"""Tests for lsne.chain.xatu / itr / text / labels on the Xatu fixture (docs/TRACE_SOURCE_SPEC.md §9).

Fixture: tests/fixtures/xatu (ethPandaOps Xatu, CC BY 4.0; see its README).
"""
import functools
import http.server
import json
import os
import shutil
import socketserver
import threading
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest

from lsne.chain import itr, labels, text, xatu

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "xatu")
DRAIN = "0x0ec3f2488a93839524add10ea229e773f6bc891b4eb4794c3337d4495263790b"
WITHDRAW_TXS = ("0xf088dda9ba4a859d73a2ab760be508a6859b493685ff58ffd7f7e45b316caf50",
                "0x2fffeca979da48f1cf416b136ebf966c2319e55470700e9271df356c6e9b71c6")
WITHDRAW_FEASIBLE = "0xd97337573a1e7de0cfac529c3999b448968aa14c0876c15585acd80b8c67c2fe"   # fork block 1920004
CALL_CODE_TX = "0x29c0938e0933990d9965c64130c89189501b6a531d5a53ad4d907c9c698052f6"
OOG_TX = "0x21d7e78b330072bffe43e26d8d5571bc45691b7d14a454f4d7871c61a58b4414"
REVERTED_HUGE = "0x5ed540cfe135fb0501b0de15815d60b865e290e4b0d34d1ba488792d999e3175"      # chunk 1978000
DAO_CREATION = "0xe9ebfecc2fa10100db51a4408d18193b3ac504584b51a4e55bdef1318f0a30f9"      # chunk 1428000
REFUND_TX = "0xf5097cbcdb67f5a8f51fa602157725ba06bb37fc56bfdcb395b01174a2b25372"
ONE_WEI_TX = "0x7f3342ede04b21f8454e06088e3ef7f065b819f788bad88409a93b2b195e8e66"
WEI = 10 ** 18


@pytest.fixture(scope="module")
def recs_1718():
    return itr.load_records(1718000, FIX)


@pytest.fixture(scope="module")
def recs_1920():
    return itr.load_records(1920000, FIX)


@pytest.fixture(scope="module")
def all_recs(recs_1718, recs_1920):
    return list(recs_1718) + list(recs_1920)


@pytest.fixture(scope="module")
def drain(recs_1718):
    return next(r for r in recs_1718 if r.tx_hash == DRAIN)


# ======================================================================================
# xatu: URLs and normalised reading
# ======================================================================================
def test_chunk_url_and_starts():
    assert xatu.chunk_url("traces", 1718000) == (
        "https://data.ethpandaops.io/xatu/mainnet/databases/default/canonical_execution_traces/1000/1718000.parquet")
    assert xatu.chunk_url("canonical_execution_logs", 0).endswith("/canonical_execution_logs/1000/0.parquet")
    with pytest.raises(ValueError):
        xatu.chunk_url("traces", 1718001)
    assert xatu.chunk_starts(1718497, 1720000) == [1718000, 1719000]
    assert xatu.chunk_starts(1718000, 1718001) == [1718000]
    assert xatu.chunk_starts(5, 5) == []
    assert xatu.chunk_path("cache", "traces", 1000) == os.path.join("cache", "canonical_execution_traces_1000.parquet")


def test_read_chunk_lowercases_native_transfers_and_decodes_le_values():
    raw = pq.read_table(os.path.join(FIX, "canonical_execution_native_transfers_1718000.parquet")).to_pylist()
    raw_drain = [r for r in raw if r["transaction_hash"].lower() == DRAIN.encode()]
    assert any(r["transaction_hash"] != r["transaction_hash"].lower() for r in raw_drain)   # upper-case in the file
    rows = xatu.read_chunk("native_transfers", 1718000, FIX)
    assert len(rows) == len(raw)
    for r in rows:
        assert r["transaction_hash"] == r["transaction_hash"].lower()
        assert r["from_address"] == r["from_address"].lower() and r["to_address"] == r["to_address"].lower()
        assert isinstance(r["value"], int)
    drain_rows = [r for r in rows if r["transaction_hash"] == DRAIN]
    assert len(drain_rows) == 81
    assert abs(sum(r["value"] for r in drain_rows) / WEI - 2597.76) < 0.01
    # little-endian: the first transfer is 137.62451099999999 ETH
    first = min(drain_rows, key=lambda r: r["transfer_index"])
    assert first["value"] == 137624510999999990000


def test_read_chunk_traces_values_and_trace_address():
    rows = xatu.read_chunk("traces", 1718000, FIX)
    drain = [r for r in rows if r["transaction_hash"] == DRAIN]
    assert len(drain) == 81
    root = [r for r in drain if r["trace_address"] == ()]
    assert len(root) == 1 and root[0]["action_value"] == 137624510999999990000
    paths = {r["trace_address"] for r in drain}
    assert (1, 4, 0) in paths and all(isinstance(p, tuple) for p in paths)
    assert max(len(p) for p in paths) == 28
    assert all(isinstance(r["action_value"], int) for r in drain)
    assert {r["action_type"] for r in drain} == {"call", "create"}
    assert xatu.parse_trace_address(None) == () and xatu.parse_trace_address("") == ()
    assert xatu.parse_trace_address(b"1_4_0") == (1, 4, 0)


def test_block_level_rows_detected_and_normalised():
    rows = xatu.read_chunk("traces", 1920000, FIX)
    blk = [r for r in rows if xatu.is_block_level(r["transaction_hash"])]
    assert len(blk) == 1
    assert blk[0]["transaction_hash"] == xatu.ZERO_HASH
    assert blk[0]["action_type"] == "reward" and blk[0]["block_number"] == 1920000
    assert xatu.is_block_level(b"0x" + b"\x00" * 64)
    assert xatu.is_block_level("0x" + "0" * 64)
    assert not xatu.is_block_level(DRAIN)
    tx = xatu.read_chunk("transaction", 1920000, FIX)
    assert all(isinstance(r["value"], int) for r in tx)
    blocks = {r["block_number"]: r for r in xatu.read_chunk("block", 1920000, FIX)}
    assert blocks[1920000]["extra_data_string"] == "dao-hard-fork"


def test_load_chunk_rows_skips_missing_tables():
    rows = xatu.load_chunk_rows(1718000, FIX)
    assert set(rows) == {"traces", "native_transfers", "logs", "storage_diffs"}
    assert set(xatu.load_chunk_rows(1920000, FIX)) == set(xatu.TABLES)


# ======================================================================================
# itr: records
# ======================================================================================
def test_build_records_drain(drain):
    assert drain.n_nodes == 81
    assert drain.max_depth == 28
    assert drain.block_number == 1718497 and drain.tx_index == 0
    assert drain.frm == "0x969837498944ae1dc0dcac2d0c65634c88729b2d"
    f = labels.sharing_features(drain)
    # structural counts (one native_transfers row per trace node, zero-value calls included)
    assert f["n_call_targets"] == 9 and f["n_call_sources"] == 6
    # gross counts over the 20 value-carrying (all effective) rows; net flows: 3 sources, 3 recipients
    assert f["fan_out_eth_gross"] == 3 and f["fan_in_eth_gross"] == 4 and f["n_native_transfers_pos"] == 20
    assert f["fan_out_eth"] == 3 and f["fan_in_eth"] == 3 and f["n_transfers_effective"] == 20
    assert len(drain.native_transfers) == 81
    assert abs(sum(t["value"] for t in drain.native_transfers) / WEI - 2597.76) < 0.01
    nodes = {n.internal_index: n for n in drain.nodes()}
    assert len(nodes) == 81
    for t in drain.native_transfers:
        n = nodes[t["internal_index"]]
        assert n.value == t["value"]
        assert n.frm == t["from_address"]
        assert n.target == t["to_address"]
    # logs and storage diffs attached by address and feasibility (see _assert_feasible)
    dfs = drain.nodes("dfs")
    assert sum(len(n.logs) for n in dfs) == 31
    assert sum(len(n.state_writes) for n in dfs) == 32
    _assert_feasible(drain)
    # children are sorted by trace address; DFS order equals sorted trace addresses
    assert [n.path for n in dfs] == sorted(n.path for n in dfs)
    assert drain.gas_source == "trace_estimate" and drain.gas_used > drain.root.gas_used


def _assert_feasible(rec):
    """Every log / storage diff sits on a frame that acts on its address (call target, created
    contract, or the caller of a delegate_call / call_code), outside failed subtrees, and the
    frame's own gas covers the minimum cost of everything attached to it."""
    failed = itr.failed_nodes(rec.root)
    for n in rec.nodes("dfs"):
        if not (n.logs or n.state_writes):
            continue
        assert id(n) not in failed and n.call_type not in ("static_call", "suicide")
        for item in n.logs + n.state_writes:
            assert item["address"] == itr.context_address(n) and item["attach"] == "feasible"
        cost = itr.SSTORE_MIN_GAS * len(n.state_writes) + sum(itr.log_min_gas(lg) for lg in n.logs)
        assert cost <= itr.self_gas(n), (n.path, cost, itr.self_gas(n))
    # logs follow the execution order (Euler tour of the call tree)
    tour = itr._euler_tour(rec.root)
    logs = sorted(((lg["log_index"], n) for n in rec.nodes() for lg in n.logs), key=lambda x: x[0])
    pos, last = {}, -1
    for p, n in enumerate(tour):
        pos.setdefault(id(n), []).append(p)
    for _, n in logs:
        nxt = [p for p in pos[id(n)] if p >= last]
        assert nxt, "logs out of execution order"
        last = nxt[0]


def test_attach_effects_by_feasibility(recs_1920, drain):
    """WithdrawDAO.withdraw (0xd97337…): the three DAO balance writes and the Transfer log come
    from transferFrom (node [1], 29 490 gas), not from the preceding balanceOf (node [0], 788 gas)
    that the old first-node-by-address rule picked."""
    by = {r.tx_hash: r for r in recs_1920}
    r = by[WITHDRAW_FEASIBLE]
    nodes = {n.path: n for n in r.nodes()}
    assert nodes[(0,)].selector == "0x70a08231" and nodes[(1,)].selector == "0x23b872dd"
    assert not nodes[(0,)].state_writes and not nodes[(0,)].logs
    assert len(nodes[(1,)].state_writes) == 3 and len(nodes[(1,)].logs) == 1
    assert {w["address"] for w in nodes[(1,)].state_writes} == {labels.DAO_ADDRESSES["dao"]}
    for rec in recs_1920 + [drain]:
        _assert_feasible(rec)
    # delegate_call / call_code frames act on their caller's storage, failed frames on nothing
    root = itr.CallNode(path=(), call_type="call", action_type="call", frm="0xa", to="0xb", value=0, gas=10 ** 6,
                        gas_used=51000, selector=None, args=[], error=None, created=None)    # 1 000 self gas
    lib = itr.CallNode(path=(0,), call_type="delegate_call", action_type="call", frm="0xb", to="0xc", value=0,
                       gas=10 ** 6, gas_used=30000, selector=None, args=[], error=None, created=None)
    bad = itr.CallNode(path=(1,), call_type="call", action_type="call", frm="0xb", to="0xb", value=0, gas=10 ** 6,
                       gas_used=20000, selector=None, args=[], error="out of gas", created=None)
    root.children = [lib, bad]
    w = [dict(address="0xb", slot="0x1"), dict(address="0xb", slot="0x2"), dict(address="0xd", slot="0x3")]
    counts = itr.attach_effects(root, [], w)
    assert [x["slot"] for x in lib.state_writes] == ["0x1", "0x2"] and not bad.state_writes
    assert counts == {"feasible": 2, "fallback": 0, "root": 1} and root.state_writes[0]["slot"] == "0x3"
    assert root.state_writes[0]["attach"] == "root" and itr.self_gas(root) == 1000


def test_build_records_counts_and_block_level_excluded(recs_1718, recs_1920):
    assert len(recs_1718) == 5
    assert len(recs_1920) == 10
    for r in list(recs_1718) + list(recs_1920):
        assert not xatu.is_block_level(r.tx_hash)
        assert r.tx_hash == r.tx_hash.lower()
        assert r.n_nodes == len(r.nodes()) and r.max_depth == max(n.depth for n in r.nodes())
    assert [(r.block_number, r.tx_index) for r in recs_1920] == sorted((r.block_number, r.tx_index) for r in recs_1920)


def test_build_records_transaction_fields(recs_1920):
    tx = {r["transaction_hash"]: r for r in xatu.read_chunk("transaction", 1920000, FIX)}
    for r in recs_1920:
        row = tx[r.tx_hash]
        assert r.gas_source == "transaction"
        assert r.gas_used == row["gas_used"] and r.success == row["success"] and r.value == row["value"]
        assert r.frm == row["from_address"]
    kinds = {n.call_type for r in recs_1920 for n in r.nodes()}
    assert {"call", "create", "call_code", "suicide"} <= kinds
    create = next(r for r in recs_1920 if r.root.call_type == "create")
    assert create.to is None and create.root.created is not None and create.root.selector is None
    assert sum(len(r.erc20_transfers) for r in recs_1920) == 4
    assert sum(len(n.state_reads) for r in recs_1920 for n in r.nodes()) == 49


def test_build_records_accepts_raw_pyarrow_rows():
    raw = {t: pq.read_table(os.path.join(FIX, f"canonical_execution_{t}_1718000.parquet")).to_pylist()
           for t in ("traces", "native_transfers", "logs", "storage_diffs")}
    from_raw = {r.tx_hash: r for r in itr.build_records(raw)}
    from_norm = {r.tx_hash: r for r in itr.build_records(xatu.load_chunk_rows(1718000, FIX))}
    assert set(from_raw) == set(from_norm)
    for h in from_norm:
        assert itr.serialize(from_raw[h]) == itr.serialize(from_norm[h])
        assert labels.sharing_features(from_raw[h]) == labels.sharing_features(from_norm[h])


def test_record_dict_roundtrip(all_recs):
    for r in all_recs:
        back = itr.TxRecord.from_dict(json.loads(json.dumps(r.to_dict())))
        assert itr.serialize(back) == itr.serialize(r)
        assert back.native_transfers == r.native_transfers
        assert text.render_text(back) == text.render_text(r)


# ======================================================================================
# itr: serialisation
# ======================================================================================
def test_numeric_token_edge_cases():
    assert itr.numeric_token(0) == "N0"
    assert itr.numeric_token(1) == "N10e-1"
    assert itr.numeric_token(9) == "N90e-1"
    assert itr.numeric_token(10) == "N10e0"
    assert itr.numeric_token(99) == "N99e0"
    assert itr.numeric_token(100) == "N10e1"
    assert itr.numeric_token(1254) == "N13e2"
    assert itr.numeric_token(995) == "N10e2"            # rounding carries into the exponent
    assert itr.numeric_token(5 * 10 ** 18) == "N50e17"
    assert itr.numeric_token(2 ** 256 - 1) == "N12e76"
    with pytest.raises(ValueError):
        itr.numeric_token(-1)
    allowed = set(itr.numeric_tokens())
    for x in (0, 1, 7, 99, 100, 1254, 995, 5 * 10 ** 18, 2 ** 256 - 1, 123456789):
        assert itr.numeric_token(x) in allowed


def test_serialize_structure(drain, all_recs):
    toks = itr.serialize(drain)
    assert toks[0] == "[TX]" and toks[-1] == "[/TX]"
    assert toks.count("[CALL]") == drain.n_nodes
    assert "[D28]" in toks and "[D29]" not in toks
    assert toks.count("[LOG]") == 31
    assert toks.count("[W]") == 32
    # breadth-first: call depths never decrease
    depths = [int(toks[i + 1][2:-1]) for i, t in enumerate(toks) if t == "[CALL]"]
    assert depths == sorted(depths) and depths[0] == 0
    assert toks.count("[START]") == toks.count("[END]")
    # options
    assert "[STATE]" not in itr.serialize(drain, include_state=False)
    assert "[LOG]" not in itr.serialize(drain, include_logs=False)
    clipped = itr.serialize(drain, max_depth_token=10)
    assert "[D10]" in clipped and "[D11]" not in clipped
    # every non-address/selector/topic token is a closed-class token known to the vocab builder
    fixed = set(itr.fixed_tokens())
    for r in all_recs:
        for t in itr.serialize(r):
            assert t in fixed or t[:2] in ("A:", "F:", "T:"), t
    err = [r for r in all_recs if not r.success]
    assert err and all(any(t.startswith("[ERR:") for t in itr.serialize(r)) for r in err)


def test_serialize_args():
    word_addr = "0x" + "0" * 24 + "b656b2a9c3b2416437a811e07466ca712f5a5b5a"
    word_small = "0x" + format(59, "064x")
    word_big = "0x" + format(2 ** 70, "064x")
    node = itr.CallNode(path=(), call_type="call", action_type="call", frm="0x" + "1" * 40, to="0x" + "2" * 40,
                        value=0, gas=21000, gas_used=0, selector="0xa9059cbb",
                        args=[word_addr, word_small, word_big, "0x1234", word_small], error=None, created=None)
    rec = itr.TxRecord(tx_hash="0x" + "a" * 64, block_number=1, tx_index=0, frm=node.frm, to=node.to, value=0,
                       gas_used=21000, success=True, root=node, n_nodes=1, max_depth=0)
    toks = itr.serialize(rec, max_args=4)
    i = toks.index("[INs]")
    # 2**70 (an 18-decimal token amount of ~1 180 units) keeps its magnitude; short words are [WORD]
    assert toks[i + 1:toks.index("[OUTs]")] == ["A:0xb656b2a9c3b2416437a811e07466ca712f5a5b5a", "N59e0",
                                                "N12e20", "[WORD]"]
    # an address with leading zero bytes (< 2**128) is an address only when the tx touches it;
    # values >= 2**160 (hashes) stay [WORD]
    lead0 = "00000000ab" + "c" * 30
    word_lead0 = "0x" + "0" * 24 + lead0
    word_hash = "0x" + "f" * 64
    node.args = [word_lead0, word_hash]
    assert itr.serialize(rec)[i + 1:i + 3] == [itr.numeric_token(int(lead0, 16)), "[WORD]"]
    node.to = "0x" + lead0
    assert itr.serialize(rec)[i + 1:i + 3] == ["A:0x" + lead0, "[WORD]"]
    assert itr._arg_token("0x" + format(2 ** 100, "064x"), None) == itr.numeric_token(2 ** 100)
    assert itr._arg_token("0x" + format(2 ** 159, "064x"), None) == "A:0x" + format(2 ** 159, "040x")
    assert itr._arg_token("0x" + format(2 ** 160, "064x"), None) == "[WORD]"


def test_error_kinds_of_xatu_strings():
    """Every distinct trace error of the 2016 chunks maps to a specific kind."""
    assert itr.error_kind("OutOfFunds") == "out_of_funds"
    assert itr.error_kind("Bad jump destination") == "bad_jump"
    assert itr.error_kind("Out of gas") == "out_of_gas"
    assert itr.error_kind("CallTooDeep") == "call_depth"
    assert itr.error_kind("calltoodeep") == "call_depth"
    assert itr.error_kind("InvalidOperandOOG") == "out_of_gas"
    assert itr.error_kind("EOFOpcodeDisabledInLegacy") == "invalid_opcode"
    assert itr.error_kind("Reverted") == "reverted" and itr.error_kind(None) is None
    assert itr.error_kind("something new") == "other"


def test_deep_record_dict_roundtrip_is_iterative():
    """A call chain of depth 3 000 (the call-depth attacks reach 1 025) converts without recursion."""
    root = itr.CallNode(path=(), call_type="call", action_type="call", frm="0x" + "1" * 40, to="0x" + "2" * 40,
                        value=0, gas=1, gas_used=0, selector=None, args=[], error=None, created=None)
    n = root
    for d in range(3000):
        c = itr.CallNode(path=n.path + (0,), call_type="call", action_type="call", frm=n.to, to=n.to, value=0,
                         gas=1, gas_used=0, selector=None, args=[], error=None, created=None)
        n.children.append(c)
        n = c
    rec = itr.TxRecord(tx_hash="0x" + "b" * 64, block_number=1, tx_index=0, frm=root.frm, to=root.to, value=0,
                       gas_used=0, success=True, root=root, n_nodes=3001, max_depth=3000)
    d = json.loads(json.dumps(rec.to_dict()))
    back = itr.TxRecord.from_dict(d)
    assert back.max_depth == 3000 and len(back.nodes()) == 3001
    assert max(m.depth for m in back.nodes()) == 3000 and back.nodes()[-1].path == (0,) * 3000


def test_node_spans_and_visible_nodes(drain):
    toks = itr.serialize(drain)
    spans = itr.node_spans(drain)
    assert len(spans) == drain.n_nodes and [n for n, _, _ in spans] == drain.nodes("bfs")
    for n, a, b in spans:
        assert toks[a:a + 2] == ["[START]", "[CALL]"] and toks[b - 1] == "[END]"
    vis = itr.visible_nodes(drain, 512)
    assert 0 < len(vis) < drain.n_nodes and all(b <= 512 for n, a, b in spans if n in vis)
    assert len(itr.visible_nodes(drain, len(toks))) == drain.n_nodes
    f = labels.sharing_features(drain, visible_len=60)
    assert f["visible_len"] == 60 and f["fan_out_eth_visible"] <= f["fan_out_eth"]
    assert f["n_value_calls_visible"] < f["n_value_calls"]


def test_serialize_oov_with_small_vocab(drain):
    known_addr = drain.root.frm
    known_sel = drain.root.selector
    known_topic = next(lg["topic0"] for n in drain.nodes() for lg in n.logs)
    vocab = SimpleNamespace(tokens=itr.fixed_tokens() + [f"A:{known_addr}", f"F:{known_sel}", f"T:{known_topic}"],
                            top_addresses=[known_addr], top_selectors=[known_sel], top_topics=[known_topic], meta={})
    raw = itr.serialize(drain)
    toks = itr.serialize(drain, vocab)
    assert len(toks) == len(raw)
    assert f"A:{known_addr}" in toks and f"F:{known_sel}" in toks and f"T:{known_topic}" in toks
    assert "[A_OOV]" in toks and "[F_OOV]" in toks and "[T_OOV]" in toks
    vocab_set = set(vocab.tokens)
    assert all(t in vocab_set for t in toks)
    for a, b in zip(raw, toks):
        if a != b:
            assert b in ("[A_OOV]", "[F_OOV]", "[T_OOV]") and a[:2] in ("A:", "F:", "T:")
    # a vocab given only via top_* lists (tokens without the open classes) works too
    v2 = SimpleNamespace(tokens=itr.fixed_tokens(), top_addresses=[known_addr], top_selectors=[], top_topics=[])
    assert f"A:{known_addr}" in itr.serialize(drain, v2)


def test_tree_text(drain):
    s = itr.tree_text(drain)
    assert DRAIN in s and "nodes 81 depth 28" in s
    assert len(s.splitlines()) > 81


# ======================================================================================
# text
# ======================================================================================
def test_render_text_deterministic_and_mentions_root(all_recs, drain):
    for r in all_recs:
        a, b = text.render_text(r), text.render_text(r)
        assert a == b and a
        short_from = f"{r.root.frm[:6]}…{r.root.frm[-4:]}"
        assert short_from in a
        tgt = r.root.target
        if tgt:
            assert f"{tgt[:6]}…{tgt[-4:]}" in a
    s = text.render_text(drain, max_nodes=12)
    assert "69 further calls" in s
    assert "Inside that call" in s
    assert "ETH" in s
    full = text.render_text(drain, address_style="full")
    assert drain.root.frm in full and drain.root.to in full
    named = text.render_text(drain, address_style="named")
    assert "the DAO" in named
    with pytest.raises(ValueError):
        text.render_text(drain, address_style="weird")


def test_render_text_failure_and_withdraw(recs_1920):
    by = {r.tx_hash: r for r in recs_1920}
    ok = text.render_text(by[WITHDRAW_TXS[0]], address_style="named")
    assert "the WithdrawDAO contract" in ok and "160 ETH" in ok
    bad = text.render_text(by[WITHDRAW_TXS[1]])
    assert "The transaction fails (bad jump destination)." in bad


def test_format_eth():
    assert text.format_eth(0) == "0"
    assert text.format_eth(2597758105043150400000) == "2600"
    assert text.format_eth(23 * 10 ** 17) == "2.3"
    assert text.format_eth(5 * 10 ** 16) == "0.050"
    assert text.format_eth(1) == "1.0e-18"
    assert text.format_eth(9_960_000_000_000_000_000) == "10"


# ======================================================================================
# labels
# ======================================================================================
def test_era_of():
    assert labels.era_of(0) == "pre_dao"
    assert labels.era_of(1_427_999) == "pre_dao"
    assert labels.era_of(1_428_756) == "pre_dao"         # the DAO is created in block 1 428 757
    assert labels.era_of(1_428_757) == "dao_active"
    assert labels.era_of(1_718_496) == "dao_active"
    assert labels.era_of(1_718_497) == "drain_crisis"
    assert labels.era_of(1_919_999) == "drain_crisis"
    assert labels.era_of(1_920_000) == "fork"
    assert labels.era_of(1_920_009) == "fork"
    assert labels.era_of(1_920_010) == "post_fork"


def test_labels_drain(drain):
    f = labels.sharing_features(drain)
    assert f["dao_involved"] and not f["withdraw_dao_involved"]
    assert f["era"] == "drain_crisis"
    assert f["max_depth"] == 28 and f["n_nodes"] == 81
    assert f["n_state_writes"] == 32
    assert f["fan_out_eth"] == 3 and f["fan_in_eth"] == 3 and f["fan_definition"] == "net_effective"
    assert f["n_call_targets"] == 9 and f["n_call_sources"] == 6
    assert f["has_value_transfer"] and abs(f["total_eth"] - 2597.76) < 0.01 and f["net_eth"] < f["total_eth"]
    assert labels.sharing_label(f) == "many_to_many"
    assert labels.sharing_label(f, structural=True) == "many_to_many"


def test_labels_withdraw_dao(recs_1920):
    by = {r.tx_hash: r for r in recs_1920}
    for h in WITHDRAW_TXS:
        f = labels.sharing_features(by[h])
        assert f["withdraw_dao_involved"] and f["dao_involved"]
        assert f["era"] == "post_fork"
        assert f["fan_out_erc20"] <= 1
    # the withdrawal: 3 call targets but one economic ETH transfer (160 ETH) -> "single"
    fw = labels.sharing_features(by[WITHDRAW_TXS[0]])
    assert (fw["n_call_targets"], fw["fan_out_eth"], fw["fan_in_eth"]) == (3, 1, 1)
    assert labels.sharing_label(fw) == "single" and labels.sharing_label(fw, structural=True) == "many_to_many"
    # zero-value ERC-20 rows and zero-value calls do not count: the failed withdrawal moves nothing
    ff = labels.sharing_features(by[WITHDRAW_TXS[1]])
    assert ff["fan_out_eth"] == ff["fan_in_eth"] == 0 and ff["n_call_targets"] == 2
    assert labels.sharing_label(ff) == "none"
    others = [r for r in recs_1920 if r.tx_hash not in WITHDRAW_TXS + (WITHDRAW_FEASIBLE,)]
    assert not any(labels.sharing_features(r)["withdraw_dao_involved"] for r in others)
    plain = by["0x6f75b64d9364b71b43cde81a889f95df72e6be004b28477f9083ed0ee471a7f9"]
    fp = labels.sharing_features(plain)
    assert fp["era"] == "fork" and labels.sharing_label(fp) == "single" and not fp["dao_involved"]


def _recs(start):
    return {r.tx_hash: r for r in itr.load_records(start, FIX)}


def test_transfer_flags_and_effective_transfers(recs_1920):
    """Every native transfer row is flagged; reverted (failed subtree / failed tx) and phantom
    (call_code / delegate_call) rows move no ETH and count nowhere."""
    by = {r.tx_hash: r for r in recs_1920}
    for r in recs_1920:
        assert all({"reverted", "phantom"} <= set(t) for t in r.native_transfers)
    cc = by[CALL_CODE_TX]                       # call_code row with 0.168 ETH: no balance change (balance_diffs)
    ph = [t for t in cc.native_transfers if t["phantom"]]
    assert len(ph) == 1 and ph[0]["value"] > 0
    eff = itr.effective_transfers(cc)
    assert len(eff) == 1 and abs(eff[0]["value"] / WEI - 0.1679) < 1e-3
    f = labels.sharing_features(cc)
    assert (f["fan_out_eth"], f["fan_in_eth"], f["n_transfers_phantom"]) == (1, 1, 1) and abs(f["total_eth"] - 0.168) < 1e-3
    t = text.render_text(cc)
    assert "In total 0.17 ETH moves in 1 transfer from 1 sender to 1 receiver." in t
    oog = by[OOG_TX]                             # failed transaction: nothing moved
    assert not oog.success and all(x["reverted"] for x in oog.native_transfers)
    fo = labels.sharing_features(oog)
    assert fo["fan_out_eth"] == fo["fan_in_eth"] == 0 and fo["total_eth"] == 0 and labels.sharing_label(fo) == "none"
    assert "No ETH moves. One attempted transfer is reverted." in text.render_text(oog)
    # flags are recomputed for records without them (older dicts)
    d = cc.to_dict()
    for x in d["native_transfers"]:
        x.pop("reverted"), x.pop("phantom")
    assert len(itr.effective_transfers(itr.TxRecord.from_dict(d))) == 1


def test_reverted_huge_transfer_does_not_count():
    """0x5ed540cf… (chunk 1978000): node (26,) carries 2**256 wei with error OutOfFunds.  Label,
    total and text agree: four receivers of 0.019 ETH, the attempt is reverted."""
    r = _recs(1978000)[REVERTED_HUGE]
    node = next(n for n in r.nodes() if n.path == (26,))
    assert node.error and node.value >= 2 ** 255
    f = labels.sharing_features(r)
    assert f["fan_out_eth_gross"] == 4 and f["n_transfers_reverted"] == 1
    assert abs(f["total_eth"] - 0.019) < 0.001 and f["total_eth"] < 1
    assert f["fan_out_eth"] <= 4 and f["fan_out_eth"] < 5
    t = text.render_text(r)
    assert "to 4 receivers" in t and "One attempted transfer is reverted." in t and "0.019 ETH" in t


def test_net_flows_refund_and_routing():
    """Pass-through contracts and refunds cancel in the net flows."""
    rows = [dict(from_address="0xa", to_address="0xc", value=10), dict(from_address="0xc", to_address="0xb", value=10)]
    assert itr.net_flows(rows) == {"0xa": -10, "0xb": 10} and itr.flow_counts(itr.net_flows(rows)) == (1, 1)
    refund = [dict(from_address="0xa", to_address="0xc", value=7), dict(from_address="0xc", to_address="0xa", value=7)]
    assert itr.net_flows(refund) == {}
    split = [dict(from_address="0xa", to_address="0xc", value=9)] + [
        dict(from_address="0xc", to_address=f"0x{i}", value=3) for i in range(3)]
    assert itr.flow_counts(itr.net_flows(split)) == (3, 1)
    by = _recs(1428000)
    r = by[REFUND_TX]                        # 0.010153 ETH to a contract and straight back to the sender
    f = labels.sharing_features(r)
    assert (f["fan_out_eth_gross"], f["fan_in_eth_gross"]) == (2, 2)
    assert f["fan_out_eth"] == f["fan_in_eth"] == 0 and labels.sharing_label(f) == "none"
    assert "Net, no address gains or loses ETH." in text.render_text(r)
    w = by[ONE_WEI_TX]                       # 1 wei + a reverted OutOfFunds attempt: single, not many_to_many
    fw = labels.sharing_features(w)
    assert labels.sharing_label(fw) == "single" and fw["n_transfers_reverted"] >= 1
    assert "1 transfer from 1 sender to 1 receiver" in text.render_text(w)


def test_dao_era_list_and_mentions():
    by = _recs(1428000)
    c = by[DAO_CREATION]
    assert c.block_number == 1_428_757 and labels.era_of(c.block_number) == "dao_active"
    assert labels.sharing_features(c)["dao_involved"]
    assert c.address_kinds.get(labels.DAO_ADDRESSES["dao"]) == "contract"
    assert len(labels.DAO_FORK_LIST) == 116 == len(set(labels.DAO_FORK_LIST))
    assert labels.DAO_ADDRESSES["dao"] in labels.DAO_FORK_LIST and labels.DAO_ADDRESSES["extra_balance"] in labels.DAO_FORK_LIST
    dark = "0x304a554a310c7e546dfe434669c62820b7d83490"          # the child DAO of the drain
    assert dark in labels.DAO_INVOLVED
    child = itr.CallNode(path=(), call_type="call", action_type="call", frm="0x" + "1" * 40, to=dark, value=0,
                         gas=50000, gas_used=0, selector="0xa9059cbb", args=[], error=None, created=None)
    rec = itr.TxRecord(tx_hash="0x" + "c" * 64, block_number=1_800_000, tx_index=0, frm=child.frm, to=dark, value=0,
                       gas_used=21000, success=True, root=child, n_nodes=1, max_depth=0)
    assert labels.sharing_features(rec)["dao_involved"]
    # a DAO address only as a call argument: ambiguous, excluded from the DAO probe's negatives
    arg = "0x" + "0" * 24 + labels.DAO_ADDRESSES["dao"][2:]
    other = itr.CallNode(path=(), call_type="call", action_type="call", frm="0x" + "1" * 40, to="0x" + "3" * 40,
                         value=0, gas=50000, gas_used=0, selector="0x095ea7b3", args=[arg], error=None, created=None)
    rec2 = itr.TxRecord(tx_hash="0x" + "d" * 64, block_number=1_800_000, tx_index=1, frm=other.frm, to=other.to,
                        value=0, gas_used=21000, success=True, root=other, n_nodes=1, max_depth=0)
    f2 = labels.sharing_features(rec2)
    assert not f2["dao_involved"] and f2["dao_mentioned"]
    _, neg, mt = labels.probe_sets([rec, rec2], None, "dao", n=5, gas_bins=1)
    assert not neg and mt.attrs["n_excluded"] == 1
    assert f"A:{dark}" not in labels.mask_tokens([f"A:{dark}"], "dao")


def test_address_kinds_in_text(recs_1920):
    """Contract / account wording from code evidence: the DAO is a contract in every text."""
    by = {r.tx_hash: r for r in recs_1920}
    r = by[WITHDRAW_FEASIBLE]
    assert r.address_kinds[labels.DAO_ADDRESSES["dao"]] == "contract"
    assert r.address_kinds[r.frm] == "account"
    plain = by["0x6f75b64d9364b71b43cde81a889f95df72e6be004b28477f9083ed0ee471a7f9"]
    assert "to account 0x53d2…8a3d" in text.render_text(plain)
    # a failed call to an address without evidence is neither called account nor contract
    oog = by[OOG_TX]
    oog2 = itr.TxRecord.from_dict({**oog.to_dict(), "address_kinds": {}})
    assert " to address " in text.render_text(oog2)
    dao_call = itr.TxRecord.from_dict({**oog.to_dict(), "address_kinds": {oog.root.to: "contract"}})
    assert " to contract " in text.render_text(dao_call)


def test_sharing_label_rules():
    base = dict(has_value_transfer=True, fan_out_erc20=0, fan_in_erc20=0)
    lab = lambda fo, fi, **kw: labels.sharing_label(dict(base, fan_out_eth=fo, fan_in_eth=fi, **kw))  # noqa: E731
    assert lab(5, 1) == "distribution"
    assert lab(1, 5) == "pooling"
    assert lab(3, 2) == "many_to_many"
    assert lab(1, 1) == "single"
    assert lab(3, 1) == "other"
    assert lab(1, 1, has_value_transfer=False) == "none"
    assert labels.sharing_label(dict(base, fan_out_eth=4, fan_in_eth=1), k=4) == "distribution"
    f = dict(base, fan_out_eth=5, fan_in_eth=1, n_call_targets=9, n_call_sources=6)
    assert labels.sharing_label(f) == "distribution"
    assert labels.sharing_label(f, structural=True) == "many_to_many"
    erc = dict(has_value_transfer=True, fan_out_eth=0, fan_in_eth=0, fan_out_erc20=1, fan_in_erc20=1)
    assert labels.sharing_label(erc) == "single"


def test_probe_sets_matched_on_era(all_recs):
    df = labels.labels_frame(all_recs)
    assert set(df.columns) >= {"fan_out_eth", "fan_in_eth", "fan_out_erc20", "fan_in_erc20", "n_value_calls",
                               "n_state_writes", "n_distinct_slots_written", "dao_involved",
                               "withdraw_dao_involved", "max_depth", "n_nodes", "era", "label"}
    # net fan-out in the fixture is at most 3 (the drain), so the probe uses k=2: positives drain and
    # 0x124532… (both drain_crisis; the call_code transaction 0x29c093… has one effective transfer);
    # gas deciles: in this small pool high fan-out and fan-out 1 share no decile, so the default
    # (min_pairs = min(n, 20)) falls back to era-only matching
    pos, neg, mt = labels.probe_sets(all_recs, df, "fanout", n=10, k=2)
    assert len(pos) == len(neg) == len(mt) == 2
    assert mt.attrs["visible"] is True and (mt["pos_fan_out_eth_visible"] >= 2).all()
    assert (mt["neg_fan_out_eth_visible"] == 1).all()
    assert all(mt["era"].notna()) and list(mt.columns)[:3] == ["pair", "era", "gas_decile"]
    assert set(mt["matching"]) == {"era_only"} and mt.attrs["matching"] == "era_only"
    assert np.isfinite(mt["gas_smd"]).all() and mt["gas_smd"].nunique() == 1
    assert mt.attrs["log_gas_smd"] == pytest.approx(float(mt["log_gas_smd"].iloc[0]))
    assert mt["gas_smd"].iloc[0] == pytest.approx(labels.standardized_mean_difference(mt["pos_gas"], mt["neg_gas"]))
    # without the fallback the gas-decile matching is kept (fewer pairs)
    strict = labels.probe_sets(all_recs, df, "fanout", n=10, k=2, min_pairs=0)[2]
    assert set(strict["matching"]) <= {"era+gas_decile"} and len(strict) < 2
    pos, neg, mt = labels.probe_sets(all_recs, df, "fanout", n=10, k=2, gas_bins=1)
    assert len(pos) == len(neg) == len(mt) >= 2 and set(mt["matching"]) == {"era_only"}
    for p, q, (_, row) in zip(pos, neg, mt.iterrows()):
        assert labels.era_of(p.block_number) == labels.era_of(q.block_number) == row["era"]
        assert labels.sharing_features(p)["fan_out_eth"] >= 2
        assert labels.sharing_features(q)["fan_out_eth"] == 1
        assert row["pos_tx"] == p.tx_hash and row["neg_tx"] == q.tx_hash
    assert len({r.tx_hash for r in pos + neg}) == 2 * len(pos)
    assert len(labels.probe_sets(all_recs, df, "fanout", n=10)[0]) == 0          # no economic fan-out >= 5 here
    pos, neg, mt = labels.probe_sets(all_recs, None, "dao", n=10, seed=1, gas_bins=2)
    assert len(pos) >= 2
    assert set(mt["era"]) <= {"drain_crisis", "post_fork"}
    for p, q in zip(pos, neg):
        assert labels.era_of(p.block_number) == labels.era_of(q.block_number)
        assert labels.sharing_features(p)["dao_involved"] and not labels.sharing_features(q)["dao_involved"]
    again = labels.probe_sets(all_recs, None, "dao", n=10, seed=1, gas_bins=2)[2]
    assert again.equals(mt)
    assert len(labels.probe_sets(all_recs, df, "fanout", n=1, k=2, gas_bins=1)[0]) == 1
    with pytest.raises(ValueError):
        labels.probe_sets(all_recs, df, "nope", n=1)
    # features of an older serialisation (fan_out_eth counted zero-value call rows) are refused
    stale = df.drop(columns=["n_call_targets", "n_call_sources"]).assign(fan_out_eth_pos=0)
    with pytest.raises(ValueError, match="older serialisation"):
        labels.probe_sets(all_recs, stale, "fanout", n=10)


def _synthetic_frame(rows):
    """labels_df + stand-in records for probe_sets from (era, fan_out_eth, n_tokens[, gas, dao]) tuples."""
    recs, out = [], []
    for i, r in enumerate(rows):
        era, fo, ntok = r[:3]
        gas = r[3] if len(r) > 3 else 21000 + i
        dao = r[4] if len(r) > 4 else False
        h = f"0x{i:064x}"
        recs.append(SimpleNamespace(tx_hash=h))
        out.append(dict(tx_hash=h, era=era, fan_out_eth=fo, dao_involved=dao, gas_used=gas, n_tokens=ntok,
                        n_nodes=max(1, ntok // 20), n_value_calls=fo, n_call_targets=fo))
    return recs, __import__("pandas").DataFrame(out)


def test_probe_sets_length_matching_caliper_and_balance():
    rng = np.random.default_rng(0)
    rows = [("post_fork", 3, int(x)) for x in rng.integers(60, 400, 40)]          # positives, longer on average
    rows += [("post_fork", 1, int(x)) for x in rng.integers(10, 300, 400)]        # negatives
    rows += [("dao_active", 2, 50), ("dao_active", 1, 52), ("dao_active", 1, 500)]
    recs, df = _synthetic_frame(rows)
    pos, neg, mt = labels.probe_sets(recs, df, "fanout", n=100, k=2, match="length", length_ratio=1.25, seed=3)
    assert len(pos) == len(neg) == len(mt) and len(mt) >= 30
    assert set(mt["matching"]) == {"era+log_length"} and (mt["gas_decile"] == -1).all()
    by = df.set_index("tx_hash")
    for p, q, (_, row) in zip(pos, neg, mt.iterrows()):
        assert by.at[p.tx_hash, "era"] == by.at[q.tx_hash, "era"] == row["era"]
        assert by.at[p.tx_hash, "fan_out_eth"] >= 2 and by.at[q.tx_hash, "fan_out_eth"] == 1
        assert abs(np.log(row["pos_n_tokens"] / row["neg_n_tokens"])) <= np.log(1.25) + 1e-9
    assert len({r.tx_hash for r in pos}) == len(pos) and len({r.tx_hash for r in neg}) == len(neg)
    # the dao_active pair is the nearest admissible length (50 -> 52, not 500)
    da = mt[mt["era"] == "dao_active"]
    assert list(zip(da["pos_n_tokens"], da["neg_n_tokens"])) == [(50, 52)]
    # balance: matched on log length, so the SMD is small although the pools differ a lot
    pool_smd = labels.standardized_mean_difference(np.log(df[df.fan_out_eth >= 2].n_tokens),
                                                   np.log(df[df.fan_out_eth == 1].n_tokens))
    assert pool_smd > 0.5 and abs(mt.attrs["length_smd"]) < 0.1 and mt["length_smd"].nunique() == 1
    assert mt.attrs["length_ratio_max"] <= 1.25 + 1e-9
    assert np.isfinite(mt.attrs["gas_smd"]) and np.isfinite(mt.attrs["log_gas_smd"])
    bal = mt.attrs["balance"]
    assert {"n_tokens", "gas_used", "n_nodes", "n_value_calls"} <= set(bal)
    assert bal["n_tokens"]["log_smd"] == pytest.approx(mt.attrs["length_smd"])
    assert mt.attrs["pool_pos"] == 41 and mt.attrs["pool_neg"] == 402
    assert mt.attrs["n_matched_total"] == len(mt) + 0 and mt.attrs["n_pos_unmatched"] == 41 - len(mt)
    again = labels.probe_sets(recs, df, "fanout", n=100, k=2, match="length", length_ratio=1.25, seed=3)[2]
    assert again.equals(mt)
    # n caps the number of pairs (random subset of the matched pairs)
    assert len(labels.probe_sets(recs, df, "fanout", n=5, k=2, match="length")[0]) == 5
    with pytest.raises(ValueError, match="match must be"):
        labels.probe_sets(recs, df, "fanout", n=5, match="nope")


def test_probe_sets_length_matching_hardest_first_and_no_fallback():
    # positive A (100 tokens) admits only the 110-token negative; positive B (115) admits 110 and 130.
    # Nearest-first in random order could give B the 110 and leave A unmatched; hardest-first matches both.
    rows = [("post_fork", 5, 100), ("post_fork", 5, 115), ("post_fork", 1, 110), ("post_fork", 1, 130)]
    recs, df = _synthetic_frame(rows)
    for seed in range(5):
        _, _, mt = labels.probe_sets(recs, df, "fanout", n=10, k=5, match="length", seed=seed)
        assert sorted(zip(mt["pos_n_tokens"], mt["neg_n_tokens"])) == [(100, 110), (115, 130)]
    # no admissible negative -> unmatched, no silent fallback to era-only matching
    recs, df = _synthetic_frame([("post_fork", 5, 1000), ("post_fork", 1, 20), ("post_fork", 1, 30)])
    pos, neg, mt = labels.probe_sets(recs, df, "fanout", n=10, k=5, match="length")
    assert len(pos) == 0 and mt.attrs["n_pos_unmatched"] == 1 and mt.attrs["n_pairs"] == 0
    # matching on the visible length (min(n, cap)): 1000 and 600 tokens both look like 512
    recs, df = _synthetic_frame([("post_fork", 5, 1000), ("post_fork", 1, 600), ("post_fork", 1, 20)])
    _, _, mt = labels.probe_sets(recs, df, "fanout", n=10, k=5, match="length", length_cap=512)
    assert list(zip(mt["pos_n_tokens"], mt["neg_n_tokens"])) == [(1000, 600)] and mt.attrs["length_cap"] == 512
    # length matching on stand-in records needs the token counts
    with pytest.raises(ValueError, match="n_tokens"):
        labels.probe_sets(recs, df.drop(columns=["n_tokens"]), "fanout", n=10, k=5, match="length")


def test_probe_sets_length_matching_on_records(all_recs):
    # without an n_tokens column the records are serialised; the fixture's fan-out >= 2 transactions are long,
    # so a generous ratio is needed to find partners at all
    pos, neg, mt = labels.probe_sets(all_recs, None, "fanout", n=10, k=2, match="length", length_ratio=30.0)
    assert len(pos) >= 1
    for p, q, (_, row) in zip(pos, neg, mt.iterrows()):
        assert row["pos_n_tokens"] == len(itr.serialize(p)) and row["neg_n_tokens"] == len(itr.serialize(q))
        assert labels.era_of(p.block_number) == labels.era_of(q.block_number)
        assert abs(np.log(row["pos_n_tokens"] / row["neg_n_tokens"])) <= np.log(30.0) + 1e-9
    # the gas matching of the spec is unchanged (no token counts computed for records)
    gas = labels.probe_sets(all_recs, None, "fanout", n=10, k=2)[2]
    assert "pos_n_tokens" not in gas.columns and gas.attrs["match"] == "gas"


def test_mask_tokens(drain, all_recs):
    toks = itr.serialize(drain)
    dao_toks = {f"A:{a}" for a in labels.DAO_INVOLVED}           # EIP-779 list + creator + WithdrawDAO
    assert any(t == "A:0x304a554a310c7e546dfe434669c62820b7d83490" for t in toks)   # the darkDAO
    n_dao = sum(t in dao_toks for t in toks)
    assert n_dao > 0
    m = labels.mask_tokens(toks, "dao")
    assert len(m) == len(toks) and not any(t in dao_toks for t in m)
    assert m.count("[A_OOV]") == toks.count("[A_OOV]") + n_dao
    assert [a for a, b in zip(toks, m) if a != b] == [t for t in toks if t in dao_toks]
    # the same on vocabulary-mapped tokens (the DAO addresses are always in the vocabulary)
    vocab = SimpleNamespace(tokens=itr.fixed_tokens(), top_addresses=sorted(labels.DAO_INVOLVED),
                            top_selectors=[], top_topics=[])
    mapped = itr.serialize(drain, vocab)
    assert labels.mask_tokens(mapped, "dao").count("[A_OOV]") == mapped.count("[A_OOV]") + n_dao
    # WithdrawDAO transactions lose their identity too
    for r in all_recs:
        if labels.sharing_features(r)["withdraw_dao_involved"]:
            assert f"A:{labels.DAO_ADDRESSES['withdraw_dao']}" not in labels.mask_tokens(itr.serialize(r), "dao")
    allm = labels.mask_tokens(toks + ["A:none"], "all_addresses")
    assert not any(t.startswith("A:0x") for t in allm) and allm[-1] == "A:none"
    assert allm.count("[A_OOV]") == sum(t.startswith("A:0x") for t in toks)
    assert labels.mask_tokens(toks, "none") == toks and labels.mask_tokens(toks, None) == toks
    custom = labels.mask_tokens(toks, [drain.root.frm.upper().replace("0X", "0x")])
    assert f"A:{drain.root.frm}" not in custom and custom.count("[A_OOV]") > toks.count("[A_OOV]")
    with pytest.raises(ValueError, match="mask must be"):
        labels.mask_tokens(toks, "everything")
    assert labels.surface_features(m)["n_a_oov"] == m.count("[A_OOV]")
    assert labels.surface_features(toks)["n_a_oov"] == 0
    assert "n_a_oov" in labels.SURFACE_FEATURES_OOV and "n_a_oov" not in labels.SURFACE_FEATURES


def test_standardized_mean_difference():
    assert labels.standardized_mean_difference([1, 2, 3], [1, 2, 3]) == 0.0
    assert labels.standardized_mean_difference([2, 3, 4], [1, 2, 3]) == pytest.approx(1.0)
    assert np.isnan(labels.standardized_mean_difference([1], [1, 2]))


def test_surface_features_and_baseline(drain, all_recs):
    toks = itr.serialize(drain)
    sf = labels.surface_features(toks)
    assert sf["n_tokens"] == len(toks) and sf["n_call"] == drain.n_nodes == 81
    assert sf["n_value_tokens"] == sum(t.startswith("V:") for t in toks)
    assert sf["n_value_calls_nonzero"] == labels.sharing_features(drain)["n_value_calls"] == 20
    assert labels.surface_features(toks, max_len=10)["n_tokens"] == 10
    # perfectly separable by length -> AUC 1 for n_tokens and the combination
    short = [["[TX]", "[START]", "[CALL]", "V:N0", "[END]", "[/TX]"]] * 6
    long_ = [itr.serialize(r) for r in all_recs[:6]]
    long_ = [t + ["[START]", "[CALL]", "V:N10e17", "[END]"] * 3 for t in long_]
    res = labels.surface_baseline_auc(long_, short, seed=0)
    assert res["n_tokens"] == 1.0 and res["combined"] == 1.0 and res["n_pairs"] == 6
    assert res["best_single"] in labels.SURFACE_FEATURES
    flipped = labels.surface_baseline_auc(short, long_, seed=0)
    assert flipped["n_tokens"] == 1.0                                   # orientation fitted on the training half
    assert np.isnan(labels.surface_baseline_auc(short[:3], long_[:3])["combined"])


# ======================================================================================
# download (local HTTP server; the real endpoint only in the network-marked test)
# ======================================================================================
class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass


@pytest.fixture
def local_xatu(tmp_path):
    root = tmp_path / "srv"
    d = root / "canonical_execution_traces" / "1000"
    d.mkdir(parents=True)
    shutil.copy(os.path.join(FIX, "canonical_execution_traces_1920000.parquet"), d / "1920000.parquet")
    handler = functools.partial(_QuietHandler, directory=str(root))
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_download_chunks_resume_and_errors(local_xatu, tmp_path):
    cache = str(tmp_path / "cache")
    df = xatu.download_chunks("traces", 1920000, 1922000, cache_dir=cache, base_url=local_xatu, retries=1,
                              backoff=0.01, max_workers=2)
    assert list(df.columns) == ["table", "start", "path", "bytes", "ok", "error", "cached"]
    assert list(df["start"]) == [1920000, 1921000]
    ok = df.set_index("start")
    assert ok.loc[1920000, "ok"] and not ok.loc[1920000, "cached"] and ok.loc[1920000, "bytes"] > 0
    assert not ok.loc[1921000, "ok"] and "404" in ok.loc[1921000, "error"]
    assert not os.path.exists(xatu.chunk_path(cache, "traces", 1921000))
    assert not [f for f in os.listdir(cache) if ".tmp." in f]
    rows = xatu.read_chunk("traces", 1920000, cache)
    assert len(rows) == 33
    again = xatu.download_chunks(["traces"], 1920000, 1921000, cache_dir=cache, base_url=local_xatu)
    assert bool(again.loc[0, "ok"]) and bool(again.loc[0, "cached"])
    # a truncated (non-Parquet) cache file is replaced
    with open(xatu.chunk_path(cache, "traces", 1920000), "wb") as fh:
        fh.write(b"PAR1garbage")
    fixed = xatu.download_chunks("traces", 1920000, 1921000, cache_dir=cache, base_url=local_xatu)
    assert bool(fixed.loc[0, "ok"]) and not bool(fixed.loc[0, "cached"])
    assert len(xatu.read_chunk("traces", 1920000, cache)) == 33
    assert xatu.cached_starts(cache) == [1920000]
    every = xatu.download_chunks("traces", 1918000, 1922000, cache_dir=cache, base_url=local_xatu, every=2,
                                 retries=0)
    assert list(every["start"]) == [1918000, 1920000]


@pytest.mark.network
def test_download_chunks_network_tiny(tmp_path):
    requests = pytest.importorskip("requests")
    try:
        requests.head(xatu.chunk_url("erc20_transfers", 0), timeout=20)
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"Xatu not reachable: {exc}")
    cache = str(tmp_path / "xatu")
    df = xatu.download_chunks("erc20_transfers", 0, 2000, cache_dir=cache, retries=2, backoff=0.5)
    assert df["ok"].all() and not df["cached"].any(), df.to_dict("records")
    assert xatu.read_chunk("erc20_transfers", 0, cache) == []        # no ERC-20 transfers in blocks 0..999
    again = xatu.download_chunks("erc20_transfers", 0, 2000, cache_dir=cache)
    assert again["ok"].all() and again["cached"].all()
