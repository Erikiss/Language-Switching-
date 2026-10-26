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
    # economic fan-out / fan-in: only the 20 value-carrying rows
    assert f["fan_out_eth"] == 3 and f["fan_in_eth"] == 4 and f["n_native_transfers_pos"] == 20
    assert len(drain.native_transfers) == 81
    assert abs(sum(t["value"] for t in drain.native_transfers) / WEI - 2597.76) < 0.01
    nodes = {n.internal_index: n for n in drain.nodes()}
    assert len(nodes) == 81
    for t in drain.native_transfers:
        n = nodes[t["internal_index"]]
        assert n.value == t["value"]
        assert n.frm == t["from_address"]
        assert n.target == t["to_address"]
    # logs and storage attached by address to the first DFS node with that target, else root
    dfs = drain.nodes("dfs")
    assert sum(len(n.logs) for n in dfs) == 31
    assert sum(len(n.state_writes) for n in dfs) == 32
    first = {}
    for n in dfs:
        for a in (n.to, n.created):
            if a and a not in first:
                first[a] = n
    for n in dfs:
        for item in n.logs + n.state_writes:
            assert first.get(item["address"], drain.root) is n
    # children are sorted by trace address; DFS order equals sorted trace addresses
    assert [n.path for n in dfs] == sorted(n.path for n in dfs)
    assert drain.gas_source == "trace_estimate" and drain.gas_used > drain.root.gas_used


def test_build_records_counts_and_block_level_excluded(recs_1718, recs_1920):
    assert len(recs_1718) == 5
    assert len(recs_1920) == 9
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
    assert sum(len(r.erc20_transfers) for r in recs_1920) == 3
    assert sum(len(n.state_reads) for r in recs_1920 for n in r.nodes()) == 41


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
    assert toks[i + 1:toks.index("[OUTs]")] == ["A:0xb656b2a9c3b2416437a811e07466ca712f5a5b5a", "N59e0",
                                                "[WORD]", "[WORD]"]


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
    assert labels.era_of(1_428_000) == "dao_active"
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
    assert f["fan_out_eth"] == 3 and f["fan_in_eth"] == 4 and f["fan_definition"] == "value_gt_0"
    assert f["n_call_targets"] == 9 and f["n_call_sources"] == 6
    assert f["has_value_transfer"] and abs(f["total_eth"] - 2597.76) < 0.01
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
    others = [r for r in recs_1920 if r.tx_hash not in WITHDRAW_TXS]
    assert not any(labels.sharing_features(r)["withdraw_dao_involved"] for r in others)
    plain = by["0x6f75b64d9364b71b43cde81a889f95df72e6be004b28477f9083ed0ee471a7f9"]
    fp = labels.sharing_features(plain)
    assert fp["era"] == "fork" and labels.sharing_label(fp) == "single" and not fp["dao_involved"]


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
    # economic fan-out in the fixture is at most 3 (the drain), so the probe uses k=2:
    # positives drain, 0x124532… (drain_crisis), 0x29c093… (post_fork)
    # gas deciles: in this small pool high fan-out and fan-out 1 share (almost) no decile, so
    # the default (min_pairs = min(n, 20)) falls back to era-only matching
    pos, neg, mt = labels.probe_sets(all_recs, df, "fanout", n=10, k=2)
    assert len(pos) == len(neg) == len(mt) == 3
    assert all(mt["era"].notna()) and list(mt.columns)[:3] == ["pair", "era", "gas_decile"]
    assert set(mt["matching"]) == {"era_only"} and mt.attrs["matching"] == "era_only"
    assert np.isfinite(mt["gas_smd"]).all() and mt["gas_smd"].nunique() == 1
    assert mt.attrs["log_gas_smd"] == pytest.approx(float(mt["log_gas_smd"].iloc[0]))
    assert mt["gas_smd"].iloc[0] == pytest.approx(labels.standardized_mean_difference(mt["pos_gas"], mt["neg_gas"]))
    # without the fallback the gas-decile matching is kept (fewer pairs)
    strict = labels.probe_sets(all_recs, df, "fanout", n=10, k=2, min_pairs=0)[2]
    assert set(strict["matching"]) <= {"era+gas_decile"} and len(strict) < 3
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
    assert len(rows) == 29
    again = xatu.download_chunks(["traces"], 1920000, 1921000, cache_dir=cache, base_url=local_xatu)
    assert bool(again.loc[0, "ok"]) and bool(again.loc[0, "cached"])
    # a truncated (non-Parquet) cache file is replaced
    with open(xatu.chunk_path(cache, "traces", 1920000), "wb") as fh:
        fh.write(b"PAR1garbage")
    fixed = xatu.download_chunks("traces", 1920000, 1921000, cache_dir=cache, base_url=local_xatu)
    assert bool(fixed.loc[0, "ok"]) and not bool(fixed.loc[0, "cached"])
    assert len(xatu.read_chunk("traces", 1920000, cache)) == 29
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
