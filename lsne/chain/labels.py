"""'Sharing' labels, DAO-era segmentation and matched probe sets on :class:`TxRecord` objects.

Fan-out / fan-in follow docs/TRACE_SOURCE_SPEC.md §0/§4: distinct receivers / senders over
the transaction's ``native_transfers`` rows.  Xatu writes one such row per trace node,
*including zero-value calls* (the DAO drain: 81 rows, 6 senders, 9 receivers, of which
only 20 rows carry ETH: 4 senders, 3 receivers).  The spec's numbers (fan-out 9 / fan-in 6
for the drain; median 1, 99th percentile 6, max 51 in chunk 1718000) are reproduced only
by counting all rows, so ``fan_out_eth`` / ``fan_in_eth`` do that; the value-only counts are
reported as ``fan_out_eth_pos`` / ``fan_in_eth_pos`` and can drive the label with
``sharing_label(..., value_only=True)``.  ``"none"`` (no value transfer) means: no native
transfer with value > 0 and no ERC-20 transfer with value > 0.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .itr import TxRecord, iter_nodes

DAO_ADDRESSES = {"dao": "0xbb9bc244d798123fde783fcc1c72d3bb8c189413",
                 "extra_balance": "0x807640a13483f8ac783c557fcdf27be11ea4ac7a",
                 "creator": "0x4a574510c7014e4ae985403536074abe582adfc8",
                 "withdraw_dao": "0xbf4ed7b27f1d666546e30d74d50d173d20bca754"}
ERA_BOUNDS = {"pre_dao": (0, 1_428_000), "dao_active": (1_428_000, 1_718_497),
              "drain_crisis": (1_718_497, 1_920_000), "fork": (1_920_000, 1_920_010),
              "post_fork": (1_920_010, 10 ** 9)}
ERAS = tuple(ERA_BOUNDS)
LABELS = ("distribution", "pooling", "many_to_many", "single", "none", "other")
_DAO_SET = frozenset(DAO_ADDRESSES.values())


def era_of(block: int) -> str:
    """Era name for a block number (half-open bounds ``[lo, hi)``)."""
    b = int(block)
    for name, (lo, hi) in ERA_BOUNDS.items():
        if lo <= b < hi:
            return name
    raise ValueError(f"block {block} outside every era")


def touched_addresses(rec: TxRecord) -> set:
    """Every address the transaction touches: call endpoints, created contracts, log
    emitters, written / read storage owners, transfer endpoints and ERC-20 token contracts."""
    out = set()
    for n in iter_nodes(rec.root, "dfs"):
        out.update(a for a in (n.frm, n.to, n.created) if a)
        out.update(x.get("address") for x in n.logs if x.get("address"))
        out.update(x.get("address") for x in n.state_writes if x.get("address"))
        out.update(x.get("address") for x in n.state_reads if x.get("address"))
    for t in rec.native_transfers:
        out.update(a for a in (t.get("from_address"), t.get("to_address")) if a)
    for t in rec.erc20_transfers:
        out.update(a for a in (t.get("from_address"), t.get("to_address"), t.get("erc20")) if a)
    out.update(a for a in (rec.frm, rec.to) if a)
    return {str(a).lower() for a in out}


def sharing_features(rec: TxRecord) -> dict:
    """Per-transaction sharing features (addresses lower-cased).

    Spec keys: fan_out_eth, fan_in_eth, fan_out_erc20, fan_in_erc20, n_value_calls,
    n_state_writes, n_distinct_slots_written, dao_involved, withdraw_dao_involved,
    max_depth, n_nodes, era.  Extra keys: fan_out_eth_pos, fan_in_eth_pos (value > 0 rows
    only), n_native_transfers, n_erc20_transfers, total_eth, has_value_transfer, tx_hash,
    block_number, gas_used, success.
    """
    nat = rec.native_transfers
    erc = rec.erc20_transfers
    lower = lambda a: str(a).lower() if a else a  # noqa: E731
    pos = [t for t in nat if t.get("value")]
    erc_pos = [t for t in erc if t.get("value")]
    nodes = list(iter_nodes(rec.root, "dfs"))
    writes = [w for n in nodes for w in n.state_writes]
    touched = touched_addresses(rec)
    return dict(
        fan_out_eth=len({lower(t.get("to_address")) for t in nat}),
        fan_in_eth=len({lower(t.get("from_address")) for t in nat}),
        fan_out_erc20=len({lower(t.get("to_address")) for t in erc}),
        fan_in_erc20=len({lower(t.get("from_address")) for t in erc}),
        n_value_calls=sum(1 for n in nodes if n.value > 0),
        n_state_writes=len(writes),
        n_distinct_slots_written=len({(lower(w.get("address")), lower(w.get("slot"))) for w in writes}),
        dao_involved=bool(touched & _DAO_SET),
        withdraw_dao_involved=DAO_ADDRESSES["withdraw_dao"] in touched,
        max_depth=int(rec.max_depth),
        n_nodes=int(rec.n_nodes),
        era=era_of(rec.block_number),
        fan_out_eth_pos=len({lower(t.get("to_address")) for t in pos}),
        fan_in_eth_pos=len({lower(t.get("from_address")) for t in pos}),
        n_native_transfers=len(nat),
        n_erc20_transfers=len(erc),
        total_eth=sum(int(t["value"]) for t in pos) / 1e18,
        has_value_transfer=bool(pos or erc_pos),
        tx_hash=rec.tx_hash,
        block_number=int(rec.block_number),
        gas_used=int(rec.gas_used),
        success=bool(rec.success),
    )


def sharing_label(feat: dict, k: int = 5, value_only: bool = False) -> str:
    """"distribution" (one sender, fan_out >= k), "pooling" (one receiver, fan_in >= k),
    "many_to_many" (fan_out >= 2 and fan_in >= 2), "single" (fan_out == fan_in == 1),
    "none" (no value transfer), else "other".  Counts from native transfers (all rows,
    or value-carrying rows with ``value_only``); without native rows the ERC-20 counts."""
    has_value = feat.get("has_value_transfer")
    if value_only:
        fo, fi = feat.get("fan_out_eth_pos", 0), feat.get("fan_in_eth_pos", 0)
    else:
        fo, fi = feat.get("fan_out_eth", 0), feat.get("fan_in_eth", 0)
    if fo == 0 and fi == 0:
        fo, fi = feat.get("fan_out_erc20", 0), feat.get("fan_in_erc20", 0)
    if has_value is None:
        has_value = (fo + fi) > 0
    if not has_value or (fo == 0 and fi == 0):
        return "none"
    k = int(k)
    if fi == 1 and fo >= k:
        return "distribution"
    if fo == 1 and fi >= k:
        return "pooling"
    if fo >= 2 and fi >= 2:
        return "many_to_many"
    if fo == 1 and fi == 1:
        return "single"
    return "other"


def labels_frame(records: Sequence[TxRecord], k: int = 5, value_only: bool = False) -> pd.DataFrame:
    """One row per record: :func:`sharing_features` plus ``label`` (input of :func:`probe_sets`)."""
    rows = []
    for r in records:
        f = sharing_features(r)
        f["label"] = sharing_label(f, k=k, value_only=value_only)
        rows.append(f)
    return pd.DataFrame(rows)


def _gas_deciles(gas: pd.Series, bins: int = 10) -> pd.Series:
    """Quantile bin 0..bins-1 by rank (ties broken by position), robust to repeated values."""
    n = len(gas)
    bins = max(1, int(bins))
    if n == 0:
        return pd.Series([], dtype=int, index=gas.index)
    ranks = gas.rank(method="first").to_numpy() - 1
    return pd.Series(np.minimum((ranks * bins // n).astype(int), bins - 1), index=gas.index)


def probe_sets(records: Sequence[TxRecord], labels_df: Optional[pd.DataFrame], kind: str, n: int,
               seed: int = 0, k: int = 5, eras: Optional[Sequence[str]] = None, gas_bins: int = 10):
    """Matched positive / negative record sets for a sharing probe.

    * ``kind="fanout"``: positives ``fan_out_eth >= k``, negatives ``fan_out_eth == 1``;
    * ``kind="dao"``: positives ``dao_involved``, negatives not, restricted to ``eras``
      (default ``("drain_crisis", "post_fork")``).

    Pairs are matched 1:1 within (era, gas_used decile) strata (deciles over the candidate
    pool; inside a stratum the sampled members are paired by gas rank; ``gas_bins=1``
    matches on era only); at most ``n`` pairs, sampled with ``seed``.  Returns ``(positive, negative,
    match_table)`` with ``match_table`` columns pair, era, gas_decile, pos_tx, neg_tx,
    pos_gas, neg_gas, pos_fan_out_eth, neg_fan_out_eth, pos_dao_involved, neg_dao_involved.
    """
    records = list(records)
    df = labels_frame(records, k=k) if labels_df is None else labels_df.copy()
    if len(df) != len(records):
        raise ValueError("labels_df must have one row per record (same order)")
    if "tx_hash" in df.columns:
        rec_hashes = [r.tx_hash for r in records]
        if list(df["tx_hash"]) != rec_hashes:
            pos_of = {h: i for i, h in enumerate(rec_hashes)}
            if set(df["tx_hash"]) != set(rec_hashes):
                raise ValueError("labels_df tx_hash values do not match the records")
            df = df.assign(_rec=[pos_of[h] for h in df["tx_hash"]])
    if "_rec" not in df.columns:
        df = df.assign(_rec=np.arange(len(df)))
    df = df.reset_index(drop=True)
    if kind == "fanout":
        is_pos = df["fan_out_eth"] >= int(k)
        is_neg = df["fan_out_eth"] == 1
        if eras is not None:
            keep = df["era"].isin(list(eras))
            is_pos, is_neg = is_pos & keep, is_neg & keep
    elif kind == "dao":
        eras = ("drain_crisis", "post_fork") if eras is None else tuple(eras)
        keep = df["era"].isin(list(eras))
        is_pos = df["dao_involved"].astype(bool) & keep
        is_neg = ~df["dao_involved"].astype(bool) & keep
    else:
        raise ValueError(f"kind must be 'fanout' or 'dao', got {kind!r}")
    pool = df[is_pos | is_neg].copy()
    pool["_pos"] = is_pos[pool.index].to_numpy()
    pool["gas_decile"] = _gas_deciles(pool["gas_used"].astype(float), gas_bins)
    rng = np.random.default_rng(int(seed))
    pairs = []
    for (era, dec), grp in pool.groupby(["era", "gas_decile"], sort=True):
        p = grp[grp["_pos"]].index.to_numpy()
        q = grp[~grp["_pos"]].index.to_numpy()
        m = min(len(p), len(q))
        if m == 0:
            continue
        p = rng.permutation(p)[:m]
        q = rng.permutation(q)[:m]
        # within the stratum pair by gas rank (closest matching for the sampled members)
        p = sorted(p, key=lambda i: (float(pool.at[i, "gas_used"]), int(i)))
        q = sorted(q, key=lambda i: (float(pool.at[i, "gas_used"]), int(i)))
        pairs += [(era, int(dec), int(a), int(b)) for a, b in zip(p, q)]
    if len(pairs) > int(n):
        idx = np.sort(rng.choice(len(pairs), size=int(n), replace=False))
        pairs = [pairs[i] for i in idx]
    rows, pos_recs, neg_recs = [], [], []
    for i, (era, dec, a, b) in enumerate(pairs):
        ra, rb = df.loc[a], df.loc[b]
        pos_recs.append(records[int(ra["_rec"])])
        neg_recs.append(records[int(rb["_rec"])])
        rows.append(dict(pair=i, era=era, gas_decile=dec, pos_tx=ra.get("tx_hash"), neg_tx=rb.get("tx_hash"),
                         pos_gas=int(ra["gas_used"]), neg_gas=int(rb["gas_used"]),
                         pos_fan_out_eth=int(ra["fan_out_eth"]), neg_fan_out_eth=int(rb["fan_out_eth"]),
                         pos_dao_involved=bool(ra["dao_involved"]), neg_dao_involved=bool(rb["dao_involved"])))
    cols = ["pair", "era", "gas_decile", "pos_tx", "neg_tx", "pos_gas", "neg_gas", "pos_fan_out_eth",
            "neg_fan_out_eth", "pos_dao_involved", "neg_dao_involved"]
    return pos_recs, neg_recs, pd.DataFrame(rows, columns=cols)
