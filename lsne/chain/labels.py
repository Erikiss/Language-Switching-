"""'Sharing' labels, DAO-era segmentation, matched probe sets and the surface baseline.

**Economic fan-out / fan-in** (docs/TRACE_SOURCE_SPEC.md §4, DESIGN §4.9): ``fan_out_eth`` /
``fan_in_eth`` are the numbers of distinct receivers / senders over the transaction's
``native_transfers`` rows **with value > 0**, ``fan_out_erc20`` / ``fan_in_erc20`` the same over
the ``erc20_transfers`` rows with value > 0.  Xatu writes one ``native_transfers`` row per
trace node, *including zero-value calls*; counting every row measures call targets, not
economic transfers (the DAO drain: 81 rows with 9 receivers / 6 senders, of which only 20
rows carry ETH: 3 receivers, 4 senders).  Those structural counts are kept as
``n_call_targets`` / ``n_call_sources`` (they reproduce the §0 statistics of chunk 1718000:
median 1, 99th percentile 6, max 51) and can drive the label with
``sharing_label(..., structural=True)``.  ``"none"`` (no value transfer) means: no native
transfer with value > 0 and no ERC-20 transfer with value > 0.

**Probe sets** (:func:`probe_sets`) are matched 1:1 on era and gas_used decile; when that
yields fewer than ``min_pairs`` pairs the matching falls back to era only, and the match
table records which matching was used and the standardized mean difference of gas
(``gas_smd``, ``log_gas_smd``) between the matched positives and negatives.

**Surface baseline** (:func:`surface_baseline_auc`): split-half AUC of trivial length /
count statistics of the token sequence (token count, ``[CALL]`` count, ``V:`` token count,
value-carrying calls), with exactly the split of :func:`lsne.chain.source.holdout_auc`, so a
sharing direction can be compared with what length and counts alone achieve.
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
FAN_DEFINITION = "value_gt_0"
SURFACE_FEATURES = ("n_tokens", "n_call", "n_value_tokens", "n_value_calls_nonzero")
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

    Spec keys: fan_out_eth, fan_in_eth, fan_out_erc20, fan_in_erc20 (economic: value > 0
    rows only), n_value_calls, n_state_writes, n_distinct_slots_written, dao_involved,
    withdraw_dao_involved, max_depth, n_nodes, era.  Structural keys: n_call_targets,
    n_call_sources (distinct receivers / senders over *all* native_transfers rows, i.e. one
    per trace node including zero-value calls).  Extra keys: n_native_transfers,
    n_native_transfers_pos, n_erc20_transfers, total_eth, has_value_transfer, fan_definition,
    tx_hash, block_number, gas_used, success.
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
        fan_out_eth=len({lower(t.get("to_address")) for t in pos}),
        fan_in_eth=len({lower(t.get("from_address")) for t in pos}),
        fan_out_erc20=len({lower(t.get("to_address")) for t in erc_pos}),
        fan_in_erc20=len({lower(t.get("from_address")) for t in erc_pos}),
        n_value_calls=sum(1 for n in nodes if n.value > 0),
        n_state_writes=len(writes),
        n_distinct_slots_written=len({(lower(w.get("address")), lower(w.get("slot"))) for w in writes}),
        dao_involved=bool(touched & _DAO_SET),
        withdraw_dao_involved=DAO_ADDRESSES["withdraw_dao"] in touched,
        max_depth=int(rec.max_depth),
        n_nodes=int(rec.n_nodes),
        era=era_of(rec.block_number),
        n_call_targets=len({lower(t.get("to_address")) for t in nat}),
        n_call_sources=len({lower(t.get("from_address")) for t in nat}),
        n_native_transfers=len(nat),
        n_native_transfers_pos=len(pos),
        n_erc20_transfers=len(erc),
        total_eth=sum(int(t["value"]) for t in pos) / 1e18,
        has_value_transfer=bool(pos or erc_pos),
        fan_definition=FAN_DEFINITION,
        tx_hash=rec.tx_hash,
        block_number=int(rec.block_number),
        gas_used=int(rec.gas_used),
        success=bool(rec.success),
    )


def sharing_label(feat: dict, k: int = 5, structural: bool = False) -> str:
    """"distribution" (one sender, fan_out >= k), "pooling" (one receiver, fan_in >= k),
    "many_to_many" (fan_out >= 2 and fan_in >= 2), "single" (fan_out == fan_in == 1),
    "none" (no value transfer), else "other".  Counts from the value-carrying native
    transfers (``structural=True``: from all native rows, ``n_call_targets`` /
    ``n_call_sources``); without native value transfers the value-carrying ERC-20 counts."""
    has_value = feat.get("has_value_transfer")
    if structural:
        fo, fi = feat.get("n_call_targets", 0), feat.get("n_call_sources", 0)
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


def labels_frame(records: Sequence[TxRecord], k: int = 5, structural: bool = False) -> pd.DataFrame:
    """One row per record: :func:`sharing_features` plus ``label`` (input of :func:`probe_sets`)."""
    rows = []
    for r in records:
        f = sharing_features(r)
        f["label"] = sharing_label(f, k=k, structural=structural)
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


def standardized_mean_difference(a, b) -> float:
    """``(mean(a) - mean(b)) / sqrt((var(a) + var(b)) / 2)`` (sample variances); NaN when
    either side has fewer than 2 values, 0 when both sides are constant and equal."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    diff = float(a.mean() - b.mean())
    sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    if sd == 0.0:
        return 0.0 if diff == 0.0 else float(np.sign(diff) * np.inf)
    return diff / sd


def _match_pairs(pool: pd.DataFrame, gas_bins: int, rng: np.random.Generator) -> list:
    pool = pool.copy()
    pool["gas_decile"] = _gas_deciles(pool["gas_used"].astype(float), gas_bins)
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
    return pairs


def probe_sets(records: Sequence[TxRecord], labels_df: Optional[pd.DataFrame], kind: str, n: int,
               seed: int = 0, k: int = 5, eras: Optional[Sequence[str]] = None, gas_bins: int = 10,
               min_pairs: Optional[int] = None):
    """Matched positive / negative record sets for a sharing probe.

    * ``kind="fanout"``: positives ``fan_out_eth >= k``, negatives ``fan_out_eth == 1``
      (economic fan-out, value > 0 transfers);
    * ``kind="dao"``: positives ``dao_involved``, negatives not, restricted to ``eras``
      (default ``("drain_crisis", "post_fork")``).

    Pairs are matched 1:1 within (era, gas_used decile) strata (deciles over the candidate
    pool; inside a stratum the sampled members are paired by gas rank).  When that yields
    fewer than ``min_pairs`` pairs (default ``min(n, 20)``) the matching falls back to era
    only (``gas_bins=1`` requests era-only matching directly).  At most ``n`` pairs, sampled
    with ``seed``.  Returns ``(positive, negative, match_table)``; ``match_table`` columns:
    pair, era, gas_decile, pos_tx, neg_tx, pos_gas, neg_gas, pos_fan_out_eth,
    neg_fan_out_eth, pos_dao_involved, neg_dao_involved, matching ("era+gas_decile" or
    "era_only"), gas_smd and log_gas_smd (standardized mean difference of gas_used resp.
    log gas_used, positives minus negatives, over all matched pairs; the same value in every
    row; also in ``match_table.attrs``).
    """
    records = list(records)
    df = labels_frame(records, k=k) if labels_df is None else labels_df.copy()
    if len(df) != len(records):
        raise ValueError("labels_df must have one row per record (same order)")
    if "n_call_targets" not in df.columns and "fan_out_eth_pos" in df.columns:
        raise ValueError("labels_df comes from an older serialisation whose fan_out_eth counted zero-value call "
                         "rows; re-run `python -m lsne.chain.cli serialize ... --overwrite`")
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
    min_pairs = min(int(n), 20) if min_pairs is None else int(min_pairs)
    rng = np.random.default_rng(int(seed))
    gas_bins = max(1, int(gas_bins))
    matching = "era+gas_decile" if gas_bins > 1 else "era_only"
    pairs = _match_pairs(pool, gas_bins, rng)
    if gas_bins > 1 and len(pairs) < min_pairs:
        era_pairs = _match_pairs(pool, 1, np.random.default_rng(int(seed)))
        if len(era_pairs) > len(pairs):
            pairs, matching = era_pairs, "era_only"
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
            "neg_fan_out_eth", "pos_dao_involved", "neg_dao_involved", "matching", "gas_smd", "log_gas_smd"]
    mt = pd.DataFrame(rows, columns=cols[:-3])
    pg, ng = mt["pos_gas"].to_numpy(float), mt["neg_gas"].to_numpy(float)
    smd = standardized_mean_difference(pg, ng)
    log_smd = standardized_mean_difference(np.log(np.maximum(pg, 1.0)), np.log(np.maximum(ng, 1.0)))
    mt = mt.assign(matching=matching, gas_smd=smd, log_gas_smd=log_smd)[cols]
    mt.attrs.update(matching=matching, gas_smd=smd, log_gas_smd=log_smd, n_pairs=len(mt), min_pairs=min_pairs,
                    gas_bins=gas_bins)
    return pos_recs, neg_recs, mt


# ======================================================================================
# surface-feature baseline
# ======================================================================================
def surface_features(tokens: Sequence[str], max_len: Optional[int] = None) -> dict:
    """Trivial length / count statistics of an ITR token list (raw or vocabulary-mapped;
    ``max_len``: only the first ``max_len`` tokens, i.e. what the model sees): ``n_tokens``,
    ``n_call`` (``[CALL]``), ``n_value_tokens`` (every ``V:`` token: calls, state writes and
    reads) and ``n_value_calls_nonzero`` (``[CALL]`` entries whose ``V:`` token is not ``V:N0``)."""
    toks = list(tokens)[: int(max_len)] if max_len is not None else list(tokens)
    n_call = n_val = n_nz = 0
    in_call = False
    for t in toks:
        if t == "[CALL]":
            n_call += 1
            in_call = True
        elif t == "[END]" or t == "[START]":
            in_call = False
        elif t.startswith("V:"):
            n_val += 1
            if in_call and t != "V:N0":
                n_nz += 1
            in_call = False
    return dict(n_tokens=len(toks), n_call=n_call, n_value_tokens=n_val, n_value_calls_nonzero=n_nz)


def _auc(p: np.ndarray, q: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float((p[:, None] > q[None, :]).mean() + 0.5 * (p[:, None] == q[None, :]).mean())


def surface_baseline_auc(pos_tokens: Sequence[Sequence[str]], neg_tokens: Sequence[Sequence[str]], seed: int = 0,
                         max_len: Optional[int] = None,
                         features: Sequence[str] = SURFACE_FEATURES) -> dict:
    """Split-half AUC of the surface features of matched positive / negative token lists.

    The split is exactly the one of :func:`lsne.chain.source.holdout_auc` with the same
    ``seed`` (``default_rng(seed).permutation(min(n_pos, n_neg))``, two halves, fit on one,
    evaluate on the other, mean over both directions).  Per feature the orientation (sign)
    is fitted on the training half; ``"combined"`` is the difference of class means of the
    z-scored features (training-half statistics), the same estimator as the sharing
    direction.  Returns ``{feature: auc, ..., "combined": auc, "best_single": name}``
    (NaN with fewer than 4 pairs)."""
    n = min(len(pos_tokens), len(neg_tokens))
    feats = list(features)
    out = {f: float("nan") for f in feats}
    out.update(combined=float("nan"), best_single=None, n_pairs=int(n))
    if n < 4:
        return out
    F_pos = np.array([[surface_features(t, max_len)[f] for f in feats] for t in pos_tokens[:n]], dtype=float)
    F_neg = np.array([[surface_features(t, max_len)[f] for f in feats] for t in neg_tokens[:n]], dtype=float)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    per = {f: [] for f in feats}
    comb = []
    for a, b in (halves, halves[::-1]):
        mp, mn = F_pos[a].mean(0), F_neg[a].mean(0)
        for j, f in enumerate(feats):
            s = 1.0 if mp[j] >= mn[j] else -1.0
            per[f].append(_auc(s * F_pos[b, j], s * F_neg[b, j]))
        train = np.concatenate([F_pos[a], F_neg[a]])
        mu, sd = train.mean(0), train.std(0)
        sd[sd == 0] = 1.0
        w = (mp - mn) / sd
        if np.linalg.norm(w) > 0:
            comb.append(_auc(((F_pos[b] - mu) / sd) @ w, ((F_neg[b] - mu) / sd) @ w))
        else:
            comb.append(0.5)
    for f in feats:
        out[f] = float(np.mean(per[f]))
    out["combined"] = float(np.mean(comb))
    out["best_single"] = max(feats, key=lambda f: out[f])
    return out
