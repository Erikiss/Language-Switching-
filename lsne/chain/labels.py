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

**Length-matched probe sets** (round 2, docs/TRACE_SOURCE.md "Kontrollierte Sonden"):
``probe_sets(..., match="length")`` pairs every positive with a negative of the same era
whose ITR token count lies within a factor ``length_ratio`` (default 1.25, i.e.
``|log n_pos - log n_neg| <= log 1.25``), nearest length first, without replacement.  The
first run showed that gas-decile matching leaves the high fan-out positives 12x longer than
their negatives, so a fan-out "direction" was a length direction.  Balance is reported as
standardized mean differences (``length_smd`` on log token count, ``n_tokens_smd``,
``gas_smd``, ``log_gas_smd``; all available covariates in ``match_table.attrs["balance"]``).

**Identity masking** (:func:`mask_tokens`): the DAO probe separated the sets through the
DAO address tokens themselves (always in the vocabulary).  ``mask_tokens(tokens, "dao")``
replaces the four DAO identities (DAO, extraBalance, creator, WithdrawDAO) by ``[A_OOV]``
before encoding, in *both* sets, so the separation has to come from context;
``"all_addresses"`` replaces every address token (stricter: also removes the ``[A_OOV]``
count difference the DAO masking itself creates).

**Surface baseline** (:func:`surface_baseline_auc`): split-half AUC of trivial length /
count statistics of the token sequence (token count, ``[CALL]`` count, ``V:`` token count,
value-carrying calls; optionally the ``[A_OOV]`` count), with exactly the split of
:func:`lsne.chain.source.holdout_auc`, so a sharing direction can be compared with what
length and counts alone achieve.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .itr import TxRecord, iter_nodes, serialize

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
#: surface features for masked probes: the ``[A_OOV]`` count is a shortcut masking can create
SURFACE_FEATURES_OOV = SURFACE_FEATURES + ("n_a_oov",)
LENGTH_RATIO = 1.25
MATCH_KINDS = ("gas", "length")
MASKS = ("none", "dao", "all_addresses")
MASK_TOKEN = "[A_OOV]"
#: covariates whose balance (SMD) is reported for matched probe sets when available
BALANCE_COVARIATES = ("n_tokens", "gas_used", "n_nodes", "n_value_calls", "max_depth", "n_state_writes")
_DAO_SET = frozenset(DAO_ADDRESSES.values())
_DAO_TOKENS = frozenset(f"A:{a}" for a in DAO_ADDRESSES.values())


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


def _match_pairs_length(pool: pd.DataFrame, ratio: float, rng: np.random.Generator, length_col: str) -> tuple:
    """1:1 nearest-length matching within era with a caliper ``|log n_pos - log n_neg| <= log ratio``.

    Negatives of equal length form one bucket whose members are drawn in a seeded random
    order (many simple transactions share a token count); positives are processed in order
    of increasing number of admissible negatives (hardest first, ties in seeded random
    order), each takes a member of the nearest non-empty length bucket inside the caliper.
    Returns ``(pairs, n_unmatched)`` with pairs ``(era, -1, pos_index, neg_index)``."""
    caliper = float(np.log(float(ratio))) + 1e-12
    pairs, n_unmatched = [], 0
    for era, grp in pool.groupby("era", sort=True):
        P = grp[grp["_pos"]]
        Q = grp[~grp["_pos"]]
        if P.empty:
            continue
        if Q.empty:
            n_unmatched += len(P)
            continue
        qlen = np.maximum(Q[length_col].to_numpy(float), 1.0)
        uniq, inv = np.unique(qlen, return_inverse=True)
        log_u = np.log(uniq)
        qidx = Q.index.to_numpy()
        counts = np.bincount(inv, minlength=len(uniq))
        cum = np.concatenate([[0], np.cumsum(counts)])
        grouped = qidx[np.argsort(inv, kind="stable")]
        buckets = [list(rng.permutation(grouped[cum[j]:cum[j + 1]])) for j in range(len(uniq))]
        plog = np.log(np.maximum(P[length_col].to_numpy(float), 1.0))
        lo = np.searchsorted(log_u, plog - caliper, side="left")
        hi = np.searchsorted(log_u, plog + caliper, side="right")
        n_cand = cum[hi] - cum[lo]
        order = np.lexsort((rng.permutation(len(P)), n_cand))      # fewest candidates first, random ties
        pidx = P.index.to_numpy()
        for i in order:
            js = np.arange(lo[i], hi[i])
            best = None
            for j in js[np.argsort(np.abs(log_u[js] - plog[i]), kind="stable")]:
                if buckets[j]:
                    best = j
                    break
            if best is None:
                n_unmatched += 1
                continue
            pairs.append((era, -1, int(pidx[i]), int(buckets[best].pop())))
    return pairs, n_unmatched


def balance_table(pos_df: pd.DataFrame, neg_df: pd.DataFrame,
                  covariates: Sequence[str] = BALANCE_COVARIATES) -> dict:
    """Standardized mean differences (positives minus negatives) of matched sets for every
    available covariate, plus the log-scale SMD for the heavy-tailed counts
    (``log_<cov>`` = SMD of ``log(max(x, 1))``) and the means of both sides."""
    out = {}
    for c in covariates:
        if c not in pos_df.columns or c not in neg_df.columns:
            continue
        a = pos_df[c].to_numpy(float)
        b = neg_df[c].to_numpy(float)
        out[c] = dict(smd=standardized_mean_difference(a, b),
                      log_smd=standardized_mean_difference(np.log(np.maximum(a, 1.0)), np.log(np.maximum(b, 1.0))),
                      mean_pos=float(np.mean(a)) if len(a) else float("nan"),
                      mean_neg=float(np.mean(b)) if len(b) else float("nan"),
                      median_pos=float(np.median(a)) if len(a) else float("nan"),
                      median_neg=float(np.median(b)) if len(b) else float("nan"))
    return out


def probe_sets(records: Sequence[TxRecord], labels_df: Optional[pd.DataFrame], kind: str, n: int,
               seed: int = 0, k: int = 5, eras: Optional[Sequence[str]] = None, gas_bins: int = 10,
               min_pairs: Optional[int] = None, match: str = "gas", length_ratio: float = LENGTH_RATIO,
               length_col: str = "n_tokens", length_cap: Optional[int] = None):
    """Matched positive / negative record sets for a sharing probe.

    * ``kind="fanout"``: positives ``fan_out_eth >= k``, negatives ``fan_out_eth == 1``
      (economic fan-out, value > 0 transfers);
    * ``kind="dao"``: positives ``dao_involved``, negatives not, restricted to ``eras``
      (default ``("drain_crisis", "post_fork")``).

    ``match="gas"`` (default, the spec): pairs are matched 1:1 within (era, gas_used decile)
    strata (deciles over the candidate pool; inside a stratum the sampled members are paired
    by gas rank).  When that yields fewer than ``min_pairs`` pairs (default ``min(n, 20)``)
    the matching falls back to era only (``gas_bins=1`` requests era-only matching directly).

    ``match="length"``: 1:1 within era on the log ITR token count (column ``length_col`` of
    ``labels_df``, computed by serialising the records when absent; ``length_cap`` matches
    on ``min(n_tokens, cap)``, the length the model sees), caliper
    ``|log n_pos - log n_neg| <= log(length_ratio)``, nearest length first, without
    replacement (:func:`_match_pairs_length`); positives without an admissible negative stay
    unmatched (counted in ``attrs["n_pos_unmatched"]``) -- there is no silent fallback.

    At most ``n`` pairs, sampled with ``seed``.  Returns ``(positive, negative,
    match_table)``; ``match_table`` columns: pair, era, gas_decile (-1 for length matching),
    pos_tx, neg_tx, pos_gas, neg_gas, pos_fan_out_eth, neg_fan_out_eth, pos_dao_involved,
    neg_dao_involved, matching ("era+gas_decile", "era_only" or "era+log_length"), gas_smd
    and log_gas_smd (standardized mean difference of gas_used resp. log gas_used, positives
    minus negatives, over all matched pairs; the same value in every row), and when token
    counts are available pos_n_tokens, neg_n_tokens, length_smd (SMD of log token count) and
    n_tokens_smd.  ``match_table.attrs`` repeats the summary values and holds ``balance``
    (:func:`balance_table` over the available covariates), the pool sizes and the matching
    parameters.
    """
    if match not in MATCH_KINDS:
        raise ValueError(f"match must be one of {MATCH_KINDS}, got {match!r}")
    records = list(records)
    df = labels_frame(records, k=k) if labels_df is None else labels_df.copy()
    if len(df) != len(records):
        raise ValueError("labels_df must have one row per record (same order)")
    if match == "length" and length_col not in df.columns:
        if not all(isinstance(r, TxRecord) for r in records):
            raise ValueError(f"length matching needs a {length_col!r} column in labels_df (or TxRecord records)")
        df[length_col] = [len(serialize(r)) for r in records]
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
    n_unmatched = None
    if match == "length":
        matching = "era+log_length"
        if length_cap is not None:
            pool["_len"] = np.minimum(pool[length_col].to_numpy(float), float(length_cap))
        else:
            pool["_len"] = pool[length_col].to_numpy(float)
        pairs, n_unmatched = _match_pairs_length(pool, float(length_ratio), rng, "_len")
    else:
        matching = "era+gas_decile" if gas_bins > 1 else "era_only"
        pairs = _match_pairs(pool, gas_bins, rng)
        if gas_bins > 1 and len(pairs) < min_pairs:
            era_pairs = _match_pairs(pool, 1, np.random.default_rng(int(seed)))
            if len(era_pairs) > len(pairs):
                pairs, matching = era_pairs, "era_only"
    n_matched_total = len(pairs)
    if len(pairs) > int(n):
        idx = np.sort(rng.choice(len(pairs), size=int(n), replace=False))
        pairs = [pairs[i] for i in idx]
    has_len = length_col in df.columns
    rows, pos_recs, neg_recs = [], [], []
    for i, (era, dec, a, b) in enumerate(pairs):
        ra, rb = df.loc[a], df.loc[b]
        pos_recs.append(records[int(ra["_rec"])])
        neg_recs.append(records[int(rb["_rec"])])
        row = dict(pair=i, era=era, gas_decile=dec, pos_tx=ra.get("tx_hash"), neg_tx=rb.get("tx_hash"),
                   pos_gas=int(ra["gas_used"]), neg_gas=int(rb["gas_used"]),
                   pos_fan_out_eth=int(ra["fan_out_eth"]), neg_fan_out_eth=int(rb["fan_out_eth"]),
                   pos_dao_involved=bool(ra["dao_involved"]), neg_dao_involved=bool(rb["dao_involved"]))
        if has_len:
            row.update(pos_n_tokens=int(ra[length_col]), neg_n_tokens=int(rb[length_col]))
        rows.append(row)
    cols = ["pair", "era", "gas_decile", "pos_tx", "neg_tx", "pos_gas", "neg_gas", "pos_fan_out_eth",
            "neg_fan_out_eth", "pos_dao_involved", "neg_dao_involved"]
    len_cols = ["pos_n_tokens", "neg_n_tokens"] if has_len else []
    mt = pd.DataFrame(rows, columns=cols + len_cols)
    pg, ng = mt["pos_gas"].to_numpy(float), mt["neg_gas"].to_numpy(float)
    smd = standardized_mean_difference(pg, ng)
    log_smd = standardized_mean_difference(np.log(np.maximum(pg, 1.0)), np.log(np.maximum(ng, 1.0)))
    mt = mt.assign(matching=matching, gas_smd=smd, log_gas_smd=log_smd)
    summary = dict(matching=matching, gas_smd=smd, log_gas_smd=log_smd)
    if has_len:
        pl, nl = mt["pos_n_tokens"].to_numpy(float), mt["neg_n_tokens"].to_numpy(float)
        summary.update(length_smd=standardized_mean_difference(np.log(np.maximum(pl, 1.0)), np.log(np.maximum(nl, 1.0))),
                       n_tokens_smd=standardized_mean_difference(pl, nl))
        mt = mt.assign(length_smd=summary["length_smd"], n_tokens_smd=summary["n_tokens_smd"])
        if len(mt):
            lr = np.abs(np.log(np.maximum(pl, 1.0)) - np.log(np.maximum(nl, 1.0)))
            summary.update(length_abs_log_ratio_median=float(np.median(lr)), length_ratio_max=float(np.exp(lr.max())))
    pos_rows = df.loc[[p[2] for p in pairs]] if pairs else df.iloc[:0]
    neg_rows = df.loc[[p[3] for p in pairs]] if pairs else df.iloc[:0]
    mt.attrs.update(summary)
    mt.attrs.update(n_pairs=len(mt), min_pairs=min_pairs, gas_bins=gas_bins, match=match,
                    length_ratio=float(length_ratio) if match == "length" else None,
                    length_cap=None if length_cap is None else int(length_cap),
                    pool_pos=int(pool["_pos"].sum()), pool_neg=int((~pool["_pos"]).sum()),
                    n_matched_total=int(n_matched_total), n_pos_unmatched=n_unmatched,
                    eras=mt["era"].value_counts().to_dict(),
                    balance=balance_table(pos_rows, neg_rows))
    return pos_recs, neg_recs, mt


# ======================================================================================
# identity masking
# ======================================================================================
def mask_tokens(tokens: Sequence[str], mask: Union[str, Iterable[str], None] = "dao",
                replacement: str = MASK_TOKEN) -> list:
    """Replace identity tokens of an ITR token list (raw or vocabulary-mapped) by ``[A_OOV]``.

    * ``"none"`` / ``None``: unchanged copy;
    * ``"dao"``: the four DAO identities of :data:`DAO_ADDRESSES` (DAO, extraBalance,
      creator, WithdrawDAO) wherever they occur (call endpoints, arguments, state writes /
      reads, log emitters);
    * ``"all_addresses"``: every address token ``A:0x…`` (``A:none`` -- no address -- stays);
    * an iterable of addresses (``0x…``, any case) or ``A:``-tokens: exactly those.

    Masking is applied to positives and negatives alike, before encoding."""
    toks = [str(t) for t in tokens]
    if mask is None or (isinstance(mask, str) and mask == "none"):
        return toks
    if isinstance(mask, str) and mask == "dao":
        targets = _DAO_TOKENS
    elif isinstance(mask, str) and mask == "all_addresses":
        return [replacement if (t.startswith("A:") and t != "A:none") else t for t in toks]
    elif isinstance(mask, str):
        raise ValueError(f"mask must be one of {MASKS} or an iterable of addresses, got {mask!r}")
    else:
        targets = frozenset("A:" + (str(a)[2:] if str(a).startswith("A:") else str(a)).lower() for a in mask)
    return [replacement if t in targets else t for t in toks]


# ======================================================================================
# surface-feature baseline
# ======================================================================================
def surface_features(tokens: Sequence[str], max_len: Optional[int] = None) -> dict:
    """Trivial length / count statistics of an ITR token list (raw or vocabulary-mapped;
    ``max_len``: only the first ``max_len`` tokens, i.e. what the model sees): ``n_tokens``,
    ``n_call`` (``[CALL]``), ``n_value_tokens`` (every ``V:`` token: calls, state writes and
    reads), ``n_value_calls_nonzero`` (``[CALL]`` entries whose ``V:`` token is not ``V:N0``)
    and ``n_a_oov`` (``[A_OOV]`` tokens; 0 for raw tokens unless masked)."""
    toks = list(tokens)[: int(max_len)] if max_len is not None else list(tokens)
    n_call = n_val = n_nz = n_oov = 0
    in_call = False
    for t in toks:
        if t == MASK_TOKEN:
            n_oov += 1
        elif t == "[CALL]":
            n_call += 1
            in_call = True
        elif t == "[END]" or t == "[START]":
            in_call = False
        elif t.startswith("V:"):
            n_val += 1
            if in_call and t != "V:N0":
                n_nz += 1
            in_call = False
    return dict(n_tokens=len(toks), n_call=n_call, n_value_tokens=n_val, n_value_calls_nonzero=n_nz,
                n_a_oov=n_oov)


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
