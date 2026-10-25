"""Statistics helpers: cluster (family) bootstrap, paired contrasts, Benjamini-Hochberg
correction, Cohen's d and bootstrap p-values.

Families are the statistical unit of the design (DESIGN §9): every confidence interval
is a *cluster* bootstrap that resamples whole clusters (families) with replacement and
recomputes the statistic over the concatenated rows.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd


# ======================================================================================
# cluster bootstrap
# ======================================================================================
def _cluster_arrays(df: pd.DataFrame, value_col: str, cluster_col: str) -> tuple[list[np.ndarray], list]:
    """Per-cluster value arrays (rows with NaN values dropped) and the cluster keys."""
    sub = df[[value_col, cluster_col]].dropna(subset=[value_col])
    groups, keys = [], []
    for key, g in sub.groupby(cluster_col, sort=True):
        groups.append(g[value_col].to_numpy(dtype=np.float64))
        keys.append(key)
    return groups, keys


def cluster_bootstrap(df: pd.DataFrame, value_col: str, cluster_col: str, n_boot: int = 10000,
                      seed: int = 0, stat: Callable = np.mean, ci: float = 0.95) -> dict:
    """Cluster bootstrap of ``stat`` over ``df[value_col]``.

    Clusters (``df[cluster_col]``) are resampled with replacement; for each resample the
    rows of the drawn clusters are concatenated and ``stat`` recomputed.  Rows with NaN
    values are dropped.  ``mean`` is the point estimate ``stat`` on the full data (not
    the bootstrap mean); ``ci_low``/``ci_high`` are percentile bounds at level ``ci``.

    Returns dict(mean, ci_low, ci_high, n_clusters, n_rows, se, boot) where ``boot`` is
    the array of bootstrap statistics (NaN-filled if there are no rows).
    """
    groups, _ = _cluster_arrays(df, value_col, cluster_col)
    n_clusters = len(groups)
    n_rows = int(sum(g.shape[0] for g in groups))
    n_boot = int(n_boot)
    if n_clusters == 0:
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan, n_clusters=0, n_rows=0, se=np.nan,
                    boot=np.full(n_boot, np.nan))
    all_values = np.concatenate(groups)
    point = float(stat(all_values))
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    if stat is np.mean:
        # fast path: mean of concatenated rows = sum of cluster sums / sum of cluster sizes
        sums = np.array([g.sum() for g in groups], dtype=np.float64)
        sizes = np.array([g.shape[0] for g in groups], dtype=np.float64)
        boot = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    else:
        boot = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            boot[i] = float(stat(np.concatenate([groups[j] for j in idx[i]])))
    alpha = (1.0 - float(ci)) / 2.0
    finite = boot[np.isfinite(boot)]
    if finite.size:
        lo, hi = (float(v) for v in np.percentile(finite, [100 * alpha, 100 * (1 - alpha)]))
        se = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
    else:
        lo = hi = se = np.nan
    return dict(mean=point, ci_low=lo, ci_high=hi, n_clusters=int(n_clusters), n_rows=n_rows,
                se=se, boot=boot)


# ======================================================================================
# paired contrast
# ======================================================================================
def _as_list(cols) -> list:
    if isinstance(cols, (str, bytes)):
        return [cols]
    return list(cols)


def paired_contrast(df: pd.DataFrame, value_col: str, cond_col: str, a, b, pair_cols, cluster_col: str,
                    n_boot: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    """Paired contrast ``a - b`` between two levels of ``cond_col`` with a cluster bootstrap.

    Rows with ``cond_col == a`` and ``cond_col == b`` are merged on ``pair_cols`` (inner
    join; duplicate rows per (pair, condition) are averaged first).  The per-pair
    difference ``value_a - value_b`` is then cluster-bootstrapped over ``cluster_col``
    (taken from the ``a`` rows if it is not one of the pair columns).

    Returns dict(mean, ci_low, ci_high, n_clusters, n_rows, n_pairs, se, p_boot, d, boot,
    mean_a, mean_b) where ``p_boot`` is the two-sided bootstrap p-value for a zero
    difference and ``d`` the paired Cohen's d (mean difference / sd of differences).
    """
    pair_cols = _as_list(pair_cols)
    cluster_in_pairs = cluster_col in pair_cols

    def side(level, tag):
        sub = df[df[cond_col] == level]
        g = sub.groupby(pair_cols, sort=False)
        agg = g[value_col].mean().rename(f"value_{tag}").reset_index()
        if not cluster_in_pairs:
            agg[f"cluster_{tag}"] = g[cluster_col].first().to_numpy()
        return agg

    da, db = side(a, "a"), side(b, "b")
    merged = da.merge(db, on=pair_cols, how="inner")
    cl = cluster_col if cluster_in_pairs else "cluster_a"
    merged["diff"] = merged["value_a"] - merged["value_b"]
    res = cluster_bootstrap(merged, "diff", cl, n_boot=n_boot, seed=seed, stat=np.mean, ci=ci)
    diffs = merged["diff"].to_numpy(dtype=np.float64)
    diffs = diffs[np.isfinite(diffs)]
    sd = float(diffs.std(ddof=1)) if diffs.size > 1 else np.nan
    d = float(diffs.mean() / sd) if diffs.size > 1 and sd > 0 else np.nan
    res.update(n_pairs=int(diffs.size), p_boot=bootstrap_p_two_sided(res["boot"], 0.0), d=d,
               mean_a=float(merged["value_a"].mean()) if len(merged) else np.nan,
               mean_b=float(merged["value_b"].mean()) if len(merged) else np.nan)
    return res


# ======================================================================================
# multiple comparisons, effect sizes, bootstrap p
# ======================================================================================
def bh_correct(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR).  NaN entries are left NaN and do not
    count towards the number of tests.  Adjusted values are monotone in the raw p-values
    and clipped to 1."""
    p = np.asarray(pvals, dtype=np.float64).reshape(-1)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    # adj_i = min_{j>=i} p_j m/j >= p_i holds exactly; the maximum removes 1-ulp rounding
    adj = np.clip(np.maximum(adj, pv[order]), 0.0, 1.0)
    res = np.empty(m)
    res[order] = adj
    out[ok] = res
    return out


def cohen_d(x, y) -> float:
    """Cohen's d between two independent samples with the pooled standard deviation
    ``sqrt(((n_x-1) s_x^2 + (n_y-1) s_y^2) / (n_x+n_y-2))``.  NaN if either sample has
    fewer than two values or the pooled sd is 0."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    nx, ny = x.size, y.size
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if not np.isfinite(pooled) or pooled == 0.0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled)


def bootstrap_p_two_sided(boot_samples, null: float = 0.0) -> float:
    """Two-sided percentile-bootstrap p-value for the hypothesis that the statistic equals
    ``null``: ``min(1, 2 * min(P(boot <= null), P(boot >= null)))`` with the
    ``(k + 1) / (n + 1)`` continuity correction.  NaN if no finite bootstrap values."""
    b = np.asarray(boot_samples, dtype=np.float64).reshape(-1)
    b = b[np.isfinite(b)]
    n = b.size
    if n == 0:
        return float("nan")
    le = (np.sum(b <= null) + 1.0) / (n + 1.0)
    ge = (np.sum(b >= null) + 1.0) / (n + 1.0)
    return float(min(1.0, 2.0 * min(le, ge)))
