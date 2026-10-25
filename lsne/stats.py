"""Statistics helpers: cluster (family) bootstrap, paired contrasts, Benjamini-Hochberg
correction, Cohen's d and bootstrap p-values.

Families are the statistical unit of the design (DESIGN §9): every confidence interval
is a *cluster* bootstrap that resamples whole clusters (families) with replacement.
Two estimators exist (``stat_level`` of :func:`cluster_bootstrap`):

``'cluster'``
    the statistic is first computed within every cluster (the family mean) and then
    over the drawn clusters: the mean of family means, every family weighing the same
    whatever its number of rows.  This is the pre-registered estimator of the
    confirmatory contrasts ("Familienmittel der Differenz" with a family-level cluster
    bootstrap, PREREGISTRATION §2) and the default of :func:`paired_contrast`.
``'row'``
    the statistic is recomputed over the concatenated rows of the drawn clusters (every
    row weighs the same, so a family with more rows weighs more); the default of
    :func:`cluster_bootstrap`, kept for the descriptive tables.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

STAT_LEVELS = ("row", "cluster")


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
                      seed: int = 0, stat: Callable = np.mean, ci: float = 0.95,
                      stat_level: str = "row") -> dict:
    """Cluster bootstrap of ``stat`` over ``df[value_col]``.

    Clusters (``df[cluster_col]``) are resampled with replacement ``n_boot`` times;
    ``stat_level`` selects the estimator (module notes):

    * ``'row'``: for each resample the rows of the drawn clusters are concatenated and
      ``stat`` recomputed over them (every row weighs the same);
    * ``'cluster'``: ``stat`` is first applied within every cluster (``cluster_values``,
      one value per cluster, e.g. the family mean) and then over the drawn cluster
      values (every cluster weighs the same) -- the pre-registered estimator of the
      confirmatory contrasts (family mean of the per-family statistic).

    Rows with NaN values are dropped.  ``mean`` is the point estimate (``stat`` on the
    full data at the chosen level, not the bootstrap mean); ``ci_low``/``ci_high`` are
    percentile bounds at level ``ci``.  The same ``seed`` draws the same clusters at
    either level.

    Returns dict(mean, ci_low, ci_high, n_clusters, n_rows, se, boot, stat_level,
    cluster_values, cluster_keys) where ``boot`` is the array of bootstrap statistics
    (NaN-filled if there are no rows) and ``cluster_values`` the per-cluster ``stat``
    (one per entry of ``cluster_keys``, sorted).
    """
    if stat_level not in STAT_LEVELS:
        raise ValueError(f"stat_level must be one of {STAT_LEVELS}, got {stat_level!r}")
    groups, keys = _cluster_arrays(df, value_col, cluster_col)
    n_clusters = len(groups)
    n_rows = int(sum(g.shape[0] for g in groups))
    n_boot = int(n_boot)
    if n_clusters == 0:
        return dict(mean=np.nan, ci_low=np.nan, ci_high=np.nan, n_clusters=0, n_rows=0, se=np.nan,
                    boot=np.full(n_boot, np.nan), stat_level=stat_level, cluster_values=np.zeros(0),
                    cluster_keys=[])
    cluster_values = np.array([float(stat(g)) for g in groups], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    if stat_level == "cluster":
        point = float(stat(cluster_values))
        if stat is np.mean:
            boot = cluster_values[idx].mean(axis=1)
        else:
            boot = np.array([float(stat(cluster_values[idx[i]])) for i in range(n_boot)], dtype=np.float64)
    else:
        point = float(stat(np.concatenate(groups)))
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
                se=se, boot=boot, stat_level=stat_level, cluster_values=cluster_values, cluster_keys=keys)


# ======================================================================================
# paired contrast
# ======================================================================================
def _as_list(cols) -> list:
    if isinstance(cols, (str, bytes)):
        return [cols]
    return list(cols)


def paired_contrast(df: pd.DataFrame, value_col: str, cond_col: str, a, b, pair_cols, cluster_col: str,
                    n_boot: int = 10000, seed: int = 0, ci: float = 0.95, stat_level: str = "cluster") -> dict:
    """Paired contrast ``a - b`` between two levels of ``cond_col`` with a cluster bootstrap.

    Rows with ``cond_col == a`` and ``cond_col == b`` are merged on ``pair_cols`` (inner
    join; duplicate rows per (pair, condition) are averaged first).  Pairs present on
    one side only are dropped and counted: ``n_dropped_a`` / ``n_dropped_b`` are the
    aggregated pair keys of the ``a`` / ``b`` side without a partner.  The per-pair
    difference ``value_a - value_b`` is then cluster-bootstrapped over ``cluster_col``
    at ``stat_level`` (:func:`cluster_bootstrap`; default ``'cluster'``, the family mean
    of the per-family mean difference -- the pre-registered confirmatory estimator;
    ``'row'`` weighs every pair the same).  The cluster of a pair is ``cluster_col``
    itself when it is one of the pair columns, otherwise the cluster of its ``a`` rows; a
    pair whose rows disagree on the cluster (within one side or between the sides)
    triggers a ``UserWarning``, since the bootstrap then resamples it under the ``a``
    cluster only.

    Returns dict(mean, ci_low, ci_high, n_clusters, n_rows, n_pairs, n_dropped_a,
    n_dropped_b, se, p_boot, d, boot, mean_a, mean_b, stat_level, cluster_values,
    cluster_keys) where ``n_pairs`` counts the matched pairs with a finite difference,
    ``p_boot`` is the two-sided bootstrap p-value for a zero difference, ``d`` the
    paired Cohen's d (mean difference / sd of differences) at the level of the statistic
    (over per-cluster mean differences for ``'cluster'``, over pairs for ``'row'``) and
    ``mean_a`` / ``mean_b`` the condition means at that level (``mean_a - mean_b ==
    mean``).
    """
    if stat_level not in STAT_LEVELS:
        raise ValueError(f"stat_level must be one of {STAT_LEVELS}, got {stat_level!r}")
    pair_cols = _as_list(pair_cols)
    cluster_in_pairs = cluster_col in pair_cols

    def side(level, tag):
        sub = df[df[cond_col] == level]
        g = sub.groupby(pair_cols, sort=False)
        agg = g[value_col].mean().rename(f"value_{tag}").reset_index()
        if not cluster_in_pairs:
            agg[f"cluster_{tag}"] = g[cluster_col].first().to_numpy()
            n_mixed = int((g[cluster_col].nunique() > 1).sum())
            if n_mixed:
                warnings.warn(f"paired_contrast: {n_mixed} pair(s) of {cond_col} == {level!r} span more than one "
                              f"{cluster_col!r}; the first cluster of each pair is used", UserWarning, stacklevel=3)
        return agg

    da, db = side(a, "a"), side(b, "b")
    merged = da.merge(db, on=pair_cols, how="inner")
    n_dropped_a, n_dropped_b = int(len(da) - len(merged)), int(len(db) - len(merged))
    if not cluster_in_pairs and len(merged):
        ca, cb = merged["cluster_a"], merged["cluster_b"]
        n_diff = int(((ca != cb) & ~(ca.isna() & cb.isna())).sum())
        if n_diff:
            warnings.warn(f"paired_contrast: {n_diff} pair(s) have different {cluster_col!r} on the {a!r} and "
                          f"{b!r} side; the {a!r} cluster is used", UserWarning, stacklevel=2)
    cl = cluster_col if cluster_in_pairs else "cluster_a"
    merged["diff"] = merged["value_a"] - merged["value_b"]
    res = cluster_bootstrap(merged, "diff", cl, n_boot=n_boot, seed=seed, stat=np.mean, ci=ci, stat_level=stat_level)
    ok = merged["diff"].notna()
    if stat_level == "cluster":
        per = merged[ok].groupby(cl, sort=True)[["value_a", "value_b", "diff"]].mean()
    else:
        per = merged.loc[ok, ["value_a", "value_b", "diff"]]
    diffs = per["diff"].to_numpy(dtype=np.float64)
    sd = float(diffs.std(ddof=1)) if diffs.size > 1 else np.nan
    d = float(diffs.mean() / sd) if diffs.size > 1 and sd > 0 else np.nan
    res.update(n_pairs=int(ok.sum()), n_dropped_a=n_dropped_a, n_dropped_b=n_dropped_b,
               p_boot=bootstrap_p_two_sided(res["boot"], 0.0), d=d,
               mean_a=float(per["value_a"].mean()) if len(per) else np.nan,
               mean_b=float(per["value_b"].mean()) if len(per) else np.nan)
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
