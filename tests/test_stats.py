import numpy as np
import pandas as pd
import pytest

from lsne import stats as S


def _clustered(seed=0, n_clusters=20, per_cluster=10, mu=1.0):
    rng = np.random.default_rng(seed)
    effects = rng.normal(0, 0.5, n_clusters)
    rows = []
    for c in range(n_clusters):
        for _ in range(per_cluster):
            rows.append(dict(family=f"F{c:02d}", value=mu + effects[c] + rng.normal(0, 1)))
    return pd.DataFrame(rows)


def test_cluster_bootstrap_recovers_mean_and_covers_truth():
    df = _clustered(seed=1, mu=1.0)
    res = S.cluster_bootstrap(df, "value", "family", n_boot=2000, seed=0)
    assert res["mean"] == pytest.approx(df["value"].mean())
    assert res["n_clusters"] == 20 and res["n_rows"] == 200
    assert res["ci_low"] < res["mean"] < res["ci_high"]
    assert res["ci_low"] < 1.0 < res["ci_high"]            # truth inside the CI
    assert res["boot"].shape == (2000,) and np.isfinite(res["boot"]).all()
    assert 0 < res["se"] < 0.5
    # deterministic
    res2 = S.cluster_bootstrap(df, "value", "family", n_boot=2000, seed=0)
    assert np.array_equal(res["boot"], res2["boot"])
    # general statistic path (median) agrees with the fast mean path in spirit
    med = S.cluster_bootstrap(df, "value", "family", n_boot=300, seed=0, stat=np.median)
    assert med["mean"] == pytest.approx(df["value"].median())
    assert med["ci_low"] < med["mean"] < med["ci_high"]
    # fast path for np.mean equals the generic concatenation path
    gen = S.cluster_bootstrap(df, "value", "family", n_boot=200, seed=3, stat=lambda v: float(np.mean(v)))
    fast = S.cluster_bootstrap(df, "value", "family", n_boot=200, seed=3)
    assert np.allclose(gen["boot"], fast["boot"])
    # NaN rows are dropped; empty input is handled
    df2 = df.copy()
    df2.loc[0, "value"] = np.nan
    assert S.cluster_bootstrap(df2, "value", "family", n_boot=50)["n_rows"] == 199
    empty = S.cluster_bootstrap(df.iloc[:0], "value", "family", n_boot=50)
    assert np.isnan(empty["mean"]) and empty["n_clusters"] == 0


def test_cluster_bootstrap_wider_than_naive_with_cluster_effects():
    df = _clustered(seed=2, n_clusters=10, per_cluster=30, mu=0.0)
    df["family_big"] = df["family"]
    rows_as_clusters = df.assign(row=np.arange(len(df)))
    cl = S.cluster_bootstrap(df, "value", "family", n_boot=2000, seed=0)
    naive = S.cluster_bootstrap(rows_as_clusters, "value", "row", n_boot=2000, seed=0)
    assert (cl["ci_high"] - cl["ci_low"]) > (naive["ci_high"] - naive["ci_low"])


def test_paired_contrast():
    rng = np.random.default_rng(0)
    rows = []
    for fam in range(12):
        base = rng.normal(0, 1)
        for K in (2, 4, 8):
            rows.append(dict(family=fam, K=K, cond="mixed", value=base + 1.0 + rng.normal(0, 0.1)))
            rows.append(dict(family=fam, K=K, cond="mono", value=base + rng.normal(0, 0.1)))
    df = pd.DataFrame(rows)
    res = S.paired_contrast(df, "value", "cond", "mixed", "mono", ["family", "K"], "family",
                            n_boot=2000, seed=0)
    assert res["n_pairs"] == 36 and res["n_clusters"] == 12
    assert res["mean"] == pytest.approx(1.0, abs=0.1)
    assert res["ci_low"] > 0.8 and res["ci_high"] < 1.2
    assert res["p_boot"] < 0.01 and res["d"] > 3
    assert res["mean_a"] - res["mean_b"] == pytest.approx(res["mean"])
    # cluster column not part of the pair columns (taken from the a-side rows)
    df["item"] = df["family"].astype(str) + "-" + df["K"].astype(str)
    res2 = S.paired_contrast(df, "value", "cond", "mixed", "mono", "item", "family", n_boot=200, seed=0)
    assert res2["n_pairs"] == 36 and res2["n_clusters"] == 12
    assert res2["mean"] == pytest.approx(res["mean"])
    # duplicate rows per (pair, condition) are averaged, not multiplied
    dup = pd.concat([df, df], ignore_index=True)
    res3 = S.paired_contrast(dup, "value", "cond", "mixed", "mono", ["family", "K"], "family", n_boot=100)
    assert res3["n_pairs"] == 36 and res3["mean"] == pytest.approx(res["mean"])
    # null contrast: CI covers 0
    df0 = df.copy()
    df0.loc[df0["cond"] == "mixed", "value"] -= 1.0
    res0 = S.paired_contrast(df0, "value", "cond", "mixed", "mono", ["family", "K"], "family",
                             n_boot=1000, seed=0)
    assert res0["ci_low"] < 0 < res0["ci_high"] and res0["p_boot"] > 0.05


def test_bh_correct_monotone_and_known_values():
    p = np.array([0.01, 0.04, 0.03])
    adj = S.bh_correct(p)
    assert adj == pytest.approx([0.03, 0.04, 0.04])
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 50)
    adj = S.bh_correct(p)
    assert (adj >= p).all() and (adj <= 1).all()
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= 0)                   # monotone in the raw p-values
    # NaNs stay NaN and do not count as tests
    adj2 = S.bh_correct([0.01, np.nan, 0.04, 0.03])
    assert np.isnan(adj2[1]) and adj2[[0, 2, 3]] == pytest.approx([0.03, 0.04, 0.04])
    assert S.bh_correct([0.5]) == pytest.approx([0.5])
    assert S.bh_correct([]).shape == (0,)


def test_cohen_d_and_bootstrap_p():
    assert S.cohen_d([1, 2, 3, 4], [2, 3, 4, 5]) == pytest.approx(-1 / np.sqrt(5 / 3))
    assert S.cohen_d([0, 0, 0], [1, 1, 1]) != S.cohen_d([0, 0, 0], [1, 1, 1])   # nan (zero sd)
    assert np.isnan(S.cohen_d([1], [1, 2]))
    rng = np.random.default_rng(0)
    assert S.bootstrap_p_two_sided(rng.normal(0, 1, 1000)) > 0.5
    assert S.bootstrap_p_two_sided(rng.normal(5, 1, 1000)) < 0.01
    assert S.bootstrap_p_two_sided(rng.normal(5, 1, 1000), null=5.0) > 0.5
    assert S.bootstrap_p_two_sided([]) != S.bootstrap_p_two_sided([])                # nan
    assert S.bootstrap_p_two_sided([1.0, 1.0, 1.0]) == pytest.approx(min(1.0, 2 * (1 / 4)))
