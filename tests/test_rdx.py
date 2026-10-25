"""Tests for lsne.rdx (Representational Difference Explanation): synthetic bases where
possible, pythia-14m (CPU) for the model-based table and the candidate bases."""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite
from lsne.directions import Direction
from lsne.rdx import (RDX_COLS, bases_from_directions, bases_from_heads, bases_from_pca, bases_from_sae,
                      basis_attribution, difference_vector, explained_fraction, family_stability, random_bases,
                      rdx_score, rdx_table, stability, variant_states, variant_states_by_site)
from lsne.sae import TopKSAE


def orthonormal(d=32, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    return Q.T                                           # rows orthonormal


def random_sae(d, F=48, k=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(F, d, generator=g)
    W = W / W.norm(dim=-1, keepdim=True)
    return TopKSAE.from_tensors(W, torch.zeros(F), W.clone(), torch.zeros(d), k)


def variant(tok, text):
    return types.SimpleNamespace(ids=tok(text, add_special_tokens=False)["input_ids"], target_id=0)


# --------------------------------------------------------------------------------------
def test_difference_vector_is_mean_difference():
    rng = np.random.default_rng(0)
    A, B = rng.standard_normal((5, 7)), rng.standard_normal((3, 7))
    delta = difference_vector(A, B)
    assert delta.shape == (7,) and np.allclose(delta, A.mean(0) - B.mean(0))
    assert np.allclose(difference_vector(torch.tensor(A), torch.tensor(B)), delta)    # torch accepted
    with pytest.raises(ValueError):
        difference_vector(A, rng.standard_normal((3, 6)))


def test_explained_fraction_span_orthogonal_and_range():
    Q = orthonormal(32)
    basis = Q[:4]                                        # 4 orthonormal rows
    delta_in = np.array([1.0, -2.0, 0.5, 3.0]) @ basis
    assert np.isclose(explained_fraction(delta_in, basis), 1.0)
    assert explained_fraction(Q[10], basis) < 1e-12      # orthogonal
    rng = np.random.default_rng(1)
    for _ in range(5):
        f = explained_fraction(rng.standard_normal(32), basis)
        assert 0.0 <= f <= 1.0
    # a mixed vector: exact fraction of the squared norm in the span
    v = 3.0 * Q[0] + 4.0 * Q[20]
    assert np.isclose(explained_fraction(v, basis), 9.0 / 25.0)
    # non-orthonormal / linearly dependent rows span the same space -> same fraction
    M = np.stack([2.0 * basis[0] + basis[1], basis[1] - 0.5 * basis[2], 5.0 * basis[2], basis[3], basis[0] + basis[3]])
    assert np.isclose(explained_fraction(v, M), 9.0 / 25.0)
    # k >= d -> everything explained; empty basis -> 0; zero delta -> nan
    assert np.isclose(explained_fraction(v, Q), 1.0)
    assert explained_fraction(v, np.zeros((0, 32))) == 0.0
    assert np.isnan(explained_fraction(np.zeros(32), basis))
    with pytest.raises(ValueError):
        explained_fraction(v, np.ones((2, 31)))


def test_random_bases_shape_norms_and_seed():
    R = random_bases(16, 3, 5, seed=0)
    assert R.shape == (5, 3, 16) and np.allclose(np.linalg.norm(R, axis=-1), 1.0)
    assert np.array_equal(R, random_bases(16, 3, 5, seed=0))
    assert not np.allclose(R, random_bases(16, 3, 5, seed=1))


def test_rdx_score_large_z_for_in_span_delta_and_random_mean_near_k_over_d():
    d, k = 64, 4
    Q = orthonormal(d, seed=2)
    basis = Q[:k]
    delta = np.array([1.0, 2.0, -1.0, 0.5]) @ basis
    res = rdx_score(delta, basis, n_random=32, seed=0)
    assert set(res) >= {"explained", "random_mean", "random_sd", "z", "k", "n_random", "p_random", "random"}
    assert res["k"] == k and res["n_random"] == 32 and res["random"].shape == (32,)
    assert np.isclose(res["explained"], 1.0) and res["z"] > 5.0 and res["p_random"] == 0.0
    assert abs(res["random_mean"] - k / d) < 0.05                       # expectation of a random k-basis
    # a random basis is not better than random: |z| small, p_random not extreme
    res_r = rdx_score(delta, random_bases(d, k, 1, seed=7)[0], n_random=32, seed=0)
    assert abs(res_r["z"]) < 3.0
    # deterministic in the seed
    res2 = rdx_score(delta, basis, n_random=32, seed=0)
    assert res2["random_mean"] == res["random_mean"] and res2["z"] == res["z"]
    # degenerate: k = d -> every basis explains everything, z = 0
    res_full = rdx_score(delta, Q, n_random=4, seed=0)
    assert np.isclose(res_full["explained"], 1.0) and res_full["z"] == 0.0


def test_basis_attribution_recovers_planted_coefficients():
    rng = np.random.default_rng(3)
    basis = rng.standard_normal((3, 20))
    planted = np.array([2.0, -1.0, 0.5])
    delta = planted @ basis
    att = basis_attribution(delta, basis, names=["x", "y", "z"])
    assert list(att.columns) == ["name", "coef", "norm", "abs_share", "cos"]
    coef = att.set_index("name").loc[["x", "y", "z"], "coef"].values
    assert np.allclose(coef, planted, atol=1e-8)
    assert np.isclose(att.abs_share.sum(), 1.0) and att.abs_share.is_monotonic_decreasing
    w = np.abs(planted) * np.linalg.norm(basis, axis=1)
    assert np.isclose(att.set_index("name").loc["x", "abs_share"], w[0] / w.sum())
    assert att.iloc[0]["name"] == ["x", "y", "z"][int(np.argmax(w))]
    assert ((att.cos >= -1) & (att.cos <= 1)).all()
    assert list(basis_attribution(delta, basis).name) and basis_attribution(delta, basis).name.str.startswith("b").all()
    with pytest.raises(ValueError):
        basis_attribution(delta, basis, names=["only", "two"])


# --------------------------------------------------------------------------------------
def test_variant_states_and_rdx_table_on_model(mb, tok):
    d = mb.d_model
    A = [variant(tok, "The court ruled that the plaintiff"), variant(tok, "The judge said the defendant"),
         dict(ids=tok("The statute was appealed by the attorney", add_special_tokens=False)["input_ids"])]
    B = [variant(tok, "The prion protein misfolded into amyloid"), variant(tok, "Amyloid fibrils aggregate in the brain")]
    sites = [HookSite(0, "block_input"), HookSite(mb.n_layers // 2, "block_output")]
    S = variant_states(mb, A, sites[1], readout="last")
    assert S.shape == (3, d) and np.isfinite(S).all()
    assert variant_states(mb, A, sites[1], readout="mean").shape == (3, d)
    assert not np.allclose(S, variant_states(mb, A, sites[1], readout="mean"))
    multi = variant_states_by_site(mb, A, sites)                          # one forward per variant, all sites
    assert set(multi) == {s.key for s in sites} and np.allclose(multi[sites[1].key], S, atol=1e-5)
    assert np.allclose(multi[sites[0].key], variant_states(mb, A, sites[0]), atol=1e-5)
    sae = random_sae(d)
    dirs = [Direction("dA", torch.randn(d, generator=torch.Generator().manual_seed(1)), sites[0]),
            Direction("dB", torch.randn(d, generator=torch.Generator().manual_seed(2)), sites[0])]
    bases = {"rand": random_bases(d, 4, 1, seed=0)[0],                  # plain matrix, shared
             "sae": bases_from_sae(sae, [0, 1, 2]),                       # (matrix, names)
             "dirs": dirs,                                                # list of Directions
             "heads": {s.key: bases_from_heads(mb, A[0], s.layer) for s in sites},   # per site
             "bad": np.ones((2, d + 1))}                                  # wrong dimension -> NaN row
    df = rdx_table(mb, {"law_vs_mis": (A, B)}, bases, sites=sites, n_random=8, seed=0)
    assert list(df.columns) == RDX_COLS
    assert len(df) == len(bases) * len(sites)
    assert set(df.site) == {s.key for s in sites} and set(df.basis) == set(bases)
    good = df[df.basis != "bad"]
    assert np.isfinite(good[["explained", "random_mean", "random_sd", "z", "delta_norm", "top_share"]].values).all()
    assert ((good.explained >= 0) & (good.explained <= 1)).all()
    assert (good.n_a == 3).all() and (good.n_b == 2).all() and (good.delta_norm > 0).all()
    assert good[good.basis == "heads"].k.unique().tolist() == [mb.n_heads]
    assert good[good.basis == "dirs"].top_component.isin(["dA", "dB"]).all()
    assert good[good.basis == "sae"].top_component.str.startswith("sae").all()
    assert df[df.basis == "bad"].explained.isna().all()
    # the same table from pre-collected activations keyed by site (mb not needed)
    acts = {"law_vs_mis": ({s.key: variant_states(mb, A, s) for s in sites}, {s.key: variant_states(mb, B, s) for s in sites})}
    df2 = rdx_table(None, acts, {"rand": bases["rand"]}, sites=[s.key for s in sites], n_random=8, seed=0)
    m1 = df[df.basis == "rand"].set_index("site")
    assert np.allclose(df2.set_index("site").loc[m1.index, "explained"], m1.explained)
    # plain [n, d] matrices are labelled "all"
    df3 = rdx_table(None, {"c": (acts["law_vs_mis"][0][sites[0].key], acts["law_vs_mis"][1][sites[0].key])},
                    {"rand": bases["rand"]}, n_random=4)
    assert df3.site.tolist() == ["all"] and np.isfinite(df3.z).all()
    with pytest.raises(ValueError):
        rdx_table(mb, {"c": (A, B)}, bases, sites=None)                  # variants need HookSites


def test_stability_aggregates_across_families():
    rows = lambda e, z: pd.DataFrame([dict(contrast="c", basis="b", site="s", k=2, explained=e, random_mean=0.1,
                                           random_sd=0.05, z=z, p_random=0.0, delta_norm=1.0, n_a=2, n_b=2,
                                           top_component="x", top_share=0.5)])
    st = stability({"F1": rows(0.4, 3.0), "F2": rows(0.6, 1.0), "F3": rows(0.5, 2.0)})
    assert len(st) == 1
    r = st.iloc[0]
    assert r.n_families == 3 and np.isclose(r.explained_mean, 0.5) and np.isclose(r.explained_min, 0.4)
    assert np.isclose(r.z_mean, 2.0) and np.isclose(r.z_min, 1.0) and np.isclose(r.frac_z_ge_2, 2 / 3)
    assert np.isclose(r.explained_sd, np.std([0.4, 0.6, 0.5], ddof=1))
    assert stability({}).empty and stability({"F1": rows(0.1, 0.0).iloc[0:0]}).empty


def test_family_stability_ci_contains_mean():
    rng = np.random.default_rng(5)
    d = 24
    basis = orthonormal(d, seed=1)[:3]
    fams = {}
    for i in range(6):
        shift = (0.5 + 0.2 * i) * basis[0]                                # planted in-span difference
        fams[f"F{i}"] = (rng.standard_normal((8, d)) + shift, rng.standard_normal((8, d)))
    res = family_stability(None, fams, basis, n_boot=100, seed=0)
    assert set(res) >= {"explained_mean", "ci_low", "ci_high", "n_families", "per_family"}
    assert res["n_families"] == 6 and len(res["per_family"]) == 6
    assert list(res["per_family"].columns) == ["family", "explained", "z", "delta_norm", "n_a", "n_b"]
    assert res["ci_low"] <= res["explained_mean"] <= res["ci_high"]
    assert np.isclose(res["explained_mean"], res["per_family"].explained.mean())
    assert ((res["per_family"].explained >= 0) & (res["per_family"].explained <= 1)).all()
    # site-keyed activations are accepted too
    fams_site = {f: ({"L0": a}, {"L0": b}) for f, (a, b) in fams.items()}
    res2 = family_stability(None, fams_site, basis, site="L0", n_boot=0)
    assert np.isclose(res2["explained_mean"], res["explained_mean"]) and res2["ci_low"] == res2["ci_high"]


def test_candidate_bases(mb, tok):
    d, H = mb.d_model, mb.n_heads
    sae = random_sae(d)
    M, names = bases_from_sae(sae, [3, 1])
    assert M.shape == (2, d) and names == ["sae3", "sae1"] and np.allclose(np.linalg.norm(M, axis=1), 1.0)
    assert np.allclose(M[0], sae.W_dec[3].numpy() / sae.W_dec[3].norm().item())
    v = variant(tok, "The report is about the soldier")
    Mh, nh = bases_from_heads(mb, v, layer=1)
    assert Mh.shape == (H, d) and nh == [f"L1H{h}" for h in range(H)]
    assert np.allclose(np.linalg.norm(Mh, axis=1), 1.0)
    Mh0, _ = bases_from_heads(mb, v, layer=1, position=0)
    assert not np.allclose(Mh, Mh0)
    dirs = [Direction("a", torch.randn(d, generator=torch.Generator().manual_seed(0)), HookSite(0)),
            Direction("b", torch.randn(d, generator=torch.Generator().manual_seed(1)), HookSite(0))]
    Md, nd = bases_from_directions(dirs)
    assert Md.shape == (2, d) and nd == ["a", "b"] and np.allclose(Md[0], dirs[0].vec.numpy())
    rng = np.random.default_rng(0)
    scales = np.ones(d)
    scales[0] = 30.0                                                        # one dominant axis
    X = rng.standard_normal((40, d)) * scales[None]
    Mp, npc = bases_from_pca(X, 3)
    assert Mp.shape == (3, d) and npc == ["pc0", "pc1", "pc2"]
    assert np.allclose(Mp @ Mp.T, np.eye(3), atol=1e-8)
    assert np.argmax(np.abs(Mp[0])) == 0                                   # top PC along the widest axis
    assert bases_from_pca(X[:2], 5)[0].shape[0] == 1                       # clipped to n - 1
    with pytest.raises(ValueError):
        bases_from_sae(sae, [])
