"""Tests for lsne.sharing: synthetic TopK SAEs (TopKSAE.from_tensors) with random
activations for the sharing index and network activity; pythia-14m (CPU) for head
profiles and the variant-based report."""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite, layer_grid
from lsne.heads import head_contributions
from lsne.sae import TopKSAE
from lsne.sharing import (ACTIVITY_KEYS, SHARING_COLS, active_feature_sets, active_features, feature_frequency,
                          frequency_sharing, head_profile, head_profile_cosines, head_profile_similarity,
                          head_profiles, jaccard, network_activity, pairwise_sharing, sharing_index,
                          sharing_report, sharing_table)


def random_sae(d=32, F=96, k=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    W_enc = torch.randn(F, d, generator=g) / d ** 0.5
    b_enc = 0.1 * torch.randn(F, generator=g)
    W_dec = torch.randn(F, d, generator=g)
    W_dec = W_dec / W_dec.norm(dim=-1, keepdim=True)
    return TopKSAE.from_tensors(W_enc, b_enc, W_dec, 0.1 * torch.randn(d, generator=g), k)


def orthonormal_sae(d=32, k=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(d, d, generator=g))
    W = Q.T.contiguous()
    return TopKSAE.from_tensors(W, torch.zeros(d), W.clone(), torch.zeros(d), k), W


def variant(tok, text):
    return types.SimpleNamespace(ids=tok(text, add_special_tokens=False)["input_ids"], target_id=0)


# --------------------------------------------------------------------------------------
def test_active_features_topk_and_threshold():
    sae = random_sae(d=32, F=96, k=8)
    x = torch.randn(10, 32, generator=torch.Generator().manual_seed(1))
    sets = active_feature_sets(sae, x)
    assert len(sets) == 10 and all(0 < len(s) <= 8 for s in sets)
    top_acts, top_idx, pre = sae.encode(x)
    for i in range(10):
        assert sets[i] == {int(j) for j, a in zip(top_idx[i], top_acts[i]) if float(a) > 0}
    union = active_features(sae, x)
    assert union == set().union(*sets) and all(0 <= j < 96 for j in union)
    assert active_features(sae, x.numpy()) == union                              # numpy accepted
    assert active_features(sae, x[0]) == sets[0]                                 # 1-D input
    # relu(pre) > threshold (not restricted to the top-k)
    full = active_feature_sets(sae, x, topk_only=False)
    assert all(full[i] >= sets[i] for i in range(10))
    assert all(full[i] == {int(j) for j in (torch.relu(pre[i]) > 0).nonzero().reshape(-1).tolist()} for i in range(10))
    hi = active_feature_sets(sae, x, threshold=float(top_acts.max()) + 1.0)
    assert all(len(s) == 0 for s in hi)
    # frequency
    freq = feature_frequency(sets, 96)
    assert freq.shape == (96,) and np.isclose(freq.sum(), np.mean([len(s) for s in sets]))
    assert ((freq >= 0) & (freq <= 1)).all() and feature_frequency([], 5).tolist() == [0.0] * 5


def test_sharing_index_identical_disjoint_and_global():
    assert jaccard({1, 2}, {2, 3}) == 1 / 3 and jaccard(set(), set()) == 1.0
    sae, W = orthonormal_sae(d=32, k=4)
    g = torch.Generator().manual_seed(2)
    # exactly k = 4 planted components per state -> TopK recovers the supports exactly
    xa = torch.stack([(torch.rand(4, generator=g) + 0.5) @ W[[0, 1, 2, 3]] for _ in range(6)])
    xb = torch.stack([(torch.rand(4, generator=g) + 0.5) @ W[[10, 11, 12, 13]] for _ in range(6)])
    xc = torch.stack([(torch.rand(4, generator=g) + 0.5) @ W[[0, 1, 12, 13]] for _ in range(6)])
    sets = {"a": active_features(sae, xa), "b": active_features(sae, xb), "c": active_features(sae, xc)}
    assert sets["a"] == {0, 1, 2, 3} and sets["b"] == {10, 11, 12, 13} and sets["c"] == {0, 1, 12, 13}
    df = sharing_index(sets)
    assert isinstance(df, pd.DataFrame) and list(df.columns) == SHARING_COLS
    assert df.kind.tolist() == ["pair", "pair", "pair", "global"]
    assert df[["a", "b"]].values.tolist()[:3] == [["a", "b"], ["a", "c"], ["b", "c"]]
    r = df.set_index(["a", "b"])
    assert r.loc[("a", "b"), "jaccard"] == 0.0 and r.loc[("a", "b"), "n_shared"] == 0
    assert np.isclose(r.loc[("a", "c"), "jaccard"], 2 / 6) and r.loc[("a", "c"), "n_union"] == 6
    assert np.isclose(r.loc[("b", "c"), "jaccard"], 2 / 6) and r.loc[("b", "c"), "n_shared"] == 2
    glob = df[df.kind == "global"].iloc[0]
    assert glob.a == "global" and glob.n_shared == 0 and glob.n_union == 8 and glob.jaccard == 0.0
    # identical conditions -> jaccard 1 everywhere
    same = sharing_index({"x": sets["a"], "y": set(sets["a"])})
    assert (same.jaccard == 1.0).all() and (same.n_shared == 4).all()
    assert ((df.jaccard >= 0) & (df.jaccard <= 1)).all()
    # frequency-based variant (second signature) and pairwise table
    fa = feature_frequency(active_feature_sets(sae, xa), sae.num_latents)
    fb = feature_frequency(active_feature_sets(sae, xb), sae.num_latents)
    ident = sharing_index(fa, fa, top_m=10)
    assert ident == frequency_sharing(fa, fa, top_m=10)
    assert ident["jaccard_topm"] == 1.0 and ident["jaccard_active"] == 1.0 and ident["weighted_overlap"] == 1.0
    disj = frequency_sharing(fa, fb)
    assert disj["jaccard_topm"] == 0.0 and disj["jaccard_active"] == 0.0 and disj["weighted_overlap"] == 0.0
    pw = pairwise_sharing({"a": fa, "b": fb, "a2": fa})
    assert list(pw.columns) == ["a", "b", "jaccard_topm", "jaccard_active", "weighted_overlap"] and len(pw) == 3
    assert pw.set_index(["a", "b"]).loc[("a", "a2"), "weighted_overlap"] == 1.0
    with pytest.raises(ValueError):
        sharing_index(fa)
    with pytest.raises(ValueError):
        frequency_sharing(fa, fa[:-1])


def test_network_activity_fields_and_ranges():
    sae = random_sae(d=32, F=96, k=8)
    x = torch.randn(40, 32, generator=torch.Generator().manual_seed(3))
    res = network_activity(sae, x)
    assert list(res) == ACTIVITY_KEYS
    assert all(np.isfinite(v) for k, v in res.items() if isinstance(v, (int, float)))
    assert 0 <= res["frac_active"] <= 1 and res["n_rows"] == 40 and res["num_latents"] == 96 and res["k"] == 8
    assert np.isclose(res["frac_active"], res["n_active_mean"] / 96)
    assert 1 <= res["participation_ratio"] <= 96 and np.isclose(res["participation_ratio_frac"], res["participation_ratio"] / 96)
    assert 0 <= res["frac_features_active"] <= 1 and res["mean_mass"] > 0 and res["topk_only"] is False
    # cross-check against the dense relu code
    _, _, pre = sae.encode(x)
    A = torch.relu(pre)
    assert np.isclose(res["n_active_mean"], float((A > 0).float().sum(-1).mean()))
    assert np.isclose(res["mean_mass"], float(A.sum(-1).mean()), rtol=1e-5)
    a = A.mean(0)
    assert np.isclose(res["participation_ratio"], float(a.sum() ** 2 / (a * a).sum()), rtol=1e-5)
    # top-k code: at most k active per state and the same mass as the sparse code
    rk = network_activity(sae, x, topk_only=True)
    assert rk["n_active_mean"] <= 8 and rk["topk_only"] is True
    assert np.isclose(rk["mean_mass"], float(sae.encode(x)[0].sum(-1).mean()), rtol=1e-5)
    # chunking is exact; a high threshold switches everything off
    assert network_activity(sae, x, chunk=7)["participation_ratio"] == res["participation_ratio"]
    off = network_activity(sae, x, threshold=1e6)
    assert off["frac_active"] == 0.0 and off["n_active_mean"] == 0.0 and off["frac_features_active"] == 0.0
    with pytest.raises(ValueError):
        network_activity(sae, torch.zeros(0, 32))


# --------------------------------------------------------------------------------------
def test_head_profile_length_H(mb, sample_ids):
    H = mb.n_heads
    layer = mb.n_layers // 2
    p = head_profile(mb, sample_ids, layer)
    assert p.shape == (H,) and np.isfinite(p).all() and (p >= 0).all() and p.max() > 0
    c = head_contributions(mb, sample_ids, layer)
    assert np.allclose(p, c[:, -1].norm(dim=-1).numpy(), atol=1e-6)
    assert np.allclose(head_profile(mb, sample_ids, layer, position=2), c[:, 2].norm(dim=-1).numpy(), atol=1e-6)
    assert np.allclose(head_profile(mb, sample_ids[0].tolist(), layer), p)            # list of ids accepted
    sim = head_profile_similarity({"a": p, "b": p, "c": np.ones(H)})
    assert list(sim.columns) == ["a", "b", "cosine"] and len(sim) == 3
    assert np.isclose(sim.cosine.iloc[0], 1.0) and ((sim.cosine >= -1) & (sim.cosine <= 1)).all()


def test_head_profile_cosines_over_layer_grid(mb, tok):
    cond = {"law": [variant(tok, "The court ruled that the plaintiff"), variant(tok, "The judge said the defendant")],
            "mis": [variant(tok, "The prion protein misfolded into amyloid"),
                    dict(ids=tok("Amyloid fibrils aggregate in the brain", add_special_tokens=False)["input_ids"])]}
    grid = [0, mb.n_layers // 2]
    profs = head_profiles(mb, cond, grid)
    assert set(profs) == set(grid) and all(set(p) == {"law", "mis"} for p in profs.values())
    assert all(v.shape == (mb.n_heads,) for p in profs.values() for v in p.values())
    exp = np.mean([head_profile(mb, v, 0) for v in cond["law"]], axis=0)
    assert np.allclose(profs[0]["law"], exp)
    df = head_profile_cosines(mb, cond, grid)
    assert list(df.columns) == ["a", "b", "layer", "cosine"] and len(df) == len(grid)
    assert df.layer.tolist() == grid and ((df.cosine >= 0) & (df.cosine <= 1)).all()    # norms are non-negative
    same = head_profile_cosines(mb, {"x": cond["law"], "y": list(cond["law"])}, grid)
    assert np.allclose(same.cosine, 1.0)
    default = head_profile_cosines(mb, cond)
    assert default.layer.tolist() == list(layer_grid(mb.n_layers))


def test_sharing_report_and_table(mb, tok):
    d = mb.d_model
    sae = random_sae(d=d, F=64, k=6, seed=4)
    site = HookSite(mb.n_layers // 2, "block_output")
    groups = {"law": [variant(tok, "The court ruled that the plaintiff"), variant(tok, "The judge said the defendant"),
                      variant(tok, "The statute was appealed by the attorney")],
              "mis": [variant(tok, "The prion protein misfolded into amyloid"),
                      variant(tok, "Amyloid fibrils aggregate in the brain")]}
    rep = sharing_report(mb, sae, groups, site=site, layer_grid=[0, 1])
    assert set(rep) >= {"sharing", "activity", "heads", "sets", "site", "n"}
    assert rep["site"] == site.key and rep["n"] == {"law": 3, "mis": 2}
    sh = rep["sharing"]
    assert list(sh.columns) == SHARING_COLS and sh.kind.tolist() == ["pair", "global"]
    assert ((sh.jaccard >= 0) & (sh.jaccard <= 1)).all()
    assert sh.iloc[0].n_a == len(rep["sets"]["law"]) and 0 < sh.iloc[0].n_a <= 3 * sae.k
    act = rep["activity"]
    assert list(act.columns) == ["condition"] + ACTIVITY_KEYS and act.condition.tolist() == ["law", "mis"]
    assert np.isfinite(act[["frac_active", "mean_mass", "participation_ratio", "n_active_mean"]].values).all()
    assert (act.n_active_mean <= sae.k).all() and act.n_rows.tolist() == [3, 2]
    heads = rep["heads"]
    assert list(heads.columns) == ["a", "b", "layer", "cosine"] and heads.layer.tolist() == [0, 1]
    # the same from pre-pooled states (no model, no head profiles)
    from lsne.rdx import variant_states
    states = {c: variant_states(mb, vs, site) for c, vs in groups.items()}
    rep2 = sharing_report(None, sae, states)
    assert rep2["heads"] is None and rep2["site"] is None
    pd.testing.assert_frame_equal(rep2["sharing"], sh)
    pd.testing.assert_frame_equal(rep2["activity"], act)
    # flat table
    tab = sharing_table(mb, sae, groups, site=site, layer_grid=[0])
    assert tab.kind.tolist() == ["sharing", "sharing", "activity", "activity", "heads"]
    assert (tab.site == site.key).all()
    assert tab[tab.kind == "sharing"].jaccard.tolist() == sh.jaccard.tolist()
    assert tab[tab.kind == "activity"].a.tolist() == ["law", "mis"]
    assert np.isclose(tab[tab.kind == "heads"].cosine.iloc[0], heads.cosine.iloc[0])
    assert tab[tab.kind == "activity"].frac_active.notna().all() and tab[tab.kind == "sharing"].frac_active.isna().all()
    with pytest.raises(TypeError):
        sharing_report(mb, sae, groups, site=None)
