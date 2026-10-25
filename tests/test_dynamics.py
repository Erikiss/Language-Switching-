"""Tests for lsne.dynamics (non-equilibrium diagnostics) on pythia-14m."""
import numpy as np
import pytest
import torch

from lsne.config import HookSite
from lsne.model import run_with_cache
from lsne.dynamics import (DamageMap, ca_complexity, ca_symbolize, damage_spreading,
                           growth_exponent, lexicon_from_axis, logit_lens_trajectory, lz76,
                           nonmonotonicity_index, spreading_front, update_norm_profile)


@pytest.fixture(scope="module")
def cache(mb, sample_ids):
    return run_with_cache(mb, sample_ids, "all_block_inputs")


@pytest.fixture(scope="module")
def direction(mb):
    return torch.randn(mb.d_model, generator=torch.Generator().manual_seed(7))


def test_damage_spreading_causal_structure(mb, sample_ids, cache, direction):
    L, T = mb.n_layers, sample_ids.shape[1]
    layer, pos, eps = 1, T // 2, 0.05
    site = HookSite(layer, "block_input")
    dm = damage_spreading(mb, sample_ids, site, pos, direction, eps, ref_cache=cache, target_id=5)
    assert isinstance(dm, DamageMap)
    assert dm.D.shape == (L + 1, T) and dm.cos.shape == (L + 1, T)
    assert dm.layers == [s.key for s in cache.sites] and len(dm.layers) == L + 1
    assert dm.inject_layer == layer and dm.inject_pos == pos and dm.eps == eps
    # snapshots before the injection layer are untouched
    assert np.all(dm.D[:layer] == 0.0) and np.all(dm.cos[:layer] == 1.0)
    # causality: earlier positions are never perturbed
    assert np.all(dm.D[:, :pos] < 1e-6)
    # at the injection snapshot only the injected position moves, by the relative dose eps
    assert dm.D[layer, pos] == pytest.approx(eps, rel=1e-3)
    assert np.all(dm.D[layer, pos + 1:] == 0.0)
    # downstream the perturbation spreads to later positions
    assert dm.D[L, pos:].max() > 0 and (dm.D[L, pos + 1:] > 0).any()
    assert np.isfinite(dm.logp_target_ref) and np.isfinite(dm.logp_target_pert)
    assert dm.audit["rel_err"] < 1e-4
    # a cache computed inside (ref_cache=None) gives the same map
    dm2 = damage_spreading(mb, sample_ids, site, pos, direction, eps)
    assert np.allclose(dm.D, dm2.D)
    assert np.isnan(dm2.logp_target_ref)


def test_damage_spreading_zero_dose_and_block_output(mb, sample_ids, cache, direction):
    L, T = mb.n_layers, sample_ids.shape[1]
    site = HookSite(2, "block_input")
    dm0 = damage_spreading(mb, sample_ids, site, 3, direction, 0.0, ref_cache=cache)
    assert np.all(dm0.D == 0.0) and np.all(dm0.cos == 1.0)
    # block output of layer l enters at snapshot l+1
    site_out = HookSite(2, "block_output")
    dm = damage_spreading(mb, sample_ids, site_out, 3, direction, 0.02, ref_cache=cache)
    assert dm.inject_layer == 3
    assert np.all(dm.D[:3] == 0.0) and dm.D[3, 3] == pytest.approx(0.02, rel=1e-3)
    # last block output: injection shows up in the final snapshot only
    dm_last = damage_spreading(mb, sample_ids, HookSite(L - 1, "block_output"), T - 1, direction, 0.02,
                               ref_cache=cache)
    assert dm_last.inject_layer == L and np.all(dm_last.D[:L] == 0.0) and dm_last.D[L, T - 1] > 0
    # a Direction-like object (with .vec) is accepted
    class Dir:  # noqa: D401 - duck type
        vec = direction
    dm3 = damage_spreading(mb, sample_ids, site, 3, Dir(), 0.02, ref_cache=cache)
    assert np.allclose(dm3.D, damage_spreading(mb, sample_ids, site, 3, direction, 0.02, ref_cache=cache).D)


def test_growth_exponent_and_front(mb, sample_ids, cache, direction):
    L, T = mb.n_layers, sample_ids.shape[1]
    pos = T // 2
    dm = damage_spreading(mb, sample_ids, HookSite(1, "block_input"), pos, direction, 0.01, ref_cache=cache)
    positions = list(range(pos, T))
    g = growth_exponent(dm, positions)
    for k in ("lam", "intercept", "r2"):
        assert np.isfinite(g[k]), k
    assert g["per_position"].shape == (len(positions),) and np.isfinite(g["per_position"]).all()
    assert list(g["layers"]) == list(range(dm.inject_layer + 1, L + 1))
    g2 = growth_exponent(dm, positions, layer_from=dm.inject_layer)
    assert list(g2["layers"]) == list(range(dm.inject_layer, L + 1)) and np.isfinite(g2["lam"])
    # positions before the injection carry only the clipped floor -> zero slope
    g0 = growth_exponent(dm, [0])
    assert g0["lam"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        growth_exponent(dm, [])
    # spreading front
    front = spreading_front(dm, 1e-9)
    assert front.shape == (L + 1,) and front.dtype.kind == "i"
    assert np.all(front[:dm.inject_layer] == 0) and front[dm.inject_layer] == 0
    assert front[L] == (T - 1 - pos) and np.all(front >= 0)
    assert np.all(spreading_front(dm, 1e9) == 0)


def test_logit_lens_trajectory(mb, sample_ids, cache):
    L, T = mb.n_layers, sample_ids.shape[1]
    tgt = int(sample_ids[0, -1])
    tr = logit_lens_trajectory(mb, cache, target_id=tgt)
    for k in ("kl", "target_rank", "target_logp"):
        assert tr[k].shape == (L + 1, T), k
    assert np.all(tr["kl"] >= 0) and np.all(tr["kl"][-1] < 1e-5)
    assert tr["kl"][0].mean() > tr["kl"][-1].mean()
    logp_model = torch.log_softmax(cache.logits[0, -1], -1)
    assert tr["target_logp"][-1, -1] == pytest.approx(float(logp_model[tgt]), abs=1e-4)
    top1 = int(logp_model.argmax())
    tr_top = logit_lens_trajectory(mb, cache, target_id=top1, positions=[T - 1])
    assert tr_top["target_rank"].shape == (L + 1, 1) and tr_top["target_rank"][-1, 0] == 0
    # per-position targets: the actual next tokens
    nxt = sample_ids[0, 1:].tolist()
    tr_seq = logit_lens_trajectory(mb, cache, target_id=nxt, positions=list(range(T - 1)))
    assert tr_seq["target_logp"].shape == (L + 1, T - 1) and np.isfinite(tr_seq["target_logp"]).all()
    # no target -> NaN target arrays, KL unchanged
    tr_none = logit_lens_trajectory(mb, cache)
    assert np.isnan(tr_none["target_rank"]).all() and np.allclose(tr_none["kl"], tr["kl"])
    with pytest.raises(ValueError):
        logit_lens_trajectory(mb, cache, target_id=[1, 2], positions=[0])


def test_nonmonotonicity_index():
    assert nonmonotonicity_index([5, 4, 3, 2, 1]) == 0.0
    assert nonmonotonicity_index([1, 0, 1]) == pytest.approx(0.5)
    assert nonmonotonicity_index([1, 2, 3]) == 1.0
    assert nonmonotonicity_index([2, 2, 2]) == 0.0
    assert nonmonotonicity_index([1.0]) == 0.0
    assert nonmonotonicity_index(np.array([3, 1, np.nan, 2])) == 0.0   # NaN increments ignored


def test_update_norm_profile(mb, cache):
    L = mb.n_layers
    T = cache.resid[cache.sites[0].key].shape[1]
    u = update_norm_profile(cache)
    assert u.shape == (L, T) and np.all(u > 0) and np.isfinite(u).all()
    H0 = cache.resid[cache.sites[0].key][0]
    H1 = cache.resid[cache.sites[1].key][0]
    expect = ((H1 - H0).norm(dim=-1) / (H0.norm(dim=-1) + 1e-12)).numpy()
    assert np.allclose(u[0], expect, rtol=1e-5)


def _lexicon(tok):
    def ids(words):
        return [tok(w, add_special_tokens=False)["input_ids"][0] for w in words]
    return dict(E=ids([" the", " and", " is", " of", " to"]), D=ids([" der", " die", " und", " ist", " nicht"]))


def test_ca_symbolize(mb, tok, cache, direction):
    L = mb.n_layers
    T = cache.resid[cache.sites[0].key].shape[1]
    S = ca_symbolize(mb, cache, _lexicon(tok))
    assert S.shape == (L + 1, T) and S.dtype == np.int8
    assert set(np.unique(S).tolist()) <= {0, 1, 2}
    # English text at the final snapshot: at least one site reads as E, none dominated by D
    assert (S[-1] == 1).any()
    # several thresholds -> stacked; a stricter threshold never adds E/D symbols
    S2 = ca_symbolize(mb, cache, _lexicon(tok), thresholds=(0.5, 0.9))
    assert S2.shape == (2, L + 1, T) and np.array_equal(S2[0], S)
    assert ((S2[1] != 0) <= (S2[0] != 0)).all()
    # SPEC argument order (lang_axis before lexicon) is tolerated
    assert np.array_equal(ca_symbolize(mb, cache, None, _lexicon(tok)), S)
    # lang_axis fallback derives the markers through the unembedding
    lex = lexicon_from_axis(mb, direction, top_k=10)
    assert len(lex["E"]) == 10 and len(lex["D"]) == 10 and not set(lex["E"]) & set(lex["D"])
    S3 = ca_symbolize(mb, cache, {}, lang_axis=direction)
    assert S3.shape == (L + 1, T)
    with pytest.raises(ValueError):
        ca_symbolize(mb, cache, dict(E=[], D=[1]))


def test_lz76_and_ca_complexity():
    assert lz76([]) == 0 and lz76([1]) == 1
    assert lz76([0] * 50) == 2
    assert lz76([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0]) == 6   # classic textbook example
    rng = np.random.default_rng(0)
    rand = rng.integers(0, 3, size=400)
    assert lz76(rand) > 10 * lz76([0] * 400)
    const = np.zeros((7, 30), dtype=np.int8)
    c = ca_complexity(const)
    assert set(c) == {"spatial_entropy", "temporal_entropy", "activity", "lz76", "block_entropy_2x2"}
    assert c["spatial_entropy"] == 0 and c["temporal_entropy"] == 0 and c["activity"] == 0
    assert c["block_entropy_2x2"] == 0 and 0 < c["lz76"] < 0.1
    field = rng.integers(0, 3, size=(7, 30)).astype(np.int8)
    r = ca_complexity(field)
    assert r["spatial_entropy"] > 1.0 and r["temporal_entropy"] > 1.0
    assert 0.5 < r["activity"] < 0.8 and r["lz76"] > c["lz76"] and r["block_entropy_2x2"] > 3
    # checkerboard: maximal activity, two 2x2 patterns
    cb = np.indices((6, 11)).sum(0) % 2          # 5 x 10 blocks -> both patterns 25 times
    r2 = ca_complexity(cb)
    assert r2["activity"] == 1.0 and r2["block_entropy_2x2"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ca_complexity(np.zeros(5))
