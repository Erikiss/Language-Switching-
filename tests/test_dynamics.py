"""Tests for lsne.dynamics (non-equilibrium diagnostics) on pythia-14m."""
import numpy as np
import pytest
import torch

from lsne.config import HookSite
from lsne.model import run_with_cache
from lsne.dynamics import (CA_MASS_FLOOR_FACTOR, DamageMap, ca_complexity, ca_marker_mass, ca_mass_floor,
                           ca_symbolize, damage_spreading, damage_spreading_batch, growth_exponent, lexicon_from_axis,
                           logit_lens_trajectory, lz76, nonmonotonicity_index, spreading_front,
                           symbolize_masses, update_norm_profile)


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


def test_damage_spreading_batch_matches_sequential(mb, sample_ids, cache, direction):
    """N (direction, dose) pairs in chunked forwards with the reference as row 0 of every
    chunk = N calls of damage_spreading (D, cos, log p(target), delta norm, Soll/Ist audit),
    with exact zeros before the injection layer and before the injection position."""
    L, T = mb.n_layers, sample_ids.shape[1]
    g = torch.Generator().manual_seed(11)
    dirs = [direction] + [torch.randn(mb.d_model, generator=g) for _ in range(2)]
    doses = [0.0, 0.01, 0.1, 1.0]
    pairs = [(d, e) for d in dirs for e in doses]
    # 12 pairs: chunks of 5+5+2 perturbed rows / one chunk / one row per forward
    cases = [(HookSite(1, "block_input"), T // 2, 6), (HookSite(2, "block_output"), 3, 64),
             (HookSite(1, "mlp_output"), T - 1, 2), (HookSite(L - 1, "block_input"), T - 1, 13)]
    for site, pos, bs in cases:
        maps = damage_spreading_batch(mb, sample_ids, site, pos, [d for d, _ in pairs], [e for _, e in pairs],
                                      target_id=5, batch_size=bs)
        assert len(maps) == len(pairs)
        for (d, e), b in zip(pairs, maps):
            s = damage_spreading(mb, sample_ids, site, pos, d, e, ref_cache=cache, target_id=5)
            assert isinstance(b, DamageMap) and b.D.shape == s.D.shape == (L + 1, T) and b.cos.shape == (L + 1, T)
            assert b.layers == s.layers and b.inject_layer == s.inject_layer and b.inject_pos == pos
            assert b.eps == e and b.site == site.key
            # nothing upstream of the injection moves: exact zeros, exact unit overlap
            assert np.all(b.D[:b.inject_layer] == 0.0) and np.all(b.cos[:b.inject_layer] == 1.0)
            assert np.all(b.D[:, :pos] == 0.0) and np.all(b.cos[:, :pos] == 1.0)
            big = s.D > 1e-6
            assert np.allclose(b.D[big], s.D[big], rtol=1e-4, atol=0.0), (site.key, e)
            assert np.all(np.abs(b.D[~big] - s.D[~big]) <= 1e-6)
            assert np.allclose(b.cos, s.cos, rtol=0.0, atol=1e-6)
            assert b.logp_target_ref == pytest.approx(s.logp_target_ref, abs=1e-4)
            assert b.logp_target_pert == pytest.approx(s.logp_target_pert, abs=1e-4)
            assert b.delta_norm == pytest.approx(s.delta_norm, rel=1e-5, abs=1e-12)
            assert {"intended_norm", "written_norm", "rel_err", "dtype"} <= set(b.audit)
            assert b.audit["intended_norm"] == pytest.approx(s.audit["intended_norm"], rel=1e-5, abs=1e-12)
            assert b.audit["rel_err"] < 1e-4 and b.audit["dtype"] == s.audit["dtype"]
            if e == 0.0:
                assert np.all(b.D == 0.0) and np.all(b.cos == 1.0) and b.logp_target_pert == b.logp_target_ref
        if site.site == "block_input":
            # at the injection snapshot only the injected position moves, by the relative dose
            for (d, e), b in zip(pairs, maps):
                assert b.D[site.layer, pos] == pytest.approx(e, rel=1e-3, abs=1e-12)
                assert np.all(b.D[site.layer, pos + 1:] == 0.0)
    # input forms: [N, d] tensor + one dose for all rows == Direction-like objects + a dose list
    site, pos = HookSite(1, "block_input"), T // 2

    class Dir:  # noqa: D401 - duck type
        def __init__(self, v):
            self.vec = v
    m1 = damage_spreading_batch(mb, sample_ids, site, pos, torch.stack(dirs), 0.1, batch_size=3)
    m2 = damage_spreading_batch(mb, sample_ids, site, pos, [Dir(d) for d in dirs], [0.1] * 3)
    assert len(m1) == len(m2) == 3 and all(np.allclose(a.D, b.D, rtol=1e-4, atol=1e-7) for a, b in zip(m1, m2))
    assert np.isnan(m1[0].logp_target_ref) and np.isnan(m1[0].logp_target_pert)
    assert damage_spreading_batch(mb, sample_ids, site, pos, [], []) == []
    with pytest.raises(ValueError):
        damage_spreading_batch(mb, sample_ids, site, pos, dirs, [0.1, 0.2])
    with pytest.raises(ValueError):
        damage_spreading_batch(mb, sample_ids, site, pos, dirs, 0.1, batch_size=1)
    with pytest.raises(ValueError):
        damage_spreading_batch(mb, sample_ids, site, T, dirs, 0.1)


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


def test_ca_symbolize_mass_floor(mb, tok, cache):
    """N is a real symbol: sites with negligible marker mass are N, not labelled by the
    sign of a ~1e-4 difference (finding: floor-less rule == sign(mass_D - mass_E))."""
    L = mb.n_layers
    T = cache.resid[cache.sites[0].key].shape[1]
    lex = _lexicon(tok)
    mE, mD = ca_marker_mass(mb, cache, lex)
    assert mE.shape == mD.shape == (L + 1, T) and (mE >= 0).all() and (mD >= 0).all() and (mE + mD <= 1 + 1e-6).all()
    floor = ca_mass_floor("auto", len(lex["E"]) + len(lex["D"]), mb.vocab)
    assert floor == pytest.approx(CA_MASS_FLOOR_FACTOR * 10 / mb.vocab)
    assert ca_mass_floor(0.0, 10, mb.vocab) == 0.0 and ca_mass_floor(0.01, 10, mb.vocab) == 0.01
    with pytest.raises(ValueError):
        ca_mass_floor("median", 10, mb.vocab)
    with pytest.raises(ValueError):
        ca_mass_floor(-1.0, 10, mb.vocab)
    # pure symbolisation rule
    S = symbolize_masses(np.array([[1e-6, 0.2, 0.3]]), np.array([[2e-6, 0.1, 0.3]]), mass_floor=1e-3)
    assert S.tolist() == [[0, 1, 0]]                       # negligible -> N, tie -> N
    assert symbolize_masses(np.array([1e-6]), np.array([2e-6]), mass_floor=0.0).tolist() == [2]
    assert symbolize_masses(np.array([0.2, 0.2]), np.array([0.1, 0.5]), thresholds=(0.5, 0.9)).shape == (2, 2)
    with pytest.raises(ValueError):
        symbolize_masses(np.zeros(3), np.zeros(2))
    # on the model: the floor-less rule has (almost) no N; the default floor labels the
    # sites of negligible marker mass N and nothing else changes
    S0 = ca_symbolize(mb, cache, lex, mass_floor=0.0)
    S_def = ca_symbolize(mb, cache, lex)
    assert np.array_equal(S_def, ca_symbolize(mb, cache, lex, mass_floor="auto"))
    tot = mE + mD
    assert np.array_equal(S_def == 0, (tot < floor) | (S0 == 0))
    assert (S_def[tot >= floor] == S0[tot >= floor]).all()
    assert (S0 == 0).mean() < 0.05 and (S_def == 0).any()
    assert (S_def[0] == 0).mean() > (S_def[-1] == 0).mean()       # embeddings carry less marker evidence
    assert (S_def[-1] == 1).any()                                   # English text still reads E at the end
    # a floor above every possible mass -> all N; a stricter floor never adds E/D symbols
    assert (ca_symbolize(mb, cache, lex, mass_floor=1.5) == 0).all()
    S_hi = ca_symbolize(mb, cache, lex, mass_floor=10 * floor)
    assert ((S_hi != 0) <= (S_def != 0)).all() and (S_hi == 0).sum() >= (S_def == 0).sum()


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
