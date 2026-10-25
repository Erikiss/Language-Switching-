"""Tests for lsne.heads (pythia-14m, CPU): head layout / reconstruction, ablation,
direct logit effects, head-level steering and the worst-head selection."""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite
from lsne.model import _first_tensor, run_with_cache
from lsne.heads import (HeadInjector, ablate_heads, dense_bias, detect_head_layout, head_ablation_effects,
                        head_contributions, head_dim, head_direct_logit_effect, head_output_weight,
                        head_preimage, head_slices, steer_at_head, verify_head_reconstruction, worst_head)


def _logits(mb, ids):
    with torch.no_grad():
        return mb.model(input_ids=ids, use_cache=False).logits.float()


@pytest.fixture(scope="module")
def layer(mb):
    return mb.n_layers // 2


@pytest.fixture(scope="module")
def base_logits(mb, sample_ids):
    return _logits(mb, sample_ids)


# --------------------------------------------------------------------------------------
def test_head_layout_and_slices(mb, layer):
    layout = detect_head_layout(mb, 0)
    assert layout in ("contiguous", "interleaved")
    H, d, dh = mb.n_heads, mb.d_model, head_dim(mb)
    assert dh * H == d
    sl = head_slices(mb, layer)
    assert len(sl) == H and all(isinstance(s, slice) for s in sl)
    cols = sorted(sum((list(range(*s.indices(d))) for s in sl), []))
    assert cols == list(range(d))                          # the slices partition the dense input
    assert head_output_weight(mb, layer, 0).shape == (d, dh)
    with pytest.raises(ValueError):
        head_slices(mb, mb.n_layers)


def test_head_contributions_reconstruct_attention_output(mb, sample_ids, layer):
    T = sample_ids.shape[1]
    c, out = head_contributions(mb, sample_ids, layer, return_output=True)
    assert c.shape == (mb.n_heads, T, mb.d_model) and out.shape == (1, T, mb.d_model)
    assert torch.isfinite(c).all()
    recon = c.sum(0) + dense_bias(mb, layer)
    assert torch.allclose(recon, out[0], atol=1e-4)
    rep = verify_head_reconstruction(mb, sample_ids, layer)
    assert rep["max_abs_err"] < 1e-4 and rep["n_heads"] == mb.n_heads
    # independent cross-check against the recorded attention output site
    co = run_with_cache(mb, sample_ids, [HookSite(layer, "attn_output")])
    assert torch.allclose(co.resid[HookSite(layer, "attn_output").key][0], recon, atol=1e-4)


def test_zero_ablation_of_all_heads_gives_bias_only_output(mb, sample_ids, layer, base_logits):
    store = {}

    def post(module, args, out):
        store["o"] = _first_tensor(out).detach().float()

    handle = mb.layers[layer].attention.register_forward_hook(post)
    try:
        with ablate_heads(mb, layer, "all", mode="zero") as ab:
            lo = _logits(mb, sample_ids)
        assert ab.n_calls == 1
    finally:
        handle.remove()
    bias = dense_bias(mb, layer)[None].expand_as(store["o"][0])
    assert torch.allclose(store["o"][0], bias, atol=1e-5)
    assert not torch.allclose(lo, base_logits)
    # the model is restored once the context has exited
    assert torch.equal(_logits(mb, sample_ids), base_logits)


def test_mean_ablation_replaces_slice_by_position_mean_and_restores(mb, sample_ids, layer, base_logits):
    store = {}

    def pre(module, args, kwargs):
        store.setdefault("x", []).append(args[0].detach().float().clone())

    dense = mb.layers[layer].attention.dense
    handle = dense.register_forward_pre_hook(pre, with_kwargs=True)
    try:
        _logits(mb, sample_ids)                                   # unablated dense input
    finally:
        handle.remove()
    with ablate_heads(mb, layer, [1], mode="mean") as ab:
        # pre-hooks run in registration order: register the recorder *after* the ablation hook
        handle = dense.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            lo = _logits(mb, sample_ids)
        finally:
            handle.remove()
    x_raw, x_abl = store["x"][0][0], store["x"][1][0]              # [T, H*dh]
    s = head_slices(mb, layer)[1]
    assert torch.allclose(x_abl[:, s], x_raw[:, s].mean(0, keepdim=True).expand_as(x_abl[:, s]), atol=1e-6)
    other = [h for h in range(mb.n_heads) if h != 1]
    for h in other:
        assert torch.equal(x_abl[:, head_slices(mb, layer)[h]], x_raw[:, head_slices(mb, layer)[h]])
    assert ab.n_calls == 1 and 1 in ab.last_mean
    assert not torch.allclose(lo, base_logits)
    assert torch.equal(_logits(mb, sample_ids), base_logits)
    # explicit mean source
    ms = torch.zeros(mb.d_model)
    with ablate_heads(mb, layer, [0, 2], mode="mean", mean_source=ms):
        lo2 = _logits(mb, sample_ids)
    with ablate_heads(mb, layer, [0, 2], mode="zero"):
        lo3 = _logits(mb, sample_ids)
    assert torch.allclose(lo2, lo3, atol=1e-5)                   # zero mean source == zero ablation
    with pytest.raises(ValueError):
        ablate_heads(mb, layer, [mb.n_heads], mode="zero")
    with pytest.raises(ValueError):
        ablate_heads(mb, layer, [0], mode="bogus")


def test_head_direct_logit_effect_shape_and_finiteness(mb, sample_ids, layer):
    e = head_direct_logit_effect(mb, sample_ids, layer, target_id=5)
    assert e.shape == (mb.n_heads,) and torch.isfinite(e).all()
    full = head_direct_logit_effect(mb, sample_ids, layer, target_id=None, position=-1)
    assert full.shape == (mb.n_heads, mb.vocab) and torch.isfinite(full).all()
    assert torch.allclose(full[:, 5], e)
    e_layer = head_direct_logit_effect(mb, sample_ids, layer, target_id=5, h_ref="layer", position=3)
    assert e_layer.shape == (mb.n_heads,) and torch.isfinite(e_layer).all()
    with pytest.raises(ValueError):
        head_direct_logit_effect(mb, sample_ids, layer, 5, h_ref="nope")


def test_head_injector_preimage_and_locality(mb, sample_ids, layer, base_logits):
    d, dh = mb.d_model, head_dim(mb)
    T = sample_ids.shape[1]
    direction = torch.randn(d, generator=torch.Generator().manual_seed(0))
    u, realised = head_preimage(mb, layer, 1, direction)
    W = head_output_weight(mb, layer, 1)
    assert u.shape == (dh,) and torch.allclose(W @ u, realised, atol=1e-5)
    # least squares: the residual is orthogonal to the column space of W_O^h
    assert float((W.T @ (direction - realised)).abs().max()) < 1e-3
    store = {}

    def post(module, args, out):
        store["o"] = _first_tensor(out).detach().float()[0]

    handle = mb.layers[layer].attention.register_forward_hook(post)
    try:
        _logits(mb, sample_ids)
        o_base = store["o"].clone()
        with HeadInjector(mb, layer, 1, direction, positions=[T - 1]) as inj:
            lo = _logits(mb, sample_ids)
        o_inj = store["o"]
    finally:
        handle.remove()
    assert inj.n_calls == 1 and inj.audit["space"] == "resid" and 0 < inj.audit["rel_err"] < 1
    # the attention output changed by exactly the realised vector at the last position only
    assert torch.allclose(o_inj[-1] - o_base[-1], realised, atol=1e-4)
    assert torch.allclose(o_inj[:-1], o_base[:-1], atol=1e-6)
    assert torch.allclose(lo[0, :-1], base_logits[0, :-1], atol=1e-5)
    assert not torch.allclose(lo[0, -1], base_logits[0, -1])
    # head-space delta
    with HeadInjector(mb, layer, 0, torch.ones(dh), positions=[0], space="head") as inj2:
        _logits(mb, sample_ids)
    assert inj2.audit["space"] == "head" and inj2.audit["rel_err"] < 1e-6
    assert torch.equal(_logits(mb, sample_ids), base_logits)


def test_steer_at_head_changes_only_last_position(mb, sample_ids, layer, base_logits):
    T = sample_ids.shape[1]
    direction = torch.randn(mb.d_model, generator=torch.Generator().manual_seed(1))
    last, info = steer_at_head(mb, sample_ids, layer, 2, direction, eps=0.5, position=T - 1, return_info=True)
    assert last.shape == (mb.vocab,) and torch.isfinite(last).all()
    assert torch.allclose(info["logits_full"][:-1], base_logits[0, :-1], atol=1e-5)
    assert not torch.allclose(last, base_logits[0, -1])
    assert torch.allclose(info["logits_full"][-1], last)
    # dose: the realised residual delta has RMS eps * RMS(head contribution)
    assert np.isclose(info["rms_delta"], 0.5 * info["rms_head"], rtol=1e-4)
    assert info["cos_realised_direction"] > 0
    # same call without info returns the same logits; Direction objects are accepted
    from lsne.directions import Direction
    last2 = steer_at_head(mb, sample_ids, layer, 2, Direction("x", direction, HookSite(layer)), 0.5, -1)
    # identical up to float32 round-off of the scaling (u differs by ~3e-8, amplified by the later blocks)
    assert float((last2 - last).abs().max()) < 5e-3
    # a dose at an earlier position changes that position's logits but not the earlier ones
    last3, info3 = steer_at_head(mb, sample_ids, layer, 2, direction, 0.5, position=3, return_info=True)
    assert torch.allclose(info3["logits_full"][:3], base_logits[0, :3], atol=1e-5)
    assert not torch.allclose(info3["logits_full"][3], base_logits[0, 3])
    assert torch.equal(_logits(mb, sample_ids), base_logits)


def test_head_ablation_effects_table(mb, tok, layer):
    def variant(text, fam, K, C):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        return types.SimpleNamespace(ids=ids, target_id=tok(" soldier", add_special_tokens=False)["input_ids"][0],
                                     family_id=fam, formulation="A", K=K, C=C)
    vs = [variant("The report is about the", "F01", 0, "C1"),
          dict(ids=tok("Der Bericht handelt von dem", add_special_tokens=False)["input_ids"],
               target_id=tok(" soldier", add_special_tokens=False)["input_ids"][0], family_id="F02", K=2, C="C2")]
    grid = [0, layer]
    df = head_ablation_effects(mb, vs, grid, mode="mean")
    assert list(df.columns) == ["variant", "family_id", "formulation", "K", "C", "layer", "head",
                                "logp_base", "logp_ablated", "delta"]
    assert len(df) == len(vs) * len(grid) * mb.n_heads
    assert np.allclose(df.delta, df.logp_ablated - df.logp_base)
    assert df.groupby("variant").logp_base.nunique().max() == 1
    assert set(df.family_id) == {"F01", "F02"} and set(df.layer) == set(grid)
    assert np.isfinite(df[["logp_base", "logp_ablated", "delta"]].values).all()
    assert len(head_ablation_effects(mb, vs, [0], mode="zero", limit=1)) == mb.n_heads


def test_worst_head_on_synthetic_table():
    rng = np.random.default_rng(0)
    fams = ["F01", "F02", "F03", "F04"]
    rows = []
    for fam in fams:
        for layer in (1, 3):
            for head in range(4):
                for rep in range(3):
                    delta = 0.05 * rng.standard_normal()
                    if (layer, head) == (3, 2) and fam in ("F01", "F02"):
                        delta -= 2.0                       # planted: costly on the weak families
                    if (layer, head) == (1, 0):
                        delta -= 1.0                       # costly everywhere -> not "worst" relative to stable
                    rows.append(dict(variant=rep, family_id=fam, formulation="A", K=2, C="C1", layer=layer,
                                     head=head, logp_base=-3.0, logp_ablated=-3.0 + delta, delta=delta))
    eff = pd.DataFrame(rows)
    res = worst_head(eff, families_weak=["F01", "F02"], families_stable=["F03", "F04"])
    assert (res["layer"], res["head"]) == (3, 2)
    assert res["score"] < -1.5 and np.isclose(res["loss"], -res["score"])
    tab = res["table"]
    assert len(tab) == 8 and set(tab.columns) >= {"layer", "head", "mean_delta_weak", "mean_delta_stable", "score"}
    assert tab.iloc[0]["layer"] == 3 and tab.iloc[0]["head"] == 2       # sorted, worst first
    assert np.isclose(tab.score, tab.mean_delta_weak - tab.mean_delta_stable).all()
    assert (tab.n_weak == 6).all() and (tab.n_stable == 6).all()
    # the uniformly costly head is not selected when stable families are given
    res_abs = worst_head(eff, families_weak=fams, families_stable=[])
    assert (res_abs["layer"], res_abs["head"]) == (3, 2) or (res_abs["layer"], res_abs["head"]) == (1, 0)
    empty = worst_head(eff.iloc[0:0], ["F01"], ["F03"])
    assert empty["layer"] is None and empty["table"].empty
