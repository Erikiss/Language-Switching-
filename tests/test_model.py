import copy
import dataclasses
import math
import warnings

import torch
import pytest

from lsne.config import HookSite
from lsne.model import (AUDIT_REL_ERR_MAX, Injector, ModelBundle, all_block_inputs, entropy_gradient,
                        generate_steered, logit_lens, make_dose_delta, next_token_stats, run_with_cache)


def test_cache_matches_hidden_states(mb, sample_ids):
    co = run_with_cache(mb, sample_ids, "all_block_inputs")
    with torch.no_grad():
        out = mb.model(input_ids=sample_ids, output_hidden_states=True, use_cache=False)
    L = mb.n_layers
    assert len(co.sites) == L + 1
    for l in range(L):
        assert torch.allclose(co.resid[f"L{l}.block_input"], out.hidden_states[l], atol=1e-5)
    # last block output -> final LN -> hidden_states[L]
    last = co.resid[f"L{L-1}.block_output"]
    assert torch.allclose(mb.final_ln(last), out.hidden_states[L], atol=1e-5)
    assert torch.allclose(logit_lens(mb, last), out.logits.float(), atol=1e-4)


def test_injection_is_local_and_audited(mb, sample_ids):
    site = HookSite(mb.n_layers - 1, "block_input")
    co = run_with_cache(mb, sample_ids, [site])
    h = co.resid[site.key][0, -1]
    torch.manual_seed(1)
    d = torch.randn(mb.d_model)
    delta = make_dose_delta(h, d, 0.01)
    assert torch.isclose(delta.pow(2).mean().sqrt(), 0.01 * h.pow(2).mean().sqrt(), rtol=1e-4)
    with Injector(mb, site, delta, positions=[sample_ids.shape[1] - 1]) as inj:
        with torch.no_grad():
            lo = mb.model(input_ids=sample_ids, use_cache=False).logits.float()
    assert inj.n_calls == 1
    assert inj.audit["rel_err"] < 1e-4          # FP32 Soll/Ist (prereg threshold); ~3e-6 observed
    assert not torch.allclose(lo[0, -1], co.logits[0, -1])
    assert torch.allclose(lo[0, :-1], co.logits[0, :-1], atol=1e-5)   # causal locality


def test_replace_mode_restores_reference(mb, sample_ids):
    site = HookSite(2, "block_input")
    co = run_with_cache(mb, sample_ids, [site])
    ref = co.resid[site.key][0, 3:6]
    with Injector(mb, site, ref, positions=[3, 4, 5], mode="replace"):
        with torch.no_grad():
            lo = mb.model(input_ids=sample_ids, use_cache=False).logits.float()
    assert torch.allclose(lo, co.logits, atol=1e-5)


def test_entropy_gradient(mb, sample_ids):
    site = HookSite(mb.n_layers - 1, "block_input")
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # float() of a grad-requiring tensor warns in torch
        g, H = entropy_gradient(mb, sample_ids, site, sample_ids.shape[1] - 1)
    assert g.shape == (mb.d_model,) and isinstance(H, float)
    assert H > 0 and torch.isfinite(g).all() and g.norm() > 0
    # first-order check: moving along +g increases entropy
    co = run_with_cache(mb, sample_ids, [site])
    h = co.resid[site.key][0, -1]
    delta = make_dose_delta(h, g, 0.002)
    with Injector(mb, site, delta, positions=[sample_ids.shape[1] - 1]):
        with torch.no_grad():
            lo = mb.model(input_ids=sample_ids, use_cache=False).logits.float()
    H2 = float(next_token_stats(lo[:, -1], 0)["entropy"][0])
    assert H2 > H


def test_next_token_stats(mb, sample_ids):
    co = run_with_cache(mb, sample_ids, [])
    st = next_token_stats(co.logits[:, -1], target_id=5, freq_ids=torch.tensor([13, 15, 253]))
    assert set(st) >= {"logp_target", "p_target", "entropy", "top1_id", "p_freq"}
    assert 0 < float(st["p_freq"][0]) < 1


def test_generate_steered_schedule(mb, sample_ids):
    site = HookSite(mb.n_layers // 2, "block_input")
    d = torch.randn(mb.d_model, generator=torch.Generator().manual_seed(3))
    r0 = generate_steered(mb, sample_ids, site, d, lambda k: 0.0, n_new=6, n_samples=2, seed=0)
    r1 = generate_steered(mb, sample_ids, site, None, lambda k: 0.0, n_new=6, n_samples=2, seed=0)
    assert r0["tokens"].shape == (2, 6) and r0["logprobs"].shape == (2, 6)
    assert torch.equal(r0["tokens"], r1["tokens"])          # zero dose == unsteered
    # nothing written -> nothing to audit; the run is not invalid (prereg §4)
    for r in (r0, r1):
        assert r["audit"]["n_dosed"] == 0 and math.isnan(r["audit"]["rel_err"]) and r["audit"]["valid"] is True
    r2 = generate_steered(mb, sample_ids, site, d, lambda k: 0.5, n_new=6, n_samples=2, seed=0)
    assert torch.allclose(r2["eps"], torch.full((6,), 0.5))
    assert not torch.equal(r2["tokens"], r0["tokens"]) or not torch.allclose(r2["logprobs"], r0["logprobs"])
    # FP32 Soll/Ist of the first dosed step (the last prompt position produces token 0); ~7e-8 observed
    a = r2["audit"]
    assert a["dtype"] == "torch.float32" and a["n_dosed"] == 1 and a["positions"] == [sample_ids.shape[1] - 1]
    assert a["eps"] == [0.5] and a["intended_norm"] > 0
    assert a["rel_err"] <= AUDIT_REL_ERR_MAX and a["valid"] is True


def test_generate_steered_audit_first_dosed_step(mb, sample_ids):
    """The audit covers the first *dosed* step, not the first hook call (the prompt chunk)."""
    site = HookSite(mb.n_layers // 2, "block_input")
    d = torch.randn(mb.d_model, generator=torch.Generator().manual_seed(3))
    T0 = sample_ids.shape[1]
    r = generate_steered(mb, sample_ids, site, d, lambda k: 0.0 if k < 2 else 0.5, n_new=4, n_samples=2, seed=0)
    assert torch.allclose(r["eps"], torch.tensor([0.0, 0.0, 0.5, 0.5]))
    a = r["audit"]
    assert a["positions"] == [T0 - 1 + 2] and a["eps"] == [0.5] and a["n_dosed"] == 1
    assert a["intended_norm"] > 0 and a["rel_err"] <= AUDIT_REL_ERR_MAX and a["valid"] is True


def test_generate_steered_audit_bf16(mb, sample_ids):
    """A bfloat16 copy of the model reports a finite Soll/Ist deviation; it exceeds the
    FP32 pre-registration threshold (~4e-3 observed at eps=0.5), so the run is invalid."""
    site = HookSite(mb.n_layers // 2, "block_input")
    d = torch.randn(mb.d_model, generator=torch.Generator().manual_seed(3))
    mb16 = ModelBundle(model=copy.deepcopy(mb.model).to(torch.bfloat16), tokenizer=mb.tokenizer,
                       spec=dataclasses.replace(mb.spec, dtype="bfloat16"), device=mb.device)
    assert mb16.dtype == torch.bfloat16
    r32 = generate_steered(mb, sample_ids, site, d, lambda k: 0.5, n_new=4, n_samples=2, seed=0)
    r16 = generate_steered(mb16, sample_ids, site, d, lambda k: 0.5, n_new=4, n_samples=2, seed=0)
    assert r16["tokens"].shape == (2, 4)
    a = r16["audit"]
    assert a["dtype"] == "torch.bfloat16" and a["n_dosed"] == 1 and a["positions"] == [sample_ids.shape[1] - 1]
    assert math.isfinite(a["rel_err"]) and a["rel_err"] > r32["audit"]["rel_err"]
    assert a["rel_err"] > AUDIT_REL_ERR_MAX and a["valid"] is False
