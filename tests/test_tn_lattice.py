"""Tests for lsne.tn_lattice (pythia-14m, CPU): QKV layout, exact linear response,
factorised lattice, TNCO / opt_einsum paths and the comparison table."""
import importlib.util

import numpy as np
import opt_einsum as oe
import pytest
import torch

from lsne.config import HookSite
from lsne.model import run_with_cache
from lsne.tn_lattice import (LatticeTN, attention_from_heads, build_factorized_lattice,
                             compare_responses, contract, cost_reference, einsum_expression,
                             factorized_response, head_weights, linear_response_jvp,
                             optimize_path, pca_basis)

HAS_TNCO = importlib.util.find_spec("tnco") is not None
T_W = 8


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def setup(mb, sample_ids):
    """Reference cache (with attention patterns), window, perturbation site and direction."""
    T = int(sample_ids.shape[1])
    L, d = mb.n_layers, mb.d_model
    window = slice(T - T_W, T)
    position = T - 5                                  # inside the window, not the last token
    cache = run_with_cache(mb, sample_ids, "all_block_inputs", need_attn=True)
    direction = torch.randn(d, generator=torch.Generator().manual_seed(0))
    direction = direction / direction.norm()
    return dict(T=T, L=L, d=d, window=window, position=position, cache=cache,
                direction=direction, site=HookSite(L - 1, "block_input"))


@pytest.fixture(scope="module")
def jvp_last(mb, sample_ids, setup):
    """Exact first-order response at the output of the last block (window positions)."""
    s = setup
    out = linear_response_jvp(mb, sample_ids, s["site"], s["position"], s["direction"],
                              [s["L"]], s["window"], method="jvp")
    assert out.shape == (1, T_W, s["d"])
    return out[0]


@pytest.fixture(scope="module")
def lattice_3(mb, setup):
    """Three-block lattice in an r = min(32, d) PCA basis (the TN used for path tests)."""
    s = setup
    L = s["L"]
    r = min(32, s["d"])
    acts = torch.cat([s["cache"].resid[f"L{l}.block_input"][0] for l in range(L)], 0)
    P = pca_basis(acts, r)
    l0 = max(L - 3, 0)
    return build_factorized_lattice(mb, s["cache"], HookSite(l0, "block_input"), s["window"], P,
                                    range(l0, L), position=s["position"], direction=s["direction"])


# --------------------------------------------------------------------------------------
# (a) QKV layout
# --------------------------------------------------------------------------------------
def test_qkv_layout(mb):
    """Per-head W_V / W_O extracted from query_key_value / dense reproduce the attention
    module's output (random input, recorded softmax pattern) to 1e-4."""
    gen = torch.Generator().manual_seed(123)
    ids = torch.randint(0, mb.vocab, (1, 16), generator=gen)
    layer = min(2, mb.n_layers - 1)
    att = mb.layers[layer].attention
    store = {}

    def pre(module, args, kwargs):
        store["x"] = (args[0] if args else kwargs["hidden_states"]).detach().clone()

    def post(module, args, out):
        store["out"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

    h1 = att.register_forward_pre_hook(pre, with_kwargs=True)
    h2 = att.register_forward_hook(post)
    try:
        co = run_with_cache(mb, ids, [], need_attn=True)
    finally:
        h1.remove()
        h2.remove()
    rec = attention_from_heads(mb, layer, store["x"][0], co.attn[layer][0])
    assert rec.shape == store["out"][0].shape
    assert float((rec - store["out"][0]).abs().max()) < 1e-4
    hw = head_weights(mb, layer)
    hs = mb.d_model // mb.n_heads
    assert hw["W_V"].shape == (mb.n_heads, hs, mb.d_model)
    assert hw["W_O"].shape == (mb.n_heads, mb.d_model, hs)
    # the blocked [Q | K | V] hypothesis must NOT reproduce the output (layout is per head)
    W = att.query_key_value.weight.detach().float()
    b = att.query_key_value.bias.detach().float()
    x = store["x"][0].float()
    wrong = hw["b_O"].clone().expand(x.shape[0], -1).clone()
    for h in range(mb.n_heads):
        v = x @ W[2 * mb.d_model + h * hs: 2 * mb.d_model + (h + 1) * hs].T + b[2 * mb.d_model + h * hs: 2 * mb.d_model + (h + 1) * hs]
        wrong = wrong + (co.attn[layer][0, h] @ v) @ hw["W_O"][h].T
    assert float((wrong - store["out"][0]).abs().max()) > 1e-3


# --------------------------------------------------------------------------------------
# pca basis
# --------------------------------------------------------------------------------------
def test_pca_basis_orthonormal(mb, setup):
    s = setup
    acts = s["cache"].resid["L0.block_input"][0]
    r = min(32, s["d"])
    P = pca_basis(acts, r)
    assert P.shape == (s["d"], r) and P.dtype == torch.float32
    assert torch.allclose(P.T @ P, torch.eye(r), atol=1e-5)
    # first direction carries the most variance
    Xc = acts - acts.mean(0, keepdim=True)
    var = (Xc @ P).var(0)
    assert var[0] >= var[1:].max()
    # deterministic
    assert torch.equal(P, pca_basis(acts.numpy(), r))
    # r clipped to d; N < r completed with an orthonormal complement (shape always [d, r])
    Pd = pca_basis(acts, 10 * s["d"])
    assert Pd.shape == (s["d"], s["d"])
    assert torch.allclose(Pd.T @ Pd, torch.eye(s["d"]), atol=1e-4)
    assert torch.allclose(Pd[:, :acts.shape[0]], pca_basis(acts, acts.shape[0]))


# --------------------------------------------------------------------------------------
# (b) exact linear response: jvp vs finite differences
# --------------------------------------------------------------------------------------
def test_jvp_matches_finite_differences(mb, sample_ids, setup, jvp_last):
    s = setup
    fd = linear_response_jvp(mb, sample_ids, s["site"], s["position"], s["direction"],
                             [s["L"]], s["window"], method="fd")[0]
    rel = np.linalg.norm(fd - jvp_last) / np.linalg.norm(jvp_last)
    assert np.isfinite(jvp_last).all() and np.linalg.norm(jvp_last) > 0
    assert rel < 1e-2, rel
    # causality: positions before the perturbation are untouched, the perturbed one is not
    rel_pos = s["position"] - s["window"].start
    assert np.abs(jvp_last[:rel_pos]).max() < 1e-6
    assert np.linalg.norm(jvp_last[rel_pos]) > 0
    # linearity: doubling the direction doubles the response; several target layers stack
    two = linear_response_jvp(mb, sample_ids, s["site"], s["position"], 2 * s["direction"],
                              [s["L"] - 1, s["L"]], s["window"], method="jvp")
    assert two.shape == (2, T_W, s["d"])
    assert np.allclose(two[1], 2 * jvp_last, rtol=1e-4, atol=1e-5)
    # at the injection snapshot itself the response is the direction at the position
    expect = np.zeros((T_W, s["d"]), dtype=np.float32)
    expect[rel_pos] = 2 * s["direction"].numpy()
    assert np.allclose(two[0], expect, atol=1e-5)


# --------------------------------------------------------------------------------------
# (c) factorised lattice, full basis, last block
# --------------------------------------------------------------------------------------
def test_factorized_last_layer_within_50pct(mb, setup, jvp_last):
    """With r = d (no truncation) the only approximations are the frozen attention
    patterns and the frozen LayerNorm scale; the factorised response of the last block
    must be within 50% (Frobenius) of the exact JVP."""
    s = setup
    L, d = s["L"], s["d"]
    P = torch.eye(d)
    resp, info = factorized_response(mb, s["cache"], s["site"], s["position"], s["direction"],
                                     s["window"], P, range(L - 1, L), backend="opt_einsum",
                                     return_info=True)
    assert resp.shape == (T_W, d)
    rel = np.linalg.norm(resp - jvp_last) / np.linalg.norm(jvp_last)
    assert rel < 0.5, rel
    tn = info["tn"]
    assert isinstance(tn, LatticeTN)
    assert any("frozen attention" in a for a in tn.meta["approximations"])
    # the exact-LN variant must not be worse than the frozen one
    resp_x = factorized_response(mb, s["cache"], s["site"], s["position"], s["direction"],
                                 s["window"], P, range(L - 1, L), backend="opt_einsum", ln_mode="exact")
    rel_x = np.linalg.norm(resp_x - jvp_last) / np.linalg.norm(jvp_last)
    assert rel_x <= rel + 1e-6
    # causality is preserved by the lattice as well
    rel_pos = s["position"] - s["window"].start
    assert np.abs(resp[:rel_pos]).max() < 1e-6


def test_lattice_structure(mb, setup, lattice_3):
    s = setup
    tn = lattice_3
    H, r, L = mb.n_heads, tn.meta["r"], s["L"]
    layers = tn.meta["layers"]
    assert tn.n_tensors == 1 + 2 * len(layers)
    assert tn.output_inds == (f"t{L}", f"i{L}")
    assert tn.tensors[0][0] == (f"t{layers[0]}", f"i{layers[0]}")
    dims = tn.dims
    for l in layers:
        assert dims[f"b{l}"] == 2 + H and dims[f"t{l}"] == T_W and dims[f"i{l}"] == r
        G = dict(tn.tensors)[(f"b{l}", f"t{l + 1}", f"t{l}")]
        K = dict(tn.tensors)[(f"b{l}", f"t{l}", f"i{l + 1}", f"i{l}")]
        assert G.shape == (2 + H, T_W, T_W) and K.shape == (2 + H, T_W, r, r)
        assert K.size == (2 + H) * T_W * r * r                     # memory per layer
        # skip branch: identities; MLP branch: positional identity in G
        assert np.allclose(G[0], np.eye(T_W)) and np.allclose(G[-1], np.eye(T_W))
        assert np.allclose(K[0], np.eye(r)[None])
        # heads: causal softmax patterns restricted to the window
        for h in range(H):
            A = G[1 + h]
            assert np.allclose(np.triu(A, 1), 0.0, atol=1e-6)
            assert np.all(A >= -1e-7)
            assert np.allclose(K[1 + h], K[1 + h][:1], atol=1e-6)   # broadcast over t
    # source: onehot(position) (x) P^T direction
    src = tn.tensors[0][1]
    rel_pos = s["position"] - s["window"].start
    assert np.abs(src[np.arange(T_W) != rel_pos]).max() == 0
    P = tn.meta["basis"]
    assert np.allclose(src[rel_pos], P.T @ s["direction"].numpy(), atol=1e-5)
    # einsum expression is well formed
    expr, names = einsum_expression(tn)
    assert expr.count(",") == tn.n_tensors - 1 and "->" in expr
    assert len(set(names.values())) == len(names)


def test_lattice_validation(mb, setup):
    s = setup
    P = torch.eye(s["d"])
    with pytest.raises(ValueError):          # site does not match layers.start
        build_factorized_lattice(mb, s["cache"], HookSite(0, "block_input"), s["window"], P, range(1, 2))
    with pytest.raises(ValueError):          # position outside the window
        build_factorized_lattice(mb, s["cache"], HookSite(s["L"] - 1, "block_input"), s["window"], P,
                                 range(s["L"] - 1, s["L"]), position=0)
    cache_no_attn = run_with_cache(mb, torch.tensor([[1, 2, 3, 4]]), "all_block_inputs", need_attn=False)
    with pytest.raises(ValueError):          # attention patterns required
        build_factorized_lattice(mb, cache_no_attn, HookSite(0, "block_input"), slice(0, 4), P, range(0, 1))


# --------------------------------------------------------------------------------------
# (d) + (e) contraction paths
# --------------------------------------------------------------------------------------
def test_opt_einsum_path_and_contract(lattice_3):
    tn = lattice_3
    info = optimize_path(tn, backend="opt_einsum")
    assert info["backend"] == "opt_einsum"
    assert np.isfinite(info["cost_log10"]) and info["runtime_s"] >= 0
    assert isinstance(info["path"], list) and all(len(p) == 2 for p in info["path"])
    y = contract(tn, info["path"])
    assert y.shape == (T_W, tn.meta["r"]) and np.isfinite(y).all()
    # explicit path == plain einsum result; dict input accepted
    expr, _ = einsum_expression(tn)
    ref = oe.contract(expr, *tn.arrays, optimize="greedy")
    assert np.allclose(y, ref, rtol=1e-4, atol=1e-6)
    assert np.allclose(contract(tn, info), y)
    ref_costs = cost_reference(tn)
    assert np.isfinite(ref_costs["greedy_cost_log10"]) and np.isfinite(ref_costs["dp_cost_log10"])
    assert ref_costs["dp_cost_log10"] <= ref_costs["greedy_cost_log10"] + 1e-9
    assert ref_costs["naive_cost_log10"] >= ref_costs["dp_cost_log10"]


@pytest.mark.skipif(not HAS_TNCO, reason="tnco not installed")
def test_tnco_path_matches_opt_einsum(lattice_3):
    tn = lattice_3
    t_info = optimize_path(tn, backend="tnco", n_steps=300, n_runs=2, seed=0)
    assert t_info["backend"] == "tnco", t_info.get("fallback_reason")
    o_info = optimize_path(tn, backend="opt_einsum")
    for info in (t_info, o_info):
        assert np.isfinite(info["cost_log10"]) and np.isfinite(info["cost_log10_opt_einsum"])
    # (e) the tnco path is a list of index pairs accepted by opt_einsum
    path = t_info["path"]
    assert isinstance(path, list) and len(path) == tn.n_tensors - 1
    assert all(isinstance(p, tuple) and len(p) == 2 and all(isinstance(i, int) for i in p) for p in path)
    expr, _ = einsum_expression(tn)
    _, oe_info = oe.contract_path(expr, *tn.arrays, optimize=path)
    assert np.isfinite(float(oe_info.opt_cost))
    # (d) both backends contract to the same array
    y_t = contract(tn, path)
    y_o = contract(tn, o_info["path"])
    assert np.allclose(y_t, y_o, rtol=1e-4, atol=1e-6)
    # tnco's own cost must not be far from the optimal opt_einsum cost (same path cost)
    assert abs(t_info["cost_log10_opt_einsum"] - o_info["cost_log10"]) < 1.0


def test_optimize_path_fallback_on_tnco_failure(lattice_3, monkeypatch):
    """Any tnco failure falls back to opt_einsum and reports the reason."""
    import lsne.tn_lattice as tl

    def boom(tn, **kw):
        raise RuntimeError("simulated tnco failure")

    monkeypatch.setattr(tl, "_tnco_path", boom)
    info = optimize_path(lattice_3, backend="tnco")
    assert info["backend"] == "opt_einsum" and "simulated" in info["fallback_reason"]
    assert np.isfinite(info["cost_log10"])
    with pytest.raises(ValueError):
        optimize_path(lattice_3, backend="nope")


# --------------------------------------------------------------------------------------
# multi-layer chain and comparison table
# --------------------------------------------------------------------------------------
def test_multilayer_chain_full_basis(mb, sample_ids, setup):
    """Full-basis lattice over the last two blocks: the TN composes the per-block maps
    (the response equals applying the single-block lattices in sequence) and it stays
    in the same ballpark as the exact JVP."""
    s = setup
    L, d = s["L"], s["d"]
    P = torch.eye(d)
    site = HookSite(L - 2, "block_input")
    resp = factorized_response(mb, s["cache"], site, s["position"], s["direction"], s["window"], P,
                               range(L - 2, L), backend="opt_einsum")
    # sequential composition: block L-2 then block L-1 applied to each output column
    r1 = factorized_response(mb, s["cache"], site, s["position"], s["direction"], s["window"], P,
                             range(L - 2, L - 1), backend="opt_einsum")           # [T_w, d]
    seq = np.zeros_like(resp)
    for t in range(T_W):
        if np.linalg.norm(r1[t]) == 0:
            continue
        seq += factorized_response(mb, s["cache"], HookSite(L - 1, "block_input"), s["window"].start + t,
                                   torch.from_numpy(r1[t]), s["window"], P, range(L - 1, L),
                                   backend="opt_einsum")
    assert np.allclose(resp, seq, rtol=1e-3, atol=1e-4)
    jvp = linear_response_jvp(mb, sample_ids, site, s["position"], s["direction"], [L], s["window"])[0]
    rel = np.linalg.norm(resp - jvp) / np.linalg.norm(jvp)
    assert np.isfinite(rel) and rel < 1.0, rel


def test_compare_responses(setup, jvp_last):
    s = setup
    rng = np.random.default_rng(0)
    measured = jvp_last * 1.1 + 0.01 * rng.standard_normal(jvp_last.shape)
    fact = jvp_last * 0.8
    df = compare_responses(measured, jvp_last, fact)
    cols = {"position", "measured_norm", "linear_norm", "factorized_norm", "nonlinearity", "factorization_error"}
    assert cols <= set(df.columns) and len(df) == T_W
    rel_pos = s["position"] - s["window"].start
    row = df.iloc[rel_pos]
    assert abs(row["factorization_error"] - 0.2) < 1e-4
    assert 0 < row["nonlinearity"] < 0.2
    assert row["linear_norm"] > 0
    # torch inputs and explicit positions are accepted
    df2 = compare_responses(torch.from_numpy(measured), torch.from_numpy(jvp_last), fact,
                            positions=range(s["window"].start, s["window"].stop))
    assert list(df2["position"]) == list(range(s["window"].start, s["window"].stop))
    with pytest.raises(ValueError):
        compare_responses(measured[:3], jvp_last, fact)
