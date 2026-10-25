"""Tensor-network bridge: the linearised transformer on the (layer x position) lattice.

DESIGN §5 / SPEC ``lsne/tn_lattice.py``.  Three levels of "response" of the residual
stream to a small perturbation at one lattice site are compared:

(a) the measured non-linear perturbation (``lsne.dynamics.damage_spreading``, not
    imported here),
(b) the exact first-order response obtained by forward-mode autodiff through the
    model (``linear_response_jvp``), and
(c) the *factorised* linear response: the Jacobian of every block at the working
    point is written as a sum of branches (skip + one branch per attention head +
    MLP) with frozen attention patterns and frozen LayerNorm scales, projected onto an
    r-dimensional basis of the residual stream and restricted to a window of
    positions.  This is a genuine tensor network on the lattice whose contraction
    order is non-trivial; the order is optimised with TNCO (fallback: opt_einsum) and
    the network is contracted with opt_einsum (``build_factorized_lattice``,
    ``optimize_path``, ``contract``, ``factorized_response``).

Model structure (GPT-NeoX, parallel residual)::

    h_{l+1} = h_l + Attn_l(LN1_l(h_l)) + MLP_l(LN2_l(h_l))
    Attn_l(x)_{t'} = sum_h sum_t A^h[t', t] (x_t W_V^h^T + b_V^h) W_O^h^T + b_O

Weight layout (verified empirically on pythia-14m, see ``head_weights`` and
``tests/test_tn_lattice.py::test_qkv_layout``): ``attention.query_key_value.weight``
has shape ``[3d, d]`` and its rows are grouped *per head* as ``[q_h | k_h | v_h]``
(each ``head_size`` rows); ``attention.dense.weight[:, h*hs:(h+1)*hs]`` is ``W_O^h``.

Factorised encoding of block ``l`` (window of ``T_w`` positions, basis ``P [d, r]``):

* branch index ``b_l`` of size ``2 + H`` (0 = skip, 1..H = heads, H+1 = MLP),
* ``G_l[b_l, t_{l+1}, t_l]``: skip -> ``I``; head h -> ``A^h`` restricted to the window
  (rows = output / query positions); MLP -> ``I``,
* ``K_l[b_l, t_l, i_{l+1}, i_l]``: skip -> ``I_r`` for every ``t_l``; head h ->
  ``M^h = P^T W_O^h W_V^h diag(gamma1) / sigma1_mean P`` broadcast over ``t_l``
  (``sigma1_mean`` = mean LN1 std over the window: frozen-LN approximation, the LN
  mean-centering / normalisation projection is dropped); MLP ->
  ``M_t = P^T W_out diag(act'(z_t)) W_in diag(gamma2) / sigma2_t P`` with ``z_t`` the
  MLP pre-activation at position ``t`` recomputed from the cache.

The map of one block is ``sum_{b_l, t_l} G_l[b, t', t] K_l[b, t, i', i]``: ``t_l`` is a
hyper-index shared by ``G_l`` and ``K_l`` (and by ``G_{l-1}``).  Memory per block is
``(2+H) * T_w * r * r`` floats for ``K_l``.  The source vector is
``v[t_0, i_0] = onehot(position) (x) (P^T direction)``; the open indices are
``(t_L, i_L)`` at the block output of the last layer in ``layers``.

Approximations of level (c) relative to level (b), all documented in
``LatticeTN.meta['approximations']``:

1. frozen attention patterns (the first-order change of ``A^h`` through q / k is
   ignored; only the value path is linearised),
2. frozen LayerNorm: ``d LN(h) ~ diag(gamma) / sigma`` (centering and the projection
   on the normalised state are dropped; the head scale is the window-mean ``sigma``),
3. truncation to the basis ``P`` (exact for ``r = d``),
4. the window restriction is exact by causality as long as the window starts at or
   before the perturbed position (positions before the window are never perturbed).
"""
from __future__ import annotations

import contextlib
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np
import opt_einsum as oe
import pandas as pd
import torch

from .config import HookSite
from .model import CacheOut, Injector, ModelBundle, _first_tensor, _get_hidden

__all__ = [
    "LatticeTN", "pca_basis", "head_weights", "attention_from_heads",
    "linear_response_jvp", "build_factorized_lattice", "einsum_expression",
    "optimize_path", "contract", "cost_reference", "factorized_response",
    "compare_responses",
]


# --------------------------------------------------------------------------------------
# data container
# --------------------------------------------------------------------------------------
@dataclass
class LatticeTN:
    """A tensor network on the (layer x position) lattice.

    ``tensors``: list of ``(index_names, numpy array)`` in a fixed order (the source
    vector first, then ``G_l, K_l`` for every block); ``output_inds``: ordered tuple of
    the open index names; ``meta``: dimensions, layer list, window, basis, branch names
    and the list of approximations.
    """
    tensors: list[tuple[tuple[str, ...], np.ndarray]]
    output_inds: tuple[str, ...]
    meta: dict = field(default_factory=dict)

    # -- convenience ---------------------------------------------------------------
    @property
    def inds(self) -> list[tuple[str, ...]]:
        return [inds for inds, _ in self.tensors]

    @property
    def arrays(self) -> list[np.ndarray]:
        return [arr for _, arr in self.tensors]

    @property
    def dims(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for inds, arr in self.tensors:
            for name, n in zip(inds, arr.shape):
                if out.setdefault(name, int(n)) != int(n):
                    raise ValueError(f"inconsistent dimension for index {name}")
        return out

    @property
    def n_tensors(self) -> int:
        return len(self.tensors)

    def nbytes(self) -> int:
        return int(sum(arr.nbytes for _, arr in self.tensors))


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _to_tensor(x, dtype=torch.float32, device=None) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x)
    t = t.to(dtype=dtype)
    if device is not None:
        t = t.to(device)
    return t


def _window_positions(window: slice, T: int) -> list[int]:
    pos = list(range(*window.indices(T)))
    if not pos:
        raise ValueError(f"window {window} is empty for T={T}")
    if any(b - a != 1 for a, b in zip(pos[:-1], pos[1:])):
        raise ValueError("window must be a contiguous slice with step 1")
    return pos


def _snapshot_index(site: HookSite) -> int:
    """Residual-snapshot index (0..L) at which a perturbation at ``site`` enters.

    A block input of layer l is snapshot l; the block / MLP / attention output of
    layer l is added to the residual and therefore first appears at snapshot l+1."""
    return site.layer if site.site == "block_input" else site.layer + 1


def _site_for_snapshot(mb: ModelBundle, snap: int) -> HookSite:
    L = mb.n_layers
    if snap < 0 or snap > L:
        raise ValueError(f"snapshot {snap} out of range 0..{L}")
    return HookSite(snap, "block_input") if snap < L else HookSite(L - 1, "block_output")


@contextlib.contextmanager
def _record_live(mb: ModelBundle, sites: Sequence[HookSite], store: dict):
    """Record residuals at ``sites`` *without* detaching (keeps forward-mode tangents)."""
    handles = []
    try:
        for s in sites:
            mod = mb.module_for(s)
            if s.site == "block_input":
                def pre(module, args, kwargs, _k=s.key):
                    store[_k] = _get_hidden(args, kwargs)
                handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(module, args, out, _k=s.key):
                    store[_k] = _first_tensor(out)
                handles.append(mod.register_forward_hook(post))
        yield
    finally:
        for h in handles:
            h.remove()


def _ln_scale(ln: torch.nn.Module, h: torch.Tensor) -> torch.Tensor:
    """Per-position LayerNorm scale ``sigma_t = sqrt(var(h_t) + eps)`` for ``h [T, d]``."""
    var = h.float().var(dim=-1, unbiased=False)
    return (var + float(ln.eps)).sqrt()


def _activation_derivative(act: Callable, z: torch.Tensor) -> torch.Tensor:
    """Elementwise derivative ``act'(z)`` of an elementwise activation via autograd
    (exact for any ``hidden_act``; for exact GELU this equals ``Phi(z) + z phi(z)``)."""
    with torch.enable_grad():
        zz = z.detach().clone().requires_grad_(True)
        y = act(zz)
        (g,) = torch.autograd.grad(y.sum(), zz)
    return g.detach()


# --------------------------------------------------------------------------------------
# 1. PCA basis
# --------------------------------------------------------------------------------------
def pca_basis(acts, r: int) -> torch.Tensor:
    """Orthonormal basis ``P [d, r]`` of the top-``r`` principal directions of the
    activations ``acts [N, d]`` (SVD of the centred matrix; float32 output).

    ``r`` is clipped to ``d``.  Columns are ordered by decreasing variance and their
    signs are fixed (largest-magnitude entry positive) for determinism.  If fewer than
    ``r`` principal directions exist (``N < r``) the basis is completed with a
    deterministic orthonormal complement so that the shape is always ``[d, r]``."""
    X = _to_tensor(acts, dtype=torch.float64).reshape(-1, int(np.shape(acts)[-1]))
    N, d = X.shape
    r = int(max(1, min(r, d)))
    Xc = X - X.mean(0, keepdim=True)
    # torch.linalg.svd on the [N, d] matrix; right singular vectors span the data.
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    P = Vh[: min(r, N)].T.contiguous()                         # [d, r0]
    sign = torch.sign(P[P.abs().argmax(0), torch.arange(P.shape[1])])
    sign[sign == 0] = 1.0
    P = P * sign[None, :]
    if P.shape[1] < r:
        # complete with the orthogonal complement (QR of [P | I_d] keeps span(P) first)
        Q, _ = torch.linalg.qr(torch.cat([P, torch.eye(d, dtype=P.dtype)], 1))
        P = torch.cat([P, Q[:, P.shape[1]: r]], 1)
    return P.to(torch.float32)


# --------------------------------------------------------------------------------------
# head-wise weights (verified layout)
# --------------------------------------------------------------------------------------
def head_weights(mb: ModelBundle, layer: int) -> dict[str, torch.Tensor]:
    """Per-head attention weights of block ``layer``.

    Returns ``W_Q, W_K, W_V [H, hs, d]``, ``b_Q, b_K, b_V [H, hs]`` (zeros if the model
    has no attention bias), ``W_O [H, d, hs]`` (so that the head's contribution to the
    attention output is ``z_h @ W_O[h].T``) and ``b_O [d]``.

    Layout (GPT-NeoX, verified on pythia-14m): ``query_key_value.weight`` is
    ``[3d, d]`` with rows grouped per head as ``[q_h | k_h | v_h]``;
    ``dense.weight[:, h*hs:(h+1)*hs]`` is the output projection of head ``h``."""
    att = mb.layers[layer].attention
    H, d = mb.n_heads, mb.d_model
    hs = d // H
    W = att.query_key_value.weight.detach().float()            # [3d, d]
    b = att.query_key_value.bias
    b = b.detach().float() if b is not None else torch.zeros(3 * d, device=W.device)
    W = W.view(H, 3, hs, d)
    b = b.view(H, 3, hs)
    Wo = att.dense.weight.detach().float()                     # [d, d]
    bo = att.dense.bias
    bo = bo.detach().float() if bo is not None else torch.zeros(d, device=W.device)
    return dict(
        W_Q=W[:, 0], W_K=W[:, 1], W_V=W[:, 2],
        b_Q=b[:, 0], b_K=b[:, 1], b_V=b[:, 2],
        W_O=Wo.view(d, H, hs).permute(1, 0, 2).contiguous(),   # [H, d, hs]
        b_O=bo,
    )


def attention_from_heads(mb: ModelBundle, layer: int, x: torch.Tensor,
                         pattern: torch.Tensor) -> torch.Tensor:
    """Reproduce the attention module output of block ``layer`` from the per-head
    weights of ``head_weights`` and a recorded softmax pattern.

    ``x [T, d]`` is the *normalised* block input ``LN1(h)``, ``pattern [H, T, T]`` the
    softmax attention weights (rows = query positions).  Used to verify the weight
    layout against the module's own output."""
    hw = head_weights(mb, layer)
    x = x.float()
    out = hw["b_O"].clone().expand(x.shape[0], -1).clone()
    for h in range(mb.n_heads):
        v = x @ hw["W_V"][h].T + hw["b_V"][h]                  # [T, hs]
        z = pattern[h].float() @ v                              # [T, hs]
        out = out + z @ hw["W_O"][h].T
    return out


# --------------------------------------------------------------------------------------
# 2. exact linear response (forward-mode AD)
# --------------------------------------------------------------------------------------
def _response_function(mb: ModelBundle, input_ids: torch.Tensor, site_from: HookSite,
                       position: int, sites_to: list[HookSite], w_pos: list[int]):
    """Build ``f(delta [d]) -> Tensor[len(sites_to), T_w, d]``: run the model with
    ``delta`` added at ``(site_from, position)`` and return the residuals at
    ``sites_to`` restricted to the window positions ``w_pos``."""
    T = int(input_ids.shape[1])
    onehot = torch.zeros(T, device=mb.device)
    onehot[position] = 1.0
    idx = torch.as_tensor(w_pos, device=mb.device)

    def f(delta: torch.Tensor) -> torch.Tensor:
        store: dict[str, torch.Tensor] = {}
        full = (onehot[:, None] * delta.to(mb.device)[None, :])[None]      # [1, T, d], no in-place ops
        # The injector is registered first so that a recording hook on the same module
        # (layers_to containing the injection snapshot) sees the perturbed residual.
        with Injector(mb, site_from, full), _record_live(mb, sites_to, store):
            mb.model(input_ids=input_ids, use_cache=False)
        return torch.stack([store[s.key][0].index_select(0, idx) for s in sites_to], 0)

    return f


def _fd_response(f: Callable, direction: torch.Tensor, step: float) -> torch.Tensor:
    """Central finite differences with one Richardson extrapolation step:
    ``D(h) = (f(h u) - f(-h u)) / 2h``; ``D_R = (4 D(h/2) - D(h)) / 3`` (error O(h^4))."""
    with torch.no_grad():
        def D(h):
            return (f(h * direction) - f(-h * direction)) / (2.0 * h)
        return (4.0 * D(step / 2.0) - D(step)) / 3.0


def linear_response_jvp(mb: ModelBundle, input_ids: torch.Tensor, site_from: HookSite,
                        position: int, direction: torch.Tensor, layers_to: list[int],
                        window: slice, method: str = "auto",
                        fd_step_rel: float = 0.05) -> np.ndarray:
    """Exact first-order response of the residual stream to a unit perturbation.

    ``direction [d]`` is added at ``(site_from, position)``; the returned array
    ``[len(layers_to), T_w, d]`` holds the directional derivative of the block inputs
    at ``layers_to`` (index ``n_layers`` = output of the last block) restricted to the
    positions in ``window``.  Because the response is linear in the perturbation the
    scale of ``direction`` only rescales the output.

    ``method``: ``"jvp"`` (forward-mode AD, ``torch.func.jvp`` through the model with
    the injection hook), ``"fd"`` (central finite differences with Richardson
    extrapolation, step ``fd_step_rel * ||h_ref||``), ``"auto"`` (jvp, falling back to
    fd if forward-mode AD fails for this model).  No tensor network is involved."""
    input_ids = input_ids.to(mb.device)
    T = int(input_ids.shape[1])
    L = mb.n_layers
    if not (0 <= position < T):
        raise ValueError(f"position {position} out of range for T={T}")
    w_pos = _window_positions(window, T)
    sites_to = [_site_for_snapshot(mb, int(l)) for l in layers_to]
    direction = _to_tensor(direction, device=mb.device)
    f = _response_function(mb, input_ids, site_from, position, sites_to, w_pos)

    if method not in ("auto", "jvp", "fd"):
        raise ValueError(method)
    out: Optional[torch.Tensor] = None
    if method in ("auto", "jvp"):
        try:
            with torch.no_grad():
                _, out = torch.func.jvp(f, (torch.zeros(mb.d_model, device=mb.device),), (direction,))
        except Exception as e:                       # pragma: no cover - fallback path
            if method == "jvp":
                raise
            warnings.warn(f"torch.func.jvp failed ({e!r}); using finite differences")
            out = None
    if out is None:
        # Finite-difference fallback (documented): step relative to the reference norm.
        store: dict[str, torch.Tensor] = {}
        with torch.no_grad(), _record_live(mb, [site_from], store):
            mb.model(input_ids=input_ids, use_cache=False)
        h_ref = store[site_from.key][0, position].float()
        step = float(fd_step_rel * h_ref.norm().clamp_min(1e-6) / direction.norm().clamp_min(1e-30))
        out = _fd_response(f, direction, step)
    return out.detach().float().cpu().numpy()


# --------------------------------------------------------------------------------------
# 3. factorised lattice
# --------------------------------------------------------------------------------------
def _block_factors(mb: ModelBundle, cache: CacheOut, layer: int, w_pos: list[int],
                   P: torch.Tensor, ln_mode: str = "frozen") -> tuple[torch.Tensor, torch.Tensor, dict]:
    """``G [2+H, T_w, T_w]`` and ``K [2+H, T_w, r, r]`` of block ``layer`` (float32)."""
    key = f"L{layer}.block_input"
    if key not in cache.resid:
        raise KeyError(f"cache lacks {key}; build it with sites='all_block_inputs'")
    if cache.attn is None:
        raise ValueError("cache.attn is None; run_with_cache(..., need_attn=True) is required")
    lyr = mb.layers[layer]
    H, d = mb.n_heads, mb.d_model
    r = P.shape[1]
    T_w = len(w_pos)
    idx = torch.as_tensor(w_pos)
    h = cache.resid[key][0].detach().float().cpu()                     # [T, d]
    h_w = h.index_select(0, idx)                                       # [T_w, d]
    A = cache.attn[layer][0].detach().float().cpu()                    # [H, T, T]
    A_w = A[:, idx][:, :, idx]                                         # [H, T_w, T_w]
    P = P.float().cpu()
    ln1, ln2 = lyr.input_layernorm, lyr.post_attention_layernorm
    g1 = ln1.weight.detach().float().cpu()
    g2 = ln2.weight.detach().float().cpu()
    sig1 = _ln_scale(ln1, h_w)                                         # [T_w]
    sig2 = _ln_scale(ln2, h_w)                                         # [T_w]

    def ln_jacobian(gamma: torch.Tensor, sig: torch.Tensor, hh: torch.Tensor, per_pos: bool) -> torch.Tensor:
        """``J_t [T_w, d, d]`` (or broadcast ``[1, d, d]``) of the LayerNorm at the window."""
        if ln_mode == "frozen":
            if per_pos:
                return (gamma[None, :, None] / sig[:, None, None]) * torch.eye(d)[None]
            return (gamma[:, None] / sig.mean()) * torch.eye(d)[None]
        if ln_mode == "exact":
            # exact LN Jacobian: gamma/sigma (I - 11^T/d - n n^T/d), n = normalised state
            n = (hh - hh.mean(-1, keepdim=True)) / sig[:, None]
            proj = torch.eye(d)[None] - torch.ones(d, d)[None] / d - n[:, :, None] * n[:, None, :] / d
            return (gamma[None, :, None] / sig[:, None, None]) * proj
        raise ValueError(f"unknown ln_mode {ln_mode}")

    hw = head_weights(mb, layer)
    G = torch.zeros(2 + H, T_w, T_w)
    K = torch.zeros(2 + H, T_w, r, r)
    eye_T = torch.eye(T_w)
    # skip
    G[0] = eye_T
    K[0] = torch.eye(r)[None].expand(T_w, r, r)
    # heads: J_t = W_O^h W_V^h J_LN1,t
    J1 = ln_jacobian(g1, sig1, h_w, per_pos=False)                     # [1 or T_w, d, d]
    for hidx in range(H):
        WOV = hw["W_O"][hidx].cpu() @ hw["W_V"][hidx].cpu()            # [d, d]
        M = torch.einsum("ri,ij,tjk,ks->trs", P.T, WOV, J1, P)         # [1 or T_w, r, r]
        G[1 + hidx] = A_w[hidx]
        K[1 + hidx] = M.expand(T_w, r, r)
    # MLP: J_t = W_out diag(act'(z_t)) W_in J_LN2,t
    with torch.no_grad():
        x2 = ln2(h_w.to(mb.device)).float().cpu()                      # LN2(h) at the window
        W_in = lyr.mlp.dense_h_to_4h.weight.detach().float().cpu()     # [4d, d]
        b_in = lyr.mlp.dense_h_to_4h.bias
        b_in = b_in.detach().float().cpu() if b_in is not None else torch.zeros(W_in.shape[0])
        W_out = lyr.mlp.dense_4h_to_h.weight.detach().float().cpu()    # [d, 4d]
        z = x2 @ W_in.T + b_in                                         # [T_w, 4d] pre-activation
    gprime = _activation_derivative(lyr.mlp.act, z.to(mb.device)).cpu()   # [T_w, 4d]
    J2 = ln_jacobian(g2, sig2, h_w, per_pos=True)                      # [T_w, d, d]
    inner = torch.einsum("ij,tjk,ks->tis", W_in, J2, P)                # [T_w, 4d, r]
    inner = gprime[:, :, None] * inner
    M_mlp = torch.einsum("ri,ij,tjs->trs", P.T, W_out, inner)          # [T_w, r, r]
    G[1 + H] = eye_T
    K[1 + H] = M_mlp
    info = dict(sigma1_mean=float(sig1.mean()), sigma2=sig2.numpy().tolist())
    return G, K, info


def build_factorized_lattice(mb: ModelBundle, cache: CacheOut, site_from: HookSite,
                             window: slice, basis, layers: range,
                             position: Optional[int] = None, direction=None,
                             ln_mode: str = "frozen", dtype=np.float32) -> LatticeTN:
    """Factorised linear-response tensor network for the blocks in ``layers``.

    ``cache`` must come from ``run_with_cache(mb, ids, "all_block_inputs", need_attn=True)``
    for the *unperturbed* input.  ``site_from`` fixes the residual snapshot where the
    perturbation enters (block input of ``layers.start``, or the output of block
    ``layers.start - 1``); ``position`` / ``direction`` define the source vector
    ``v[t_0, i_0] = onehot(position - window.start) (x) (P^T direction)``.  If they are
    omitted the source is a unit vector at the last window position along the first
    basis vector (the source can be replaced afterwards, see ``factorized_response``).

    Tensors (in order): source ``v(t_0, i_0)``, then for every block ``l``:
    ``G_l(b_l, t_{l+1}, t_l)`` and ``K_l(b_l, t_l, i_{l+1}, i_l)``.  Open indices:
    ``(t_L, i_L)`` with ``L = layers.stop`` (block output of the last block).

    ``ln_mode``: ``"frozen"`` (spec: ``diag(gamma)/sigma``, window-mean sigma for the
    heads) or ``"exact"`` (full per-position LayerNorm Jacobian incl. centering, for
    heads and MLP; same memory)."""
    layers = list(layers)
    if not layers:
        raise ValueError("layers is empty")
    if any(b - a != 1 for a, b in zip(layers[:-1], layers[1:])):
        raise ValueError("layers must be a contiguous increasing range")
    L = mb.n_layers
    if layers[0] < 0 or layers[-1] >= L:
        raise ValueError(f"layers {layers} out of range 0..{L - 1}")
    if _snapshot_index(site_from) != layers[0]:
        raise ValueError(f"site_from {site_from.key} enters at snapshot {_snapshot_index(site_from)} "
                         f"but layers start at {layers[0]}")
    any_resid = next(iter(cache.resid.values()))
    T = int(any_resid.shape[1])
    w_pos = _window_positions(window, T)
    T_w = len(w_pos)
    P = _to_tensor(basis).float().cpu()
    if P.shape[0] != mb.d_model:
        raise ValueError(f"basis must be [d={mb.d_model}, r], got {tuple(P.shape)}")
    r = int(P.shape[1])
    if position is None:
        position = w_pos[-1]
    if not (w_pos[0] <= position <= w_pos[-1]):
        raise ValueError(f"position {position} must lie inside the window {w_pos[0]}..{w_pos[-1]} "
                         "(the window restriction is exact only then)")
    if direction is None:
        direction = P[:, 0].clone()
    direction = _to_tensor(direction).float().cpu()

    tensors: list[tuple[tuple[str, ...], np.ndarray]] = []
    l0 = layers[0]
    src = torch.zeros(T_w, r)
    src[position - w_pos[0]] = P.T @ direction
    tensors.append(((f"t{l0}", f"i{l0}"), src.numpy().astype(dtype)))
    per_layer: dict[int, dict] = {}
    for l in layers:
        G, K, info = _block_factors(mb, cache, l, w_pos, P, ln_mode=ln_mode)
        tensors.append(((f"b{l}", f"t{l + 1}", f"t{l}"), G.numpy().astype(dtype)))
        tensors.append(((f"b{l}", f"t{l}", f"i{l + 1}", f"i{l}"), K.numpy().astype(dtype)))
        per_layer[l] = info
    L_out = layers[-1] + 1
    output_inds = (f"t{L_out}", f"i{L_out}")
    meta = dict(
        d=mb.d_model, r=r, T_w=T_w, n_heads=mb.n_heads, n_branches=2 + mb.n_heads,
        branches=["skip"] + [f"head{h}" for h in range(mb.n_heads)] + ["mlp"],
        layers=layers, window=(w_pos[0], w_pos[-1] + 1), position=position,
        site_from=site_from.key, basis=P.numpy().astype(np.float32),
        direction=direction.numpy().astype(np.float32), ln_mode=ln_mode,
        output_snapshot=L_out, per_layer=per_layer,
        n_tensors=len(tensors), dtype=np.dtype(dtype).name,
        approximations=[
            "frozen attention patterns (value path only; dA/dq, dA/dk ignored)",
            ("frozen LayerNorm: diag(gamma)/sigma, centering and normalisation projection dropped; "
             "head sigma = window mean" if ln_mode == "frozen" else
             "exact per-position LayerNorm Jacobian"),
            f"residual stream truncated to r={r} of d={mb.d_model} basis directions",
            "position window restriction (exact by causality for positions >= window start)",
        ],
    )
    tn = LatticeTN(tensors=tensors, output_inds=output_inds, meta=meta)
    meta["dims"] = tn.dims
    return tn


# --------------------------------------------------------------------------------------
# 4./5. contraction path and contraction
# --------------------------------------------------------------------------------------
def einsum_expression(tn: LatticeTN) -> tuple[str, dict[str, str]]:
    """opt_einsum expression ``"ab,bc,...->xy"`` for ``tn`` and the index -> symbol map."""
    names: dict[str, str] = {}
    for inds in tn.inds:
        for name in inds:
            if name not in names:
                names[name] = oe.get_symbol(len(names))
    for name in tn.output_inds:
        if name not in names:
            raise ValueError(f"output index {name} not present in the network")
    lhs = ",".join("".join(names[n] for n in inds) for inds in tn.inds)
    rhs = "".join(names[n] for n in tn.output_inds)
    return f"{lhs}->{rhs}", names


def _normalize_path(path) -> list[tuple[int, ...]]:
    return [tuple(int(i) for i in step) for step in path]


def _oe_path_cost(tn: LatticeTN, path) -> float:
    """opt_einsum FLOP count of an explicit path (same convention for every backend)."""
    expr, _ = einsum_expression(tn)
    _, info = oe.contract_path(expr, *tn.arrays, optimize=_normalize_path(path))
    return float(info.opt_cost)


def _opt_einsum_path(tn: LatticeTN, optimize: Optional[str] = None, **kw) -> dict:
    t0 = time.perf_counter()
    expr, _ = einsum_expression(tn)
    if optimize is None:
        optimize = "dp" if tn.n_tensors <= 12 else "greedy"
    path, info = oe.contract_path(expr, *tn.arrays, optimize=optimize)
    path = _normalize_path(path)
    cost = float(info.opt_cost)
    return dict(path=path, cost_log10=math.log10(cost) if cost > 0 else float("-inf"),
                cost_log10_opt_einsum=math.log10(cost) if cost > 0 else float("-inf"),
                runtime_s=time.perf_counter() - t0, backend="opt_einsum",
                optimizer=str(optimize), largest_intermediate_log10=math.log10(max(float(info.largest_intermediate), 1.0)))


def _tnco_path(tn: LatticeTN, **kw) -> dict:
    """Optimise the contraction order with TNCO (simulated annealing, infinite memory).

    ``fuse=0`` and ``decompose_hyper_inds=False`` keep the optimiser's tensor list equal
    to ``tn.tensors`` so that the returned linear path indexes our arrays directly."""
    from tnco.app import Optimizer, Tensor, TensorNetwork      # ImportError -> fallback
    t0 = time.perf_counter()
    tn_obj = TensorNetwork([Tensor(inds=tuple(inds), dims=tuple(int(n) for n in arr.shape))
                            for inds, arr in tn.tensors], output_inds=tuple(tn.output_inds))
    opt = Optimizer(method=kw.get("method", "sa"), n_jobs=int(kw.get("n_jobs", 1)),
                    seed=int(kw.get("seed", 0)), verbose=int(kw.get("verbose", 0)))
    betas = tuple(kw.get("betas", (0.0, 1e4)))
    res = opt.optimize(tn_obj, betas=betas, n_steps=int(kw.get("n_steps", 500)),
                       n_runs=int(kw.get("n_runs", 4)), timeout=kw.get("timeout", None),
                       fuse=0, decompose_hyper_inds=False)
    # `res` is (new_tn, results); results is a list sorted by cost (keep our own order).
    results = res[1] if isinstance(res, tuple) else res
    results = list(results)
    best = sorted(results, key=lambda rr: rr.cost)[0]
    path = _normalize_path(best.path)
    if len(path) != max(tn.n_tensors - 1, 0):
        raise RuntimeError(f"tnco returned a path of {len(path)} steps for {tn.n_tensors} tensors")
    cost = best.cost
    cost_log10 = float(cost.log10()) if hasattr(cost, "log10") else (math.log10(float(cost)) if float(cost) > 0 else float("-inf"))
    oe_cost = _oe_path_cost(tn, path)
    return dict(path=path, cost_log10=cost_log10,
                cost_log10_opt_einsum=math.log10(oe_cost) if oe_cost > 0 else float("-inf"),
                runtime_s=time.perf_counter() - t0, backend="tnco",
                tnco_runtime_s=float(getattr(best, "runtime_s", float("nan"))),
                n_runs=len(results), all_costs_log10=[
                    float(rr.cost.log10()) if hasattr(rr.cost, "log10") else math.log10(float(rr.cost))
                    for rr in results])


def optimize_path(tn: LatticeTN, backend: str = "tnco", **kw) -> dict:
    """Find a contraction path for ``tn``.

    ``backend="tnco"``: simulated-annealing optimiser of TNCO (kwargs ``betas``,
    ``n_steps``, ``n_runs``, ``seed``, ``n_jobs``, ``timeout``); the best of ``n_runs``
    is returned.  On ``ImportError`` or any TNCO failure the opt_einsum optimiser is used
    (``"dp"`` for <= 12 tensors, else ``"greedy"``; override with ``optimize=``).
    ``backend="opt_einsum"`` uses opt_einsum directly.

    Returns ``dict(path, cost_log10, runtime_s, backend, ...)``.  ``cost_log10`` is the
    backend's own cost model (TNCO counts multiply-adds per pairwise contraction,
    opt_einsum counts FLOPs); ``cost_log10_opt_einsum`` evaluates *every* path with the
    opt_einsum FLOP count so that backends can be compared."""
    if backend == "tnco":
        try:
            return _tnco_path(tn, **kw)
        except ImportError as e:
            out = _opt_einsum_path(tn, kw.get("optimize"))
            out["fallback_reason"] = f"tnco not importable: {e!r}"
            return out
        except Exception as e:                        # any tnco failure -> fallback
            out = _opt_einsum_path(tn, kw.get("optimize"))
            out["fallback_reason"] = f"tnco failed: {e!r}"
            return out
    if backend == "opt_einsum":
        return _opt_einsum_path(tn, kw.get("optimize"))
    raise ValueError(f"unknown backend {backend}")


def contract(tn: LatticeTN, path) -> np.ndarray:
    """Contract ``tn`` along the explicit linear ``path`` (list of index pairs) with
    opt_einsum; output axes follow ``tn.output_inds``."""
    expr, _ = einsum_expression(tn)
    if isinstance(path, dict):
        path = path["path"]
    out = oe.contract(expr, *tn.arrays, optimize=_normalize_path(path))
    return np.asarray(out)


def cost_reference(tn: LatticeTN, max_dp_tensors: int = 40) -> dict:
    """opt_einsum reference costs (log10 FLOPs) of the greedy and dynamic-programming
    optimisers, plus the naive single-einsum cost (``naive_cost``)."""
    expr, _ = einsum_expression(tn)
    out: dict[str, Any] = {}
    for name in ("greedy", "dp"):
        if name == "dp" and tn.n_tensors > max_dp_tensors:
            out[f"{name}_cost_log10"] = float("nan")
            out[f"{name}_path"] = None
            continue
        t0 = time.perf_counter()
        try:
            path, info = oe.contract_path(expr, *tn.arrays, optimize=name)
        except Exception:                                  # pragma: no cover
            out[f"{name}_cost_log10"] = float("nan")
            out[f"{name}_path"] = None
            continue
        out[f"{name}_cost_log10"] = math.log10(float(info.opt_cost))
        out[f"{name}_path"] = _normalize_path(path)
        out[f"{name}_runtime_s"] = time.perf_counter() - t0
        out["naive_cost_log10"] = math.log10(float(info.naive_cost))
    return out


# --------------------------------------------------------------------------------------
# 6. factorised response
# --------------------------------------------------------------------------------------
def factorized_response(mb: ModelBundle, cache: CacheOut, site_from: HookSite, position: int,
                        direction, window: slice, basis, layers: range,
                        backend: str = "tnco", ln_mode: str = "frozen",
                        return_info: bool = False, **kw):
    """Linear response at the block output of ``layers[-1]`` (window positions, mapped
    back to the full residual dimension): ``[T_w, d] = (contract(TN)[t, :]) @ P^T``.

    Builds the lattice, optimises the path with ``backend`` and contracts.  With
    ``return_info=True`` also returns ``dict(tn, path_info)``."""
    tn = build_factorized_lattice(mb, cache, site_from, window, basis, layers,
                                  position=position, direction=direction, ln_mode=ln_mode)
    info = optimize_path(tn, backend=backend, **kw)
    y = contract(tn, info["path"])                       # [T_w, r]
    P = tn.meta["basis"]                                 # [d, r]
    resp = np.asarray(y, dtype=np.float64) @ P.T.astype(np.float64)      # [T_w, d]
    resp = resp.astype(np.float32)
    if return_info:
        return resp, dict(tn=tn, path_info=info)
    return resp


# --------------------------------------------------------------------------------------
# 7. comparison
# --------------------------------------------------------------------------------------
def _as_2d(x, name: str) -> np.ndarray:
    if hasattr(x, "D") and not hasattr(x, "shape"):
        raise TypeError(f"{name}: pass the residual differences [T_w, d], not a DamageMap "
                        "(DamageMap only stores relative norms)")
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 3 and x.shape[0] == 1:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"{name} must be [T_w, d], got {x.shape}")
    return x


def compare_responses(measured, linear, factorized, positions: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Per-position comparison of the three response levels (DESIGN §5, D8).

    ``measured [T_w, d]``: residual differences ``h' - h`` of the perturbed run (already
    divided by the dose if the linear response is per unit perturbation);
    ``linear [T_w, d]``: exact first-order response (``linear_response_jvp``);
    ``factorized [T_w, d]``: tensor-network response (``factorized_response``).

    Columns: ``position`` (window-relative or ``positions``), ``measured_norm``,
    ``linear_norm``, ``factorized_norm``, ``nonlinearity = |measured - linear| / |linear|``,
    ``factorization_error = |linear - factorized| / |linear|``, plus the cosines
    ``cos_measured_linear`` and ``cos_linear_factorized``."""
    m = _as_2d(measured, "measured")
    li = _as_2d(linear, "linear")
    fa = _as_2d(factorized, "factorized")
    if not (m.shape == li.shape == fa.shape):
        raise ValueError(f"shape mismatch: {m.shape}, {li.shape}, {fa.shape}")
    T_w = m.shape[0]
    pos = list(range(T_w)) if positions is None else [int(p) for p in positions]
    if len(pos) != T_w:
        raise ValueError("positions must have one entry per row")
    eps = 1e-30
    ln = np.linalg.norm(li, axis=1)
    mn = np.linalg.norm(m, axis=1)
    fn = np.linalg.norm(fa, axis=1)

    def cos(a, b):
        return (a * b).sum(1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), eps)

    df = pd.DataFrame(dict(
        position=pos,
        measured_norm=mn, linear_norm=ln, factorized_norm=fn,
        nonlinearity=np.linalg.norm(m - li, axis=1) / np.maximum(ln, eps),
        factorization_error=np.linalg.norm(li - fa, axis=1) / np.maximum(ln, eps),
        cos_measured_linear=cos(m, li),
        cos_linear_factorized=cos(li, fa),
    ))
    return df
