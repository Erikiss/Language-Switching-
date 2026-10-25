"""Per-head attribution and head-level steering for GPT-NeoX (Pythia) models.

DESIGN §4.7 ("improve the worst head"): the attention output of block ``l`` is

    attn_out = dense(concat_h(z_h)) + b_O = sum_h z_h @ W_O[:, slice_h].T + b_O

where ``z_h [T, dh]`` is the attention-weighted value vector of head ``h`` and ``W_O =
dense.weight [d, H*dh]``.  Everything in this module works on the *input of
``layers[l].attention.dense``* (captured / modified with a forward pre-hook), so that the
per-head decomposition is exact and no attention pattern is needed:

* :func:`head_slices` / :func:`detect_head_layout` - which columns of the dense input
  belong to which head.  The layout is **verified at runtime** (once per model) by
  comparing the dense input of a one-token forward with the per-head value vectors
  ``LN1(h) W_V^h^T + b_V^h`` (for a single position the attention pattern is the identity,
  so ``z_h == v_h`` exactly).  Both the contiguous layout (``x[..., h*dh:(h+1)*dh]``, what
  transformers' ``GPTNeoXAttention`` produces) and an interleaved layout (``x[..., h::H]``)
  are recognised.
* :func:`head_contributions` - ``c_h = z_h @ W_O^h^T`` ``[H, T, d]`` (sum + bias == attention output).
* :func:`head_direct_logit_effect` - lens of each head's contribution on a target logit.
* :func:`ablate_heads` - zero / mean ablation of head slices (context manager).
* :func:`head_ablation_effects`, :func:`worst_head` - ablation table over stimulus
  variants and the "worst head" selection (largest loss on weak families relative to
  stable families).
* :class:`HeadInjector`, :func:`steer_at_head` - steering *at the head output*: a
  residual-space direction is mapped into head space through the least-squares
  pre-image of ``W_O^h`` and dosed relative to the RMS of the head's own contribution.
"""
from __future__ import annotations

import contextlib
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from .model import ModelBundle, _first_tensor, logit_lens, rms, run_with_cache
from .config import HookSite

try:                                           # verified QKV layout helper (tn_lattice)
    from .tn_lattice import head_weights as _tn_head_weights
except Exception:                              # pragma: no cover - opt_einsum missing etc.
    _tn_head_weights = None

__all__ = [
    "head_dim", "head_slices", "detect_head_layout", "dense_bias", "head_output_weight",
    "head_contributions", "verify_head_reconstruction", "head_direct_logit_effect",
    "HeadAblation", "ablate_heads", "head_ablation_effects", "worst_head",
    "HeadInjector", "head_preimage", "steer_at_head",
]

_EPS = 1e-30
_LAYOUTS = ("contiguous", "interleaved")
_LAYOUT_CACHE: dict[int, str] = {}


# --------------------------------------------------------------------------------------
# structure helpers
# --------------------------------------------------------------------------------------
def head_dim(mb: ModelBundle) -> int:
    """Per-head width ``dh = d_model // n_heads`` (read from the model config)."""
    H, d = mb.n_heads, mb.d_model
    if d % H != 0:
        raise ValueError(f"d_model={d} is not divisible by n_heads={H}")
    return d // H


def _attention(mb: ModelBundle, layer: int):
    L = mb.n_layers
    if not (0 <= int(layer) < L):
        raise ValueError(f"layer {layer} out of range 0..{L - 1}")
    return mb.layers[int(layer)].attention


def _dense_input(args, kwargs) -> torch.Tensor:
    if args:
        return args[0]
    return kwargs["input"]


def _set_dense_input(args, kwargs, x):
    if args:
        return (x,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["input"] = x
    return args, kwargs


def _value_weights(mb: ModelBundle, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``W_V [H, dh, d]`` and ``b_V [H, dh]`` of block ``layer`` (QKV rows grouped per head
    as ``[q_h | k_h | v_h]``; the layout verified in ``lsne.tn_lattice``)."""
    if _tn_head_weights is not None:
        hw = _tn_head_weights(mb, layer)
        return hw["W_V"].cpu(), hw["b_V"].cpu()
    att = _attention(mb, layer)                                    # pragma: no cover
    H, d = mb.n_heads, mb.d_model
    dh = d // H
    W = att.query_key_value.weight.detach().float().cpu().view(H, 3, dh, d)
    b = att.query_key_value.bias
    b = b.detach().float().cpu().view(H, 3, dh) if b is not None else torch.zeros(H, 3, dh)
    return W[:, 2], b[:, 2]


def _layout_slices(layout: str, H: int, dh: int) -> list[slice]:
    if layout == "contiguous":
        return [slice(h * dh, (h + 1) * dh) for h in range(H)]
    if layout == "interleaved":
        return [slice(h, None, H) for h in range(H)]
    raise ValueError(f"unknown head layout {layout!r}")


def detect_head_layout(mb: ModelBundle, layer: int = 0, atol: float = 1e-4,
                       force: bool = False) -> str:
    """Verify (once per model, cached) how the heads are laid out in the input of
    ``attention.dense``.

    A one-token forward is run; at a single position every attention pattern is the
    identity, so head ``h``'s value vector ``LN1(h_0) W_V^h^T + b_V^h`` must equal its
    slice of the dense input.  Returns ``"contiguous"`` (``x[..., h*dh:(h+1)*dh]``) or
    ``"interleaved"`` (``x[..., h::H]``); raises ``RuntimeError`` if neither layout
    reproduces the captured input within ``atol``.
    """
    key = id(mb.model)
    if not force and key in _LAYOUT_CACHE:
        return _LAYOUT_CACHE[key]
    H, d = mb.n_heads, mb.d_model
    dh = head_dim(mb)
    lyr = mb.layers[int(layer)]
    store: dict[str, torch.Tensor] = {}

    def ln_hook(module, args, out):
        store["ln1"] = _first_tensor(out).detach()

    def dense_pre(module, args, kwargs):
        store["x"] = _dense_input(args, kwargs).detach()

    h1 = lyr.input_layernorm.register_forward_hook(ln_hook)
    h2 = lyr.attention.dense.register_forward_pre_hook(dense_pre, with_kwargs=True)
    try:
        tok_id = int(mb.tokenizer.encode("the", add_special_tokens=False)[0]) if hasattr(mb.tokenizer, "encode") else 0
        ids = torch.tensor([[tok_id]], dtype=torch.long, device=mb.device)
        with torch.no_grad():
            mb.model(input_ids=ids, use_cache=False)
    finally:
        h1.remove()
        h2.remove()
    x = store["x"][0, 0].float().cpu()                              # [H*dh]
    ln1 = store["ln1"][0, 0].float().cpu()                          # [d]
    W_V, b_V = _value_weights(mb, int(layer))
    v = torch.stack([ln1 @ W_V[h].T + b_V[h] for h in range(H)], 0)   # [H, dh]
    errs = {}
    for layout in _LAYOUTS:
        sl = _layout_slices(layout, H, dh)
        errs[layout] = max(float((x[s] - v[h]).abs().max()) for h, s in enumerate(sl))
    best = min(errs, key=errs.get)
    scale = float(v.abs().max().clamp_min(1.0))
    if errs[best] > atol * scale:
        raise RuntimeError(f"no known head layout reproduces the dense input (errors {errs})")
    if H > 1 and dh > 1 and abs(errs["contiguous"] - errs["interleaved"]) <= atol * scale:
        warnings.warn("head layouts are indistinguishable on this model; assuming contiguous")
        best = "contiguous"
    _LAYOUT_CACHE[key] = best
    return best


def head_slices(mb: ModelBundle, layer: int) -> list[slice]:
    """Column slices of the ``dense`` (W_O) input that belong to each head of block
    ``layer`` (also valid for ``dense.weight[:, slice]``).  The layout is verified with
    :func:`detect_head_layout` on first use."""
    _attention(mb, layer)                       # range check
    layout = detect_head_layout(mb, 0)
    return _layout_slices(layout, mb.n_heads, head_dim(mb))


def dense_bias(mb: ModelBundle, layer: int) -> torch.Tensor:
    """``b_O [d]`` of block ``layer`` (zeros if the dense projection has no bias)."""
    att = _attention(mb, layer)
    b = att.dense.bias
    if b is None:
        return torch.zeros(mb.d_model)
    return b.detach().float().cpu()


def head_output_weight(mb: ModelBundle, layer: int, head: int) -> torch.Tensor:
    """``W_O^h = dense.weight[:, slice_h]`` as ``[d, dh]`` (float32 CPU)."""
    sl = head_slices(mb, layer)
    if not (0 <= int(head) < len(sl)):
        raise ValueError(f"head {head} out of range 0..{len(sl) - 1}")
    W = _attention(mb, layer).dense.weight.detach().float().cpu()
    return W[:, sl[int(head)]].contiguous()


# --------------------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------------------
@contextlib.contextmanager
def _capture_dense(mb: ModelBundle, layer: int, store: dict, with_output: bool = False):
    """Record the dense input ``x [B,T,H*dh]`` (and optionally the attention module
    output ``[B,T,d]``) of block ``layer``."""
    att = _attention(mb, layer)
    handles = []

    def pre(module, args, kwargs):
        store["x"] = _dense_input(args, kwargs).detach()

    handles.append(att.dense.register_forward_pre_hook(pre, with_kwargs=True))
    if with_output:
        def post(module, args, out):
            store["attn_out"] = _first_tensor(out).detach()
        handles.append(att.register_forward_hook(post))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


def _forward(mb: ModelBundle, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return mb.model(input_ids=input_ids.to(mb.device), use_cache=False).logits.float()


def head_contributions(mb: ModelBundle, input_ids: torch.Tensor, layer: int,
                       return_output: bool = False):
    """Per-head direct contribution of block ``layer`` to the residual stream,
    ``c_h = x[..., slice_h] @ W_O[:, slice_h].T`` for every position: ``[H, T, d]``
    (float32 CPU; ``[H, B, T, d]`` for a batch of more than one sequence).

    ``sum_h c_h + dense_bias == attention output`` (see
    :func:`verify_head_reconstruction`).  With ``return_output=True`` the captured
    attention output ``[B, T, d]`` is returned as well."""
    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids[None]
    store: dict = {}
    with _capture_dense(mb, layer, store, with_output=True):
        _forward(mb, input_ids)
    x = store["x"].float().cpu()                                   # [B, T, H*dh]
    W = _attention(mb, layer).dense.weight.detach().float().cpu()  # [d, H*dh]
    sl = head_slices(mb, layer)
    contrib = torch.stack([x[..., s] @ W[:, s].T for s in sl], 0)  # [H, B, T, d]
    if x.shape[0] == 1:
        contrib = contrib[:, 0]
    if return_output:
        return contrib, store["attn_out"].float().cpu()
    return contrib


def verify_head_reconstruction(mb: ModelBundle, input_ids: torch.Tensor, layer: int) -> dict:
    """Check ``sum_h c_h + b_O == attention output`` for ``input_ids``; returns
    ``dict(max_abs_err, rel_err, layout, n_heads, head_dim)``."""
    c, out = head_contributions(mb, input_ids, layer, return_output=True)
    if c.dim() == 3:
        c = c[:, None]
    recon = c.sum(0) + dense_bias(mb, layer)
    err = (recon - out).abs().max()
    return dict(max_abs_err=float(err), rel_err=float(err / out.abs().max().clamp_min(_EPS)),
                layout=detect_head_layout(mb, 0), n_heads=mb.n_heads, head_dim=head_dim(mb))


# --------------------------------------------------------------------------------------
# direct logit effect
# --------------------------------------------------------------------------------------
def _residual_at(mb: ModelBundle, input_ids: torch.Tensor, site: HookSite, position: int) -> torch.Tensor:
    co = run_with_cache(mb, input_ids, [site])
    return co.resid[site.key][0, position].detach().float().cpu()


def head_direct_logit_effect(mb: ModelBundle, input_ids: torch.Tensor, layer: int,
                             target_id: Optional[int], position: int = -1,
                             h_ref: str = "final") -> torch.Tensor:
    """Direct effect of every head of block ``layer`` on the target logit at ``position``:
    ``logit_lens(h + c_h)[target] - logit_lens(h)[target]`` with ``c_h`` the head's
    contribution (:func:`head_contributions`) and ``h`` the reference residual at which
    the lens (final LayerNorm + unembedding) is linearised.

    ``h_ref="final"``: ``h`` is the final residual (output of the last block, where the
    lens equals the true logits); ``h_ref="layer"``: ``h`` is the input of block
    ``layer`` (the residual the head is written into).  Returns ``Tensor[H]`` (float32
    CPU); with ``target_id=None`` the full lens difference ``[H, V]`` is returned."""
    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids[None]
    T = int(input_ids.shape[1])
    pos = int(position) % T
    c = head_contributions(mb, input_ids, layer)                 # [H, T, d]
    if c.dim() == 4:
        c = c[:, 0]
    c_pos = c[:, pos]                                              # [H, d]
    if h_ref == "final":
        site = HookSite(mb.n_layers - 1, "block_output")
    elif h_ref == "layer":
        site = HookSite(int(layer), "block_input")
    else:
        raise ValueError(f"h_ref must be 'final' or 'layer', got {h_ref!r}")
    h = _residual_at(mb, input_ids, site, pos)                     # [d]
    lens_h = logit_lens(mb, h.to(mb.device))[None]                 # [1, V]
    lens_hc = logit_lens(mb, (h[None] + c_pos).to(mb.device))     # [H, V]
    delta = (lens_hc - lens_h).float().cpu()
    if target_id is None:
        return delta
    return delta[:, int(target_id)].contiguous()


# --------------------------------------------------------------------------------------
# ablation
# --------------------------------------------------------------------------------------
class HeadAblation:
    """Context manager that zero- or mean-ablates head slices at the ``dense`` input of
    block ``layer``.

    ``mode="zero"``: the slices are set to 0.  ``mode="mean"``: every position of a head
    slice is replaced by the mean of that slice over the positions of the same forward
    pass (per sequence), or by the corresponding slice of ``mean_source`` (``Tensor[H*dh]``,
    e.g. a corpus mean of the dense input, or ``Tensor[H, dh]``).  ``positions`` restricts
    the ablation to those positions (default: all).  ``n_calls`` counts hook calls and
    ``last_mean`` keeps the per-head mean vectors used in the last call."""

    def __init__(self, mb: ModelBundle, layer: int, heads: Sequence[int], mode: str = "mean",
                 mean_source: Optional[torch.Tensor] = None, positions=None):
        if mode not in ("zero", "mean"):
            raise ValueError(f"mode must be 'zero' or 'mean', got {mode!r}")
        self.mb, self.layer, self.mode = mb, int(layer), mode
        H = mb.n_heads
        self.heads = sorted({int(h) for h in heads})
        if any(h < 0 or h >= H for h in self.heads):
            raise ValueError(f"heads {self.heads} out of range 0..{H - 1}")
        self.slices = head_slices(mb, self.layer)
        self.positions = None if positions is None else [int(p) for p in positions]
        self.mean_source = None
        if mean_source is not None:
            ms = torch.as_tensor(mean_source).detach().float().cpu().reshape(-1)
            if ms.numel() != H * head_dim(mb):
                raise ValueError(f"mean_source must have {H * head_dim(mb)} entries, got {ms.numel()}")
            self.mean_source = ms
        self._handle = None
        self.n_calls = 0
        self.last_mean: dict[int, torch.Tensor] = {}

    def _edit(self, x: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        x_new = x.clone()
        if self.positions is None:
            pos, n_pos = slice(None), x.shape[1]
        else:
            pos, n_pos = torch.as_tensor(self.positions, device=x.device), len(self.positions)
        for h in self.heads:
            s = self.slices[h]
            if self.mode == "zero":
                x_new[:, pos, s] = 0.0
            else:
                if self.mean_source is not None:
                    m = self.mean_source[s].to(x.device, x.dtype)[None, None, :]   # [1, 1, dh]
                else:
                    m = x[:, :, s].mean(dim=1, keepdim=True)                       # [B, 1, dh]
                x_new[:, pos, s] = m.expand(x.shape[0], n_pos, -1)
                self.last_mean[h] = m.detach().float().cpu()
        return x_new

    def __enter__(self):
        att = _attention(self.mb, self.layer)

        def pre(module, args, kwargs):
            x = _dense_input(args, kwargs)
            return _set_dense_input(args, kwargs, self._edit(x))

        self._handle = att.dense.register_forward_pre_hook(pre, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def ablate_heads(mb: ModelBundle, layer: int, heads: Sequence[int], mode: str = "mean",
                 mean_source: Optional[torch.Tensor] = None, positions=None) -> HeadAblation:
    """Context manager: zero- or mean-ablate ``heads`` of block ``layer`` at the dense
    input (see :class:`HeadAblation`).  ``heads="all"`` ablates every head."""
    if isinstance(heads, str) and heads == "all":
        heads = list(range(mb.n_heads))
    return HeadAblation(mb, layer, heads, mode=mode, mean_source=mean_source, positions=positions)


def _vattr(v, name, default=None):
    if isinstance(v, dict):
        return v.get(name, default)
    return getattr(v, name, default)


def _variant_ids(v) -> torch.Tensor:
    ids = _vattr(v, "ids")
    if ids is None:
        raise ValueError("variant has no 'ids'")
    if isinstance(ids, torch.Tensor):
        return ids.detach().long().reshape(1, -1)
    return torch.as_tensor(list(ids), dtype=torch.long).reshape(1, -1)


def head_ablation_effects(mb: ModelBundle, variants: Sequence, layer_grid: Sequence[int],
                          target_id_attr: str = "target_id", mode: str = "mean",
                          limit: Optional[int] = None, mean_source: Optional[dict] = None) -> pd.DataFrame:
    """Ablation effect of every head in ``layer_grid`` on the target log-probability of
    every variant (objects / dicts with ``ids``, ``<target_id_attr>`` and optionally
    ``family_id, formulation, K, C``).

    Columns: ``variant, family_id, formulation, K, C, layer, head, logp_base,
    logp_ablated, delta`` with ``delta = logp_ablated - logp_base`` (negative = the head
    was useful for the target).  ``mean_source`` may map ``layer -> Tensor[H*dh]``.
    ``limit`` restricts the number of variants (smoke tests)."""
    rows: list[dict] = []
    vs = list(variants)[: (int(limit) if limit is not None else None)]
    H = mb.n_heads
    for vi, v in enumerate(vs):
        ids = _variant_ids(v)
        tid = int(_vattr(v, target_id_attr))
        base = float(torch.log_softmax(_forward(mb, ids)[0, -1], -1)[tid])
        meta = dict(variant=vi, family_id=_vattr(v, "family_id"), formulation=_vattr(v, "formulation"),
                    K=_vattr(v, "K"), C=_vattr(v, "C"))
        for layer in layer_grid:
            ms = mean_source.get(int(layer)) if isinstance(mean_source, dict) else None
            for h in range(H):
                with ablate_heads(mb, int(layer), [h], mode=mode, mean_source=ms):
                    lp = float(torch.log_softmax(_forward(mb, ids)[0, -1], -1)[tid])
                rows.append(dict(meta, layer=int(layer), head=h, logp_base=base, logp_ablated=lp,
                                 delta=lp - base))
    cols = ["variant", "family_id", "formulation", "K", "C", "layer", "head", "logp_base",
            "logp_ablated", "delta"]
    return pd.DataFrame(rows, columns=cols)


def worst_head(effects: pd.DataFrame, families_weak: Sequence[str],
               families_stable: Sequence[str]) -> dict:
    """"Worst head" of an ablation table (:func:`head_ablation_effects`).

    Per ``(layer, head)``: ``mean_delta_weak`` (mean ``delta`` over the weak families),
    ``mean_delta_stable`` and ``score = mean_delta_weak - mean_delta_stable``.  Because
    ``delta = logp_ablated - logp_base``, the head whose ablation costs most on the weak
    families relative to the stable ones has the **smallest** (most negative) score; it
    is returned as ``dict(layer, head, score, loss=-score, table)`` (ties: lowest layer,
    then lowest head).  Families missing from the table contribute nothing; an empty
    family set gives a mean of 0."""
    weak = {str(f) for f in families_weak}
    stable = {str(f) for f in families_stable}
    df = effects.copy()
    df["family_id"] = df["family_id"].astype(str)
    rows = []
    for (layer, head), g in df.groupby(["layer", "head"], sort=True):
        gw = g[g["family_id"].isin(weak)]
        gs = g[g["family_id"].isin(stable)]
        mw = float(gw["delta"].mean()) if len(gw) else 0.0
        ms = float(gs["delta"].mean()) if len(gs) else 0.0
        rows.append(dict(layer=int(layer), head=int(head), mean_delta_weak=mw, mean_delta_stable=ms,
                         n_weak=int(len(gw)), n_stable=int(len(gs)), score=mw - ms))
    table = pd.DataFrame(rows, columns=["layer", "head", "mean_delta_weak", "mean_delta_stable",
                                        "n_weak", "n_stable", "score"])
    if table.empty:
        return dict(layer=None, head=None, score=float("nan"), loss=float("nan"), table=table)
    table = table.sort_values(["score", "layer", "head"], kind="mergesort").reset_index(drop=True)
    best = table.iloc[0]
    return dict(layer=int(best["layer"]), head=int(best["head"]), score=float(best["score"]),
                loss=-float(best["score"]), table=table)


# --------------------------------------------------------------------------------------
# head-level steering
# --------------------------------------------------------------------------------------
def head_preimage(mb: ModelBundle, layer: int, head: int, delta_resid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Least-squares pre-image of a residual-space vector through ``W_O^h``:
    ``u = argmin_u ||W_O^h u - delta||`` (``torch.linalg.lstsq``).  Returns
    ``(u [dh], realised = W_O^h u [d])``; ``realised`` is the projection of ``delta`` onto
    the (at most ``dh``-dimensional) column space of ``W_O^h``."""
    W = head_output_weight(mb, layer, head)                        # [d, dh]
    delta = torch.as_tensor(delta_resid).detach().float().cpu().reshape(-1)
    if delta.numel() != W.shape[0]:
        raise ValueError(f"delta has {delta.numel()} entries, expected d={W.shape[0]}")
    u = torch.linalg.lstsq(W.double(), delta.double()[:, None]).solution[:, 0].float()
    return u, W @ u


class HeadInjector:
    """Context manager adding a delta to head ``head``'s slice of the ``dense`` input of
    block ``layer`` at ``positions`` (default: all positions).

    ``delta`` is ``Tensor[dh]`` in head space, or ``Tensor[d]`` in residual space, which is
    mapped into head space through the least-squares pre-image of ``W_O^h``
    (:func:`head_preimage`); ``space="auto"`` decides by length (ambiguous when
    ``dh == d``: pass ``space`` explicitly).  ``audit`` holds ``requested_norm``,
    ``realised_norm`` and ``rel_err = ||W_O u - delta|| / ||delta||`` (the part of a
    residual-space delta that the head cannot write), plus ``written_norm`` after the
    first call."""

    def __init__(self, mb: ModelBundle, layer: int, head: int, delta: torch.Tensor,
                 positions=None, space: str = "auto"):
        self.mb, self.layer, self.head = mb, int(layer), int(head)
        dh, d = head_dim(mb), mb.d_model
        delta = torch.as_tensor(delta).detach().float().cpu().reshape(-1)
        if space == "auto":
            if delta.numel() == dh and dh != d:
                space = "head"
            elif delta.numel() == d and dh != d:
                space = "resid"
            elif delta.numel() == d:
                raise ValueError("dh == d: pass space='head' or space='resid' explicitly")
            else:
                raise ValueError(f"delta has {delta.numel()} entries; expected dh={dh} or d={d}")
        if space == "head":
            if delta.numel() != dh:
                raise ValueError(f"head-space delta must have dh={dh} entries")
            u = delta
            realised = head_output_weight(mb, self.layer, self.head) @ u
            requested = realised
        elif space == "resid":
            u, realised = head_preimage(mb, self.layer, self.head, delta)
            requested = delta
        else:
            raise ValueError(f"space must be 'auto', 'head' or 'resid', got {space!r}")
        self.space = space
        self.u = u
        self.realised = realised
        self.positions = None if positions is None else [int(p) for p in positions]
        self.slice = head_slices(mb, self.layer)[self.head]
        self.audit = dict(space=space, requested_norm=float(requested.norm()),
                          realised_norm=float(realised.norm()),
                          rel_err=float((realised - requested).norm() / requested.norm().clamp_min(_EPS)),
                          cos=float(realised @ requested / (realised.norm() * requested.norm()).clamp_min(_EPS)))
        self._handle = None
        self.n_calls = 0

    def _edit(self, x: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        x_new = x.clone()
        u = self.u.to(x.device, x.dtype)
        if self.positions is None:
            x_new[:, :, self.slice] = x_new[:, :, self.slice] + u
        else:
            pos = torch.as_tensor(self.positions, device=x.device)
            x_new[:, pos, self.slice] = x_new[:, pos, self.slice] + u
        if "written_norm" not in self.audit:
            self.audit["written_norm"] = float((x_new.float() - x.float()).norm())
        return x_new

    def __enter__(self):
        att = _attention(self.mb, self.layer)

        def pre(module, args, kwargs):
            x = _dense_input(args, kwargs)
            return _set_dense_input(args, kwargs, self._edit(x))

        self._handle = att.dense.register_forward_pre_hook(pre, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def _as_vec(direction) -> torch.Tensor:
    vec = getattr(direction, "vec", direction)
    return torch.as_tensor(vec).detach().float().cpu().reshape(-1)


def steer_at_head(mb: ModelBundle, input_ids: torch.Tensor, layer: int, head: int,
                  direction, eps: float, position: int = -1, return_info: bool = False):
    """Last-position logits ``[V]`` when the residual-space ``direction`` (``Tensor[d]`` or
    a ``Direction``) is applied *through head ``head``* of block ``layer`` at ``position``.

    The direction is mapped into head space by the least-squares pre-image through
    ``W_O^h`` and rescaled so that the realised residual-space delta ``W_O^h u`` has
    RMS ``eps * RMS(c_h[position])`` where ``c_h`` is the head's own contribution at that
    position (dose relative to the head, not to the residual).  Only ``position`` is
    modified, so earlier positions' logits are unchanged (causality).  With
    ``return_info=True`` returns ``(logits_last, info)`` where ``info`` has
    ``logits_full [T, V]``, ``u``, ``realised`` (the residual delta actually written),
    ``cos_realised_direction``, ``rms_head``, ``rms_delta`` and ``eps``."""
    input_ids = torch.as_tensor(input_ids, dtype=torch.long)
    if input_ids.dim() == 1:
        input_ids = input_ids[None]
    T = int(input_ids.shape[1])
    pos = int(position) % T
    vec = _as_vec(direction)
    if vec.numel() != mb.d_model:
        raise ValueError(f"direction has {vec.numel()} entries, expected d={mb.d_model}")
    c = head_contributions(mb, input_ids, layer)
    if c.dim() == 4:
        c = c[:, 0]
    rms_head = float(rms(c[int(head), pos]))
    unit_dir = vec / rms(vec).clamp_min(_EPS)                      # RMS-unit direction
    u0, r0 = head_preimage(mb, layer, head, unit_dir)
    rms_r0 = float(rms(r0))
    if rms_r0 <= 0.0 or not np.isfinite(rms_r0):
        raise ValueError("direction has no component in the column space of W_O^h")
    target_rms = float(eps) * rms_head
    u = u0 * (target_rms / max(rms_r0, _EPS))
    realised = r0 * (target_rms / max(rms_r0, _EPS))
    with HeadInjector(mb, layer, head, u, positions=[pos], space="head"):
        logits = _forward(mb, input_ids)[0].cpu()                  # [T, V]
    last = logits[-1].contiguous()
    if not return_info:
        return last
    info = dict(logits_full=logits, u=u, realised=realised, rms_head=rms_head,
                rms_delta=float(rms(realised)), eps=float(eps), position=pos,
                cos_realised_direction=float(realised @ vec / (realised.norm() * vec.norm()).clamp_min(_EPS)),
                preimage_rel_err=float((r0 - unit_dir).norm() / unit_dir.norm().clamp_min(_EPS)))
    return last, info
