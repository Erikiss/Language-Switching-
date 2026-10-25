"""Cross-model activation transplant (DESIGN section 4.9): a pluggable *source model*
(BlockGPT-like trace model, or an HF causal LM as stand-in), a linear *alignment*
between the residual spaces of source and target, and two injection routes into the
target (Pythia):

* **Injection A, residual substitution** (:func:`substitute_residual`): the target's
  residual at ``(site, positions)`` is *replaced* by transferred states
  (``Injector(mode="replace")``, no addition).
* **Injection B, KV-cache planting** (:func:`plant_kv_prefix`, :func:`plant_kv_cache`):
  a virtual prefix of ``n`` placeholder tokens whose block inputs are overwritten at
  every chosen layer by transferred states; the prefix's keys/values at those layers are
  thus computed from the transplanted states (rotary at the prefix positions included),
  the real tokens see them through attention and their own residual stream is untouched.
* **Injection C** (:func:`transplant_experiment`, mode ``"both"``): A on the real
  positions plus B in the prefix.

Alignment (:func:`fit_alignment`): ridge regression or orthogonal Procrustes after PCA
on *paired* states (the same item in both models), with R^2 on hold-out pairs and a
permutation null (hold-out targets permuted); :func:`layer_alignments` fits one map per
target layer (what injection B needs to plant *all* layers).  The functional transfer
test (:func:`functional_transfer_test`) validates the machinery with a known feature, the
language axis: the axis found in the *source* model, transferred into the target, must
move the German fraction of steered continuations in the same direction as the axis
found natively in the target, and it must do so *better than the null* (the same map
refitted on permuted pairs).  With the same model as source and target the alignment is
a symmetric projector and the test is a plumbing smoke test only; the informative
stand-in is :class:`RotatedSourceModel` (a known isometry of the target's states that the
alignment has to undo, which also exercises the ``W`` vs. ``W^T`` convention) or a
different source layer than the target site.

Conventions: states are row vectors, ``H_T = H_S @ W + b`` with ``W [d_S, d_T]``;
a direction ``v_S`` maps to ``W^T v_S`` (no bias, unit-normalised).  "Layer ``l``" of a
residual snapshot means the block input of block ``l`` for ``0 <= l < L`` and the output
of the last block for ``l == L`` (the :func:`lsne.model.all_block_inputs` convention).
Nothing here hard-codes ``d_model``/``n_layers``; everything is read from the models.
"""
from __future__ import annotations

import contextlib
import math
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np
import pandas as pd
import torch

from .config import HookSite, ModelSpec
from .directions import cosine, diff_of_means, random_directions, unit
from .generation import block_fraction, language_trajectory
from .model import Injector, ModelBundle, generate_steered, load_model, next_token_stats, rms, run_with_cache

_EPS = 1e-12


# ======================================================================================
# source adapters
# ======================================================================================
@runtime_checkable
class SourceModel(Protocol):
    """Pluggable source adapter (BlockGPT-like trace model or any causal LM).

    ``encode(item)`` turns an item (a text, a transaction trace, a token list, ...) into
    ``LongTensor[1, T]``; ``residuals(ids, layers)`` returns the residual snapshot per
    requested layer as ``Tensor[T, d_S]`` (block input of layer ``l`` for ``l < n_layers``
    and the last block's output for ``l == n_layers``).
    """
    name: str
    d_model: int
    n_layers: int

    def encode(self, item) -> torch.Tensor: ...

    def residuals(self, ids: torch.Tensor, layers: Optional[Sequence[int]] = None) -> dict[int, torch.Tensor]: ...


def _as_ids(item, device=None) -> torch.Tensor:
    """A token list / 1-D or 2-D tensor -> ``LongTensor[1, T]``."""
    ids = torch.as_tensor(item, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids[None]
    if ids.dim() != 2 or ids.shape[0] != 1:
        raise ValueError(f"ids must be [1,T] (or [T]), got {tuple(ids.shape)}")
    return ids.to(device) if device is not None else ids


def _snapshot_site(n_layers: int, layer: int) -> HookSite:
    """Residual snapshot ``layer`` -> HookSite (block input, or last block output for L)."""
    if 0 <= layer < n_layers:
        return HookSite(layer, "block_input")
    if layer == n_layers:
        return HookSite(n_layers - 1, "block_output")
    raise ValueError(f"layer {layer} outside 0..{n_layers}")


def _snapshot_layer(site: HookSite, n_layers: int) -> int:
    """Inverse of :func:`_snapshot_site` for residual sites."""
    if site.site == "block_input":
        return int(site.layer)
    if site.site == "block_output":
        return int(site.layer) + 1
    raise ValueError(f"{site.key} is not a residual-stream snapshot (block_input/block_output)")


def target_residuals(mb: ModelBundle, ids: torch.Tensor, layers: Optional[Sequence[int]] = None) -> dict[int, torch.Tensor]:
    """Residual snapshots of the target bundle, ``{layer: Tensor[T, d_T]}`` (float32, CPU),
    same layer convention as :class:`SourceModel`."""
    L = mb.n_layers
    layers = list(range(L + 1)) if layers is None else [int(l) for l in layers]
    sites = [_snapshot_site(L, l) for l in layers]
    co = run_with_cache(mb, _as_ids(ids, mb.device), sites)
    return {l: co.resid[s.key][0].detach().float().cpu() for l, s in zip(layers, sites)}


class HFSourceModel:
    """Source adapter around :func:`lsne.model.load_model` (any GPT-NeoX / HF causal LM
    whose blocks expose their inputs; works for the Pythia family as stand-ins).

    ``HFSourceModel(spec)`` loads the model; :meth:`from_bundle` wraps an already
    loaded :class:`ModelBundle` (used when source and target are the same model).
    """

    def __init__(self, spec: Optional[ModelSpec] = None, bundle: Optional[ModelBundle] = None):
        if bundle is None:
            if spec is None:
                raise ValueError("HFSourceModel needs a ModelSpec or a bundle")
            bundle = load_model(spec)
        self.mb = bundle
        self.spec = bundle.spec
        self.name = f"{self.spec.name}@{self.spec.revision}"
        self.d_model = int(bundle.d_model)
        self.n_layers = int(bundle.n_layers)

    @classmethod
    def from_bundle(cls, mb: ModelBundle) -> "HFSourceModel":
        return cls(spec=mb.spec, bundle=mb)

    @property
    def tokenizer(self):
        return self.mb.tokenizer

    def encode(self, item) -> torch.Tensor:
        if isinstance(item, str):
            return self.mb.encode(item)
        return _as_ids(item, self.mb.device)

    def residuals(self, ids: torch.Tensor, layers: Optional[Sequence[int]] = None) -> dict[int, torch.Tensor]:
        return target_residuals(self.mb, ids, layers)

    def __repr__(self) -> str:
        return f"HFSourceModel({self.name}, d={self.d_model}, L={self.n_layers})"


class TraceSourceStub:
    """Deterministic random-weight stand-in for a BlockGPT-like trace model.

    It documents the :class:`SourceModel` interface for interface tests and nothing
    more: **not a trained model**.  Items are texts (words hashed to token ids), token
    lists or tensors.  The residual stack is a token embedding followed by ``n_layers``
    fixed random, causal layer maps (``h <- h + 0.5 tanh(h A_l) + 0.25 cummean(h) B_l``),
    so states depend on the left context the way a causal trace model's would.
    """

    def __init__(self, d_model: int = 64, n_layers: int = 4, vocab: int = 1024, seed: int = 0):
        self.d_model, self.n_layers, self.vocab, self.seed = int(d_model), int(n_layers), int(vocab), int(seed)
        self.name = f"trace-stub(d={self.d_model},L={self.n_layers},vocab={self.vocab},seed={self.seed})"
        g = torch.Generator().manual_seed(self.seed)
        self.embed = torch.randn(self.vocab, self.d_model, generator=g)
        self.maps: list[tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(self.n_layers):
            A = torch.randn(self.d_model, self.d_model, generator=g) / math.sqrt(self.d_model)
            B = torch.randn(self.d_model, self.d_model, generator=g) / math.sqrt(self.d_model)
            self.maps.append((A, B))

    def encode(self, item) -> torch.Tensor:
        if isinstance(item, str):
            words = item.split() or [""]
            ids = [zlib.crc32(w.encode("utf-8")) % self.vocab for w in words]
            return torch.tensor([ids], dtype=torch.long)
        return _as_ids(item)

    def residuals(self, ids: torch.Tensor, layers: Optional[Sequence[int]] = None) -> dict[int, torch.Tensor]:
        ids = _as_ids(ids)
        if int(ids.max()) >= self.vocab or int(ids.min()) < 0:
            raise ValueError("token id outside the stub vocabulary")
        h = self.embed[ids[0]]                                  # [T, d]
        states = {0: h}
        T = h.shape[0]
        denom = torch.arange(1, T + 1, dtype=h.dtype)[:, None]
        for l, (A, B) in enumerate(self.maps):
            ctx = h.cumsum(0) / denom                            # causal running mean
            h = h + 0.5 * torch.tanh(h @ A) + 0.25 * (ctx @ B)
            states[l + 1] = h
        layers = list(range(self.n_layers + 1)) if layers is None else [int(l) for l in layers]
        out = {}
        for l in layers:
            if l not in states:
                raise ValueError(f"layer {l} outside 0..{self.n_layers}")
            out[l] = states[l].clone()
        return out

    def __repr__(self) -> str:
        return f"TraceSourceStub({self.name})"


class RotatedSourceModel:
    """Validation stand-in: the residuals of ``base`` seen through a *known* affine
    isometry, ``scale * H @ Q + shift`` with a seeded orthogonal ``Q [d, d]``.

    With the target as ``base`` the alignment ``H_T ~ H_S W + b`` has to recover
    ``W = Q^T / scale`` on the span of the paired states, so the self-consistency check
    is no longer tautological: an identity or transposed ``W`` leaves the transferred
    direction rotated (cosine ~ 0 with the native axis), a permuted pairing gives no
    R^2.  Pass ``Q`` explicitly to use another orthogonal matrix.
    """

    def __init__(self, base: SourceModel, seed: int = 0, scale: float = 1.0, shift: float = 0.0,
                 Q: Optional[torch.Tensor] = None):
        self.base = base
        self.d_model, self.n_layers = int(base.d_model), int(base.n_layers)
        if Q is None:
            g = torch.Generator().manual_seed(int(seed))
            Q, _ = torch.linalg.qr(torch.randn(self.d_model, self.d_model, generator=g))
        Q = torch.as_tensor(Q).detach().float().cpu()
        if Q.shape != (self.d_model, self.d_model) or not torch.allclose(Q.T @ Q, torch.eye(self.d_model), atol=1e-4):
            raise ValueError(f"Q must be orthogonal [{self.d_model}, {self.d_model}]")
        self.Q, self.seed, self.scale, self.shift = Q, int(seed), float(scale), float(shift)
        if self.scale == 0.0:
            raise ValueError("scale must be non-zero")
        self.name = f"rotated({base.name}, seed={self.seed}, scale={self.scale}, shift={self.shift})"

    @property
    def W_expected(self) -> torch.Tensor:
        """The map ``[d, d]`` that undoes the isometry (``H_base = H_rot @ W_expected + b``)."""
        return self.Q.T / self.scale

    def encode(self, item) -> torch.Tensor:
        return self.base.encode(item)

    def residuals(self, ids: torch.Tensor, layers: Optional[Sequence[int]] = None) -> dict[int, torch.Tensor]:
        out = self.base.residuals(ids, layers)
        return {l: self.scale * (torch.as_tensor(h).detach().float().cpu() @ self.Q) + self.shift for l, h in out.items()}

    def __repr__(self) -> str:
        return f"RotatedSourceModel({self.name})"


# ======================================================================================
# alignment
# ======================================================================================
@dataclass
class Alignment:
    """Linear map ``H_T ~ H_S @ W + b`` between residual spaces (``W [d_S, d_T]``)."""
    W: torch.Tensor
    b: torch.Tensor
    method: str
    rank: Optional[int]
    layer_src: int
    layer_tgt: int
    r2_train: float
    r2_holdout: float
    r2_permuted: float
    n_pairs: int
    meta: dict = field(default_factory=dict)

    @property
    def d_src(self) -> int:
        return int(self.W.shape[0])

    @property
    def d_tgt(self) -> int:
        return int(self.W.shape[1])

    @property
    def gap(self) -> float:
        return float(self.r2_holdout - self.r2_permuted)

    def __repr__(self) -> str:
        return (f"Alignment({self.method}, {self.d_src}->{self.d_tgt}, L{self.layer_src}->L{self.layer_tgt}, "
                f"rank={self.rank}, r2 train={self.r2_train:.3f} holdout={self.r2_holdout:.3f} "
                f"permuted={self.r2_permuted:.3f}, n={self.n_pairs})")


def _np64(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _r2(Y: np.ndarray, Yhat: np.ndarray) -> float:
    """``1 - ||Y - Yhat||^2 / ||Y - mean(Y)||^2`` (NaN when Y has no variance)."""
    ss_res = float(((Y - Yhat) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0, keepdims=True)) ** 2).sum())
    if ss_tot <= _EPS:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def _derangement(n: int, rng: np.random.Generator) -> np.ndarray:
    """A seeded permutation of ``range(n)`` without fixed points (n >= 2)."""
    for _ in range(20):
        p = rng.permutation(n)
        if not np.any(p == np.arange(n)):
            return p
    return np.roll(np.arange(n), 1 + int(rng.integers(max(n - 1, 1))))


def fit_alignment(acts_src, acts_tgt, method: str = "ridge", rank: Optional[int] = None,
                  alpha: float = 1.0, holdout: float = 0.2, seed: int = 0,
                  layer_src: int = -1, layer_tgt: int = -1) -> Alignment:
    """Fit ``H_T ~ H_S @ W + b`` on paired states ``acts_src [N, d_S]``, ``acts_tgt [N, d_T]``.

    A seeded split keeps ``round(holdout * N)`` pairs out (at least one if ``holdout > 0``,
    at most ``N - 2``).  Both sides are centred with the *training* means.

    * ``method="ridge"``: ``W = (Xc^T Xc + alpha I)^-1 Xc^T Yc`` (closed form; ``alpha`` is
      an absolute penalty on the centred Gram matrix, so compare it with the eigenvalues
      of ``Xc^T Xc`` — for mean-pooled Pythia states these are O(1..100)).
    * ``method="procrustes"``: PCA of both sides to rank ``r`` (default
      ``min(d_S, d_T, N_train - 1)``), orthogonal Procrustes ``R`` between the PCA scores
      and the least-squares scalar ``s``: ``W = s P_S R P_T^T``.  ``meta`` holds
      ``P_S [d_S, r]``, ``P_T [d_T, r]``, ``R [r, r]`` and ``scale``.

    ``b = mean_y - mean_x W``.  ``r2_train`` / ``r2_holdout`` are
    ``1 - ||Y - Yhat||^2 / ||Y - mean||^2`` on the respective split, ``r2_permuted`` the
    same on the hold-out pairs with the targets permuted (seeded derangement; computed on
    the training pairs when there is no hold-out).  Seeds: ``seed`` for the split and
    ``seed + 1`` for the permutation.
    """
    X, Y = _np64(acts_src), _np64(acts_tgt)
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("acts_src / acts_tgt must be 2-D [N, d]")
    N = X.shape[0]
    if Y.shape[0] != N:
        raise ValueError(f"paired states need the same N: {N} vs {Y.shape[0]}")
    if N < 3:
        raise ValueError("need at least 3 pairs")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise ValueError("non-finite activations")
    d_S, d_T = X.shape[1], Y.shape[1]

    rng = np.random.default_rng(seed)
    perm = rng.permutation(N)
    n_hold = int(round(float(holdout) * N))
    if holdout > 0:
        n_hold = max(n_hold, 1)
    n_hold = min(n_hold, N - 2)
    hold, train = perm[:n_hold], perm[n_hold:]
    Xt, Yt = X[train], Y[train]
    mx, my = Xt.mean(0), Yt.mean(0)
    Xc, Yc = Xt - mx, Yt - my
    n_train = Xt.shape[0]
    meta: dict = dict(alpha=float(alpha), holdout=float(holdout), seed=int(seed),
                      n_train=int(n_train), n_holdout=int(n_hold), idx_holdout=hold.tolist())

    if method == "ridge":
        G = Xc.T @ Xc + float(alpha) * np.eye(d_S)
        W = np.linalg.solve(G, Xc.T @ Yc)
        rank_used = None if rank is None else int(rank)
    elif method == "procrustes":
        r_max = max(1, min(d_S, d_T, n_train - 1))
        r = r_max if rank is None else max(1, min(int(rank), r_max))
        _, _, VhX = np.linalg.svd(Xc, full_matrices=False)
        _, _, VhY = np.linalg.svd(Yc, full_matrices=False)
        P_S, P_T = VhX[:r].T, VhY[:r].T                        # [d_S, r], [d_T, r]
        Zx, Zy = Xc @ P_S, Yc @ P_T                             # PCA scores
        U, S, Vh = np.linalg.svd(Zx.T @ Zy)
        R = U @ Vh                                              # orthogonal, argmin ||Zx R - Zy||
        s = float(S.sum() / max(float((Zx ** 2).sum()), _EPS))  # least-squares scalar
        W = s * (P_S @ R @ P_T.T)
        rank_used = int(r)
        meta.update(P_S=torch.from_numpy(P_S).float(), P_T=torch.from_numpy(P_T).float(),
                    R=torch.from_numpy(R).float(), scale=s)
    else:
        raise ValueError(f"unknown method {method!r} (use 'ridge' or 'procrustes')")
    b = my - mx @ W

    r2_train = _r2(Yt, Xt @ W + b)
    rng_perm = np.random.default_rng(seed + 1)
    if n_hold > 0:
        Xh, Yh = X[hold], Y[hold]
        r2_holdout = _r2(Yh, Xh @ W + b)
        if n_hold >= 2:
            r2_perm = _r2(Yh[_derangement(n_hold, rng_perm)], Xh @ W + b)
        else:
            r2_perm = float("nan")
    else:
        r2_holdout = float("nan")
        r2_perm = _r2(Yt[_derangement(n_train, rng_perm)], Xt @ W + b)
    return Alignment(W=torch.from_numpy(W).float(), b=torch.from_numpy(b).float(), method=method,
                     rank=rank_used, layer_src=int(layer_src), layer_tgt=int(layer_tgt),
                     r2_train=float(r2_train), r2_holdout=float(r2_holdout), r2_permuted=float(r2_perm),
                     n_pairs=int(N), meta=meta)


def permute_alignment(align: Alignment, seed: int = 0) -> Alignment:
    """Control: the same map with the *target* coordinates permuted (``W[:, p]``, ``b[p]``),
    which keeps the norms but destroys the correspondence."""
    rng = np.random.default_rng(seed)
    p = torch.from_numpy(_derangement(align.d_tgt, rng)).long()
    return Alignment(W=align.W[:, p].clone(), b=align.b[p].clone(), method=align.method + "+permuted",
                     rank=align.rank, layer_src=align.layer_src, layer_tgt=align.layer_tgt,
                     r2_train=float("nan"), r2_holdout=float("nan"), r2_permuted=float("nan"),
                     n_pairs=align.n_pairs, meta=dict(permuted_from=align.method, seed=int(seed)))


def _pool_rows(H: torch.Tensor, readout: str) -> torch.Tensor:
    """``[T, d]`` -> ``[d]`` by mean or last-position readout."""
    H = torch.as_tensor(H).detach().float().cpu()
    if readout == "mean":
        return H.mean(0)
    if readout == "last":
        return H[-1].clone()
    raise ValueError(f"unknown readout {readout!r} (use 'mean' or 'last')")


def paired_states(src: SourceModel, mb_tgt: ModelBundle, pairs: Sequence[tuple], layers_src: Sequence[int],
                  layers_tgt: Sequence[int], readout: str = "mean") -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Pooled residual states of paired items: ``({layer_src: [N, d_S]}, {layer_tgt: [N, d_T]})``.
    ``pairs`` are ``(item_src, text_tgt)``; the same text in both models for a stand-in."""
    S = {l: [] for l in layers_src}
    Tt = {l: [] for l in layers_tgt}
    for item_src, text_tgt in pairs:
        hs = src.residuals(src.encode(item_src), list(layers_src))
        for l in layers_src:
            S[l].append(_pool_rows(hs[l], readout))
        ht = target_residuals(mb_tgt, mb_tgt.encode(text_tgt), list(layers_tgt))
        for l in layers_tgt:
            Tt[l].append(_pool_rows(ht[l], readout))
    return {l: torch.stack(v) for l, v in S.items()}, {l: torch.stack(v) for l, v in Tt.items()}


def layer_matching(src: SourceModel, mb_tgt: ModelBundle, pairs: Sequence[tuple], layers_src=None,
                   layers_tgt=None, readout: str = "mean", **fit_kw) -> pd.DataFrame:
    """Alignment quality for every (source layer, target layer) combination.

    Returns ``DataFrame(layer_src, layer_tgt, r2_train, r2_holdout, r2_permuted, gap, method, n_pairs)``
    with ``gap = r2_holdout - r2_permuted``; the best combination is the layer matching.
    """
    layers_src = list(range(src.n_layers + 1)) if layers_src is None else [int(l) for l in layers_src]
    layers_tgt = list(range(mb_tgt.n_layers + 1)) if layers_tgt is None else [int(l) for l in layers_tgt]
    S, Tt = paired_states(src, mb_tgt, pairs, layers_src, layers_tgt, readout)
    rows = []
    for ls in layers_src:
        for lt in layers_tgt:
            a = fit_alignment(S[ls], Tt[lt], layer_src=ls, layer_tgt=lt, **fit_kw)
            rows.append(dict(layer_src=ls, layer_tgt=lt, r2_train=a.r2_train, r2_holdout=a.r2_holdout,
                             r2_permuted=a.r2_permuted, gap=a.gap, method=a.method, n_pairs=a.n_pairs))
    return pd.DataFrame(rows)


def layer_alignments(src: SourceModel, mb_tgt: ModelBundle, pairs: Sequence[tuple], layers_tgt=None, layers_src=None,
                     readout: str = "mean", **fit_kw) -> dict[int, Alignment]:
    """One :class:`Alignment` per target layer, ``{layer_tgt: Alignment}`` — the input
    :func:`transplant_experiment` needs to plant the KV prefix at *every* target layer
    (DESIGN 4.9, injection B).  For each target layer (default all ``0..L``) the source
    layer with the largest ``gap = r2_holdout - r2_permuted`` among ``layers_src``
    (default all) is taken (layer matching); ``fit_kw`` -> :func:`fit_alignment`.
    """
    layers_src = list(range(src.n_layers + 1)) if layers_src is None else [int(l) for l in layers_src]
    layers_tgt = list(range(mb_tgt.n_layers + 1)) if layers_tgt is None else [int(l) for l in layers_tgt]
    S, Tt = paired_states(src, mb_tgt, pairs, layers_src, layers_tgt, readout)
    out: dict[int, Alignment] = {}
    for lt in layers_tgt:
        best = None
        for ls in layers_src:
            a = fit_alignment(S[ls], Tt[lt], layer_src=ls, layer_tgt=lt, **fit_kw)
            key = a.gap if np.isfinite(a.gap) else -np.inf
            if best is None or key > best[0]:
                best = (key, a)
        out[lt] = best[1]
    return out


def transfer_direction(align: Alignment, v_src) -> torch.Tensor:
    """Direction transfer ``unit(W^T v_S)`` (no bias)."""
    v = torch.as_tensor(v_src).detach().float().cpu().reshape(-1)
    if v.numel() != align.d_src:
        raise ValueError(f"v_src has {v.numel()} entries, alignment expects d_S={align.d_src}")
    return unit(align.W.T @ v)


def transfer_states(align: Alignment, H_src) -> torch.Tensor:
    """State transfer ``H_S @ W + b`` for ``H_src [n, d_S]`` (or ``[d_S]``) -> ``[n, d_T]``."""
    H = torch.as_tensor(H_src).detach().float().cpu()
    single = H.dim() == 1
    H = H.reshape(-1, H.shape[-1])
    if H.shape[1] != align.d_src:
        raise ValueError(f"H_src has d={H.shape[1]}, alignment expects d_S={align.d_src}")
    out = H @ align.W + align.b
    return out[0] if single else out


# ======================================================================================
# injection A: residual substitution
# ======================================================================================
def _norm_positions(positions, T: int) -> list[int]:
    pos = [int(p) + T if int(p) < 0 else int(p) for p in positions]
    if any(p < 0 or p >= T for p in pos):
        raise ValueError(f"positions {positions} outside 0..{T - 1}")
    return pos


def substitute_residual(mb: ModelBundle, input_ids: torch.Tensor, site: HookSite, positions: Sequence[int],
                        H_new, return_info: bool = False):
    """Replace the target's residual at ``site`` and ``positions`` by ``H_new [len(positions), d_T]``
    (``Injector(mode="replace")``, no addition) and return the last-position logits ``[V]``.
    With ``return_info=True`` returns ``(logits, info)`` where ``info`` carries the Injector
    audit (written norm), the full logits ``[T, V]`` and the positions."""
    ids = _as_ids(input_ids, mb.device)
    T = ids.shape[1]
    pos = _norm_positions(positions, T)
    H = torch.as_tensor(H_new).detach().float()
    if H.dim() == 1:
        H = H[None]
    if H.shape[0] != len(pos) or H.shape[1] != mb.d_model:
        raise ValueError(f"H_new must be [{len(pos)}, {mb.d_model}], got {tuple(H.shape)}")
    with torch.no_grad(), Injector(mb, site, H.to(mb.device), positions=pos, mode="replace") as inj:
        out = mb.model(input_ids=ids, use_cache=False)
    logits = out.logits.float()[0].cpu()
    if inj.n_calls != 1:
        warnings.warn(f"substitute_residual: hook at {site.key} fired {inj.n_calls} times")
    if return_info:
        return logits[-1], dict(audit=dict(inj.audit), n_calls=inj.n_calls, logits=logits, positions=pos, site=site.key)
    return logits[-1]


# ======================================================================================
# injection B: KV-cache planting
# ======================================================================================
def _prefix_states(mb: ModelBundle, H_by_layer: dict, layers) -> tuple[list[tuple[int, HookSite, torch.Tensor]], int]:
    """Validate ``H_by_layer`` and select layers; returns ``([(layer, site, H[n, d])], n)``."""
    L = mb.n_layers
    keys = sorted(int(k) for k in H_by_layer) if layers is None else [int(l) for l in layers]
    sel, n = [], None
    for l in keys:
        if l not in H_by_layer:
            raise KeyError(f"layer {l} not in H_by_layer")
        H = torch.as_tensor(H_by_layer[l]).detach().float()
        if H.dim() == 1:
            H = H[None]
        if H.dim() != 2 or H.shape[1] != mb.d_model:
            raise ValueError(f"H_by_layer[{l}] must be [n, {mb.d_model}], got {tuple(H.shape)}")
        if n is None:
            n = int(H.shape[0])
        elif H.shape[0] != n:
            raise ValueError("all entries of H_by_layer must have the same number of prefix rows")
        sel.append((l, _snapshot_site(L, l), H.to(mb.device)))
    return sel, (0 if n is None else n)


def _plain_forward(mb: ModelBundle, ids: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = mb.model(input_ids=ids, use_cache=False).logits.float()[0].cpu()
    return dict(logits_last=logits[-1], logits=logits, n_prefix=0, audits={}, layers=[])


def plant_kv_prefix(mb: ModelBundle, input_ids: torch.Tensor, H_by_layer: dict, placeholder_id: int = 0,
                    layers: Optional[Sequence[int]] = None) -> dict:
    """KV-cache planting through a virtual prefix, in one forward pass.

    Runs ``[placeholder] * n + input_ids`` with one ``Injector(mode="replace")`` per layer
    in ``H_by_layer`` (default: all of its layers; ``layers`` restricts them) on that
    layer's block input at the prefix positions ``0..n-1`` (``layer == n_layers`` means the
    last block's output, which no real token can attend to).  Each layer's block input at
    the prefix is overwritten independently, so the prefix's keys/values at every planted
    layer come from the transplanted states (rotary applied at the prefix positions) while
    the real tokens' own residual stream is untouched.  Returns
    ``dict(logits_last [V], logits [T, V], n_prefix, audits, layers)`` for the real tokens.
    """
    ids = _as_ids(input_ids, mb.device)
    sel, n = _prefix_states(mb, H_by_layer, layers)
    if n == 0:
        return _plain_forward(mb, ids)
    full = torch.cat([torch.full((1, n), int(placeholder_id), dtype=torch.long, device=mb.device), ids], 1)
    audits = {}
    with torch.no_grad(), contextlib.ExitStack() as stack:
        injs = {l: stack.enter_context(Injector(mb, site, H, positions=list(range(n)), mode="replace"))
                for l, site, H in sel}
        out = mb.model(input_ids=full, use_cache=False)
        for l, inj in injs.items():
            audits[l] = dict(inj.audit, n_calls=inj.n_calls)
    logits = out.logits.float()[0, n:].cpu()
    return dict(logits_last=logits[-1], logits=logits, n_prefix=n, audits=audits, layers=[l for l, _, _ in sel])


def plant_kv_cache(mb: ModelBundle, input_ids: torch.Tensor, H_by_layer: dict, placeholder_id: int = 0,
                   layers: Optional[Sequence[int]] = None, strict: bool = False) -> dict:
    """Same effect as :func:`plant_kv_prefix` through transformers' cache API.

    Step 1 runs the placeholder prefix alone with ``use_cache=True`` and the replace
    Injectors active, which fills a ``DynamicCache`` with the prefix's keys/values computed
    from the transplanted states.  Step 2 runs the real tokens with ``past_key_values`` set
    to that cache, an attention mask of ones over prefix + real tokens and
    ``position_ids`` starting at ``n`` (the Injectors are *not* active here: the real
    tokens' residual stream stays untouched).  Verified against :func:`plant_kv_prefix`
    (transformers 5.20, GPT-NeoX): identical logits when every layer is planted (the
    prefix K/V are then computed from the same overwritten inputs in both routes); when
    only a subset of layers is planted the un-planted prefix layers are computed in a
    length-``n`` forward here vs. inside the length-``n+T`` forward there, which differs by
    the cache API's own fp32 noise (~1e-3 in the logits on CPU, the same as a plain
    uninjected two-pass forward).  If the cache API fails, falls back to
    :func:`plant_kv_prefix` with a warning (``strict=True`` raises instead).
    """
    ids = _as_ids(input_ids, mb.device)
    sel, n = _prefix_states(mb, H_by_layer, layers)
    if n == 0:
        return _plain_forward(mb, ids)
    T = ids.shape[1]
    try:
        prefix = torch.full((1, n), int(placeholder_id), dtype=torch.long, device=mb.device)
        audits = {}
        with torch.no_grad(), contextlib.ExitStack() as stack:
            injs = {l: stack.enter_context(Injector(mb, site, H, positions=list(range(n)), mode="replace"))
                    for l, site, H in sel}
            out1 = mb.model(input_ids=prefix, use_cache=True)
            for l, inj in injs.items():
                audits[l] = dict(inj.audit, n_calls=inj.n_calls)
        cache = out1.past_key_values
        if cache is None or int(cache.get_seq_length()) != n:
            raise RuntimeError("prefix forward returned no usable cache")
        attn = torch.ones(1, n + T, dtype=torch.long, device=mb.device)
        pos_ids = torch.arange(n, n + T, device=mb.device)[None]
        with torch.no_grad():
            out2 = mb.model(input_ids=ids, past_key_values=cache, attention_mask=attn, position_ids=pos_ids,
                            use_cache=True)
        logits = out2.logits.float()[0].cpu()
    except Exception as e:  # noqa: BLE001 - documented fallback
        if strict:
            raise
        warnings.warn(f"plant_kv_cache: cache API failed ({type(e).__name__}: {e}); falling back to plant_kv_prefix")
        res = plant_kv_prefix(mb, ids, H_by_layer, placeholder_id=placeholder_id, layers=layers)
        res["fallback"] = True
        return res
    return dict(logits_last=logits[-1], logits=logits, n_prefix=n, audits=audits, layers=[l for l, _, _ in sel],
                cache_len=n + T, fallback=False)


# ======================================================================================
# chunk-aware replacement (works under generate(): prompt chunk, then one token per step)
# ======================================================================================
class ChunkReplacer:
    """Callable delta for :class:`lsne.model.Injector` that writes ``H[i]`` at absolute
    position ``positions[i]`` of the sequence the hook sees, tracking chunk offsets across
    calls (full prompt first, then ``[B,1,d]`` per generated token).  Rows are broadcast
    over the batch.  Use a fresh instance per forward/generate call."""

    def __init__(self, H, positions: Sequence[int]):
        self.H = torch.as_tensor(H).detach().float()
        if self.H.dim() == 1:
            self.H = self.H[None]
        self.positions = [int(p) for p in positions]
        if self.H.shape[0] != len(self.positions):
            raise ValueError("H needs one row per position")
        self.offset = 0
        self.n_written = 0
        self.written_norm = 0.0

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        B, n, d = h.shape
        lo, hi = self.offset, self.offset + n
        self.offset = hi
        sel = [(i, p - lo) for i, p in enumerate(self.positions) if lo <= p < hi]
        if not sel:
            return h
        rows = torch.tensor([i for i, _ in sel], device=h.device)
        cols = torch.tensor([q for _, q in sel], device=h.device)
        h_new = h.clone()
        h_new[:, cols, :] = self.H.to(device=h.device, dtype=h.dtype)[rows]
        with torch.no_grad():
            self.written_norm += float((h_new.float() - h.float()).norm())
        self.n_written += len(sel)
        return h_new


@contextlib.contextmanager
def replacements(mb: ModelBundle, specs: Sequence[tuple[HookSite, Sequence[int], torch.Tensor]]):
    """Enter one replace hook per ``(site, positions, H)`` spec; yields the replacers."""
    reps = [ChunkReplacer(H, positions) for _, positions, H in specs]
    with contextlib.ExitStack() as stack:
        for (site, _, _), rep in zip(specs, reps):
            stack.enter_context(Injector(mb, site, rep))
        yield reps


def _generate_fraction(mb: ModelBundle, ids: torch.Tensor, specs, n_new: int, n_samples: int, seed: int,
                       temperature: float, site_dummy: HookSite) -> dict:
    """Unsteered sampling with the replacement specs active; German fraction of the
    continuation via :func:`lsne.generation.language_trajectory`."""
    with replacements(mb, specs):
        res = generate_steered(mb, ids, site_dummy, None, lambda k: 0.0, n_new=n_new, n_samples=n_samples,
                               temperature=temperature, seed=seed)
    traj = language_trajectory(mb.tokenizer, res["tokens"])
    fD, n_ed = block_fraction(traj["labels"])
    last = traj["fD_window"][:, -1] if traj["fD_window"].shape[1] else np.full(n_samples, np.nan)
    return dict(fD=fD, n_ED=n_ed, fD_window_last=float(np.nanmean(last)) if np.isfinite(last).any() else float("nan"),
                n_switches=float(traj["n_switches"].mean()), tokens=res["tokens"],
                entropy_gen=float(res["entropies"].mean()))


# ======================================================================================
# the transplant experiment (A / B / C against controls)
# ======================================================================================
def _dose_rows(H: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-row State-2143 dose in the *source* space: ``h + eps RMS(h) v / RMS(v)``."""
    if eps == 0.0:
        return H.clone()
    v = v.float() / rms(v.float()).clamp_min(1e-30)
    return H + float(eps) * rms(H)[:, None] * v[None, :]


def _split_prompt(prompt, targets, i: int, mb: ModelBundle) -> tuple[str, Optional[int]]:
    target = None
    if isinstance(prompt, (tuple, list)):
        prompt, target = prompt[0], prompt[1]
    if targets is not None and i < len(targets):
        target = targets[i]
    if isinstance(target, str):
        target = int(mb.tokenizer(target, add_special_tokens=False)["input_ids"][0])
    return str(prompt), (None if target is None else int(target))


def transplant_experiment(mb: ModelBundle, src: SourceModel, align: Union[Alignment, dict], v_src, prompts: Sequence,
                          site: HookSite, eps: float, n_prefix: int = 8,
                          modes: Sequence[str] = ("residual", "kv", "both"),
                          controls: Sequence[str] = ("random", "identity", "permuted_alignment"),
                          gen_kw: Optional[dict] = None, native_axis=None, targets=None, seed: int = 0) -> pd.DataFrame:
    """Transplant the source feature ``v_src`` into the target per prompt x mode x control.

    Per prompt the source states ``H_S`` (at the alignment's source layer) are *dosed* along
    ``v_src`` in the source space (``eps`` relative RMS per position, the State-2143
    convention) and transferred, ``H_T = (H_S + dose) W + b``.  Modes: ``"residual"``
    substitutes ``H_T`` on the last ``k = min(T_S, T)`` real positions at ``site``
    (injection A, no addition); ``"kv"`` plants the last ``n_prefix`` transferred rows as a
    virtual prefix (injection B) at every target layer that has an alignment; ``"both"``
    combines them.  DESIGN 4.9 specifies injection B at **all** layers, which needs a
    ``{target_layer: Alignment}`` dict covering ``0..L`` (:func:`layer_alignments`); a
    single Alignment can only plant its own ``layer_tgt`` (or ``site`` when that is unset),
    i.e. the "selected layers" variant, and a ``UserWarning`` says so whenever a kv mode
    runs with it.  The planted layers are recorded per row in ``kv_layers`` (comma-joined,
    empty in ``"residual"`` mode) and ``n_kv_layers``.  Arms:
    ``control="none"`` (the transplant) plus ``"random"`` (a seeded random source direction
    of equal norm), ``"identity"`` (the target's own, undosed states at the same positions
    / a prefix of the target's own states) and ``"permuted_alignment"``
    (:func:`permute_alignment`).  Measurements: next-token ``entropy`` and ``logp_target``
    (prompts may be ``(text, target)`` pairs or ``targets`` given), the German fraction
    ``fD`` of a short unsteered continuation sampled with the replacements active
    (:func:`lsne.generation.language_trajectory`), ``cos_native`` of the transferred
    direction with ``native_axis`` (if given) and the alignment's ``r2_holdout`` /
    ``r2_permuted``.  ``gen_kw``: ``n_new`` (16), ``n_samples`` (4), ``temperature`` (1.0).
    Position-wise pairing of source and target states is exact only when both models
    share the tokenisation; otherwise the last positions are paired.
    """
    gk = dict(n_new=16, n_samples=4, temperature=1.0)
    gk.update(gen_kw or {})
    L = mb.n_layers
    site_layer = _snapshot_layer(site, L)
    kv_modes = [m for m in modes if m in ("kv", "both")]
    if isinstance(align, Alignment):
        key = align.layer_tgt if 0 <= align.layer_tgt <= L else site_layer
        aligns = {int(key): align}
        if kv_modes:
            warnings.warn(f"transplant_experiment: a single Alignment plants the KV prefix only at target layer {key} "
                          f"(modes {kv_modes}); DESIGN 4.9 injection B wants every layer 0..{L} -- pass a "
                          f"{{layer_tgt: Alignment}} dict from layer_alignments() for that. See the kv_layers column.",
                          UserWarning, stacklevel=2)
    else:
        aligns = {int(k): a for k, a in align.items()}
        if not aligns:
            raise ValueError("align dict is empty")
        bad = [l for l in aligns if not 0 <= l <= L]
        if bad:
            raise ValueError(f"target layers {bad} outside 0..{L}")
    align_res = aligns.get(site_layer, next(iter(aligns.values())))
    kv_layers = sorted(aligns)
    kv_layers_str = ",".join(str(l) for l in kv_layers)
    v_src = torch.as_tensor(v_src).detach().float().cpu().reshape(-1)
    v_T = transfer_direction(align_res, v_src)
    cos_native = cosine(v_T, native_axis) if native_axis is not None else float("nan")
    arms = ["none"] + [c for c in controls if c != "none"]
    layers_src = sorted({a.layer_src for a in aligns.values()})
    if any(l < 0 for l in layers_src):
        raise ValueError("every Alignment needs layer_src >= 0 (the source snapshot it was fitted on)")
    rows = []
    for i, p in enumerate(prompts):
        text, target = _split_prompt(p, targets, i, mb)
        ids_T = mb.encode(text)
        T = ids_T.shape[1]
        H_S = src.residuals(src.encode(text), layers_src)
        own = target_residuals(mb, ids_T, sorted(set(aligns) | {site_layer}))
        for ci, control in enumerate(arms):
            v_dose = v_src
            use = aligns
            if control == "random":
                v_dose = random_directions(v_src.numel(), 1, seed + 1000 * i + ci)[0]
            elif control == "permuted_alignment":
                use = {l: permute_alignment(a, seed + l) for l, a in aligns.items()}
            elif control not in ("none", "identity"):
                raise ValueError(f"unknown control {control!r}")
            # transferred (or own) states per target layer: [T_eff, d_T]
            H_T: dict[int, torch.Tensor] = {}
            if control == "identity":
                for l in set(aligns) | {site_layer}:
                    H_T[l] = own[l]
            else:
                for l, a in use.items():
                    H_T[l] = transfer_states(a, _dose_rows(H_S[a.layer_src], v_dose, eps))
                if site_layer not in H_T:
                    H_T[site_layer] = transfer_states(use[next(iter(use))], _dose_rows(H_S[align_res.layer_src], v_dose, eps))
            for mode in modes:
                if mode not in ("residual", "kv", "both"):
                    raise ValueError(f"unknown mode {mode!r}")
                n = min(int(n_prefix), min(h.shape[0] for h in H_T.values())) if mode in ("kv", "both") else 0
                k = min(T, H_T[site_layer].shape[0]) if mode in ("residual", "both") else 0
                ids = ids_T
                if n > 0:
                    ids = torch.cat([torch.zeros(1, n, dtype=torch.long, device=mb.device), ids_T], 1)
                specs = []
                if n > 0:
                    for l in kv_layers:
                        specs.append((_snapshot_site(L, l), list(range(n)), H_T[l][-n:]))
                if k > 0:
                    specs.append((site, [n + T - k + j for j in range(k)], H_T[site_layer][-k:]))
                with torch.no_grad(), replacements(mb, specs) as reps:
                    logits = mb.model(input_ids=ids, use_cache=False).logits.float()[:, -1].cpu()
                written = float(sum(r.written_norm for r in reps))
                st = next_token_stats(logits, target if target is not None else 0)
                gen = _generate_fraction(mb, ids, specs, gk["n_new"], gk["n_samples"], seed + i, gk["temperature"], site)
                rows.append(dict(prompt_idx=i, prompt=text, mode=mode, control=control, eps=float(eps),
                                 n_prefix=n, n_substituted=k, site=site.key,
                                 kv_layers=kv_layers_str if n > 0 else "", n_kv_layers=len(kv_layers) if n > 0 else 0,
                                 entropy=float(st["entropy"][0]),
                                 logp_target=float(st["logp_target"][0]) if target is not None else float("nan"),
                                 fD=gen["fD"], fD_window_last=gen["fD_window_last"], n_ED=gen["n_ED"],
                                 n_switches=gen["n_switches"], entropy_gen=gen["entropy_gen"],
                                 written_norm=written, cos_native=cos_native,
                                 r2_holdout=align_res.r2_holdout, r2_permuted=align_res.r2_permuted,
                                 layer_src=align_res.layer_src, method=align_res.method))
    return pd.DataFrame(rows)


# ======================================================================================
# functional transfer test (stand-in validation with the language axis)
# ======================================================================================
def _sign(x: float, tol: float = 1e-6) -> int:
    if not np.isfinite(x) or abs(x) < tol:
        return 0
    return 1 if x > 0 else -1


def _fraction_of(mb: ModelBundle, ids: torch.Tensor, site: HookSite, direction, eps: float, n_gen: int,
                 n_samples: int, seed: int, temperature: float) -> tuple[float, int]:
    """German fraction of the steered continuations and the number of E/D-labelled tokens
    it is based on (``0`` means the readout is dead and the fraction is the 0.0 convention)."""
    res = generate_steered(mb, ids, site, direction, (lambda k: float(eps)) if direction is not None else (lambda k: 0.0),
                           n_new=n_gen, n_samples=n_samples, temperature=temperature, seed=seed)
    traj = language_trajectory(mb.tokenizer, res["tokens"])
    fD, n_ed = block_fraction(traj["labels"])
    return float(fD), int(n_ed)


def functional_transfer_test(mb_tgt: ModelBundle, src: SourceModel, texts_E: Sequence[str], texts_D: Sequence[str],
                             site_src: int, site_tgt: HookSite, native_axis=None, n_gen: int = 16, n_samples: int = 4,
                             seed: int = 0, texts_neutral: Optional[Sequence[str]] = None, eps: float = 1.0,
                             prompts: Union[str, Sequence[str]] = "Yesterday we went to the market and",
                             fit_kw: Optional[dict] = None, temperature: float = 1.0, denoise_pcs: int = 1,
                             min_delta: float = 0.0, n_null: int = 10) -> dict:
    """Does a feature found in the *source* model survive the transfer functionally?

    1. Source language axis: ``diff_of_means`` of the mean-pooled source residuals (layer
       ``site_src``) of the German vs. the English sentences.
    2. Alignment ``site_src -> site_tgt`` fitted on the mean-pooled *paired* states of the
       same texts in both models (``texts_E + texts_D + texts_neutral``), ``fit_kw`` ->
       :func:`fit_alignment`.  The default is ``method="procrustes"`` (scale-free; with a
       few dozen pooled pairs it generalises better than ridge, whose absolute ``alpha``
       would have to be tuned to the residual scale of each source model).  The *null* is
       the same fit on permuted pairs (``n_null`` seeded derangements of the target rows,
       generator seed ``seed + 2``): ``cos_null_vs_native`` (mean over the draws of the
       cosine of the source axis transferred through the null map; ``cos_null_std``,
       ``cos_null_max`` and ``z_cos_vs_null`` describe the distribution) and
       ``r2_holdout_null`` / ``r2_holdout_null_max`` are what a broken pairing produces,
       so the transfer counts only if ``cos_transferred_vs_native`` and ``r2_holdout``
       beat them clearly (a single draw scatters with std ~0.3-0.4 because a random
       derangement keeps about half of the pairs within the same language).
    3. The transferred axis ``unit(W^T v_S)`` steers the target (:func:`lsne.model.generate_steered`,
       constant dose ``eps``); the German fraction of the continuations (pooled over
       ``prompts`` x ``n_samples``) is compared with the unsteered baseline and with the
       *native* axis (``native_axis`` if given, else ``diff_of_means`` in the target on the
       same texts).  Baseline and steered runs share ``seed``.

    Returns ``dict(cos_transferred_vs_native, cos_null_vs_native, cos_null_std, cos_null_max,
    z_cos_vs_null, delta_fD_transferred, delta_fD_native, sign_agreement, readout_alive,
    n_ED, r2_holdout, r2_holdout_null, r2_holdout_null_max, r2_permuted, alignment,
    alignment_null (the first null draw), ...)``.  ``n_ED`` counts the E/D-labelled generated tokens per arm
    (``baseline`` / ``transferred`` / ``native``, summed over prompts); ``readout_alive``
    is True only if every arm has at least one.  ``sign_agreement`` is a *measurement*,
    not a default: it is True only if the readout is alive **and** both deltas are
    non-zero (``|delta| > max(min_delta, 1e-6)``) **and** their signs agree; a dead
    readout or an ineffective dose (both deltas 0, e.g. ``eps=0``) gives False.  A dead
    arm additionally raises a ``UserWarning``.

    Caveat: with the same model as source and target (and the same layer) the Procrustes
    map is a symmetric projector, ``v_src == v_native`` and every criterion above is met
    by construction (it is then a plumbing smoke test).  Use :class:`RotatedSourceModel`
    or a different ``site_src`` for an informative self-consistency check.
    """
    texts_E, texts_D = list(texts_E), list(texts_D)
    texts_N = list(texts_neutral or [])
    texts_all = texts_E + texts_D + texts_N
    fk = dict(method="procrustes")
    fk.update(fit_kw or {})
    S_rows, T_rows = [], []
    for t in texts_all:
        S_rows.append(_pool_rows(src.residuals(src.encode(t), [int(site_src)])[int(site_src)], "mean"))
        co = run_with_cache(mb_tgt, mb_tgt.encode(t), [site_tgt])
        T_rows.append(co.resid[site_tgt.key][0].detach().float().cpu().mean(0))
    S, Tt = torch.stack(S_rows), torch.stack(T_rows)
    nE, nD = len(texts_E), len(texts_D)
    v_src = diff_of_means(S[nE:nE + nD], S[:nE], denoise_pcs=denoise_pcs)
    layer_tgt = _snapshot_layer(site_tgt, mb_tgt.n_layers)
    align = fit_alignment(S, Tt, seed=seed, layer_src=int(site_src), layer_tgt=layer_tgt, **fk)
    v_T = transfer_direction(align, v_src)
    native = unit(native_axis) if native_axis is not None else diff_of_means(Tt[nE:nE + nD], Tt[:nE], denoise_pcs=denoise_pcs)
    cos = cosine(v_T, native)
    # null: the same fit on permuted pairs (a broken correspondence), same split seed.  One
    # derangement is not enough: it keeps ~half of the pairs within the same language and the
    # language axis carries much of the pooled variance, so a single null cosine scatters
    # with std ~0.3-0.4 (pythia-14m); the null is therefore a distribution over n_null draws.
    n_null = max(1, int(n_null))
    rng_null = np.random.default_rng(seed + 2)
    cos_nulls, r2_nulls, align_null = [], [], None
    for _ in range(n_null):
        perm = _derangement(S.shape[0], rng_null)
        a_n = fit_alignment(S, Tt[torch.from_numpy(perm).long()], seed=seed, layer_src=int(site_src),
                            layer_tgt=layer_tgt, **fk)
        a_n.method += "+null"
        cos_nulls.append(cosine(transfer_direction(a_n, v_src), native))
        r2_nulls.append(a_n.r2_holdout)
        align_null = a_n if align_null is None else align_null
    cos_nulls, r2_nulls = np.asarray(cos_nulls, dtype=np.float64), np.asarray(r2_nulls, dtype=np.float64)
    cos_null, cos_null_std = float(cos_nulls.mean()), float(cos_nulls.std())
    z_cos = float((cos - cos_null) / cos_null_std) if cos_null_std > 0 else float("nan")
    prompt_list = [prompts] if isinstance(prompts, str) else list(prompts)
    f_base, f_tr, f_na = [], [], []
    n_ed = dict(baseline=0, transferred=0, native=0)
    for j, p in enumerate(prompt_list):
        ids = mb_tgt.encode(p)
        for arm, store, direction, e in (("baseline", f_base, None, 0.0), ("transferred", f_tr, v_T, eps),
                                         ("native", f_na, native, eps)):
            f, n = _fraction_of(mb_tgt, ids, site_tgt, direction, e, n_gen, n_samples, seed + j, temperature)
            store.append(f)
            n_ed[arm] += n
    fD_base, fD_tr, fD_na = float(np.mean(f_base)), float(np.mean(f_tr)), float(np.mean(f_na))
    d_tr, d_na = fD_tr - fD_base, fD_na - fD_base
    readout_alive = all(n > 0 for n in n_ed.values())
    if not readout_alive:
        dead = [arm for arm, n in n_ed.items() if n == 0]
        warnings.warn(f"functional_transfer_test: no E/D-labelled generated tokens in arm(s) {dead} "
                      f"(n_gen={n_gen}, n_samples={n_samples}); the readout is dead, sign_agreement is False",
                      UserWarning, stacklevel=2)
    tol = max(float(min_delta), 1e-6)
    s_tr, s_na = _sign(d_tr, tol), _sign(d_na, tol)
    sign_agreement = bool(readout_alive and s_tr != 0 and s_na != 0 and s_tr == s_na)
    return dict(cos_transferred_vs_native=float(cos), cos_null_vs_native=cos_null, cos_null_std=cos_null_std,
                cos_null_max=float(cos_nulls.max()), z_cos_vs_null=z_cos, n_null=int(n_null),
                delta_fD_transferred=float(d_tr), delta_fD_native=float(d_na),
                sign_agreement=sign_agreement, readout_alive=bool(readout_alive), n_ED=n_ed, min_delta=float(tol),
                r2_holdout=align.r2_holdout, r2_holdout_null=float(np.nanmean(r2_nulls)),
                r2_holdout_null_max=float(np.nanmax(r2_nulls)),
                r2_permuted=align.r2_permuted, r2_train=align.r2_train, fD_baseline=fD_base, fD_transferred=fD_tr,
                fD_native=fD_na, eps=float(eps), n_pairs=align.n_pairs, alignment=align, alignment_null=align_null,
                v_src=v_src, v_transferred=v_T, v_native=native, site_src=int(site_src), site_tgt=site_tgt.key)
