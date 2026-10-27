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

**Prompt pairs** (:class:`PromptPair`, :func:`as_prompt_pair`).  With a language source
both models read the same string; a trace source reads a *transaction* while Pythia reads
a *text*.  Every prompt argument of :func:`transplant_experiment` and of the one-prompt
helpers (:func:`transplanted_states`, :func:`transplant_residual`, :func:`transplant_kv`;
:func:`substitute_residual` / :func:`plant_kv_prefix` / :func:`plant_kv_cache` take the
target side of a pair as ``input_ids``) is therefore one of: a plain string (the old
behaviour: the same text on both sides), ``(text, target)`` with a string first element
(the old prompt-with-target form), a :class:`PromptPair` ``(source, text, target, ids)``,
``(source_item, text[, target])`` with a non-string first element (e.g. a
:class:`lsne.chain.itr.TxRecord`), or a dict with ``"text"`` (an ITR shard / pair-file row
is its own source item through its ``"tokens"``).  The source encodes ``pair.source``,
the target ``pair.text`` (or ``pair.ids``).
"""
from __future__ import annotations

import contextlib
import math
import warnings
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, Union, runtime_checkable

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


# ======================================================================================
# prompt pairs: the source encodes one item, the target another
# ======================================================================================
@dataclass
class PromptPair:
    """One transplant prompt with (possibly) different items for the two models.

    ``source`` is what the source model encodes (``src.encode(source)``: a trace record, ITR
    tokens or a shard row for :class:`lsne.chain.source.TraceLMSource`, token ids, or a text
    for a language source); ``text`` is what the target encodes (``mb.encode(text)``, or
    ``ids`` when given, e.g. the curated token ids of a stimulus variant); ``target`` is the
    target-side next token (id or string; optional); ``name`` labels the source item in the
    result tables (default: its ``tx_hash`` when it has one)."""
    source: Any
    text: str
    target: Optional[Union[int, str]] = None
    ids: Optional[Sequence[int]] = None
    name: Optional[str] = None

    @property
    def paired(self) -> bool:
        """True when the two sides encode different items (not one plain string)."""
        return not (isinstance(self.source, str) and self.source == self.text)


def _is_ids(x) -> bool:
    """Token ids (tensor / nested int sequence) as opposed to a prompt."""
    if isinstance(x, torch.Tensor):
        return True
    if isinstance(x, (str, bytes, dict, PromptPair)) or x is None:
        return False
    try:
        a = np.asarray(x)
    except (ValueError, TypeError):
        return False
    return a.size > 0 and np.issubdtype(a.dtype, np.integer)


def as_prompt_pair(prompt, target=None) -> PromptPair:
    """Normalise a prompt to a :class:`PromptPair` (see the module docstring).

    * ``str`` -> the same text on both sides (the old behaviour);
    * ``(text, target)`` / ``[text]`` with a **string** first element -> the old prompt with
      a target (a raw ITR string as source item therefore needs an explicit PromptPair);
    * ``(source_item, text[, target])`` with a non-string first element -> a pair;
    * ``dict`` with ``"text"`` -> source ``d["source"]`` / ``d["source_item"]`` / ``d["record"]``
      or the dict itself (a shard / pair-file row carries ``"tokens"``), target
      ``d["target"]`` / ``d["target_id"]``, ids ``d["ids"]``;
    * a :class:`PromptPair` is returned as is.
    ``target`` (if not None) overrides the prompt's own target."""
    if isinstance(prompt, PromptPair):
        pp = prompt
    elif isinstance(prompt, str):
        pp = PromptPair(source=prompt, text=prompt)
    elif isinstance(prompt, dict):
        if "text" not in prompt:
            raise ValueError("a dict prompt needs a 'text' entry (the target side)")
        src_item = prompt
        for k in ("source", "source_item", "record"):
            if prompt.get(k) is not None:
                src_item = prompt[k]
                break
        tgt = prompt.get("target", prompt.get("target_id"))
        pp = PromptPair(source=src_item, text=str(prompt["text"]), target=tgt, ids=prompt.get("ids"),
                        name=prompt.get("name") or prompt.get("tx_hash"))
    elif isinstance(prompt, (tuple, list)) and len(prompt) >= 1:
        if isinstance(prompt[0], str):
            pp = PromptPair(source=prompt[0], text=prompt[0], target=prompt[1] if len(prompt) > 1 else None)
        else:
            if len(prompt) < 2 or not isinstance(prompt[1], str):
                raise ValueError("a (source_item, text[, target]) prompt needs the target text as second element")
            pp = PromptPair(source=prompt[0], text=prompt[1], target=prompt[2] if len(prompt) > 2 else None)
    else:
        raise TypeError(f"cannot read a prompt from {type(prompt).__name__}")
    if target is not None:
        pp = PromptPair(source=pp.source, text=pp.text, target=target, ids=pp.ids, name=pp.name)
    return pp


def source_name(item, max_chars: int = 60) -> str:
    """Short label of a source item: its ``tx_hash`` when it has one, else the text (cut)."""
    h = item.get("tx_hash") if isinstance(item, dict) else getattr(item, "tx_hash", None)
    if h:
        return str(h)
    if isinstance(item, str):
        return item if len(item) <= max_chars else item[: max_chars - 3] + "..."
    if _is_ids(item):
        return f"ids[{int(torch.as_tensor(item).numel())}]"
    return type(item).__name__


def _target_ids(mb: ModelBundle, x) -> torch.Tensor:
    """Target-side ``LongTensor[1, T]`` of ids or of a prompt (pair): ``pair.ids`` or
    ``mb.encode(pair.text)``."""
    if _is_ids(x):
        return _as_ids(x, mb.device)
    pp = as_prompt_pair(x)
    if pp.ids is not None:
        return _as_ids(list(pp.ids), mb.device)
    return mb.encode(pp.text)


def _resolve_prompt(prompt, targets, i: int, mb: ModelBundle) -> tuple[PromptPair, torch.Tensor, Optional[int]]:
    """``(pair, target ids [1, T], target token id or None)`` of prompt ``i``; ``targets[i]``
    (when given) overrides the prompt's own target (the old ``_split_prompt`` rule)."""
    pp = as_prompt_pair(prompt)
    target = pp.target
    if targets is not None and i < len(targets):
        target = targets[i]
    if isinstance(target, str):
        target = int(mb.tokenizer(target, add_special_tokens=False)["input_ids"][0])
    return pp, _target_ids(mb, pp), (None if target is None else int(target))


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
    ``pairs`` are ``(item_src, text_tgt)`` tuples (always read as source item / target text,
    also when the item is a string; the same text in both models for a stand-in), or
    :class:`PromptPair` objects / dicts with ``"text"`` (:func:`as_prompt_pair`)."""
    S = {l: [] for l in layers_src}
    Tt = {l: [] for l in layers_tgt}
    for p in pairs:
        if isinstance(p, (PromptPair, dict)):
            pp = as_prompt_pair(p)
            item_src, ids_tgt = pp.source, _target_ids(mb_tgt, pp)
        else:
            item_src, text_tgt = p
            ids_tgt = _target_ids(mb_tgt, text_tgt) if _is_ids(text_tgt) else mb_tgt.encode(text_tgt)
        hs = src.residuals(src.encode(item_src), list(layers_src))
        for l in layers_src:
            S[l].append(_pool_rows(hs[l], readout))
        ht = target_residuals(mb_tgt, ids_tgt, list(layers_tgt))
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
    audit (written norm), the full logits ``[T, V]`` and the positions.  ``input_ids`` may
    also be a text or a prompt pair (its target side is encoded, :func:`as_prompt_pair`)."""
    ids = _target_ids(mb, input_ids)
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
    ``input_ids`` may also be a text or a prompt pair (its target side is encoded).
    """
    ids = _target_ids(mb, input_ids)
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
    :func:`plant_kv_prefix` with a warning (``strict=True`` raises instead).  ``input_ids``
    may also be a text or a prompt pair (its target side is encoded).
    """
    ids = _target_ids(mb, input_ids)
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
def source_directions(v_src) -> Union[torch.Tensor, dict]:
    """``v_src`` as a float CPU vector, or - for layer-consistent directions - a
    ``{layer_src: vector}`` dict (:func:`direction_at`)."""
    if isinstance(v_src, dict):
        if not v_src:
            raise ValueError("v_src dict is empty")
        return {int(l): torch.as_tensor(v).detach().float().cpu().reshape(-1) for l, v in v_src.items()}
    return torch.as_tensor(v_src).detach().float().cpu().reshape(-1)


def direction_at(v_src, layer_src: int) -> torch.Tensor:
    """The source direction dosed into the states of source layer ``layer_src``: ``v_src`` itself
    (one vector for every layer) or ``v_src[layer_src]`` of a ``{layer_src: vector}`` dict, so that
    every alignment map is dosed with the direction measured at its own source layer."""
    if isinstance(v_src, dict):
        if int(layer_src) not in v_src:
            raise KeyError(f"no source direction for source layer {layer_src}; have {sorted(v_src)}")
        return v_src[int(layer_src)]
    return v_src


def _dose_rows(H: torch.Tensor, v: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-row State-2143 dose in the *source* space: ``h + eps RMS(h) v / RMS(v)``."""
    if eps == 0.0:
        return H.clone()
    v = v.float() / rms(v.float()).clamp_min(1e-30)
    return H + float(eps) * rms(H)[:, None] * v[None, :]


def _split_prompt(prompt, targets, i: int, mb: ModelBundle) -> tuple[str, Optional[int]]:
    """``(target text, target id)`` of prompt ``i`` (kept for callers of the old helper)."""
    pp, _ids, target = _resolve_prompt(prompt, targets, i, mb)
    return str(pp.text), target


# --------------------------------------------------------------------------------------
# one-prompt helpers (prompt pairs: the source encodes pair.source, the target pair.text)
# --------------------------------------------------------------------------------------
def source_states(src: SourceModel, prompt, layers: Sequence[int]) -> dict[int, torch.Tensor]:
    """Source residuals ``{layer: [T_S, d_S]}`` of the *source side* of a prompt (pair)."""
    pp = as_prompt_pair(prompt)
    return src.residuals(src.encode(pp.source), [int(l) for l in layers])


def transplanted_states(src: SourceModel, align: Union[Alignment, dict], prompt, v_src=None, eps: float = 0.0,
                        layers_tgt: Optional[Sequence[int]] = None) -> dict[int, torch.Tensor]:
    """Transferred states ``{layer_tgt: [T_S, d_T]}`` of the source side of ``prompt``: the
    source residuals at each alignment's source layer, dosed along ``v_src`` (``eps``
    relative RMS per position, :func:`transplant_experiment` convention; no dose when
    ``v_src`` is None or ``eps == 0``; a ``{layer_src: vector}`` dict doses each map's source
    layer with its own direction, :func:`direction_at`) and mapped with ``H_S W + b``.
    ``align``: one :class:`Alignment` (keyed by its ``layer_tgt``) or a ``{layer_tgt: Alignment}``
    dict."""
    aligns = {int(align.layer_tgt): align} if isinstance(align, Alignment) else {int(k): a for k, a in align.items()}
    if layers_tgt is not None:
        aligns = {int(l): aligns[int(l)] for l in layers_tgt}
    if not aligns:
        raise ValueError("no alignment")
    H_S = source_states(src, prompt, sorted({int(a.layer_src) for a in aligns.values()}))
    v = None if v_src is None else source_directions(v_src)
    out = {}
    for l, a in aligns.items():
        H = torch.as_tensor(H_S[int(a.layer_src)]).detach().float().cpu()
        if v is not None and float(eps) != 0.0:
            H = _dose_rows(H, direction_at(v, int(a.layer_src)), float(eps))
        out[l] = transfer_states(a, H)
    return out


def transplant_residual(mb: ModelBundle, src: SourceModel, align: Union[Alignment, dict], prompt, site: HookSite,
                        v_src=None, eps: float = 0.0, n_substitute: Optional[int] = None, return_info: bool = False):
    """Injection A for one prompt (pair): the transferred states of the source side
    (:func:`transplanted_states`) replace the target's residual at ``site`` on the last
    ``k = min(n_substitute or T_S, T_S, T)`` positions of the target side (last source rows
    paired with the last target positions).  ``align``: one Alignment or a dict holding the
    site's snapshot layer.  Returns :func:`substitute_residual`'s result."""
    layer = _snapshot_layer(site, mb.n_layers)
    if isinstance(align, Alignment):
        a = align
    else:
        if int(layer) not in {int(k) for k in align}:
            raise KeyError(f"no alignment for target layer {layer} (site {site.key})")
        a = {int(k): v for k, v in align.items()}[int(layer)]
    H_T = next(iter(transplanted_states(src, a, prompt, v_src, eps).values()))
    ids = _target_ids(mb, prompt)
    T = int(ids.shape[1])
    k = min(T, int(H_T.shape[0]), int(n_substitute) if n_substitute else T)
    return substitute_residual(mb, ids, site, list(range(T - k, T)), H_T[-k:], return_info=return_info)


def transplant_kv(mb: ModelBundle, src: SourceModel, aligns: Union[Alignment, dict], prompt, v_src=None, eps: float = 0.0,
                  n_prefix: int = 8, placeholder_id: int = 0, use_cache_api: bool = False) -> dict:
    """Injection B for one prompt (pair): the last ``n = min(n_prefix, T_S)`` transferred rows
    of the source side are planted as a virtual prefix at every target layer of ``aligns``
    (:func:`plant_kv_prefix`, or :func:`plant_kv_cache` with ``use_cache_api``) in front of
    the target side's tokens."""
    if isinstance(aligns, Alignment) and not 0 <= int(aligns.layer_tgt) <= mb.n_layers:
        raise ValueError(f"the alignment's layer_tgt ({aligns.layer_tgt}) must name a target layer 0..{mb.n_layers}")
    H = transplanted_states(src, aligns, prompt, v_src, eps)
    n = min(int(n_prefix), min(int(h.shape[0]) for h in H.values()))
    H = {l: h[-n:] for l, h in H.items()} if n > 0 else {}
    fn = plant_kv_cache if use_cache_api else plant_kv_prefix
    return fn(mb, _target_ids(mb, prompt), H, placeholder_id=placeholder_id)


def _random_key(pp: PromptPair, i: int) -> int:
    """Seed key of the random-direction control of prompt ``i``: one key per *source item*
    for paired prompts (its ``name`` / :func:`source_name`), so prompts that share a source
    item -- the K=0 variant and its K=8/C1 partner on one carrier -- are dosed with the same
    random direction, as the transplant arm doses them with the same ``v_src``; the prompt
    index for plain prompts (the same text on both sides)."""
    if not pp.paired:
        return int(i)
    key = str(pp.name or source_name(pp.source))
    return int(zlib.crc32(key.encode("utf-8")) % 1_000_003)


def transplant_experiment(mb: ModelBundle, src: SourceModel, align: Union[Alignment, dict], v_src, prompts: Sequence,
                          site: HookSite, eps: float, n_prefix: int = 8,
                          modes: Sequence[str] = ("residual", "kv", "both"),
                          controls: Sequence[str] = ("random", "identity", "permuted_alignment"),
                          gen_kw: Optional[dict] = None, native_axis=None, targets=None, seed: int = 0,
                          n_substitute: Optional[int] = None, state_sites: Optional[Sequence[HookSite]] = None):
    """Transplant the source feature ``v_src`` into the target per prompt x mode x control.

    ``v_src``: one source vector, or a ``{layer_src: vector}`` dict of layer-consistent
    directions (every alignment map is dosed with the direction measured at its own source
    layer, :func:`direction_at`; rows record ``v_src_per_layer`` and ``v_src_layer``).

    ``prompts``: plain strings (the same text in both models, the old behaviour),
    ``(text, target)`` tuples, or prompt *pairs* whose source side differs from the target
    side (:class:`PromptPair`, ``(source_item, text[, target])``, dicts with ``"text"``;
    :func:`as_prompt_pair`): the source encodes ``pair.source`` (e.g. a trace record), the
    target ``pair.text`` (or ``pair.ids``).
    Per prompt the source states ``H_S`` (at the alignment's source layer) are *dosed* along
    ``v_src`` in the source space (``eps`` relative RMS per position, the State-2143
    convention) and transferred, ``H_T = (H_S + dose) W + b``.  Modes: ``"residual"``
    substitutes ``H_T`` on the last ``k = min(T_S, T)`` real positions at ``site``
    (``n_substitute`` caps ``k``; injection A, no addition); ``"kv"`` plants the last ``n_prefix`` transferred rows as a
    virtual prefix (injection B) at every target layer that has an alignment; ``"both"``
    combines them.  DESIGN 4.9 specifies injection B at **all** layers, which needs a
    ``{target_layer: Alignment}`` dict covering ``0..L`` (:func:`layer_alignments`); a
    single Alignment can only plant its own ``layer_tgt`` (or ``site`` when that is unset),
    i.e. the "selected layers" variant, and a ``UserWarning`` says so whenever a kv mode
    runs with it.  The planted layers are recorded per row in ``kv_layers`` (comma-joined,
    empty in ``"residual"`` mode) and ``n_kv_layers``.  Arms:
    ``control="none"`` (the transplant) plus ``"random"`` (a seeded random source direction
    of equal norm; one direction per source item of paired prompts, so prompts sharing a
    carrier share it, :func:`_random_key`), ``"identity"`` (the target's own, undosed states at the same positions
    / a prefix of the target's own states) and ``"permuted_alignment"``
    (:func:`permute_alignment`).  Measurements: next-token ``entropy`` and ``logp_target``
    (prompts may be ``(text, target)`` pairs or ``targets`` given), the German fraction
    ``fD`` of a short unsteered continuation sampled with the replacements active
    (:func:`lsne.generation.language_trajectory`), ``cos_native`` of the transferred
    direction with ``native_axis`` (if given) and the alignment's ``r2_holdout`` /
    ``r2_permuted``.  ``gen_kw``: ``n_new`` (16), ``n_samples`` (4), ``temperature`` (1.0).
    Position-wise pairing of source and target states is exact only when both models
    share the tokenisation; otherwise the last positions are paired.  Every row also
    records ``paired`` (the two sides encode different items), ``source_item``
    (:func:`source_name`), ``n_src_tokens`` and ``n_tgt_tokens``.

    ``state_sites``: when given, the forward pass of every row also records the target's
    residual at these sites at the last real position, and the function returns
    ``(df, states)`` with ``states[i] = {site.key: ndarray[d_T]}`` for row ``i`` of ``df``
    (the input of the sharing / network-activity readouts).
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
    v_src = source_directions(v_src)
    per_layer = isinstance(v_src, dict)
    for a in aligns.values():
        direction_at(v_src, int(a.layer_src))               # every map needs its source layer's direction
    d_src = int(next(iter(v_src.values())).numel()) if per_layer else int(v_src.numel())
    v_T = transfer_direction(align_res, direction_at(v_src, int(align_res.layer_src)))
    cos_native = cosine(v_T, native_axis) if native_axis is not None else float("nan")
    arms = ["none"] + [c for c in controls if c != "none"]
    layers_src = sorted({a.layer_src for a in aligns.values()})
    if any(l < 0 for l in layers_src):
        raise ValueError("every Alignment needs layer_src >= 0 (the source snapshot it was fitted on)")
    rec_sites = list(state_sites or [])
    rows, states = [], []
    for i, p in enumerate(prompts):
        pp, ids_T, target = _resolve_prompt(p, targets, i, mb)
        text = pp.text
        T = ids_T.shape[1]
        H_S = src.residuals(src.encode(pp.source), layers_src)
        T_S = int(next(iter(H_S.values())).shape[0])
        src_label = pp.name or source_name(pp.source)
        own = target_residuals(mb, ids_T, sorted(set(aligns) | {site_layer}))
        for ci, control in enumerate(arms):
            v_dose = v_src
            use = aligns
            if control == "random":
                v_dose = random_directions(d_src, 1, seed + 1000 * _random_key(pp, i) + ci)[0]
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
                    H_T[l] = transfer_states(a, _dose_rows(H_S[a.layer_src], direction_at(v_dose, int(a.layer_src)), eps))
                if site_layer not in H_T:
                    H_T[site_layer] = transfer_states(use[next(iter(use))], _dose_rows(H_S[align_res.layer_src],
                                                                                       direction_at(v_dose, int(align_res.layer_src)), eps))
            for mode in modes:
                if mode not in ("residual", "kv", "both"):
                    raise ValueError(f"unknown mode {mode!r}")
                n = min(int(n_prefix), min(h.shape[0] for h in H_T.values())) if mode in ("kv", "both") else 0
                k = min(T, H_T[site_layer].shape[0]) if mode in ("residual", "both") else 0
                if k and n_substitute is not None:
                    k = min(k, max(1, int(n_substitute)))
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
                    if rec_sites:
                        co = run_with_cache(mb, ids, rec_sites)
                        logits = co.logits[:, -1].cpu()
                        states.append({s.key: co.resid[s.key][0, -1].detach().float().cpu().numpy() for s in rec_sites})
                    else:
                        logits = mb.model(input_ids=ids, use_cache=False).logits.float()[:, -1].cpu()
                written = float(sum(r.written_norm for r in reps))
                st = next_token_stats(logits, target if target is not None else 0)
                gen = _generate_fraction(mb, ids, specs, gk["n_new"], gk["n_samples"], seed + i, gk["temperature"], site)
                rows.append(dict(prompt_idx=i, prompt=text, paired=bool(pp.paired), source_item=src_label,
                                 n_src_tokens=T_S, n_tgt_tokens=int(T), mode=mode, control=control, eps=float(eps),
                                 n_prefix=n, n_substituted=k, site=site.key,
                                 kv_layers=kv_layers_str if n > 0 else "", n_kv_layers=len(kv_layers) if n > 0 else 0,
                                 entropy=float(st["entropy"][0]),
                                 logp_target=float(st["logp_target"][0]) if target is not None else float("nan"),
                                 fD=gen["fD"], fD_window_last=gen["fD_window_last"], n_ED=gen["n_ED"],
                                 n_switches=gen["n_switches"], entropy_gen=gen["entropy_gen"],
                                 written_norm=written, cos_native=cos_native,
                                 r2_holdout=align_res.r2_holdout, r2_permuted=align_res.r2_permuted,
                                 layer_src=align_res.layer_src, method=align_res.method,
                                 v_src_per_layer=per_layer, v_src_layer=(int(align_res.layer_src) if per_layer else -1)))
    df = pd.DataFrame(rows)
    return (df, states) if rec_sites else df


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
