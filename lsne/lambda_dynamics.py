"""lambda-dynamics: Biggio et al. Algorithm 1 in activation space (DESIGN.md section 4.6).

Biggio et al. (2013/2017, *Evasion attacks against machine learning at test time*,
arXiv:1708.06131) descend a discriminant ``g`` while a kernel-density "mimicry" term pulls
the point into the populated region of the target class:

    F(x) = g(x) - (lam / n) sum_i k((x - x_i) / h)      s.t.  d(x, x0) <= d_max

Algorithm 1: unit step along grad F, projection onto the feasible ball, stop as soon as F
no longer decreases. Here the point ``x`` is the residual vector ``h`` at one
``(HookSite, position)`` of the model, ``g`` is one of

* ``"probe"``         g(h) = w . h + b                      (linear language probe; positive = source class)
* ``"target_logit"``  g(h) = -log p(target | h) + b         (model-own discriminant with flat regions)
* ``"entropy"``       g(h) = -H(next token | h) + b         (descending g raises the entropy)
* ``"sae_feature"``   g(h) = -relu(pre-activation_f(h)) + b (descending g activates SAE feature f)

and the target density ``p_hat`` is a KDE over *real* activations of the target condition
at the same site (:class:`KDEMimicry`), either in raw l2 coordinates or in the
Mahalanobis-whitened coordinates of the layer covariance (:func:`lsne.geometry.layer_covariance`).

Conventions
-----------
* Discriminant ``fn``/``grad`` accept a 1-D tensor on any device and return CPU float32.
* ``F(h) = g(h) - lam * density(h)`` with ``density = (1/n) sum_i k_i``, i.e. ``lam`` multiplies
  the ``(1/n) sum k`` term exactly as in the paper; :func:`lambda_scale` gives the paper's
  scale ``lam = g_range * n * bw`` (so that ``lam / (n bw)`` equals the range of ``g``).
* RBF kernel ``k_i = exp(-||z - z_i||^2 / bw)``, Laplacian kernel ``k_i = exp(-||z - z_i||_1 / bw)``.
  ``bandwidth="median"`` sets ``bw`` so that the kernel equals ``e^-1`` at the median pairwise
  distance of the references: ``bw = median^2`` (RBF) or ``bw = median`` (Laplacian). The
  gradients are the exact derivatives (for the Laplacian kernel this carries
  ``sign(z - z_i)``, where the paper's printed formula shows ``(z - z_i)``).
* ``k_nearest``: only the ``k`` nearest references (in the kernel's distance) enter the sum,
  which is then normalised by the number of points used (the paper used 50).
* Trajectory arithmetic is done in float64; ``Trajectory.points`` are stored as float32
  (``dist_from_start`` is computed from the exact float64 path).
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .config import HookSite
from .dynamics import last_logits
from .geometry import log_volume, mahalanobis
from .model import Injector, ModelBundle, _first_tensor, next_token_stats

__all__ = [
    "Discriminant", "partial_forward", "make_discriminant", "KDEMimicry", "lambda_scale",
    "Trajectory", "evasion_trajectory", "lambda_sweep", "trajectory_geometry",
    "manifold_stats", "endpoint_effects", "feature_lambda_correlation",
    "geometry_lambda_correlation",
]

_LAM_FLOOR = 1e-300
_FLAT_NORM = 1e-12


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _vec32(x) -> torch.Tensor:
    """Any 1-D array-like (torch on any device, numpy, list, Direction) -> float32 CPU, detached."""
    if hasattr(x, "vec") and isinstance(getattr(x, "vec"), torch.Tensor):   # directions.Direction
        x = x.vec
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().reshape(-1)
    return torch.as_tensor(np.asarray(x, dtype=np.float32)).reshape(-1)


def _np64(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _spearman(x, y) -> tuple[float, float, int]:
    """Spearman rho / p-value over the rows where both are finite; NaN if constant or n < 2."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = int(x.size)
    if n < 2 or np.ptp(x) == 0.0 or np.ptp(y) == 0.0:
        return float("nan"), float("nan"), n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = spearmanr(x, y)
    rho, p = float(res[0]), float(res[1])
    return rho, p, n


def _whiten_np(X: np.ndarray, U: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """``[N, d] -> [N, r]`` coordinates in which the truncated Mahalanobis distance is Euclidean."""
    return (X @ U) / np.sqrt(np.maximum(lam, _LAM_FLOOR))[None, :]


# --------------------------------------------------------------------------------------
# discriminants
# --------------------------------------------------------------------------------------
@dataclass
class Discriminant:
    """A scalar function of the residual ``h`` with its gradient.

    ``fn(h) -> Tensor[()]`` and ``grad(h) -> Tensor[d]`` accept a 1-D tensor on any device and
    return CPU float32. ``value_and_grad`` (optional) returns both from one evaluation
    (used by :func:`evasion_trajectory` to save a forward pass); when ``None`` the two
    calls are made separately.
    """
    name: str
    kind: str
    fn: Callable[[torch.Tensor], torch.Tensor]
    grad: Callable[[torch.Tensor], torch.Tensor]
    meta: dict = field(default_factory=dict)
    value_and_grad: Optional[Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]] = None

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        return self.fn(h)

    def evaluate(self, h: torch.Tensor) -> tuple[float, torch.Tensor]:
        """``(g(h) as float, grad g(h) as float32 CPU Tensor[d])``."""
        if self.value_and_grad is not None:
            v, g = self.value_and_grad(h)
            return float(v), g.detach().float().cpu()
        return float(self.fn(h)), self.grad(h).detach().float().cpu()


def partial_forward(mb: ModelBundle, input_ids: torch.Tensor, site: HookSite,
                    position: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """``f(h) -> logits[V]`` (float32, last position) when the residual at ``(site, position)``
    is REPLACED by ``h``.

    The written value is ``h`` itself (cast to the model dtype/device without detaching), so
    the returned logits are differentiable with respect to ``h`` whenever ``h.requires_grad``.
    ``f`` runs under ``torch.enable_grad()``; wrap calls in ``torch.no_grad()`` when no
    gradient is needed (the frozen model builds no graph for a detached ``h`` anyway).

    For a ``block_input`` site the input of block ``L = site.layer`` (and the keyword
    arguments the model passes to its blocks: mask, rotary embeddings, ...) is captured
    once from an unperturbed forward; ``f`` then writes ``h`` at ``position`` and runs only
    blocks ``L .. n-1`` over all positions, the final LayerNorm and the LM head on the last
    position.  Blocks ``0 .. L-1`` do not depend on ``h``, so this equals the full forward
    with the replacement (float32 round-off of the last-position head aside) at a fraction
    of its cost: one block instead of twelve for the dossier site of pythia-160m, and no
    full-vocabulary logits at every position.  Other site kinds replace the residual with
    :class:`lsne.model.Injector` in ``mode="replace"`` inside a full forward.
    """
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids[None, :]
    if ids.dim() != 2 or ids.shape[0] != 1:
        raise ValueError(f"input_ids must be [1, T], got shape {tuple(ids.shape)}")
    ids = ids.to(mb.device)
    T = int(ids.shape[1])
    pos = int(position)
    pos = pos + T if pos < 0 else pos
    if not 0 <= pos < T:
        raise ValueError(f"position {position} out of range for T={T}")
    if site.site == "block_input":
        f = _partial_forward_from_block(mb, ids, int(site.layer), pos)
    else:
        f = _partial_forward_injector(mb, ids, site, pos)
    f.site, f.position, f.n_tokens = site, pos, T   # type: ignore[attr-defined]
    return f


def _check_h(h, d: int) -> torch.Tensor:
    h = torch.as_tensor(h)
    if h.dim() != 1 or h.numel() != d:
        raise ValueError(f"h must be a 1-D tensor of length {d}, got shape {tuple(h.shape)}")
    return h


def _partial_forward_injector(mb: ModelBundle, ids: torch.Tensor, site: HookSite, pos: int):
    """Full forward with the residual at ``(site, pos)`` replaced through an Injector."""
    d = mb.d_model

    def f(h: torch.Tensor) -> torch.Tensor:
        h_dev = _check_h(h, d).to(device=mb.device)
        with torch.enable_grad():
            with Injector(mb, site, h_dev, positions=[pos], mode="replace"):
                out = mb.model(input_ids=ids, use_cache=False)
            return out.logits[0, -1].float()

    return f


def _partial_forward_from_block(mb: ModelBundle, ids: torch.Tensor, layer: int, pos: int):
    """Blocks ``layer .. n-1`` + final LN + LM head (last position) on the captured input of
    block ``layer`` with position ``pos`` replaced by ``h`` (see :func:`partial_forward`)."""
    d = mb.d_model
    cap: dict = {}

    def _pre(module, args, kwargs):
        cap["hs"] = (args[0] if args else kwargs["hidden_states"]).detach()
        cap["args"] = tuple(args[1:])
        cap["kw"] = {k: v for k, v in kwargs.items() if k != "hidden_states"}

    hd = mb.layers[layer].register_forward_pre_hook(_pre, with_kwargs=True)
    try:
        with torch.no_grad():
            last_logits(mb, ids)
    finally:
        hd.remove()
    base, extra, lkw = cap["hs"], cap["args"], cap["kw"]
    blocks = mb.layers[layer:]

    def f(h: torch.Tensor) -> torch.Tensor:
        h = _check_h(h, d)
        with torch.enable_grad():
            hs = base.clone()
            hs[0, pos] = h.to(device=hs.device, dtype=hs.dtype)          # keeps the graph to h
            for block in blocks:
                hs = _first_tensor(block(hs, *extra, **lkw))
            return mb.lm_head(mb.final_ln(hs[:, -1]))[0].float()

    return f


def _autograd_triplet(scalar_of_h: Callable[[torch.Tensor], torch.Tensor]):
    """``fn``, ``grad`` and ``value_and_grad`` for a differentiable scalar of ``h``."""
    def fn(h):
        h = _vec32(h)
        with torch.no_grad():
            return scalar_of_h(h).detach().float().cpu().reshape(())

    def value_and_grad(h):
        h = _vec32(h).clone().requires_grad_(True)
        with torch.enable_grad():
            val = scalar_of_h(h)
            (g,) = torch.autograd.grad(val, h)
        return val.detach().float().cpu().reshape(()), g.detach().float().cpu()

    def grad(h):
        return value_and_grad(h)[1]

    return fn, grad, value_and_grad


def make_discriminant(mb: ModelBundle, kind: str, site: HookSite, input_ids=None, position=None,
                      w=None, b: float = 0.0, target_id=None, sae=None, feature=None) -> Discriminant:
    """Build a :class:`Discriminant` of the given ``kind`` (see module docstring).

    ``b`` is an additive offset for every kind (``g = raw + b``); it sets the threshold
    ``g = 0`` that :func:`evasion_trajectory` reports as the boundary crossing. For
    ``"probe"`` ``w`` may be a tensor or a :class:`lsne.directions.Direction`; the gradient is
    ``w``. ``"target_logit"`` and ``"entropy"`` need ``input_ids`` and ``position`` and are
    differentiated through :func:`partial_forward`; ``"entropy"`` has ``grad g = -grad H``,
    i.e. the negative of :func:`lsne.model.entropy_gradient`. ``"sae_feature"`` needs ``sae``
    (a :class:`lsne.sae.TopKSAE`) and ``feature``; its gradient is ``-W_enc[feature]`` where
    the pre-activation is positive and zero elsewhere (ReLU subgradient).
    """
    b = float(b)
    meta: dict = dict(kind=kind, site=site.key if isinstance(site, HookSite) else str(site), b=b)
    if kind == "probe":
        if w is None:
            raise ValueError("kind='probe' needs w")
        wv = _vec32(w).clone()
        if hasattr(w, "name"):
            meta["direction"] = getattr(w, "name")
        meta.update(w_norm=float(wv.norm()), d=int(wv.numel()))

        def fn(h):
            return (_vec32(h) @ wv + b).reshape(())

        def grad(h):
            return wv.clone()

        def value_and_grad(h):
            return fn(h), wv.clone()

        return Discriminant("probe", kind, fn, grad, meta, value_and_grad)

    if kind in ("target_logit", "entropy"):
        if input_ids is None or position is None:
            raise ValueError(f"kind={kind!r} needs input_ids and position")
        f = partial_forward(mb, input_ids, site, int(position))
        meta.update(position=int(f.position), n_tokens=int(f.n_tokens))     # type: ignore[attr-defined]
        if kind == "target_logit":
            if target_id is None:
                raise ValueError("kind='target_logit' needs target_id")
            tid = int(target_id)
            meta["target_id"] = tid

            def scalar(h):
                return -torch.log_softmax(f(h), -1)[tid] + b
        else:
            def scalar(h):
                logp = torch.log_softmax(f(h), -1)
                return (logp.exp() * logp).sum() + b          # -H
        fn, grad, vag = _autograd_triplet(scalar)
        return Discriminant(kind, kind, fn, grad, meta, vag)

    if kind == "sae_feature":
        if sae is None or feature is None:
            raise ValueError("kind='sae_feature' needs sae and feature")
        j = int(feature)
        W = sae.W_enc[j].detach().float().cpu().clone()
        b_enc = float(sae.b_enc[j].detach().float().cpu())
        b_dec = sae.b_dec.detach().float().cpu().clone()
        meta.update(feature=j, w_norm=float(W.norm()))

        def pre(h):
            return float((_vec32(h) - b_dec) @ W + b_enc)

        def fn(h):
            return torch.tensor(-max(pre(h), 0.0) + b, dtype=torch.float32)

        def grad(h):
            return -W.clone() if pre(h) > 0.0 else torch.zeros_like(W)

        def value_and_grad(h):
            p = pre(h)
            g = -W.clone() if p > 0.0 else torch.zeros_like(W)
            return torch.tensor(-max(p, 0.0) + b, dtype=torch.float32), g

        return Discriminant(f"sae_feature_{j}", kind, fn, grad, meta, value_and_grad)

    raise ValueError(f"unknown discriminant kind {kind!r} "
                     "(use 'probe', 'target_logit', 'entropy' or 'sae_feature')")


# --------------------------------------------------------------------------------------
# kernel density mimicry term
# --------------------------------------------------------------------------------------
class KDEMimicry:
    """Kernel density estimate of the target class (the mimicry term of Biggio et al.).

    ``H_target [n, d]`` are real activations of the target condition. ``metric="l2"`` runs the
    KDE in raw coordinates; ``metric="mahalanobis"`` whitens references and queries with
    ``z = lam_r^-1/2 * (U_r^T h)`` (``U_r [d, r]``, ``lam_r [r]`` from
    :func:`lsne.geometry.layer_covariance`) and runs the KDE in ``z``; gradients are mapped
    back to ``h``-space with the chain rule ``dz/dh = diag(lam_r^-1/2) U_r^T``.

    ``kernel="rbf"`` (default): ``k_i = exp(-||z - z_i||^2 / bw)``; ``kernel="laplacian"``:
    ``k_i = exp(-||z - z_i||_1 / bw)``. ``bandwidth`` is a float or ``"median"`` (see module
    docstring; ``median_distance`` keeps the measured median, ``bandwidth`` the ``bw`` used).
    ``k_nearest`` restricts every evaluation to the ``k`` nearest references in the kernel's
    distance (``None`` = all); the sum is normalised by ``n_eff = min(k, n)`` points.

    ``density(h)`` returns ``(1/n_eff) sum_i k_i`` as a float, ``grad(h)`` its gradient as a
    float32 ``Tensor[d]`` in ``h``-space (closed form, paper section 3.2), and
    ``density_and_grad(h)`` both from one evaluation. All arithmetic is float64.
    """

    def __init__(self, H_target: torch.Tensor, bandwidth: Union[float, str] = "median",
                 metric: str = "l2", U_r=None, lam_r=None, k_nearest: Optional[int] = 50,
                 kernel: str = "rbf", seed: int = 0, max_pairs_sample: int = 500):
        H = _np64(H_target)
        if H.ndim == 1:
            H = H[None, :]
        if H.ndim != 2 or H.shape[0] == 0:
            raise ValueError(f"H_target must be [n, d] with n >= 1, got shape {H.shape}")
        if metric not in ("l2", "mahalanobis"):
            raise ValueError(f"metric must be 'l2' or 'mahalanobis', got {metric!r}")
        if kernel not in ("rbf", "laplacian"):
            raise ValueError(f"kernel must be 'rbf' or 'laplacian', got {kernel!r}")
        self.metric, self.kernel = metric, kernel
        self.n, self.d = int(H.shape[0]), int(H.shape[1])
        self.H = torch.as_tensor(H, dtype=torch.float32)
        self._U: Optional[np.ndarray] = None
        self._s: Optional[np.ndarray] = None
        if metric == "mahalanobis":
            if U_r is None or lam_r is None:
                raise ValueError("metric='mahalanobis' needs U_r [d, r] and lam_r [r]")
            U = _np64(U_r)
            lam = _np64(lam_r).reshape(-1)
            if U.ndim != 2 or U.shape[0] != self.d or U.shape[1] != lam.size or lam.size == 0:
                raise ValueError(f"U_r must be [d={self.d}, r] with r = len(lam_r) >= 1; "
                                 f"got {U.shape} and {lam.shape}")
            self._U = U
            self._s = 1.0 / np.sqrt(np.maximum(lam, _LAM_FLOOR))
            self.Z = (H @ U) * self._s[None, :]
        else:
            self.Z = H
        self.r = int(self.Z.shape[1])
        if k_nearest is not None and int(k_nearest) < 1:
            raise ValueError("k_nearest must be >= 1 or None")
        self.k_nearest = None if k_nearest is None else int(min(int(k_nearest), self.n))
        self.n_eff = self.n if self.k_nearest is None else self.k_nearest
        self.median_distance = float("nan")
        if isinstance(bandwidth, str):
            if bandwidth != "median":
                raise ValueError(f"bandwidth must be a float or 'median', got {bandwidth!r}")
            if self.n < 2:
                raise ValueError("bandwidth='median' needs at least two references")
            med = self._median_pairwise(seed, int(max_pairs_sample))
            if not np.isfinite(med) or med <= 0.0:
                raise ValueError("median pairwise distance of the references is zero / non-finite; "
                                 "pass a numeric bandwidth")
            self.median_distance = med
            bw = med ** 2 if kernel == "rbf" else med
        else:
            bw = float(bandwidth)
        if not np.isfinite(bw) or bw <= 0.0:
            raise ValueError(f"bandwidth must be positive, got {bw}")
        self.bandwidth = float(bw)

    # ---- coordinates ---------------------------------------------------------------
    def to_metric(self, h) -> np.ndarray:
        """``h [d] -> z [r]`` (float64): whitened coordinates for ``metric="mahalanobis"``,
        the raw vector for ``metric="l2"``."""
        z = _np64(h).reshape(-1)
        if z.size != self.d:
            raise ValueError(f"expected a vector of length {self.d}, got {z.size}")
        if self._U is not None:
            z = (z @ self._U) * self._s
        return z

    def metric_distance(self, a, b) -> float:
        """l2 distance between two ``h``-vectors in the metric coordinates (``sqrt`` of the
        truncated Mahalanobis form for ``metric="mahalanobis"``)."""
        return float(np.linalg.norm(self.to_metric(a) - self.to_metric(b)))

    def _dist(self, diff: np.ndarray) -> np.ndarray:
        """Kernel distance of every row of ``diff [m, r]``: l2 (RBF) or l1 (Laplacian)."""
        if self.kernel == "rbf":
            return np.sqrt(np.einsum("ij,ij->i", diff, diff))
        return np.abs(diff).sum(1)

    def _median_pairwise(self, seed: int, max_n: int) -> float:
        Z = self.Z
        if Z.shape[0] > max_n:
            rng = np.random.default_rng(int(seed))
            Z = Z[rng.choice(Z.shape[0], size=max_n, replace=False)]
        m = Z.shape[0]
        iu = np.triu_indices(m, k=1)
        if self.kernel == "rbf":
            sq = (Z * Z).sum(1)
            D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
            D = np.sqrt(np.maximum(D2[iu], 0.0))
        else:
            D = np.abs(Z[:, None, :] - Z[None, :, :]).sum(-1)[iu]
        return float(np.median(D))

    # ---- evaluation ----------------------------------------------------------------
    def _terms(self, h) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(diff [m, r], dist [m], kernel [m])`` over the references used for ``h``."""
        z = self.to_metric(h)
        diff = z[None, :] - self.Z                       # z - z_i
        dist = self._dist(diff)
        if self.k_nearest is not None and self.k_nearest < self.n:
            idx = np.argpartition(dist, self.k_nearest - 1)[:self.k_nearest]
            diff, dist = diff[idx], dist[idx]
        if self.kernel == "rbf":
            kern = np.exp(-(dist * dist) / self.bandwidth)
        else:
            kern = np.exp(-dist / self.bandwidth)
        return diff, dist, kern

    def nearest(self, h, k: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
        """Indices and kernel distances of the ``k`` nearest references (default ``k_nearest``,
        or all), sorted ascending."""
        z = self.to_metric(h)
        dist = self._dist(z[None, :] - self.Z)
        k = self.n if k is None and self.k_nearest is None else int(min(k or self.k_nearest, self.n))
        idx = np.argsort(dist, kind="stable")[:k]
        return idx, dist[idx]

    def density(self, h) -> float:
        _, _, kern = self._terms(h)
        return float(kern.mean())

    def density_and_grad(self, h) -> tuple[float, torch.Tensor]:
        diff, dist, kern = self._terms(h)
        m = diff.shape[0]
        if self.kernel == "rbf":
            grad_z = -(2.0 / (m * self.bandwidth)) * (kern[:, None] * diff).sum(0)
        else:
            grad_z = -(1.0 / (m * self.bandwidth)) * (kern[:, None] * np.sign(diff)).sum(0)
        grad_h = self._U @ (self._s * grad_z) if self._U is not None else grad_z
        return float(kern.mean()), torch.as_tensor(grad_h, dtype=torch.float32)

    def grad(self, h) -> torch.Tensor:
        return self.density_and_grad(h)[1]

    def __repr__(self) -> str:
        return (f"KDEMimicry(n={self.n}, d={self.d}, metric={self.metric!r}, kernel={self.kernel!r}, "
                f"bandwidth={self.bandwidth:.4g}, k_nearest={self.k_nearest})")


def lambda_scale(g_values_range: float, n: int, bw: float) -> float:
    """The paper's scale for ``lam``: ``lam = g_range * n * bw``, so that ``lam / (n * bw)``
    equals the range of the discriminant ``g`` (Biggio et al. section 3.2: the mimicry
    gradient carries the prefactor ``lam / (n h)``). Pass ``n = kde.n_eff`` (the number of
    kernel terms, i.e. ``k_nearest`` when active) and ``bw = kde.bandwidth``; the relative
    sweep ``lam_rel in {0, 0.1, 0.3, 1, 3, 10, 30}`` multiplies this value."""
    return float(g_values_range) * float(n) * float(bw)


# --------------------------------------------------------------------------------------
# Algorithm 1
# --------------------------------------------------------------------------------------
@dataclass
class Trajectory:
    """One run of Algorithm 1. ``points[0]`` is the start; ``points[-1]`` the returned
    ``x* = x_m`` (the last evaluated point, which by the stopping rule did not decrease
    ``F`` by more than ``eps_stop`` relative to ``points[-2]``). ``reason`` is one of
    ``"no_decrease"`` (converged), ``"max_steps"``, ``"flat"`` (zero gradient)."""
    points: torch.Tensor              # [m+1, d] float32
    g: np.ndarray                     # [m+1]
    density: np.ndarray               # [m+1]
    F: np.ndarray                     # [m+1]
    dist_from_start: np.ndarray       # [m+1]
    lam: float
    converged: bool
    steps_to_cross: Optional[int]
    reason: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.points.shape[0] - 1)

    @property
    def endpoint(self) -> torch.Tensor:
        return self.points[-1]

    @property
    def best_index(self) -> int:
        """Index of the point with the smallest ``F``."""
        return int(np.argmin(self.F))


def _as_discriminant(g) -> Discriminant:
    if isinstance(g, Discriminant):
        return g
    raise TypeError("g must be a Discriminant (see make_discriminant)")


def evasion_trajectory(h0: torch.Tensor, g: Discriminant, kde: Optional[KDEMimicry], lam: float,
                       step: float, d_max: float, n_steps: int = 200, eps_stop: float = 1e-5,
                       metric: str = "l2") -> Trajectory:
    """Algorithm 1 of Biggio et al. in activation space.

    ``F(h) = g(h) - lam * kde.density(h)``. Each step moves ``h`` by ``step`` along the unit
    vector of ``grad g(h) - lam * grad density(h)`` (descent), then projects onto the ball
    ``d(h, h0) <= d_max`` (radial projection, which for the l2 metric is the Euclidean
    projection; ``metric="mahalanobis"`` measures ``d`` in the whitened coordinates of ``kde``,
    which must then use that metric). The loop stops when ``F(h_m) - F(h_{m-1}) > -eps_stop``
    (``converged=True``, reason ``"no_decrease"``), after ``n_steps`` (``"max_steps"``) or when
    the gradient norm is below 1e-12 (``"flat"``; ``converged=False``). Every evaluated point
    is recorded; ``steps_to_cross`` is the first ``m`` with ``g(h_m) < 0`` (``None`` if never).
    With ``kde=None`` the density is 0 (``lam`` must then be 0).
    """
    g = _as_discriminant(g)
    lam = float(lam)
    step, d_max = float(step), float(d_max)
    if step <= 0.0 or d_max < 0.0:
        raise ValueError("step must be > 0 and d_max >= 0")
    if kde is None and lam != 0.0:
        raise ValueError("lam != 0 needs a KDEMimicry")
    if metric not in ("l2", "mahalanobis"):
        raise ValueError(f"metric must be 'l2' or 'mahalanobis', got {metric!r}")
    if metric == "mahalanobis" and (kde is None or kde.metric != "mahalanobis"):
        raise ValueError("metric='mahalanobis' needs a KDEMimicry built with metric='mahalanobis'")

    x0 = _np64(h0).reshape(-1)
    d = x0.size

    def dist(x):
        if metric == "l2":
            return float(np.linalg.norm(x - x0))
        return kde.metric_distance(x, x0)

    def evaluate(x):
        val, grad_g = g.evaluate(torch.as_tensor(x, dtype=torch.float32))
        grad_g = _np64(grad_g).reshape(-1)
        if kde is None:
            dens, grad_d = 0.0, np.zeros(d)
        else:
            dens, grad_d = kde.density_and_grad(x)
            grad_d = _np64(grad_d).reshape(-1)
        return val, dens, val - lam * dens, grad_g - lam * grad_d

    pts, gs, ds, Fs, dists = [x0.copy()], [], [], [], [0.0]
    val, dens, F, direction = evaluate(x0)
    gs.append(val); ds.append(dens); Fs.append(F)
    converged, reason = False, "max_steps"
    for _m in range(int(n_steps)):
        nrm = float(np.linalg.norm(direction))
        if not np.isfinite(nrm) or nrm < _FLAT_NORM:
            reason = "flat" if np.isfinite(nrm) else "non_finite"
            break
        x = pts[-1] - step * direction / nrm
        dd = dist(x)
        if dd > d_max:
            x = x0 + (x - x0) * (d_max / dd if dd > 0 else 0.0)
            dd = dist(x)
        val, dens, F, direction = evaluate(x)
        pts.append(x); gs.append(val); ds.append(dens); Fs.append(F); dists.append(dd)
        if not np.isfinite(F):
            reason = "non_finite"
            break
        if F - Fs[-2] > -eps_stop:
            converged, reason = True, "no_decrease"
            break
    g_arr = np.asarray(gs, dtype=np.float64)
    below = np.nonzero(g_arr < 0.0)[0]
    steps_to_cross = int(below[0]) if below.size else None
    return Trajectory(points=torch.as_tensor(np.stack(pts, 0), dtype=torch.float32),
                      g=g_arr, density=np.asarray(ds, dtype=np.float64),
                      F=np.asarray(Fs, dtype=np.float64),
                      dist_from_start=np.asarray(dists, dtype=np.float64),
                      lam=lam, converged=converged, steps_to_cross=steps_to_cross, reason=reason,
                      meta=dict(step=step, d_max=d_max, n_steps_max=int(n_steps), eps_stop=float(eps_stop),
                                metric=metric, discriminant=g.name, kind=g.kind))


def lambda_sweep(h0: torch.Tensor, g: Discriminant, kde: Optional[KDEMimicry],
                 lambdas: Sequence[float], **kw) -> dict[float, Trajectory]:
    """One :func:`evasion_trajectory` per value of ``lambdas`` (keyword arguments ``step``,
    ``d_max``, ``n_steps``, ``eps_stop``, ``metric`` are passed through). Keys are the floats
    of ``lambdas`` in the given order."""
    out: dict[float, Trajectory] = {}
    for lam in lambdas:
        lam = float(lam)
        out[lam] = evasion_trajectory(h0, g, kde, lam, **kw)
    return out


# --------------------------------------------------------------------------------------
# measurements along / at the end of a path
# --------------------------------------------------------------------------------------
def trajectory_geometry(traj: Trajectory) -> dict:
    """Path descriptors: ``path_length = sum ||h_m - h_{m-1}||``, ``net_displacement =
    ||h_m - h_0||``, ``tortuosity = path_length / net_displacement`` (NaN for zero
    displacement), ``mean_turning_angle_deg`` (mean angle between consecutive non-zero step
    vectors; NaN with fewer than two), ``n_steps``, ``steps_to_cross``, ``final_g``,
    ``final_density``, ``final_F``, ``dist_from_start`` (final, in the trajectory's metric),
    plus ``lam``, ``converged``, ``reason``, ``initial_g``, ``g_drop``."""
    P = _np64(traj.points)
    steps = np.diff(P, axis=0) if P.shape[0] > 1 else np.zeros((0, P.shape[1]))
    lens = np.linalg.norm(steps, axis=1) if steps.shape[0] else np.zeros(0)
    path_length = float(lens.sum())
    net = float(np.linalg.norm(P[-1] - P[0]))
    tort = path_length / net if net > 0.0 else float("nan")
    angles = []
    for a, b, la, lb in zip(steps[:-1], steps[1:], lens[:-1], lens[1:]):
        if la > 0.0 and lb > 0.0:
            c = float(np.clip(a @ b / (la * lb), -1.0, 1.0))
            angles.append(math.degrees(math.acos(c)))
    g0, gm = float(traj.g[0]), float(traj.g[-1])
    return dict(path_length=path_length, net_displacement=net, tortuosity=float(tort),
                mean_turning_angle_deg=float(np.mean(angles)) if angles else float("nan"),
                n_steps=int(traj.n_steps), steps_to_cross=traj.steps_to_cross,
                final_g=gm, final_density=float(traj.density[-1]), final_F=float(traj.F[-1]),
                dist_from_start=float(traj.dist_from_start[-1]),
                lam=float(traj.lam), converged=bool(traj.converged), reason=traj.reason,
                initial_g=g0, g_drop=g0 - gm)


def _sae_active(sae, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(top_acts, top_idx, x_hat)`` of the TopK code for ``X [N, d]`` (float32 CPU)."""
    with torch.no_grad():
        top_acts, top_idx, _ = sae.encode(X)
        x_hat = sae.decode(top_acts, top_idx).float().cpu()
    return top_acts.float().cpu(), top_idx.cpu(), x_hat


def manifold_stats(points: torch.Tensor, X_ref: torch.Tensor, U_r=None, lam_r=None, sae=None,
                   k: int = 5) -> pd.DataFrame:
    """Per-point measures of "data support" relative to a reference cloud ``X_ref [N, d]``.

    Columns (one row per point, ``index`` = row of ``points``):

    * ``mahal_to_ref_mean``  truncated Mahalanobis distance to the reference mean
      (:func:`lsne.geometry.mahalanobis` with ``U_r, lam_r``); without a covariance the
      Euclidean distance is reported here (``metric`` column says which);
    * ``euclid_to_ref_mean`` always the Euclidean distance;
    * ``knn_mean_dist``      mean Euclidean distance to the ``k`` nearest references;
    * ``knn_mean_mahal``     the same in the whitened coordinates (NaN without covariance);
    * ``local_log_volume``   :func:`lsne.geometry.log_volume` of the cloud {point} + its ``k``
      nearest references (Euclidean neighbours);
    * ``sae_fvu``            ``||x - x_hat||^2`` of the point's TopK-SAE reconstruction divided
      by the per-sample variance of the reference cloud ``sum_i ||x_i - mean||^2 / N``
      (NaN without ``sae``);
    * ``n_shared_features_with_ref`` number of the point's active TopK features (positive
      activation) that are active in at least one of its ``k`` nearest references;
    * ``n_active_features``  number of positive TopK activations of the point.
    """
    P = _np64(points)
    if P.ndim == 1:
        P = P[None, :]
    X = _np64(X_ref)
    if X.ndim == 1:
        X = X[None, :]
    N, d = X.shape
    if P.shape[1] != d:
        raise ValueError(f"points have d={P.shape[1]} but X_ref has d={d}")
    k_eff = int(min(int(k), N))
    mu = X.mean(0) if N else np.full(d, np.nan)
    have_cov = U_r is not None and lam_r is not None
    if have_cov:
        U, lam = _np64(U_r), _np64(lam_r).reshape(-1)
        Zx = _whiten_np(X, U, lam)
    ref_var = float(((X - mu) ** 2).sum() / N) if N else float("nan")
    if sae is not None:
        pa, pi, x_hat = _sae_active(sae, torch.as_tensor(P, dtype=torch.float32))
        ra, ri, _ = _sae_active(sae, torch.as_tensor(X, dtype=torch.float32)) if N else (None, None, None)
    rows = []
    for i, p in enumerate(P):
        eu = float(np.linalg.norm(p - mu))
        row = dict(index=i, mahal_to_ref_mean=eu, euclid_to_ref_mean=eu,
                   metric="euclid", knn_mean_dist=float("nan"), knn_mean_mahal=float("nan"),
                   local_log_volume=float("nan"), sae_fvu=float("nan"),
                   n_shared_features_with_ref=np.nan, n_active_features=np.nan)
        if have_cov:
            row["mahal_to_ref_mean"] = mahalanobis(p, mu, U, lam)
            row["metric"] = "mahalanobis"
        nn_idx = np.zeros(0, dtype=int)
        if k_eff > 0:
            dist = np.linalg.norm(X - p[None, :], axis=1)
            nn_idx = np.argpartition(dist, k_eff - 1)[:k_eff]
            row["knn_mean_dist"] = float(dist[nn_idx].mean())
            row["local_log_volume"] = log_volume(np.vstack([p[None, :], X[nn_idx]]))
            if have_cov:
                zp = _whiten_np(p[None, :], U, lam)[0]
                dz = np.linalg.norm(Zx - zp[None, :], axis=1)
                row["knn_mean_mahal"] = float(np.sort(dz)[:k_eff].mean())
        if sae is not None:
            row["sae_fvu"] = float(((P[i] - _np64(x_hat[i])) ** 2).sum() / ref_var) if ref_var > 0 else float("nan")
            own = set(int(j) for a, j in zip(pa[i].tolist(), pi[i].tolist()) if a > 0.0)
            row["n_active_features"] = len(own)
            if nn_idx.size:
                ref_feats: set[int] = set()
                for r_ in nn_idx:
                    ref_feats.update(int(j) for a, j in zip(ra[r_].tolist(), ri[r_].tolist()) if a > 0.0)
                row["n_shared_features_with_ref"] = len(own & ref_feats)
        rows.append(row)
    cols = ["index", "mahal_to_ref_mean", "euclid_to_ref_mean", "metric", "knn_mean_dist",
            "knn_mean_mahal", "local_log_volume", "sae_fvu", "n_shared_features_with_ref",
            "n_active_features"]
    return pd.DataFrame(rows, columns=cols)


def endpoint_effects(mb: ModelBundle, input_ids: torch.Tensor, site: HookSite, position: int,
                     endpoints: Mapping[float, torch.Tensor], target_id: int,
                     freq_ids=None) -> pd.DataFrame:
    """Next-token effect of replacing the residual at ``(site, position)`` by each endpoint.

    One row per entry of ``endpoints`` (``lam -> Tensor[d]``): ``lam``, ``logp_target``,
    ``p_target``, ``entropy``, ``p_freq`` (mass on ``freq_ids``; NaN if None), ``top1_id`` and
    ``B = logp_target - logp_target_unperturbed`` (the unperturbed run is made once; its
    values are in ``df.attrs``: ``logp_target_unperturbed``, ``entropy_unperturbed``,
    ``p_freq_unperturbed``)."""
    f = partial_forward(mb, input_ids, site, position)
    ids = torch.as_tensor(input_ids, dtype=torch.long)
    ids = ids[None, :] if ids.dim() == 1 else ids
    tid = int(target_id)
    fi = None if freq_ids is None else torch.as_tensor(freq_ids, dtype=torch.long).reshape(-1)
    with torch.no_grad():
        base_logits = last_logits(mb, ids)[0]
        base = next_token_stats(base_logits[None, :], tid, fi)
        base_logp = float(base["logp_target"][0])
        rows = []
        for lam, h in endpoints.items():
            st = next_token_stats(f(_vec32(h))[None, :], tid, fi)
            lp = float(st["logp_target"][0])
            rows.append(dict(lam=float(lam), logp_target=lp, p_target=float(st["p_target"][0]),
                             entropy=float(st["entropy"][0]),
                             p_freq=float(st["p_freq"][0]) if fi is not None else float("nan"),
                             top1_id=int(st["top1_id"][0]), B=lp - base_logp))
    df = pd.DataFrame(rows, columns=["lam", "logp_target", "p_target", "entropy", "p_freq", "top1_id", "B"])
    df.attrs.update(logp_target_unperturbed=base_logp, entropy_unperturbed=float(base["entropy"][0]),
                    p_freq_unperturbed=float(base["p_freq"][0]) if fi is not None else float("nan"),
                    target_id=tid, site=site.key, position=int(f.position))   # type: ignore[attr-defined]
    return df


def feature_lambda_correlation(endpoints: Mapping[float, torch.Tensor], sae,
                               topk_only: bool = False) -> pd.DataFrame:
    """Spearman correlation of SAE feature activations at the endpoints with ``lam``.

    Activations are ``relu(pre)`` of ``sae.encode`` (every latent; with ``topk_only=True``
    only the TopK-selected code, zeros elsewhere). One row per feature that is active
    (``> 0``) at any endpoint: ``feature``, ``rho``, ``pvalue`` (NaN if the activation is
    constant across endpoints), ``act_at_min_lambda``, ``act_at_max_lambda``,
    ``active_only_lambda0`` (active at ``lam == 0`` and at no ``lam > 0`` endpoint),
    ``active_only_lambda_pos`` (active at some ``lam > 0`` endpoint and not at ``lam == 0``),
    ``n_active``, ``max_act``. Both booleans are False when no ``lam == 0`` endpoint exists."""
    lams = np.asarray([float(l) for l in endpoints.keys()], dtype=np.float64)
    cols = ["feature", "rho", "pvalue", "act_at_min_lambda", "act_at_max_lambda",
            "active_only_lambda0", "active_only_lambda_pos", "n_active", "max_act"]
    if lams.size == 0:
        return pd.DataFrame(columns=cols)
    X = torch.stack([_vec32(h) for h in endpoints.values()], 0)
    with torch.no_grad():
        top_acts, top_idx, pre = sae.encode(X)
        if topk_only:
            A = torch.zeros_like(pre)
            A.scatter_(-1, top_idx, top_acts.to(A.dtype))
        else:
            A = torch.relu(pre)
    A = A.float().cpu().numpy().astype(np.float64)              # [n_lam, F]
    active = A > 0.0
    feats = np.nonzero(active.any(0))[0]
    i_min, i_max = int(np.argmin(lams)), int(np.argmax(lams))
    is0 = lams == 0.0
    rows = []
    for j in feats:
        rho, p, _ = _spearman(lams, A[:, j])
        act0 = bool(active[is0, j].any()) if is0.any() else False
        actpos = bool(active[~is0, j].any())
        rows.append(dict(feature=int(j), rho=rho, pvalue=p,
                         act_at_min_lambda=float(A[i_min, j]), act_at_max_lambda=float(A[i_max, j]),
                         active_only_lambda0=bool(is0.any() and act0 and not actpos),
                         active_only_lambda_pos=bool(is0.any() and actpos and not act0),
                         n_active=int(active[:, j].sum()), max_act=float(A[:, j].max())))
    return pd.DataFrame(rows, columns=cols)


def geometry_lambda_correlation(rows: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation of every numeric column of ``rows`` (bools included) with the
    column ``lam``: one row per quantity with ``quantity, rho, pvalue, n`` (``n`` = rows where
    both values are finite; ``rho`` NaN if constant or ``n < 2``)."""
    if "lam" not in rows.columns:
        raise ValueError("rows needs a 'lam' column")
    lam = rows["lam"].to_numpy(dtype=np.float64)
    out = []
    for c in rows.columns:
        if c == "lam" or not pd.api.types.is_numeric_dtype(rows[c]):
            continue
        vals = pd.to_numeric(rows[c], errors="coerce").to_numpy(dtype=np.float64)
        rho, p, n = _spearman(lam, vals)
        out.append(dict(quantity=str(c), rho=rho, pvalue=p, n=int(n)))
    return pd.DataFrame(out, columns=["quantity", "rho", "pvalue", "n"])
