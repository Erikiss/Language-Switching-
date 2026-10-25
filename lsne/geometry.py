"""Local-geometry measures of the residual stream under controlled language switching
(DESIGN.md section 4.4; "Meaning Contracts", Godoy Ortiz 2026, arXiv:2610.30715).

All measures operate on *clouds* of residual states: a matrix ``G`` of shape ``[n, d]``
holding ``n`` realizations (stimulus variants) of one condition, pooled per sequence
with a *readout* (mean over positions or last position). Conventions, fixed here so
that every stage reports the same numbers:

* **Covariance normalisation**: sample covariance with ``1/(n-1)`` (``np.cov`` default)
  everywhere (cloud eigenvalues, layer covariance). For comparisons at equal ``n`` the
  constant is irrelevant; it is stated so that absolute log-volumes are reproducible.
* **Log-volume** (paper, eq. for the local realization cloud): ``sum_i log max(lambda_i, eps)``
  over *all* ``min(n, d)`` eigenvalues of the covariance of the centred cloud, obtained
  from an economy SVD (one of them is numerically zero because of centring and is
  clipped at ``eps``). A cloud needs ``n >= 2``; otherwise ``NaN``.
* **Participation ratio** ``(sum lambda)^2 / sum lambda^2`` on the same eigenvalues
  (ordinal indicator, as in the paper).
* **Truncated Mahalanobis** (paper eq. 1): ``d_M = sqrt((x-y)^T U_r Lambda_r^{-1} U_r^T (x-y))``
  with ``(U_r, Lambda_r)`` the rank-``r`` eigendecomposition of the covariance of *all*
  pooled states of a layer (``r = min(r, N-1, d)``; numerically-zero eigenvalues are
  dropped). :func:`mahalanobis` returns the distance (not its square). The form is only
  informative for ``N >> r``: ``N`` points whitened by their own full-rank covariance
  (``r = N-1``) form a regular simplex with all pairwise distances equal.
* **Deflation**: ``X^(-k) = X~ - (X~ V_k^T) V_k`` with ``V_k`` the top-``k`` right singular
  vectors of the centred pooled states ``X~``; ``k = 0`` is the centred cloud.
  All measures of a row with deflation level ``k`` are computed on the ``k``-deflated
  representation (log-volume, PR *and* both displacements); since ``V_k`` and ``U_r`` come
  from the same pooled matrix, the Mahalanobis displacement then simply ignores the
  deflated eigen-directions.

Stimulus "variants" are duck-typed: anything with ``.ids`` (token ids), ``.annot`` (dict
with per-token lists ``lang``/``unit``/``klass`` and ints ``tail_start``/``last_switch_pos``)
and the condition attributes ``family_id``, ``formulation``, ``K``, ``C``.
"""
from __future__ import annotations

import itertools
import warnings
from typing import Callable, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from .config import HookSite
from .model import ModelBundle, run_with_cache

__all__ = [
    "pool_states", "deflate", "layer_covariance", "mahalanobis", "log_volume",
    "participation_ratio", "displacement", "group_geometry", "pair_ordering",
    "per_position_log_volume", "align_tail_states", "enumerate_unit_masks",
    "count_switches", "unit_mask_to_segment_mask", "segment_mask_fn",
    "realization_groups", "geometry_table", "group_label", "variant_ids",
    "PAIR_CATEGORIES", "SEGMENT_NAMES",
]

ArrayLike = Union[np.ndarray, torch.Tensor, Sequence]
SEGMENT_NAMES = ("subj", "pred", "adj")
PAIR_CATEGORIES = ("exact_translation", "same_meaning_same_lang",
                   "same_meaning_cross_lang", "other_family")
_LAM_FLOOR = 1e-300


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _np(x: ArrayLike) -> np.ndarray:
    """Any array-like (torch on any device, numpy, list) -> float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _as_cloud(G: ArrayLike) -> np.ndarray:
    G = _np(G)
    if G.ndim == 1:
        G = G[None, :]
    if G.ndim != 2:
        raise ValueError(f"cloud must be [n, d], got shape {G.shape}")
    return G


def _cloud_eigenvalues(G: np.ndarray) -> np.ndarray:
    """Eigenvalues (descending) of the ``1/(n-1)`` covariance of the centred cloud,
    ``min(n, d)`` of them, via an economy SVD."""
    n = G.shape[0]
    Gc = G - G.mean(0, keepdims=True)
    s = np.linalg.svd(Gc, compute_uv=False)
    return s ** 2 / max(n - 1, 1)


def variant_ids(v) -> torch.Tensor:
    """Token ids of a duck-typed variant as a 1-D ``LongTensor``."""
    ids = getattr(v, "ids") if not isinstance(v, Mapping) else v["ids"]
    if isinstance(ids, torch.Tensor):
        return ids.detach().long().reshape(-1).cpu()
    return torch.as_tensor(list(ids), dtype=torch.long)


def _attr(v, name, default=None):
    if isinstance(v, Mapping):
        return v.get(name, default)
    return getattr(v, name, default)


def group_label(key) -> str:
    """Human-readable label for a group key ``(family_id, K, C)`` (or any key)."""
    if isinstance(key, tuple) and len(key) == 3:
        fam, K, C = key
        return f"{fam}|K{K}|{C}"
    return str(key)


# --------------------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------------------
def pool_states(resid: ArrayLike, readout: str, positions: Optional[Sequence[int]] = None) -> np.ndarray:
    """Pool a residual tensor ``[B, T, d]`` (or ``[T, d]`` -> ``B = 1``) into ``[B, d]``.

    ``readout="mean"``: mean over ``positions`` (default: all positions);
    ``readout="last"``: the state at the last position (the largest entry of
    ``positions`` if given). Negative positions count from the end.
    """
    H = _np(resid)
    if H.ndim == 2:
        H = H[None]
    if H.ndim != 3:
        raise ValueError(f"resid must be [B,T,d] or [T,d], got shape {H.shape}")
    B, T, d = H.shape
    if positions is None:
        idx = np.arange(T)
    else:
        idx = np.asarray(list(positions), dtype=int)
        idx = np.where(idx < 0, idx + T, idx)
        if idx.size == 0:
            raise ValueError("positions must not be empty")
    if readout == "mean":
        return H[:, idx, :].mean(1)
    if readout == "last":
        return H[:, int(idx.max()), :].copy()
    raise ValueError(f"unknown readout {readout!r} (use 'mean' or 'last')")


# --------------------------------------------------------------------------------------
# deflation / covariance / distances
# --------------------------------------------------------------------------------------
def deflate(X: ArrayLike, k: int, basis: Optional[ArrayLike] = None) -> tuple[np.ndarray, np.ndarray]:
    """Centre ``X [N, d]`` and remove its projection onto the top-``k`` right singular
    vectors (fit on the centred ``X`` if ``basis`` is None, else the given rows).

    Returns ``(X_deflated [N, d], basis [k', d])`` with ``k' = min(k, rank)`` orthonormal
    rows. ``k = 0`` returns the centred ``X`` and an empty ``[0, d]`` basis.
    """
    X = _as_cloud(X)
    N, d = X.shape
    Xc = X - X.mean(0, keepdims=True)
    k = int(k)
    if k <= 0:
        return Xc, np.zeros((0, d))
    if basis is None:
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        k_eff = min(k, int((s > s[0] * 1e-12).sum()) if s.size and s[0] > 0 else 0)
        V = Vt[:k_eff]
    else:
        V = _np(basis).reshape(-1, d)[:k]
        if V.shape[0]:
            gram = V @ V.T
            if not np.allclose(gram, np.eye(V.shape[0]), atol=1e-6):
                Q, _ = np.linalg.qr(V.T)
                V = Q.T
    if V.shape[0] == 0:
        return Xc, V
    return Xc - (Xc @ V.T) @ V, V


def layer_covariance(X: ArrayLike, r: int, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Rank-``r`` eigendecomposition of the ``1/(N-1)`` covariance of all pooled states
    ``X [N, d]`` of a layer: returns ``(U_r [d, r], lam_r [r])`` (descending).

    ``r`` is truncated to ``min(r, N-1, d)`` and additionally to the eigenvalues above
    ``tol * lam_max`` (numerically-zero directions carry no information and would blow
    up the Mahalanobis form).
    """
    X = _as_cloud(X)
    N, d = X.shape
    if N < 2:
        raise ValueError("layer_covariance needs at least two states")
    Xc = X - X.mean(0, keepdims=True)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = s ** 2 / (N - 1)
    r_eff = int(min(int(r), N - 1, d))
    if lam.size and lam[0] > 0 and tol > 0:
        r_eff = min(r_eff, int((lam > tol * lam[0]).sum()))
    else:
        r_eff = 0
    r_eff = max(r_eff, 0)
    return Vt[:r_eff].T.copy(), lam[:r_eff].copy()


def mahalanobis(x: ArrayLike, y: ArrayLike, U_r: ArrayLike, lam_r: ArrayLike) -> float:
    """Truncated Mahalanobis distance (paper eq. 1, square root taken):
    ``sqrt((x-y)^T U_r diag(1/lam_r) U_r^T (x-y))``. ``NaN`` if the rank is zero."""
    x = _np(x).reshape(-1)
    y = _np(y).reshape(-1)
    U = _np(U_r)
    lam = _np(lam_r).reshape(-1)
    if U.ndim != 2 or U.shape[1] != lam.size:
        raise ValueError(f"U_r must be [d, r] with r = len(lam_r); got {U.shape} and {lam.shape}")
    if lam.size == 0:
        return float("nan")
    z = U.T @ (x - y)
    return float(np.sqrt(np.sum(z * z / np.maximum(lam, _LAM_FLOOR))))


def _whiten(X: np.ndarray, U_r: np.ndarray, lam_r: np.ndarray) -> np.ndarray:
    """Coordinates in which the truncated Mahalanobis distance is Euclidean: ``[N, r]``."""
    return (X @ U_r) / np.sqrt(np.maximum(lam_r, _LAM_FLOOR))[None, :]


def log_volume(G: ArrayLike, eps: float = 1e-8) -> float:
    """``sum_i log max(lambda_i, eps)`` over all ``min(n, d)`` covariance eigenvalues of the
    centred cloud ``G [n, d]`` (economy SVD; one eigenvalue is ~0 due to centring).
    ``NaN`` for ``n < 2``."""
    G = _as_cloud(G)
    if G.shape[0] < 2:
        return float("nan")
    lam = _cloud_eigenvalues(G)
    return float(np.sum(np.log(np.maximum(lam, eps))))


def participation_ratio(G: ArrayLike) -> float:
    """``(sum lam)^2 / sum lam^2`` of the covariance eigenvalues of the centred cloud.
    ``NaN`` for ``n < 2`` or a degenerate (zero-variance) cloud."""
    G = _as_cloud(G)
    if G.shape[0] < 2:
        return float("nan")
    lam = _cloud_eigenvalues(G)
    s2 = float(np.sum(lam ** 2))
    if s2 <= 0.0:
        return float("nan")
    return float(np.sum(lam) ** 2 / s2)


def displacement(mean_a: ArrayLike, mean_b: ArrayLike, U_r: Optional[ArrayLike] = None,
                 lam_r: Optional[ArrayLike] = None) -> dict:
    """Euclidean and (if a layer covariance is given) Mahalanobis distance between two
    condition means. Returns ``dict(euclid=float, mahal=float)`` (``mahal`` NaN without
    covariance)."""
    a = _np(mean_a).reshape(-1)
    b = _np(mean_b).reshape(-1)
    out = dict(euclid=float(np.linalg.norm(a - b)), mahal=float("nan"))
    if U_r is not None and lam_r is not None:
        out["mahal"] = mahalanobis(a, b, U_r, lam_r)
    return out


# --------------------------------------------------------------------------------------
# group geometry
# --------------------------------------------------------------------------------------
def _resolve_ref(ref_group, keys: list) -> dict:
    """Map every group key to its reference key (a key, a dict key->ref, or a callable)."""
    if ref_group is None:
        return {k: keys[0] for k in keys}
    if callable(ref_group) and not isinstance(ref_group, (str, tuple)):
        out = {k: ref_group(k) for k in keys}
    elif isinstance(ref_group, Mapping):
        out = {k: ref_group.get(k, keys[0]) for k in keys}
    else:
        out = {k: ref_group for k in keys}
    for k, ref in out.items():
        if ref not in keys:
            raise KeyError(f"reference group {ref!r} for {k!r} is not among the groups")
    return out


def group_geometry(groups: Mapping, r: int = 128, deflate_k: Sequence[int] = (0, 5),
                   ref_group=None) -> pd.DataFrame:
    """Local geometry of several realization clouds sharing one layer.

    ``groups`` maps a key to a cloud ``[n_g, d]``. The layer covariance (rank ``r``) and
    the deflation bases are fit on the pooled matrix of *all* groups; each group is then
    measured at every deflation level ``k`` in ``deflate_k`` on the ``k``-deflated
    representation (see module docstring). ``ref_group`` is the key whose mean the
    displacements are measured from (default: the first group); it may also be a dict
    ``key -> ref_key`` or a callable ``key -> ref_key`` (used by :func:`geometry_table`
    for per-family references).

    Columns: group, k, n, log_volume, participation_ratio, disp_euclid, disp_mahal,
    ref_group, r (effective rank).
    """
    keys = list(groups.keys())
    if not keys:
        return pd.DataFrame(columns=["group", "k", "n", "log_volume", "participation_ratio",
                                     "disp_euclid", "disp_mahal", "ref_group", "r"])
    arrays = {}
    for key in keys:
        A = _np(groups[key])
        A = A.reshape(A.shape[0], -1) if A.ndim >= 2 else A.reshape(1, -1)
        arrays[key] = A
    X = np.concatenate([arrays[k] for k in keys], 0)
    if X.shape[0] < 2:
        raise ValueError("group_geometry needs at least two pooled states in total")
    slices, start = {}, 0
    for key in keys:
        n = arrays[key].shape[0]
        slices[key] = slice(start, start + n)
        start += n
    U_r, lam_r = layer_covariance(X, r)
    ref_map = _resolve_ref(ref_group, keys)
    rows = []
    for k in deflate_k:
        Xk, _ = deflate(X, int(k))
        means = {key: Xk[slices[key]].mean(0) for key in keys}
        for key in keys:
            Gk = Xk[slices[key]]
            ref = ref_map[key]
            disp = displacement(means[key], means[ref], U_r, lam_r)
            rows.append(dict(group=key, k=int(k), n=int(Gk.shape[0]),
                             log_volume=log_volume(Gk),
                             participation_ratio=participation_ratio(Gk),
                             disp_euclid=disp["euclid"], disp_mahal=disp["mahal"],
                             ref_group=ref, r=int(lam_r.size)))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# pair ordering (G0 gate)
# --------------------------------------------------------------------------------------
def pair_ordering(X: ArrayLike, labels: pd.DataFrame, r: int = 128, k: int = 5) -> pd.DataFrame:
    """Core-and-spread ordering of the paper on our bank (gate G0).

    ``X [N, d]`` are pooled states, ``labels`` a DataFrame (``N`` rows) with columns
    ``family`` (or ``family_id``), ``formulation`` and ``lang``. Every unordered pair is
    classified as

    * ``exact_translation``        same family + formulation, different language,
    * ``same_meaning_same_lang``   same family, different formulation, same language,
    * ``same_meaning_cross_lang``  same family, different formulation, different language,
    * ``other_family``             different family,

    (pairs with identical family, formulation and language are reported as
    ``same_variant`` only if they occur). Distances are the truncated Mahalanobis distance
    (layer covariance of ``X``, rank ``r``) and the cosine, both on the ``k``-deflated,
    centred vectors. One row per category with ``n_pairs, mean_mahal, sd_mahal,
    mean_cosine, sd_cosine``; the first row carries ``core_order_ok`` =
    ``mean_mahal(exact) < mean_mahal(same_meaning_same_lang) < mean_mahal(other_family)``
    (``False`` if any of the three is undefined).
    """
    X = _as_cloud(X)
    N = X.shape[0]
    lab = labels.reset_index(drop=True)
    if len(lab) != N:
        raise ValueError(f"labels has {len(lab)} rows but X has {N}")
    fam_col = "family" if "family" in lab.columns else "family_id"
    fam = lab[fam_col].astype(str).to_numpy()
    form = lab["formulation"].astype(str).to_numpy()
    lang = lab["lang"].astype(str).to_numpy()

    U_r, lam_r = layer_covariance(X, r)
    if lam_r.size >= N - 1:
        warnings.warn(
            f"pair_ordering: effective rank r={lam_r.size} equals N-1={N - 1}; N points whitened "
            "by their own full-rank covariance form a regular simplex (all Mahalanobis distances "
            "equal), so the ordering is uninformative. Use N >> r.", UserWarning, stacklevel=2)
    Xk, _ = deflate(X, k)
    Z = _whiten(Xk, U_r, lam_r)
    sq = (Z * Z).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    D = np.sqrt(np.maximum(D2, 0.0))
    norms = np.linalg.norm(Xk, axis=1)
    denom = np.outer(norms, norms)
    with np.errstate(invalid="ignore", divide="ignore"):
        Cs = np.where(denom > 0, (Xk @ Xk.T) / np.where(denom > 0, denom, 1.0), np.nan)

    same_fam = fam[:, None] == fam[None, :]
    same_form = form[:, None] == form[None, :]
    same_lang = lang[:, None] == lang[None, :]
    iu = np.triu_indices(N, k=1)
    cats = {
        "exact_translation": same_fam & same_form & ~same_lang,
        "same_meaning_same_lang": same_fam & ~same_form & same_lang,
        "same_meaning_cross_lang": same_fam & ~same_form & ~same_lang,
        "other_family": ~same_fam,
        "same_variant": same_fam & same_form & same_lang,
    }
    rows = []
    for name, M in cats.items():
        sel = M[iu]
        n = int(sel.sum())
        if name == "same_variant" and n == 0:
            continue
        dm = D[iu][sel]
        cs = Cs[iu][sel]
        rows.append(dict(category=name, n_pairs=n,
                         mean_mahal=float(np.nanmean(dm)) if n else float("nan"),
                         sd_mahal=float(np.nanstd(dm)) if n else float("nan"),
                         mean_cosine=float(np.nanmean(cs)) if n else float("nan"),
                         sd_cosine=float(np.nanstd(cs)) if n else float("nan")))
    df = pd.DataFrame(rows)
    m = df.set_index("category")["mean_mahal"]
    vals = [m.get("exact_translation", np.nan), m.get("same_meaning_same_lang", np.nan),
            m.get("other_family", np.nan)]
    ok = bool(np.all(np.isfinite(vals)) and vals[0] < vals[1] < vals[2])
    df["core_order_ok"] = pd.array([ok] + [pd.NA] * (len(df) - 1), dtype="boolean")
    df.attrs["r"] = int(lam_r.size)
    df.attrs["k"] = int(k)
    return df


# --------------------------------------------------------------------------------------
# per-position measures
# --------------------------------------------------------------------------------------
def per_position_log_volume(resid_group: ArrayLike, eps: float = 1e-8) -> np.ndarray:
    """Log-volume of the realization cloud at every aligned position:
    ``[n, T_aligned, d] -> [T_aligned]``."""
    R = _np(resid_group)
    if R.ndim != 3:
        raise ValueError(f"expected [n, T, d], got shape {R.shape}")
    return np.array([log_volume(R[:, t, :], eps) for t in range(R.shape[1])], dtype=np.float64)


def align_tail_states(items: Iterable[tuple[ArrayLike, int]]) -> np.ndarray:
    """Align the tails of several sequences from the END.

    ``items`` are ``(resid [T, d], tail_start)`` pairs; the common tail length is
    ``T_tail_min = min(T - tail_start)`` and the result is ``[n, T_tail_min, d]`` holding
    the last ``T_tail_min`` positions of every sequence.
    """
    arrs = [(_np(h), int(ts)) for h, ts in items]
    if not arrs:
        raise ValueError("no sequences given")
    m = min(h.shape[0] - ts for h, ts in arrs)
    if m <= 0:
        raise ValueError("a sequence has an empty tail (tail_start >= T)")
    return np.stack([h[h.shape[0] - m:] for h, _ in arrs], 0)


# --------------------------------------------------------------------------------------
# masks and realization groups
# --------------------------------------------------------------------------------------
def count_switches(mask: Sequence[str]) -> int:
    """Number of adjacent positions with different language labels."""
    m = list(mask)
    return int(sum(1 for a, b in zip(m, m[1:]) if a != b))


def enumerate_unit_masks(K: int, n_units: int = 10) -> list[str]:
    """All unit-level masks over ``{E, D}`` with exactly ``K`` switches, unit 1 = E,
    unit ``n`` = E and unit ``n-1`` = D (dossier constraints for mixed conditions).
    ``K = 0`` returns ``["E" * n_units]``; an odd ``K`` has no admissible mask."""
    K = int(K)
    n_units = int(n_units)
    if n_units < 1:
        raise ValueError("n_units must be >= 1")
    if K == 0:
        return ["E" * n_units]
    if n_units < 3:
        return []
    out = []
    for bits in itertools.product("ED", repeat=n_units - 3):
        m = "E" + "".join(bits) + "DE"
        if count_switches(m) == K:
            out.append(m)
    return out


def unit_mask_to_segment_mask(unit_mask: Sequence[str], C: str) -> list[str]:
    """Segment-level mask (``3 * n_units`` entries, order subj, pred, adj per unit)
    realizing the unit mask at structure level ``C`` (DESIGN section 2.1):

    * ``C1`` every segment of unit ``k`` carries ``m_k``;
    * ``C2`` at every switch boundary ``k -> k+1`` the adjunct of unit ``k`` already
      carries ``m_{k+1}``;
    * ``C3`` at every switch boundary the predicate *and* the adjunct of unit ``k``
      carry ``m_{k+1}`` (subject-verb agreement across the language boundary).
    """
    m = [str(c) for c in unit_mask]
    bad = sorted(set(m) - {"E", "D"})
    if bad:
        raise ValueError(f"unit mask may only contain 'E'/'D', got {bad}")
    if C not in ("C1", "C2", "C3"):
        raise ValueError(f"C must be one of C1/C2/C3, got {C!r}")
    seg = [[m[k]] * len(SEGMENT_NAMES) for k in range(len(m))]
    if C in ("C2", "C3"):
        for k in range(len(m) - 1):
            if m[k] != m[k + 1]:
                seg[k][2] = m[k + 1]            # adj_k
                if C == "C3":
                    seg[k][1] = m[k + 1]        # pred_k
    return [s for unit in seg for s in unit]


def segment_mask_fn() -> Callable[[str, str], list[str]]:
    """``lsne.stimuli.segment_masks_from_unit_mask`` if importable, else the local
    :func:`unit_mask_to_segment_mask` (both implement DESIGN section 2.1)."""
    try:
        from lsne.stimuli import segment_masks_from_unit_mask as fn  # type: ignore
        return fn
    except (ImportError, AttributeError, SyntaxError):
        return unit_mask_to_segment_mask


def realization_groups(families: Sequence, build_fn: Callable, K_levels: Sequence[int],
                       C_levels: Sequence[str]) -> dict:
    """Realization clouds per condition ``(family_id, K, C)``.

    For every family (any object/dict with ``family_id``; ``units`` gives ``n_units``,
    default 10) and every admissible unit mask with exactly ``K`` switches
    (:func:`enumerate_unit_masks`), the segment mask for structure level ``C`` is built
    (``lsne.stimuli.segment_masks_from_unit_mask`` if available) and
    ``build_fn(family, segment_mask)`` produces the variant (``None`` -> skipped).
    Families sharing a ``family_id`` (formulations A/B/C/D) pool into one group.
    ``K = 0`` yields the single group ``(family_id, 0, "C1")``; the segment-level level
    ``"seg"`` is not a realization group and is ignored.
    """
    seg_fn = segment_mask_fn()
    groups: dict = {}
    for fam in families:
        fid = _attr(fam, "family_id")
        units = _attr(fam, "units", None)
        n_units = len(units) if units is not None else 10
        for K in K_levels:
            K = int(K)
            c_list = ["C1"] if K == 0 else [c for c in C_levels if c != "seg"]
            for C in c_list:
                key = (fid, K, C)
                for um in enumerate_unit_masks(K, n_units):
                    sm = list(seg_fn(um, C))
                    v = build_fn(fam, sm)
                    if v is None:
                        continue
                    groups.setdefault(key, []).append(v)
    return groups


# --------------------------------------------------------------------------------------
# full table over sites and readouts
# --------------------------------------------------------------------------------------
def _ref_for(key, keys: list):
    """Reference group: ``(family, 0, "C1")`` of the same family, else any K=0 group of
    that family, else the first group."""
    if isinstance(key, tuple) and len(key) == 3:
        fam = key[0]
        if (fam, 0, "C1") in keys:
            return (fam, 0, "C1")
        for k2 in keys:
            if isinstance(k2, tuple) and len(k2) == 3 and k2[0] == fam and int(k2[1]) == 0:
                return k2
    return keys[0]


def geometry_table(mb: ModelBundle, groups: Mapping, sites: Sequence[HookSite],
                   readouts: Sequence[str] = ("mean", "last"), r_mean: int = 128,
                   r_last: int = 256, deflate_k: Sequence[int] = (0, 5)) -> pd.DataFrame:
    """Local geometry of every realization group at every site and readout.

    Each variant is run once (sequences differ in length) with :func:`run_with_cache` at
    ``sites``; its residual is pooled with every readout (``"mean"`` over all context
    positions, ``"last"`` position). Per ``(site, readout)`` the states of *all* groups
    are pooled to fit the layer covariance (``r = min(r_readout, N-1, d)``) and the
    deflation basis, then :func:`group_geometry` measures every group with the reference
    ``(family_id, 0, "C1")`` of the same family (else the first group).

    Columns: site, readout, k, family_id, K, C, group, n, log_volume, participation_ratio,
    disp_euclid, disp_mahal, ref_group, r.
    """
    sites = list(sites)
    uniq: dict[str, HookSite] = {}
    for s in sites:
        uniq.setdefault(s.key, s)
    sites = list(uniq.values())
    readouts = list(readouts)
    keys = list(groups.keys())
    pooled: dict[tuple[str, str], dict] = {(s.key, ro): {k: [] for k in keys}
                                           for s in sites for ro in readouts}
    for key in keys:
        for v in groups[key]:
            ids = variant_ids(v)
            if ids.numel() == 0:
                continue
            co = run_with_cache(mb, ids[None, :], sites)
            for s in sites:
                h = co.resid[s.key][0]
                for ro in readouts:
                    pooled[(s.key, ro)][key].append(pool_states(h, ro)[0])
    frames = []
    for s in sites:
        for ro in readouts:
            g = {k: np.stack(vs, 0) for k, vs in pooled[(s.key, ro)].items() if vs}
            if not g:
                continue
            gkeys = list(g.keys())
            r = r_mean if ro == "mean" else r_last
            df = group_geometry(g, r=r, deflate_k=deflate_k,
                                ref_group={k: _ref_for(k, gkeys) for k in gkeys})
            df.insert(0, "readout", ro)
            df.insert(0, "site", s.key)
            frames.append(df)
    cols = ["site", "readout", "k", "family_id", "K", "C", "group", "n", "log_volume",
            "participation_ratio", "disp_euclid", "disp_mahal", "ref_group", "r"]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)

    def _split(key):
        if isinstance(key, tuple) and len(key) == 3:
            return key[0], key[1], key[2]
        return str(key), np.nan, None

    parts = out["group"].map(_split)
    out["family_id"] = [p[0] for p in parts]
    out["K"] = [p[1] for p in parts]
    out["C"] = [p[2] for p in parts]
    out["group"] = out["group"].map(group_label)
    out["ref_group"] = out["ref_group"].map(group_label)
    return out[cols]
