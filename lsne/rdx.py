"""Representational Difference Explanation (RDX, DESIGN §4.8a).

Not absolute representations are explained but *difference vectors* between controlled
conditions, ``delta = mean(A) - mean(B)`` (:func:`difference_vector`).  For a candidate
basis ``B [k, d]`` (domain axes, SAE decoder rows, head contributions, principal
components) the **explained fraction** is ``||P_B delta||^2 / ||delta||^2`` with ``P_B``
the orthogonal projector onto ``span(B)`` (:func:`explained_fraction`; computed by least
squares, so the rows need not be orthonormal or independent).  The fraction is
z-standardised against ``n_random`` gaussian bases of the same ``k`` (:func:`rdx_score`;
the expected fraction of a random ``k``-basis is ``k/d``).  A basis explains a difference
only if it beats the random bases **and** the explanation is stable across stimulus
families (:func:`stability`, :func:`family_stability`).

Inputs are numpy arrays or torch tensors; outputs are numpy / python floats.  Contrasts
may be given as pooled activation matrices ``[n, d]``, as dicts ``site key -> [n, d]`` or
as lists of stimulus variants (objects / dicts with ``ids``), in which case the states are
collected from the model at the requested sites (:func:`variant_states`).
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .config import HookSite
from .geometry import pool_states
from .model import ModelBundle, run_with_cache

__all__ = [
    "difference_vector", "explained_fraction", "random_bases", "rdx_score", "basis_attribution",
    "variant_states", "variant_states_by_site", "rdx_table", "stability", "family_stability", "bases_from_directions",
    "bases_from_sae", "bases_from_heads", "bases_from_pca", "site_key",
]

_EPS = 1e-30
RDX_COLS = ["contrast", "basis", "site", "k", "explained", "random_mean", "random_sd", "z",
            "p_random", "delta_norm", "n_a", "n_b", "top_component", "top_share"]


# --------------------------------------------------------------------------------------
# array helpers
# --------------------------------------------------------------------------------------
def _np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _vec(x) -> np.ndarray:
    v = _np(x).reshape(-1)
    return v


def _mat(x) -> np.ndarray:
    M = _np(x)
    if M.ndim == 1:
        M = M[None, :]
    if M.ndim != 2:
        raise ValueError(f"expected a [k, d] matrix, got shape {M.shape}")
    return M


def _cloud(x) -> np.ndarray:
    A = _np(x)
    if A.ndim == 1:
        A = A[None, :]
    if A.ndim != 2 or A.shape[0] == 0:
        raise ValueError(f"expected a non-empty [n, d] activation matrix, got shape {A.shape}")
    return A


def site_key(site) -> str:
    """``HookSite -> site.key``; strings are returned unchanged."""
    if isinstance(site, HookSite):
        return site.key
    return str(site)


# --------------------------------------------------------------------------------------
# core quantities
# --------------------------------------------------------------------------------------
def difference_vector(acts_a, acts_b) -> np.ndarray:
    """``mean(acts_a) - mean(acts_b)`` over the rows of two ``[n, d]`` clouds (float64)."""
    A, B = _cloud(acts_a), _cloud(acts_b)
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"dimension mismatch: {A.shape[1]} vs {B.shape[1]}")
    return A.mean(0) - B.mean(0)


def explained_fraction(delta, basis) -> float:
    """``||P_B delta||^2 / ||delta||^2``: the fraction of the squared norm of ``delta``
    that lies in the span of the rows of ``basis [k, d]``.  The projection is the least-
    squares fit ``c = argmin ||B^T c - delta||`` (``numpy.linalg.lstsq``), so the rows may
    be non-orthonormal or linearly dependent.  Returns ``nan`` for a zero ``delta`` and
    ``0.0`` for an empty basis; the value is clipped to ``[0, 1]`` against round-off."""
    d = _vec(delta)
    n2 = float(d @ d)
    if not np.isfinite(n2) or n2 <= 0.0:
        return float("nan")
    B = _np(basis)
    if B.ndim == 1:
        B = B[None, :]
    if B.size == 0 or B.shape[0] == 0:
        return 0.0
    if B.shape[1] != d.shape[0]:
        raise ValueError(f"basis has d={B.shape[1]} but delta has d={d.shape[0]}")
    c, *_ = np.linalg.lstsq(B.T, d, rcond=None)
    proj = B.T @ c
    return float(np.clip(float(proj @ proj) / n2, 0.0, 1.0))


def random_bases(d: int, k: int, n: int, seed: int = 0) -> np.ndarray:
    """``n`` gaussian bases ``[n, k, d]`` with unit-norm rows (``numpy.random.default_rng(seed)``)."""
    rng = np.random.default_rng(int(seed))
    X = rng.standard_normal((int(n), int(k), int(d)))
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), _EPS)


def rdx_score(delta, basis, n_random: int = 32, seed: int = 0) -> dict:
    """Explained fraction of ``delta`` by ``basis`` against ``n_random`` random bases of
    the same ``k`` drawn in the ambient space of ``delta``.

    Returns ``dict(explained, random_mean, random_sd, z, k, n_random, p_random, random)``
    with ``z = (explained - random_mean) / random_sd`` (``0`` when the random fractions do
    not vary and the difference vanishes, e.g. ``k >= d``; ``+-inf`` when they do not vary
    but the difference does not vanish), ``p_random`` the share of random bases whose
    fraction is at least ``explained`` and ``random`` the ``n_random`` fractions."""
    d = _vec(delta)
    B = _mat(basis)
    k = int(B.shape[0])
    explained = explained_fraction(d, B)
    n_random = int(max(n_random, 2))
    R = random_bases(d.shape[0], k, n_random, seed)
    vals = np.array([explained_fraction(d, R[i]) for i in range(n_random)], dtype=np.float64)
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1))
    diff = explained - mean
    if np.isnan(explained):
        z = float("nan")
    elif sd > 1e-12:
        z = float(diff / sd)
    else:
        z = 0.0 if abs(diff) < 1e-12 else float(np.sign(diff) * np.inf)
    p = float(np.mean(vals >= explained)) if not np.isnan(explained) else float("nan")
    return dict(explained=explained, random_mean=mean, random_sd=sd, z=z, k=k, n_random=n_random,
                p_random=p, random=vals)


def basis_attribution(delta, basis, names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Least-squares coefficients of ``delta`` on the (non-orthogonalised) basis rows.

    Columns: ``name, coef, norm, abs_share, cos`` with ``abs_share = |coef| * ||b_i|| /
    sum_j |coef_j| * ||b_j||`` (the share of the fitted vector carried by each row) and
    ``cos`` the cosine between ``delta`` and the row; sorted by ``abs_share`` descending
    (ties: basis order)."""
    d = _vec(delta)
    B = _mat(basis)
    k = B.shape[0]
    nm = [f"b{i}" for i in range(k)] if names is None else [str(n) for n in names]
    if len(nm) != k:
        raise ValueError(f"{len(nm)} names for {k} basis rows")
    c, *_ = np.linalg.lstsq(B.T, d, rcond=None)
    norms = np.linalg.norm(B, axis=1)
    w = np.abs(c) * norms
    share = w / w.sum() if w.sum() > 0 else np.zeros_like(w)
    dn = float(np.linalg.norm(d))
    cos = (B @ d) / np.maximum(norms * dn, _EPS) if dn > 0 else np.zeros(k)
    df = pd.DataFrame(dict(name=nm, coef=c.astype(np.float64), norm=norms, abs_share=share, cos=cos))
    df["order"] = np.arange(k)
    df = df.sort_values(["abs_share", "order"], ascending=[False, True], kind="mergesort")
    return df.drop(columns="order").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# collecting states from variants
# --------------------------------------------------------------------------------------
def _ids_of(v) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.detach().long().reshape(-1).cpu()
    if isinstance(v, Mapping):
        ids = v["ids"]
    elif hasattr(v, "ids"):
        ids = getattr(v, "ids")
    else:
        ids = v
    if isinstance(ids, torch.Tensor):
        return ids.detach().long().reshape(-1).cpu()
    return torch.as_tensor([int(i) for i in np.asarray(ids).reshape(-1)], dtype=torch.long)


def _is_variant_list(x) -> bool:
    if isinstance(x, (np.ndarray, torch.Tensor, Mapping)) or not isinstance(x, (list, tuple)):
        return False
    if len(x) == 0:
        return False
    v = x[0]
    if isinstance(v, Mapping):
        return "ids" in v
    if hasattr(v, "ids"):
        return True
    return isinstance(v, (list, tuple, torch.Tensor)) and not isinstance(v, (int, float))


def variant_states_by_site(mb: ModelBundle, variants: Sequence, sites: Sequence[HookSite],
                           readout: str = "last") -> dict[str, np.ndarray]:
    """Pooled residual states ``{site.key: [n, d]}`` of ``variants`` at every site of
    ``sites``: one forward per variant (lengths differ) recording all sites, pooled with
    :func:`lsne.geometry.pool_states` (``readout`` ``"last"`` or ``"mean"``).  A variant
    is an object or dict with ``ids``, a tensor or a list of token ids."""
    if mb is None:
        raise ValueError("a ModelBundle is needed to collect states from variants")
    sites = list(sites)
    for s in sites:
        if not isinstance(s, HookSite):
            raise TypeError("sites must be HookSite objects when states are collected from variants")
    out: dict[str, list] = {s.key: [] for s in sites}
    for v in variants:
        ids = _ids_of(v)
        if ids.numel() == 0:
            continue
        co = run_with_cache(mb, ids[None, :], sites)
        for s in sites:
            h = co.resid[s.key][0].detach().float().cpu()
            out[s.key].append(pool_states(h, readout)[0])
    return {k: (np.stack(v, 0) if v else np.zeros((0, mb.d_model))) for k, v in out.items()}


def variant_states(mb: ModelBundle, variants: Sequence, site: HookSite, readout: str = "last") -> np.ndarray:
    """Pooled residual states ``[n, d]`` (float64) of ``variants`` at ``site`` (see
    :func:`variant_states_by_site`)."""
    return variant_states_by_site(mb, variants, [site], readout)[site.key]


def _states_by_site(mb, acts, sites, readout: str) -> dict[str, np.ndarray]:
    """Normalise one side of a contrast into ``{site key: [n, d]}``."""
    if isinstance(acts, Mapping):
        d = {site_key(k): _cloud(v) for k, v in acts.items()}
        if sites:
            keys = [site_key(s) for s in sites]
            missing = [k for k in keys if k not in d]
            if missing:
                raise KeyError(f"activations missing for sites {missing}")
            d = {k: d[k] for k in keys}
        return d
    if _is_variant_list(acts):
        if not sites:
            raise ValueError("variant lists need explicit HookSite objects in `sites`")
        return variant_states_by_site(mb, acts, sites, readout)
    A = _cloud(acts)
    key = site_key(sites[0]) if sites and len(sites) == 1 else "all"
    return {key: A}


def _basis_entry(val, key: str):
    """Resolve a bases-dict value for site ``key`` -> ``(matrix [k, d], names | None)`` or ``None``."""
    if val is None:
        return None
    if isinstance(val, tuple) and len(val) == 2 and not isinstance(val[0], (str, bytes)):
        M, names = val
        return _mat(M), (None if names is None else [str(n) for n in names])
    if isinstance(val, Mapping):
        sub = val.get(key, val.get("all"))
        return _basis_entry(sub, key) if sub is not None else None
    if hasattr(val, "vec") and hasattr(val, "name"):          # a single Direction
        return _mat(getattr(val, "vec")), [str(getattr(val, "name"))]
    if isinstance(val, (list, tuple)) and val and hasattr(val[0], "vec"):
        return bases_from_directions(val)
    return _mat(val), None


def rdx_table(mb: Optional[ModelBundle], contrasts: Mapping[str, tuple], bases: Mapping[str, object],
              sites: Optional[Sequence] = None, n_random: int = 32, seed: int = 0,
              readout: str = "last") -> pd.DataFrame:
    """RDX scores for every ``contrast x basis x site``.

    ``contrasts``: ``{name: (A, B)}`` where each side is a ``[n, d]`` activation matrix, a
    dict ``site key -> [n, d]`` or a list of variants (then ``mb`` and ``sites`` as
    ``HookSite`` objects are required and the states are pooled with ``readout``).
    ``bases``: ``{name: M [k, d] | (M, names) | {site key: M | (M, names)} | [Direction]}``;
    a per-site dict may hold the key ``"all"`` as a shared fallback.  ``sites`` restricts /
    orders the site keys (``HookSite`` or strings); plain matrices are labelled ``"all"``
    (or the single site given).

    Columns: ``contrast, basis, site, k, explained, random_mean, random_sd, z, p_random,
    delta_norm, n_a, n_b, top_component, top_share`` (the row of the basis with the
    largest attribution share, :func:`basis_attribution`).  Bases that do not fit a site's
    dimension are skipped with a warning-free ``NaN`` row."""
    rows: list[dict] = []
    site_list = list(sites) if sites is not None else None
    for cname, pair in contrasts.items():
        if not (isinstance(pair, (tuple, list)) and len(pair) == 2):
            raise ValueError(f"contrast {cname!r} must be a pair (acts_a, acts_b)")
        A_by, B_by = (_states_by_site(mb, pair[0], site_list, readout),
                      _states_by_site(mb, pair[1], site_list, readout))
        keys = [k for k in A_by if k in B_by]
        for key in keys:
            A, B = A_by[key], B_by[key]
            delta = difference_vector(A, B)
            dn = float(np.linalg.norm(delta))
            for bname, bval in bases.items():
                ent = _basis_entry(bval, key)
                base = dict(contrast=str(cname), basis=str(bname), site=key, n_a=int(A.shape[0]),
                            n_b=int(B.shape[0]), delta_norm=dn)
                if ent is None or ent[0].shape[1] != delta.shape[0]:
                    rows.append(dict(base, k=0 if ent is None else int(ent[0].shape[0]), explained=np.nan,
                                     random_mean=np.nan, random_sd=np.nan, z=np.nan, p_random=np.nan,
                                     top_component=None, top_share=np.nan))
                    continue
                M, names = ent
                sc = rdx_score(delta, M, n_random=n_random, seed=seed)
                att = basis_attribution(delta, M, names)
                top = att.iloc[0] if len(att) else None
                rows.append(dict(base, k=sc["k"], explained=sc["explained"], random_mean=sc["random_mean"],
                                 random_sd=sc["random_sd"], z=sc["z"], p_random=sc["p_random"],
                                 top_component=None if top is None else str(top["name"]),
                                 top_share=np.nan if top is None else float(top["abs_share"])))
    return pd.DataFrame(rows, columns=RDX_COLS)


def stability(rdx_rows_by_family: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Across-family stability of RDX rows: ``{family: rdx_table(...)}`` ->
    one row per ``(contrast, basis, site)`` with ``n_families, explained_mean,
    explained_sd, explained_min, z_mean, z_sd, z_min, frac_z_ge_2`` (share of families
    with ``z >= 2``, the preregistered minimum effect)."""
    parts = []
    for fam, df in rdx_rows_by_family.items():
        if df is None or len(df) == 0:
            continue
        d = df.copy()
        d["family"] = str(fam)
        parts.append(d)
    cols = ["contrast", "basis", "site", "n_families", "explained_mean", "explained_sd", "explained_min",
            "z_mean", "z_sd", "z_min", "frac_z_ge_2"]
    if not parts:
        return pd.DataFrame(columns=cols)
    all_ = pd.concat(parts, ignore_index=True)
    rows = []
    for (c, b, s), g in all_.groupby(["contrast", "basis", "site"], sort=True):
        e = np.asarray(g["explained"], dtype=np.float64)
        z = np.asarray(g["z"], dtype=np.float64)
        ok = np.isfinite(z)
        rows.append(dict(contrast=c, basis=b, site=s, n_families=int(g["family"].nunique()),
                         explained_mean=float(np.nanmean(e)), explained_sd=float(np.nanstd(e, ddof=1)) if len(e) > 1 else 0.0,
                         explained_min=float(np.nanmin(e)), z_mean=float(np.nanmean(z)),
                         z_sd=float(np.nanstd(z, ddof=1)) if len(z) > 1 else 0.0, z_min=float(np.nanmin(z)),
                         frac_z_ge_2=float(np.mean(z[ok] >= 2.0)) if ok.any() else float("nan")))
    return pd.DataFrame(rows, columns=cols)


def family_stability(mb: Optional[ModelBundle], contrast_by_family: Mapping[str, tuple], basis,
                     site=None, n_boot: int = 200, seed: int = 0, readout: str = "last",
                     n_random: int = 32) -> dict:
    """Explained fraction of the per-family difference vectors by ``basis`` with a
    bootstrap CI over families.

    ``contrast_by_family``: ``{family: (A, B)}`` with each side a ``[n, d]`` matrix, a dict
    ``site key -> matrix`` or a list of variants (``mb`` and a ``HookSite`` ``site`` are then
    required).  Returns ``dict(explained_mean, ci_low, ci_high, n_families, per_family)``
    where ``per_family`` has ``family, explained, z, delta_norm, n_a, n_b`` and the CI is
    the percentile bootstrap (``n_boot`` resamples of families) of the mean fraction."""
    M = _mat(basis)
    sites = [site] if site is not None else None
    rows = []
    for fam, pair in contrast_by_family.items():
        A_by = _states_by_site(mb, pair[0], sites, readout)
        B_by = _states_by_site(mb, pair[1], sites, readout)
        key = next(iter(A_by))
        A, B = A_by[key], B_by[key]
        delta = difference_vector(A, B)
        sc = rdx_score(delta, M, n_random=n_random, seed=seed)
        rows.append(dict(family=str(fam), explained=sc["explained"], z=sc["z"], delta_norm=float(np.linalg.norm(delta)),
                         n_a=int(A.shape[0]), n_b=int(B.shape[0])))
    per = pd.DataFrame(rows, columns=["family", "explained", "z", "delta_norm", "n_a", "n_b"])
    e = np.asarray(per["explained"], dtype=np.float64)
    e = e[np.isfinite(e)]
    n = int(e.shape[0])
    if n == 0:
        return dict(explained_mean=float("nan"), ci_low=float("nan"), ci_high=float("nan"), n_families=0, per_family=per)
    mean = float(e.mean())
    if n > 1 and int(n_boot) > 0:
        rng = np.random.default_rng(int(seed))
        boots = e[rng.integers(0, n, size=(int(n_boot), n))].mean(1)
        lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
    else:
        lo = hi = mean
    return dict(explained_mean=mean, ci_low=lo, ci_high=hi, n_families=n, per_family=per)


# --------------------------------------------------------------------------------------
# candidate bases
# --------------------------------------------------------------------------------------
def _unit_rows(M: np.ndarray) -> np.ndarray:
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), _EPS)


def bases_from_directions(dirs: Sequence) -> tuple[np.ndarray, list[str]]:
    """Stack ``Direction`` objects (``.vec``, ``.name``) into ``([k, d], names)``."""
    dirs = list(dirs)
    if not dirs:
        raise ValueError("no directions")
    M = np.stack([_vec(getattr(d, "vec", d)) for d in dirs], 0)
    names = [str(getattr(d, "name", f"dir{i}")) for i, d in enumerate(dirs)]
    return M, names


def bases_from_sae(sae, feature_ids: Sequence[int]) -> tuple[np.ndarray, list[str]]:
    """Unit decoder rows ``W_dec[feature_ids]`` as ``([k, d], ["sae<id>", ...])``."""
    ids = [int(i) for i in feature_ids]
    if not ids:
        raise ValueError("no feature ids")
    W = sae.W_dec.detach().float().cpu().numpy().astype(np.float64)
    return _unit_rows(W[ids]), [f"sae{i}" for i in ids]


def bases_from_heads(mb: ModelBundle, input_ids, layer: int, position: int = -1) -> tuple[np.ndarray, list[str]]:
    """Unit head-contribution directions of block ``layer`` at ``position`` (default: last
    token) for ``input_ids``: ``([H, d], ["L<layer>H<h>", ...])``
    (:func:`lsne.heads.head_contributions`)."""
    from .heads import head_contributions
    ids = _ids_of(input_ids)
    c = head_contributions(mb, ids[None, :], int(layer))          # [H, T, d]
    if c.dim() == 4:
        c = c[:, 0]
    T = int(c.shape[1])
    pos = int(position) % T
    M = _np(c[:, pos])
    return _unit_rows(M), [f"L{int(layer)}H{h}" for h in range(M.shape[0])]


def bases_from_pca(acts, k: int) -> tuple[np.ndarray, list[str]]:
    """Top-``k`` principal directions of the centred cloud ``acts [n, d]`` (right singular
    vectors, ``k`` clipped to ``min(n - 1, d)``): ``([k, d], ["pc0", ...])``."""
    X = _cloud(acts)
    Xc = X - X.mean(0, keepdims=True)
    kk = int(min(int(k), max(X.shape[0] - 1, 1), X.shape[1]))
    _, _, Vh = np.linalg.svd(Xc, full_matrices=False)
    return Vh[:kk].copy(), [f"pc{i}" for i in range(kk)]
