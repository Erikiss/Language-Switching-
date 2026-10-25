"""Representational sharing index and network activity (DESIGN §4.8c).

"Sharing" is a product metric at Meta, not a model parameter; the model-internal
analogue measured here is **representational sharing**: how strongly do different
domains / conditions use the same units?  Per ``(layer, site)``:

* **Sharing index** ``S``: Jaccard overlap of the active TopK-SAE latents between
  conditions, pairwise and global (:func:`active_features`, :func:`sharing_index`),
  frequency-weighted variants (:func:`frequency_sharing`, :func:`pairwise_sharing`) and
  the cosine of the per-head contribution-norm profiles (:func:`head_profile`,
  :func:`head_profile_cosines`).
* **Network activity** ``A``: share of latents above threshold, mean activation mass and
  the participation ratio of the latent activations (:func:`network_activity`).

``S`` and ``A`` are measurable before and after an intervention (test-time training,
:mod:`lsne.ttt`); :func:`sharing_report` / :func:`sharing_table` bundle them for a set of
conditions given as pooled states ``[n, d]`` or as stimulus variants.
"""
from __future__ import annotations

import itertools
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .config import HookSite, layer_grid as _default_layer_grid
from .model import ModelBundle
from .rdx import _ids_of, _is_variant_list, variant_states

__all__ = [
    "active_features", "active_feature_sets", "feature_frequency", "jaccard", "sharing_index",
    "frequency_sharing", "pairwise_sharing", "network_activity", "head_profile", "head_profiles",
    "head_profile_cosines", "head_profile_similarity", "sharing_report", "sharing_table",
]

_EPS = 1e-30
SHARING_COLS = ["a", "b", "jaccard", "n_a", "n_b", "n_shared", "n_union", "kind"]
ACTIVITY_KEYS = ["frac_active", "mean_mass", "participation_ratio", "participation_ratio_frac",
                 "n_active_mean", "frac_features_active", "n_rows", "num_latents", "k", "threshold", "topk_only"]


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _as_2d(acts) -> torch.Tensor:
    X = torch.as_tensor(np.asarray(acts, dtype=np.float32)) if not isinstance(acts, torch.Tensor) else acts
    X = X.detach().float().cpu()
    if X.dim() == 1:
        X = X[None, :]
    return X.reshape(-1, X.shape[-1])


def _chunks(X: torch.Tensor, chunk: int):
    for i in range(0, X.shape[0], chunk):
        yield X[i:i + chunk]


def _dense_acts(sae, xb: torch.Tensor, topk_only: bool) -> torch.Tensor:
    """Dense latent activations ``[n, F]``: TopK code (``topk_only``) or ``relu(pre)``."""
    top_acts, top_idx, pre = sae.encode(xb)
    if topk_only:
        dense = torch.zeros_like(pre)
        dense.scatter_(-1, top_idx, top_acts)
        return dense.float().cpu()
    return torch.relu(pre).float().cpu()


# --------------------------------------------------------------------------------------
# active latents and sharing
# --------------------------------------------------------------------------------------
def active_feature_sets(sae, acts, threshold: float = 0.0, topk_only: bool = True,
                        chunk: int = 1024) -> list[set[int]]:
    """Active latent indices per row of ``acts [n, d]``: the TopK-selected latents with
    activation ``> threshold`` (``topk_only=True``) or every latent with
    ``relu(pre) > threshold``."""
    X = _as_2d(acts)
    out: list[set[int]] = []
    with torch.no_grad():
        for xb in _chunks(X, chunk):
            top_acts, top_idx, pre = sae.encode(xb)
            if topk_only:
                mask = top_acts > float(threshold)
                for i in range(xb.shape[0]):
                    out.append({int(j) for j in top_idx[i][mask[i]].tolist()})
            else:
                mask = torch.relu(pre) > float(threshold)
                for i in range(xb.shape[0]):
                    out.append({int(j) for j in mask[i].nonzero().reshape(-1).tolist()})
    return out


def active_features(sae, acts, threshold: float = 0.0, topk_only: bool = True) -> set[int]:
    """Union of the active latent indices over all rows of ``acts`` (see
    :func:`active_feature_sets`)."""
    return set().union(*active_feature_sets(sae, acts, threshold=threshold, topk_only=topk_only))


def feature_frequency(sets: Sequence[set], n_latents: int) -> np.ndarray:
    """Fraction of rows in which each latent is active, ``[F]`` (zeros for an empty list)."""
    freq = np.zeros(int(n_latents), dtype=np.float64)
    sets = list(sets)
    for s in sets:
        for j in s:
            freq[int(j)] += 1.0
    return freq / max(len(sets), 1)


def jaccard(a: set, b: set) -> float:
    """``|a & b| / |a | b|`` (``1.0`` when both are empty)."""
    a, b = set(a), set(b)
    u = a | b
    return float(len(a & b) / len(u)) if u else 1.0


def _pairwise_sets(sets: Mapping[str, set]) -> pd.DataFrame:
    names = list(sets.keys())
    rows = []
    for a, b in itertools.combinations(names, 2):
        A, B = set(sets[a]), set(sets[b])
        inter, union = A & B, A | B
        rows.append(dict(a=str(a), b=str(b), jaccard=(len(inter) / len(union)) if union else 1.0,
                         n_a=len(A), n_b=len(B), n_shared=len(inter), n_union=len(union), kind="pair"))
    if names:
        all_sets = [set(sets[n]) for n in names]
        inter = set.intersection(*all_sets)
        union = set.union(*all_sets)
        rows.append(dict(a="global", b="global", jaccard=(len(inter) / len(union)) if union else 1.0,
                         n_a=len(union), n_b=len(union), n_shared=len(inter), n_union=len(union), kind="global"))
    return pd.DataFrame(rows, columns=SHARING_COLS)


def frequency_sharing(freq_a, freq_b, top_m: int = 200, min_freq: float = 0.01) -> dict:
    """Sharing between two latent-frequency vectors ``[F]``: ``jaccard_topm`` over the
    ``top_m`` most frequent latents of each side, ``jaccard_active`` over latents with
    ``freq >= min_freq`` and ``weighted_overlap = sum min(fa, fb) / sum max(fa, fb)``."""
    fa = np.asarray(freq_a, dtype=np.float64).reshape(-1)
    fb = np.asarray(freq_b, dtype=np.float64).reshape(-1)
    if fa.shape != fb.shape:
        raise ValueError(f"frequency vectors differ in length: {fa.shape[0]} vs {fb.shape[0]}")
    m = int(min(top_m, fa.shape[0]))

    def _top(f: np.ndarray) -> set:
        order = np.argsort(-f, kind="stable")[:m]
        return set(order[f[order] > 0].tolist())          # only latents that are active at all

    ta, tb = _top(fa), _top(fb)
    aa = set(np.nonzero(fa >= float(min_freq))[0].tolist())
    ab = set(np.nonzero(fb >= float(min_freq))[0].tolist())
    mx = np.maximum(fa, fb).sum()
    return dict(jaccard_topm=jaccard(ta, tb), jaccard_active=jaccard(aa, ab),
                weighted_overlap=float(np.minimum(fa, fb).sum() / mx) if mx > 0 else 1.0,
                n_topm_a=len(ta), n_topm_b=len(tb), n_active_a=len(aa), n_active_b=len(ab))


def sharing_index(sets, freq_b=None, **kw):
    """Pairwise sharing between conditions.

    ``sharing_index({cond: set[int]})`` -> ``DataFrame(a, b, jaccard, n_a, n_b, n_shared,
    n_union, kind)`` with one ``kind="pair"`` row per condition pair (dict order) and one
    ``kind="global"`` row (``a = b = "global"``: intersection of all / union of all;
    ``n_a = n_b = n_union`` there).  Called with two frequency vectors,
    ``sharing_index(freq_a, freq_b, top_m=200, min_freq=0.01)`` returns
    :func:`frequency_sharing`."""
    if isinstance(sets, Mapping):
        if freq_b is not None:
            raise ValueError("pass either a dict of sets or two frequency vectors")
        return _pairwise_sets(sets)
    if freq_b is None:
        raise ValueError("sharing_index needs a dict of sets or two frequency vectors")
    return frequency_sharing(sets, freq_b, **kw)


def pairwise_sharing(freqs: Mapping[str, np.ndarray], **kw) -> pd.DataFrame:
    """:func:`frequency_sharing` for every pair of conditions: ``DataFrame(a, b,
    jaccard_topm, jaccard_active, weighted_overlap)``."""
    names = list(freqs.keys())
    rows = []
    for a, b in itertools.combinations(names, 2):
        r = frequency_sharing(freqs[a], freqs[b], **kw)
        rows.append(dict(a=str(a), b=str(b), jaccard_topm=r["jaccard_topm"], jaccard_active=r["jaccard_active"],
                         weighted_overlap=r["weighted_overlap"]))
    return pd.DataFrame(rows, columns=["a", "b", "jaccard_topm", "jaccard_active", "weighted_overlap"])


# --------------------------------------------------------------------------------------
# network activity
# --------------------------------------------------------------------------------------
def network_activity(sae, acts, threshold: float = 0.0, topk_only: bool = False, min_freq: float = 0.01,
                     chunk: int = 1024) -> dict:
    """Activity of the SAE latents on the states ``acts [n, d]``.

    Activations are ``relu(pre)`` (default) or the TopK code (``topk_only=True``); a latent
    is *active* when its activation is ``> threshold``.  Returns
    ``frac_active`` (mean over states of the share of active latents),
    ``mean_mass`` (mean over states of the summed activation),
    ``participation_ratio`` ``(sum_j a_j)^2 / sum_j a_j^2`` of the mean activation vector
    ``a [F]`` over latents (``participation_ratio_frac = PR / F``; 0 when nothing is active),
    ``n_active_mean``, ``frac_features_active`` (share of latents active in at least
    ``min_freq`` of the states), ``n_rows``, ``num_latents``, ``k``, ``threshold``, ``topk_only``."""
    X = _as_2d(acts)
    F = int(sae.num_latents)
    n = int(X.shape[0])
    if n == 0:
        raise ValueError("no states")
    n_active = []
    mass = []
    total = torch.zeros(F, dtype=torch.float64)
    counts = torch.zeros(F, dtype=torch.float64)
    with torch.no_grad():
        for xb in _chunks(X, chunk):
            A = _dense_acts(sae, xb, topk_only)                         # [nb, F]
            act = A > float(threshold)
            n_active.append(act.sum(-1).double())
            mass.append(A.sum(-1).double())
            total += A.double().sum(0)
            counts += act.double().sum(0)
    n_active = torch.cat(n_active)
    mass = torch.cat(mass)
    a = total / n
    s1, s2 = float(a.sum()), float((a * a).sum())
    pr = (s1 * s1 / s2) if s2 > 0 else 0.0
    return dict(frac_active=float(n_active.mean() / F), mean_mass=float(mass.mean()), participation_ratio=float(pr),
                participation_ratio_frac=float(pr / F), n_active_mean=float(n_active.mean()),
                frac_features_active=float(((counts / n) >= float(min_freq)).double().mean()), n_rows=n,
                num_latents=F, k=int(getattr(sae, "k", 0)), threshold=float(threshold), topk_only=bool(topk_only))


# --------------------------------------------------------------------------------------
# head profiles
# --------------------------------------------------------------------------------------
def head_profile(mb: ModelBundle, input_ids, layer: int, position: int = -1) -> np.ndarray:
    """Per-head contribution norm ``[H]`` of block ``layer`` at ``position`` (default: the
    last token) for ``input_ids`` (:func:`lsne.heads.head_contributions`)."""
    from .heads import head_contributions
    ids = _ids_of(input_ids)
    c = head_contributions(mb, ids[None, :], int(layer))                 # [H, T, d]
    if c.dim() == 4:
        c = c[:, 0]
    pos = int(position) % int(c.shape[1])
    return c[:, pos].norm(dim=-1).numpy().astype(np.float64)


def _head_norms_all_layers(mb: ModelBundle, input_ids, layers: Sequence[int], position: int = -1) -> dict[int, np.ndarray]:
    """:func:`head_profile` for several layers from a single forward: the ``dense`` input
    of every requested block is captured with a pre-hook and ``||x[slice_h] @ W_O[:, slice_h].T||``
    is evaluated at ``position`` (the same decomposition as
    :func:`lsne.heads.head_contributions`)."""
    from .heads import head_slices
    ids = _ids_of(input_ids)
    layers = [int(l) for l in layers]
    store: dict[int, torch.Tensor] = {}
    handles = []
    for l in layers:
        def pre(module, args, kwargs, _l=l):
            store[_l] = (args[0] if args else kwargs["input"]).detach()
        handles.append(mb.layers[l].attention.dense.register_forward_pre_hook(pre, with_kwargs=True))
    try:
        with torch.no_grad():
            mb.model(input_ids=ids[None, :].to(mb.device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    out: dict[int, np.ndarray] = {}
    for l in layers:
        x = store[l][0].float().cpu()                                       # [T, H*dh]
        pos = int(position) % int(x.shape[0])
        W = mb.layers[l].attention.dense.weight.detach().float().cpu()      # [d, H*dh]
        out[l] = np.array([float((x[pos, s] @ W[:, s].T).norm()) for s in head_slices(mb, l)], dtype=np.float64)
    return out


def head_profiles(mb: ModelBundle, variants_by_condition: Mapping[str, Sequence], layer_grid: Sequence[int],
                  position: int = -1) -> dict[int, dict[str, np.ndarray]]:
    """Mean per-head contribution-norm profile per condition: ``{layer: {condition: [H]}}``
    (one forward per variant, all layers of ``layer_grid`` captured at once)."""
    layers = [int(l) for l in layer_grid]
    out: dict[int, dict[str, np.ndarray]] = {l: {} for l in layers}
    for cond, variants in variants_by_condition.items():
        vs = list(variants)
        if not vs:
            continue
        per_variant = [_head_norms_all_layers(mb, v, layers, position) for v in vs]
        for l in layers:
            out[l][str(cond)] = np.mean([p[l] for p in per_variant], axis=0)
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), _EPS))


def head_profile_similarity(profiles: Mapping[str, np.ndarray]) -> pd.DataFrame:
    """Cosine between the head profiles of every pair of conditions: ``DataFrame(a, b, cosine)``."""
    names = list(profiles.keys())
    rows = [dict(a=str(a), b=str(b), cosine=_cos(profiles[a], profiles[b])) for a, b in itertools.combinations(names, 2)]
    return pd.DataFrame(rows, columns=["a", "b", "cosine"])


def head_profile_cosines(mb: ModelBundle, variants_by_condition: Mapping[str, Sequence],
                         layer_grid: Optional[Sequence[int]] = None, position: int = -1) -> pd.DataFrame:
    """Cosine of the mean per-head contribution-norm profiles between conditions for every
    layer of ``layer_grid`` (default :func:`lsne.config.layer_grid`):
    ``DataFrame(a, b, layer, cosine)``."""
    grid = list(layer_grid) if layer_grid is not None else list(_default_layer_grid(mb.n_layers))
    profs = head_profiles(mb, variants_by_condition, grid, position)
    rows = []
    for layer in grid:
        sim = head_profile_similarity(profs[int(layer)])
        for r in sim.itertuples(index=False):
            rows.append(dict(a=r.a, b=r.b, layer=int(layer), cosine=float(r.cosine)))
    return pd.DataFrame(rows, columns=["a", "b", "layer", "cosine"])


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
def _group_states(mb, groups: Mapping[str, object], site, readout: str) -> tuple[dict[str, np.ndarray], bool]:
    """``{condition: [n, d]}`` from pooled states or variant lists; the flag says whether
    the groups were variants (head profiles are then available)."""
    states: dict[str, np.ndarray] = {}
    were_variants = False
    for cond, g in groups.items():
        if _is_variant_list(g):
            if not isinstance(site, HookSite):
                raise TypeError("a HookSite is needed to collect states from variants")
            states[str(cond)] = variant_states(mb, g, site, readout)
            were_variants = True
        else:
            states[str(cond)] = _as_2d(g).numpy().astype(np.float64)
    return states, were_variants


def sharing_report(mb: Optional[ModelBundle], sae, groups: Mapping[str, object], site=None, threshold: float = 0.0,
                   topk_only: bool = True, readout: str = "last", layer_grid: Optional[Sequence[int]] = None,
                   variants_by_condition: Optional[Mapping[str, Sequence]] = None, min_freq: float = 0.01) -> dict:
    """Sharing index and network activity for a set of conditions.

    ``groups``: ``{condition: pooled states [n, d]}`` or ``{condition: [variants]}`` (then
    ``mb`` and a ``HookSite`` ``site`` are required; states are pooled with ``readout``).
    Returns ``dict(sharing, activity, heads, sets, site, n)``: ``sharing`` =
    :func:`sharing_index` of the active latent sets (TopK latents ``> threshold`` by
    default), ``activity`` = one :func:`network_activity` row per condition (column
    ``condition`` first), ``heads`` = :func:`head_profile_cosines` over ``layer_grid`` when
    variants are available (``variants_by_condition`` or variant ``groups``), else ``None``."""
    states, were_variants = _group_states(mb, groups, site, readout)
    sets = {c: active_features(sae, X, threshold=threshold, topk_only=topk_only) for c, X in states.items()}
    sharing = sharing_index(sets)
    act_rows = []
    for c, X in states.items():
        r = network_activity(sae, X, threshold=threshold, topk_only=topk_only, min_freq=min_freq)
        act_rows.append(dict(condition=c, **r))
    activity = pd.DataFrame(act_rows, columns=["condition"] + ACTIVITY_KEYS)
    heads = None
    vbc = variants_by_condition if variants_by_condition is not None else (groups if were_variants else None)
    if vbc is not None and mb is not None:
        heads = head_profile_cosines(mb, vbc, layer_grid)
    return dict(sharing=sharing, activity=activity, heads=heads, sets=sets,
                site=(site.key if isinstance(site, HookSite) else site), n={c: int(X.shape[0]) for c, X in states.items()})


def sharing_table(mb: Optional[ModelBundle], sae, groups: Mapping[str, object], site=None, readout: str = "last",
                  threshold: float = 0.0, topk_only: bool = True, layer_grid: Optional[Sequence[int]] = None) -> pd.DataFrame:
    """Flat table of :func:`sharing_report`: ``kind="sharing"`` rows (``a, b, jaccard,
    n_a, n_b, n_shared, n_union``; the global row has ``a = b = "global"``),
    ``kind="activity"`` rows (``a`` = condition, activity metrics) and, when head profiles
    are available, ``kind="heads"`` rows (``a, b, layer, cosine``); ``site`` is recorded in
    every row."""
    rep = sharing_report(mb, sae, groups, site=site, threshold=threshold, topk_only=topk_only, readout=readout,
                         layer_grid=layer_grid)
    key = rep["site"]
    rows = []
    for r in rep["sharing"].to_dict("records"):
        rows.append(dict(kind="sharing", site=key, a=r["a"], b=r["b"], jaccard=r["jaccard"], n_a=r["n_a"], n_b=r["n_b"],
                         n_shared=r["n_shared"], n_union=r["n_union"]))
    for r in rep["activity"].to_dict("records"):
        rows.append(dict(kind="activity", site=key, a=r["condition"], b=None,
                         **{k: r[k] for k in ("frac_active", "mean_mass", "participation_ratio", "n_active_mean",
                                              "frac_features_active", "n_rows")}))
    if rep["heads"] is not None:
        for r in rep["heads"].to_dict("records"):
            rows.append(dict(kind="heads", site=key, a=r["a"], b=r["b"], layer=r["layer"], cosine=r["cosine"]))
    cols = ["kind", "site", "a", "b", "jaccard", "n_a", "n_b", "n_shared", "n_union", "frac_active", "mean_mass",
            "participation_ratio", "n_active_mean", "frac_features_active", "n_rows", "layer", "cosine"]
    return pd.DataFrame(rows, columns=cols)
