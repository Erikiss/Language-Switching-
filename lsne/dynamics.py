"""Non-equilibrium diagnostics on the (layer x position) lattice (DESIGN.md section 4.2).

"Time" is depth: the L+1 residual snapshots of :func:`lsne.model.run_with_cache` with the
``"all_block_inputs"`` preset (block inputs 0..L-1 plus the output of block L-1). Space is
the token position. Implemented here:

* D1 damage spreading: a State-2143 dose at one lattice site, the relative perturbation
  ``D[l, t]`` on every later snapshot, its depth growth exponent and spreading front;
* D2 depth monotonicity: logit-lens KL trajectory to the final prediction and a
  non-monotonicity index;
* update-norm profile between consecutive snapshots;
* D9 cellular-automaton view: symbolisation of each site as N/E/D from logit-lens mass on
  language-marker tokens, with entropy / activity / LZ76 complexity measures.

All outputs are numpy arrays or plain floats; only the model calls use torch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .config import HookSite
from .model import CacheOut, Injector, ModelBundle, all_block_inputs, logit_lens, make_dose_delta, run_with_cache

_NORM_EPS = 1e-12
_D_CLIP = 1e-12
#: Symbols of the cellular-automaton view.
CA_N, CA_E, CA_D = 0, 1, 2


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _to_ids(mb: ModelBundle, ids) -> torch.Tensor:
    if isinstance(ids, torch.Tensor):
        t = ids.detach().long()
    else:
        t = torch.as_tensor([int(i) for i in np.asarray(ids).reshape(-1)], dtype=torch.long)
    if t.dim() == 1:
        t = t[None, :]
    if t.dim() != 2 or t.shape[0] != 1:
        raise ValueError(f"expected a single sequence, got shape {tuple(t.shape)}")
    return t.to(mb.device)


def _as_vec(direction) -> torch.Tensor:
    """Direction object (``.vec``), tensor or array -> float32 CPU vector."""
    v = getattr(direction, "vec", direction)
    if isinstance(v, np.ndarray):
        v = torch.as_tensor(v)
    return torch.as_tensor(v).detach().float().cpu().reshape(-1)


def snapshot_keys(mb: ModelBundle) -> list[str]:
    """Keys of the L+1 depth snapshots in order."""
    return [s.key for s in all_block_inputs(mb)]


def _snapshot_stack(cache: CacheOut, keys: Sequence[str]) -> torch.Tensor:
    """[n_keys, T, d] float32 CPU stack of the first batch element of each snapshot."""
    return torch.stack([cache.resid[k][0].detach().float().cpu() for k in keys], 0)


def _has_keys(cache: Optional[CacheOut], keys: Sequence[str]) -> bool:
    return cache is not None and all(k in cache.resid for k in keys)


def _ols(x, y) -> tuple[float, float, float]:
    """Ordinary least squares ``y = intercept + slope * x`` -> (slope, intercept, r2).
    NaN slope for fewer than two points or constant x; r2 = 1 for a constant y."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        return float("nan"), float("nan"), float("nan")
    xm, ym = x.mean(), y.mean()
    sxx = float(((x - xm) ** 2).sum())
    if sxx <= 0:
        return float("nan"), float("nan"), float("nan")
    slope = float(((x - xm) * (y - ym)).sum() / sxx)
    intercept = float(ym - slope * xm)
    ss_res = float(((y - (intercept + slope * x)) ** 2).sum())
    ss_tot = float(((y - ym) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, float(r2)


def _entropy_bits(counts) -> float:
    c = np.asarray(counts, dtype=float)
    c = c[c > 0]
    if c.size == 0:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


# --------------------------------------------------------------------------------------
# D1: damage spreading
# --------------------------------------------------------------------------------------
@dataclass
class DamageMap:
    """Relative perturbation of every lattice site after a dose at ``(inject_layer, inject_pos)``.

    ``D[l, t] = ||h'_{l,t} - h_{l,t}|| / (||h_{l,t}|| + 1e-12)`` and
    ``cos[l, t]`` = cosine similarity (overlap) between the perturbed and the reference
    state, both over the L+1 depth snapshots ``layers``. ``inject_layer`` is the snapshot
    index at which the injection first appears (``site.layer`` for a block input,
    ``site.layer + 1`` for a block / MLP / attention output). ``logp_target_ref`` /
    ``logp_target_pert`` are log p(target) at the last position before / after the dose
    (NaN without a target).
    """
    D: np.ndarray
    cos: np.ndarray
    layers: list[str]
    inject_layer: int
    inject_pos: int
    eps: float
    site: str = ""
    logp_target_ref: float = float("nan")
    logp_target_pert: float = float("nan")
    delta_norm: float = 0.0
    audit: dict = field(default_factory=dict)

    @property
    def n_snapshots(self) -> int:
        return int(self.D.shape[0])

    @property
    def n_positions(self) -> int:
        return int(self.D.shape[1])


def damage_spreading(mb: ModelBundle, input_ids, site: HookSite, position: int, direction,
                     eps: float, ref_cache: Optional[CacheOut] = None,
                     target_id: Optional[int] = None) -> DamageMap:
    """Inject ``make_dose_delta(h_ref[position], direction, eps)`` at ``site``/``position``
    and measure the relative perturbation of every (snapshot, position) site.

    ``ref_cache`` is the unperturbed ``"all_block_inputs"`` cache of the same input (it is
    computed, or recomputed when it lacks the snapshot keys). Snapshots before the
    injection layer are identical by construction (D = 0) and, by causality, positions
    before ``position`` are unperturbed everywhere.
    """
    ids = _to_ids(mb, input_ids)
    T = int(ids.shape[1])
    position = int(position)
    if not (0 <= position < T):
        raise ValueError(f"position {position} outside [0, {T})")
    keys = snapshot_keys(mb)
    if not _has_keys(ref_cache, keys):
        ref_cache = run_with_cache(mb, ids, "all_block_inputs")
    # reference residual at the injection site (block inputs/outputs are snapshots; other
    # sites need their own recording)
    if site.key in ref_cache.resid:
        h_site = ref_cache.resid[site.key]
    else:
        h_site = run_with_cache(mb, ids, [site]).resid[site.key]
    h_ref = h_site[0, position].detach()
    delta = make_dose_delta(h_ref, _as_vec(direction).to(h_ref.device), float(eps)).to(h_ref.dtype)

    inj = Injector(mb, site, delta, positions=[position], mode="add")
    with inj:
        pert_cache = run_with_cache(mb, ids, "all_block_inputs")

    H0 = _snapshot_stack(ref_cache, keys)        # [L+1, T, d]
    H1 = _snapshot_stack(pert_cache, keys)
    diff = H1 - H0
    n0 = H0.norm(dim=-1)
    D = (diff.norm(dim=-1) / (n0 + _NORM_EPS)).numpy().astype(np.float64)
    cos = torch.nn.functional.cosine_similarity(H1, H0, dim=-1, eps=_NORM_EPS).numpy().astype(np.float64)
    cos[diff.norm(dim=-1).numpy() == 0.0] = 1.0

    inject_layer = int(site.layer) if site.site == "block_input" else int(site.layer) + 1
    lp_ref = lp_pert = float("nan")
    if target_id is not None:
        lp_ref = float(torch.log_softmax(ref_cache.logits[0, -1].float(), -1)[int(target_id)])
        lp_pert = float(torch.log_softmax(pert_cache.logits[0, -1].float(), -1)[int(target_id)])
    return DamageMap(D=D, cos=cos, layers=list(keys), inject_layer=inject_layer, inject_pos=position,
                     eps=float(eps), site=site.key, logp_target_ref=lp_ref, logp_target_pert=lp_pert,
                     delta_norm=float(delta.float().norm()), audit=dict(inj.audit))


def growth_exponent(dm: DamageMap, positions: Sequence[int], layer_from: Optional[int] = None) -> dict:
    """OLS slope ``lam`` of ``ln(mean_t D[l, t])`` (t in ``positions``) against the snapshot
    index l, for ``l = layer_from .. L`` (default ``layer_from = inject_layer + 1``).
    Also per-position slopes and the r2 of the pooled fit. D is clipped at 1e-12 before
    the logarithm. ``lam > 0`` = growing perturbation (non-equilibrium signature D1)."""
    positions = [int(p) for p in positions]
    if not positions:
        raise ValueError("positions must not be empty")
    L_last = dm.D.shape[0] - 1
    start = int(dm.inject_layer + 1 if layer_from is None else layer_from)
    start = max(0, start)
    layers = np.arange(start, L_last + 1)
    sub = np.clip(dm.D[start:, positions], _D_CLIP, None)       # [n_layers, n_pos]
    if layers.size == 0:
        return dict(lam=float("nan"), intercept=float("nan"), r2=float("nan"),
                    per_position=np.full(len(positions), np.nan), layers=layers)
    y = np.log(sub.mean(axis=1))
    lam, intercept, r2 = _ols(layers, y)
    per_pos = np.array([_ols(layers, np.log(sub[:, j]))[0] for j in range(len(positions))], dtype=float)
    return dict(lam=lam, intercept=intercept, r2=r2, per_position=per_pos, layers=layers,
                log_mean_D=y)


def spreading_front(dm: DamageMap, threshold: float) -> np.ndarray:
    """Per snapshot: the largest distance ``t - inject_pos`` at which ``D > threshold``
    (0 when no site exceeds the threshold)."""
    out = np.zeros(dm.D.shape[0], dtype=np.int64)
    for l in range(dm.D.shape[0]):
        idx = np.nonzero(dm.D[l] > threshold)[0]
        if idx.size:
            out[l] = int(idx.max()) - int(dm.inject_pos)
    return out


# --------------------------------------------------------------------------------------
# D2: logit-lens trajectory
# --------------------------------------------------------------------------------------
def _lens_logp(mb: ModelBundle, resid: torch.Tensor) -> torch.Tensor:
    """log-softmax of the logit lens of ``resid`` [..., d] -> [..., V] float32 CPU."""
    return torch.log_softmax(logit_lens(mb, resid), dim=-1).cpu()


def logit_lens_trajectory(mb: ModelBundle, cache: CacheOut, target_id=None,
                          positions: Optional[Sequence[int]] = None) -> dict:
    """Logit-lens trajectory over the depth snapshots of ``cache``.

    ``kl[l, t] = KL(p_final(t) || p_l(t))`` where ``p_l`` is the lens distribution of
    snapshot l and ``p_final`` that of the last snapshot (identical to the model's
    logits), so ``kl[-1] == 0``. ``target_id`` may be an int (same target at every
    position, e.g. the next token of the last position), a sequence of per-position
    target ids (length = number of selected positions) or None (NaN target arrays).
    ``target_rank`` is 0-based (0 = top-1). Returns dict(kl, target_rank, target_logp,
    positions, layers), each array of shape [n_snapshots, n_positions].
    """
    keys = [s.key for s in cache.sites] if cache.sites else list(cache.resid.keys())
    if not keys:
        raise ValueError("cache holds no residual snapshots")
    T = int(cache.resid[keys[0]].shape[1])
    pos = list(range(T)) if positions is None else [int(p) for p in positions]
    n_l, n_p = len(keys), len(pos)
    kl = np.zeros((n_l, n_p), dtype=np.float64)
    rank = np.full((n_l, n_p), np.nan)
    tlogp = np.full((n_l, n_p), np.nan)
    tgt = None
    if target_id is not None:
        if isinstance(target_id, (int, np.integer)) or (isinstance(target_id, torch.Tensor) and target_id.dim() == 0):
            tgt = torch.full((n_p,), int(target_id), dtype=torch.long)
        else:
            tgt = torch.as_tensor([int(t) for t in np.asarray(target_id).reshape(-1)], dtype=torch.long)
            if tgt.numel() != n_p:
                raise ValueError(f"target_id has {tgt.numel()} entries for {n_p} positions")
    with torch.no_grad():
        logp_final = _lens_logp(mb, cache.resid[keys[-1]][0, pos])          # [P, V]
        p_final = logp_final.exp()
        for l, k in enumerate(keys):
            logp_l = _lens_logp(mb, cache.resid[k][0, pos])
            kl[l] = (p_final * (logp_final - logp_l)).sum(-1).clamp_min(0.0).numpy()
            if tgt is not None:
                lt = logp_l.gather(-1, tgt[:, None]).squeeze(-1)             # [P]
                tlogp[l] = lt.numpy()
                rank[l] = (logp_l > lt[:, None]).sum(-1).numpy()
    return dict(kl=kl, target_rank=rank, target_logp=tlogp, positions=np.asarray(pos), layers=list(keys))


def nonmonotonicity_index(series) -> float:
    """Sum of positive increments divided by the total variation of a 1-D series
    (0 for a monotonically decreasing series, 1 for a monotonically increasing one, 0 when
    the total variation vanishes). NaN increments are ignored."""
    s = np.asarray(series, dtype=float).reshape(-1)
    if s.size < 2:
        return 0.0
    d = np.diff(s)
    d = d[np.isfinite(d)]
    tv = float(np.abs(d).sum())
    if tv <= 0:
        return 0.0
    return float(d[d > 0].sum() / tv)


def update_norm_profile(cache: CacheOut) -> np.ndarray:
    """Relative update norms ``||h_{l+1} - h_l|| / (||h_l|| + 1e-12)`` between consecutive
    snapshots of ``cache`` (first batch element) -> ndarray [n_snapshots - 1, T]."""
    keys = [s.key for s in cache.sites] if cache.sites else list(cache.resid.keys())
    H = _snapshot_stack(cache, keys)                                  # [L+1, T, d]
    upd = (H[1:] - H[:-1]).norm(dim=-1) / (H[:-1].norm(dim=-1) + _NORM_EPS)
    return upd.numpy().astype(np.float64)


# --------------------------------------------------------------------------------------
# D9: cellular-automaton view
# --------------------------------------------------------------------------------------
def _marker_ids(mb: ModelBundle, ids) -> torch.Tensor:
    if ids is None:
        return torch.zeros(0, dtype=torch.long)
    if isinstance(ids, torch.Tensor):
        arr = ids.detach().reshape(-1).long().cpu().numpy()
    else:
        arr = np.asarray([int(i) for i in np.asarray(list(ids)).reshape(-1)], dtype=np.int64)
    arr = np.unique(arr[(arr >= 0) & (arr < mb.vocab)])
    return torch.as_tensor(arr, dtype=torch.long)


def lexicon_from_axis(mb: ModelBundle, lang_axis, top_k: int = 25) -> dict:
    """Marker tokens derived from a language axis through the logit lens: the ``top_k``
    tokens promoted by ``+axis`` (relative to ``-axis``) are 'D' markers, the ``top_k``
    most suppressed are 'E' markers (the axis is oriented D - E, see ``lang_dom``)."""
    v = _as_vec(lang_axis).to(mb.device)
    with torch.no_grad():
        score = (logit_lens(mb, v) - logit_lens(mb, -v)).reshape(-1)
    order = torch.argsort(score, descending=True)
    return dict(D=order[:top_k].cpu().tolist(), E=order[-top_k:].cpu().tolist())


def ca_symbolize(mb: ModelBundle, cache: CacheOut, lexicon_ids=None, lang_axis=None,
                 thresholds: Sequence[float] = (0.5,)) -> np.ndarray:
    """Symbolise every lattice site (snapshot, position) as N=0 / E=1 / D=2.

    Per site the logit-lens probability mass on the English marker ids ``lexicon_ids['E']``
    (``mass_E``) and on the German markers ``lexicon_ids['D']`` (``mass_D``) is computed;
    the symbol is D if ``mass_D > thr * (mass_E + mass_D)`` and ``mass_D > mass_E``, E if
    the same holds for ``mass_E``, else N. With no usable lexicon and a ``lang_axis``
    (Direction or vector) the markers are derived from the axis through the unembedding
    (:func:`lexicon_from_axis`). Returns int8 ``[n_snapshots, T]`` for a single threshold
    and ``[n_thresholds, n_snapshots, T]`` for several.
    """
    # tolerate the SPEC argument order (lang_axis before lexicon_ids)
    if isinstance(lang_axis, dict) and not isinstance(lexicon_ids, dict):
        lexicon_ids, lang_axis = lang_axis, lexicon_ids
    lex = dict(lexicon_ids) if isinstance(lexicon_ids, dict) else {}
    E = _marker_ids(mb, lex.get("E"))
    Dm = _marker_ids(mb, lex.get("D"))
    if (E.numel() == 0 or Dm.numel() == 0) and lang_axis is not None:
        lex = lexicon_from_axis(mb, lang_axis)
        E, Dm = _marker_ids(mb, lex["E"]), _marker_ids(mb, lex["D"])
    if E.numel() == 0 or Dm.numel() == 0:
        raise ValueError("ca_symbolize needs non-empty 'E' and 'D' marker id lists (or a lang_axis)")
    thr = [float(t) for t in thresholds]
    if not thr:
        raise ValueError("thresholds must not be empty")
    keys = [s.key for s in cache.sites] if cache.sites else list(cache.resid.keys())
    T = int(cache.resid[keys[0]].shape[1])
    S = np.zeros((len(thr), len(keys), T), dtype=np.int8)
    with torch.no_grad():
        for l, k in enumerate(keys):
            p = torch.softmax(logit_lens(mb, cache.resid[k][0]), dim=-1).cpu()   # [T, V]
            mE = p[:, E].sum(-1).numpy()
            mD = p[:, Dm].sum(-1).numpy()
            tot = mE + mD
            for i, th in enumerate(thr):
                is_D = (mD > th * tot) & (mD > mE)
                is_E = (mE > th * tot) & (mE > mD)
                S[i, l][is_D] = CA_D
                S[i, l][is_E] = CA_E
    return S[0] if len(thr) == 1 else S


def lz76(seq: Sequence[int]) -> int:
    """Lempel-Ziv 1976 complexity (number of distinct phrases) of a symbol sequence,
    Kaspar & Schuster (1987) algorithm. ``lz76([]) == 0``; a constant sequence of length
    n >= 2 has complexity 2; a random sequence approaches ``n / log_b(n)``."""
    s = [int(x) for x in np.asarray(list(seq)).reshape(-1)]
    n = len(s)
    if n == 0:
        return 0
    if n == 1:
        return 1
    i, k, l = 0, 1, 1
    c, k_max = 1, 1
    while True:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
            if l + k > n:
                c += 1
                break
        else:
            if k > k_max:
                k_max = k
            i += 1
            if i == l:
                c += 1
                l += k_max
                if l + 1 > n:
                    break
                i, k, k_max = 0, 1, 1
            else:
                k = 1
    return int(c)


def ca_complexity(S: np.ndarray) -> dict:
    """Complexity measures of a symbol field ``S[n_snapshots, T]`` (values 0/1/2).

    * ``spatial_entropy``: mean over snapshots of the symbol entropy (bits) across positions;
    * ``temporal_entropy``: mean over positions of the symbol entropy across snapshots;
    * ``activity``: fraction of sites whose symbol differs from the snapshot below;
    * ``lz76``: LZ76 complexity of the row-major flattened field, normalised by
      ``n / log2(n)`` (about 1 for a random field, near 0 for a constant one);
    * ``block_entropy_2x2``: entropy (bits) of the distribution of 2x2 block patterns.
    """
    S = np.asarray(S)
    if S.ndim != 2:
        raise ValueError("S must be 2-D [n_snapshots, T]")
    n_l, T = S.shape
    symbols = np.unique(S)
    spatial = float(np.mean([_entropy_bits([(S[l] == s).sum() for s in symbols]) for l in range(n_l)])) if n_l else 0.0
    temporal = float(np.mean([_entropy_bits([(S[:, t] == s).sum() for s in symbols]) for t in range(T)])) if T else 0.0
    activity = float(np.mean(S[1:] != S[:-1])) if n_l >= 2 and T else 0.0
    n = S.size
    if n >= 2:
        lz = float(lz76(S.reshape(-1)) / (n / np.log2(n)))
    else:
        lz = 0.0
    if n_l >= 2 and T >= 2:
        blocks: dict[tuple, int] = {}
        for l in range(n_l - 1):
            for t in range(T - 1):
                key = (int(S[l, t]), int(S[l, t + 1]), int(S[l + 1, t]), int(S[l + 1, t + 1]))
                blocks[key] = blocks.get(key, 0) + 1
        block_ent = _entropy_bits(list(blocks.values()))
    else:
        block_ent = 0.0
    return dict(spatial_entropy=spatial, temporal_entropy=temporal, activity=activity,
                lz76=lz, block_entropy_2x2=float(block_ent))
