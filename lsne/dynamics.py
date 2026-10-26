"""Non-equilibrium diagnostics on the (layer x position) lattice (DESIGN.md section 4.2).

"Time" is depth: the L+1 residual snapshots of :func:`lsne.model.run_with_cache` with the
``"all_block_inputs"`` preset (block inputs 0..L-1 plus the output of block L-1). Space is
the token position. Implemented here:

* D1 damage spreading: a State-2143 dose at one lattice site, the relative perturbation
  ``D[l, t]`` on every later snapshot, its depth growth exponent and spreading front
  (one map per forward: :func:`damage_spreading`; many (direction, dose) pairs of one
  input in batched forwards with the reference as row 0: :func:`damage_spreading_batch`);
* D2 depth monotonicity: logit-lens KL trajectory to the final prediction and a
  non-monotonicity index;
* update-norm profile between consecutive snapshots;
* D9 cellular-automaton view: symbolisation of each site as N/E/D from logit-lens mass on
  language-marker tokens (E/D only above an absolute marker-mass floor, otherwise N),
  with entropy / activity / LZ76 complexity measures.

All outputs are numpy arrays or plain floats; only the model calls use torch.
"""
from __future__ import annotations

import contextlib
import inspect
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

from .config import HookSite
from .model import (CacheOut, Injector, ModelBundle, _first_tensor, _get_hidden, all_block_inputs, logit_lens,
                    make_dose_delta, run_with_cache)

_NORM_EPS = 1e-12
_D_CLIP = 1e-12
#: Default rows per forward of :func:`damage_spreading_batch` (the unperturbed reference
#: row plus up to 63 perturbed copies).  Peak memory of one chunk = the model's forward
#: activations for ``batch_size x T`` tokens (logits only for the last position) plus two
#: or three ``[batch_size, T, d]`` float32 temporaries of the per-snapshot D / cos
#: reduction; no residual snapshot is stored.  Per row at T = 320, float32, eager
#: attention (the [n_heads, T, T] scores twice, the [T, 4d] MLP intermediate twice, a few
#: [T, d] buffers): measured 20.5 MiB for pythia-160m and 15 MiB for pythia-70m (CPU peak
#: RSS slope), estimated ~27 MiB for 410m and ~40 MiB for 1b / 1.4b.  64 rows therefore
#: need ~1.3 GiB (160m) to ~2.6 GiB (1.4b) on top of the weights (0.65 GB / 5.6 GB): far
#: below 40 GB on an A100 for every Pythia model up to 1.4b, and 64 x 150-320 tokens keep
#: the GPU busy even for pythia-160m (the stage-D variants have T = 94..182).  On CPU the
#: batch size hardly matters (16, 64 and 128 rows give the same throughput).
DAMAGE_BATCH_SIZE = 64
#: Symbols of the cellular-automaton view.
CA_N, CA_E, CA_D = 0, 1, 2
#: Default absolute marker-mass floor of :func:`ca_symbolize`, as a multiple of the
#: uniform lens baseline ``n_markers / vocab``: a site is labelled E or D only if the
#: logit lens puts at least this many times the uniform mass on the marker tokens.
CA_MASS_FLOOR_FACTOR = 10.0


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


class _BatchDose:
    """Callable delta for :class:`~lsne.model.Injector` used by :func:`damage_spreading_batch`.

    Row 0 of the chunk ``h [B, T, d]`` is the unperturbed reference and stays untouched;
    row ``1 + i`` receives ``make_dose_delta(h[0, position], vecs[i], eps[i])`` at
    ``position`` - exactly the delta of :func:`damage_spreading`, built from the reference
    residual of the same forward.  ``row_audit`` holds the Soll/Ist record of every row
    (``Injector`` semantics: intended = the delta in the model dtype, written = h_new - h in
    float32, ``rel_err = ||written - intended|| / ||intended||``); ``audit`` is the record of
    the whole chunk (adopted by the Injector)."""

    def __init__(self, vecs: torch.Tensor, eps: Sequence[float], position: int):
        self.vecs = vecs
        self.eps = [float(e) for e in eps]
        self.position = int(position)
        self.audit: dict = {}
        self.row_audit: list[dict] = []
        self.delta_norms: list[float] = []

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        n = len(self.eps)
        if h.shape[0] != n + 1:
            raise RuntimeError(f"chunk of {h.shape[0]} rows for {n} doses + the reference row")
        p = self.position
        h_ref = h[0, p].detach()
        vecs = self.vecs.to(h_ref.device)
        delta = torch.stack([make_dose_delta(h_ref, vecs[i], self.eps[i]) for i in range(n)]).to(h.dtype)   # [n, d]
        h_new = h.clone()
        h_new[1:, p, :] = h_new[1:, p, :] + delta
        with torch.no_grad():
            intended = delta.float()
            written = h_new[1:, p, :].float() - h[1:, p, :].float()
            i_norm = intended.norm(dim=-1)
            w_norm = written.norm(dim=-1)
            err = (written - intended).norm(dim=-1)
            rel = err / i_norm.clamp_min(1e-30)
            dt = str(h.dtype)
            self.delta_norms = [float(x) for x in i_norm.cpu().tolist()]
            self.row_audit = [dict(intended_norm=float(a), written_norm=float(b), rel_err=float(c), dtype=dt)
                              for a, b, c in zip(i_norm.cpu().tolist(), w_norm.cpu().tolist(), rel.cpu().tolist())]
            tot = intended.norm()
            self.audit = dict(intended_norm=float(tot), written_norm=float(written.norm()),
                              rel_err=float((written - intended).norm() / tot.clamp_min(1e-30)), dtype=dt, n_rows=n)
        return h_new


@contextlib.contextmanager
def _damage_recording(mb: ModelBundle, sites: Sequence[HookSite], D_out: torch.Tensor, cos_out: torch.Tensor):
    """Hooks that reduce every depth snapshot of a chunk on the fly against its row 0:
    ``D_out[i, l, t]`` / ``cos_out[i, l, t]`` for perturbed row ``1 + i`` (same formulas as
    :func:`damage_spreading`, float32 on the model device); no snapshot is stored."""
    handles = []

    def reduce(h: torch.Tensor, l: int):
        H = h.detach().float()                                   # [B, T, d]
        ref = H[:1]
        dn = (H[1:] - ref).norm(dim=-1)                          # [B-1, T]
        D_out[:, l] = dn / (ref.norm(dim=-1) + _NORM_EPS)
        c = torch.nn.functional.cosine_similarity(H[1:], ref, dim=-1, eps=_NORM_EPS)
        cos_out[:, l] = torch.where(dn == 0, torch.ones_like(c), c)

    try:
        for l, s in enumerate(sites):
            mod = mb.module_for(s)
            if s.site == "block_input":
                def pre(module, args, kwargs, _l=l):
                    reduce(_get_hidden(args, kwargs), _l)
                handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(module, args, out, _l=l):
                    reduce(_first_tensor(out), _l)
                handles.append(mod.register_forward_hook(post))
        yield
    finally:
        for hd in handles:
            hd.remove()


def _last_logits_kwargs(mb: ModelBundle) -> dict:
    """Forward kwarg that restricts the LM head to the last position (``logits_to_keep=1``
    in transformers >= 4.50 incl. 5.x; ``num_logits_to_keep`` in older versions; empty if
    neither exists, then the full logits are computed and sliced)."""
    try:
        params = inspect.signature(mb.model.forward).parameters
    except (TypeError, ValueError):
        return {}
    for name in ("logits_to_keep", "num_logits_to_keep"):
        if name in params:
            return {name: 1}
    return {}


def last_logits(mb: ModelBundle, input_ids: torch.Tensor) -> torch.Tensor:
    """``[B, V]`` float32 logits of the last position of ``input_ids [B, T]`` with the LM
    head applied to that position only (``logits_to_keep=1``; without that kwarg the final
    hidden state of the base model goes through the head at the last position).  Runs
    inside the caller's grad mode; hooks on the blocks fire as in a full forward."""
    ids = input_ids.to(mb.device)
    kw = _last_logits_kwargs(mb)
    if kw:
        return mb.model(input_ids=ids, use_cache=False, **kw).logits[:, -1].float()
    hs = mb.model.base_model(input_ids=ids, use_cache=False).last_hidden_state
    return mb.lm_head(hs[:, -1]).float()


def _stack_dirs(directions) -> torch.Tensor:
    """[N, d] float32 CPU stack of direction objects / vectors / an [N, d] array."""
    if isinstance(directions, (torch.Tensor, np.ndarray)) and np.ndim(directions) == 2:
        return torch.as_tensor(directions).detach().float().cpu()
    vecs = [_as_vec(d) for d in directions]
    if not vecs:
        return torch.zeros(0, 0)
    return torch.stack(vecs, 0)


def damage_spreading_batch(mb: ModelBundle, input_ids, site: HookSite, position: int, directions, eps,
                           target_id: Optional[int] = None, batch_size: int = DAMAGE_BATCH_SIZE) -> list[DamageMap]:
    """Batched :func:`damage_spreading`: N (direction, dose) pairs of one input.

    ``directions`` is an ``[N, d]`` tensor / array or a sequence of N directions (objects
    with ``.vec``, tensors or arrays); ``eps`` a float (same dose for every row) or N
    floats.  The N perturbed copies run in chunks of ``batch_size`` rows (``batch_size - 1``
    perturbed copies per forward); **row 0 of every chunk is the unperturbed reference**,
    so reference and perturbed states go through the same kernels: snapshots before the
    injection layer and, by causality, positions before ``position`` come out exactly
    ``D = 0``.  Row ``1 + i`` gets ``make_dose_delta(h_ref[position], dir_i, eps_i)`` with
    ``h_ref`` the residual of row 0 at ``site``.  ``D`` / ``cos`` are reduced per snapshot
    on the model device while the forward runs (no snapshot is kept) and only the
    ``[N, L+1, T]`` arrays move to the CPU; the LM head runs on the last position only.

    Returns N :class:`DamageMap` in input order, with the meaning of the sequential maps
    (``logp_target_ref`` from the reference row of the row's own chunk, ``audit`` = the
    Soll/Ist record of the row: ``intended_norm``, ``written_norm``, ``rel_err``, ``dtype``,
    plus ``chunk_rows`` / ``chunk_rel_err`` of its forward).  Agrees with N calls of
    :func:`damage_spreading` up to float32 round-off (batch-size dependent GEMM blocking).
    Memory of a chunk: see :data:`DAMAGE_BATCH_SIZE`.
    """
    ids = _to_ids(mb, input_ids)
    T = int(ids.shape[1])
    position = int(position)
    if not (0 <= position < T):
        raise ValueError(f"position {position} outside [0, {T})")
    batch_size = int(batch_size)
    if batch_size < 2:
        raise ValueError(f"batch_size must be >= 2 (reference row + one perturbed row), got {batch_size}")
    vecs = _stack_dirs(directions)
    N = int(vecs.shape[0])
    if np.ndim(eps) == 0:
        eps_l = [float(eps)] * N
    else:
        eps_l = [float(e) for e in np.asarray(eps, dtype=float).reshape(-1)]
        if len(eps_l) != N:
            raise ValueError(f"{len(eps_l)} doses for {N} directions")
    if N == 0:
        return []
    if vecs.shape[1] != mb.d_model:
        raise ValueError(f"directions have dimension {vecs.shape[1]}, model has {mb.d_model}")
    sites = all_block_inputs(mb)
    keys = [s.key for s in sites]
    n_snap = len(sites)
    inject_layer = int(site.layer) if site.site == "block_input" else int(site.layer) + 1
    logit_kw = _last_logits_kwargs(mb)
    vecs = vecs.to(mb.device)
    out: list[DamageMap] = []
    step = batch_size - 1
    for c0 in range(0, N, step):
        idx = list(range(c0, min(N, c0 + step)))
        n = len(idx)
        dose = _BatchDose(vecs[idx[0]:idx[-1] + 1], [eps_l[i] for i in idx], position)
        D_dev = torch.zeros((n, n_snap, T), dtype=torch.float32, device=mb.device)
        cos_dev = torch.ones((n, n_snap, T), dtype=torch.float32, device=mb.device)
        chunk_ids = ids.expand(n + 1, T).contiguous()
        # injector first: at a shared module its hook runs before the recording hook, so the
        # snapshot at the injection layer already carries the dose (as in damage_spreading)
        with torch.no_grad(), Injector(mb, site, dose, mode="add"), _damage_recording(mb, sites, D_dev, cos_dev):
            res = mb.model(input_ids=chunk_ids, attention_mask=None, use_cache=False, **logit_kw)
        lp_ref = [float("nan")] * (n + 1)
        if target_id is not None:
            lp = torch.log_softmax(res.logits[:, -1].float(), -1)[:, int(target_id)]
            lp_ref = [float(x) for x in lp.cpu().tolist()]
        del res
        D_np = D_dev.cpu().numpy().astype(np.float64)
        cos_np = cos_dev.cpu().numpy().astype(np.float64)
        del D_dev, cos_dev
        for j, i in enumerate(idx):
            audit = dict(dose.row_audit[j], chunk_rows=n + 1, chunk_rel_err=float(dose.audit.get("rel_err", float("nan"))))
            out.append(DamageMap(D=D_np[j].copy(), cos=cos_np[j].copy(), layers=list(keys), inject_layer=inject_layer,
                                 inject_pos=position, eps=eps_l[i], site=site.key, logp_target_ref=lp_ref[0],
                                 logp_target_pert=lp_ref[1 + j], delta_norm=dose.delta_norms[j], audit=audit))
    return out


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


def resolve_markers(mb: ModelBundle, lexicon_ids=None, lang_axis=None) -> dict:
    """The marker lexicon :func:`ca_symbolize` will use: ``dict(E=[ids], D=[ids])`` from
    ``lexicon_ids`` or, if that is unusable, derived from ``lang_axis``
    (:func:`lexicon_from_axis`)."""
    E, Dm = _resolve_markers(mb, lexicon_ids, lang_axis)
    return dict(E=[int(i) for i in E.tolist()], D=[int(i) for i in Dm.tolist()])


def _resolve_markers(mb: ModelBundle, lexicon_ids, lang_axis) -> tuple[torch.Tensor, torch.Tensor]:
    """English / German marker id tensors from a lexicon dict or, failing that, a language
    axis (:func:`lexicon_from_axis`); raises if either list ends up empty."""
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
    return E, Dm


def ca_mass_floor(mass_floor, n_markers: int, vocab: int) -> float:
    """Absolute marker-mass floor used by :func:`ca_symbolize`.

    ``mass_floor`` is either a number (absolute probability mass, ``0`` disables the
    floor) or ``"auto"`` = ``CA_MASS_FLOOR_FACTOR * n_markers / vocab``, i.e. a fixed
    multiple of the mass a *uniform* lens distribution would put on the ``n_markers``
    marker tokens (for 12 markers and the Pythia vocabulary about 2.4e-3)."""
    if isinstance(mass_floor, str):
        if mass_floor != "auto":
            raise ValueError(f"mass_floor must be a number or 'auto', got {mass_floor!r}")
        return float(CA_MASS_FLOOR_FACTOR * int(n_markers) / max(int(vocab), 1))
    f = float(mass_floor)
    if not np.isfinite(f) or f < 0:
        raise ValueError(f"mass_floor must be finite and >= 0, got {mass_floor!r}")
    return f


def ca_marker_mass(mb: ModelBundle, cache: CacheOut, lexicon_ids=None, lang_axis=None) -> tuple[np.ndarray, np.ndarray]:
    """Logit-lens probability mass on the English and German marker tokens at every
    lattice site: ``(mass_E, mass_D)``, each ``[n_snapshots, T]`` float64."""
    E, Dm = _resolve_markers(mb, lexicon_ids, lang_axis)
    keys = [s.key for s in cache.sites] if cache.sites else list(cache.resid.keys())
    T = int(cache.resid[keys[0]].shape[1])
    mE = np.zeros((len(keys), T), dtype=np.float64)
    mD = np.zeros((len(keys), T), dtype=np.float64)
    with torch.no_grad():
        for l, k in enumerate(keys):
            p = torch.softmax(logit_lens(mb, cache.resid[k][0]), dim=-1).cpu()   # [T, V]
            mE[l] = p[:, E].sum(-1).numpy()
            mD[l] = p[:, Dm].sum(-1).numpy()
    return mE, mD


def symbolize_masses(mass_E: np.ndarray, mass_D: np.ndarray, thresholds: Sequence[float] = (0.5,),
                     mass_floor: float = 0.0) -> np.ndarray:
    """Symbol field from marker masses (pure numpy part of :func:`ca_symbolize`).

    A site is D if ``mass_D > thr * (mass_E + mass_D)``, ``mass_D > mass_E`` **and**
    ``mass_E + mass_D >= mass_floor``; E symmetrically; otherwise N.  The absolute floor
    is what makes N a real symbol: without it the rule reduces to ``sign(mass_D -
    mass_E)`` and sites where the lens puts negligible mass on *both* marker sets (early
    snapshots, mid-word positions) are labelled by noise.  Returns int8 arrays shaped
    like the inputs for one threshold, with a leading threshold axis for several."""
    mE = np.asarray(mass_E, dtype=np.float64)
    mD = np.asarray(mass_D, dtype=np.float64)
    if mE.shape != mD.shape:
        raise ValueError(f"mass arrays differ in shape: {mE.shape} vs {mD.shape}")
    thr = [float(t) for t in thresholds]
    if not thr:
        raise ValueError("thresholds must not be empty")
    floor = float(mass_floor)
    tot = mE + mD
    enough = tot >= floor if floor > 0 else np.ones_like(tot, dtype=bool)
    S = np.zeros((len(thr),) + mE.shape, dtype=np.int8)
    for i, th in enumerate(thr):
        is_D = (mD > th * tot) & (mD > mE) & enough
        is_E = (mE > th * tot) & (mE > mD) & enough
        S[i][is_D] = CA_D
        S[i][is_E] = CA_E
    return S[0] if len(thr) == 1 else S


def ca_symbolize(mb: ModelBundle, cache: CacheOut, lexicon_ids=None, lang_axis=None,
                 thresholds: Sequence[float] = (0.5,), mass_floor="auto") -> np.ndarray:
    """Symbolise every lattice site (snapshot, position) as N=0 / E=1 / D=2.

    Per site the logit-lens probability mass on the English marker ids ``lexicon_ids['E']``
    (``mass_E``) and on the German markers ``lexicon_ids['D']`` (``mass_D``) is computed;
    the symbol is D if ``mass_D > thr * (mass_E + mass_D)``, ``mass_D > mass_E`` and the
    total marker mass reaches the absolute floor ``mass_E + mass_D >= mass_floor``; E if
    the same holds for ``mass_E``; else N (no marker evidence).  ``mass_floor`` is an
    absolute probability mass or ``"auto"`` (default: :data:`CA_MASS_FLOOR_FACTOR` times
    the uniform baseline ``n_markers / vocab``, see :func:`ca_mass_floor`); ``0`` restores
    the floor-less rule, under which N only occurs for an exact tie.  With no usable
    lexicon and a ``lang_axis`` (Direction or vector) the markers are derived from the
    axis through the unembedding (:func:`lexicon_from_axis`).  Returns int8
    ``[n_snapshots, T]`` for a single threshold and ``[n_thresholds, n_snapshots, T]``
    for several.
    """
    E, Dm = _resolve_markers(mb, lexicon_ids, lang_axis)
    lex = dict(E=E.tolist(), D=Dm.tolist())
    mE, mD = ca_marker_mass(mb, cache, lex)
    floor = ca_mass_floor(mass_floor, E.numel() + Dm.numel(), mb.vocab)
    return symbolize_masses(mE, mD, thresholds, floor)


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
