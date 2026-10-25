"""Intervention directions in residual space (DESIGN.md section 3).

Every direction is a :class:`Direction`: a unit-L2 float32 CPU vector attached to a
:class:`~lsne.config.HookSite` plus a ``meta`` dict recording how it was built.

Constructions implemented here:

* ``ent``        - entropy gradient at a site/position (State-2143 construction), optionally
                   orthogonalised against an external axis (the dossier's ARI axis).
* ``lang_dom``   - difference of means D - E over segment-final residuals, "denoised" by
                   projecting out the top *shared* principal component(s), i.e. the PCs of
                   the within-class centred data (the Pain-Axis recipe of Tagliabue et al.
                   2026, DESIGN section 3). The PCs are deliberately NOT taken from the pooled
                   data: the pooled covariance contains the between-class term
                   ``n_D n_E / N^2 * diff diff^T``, so a strong language difference would be
                   PC1 and denoising would delete the signal instead of shared nuisance.
* ``lang_probe`` - L2-regularised logistic-regression weight vector E vs D (control).
* ``freq``       - mean unembedding row of the most frequent tokens minus the mean row
                   (first-order "frequency axis"; ignores the final LayerNorm).
* ``rand``       - isotropic random unit vectors (null model).

Plus the geometry report helpers ``cosine_table`` and ``unembed_projection`` and the
activation collector ``collect_activations`` shared by the axis constructions.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from .config import HookSite
from .model import ModelBundle, entropy_gradient, make_dose_delta, rms, run_with_cache

_EPS = 1e-30

# Gate for the denoised language axis: if the cosine between the denoised and the plain
# difference of means falls below this, denoising has removed a large part of the measured
# language separation (the removed "shared" PC coincides with the class difference) and
# ``language_axis`` warns (DESIGN section 3; fix round 1, finding 1).
DENOISE_GATE_COS = 0.7


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def unit(v: torch.Tensor) -> torch.Tensor:
    """Return ``v`` as a float32 CPU vector of unit L2 norm (raises on a zero vector)."""
    v = torch.as_tensor(v).detach().float().cpu().reshape(-1)
    n = float(v.norm())
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("cannot normalise a zero / non-finite vector")
    return v / n


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two vectors (float32, CPU)."""
    a = torch.as_tensor(a).detach().float().cpu().reshape(-1)
    b = torch.as_tensor(b).detach().float().cpu().reshape(-1)
    return float(a @ b / (a.norm() * b.norm()).clamp_min(_EPS))


def _as_vec(x: Union["Direction", torch.Tensor, np.ndarray]) -> torch.Tensor:
    """Accept a Direction or a raw vector and return the float32 CPU vector."""
    if isinstance(x, Direction):
        return x.vec
    return torch.as_tensor(np.asarray(x) if isinstance(x, np.ndarray) else x).detach().float().cpu().reshape(-1)


# --------------------------------------------------------------------------------------
# Direction
# --------------------------------------------------------------------------------------
@dataclass
class Direction:
    """A named unit direction in residual space at a hook site.

    ``vec`` is always stored as a unit-L2 float32 CPU tensor (normalised on construction);
    ``meta`` records provenance (entropy, sample counts, cosines, ...).
    """
    name: str
    vec: torch.Tensor
    site: HookSite
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.vec = unit(self.vec)

    @property
    def d(self) -> int:
        return int(self.vec.numel())

    def scaled_for(self, h_ref: torch.Tensor, eps: float) -> torch.Tensor:
        """Dose vector for reference residual ``h_ref``: ``eps * RMS(h_ref) * vec / RMS(vec)``
        (the State-2143 convention, see :func:`lsne.model.make_dose_delta`)."""
        return make_dose_delta(h_ref, self.vec, eps)

    def flipped(self, suffix: str = "-") -> "Direction":
        """Sign-flipped copy (control: the same axis with opposite sign)."""
        return Direction(self.name + suffix, -self.vec, self.site, dict(self.meta, flipped=True))

    def __repr__(self) -> str:  # compact, keeps the meta out of the way
        return f"Direction(name={self.name!r}, site={self.site.key}, d={self.d})"


# --------------------------------------------------------------------------------------
# ent: entropy gradient direction (State 2143)
# --------------------------------------------------------------------------------------
def entropy_direction(mb: ModelBundle, input_ids: torch.Tensor, site: HookSite, position: int,
                      remove_axis: Optional[torch.Tensor] = None) -> Direction:
    """State-2143 direction: gradient of the next-token entropy (last position) with respect
    to the residual at ``site``, restricted to ``position``.

    If ``remove_axis`` (a vector of length d) is given, its component is projected out of
    the gradient, ``g - a (g.a)/(a.a)`` (the dossier's ARI-orthogonalisation).
    ``meta`` records ``entropy``, ``grad_norm``, ``cos_to_axis`` (cosine of the raw
    gradient with the removed axis; NaN when no axis is given) and ``removed_fraction``.
    """
    g, H = entropy_gradient(mb, input_ids, site, position)
    g = g.detach().float().cpu()
    grad_norm = float(g.norm())
    meta = dict(entropy=float(H), grad_norm=grad_norm, position=int(position),
                n_tokens=int(input_ids.shape[-1]), cos_to_axis=float("nan"),
                removed_axis=False, removed_fraction=0.0)
    if remove_axis is not None:
        a = _as_vec(remove_axis)
        if a.numel() != g.numel():
            raise ValueError(f"remove_axis has length {a.numel()}, expected {g.numel()}")
        aa = float(a @ a)
        if aa <= 0.0:
            raise ValueError("remove_axis is the zero vector")
        meta["cos_to_axis"] = cosine(g, a)
        proj = a * (float(g @ a) / aa)
        g = g - proj
        meta["removed_axis"] = True
        meta["removed_fraction"] = float(proj.norm() / max(grad_norm, _EPS))
    return Direction("ent", g, site, meta)


def load_axis_npz(path: str) -> Optional[torch.Tensor]:
    """Load an external axis (e.g. the dossier's ``axes.npz`` ARI axis) as a float32 vector.

    Tries the keys ``axis``, ``ari_axis``, ``a`` and finally the first array in the file.
    Returns ``None`` if the file does not exist. A 2-D array is reduced to its first row.
    """
    if path is None or not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        arr = None
        for k in ("axis", "ari_axis", "a"):
            if k in keys:
                arr = np.asarray(z[k])
                break
        if arr is None:
            if not keys:
                return None
            arr = np.asarray(z[keys[0]])
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return None
    if arr.ndim > 1:
        arr = arr.reshape(arr.shape[0], -1)[0]
    return torch.as_tensor(np.asarray(arr, dtype=np.float32))


# --------------------------------------------------------------------------------------
# activations from stimulus variants
# --------------------------------------------------------------------------------------
def _segments_of(annot: dict, n_tokens: int) -> list[tuple[int, int, str, object]]:
    """Split a variant into segments ``(start, end_exclusive, lang, unit)``.

    A segment is a maximal run of tokens before ``tail_start`` sharing ``(unit, lang)``;
    the tail (``tail_start`` .. end) is one English ('E') segment.
    """
    lang = list(annot.get("lang", []))
    unit_ = list(annot.get("unit", []))
    tail_start = int(annot.get("tail_start", n_tokens))
    tail_start = max(0, min(tail_start, n_tokens))
    if len(lang) < n_tokens:
        lang = lang + ["E"] * (n_tokens - len(lang))
    if len(unit_) < n_tokens:
        unit_ = unit_ + [None] * (n_tokens - len(unit_))
    segs: list[tuple[int, int, str, object]] = []
    start = 0
    while start < tail_start:
        key = (unit_[start], lang[start])
        end = start + 1
        while end < tail_start and (unit_[end], lang[end]) == key:
            end += 1
        segs.append((start, end, str(lang[start]), unit_[start]))
        start = end
    if tail_start < n_tokens:
        tail_unit = unit_[tail_start] if unit_[tail_start] is not None else "tail"
        segs.append((tail_start, n_tokens, "E", tail_unit))
    return segs


def _variant_ids(v) -> torch.Tensor:
    ids = getattr(v, "ids")
    if isinstance(ids, torch.Tensor):
        t = ids.detach().long().reshape(-1)
    else:
        t = torch.as_tensor(list(ids), dtype=torch.long)
    return t


def collect_activations(mb: ModelBundle, variants: Sequence, site: HookSite,
                        which: str = "segment_last") -> tuple[torch.Tensor, list[str], pd.DataFrame]:
    """Residual activations at ``site`` for a list of stimulus variants.

    A variant is any object with ``.ids`` (token ids), ``.target_id`` and ``.annot`` (a dict
    with per-token lists ``lang``, ``unit``, ``klass`` and the int ``tail_start``);
    ``family_id``, ``formulation``, ``K``, ``C`` are read if present.

    ``which="segment_last"``: one row per segment (maximal run of equal ``(unit, lang)``
    before ``tail_start``; the tail is one 'E' segment), taken at the segment's last token.
    ``which="all_tokens"``: one row per token (tail tokens labelled 'E').

    Sequences have different lengths, so each variant is run on its own.
    Returns ``(acts [N,d] float32 CPU, labels list[str], meta DataFrame)`` where meta has the
    columns variant, family, formulation, K, C, unit, position, lang, klass, seg_start,
    seg_len, is_tail, n_tokens.
    """
    if which not in ("segment_last", "all_tokens"):
        raise ValueError(f"which must be 'segment_last' or 'all_tokens', got {which!r}")
    rows: list[dict] = []
    acts: list[torch.Tensor] = []
    labels: list[str] = []
    for vi, v in enumerate(variants):
        ids = _variant_ids(v)
        T = int(ids.numel())
        if T == 0:
            continue
        annot = dict(getattr(v, "annot", {}) or {})
        klass = list(annot.get("klass", []))
        if len(klass) < T:
            klass = klass + [None] * (T - len(klass))
        co = run_with_cache(mb, ids[None, :], [site])
        h = co.resid[site.key][0].detach().float().cpu()          # [T,d]
        base = dict(variant=vi, family=getattr(v, "family_id", None),
                    formulation=getattr(v, "formulation", None),
                    K=getattr(v, "K", None), C=getattr(v, "C", None), n_tokens=T)
        segs = _segments_of(annot, T)
        if which == "segment_last":
            for (s, e, lang, u) in segs:
                pos = e - 1
                acts.append(h[pos])
                labels.append(lang)
                rows.append(dict(base, unit=u, position=pos, lang=lang, klass=klass[pos],
                                 seg_start=s, seg_len=e - s, is_tail=(e == T and s == int(annot.get("tail_start", T)))))
        else:
            tail_start = int(annot.get("tail_start", T))
            for (s, e, lang, u) in segs:
                for pos in range(s, e):
                    acts.append(h[pos])
                    labels.append(lang)
                    rows.append(dict(base, unit=u, position=pos, lang=lang, klass=klass[pos],
                                     seg_start=s, seg_len=e - s, is_tail=pos >= tail_start))
    cols = ["variant", "family", "formulation", "K", "C", "unit", "position", "lang", "klass",
            "seg_start", "seg_len", "is_tail", "n_tokens"]
    meta = pd.DataFrame(rows, columns=cols)
    if acts:
        A = torch.stack(acts, 0)
    else:
        A = torch.zeros(0, mb.d_model)
    return A, labels, meta


# --------------------------------------------------------------------------------------
# lang_dom: difference of means with PC denoising
# --------------------------------------------------------------------------------------
def _two_classes(acts_D: torch.Tensor, acts_E: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Both class matrices as float32 CPU ``[n, d]`` (raises if a class is empty)."""
    D = torch.as_tensor(acts_D).detach().float().cpu().reshape(-1, acts_D.shape[-1])
    E = torch.as_tensor(acts_E).detach().float().cpu().reshape(-1, acts_E.shape[-1])
    if D.shape[0] == 0 or E.shape[0] == 0:
        raise ValueError("need at least one sample per class")
    if D.shape[1] != E.shape[1]:
        raise ValueError(f"feature dimensions differ: {D.shape[1]} vs {E.shape[1]}")
    return D, E


def _within_class_centred(D: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """Stack of ``D - mean(D)`` and ``E - mean(E)`` (the within-class scatter data)."""
    return torch.cat([D - D.mean(0, keepdim=True), E - E.mean(0, keepdim=True)], 0)


def within_class_pcs(acts_D: torch.Tensor, acts_E: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-``k`` principal components of the *within-class* centred data.

    Each class is centred on its own mean before the SVD, so the between-class term
    (the mean difference itself) does not enter the covariance: the components describe
    variance that is present inside the classes ("shared" nuisance such as position, token
    class or family), never the class separation. ``k`` is clipped to the rank bound
    ``min(k, (n_D - 1) + (n_E - 1), d)``.

    Returns ``(V [d, k'] with orthonormal columns, frac [k'])`` where ``frac[i]`` is the
    fraction of the within-class variance carried by component ``i`` (``k'`` may be 0).
    """
    D, E = _two_classes(acts_D, acts_E)
    Xw = _within_class_centred(D, E)
    d = int(Xw.shape[1])
    rank_max = max(D.shape[0] - 1, 0) + max(E.shape[0] - 1, 0)
    k = int(min(int(k), rank_max, d))
    if k <= 0:
        return torch.zeros(d, 0), torch.zeros(0)
    # deterministic full SVD of the within-class centred data (N and d are small)
    _, S, Vh = torch.linalg.svd(Xw.double(), full_matrices=False)
    tot = float((S ** 2).sum())
    frac = ((S[:k] ** 2) / max(tot, _EPS)).float()
    return Vh[:k].T.float(), frac                             # [d,k], orthonormal columns


def diff_of_means(acts_D: torch.Tensor, acts_E: torch.Tensor, denoise_pcs: int = 1) -> torch.Tensor:
    """Unit difference-of-means direction ``mean(D) - mean(E)``.

    With ``denoise_pcs > 0`` the top ``denoise_pcs`` principal components of the
    **within-class centred** data (:func:`within_class_pcs`: ``D - mean(D)`` stacked on
    ``E - mean(E)``) are computed and the projection of the mean difference onto those
    components is removed before normalising (Pain-Axis denoising of the top *shared*
    component, DESIGN section 3). ``denoise_pcs=0`` gives the plain difference. The number
    of PCs is clipped to ``min(denoise_pcs, (n_D - 1) + (n_E - 1), d)``.

    The PCs are intentionally not taken from the pooled data: the pooled covariance equals
    the within-class covariance plus ``n_D n_E / N^2 * diff diff^T``, so whenever the class
    difference is large it *is* the pooled PC1 and projecting it out deletes the signal.
    Within-class PCs cannot contain the mean difference, so a class shift along a direction
    with little within-class variance survives untouched, while a direction with large
    variance in both classes (a shared nuisance axis) is removed.
    """
    D, E = _two_classes(acts_D, acts_E)
    diff = D.mean(0) - E.mean(0)
    k = int(denoise_pcs)
    if k > 0:
        V, _ = within_class_pcs(D, E, k)
        if V.shape[1] > 0:
            diff = diff - V @ (V.T @ diff)
    n = float(diff.norm())
    if not np.isfinite(n) or n <= 0.0:
        raise ValueError("diff_of_means: the mean difference is zero (or lies entirely inside "
                         "the removed within-class components); nothing to normalise")
    return unit(diff)


def denoise_report(acts_D: torch.Tensor, acts_E: torch.Tensor, denoise_pcs: int = 1,
                   gate_cos: float = DENOISE_GATE_COS, warn: bool = True) -> dict:
    """Plain and denoised difference of means with the denoising gate.

    Returns a dict with ``vec_plain``, ``vec_denoised`` (unit vectors),
    ``cos_denoised_plain``, ``separation_kept`` (``(mu @ v_den) / (mu @ v_plain)`` with
    ``mu = mean(D) - mean(E)``: the fraction of the raw D-E mean separation that survives
    along the denoised axis; by construction equal to the cosine, kept under its own name
    as the gate quantity), ``cos_pcs_plain`` (``|cos|`` of each removed within-class PC with
    the plain difference), ``within_var_removed`` (fraction of within-class variance carried
    by the removed PCs), ``dprime_plain`` / ``dprime_denoised`` (mean separation along the
    axis divided by the pooled within-class standard deviation along it; NaN when the
    within-class spread is zero), ``gate_ok`` (``cos_denoised_plain >= gate_cos``),
    ``gate_cos`` and ``denoise_pcs`` (the number of PCs actually removed).

    If ``warn`` and the gate fails, a :class:`UserWarning` is emitted: the removed shared
    component overlaps strongly with the plain class difference, so most of the measured
    mean separation is gone from the denoised axis. Whether that is a loss of signal or the
    removal of nuisance that contaminated the plain difference cannot be told from the
    cosine alone; the message therefore also states whether ``dprime`` improved (nuisance
    removed, e.g. a massive-activation dimension with few samples) or degraded (signal
    discarded). This is data and model dependent, so it is reported, not hidden.
    """
    D, E = _two_classes(acts_D, acts_E)
    mu = D.mean(0) - E.mean(0)
    v_plain = diff_of_means(D, E, denoise_pcs=0)
    V, frac = within_class_pcs(D, E, int(denoise_pcs))
    v_den = diff_of_means(D, E, denoise_pcs=int(denoise_pcs)) if V.shape[1] > 0 else v_plain.clone()
    Xw = _within_class_centred(D, E)

    def dprime(v: torch.Tensor) -> float:
        sd = float((Xw @ v).pow(2).mean().sqrt()) if Xw.shape[0] > 1 else 0.0
        return float(mu @ v) / sd if sd > 0.0 else float("nan")

    cos_dp = cosine(v_den, v_plain)
    sep_plain = float(mu @ v_plain)
    sep_kept = float(mu @ v_den) / sep_plain if sep_plain > 0.0 else float("nan")
    gate_ok = bool(cos_dp >= float(gate_cos))
    rep = dict(vec_plain=v_plain, vec_denoised=v_den, cos_denoised_plain=cos_dp,
               separation_kept=sep_kept,
               cos_pcs_plain=[abs(cosine(V[:, i], v_plain)) for i in range(int(V.shape[1]))],
               within_var_removed=float(frac.sum()) if frac.numel() else 0.0,
               dprime_plain=dprime(v_plain), dprime_denoised=dprime(v_den),
               gate_ok=gate_ok, gate_cos=float(gate_cos), denoise_pcs=int(V.shape[1]))
    if warn and not gate_ok:
        dp, dd = rep["dprime_plain"], rep["dprime_denoised"]
        verdict = ("the standardised separation d' still IMPROVED, i.e. the removed component "
                   "was mostly nuisance contaminating the plain difference"
                   if np.isfinite(dp) and np.isfinite(dd) and dd >= dp else
                   "the standardised separation d' DEGRADED, i.e. the removed shared component "
                   "coincides with the class difference and signal was discarded")
        warnings.warn(
            f"diff-of-means denoising changed the axis substantially: cos(denoised, plain) = "
            f"{cos_dp:.3f} < {float(gate_cos):.2f} (mean separation kept {sep_kept:.1%}; "
            f"|cos(PC, plain)| = {[round(c, 3) for c in rep['cos_pcs_plain']]}; "
            f"d' plain -> denoised = {dp:.2f} -> {dd:.2f}): {verdict}. Report both axes; "
            f"consider denoise_pcs=0 or more homogeneous classes.", UserWarning, stacklevel=2)
    return rep


def language_axis(mb: ModelBundle, variants: Sequence, site: HookSite,
                  pool: str = "segment_last", denoise_pcs: int = 1) -> Direction:
    """``lang_dom``: denoised difference of means D - E over the residuals of ``variants``
    at ``site`` (segment-final tokens by default).

    meta: ``n_E``, ``n_D``, ``cos_denoised_plain`` (cosine between the denoised and the plain
    difference), ``separation_kept`` (gate quantity: fraction of the raw D-E mean separation
    kept along the denoised axis), ``cos_pcs_plain``, ``within_var_removed``,
    ``dprime_plain`` / ``dprime_denoised``, ``gate_ok`` / ``gate_cos`` (see
    :func:`denoise_report`), ``denoise_pcs``, ``pool``, ``mean_norm_D/E``, ``diff_norm`` and
    the plain direction itself under ``vec_plain``. A :class:`UserWarning` is emitted when
    ``cos_denoised_plain < DENOISE_GATE_COS`` (0.7).
    """
    acts, labels, _meta = collect_activations(mb, variants, site, which=pool)
    lab = np.asarray(labels)
    D = acts[torch.as_tensor(lab == "D")]
    E = acts[torch.as_tensor(lab == "E")]
    if D.shape[0] == 0 or E.shape[0] == 0:
        raise ValueError(f"language_axis needs both languages (n_E={E.shape[0]}, n_D={D.shape[0]})")
    rep = denoise_report(D, E, denoise_pcs=int(denoise_pcs), warn=True)
    v_den = rep.pop("vec_denoised")
    meta = dict(n_E=int(E.shape[0]), n_D=int(D.shape[0]), pool=pool,
                diff_norm=float((D.mean(0) - E.mean(0)).norm()),
                mean_norm_D=float(D.mean(0).norm()), mean_norm_E=float(E.mean(0).norm()))
    meta.update(rep)
    meta["denoise_pcs_requested"] = int(denoise_pcs)
    return Direction("lang_dom", v_den, site, meta)


# --------------------------------------------------------------------------------------
# lang_probe: logistic regression without sklearn
# --------------------------------------------------------------------------------------
def probe_axis(acts: torch.Tensor, labels: Sequence[str], C: float = 1.0,
               max_iter: int = 200) -> tuple[torch.Tensor, float]:
    """L2-regularised logistic regression D (1) vs E (0) on standardised features.

    Objective (sklearn convention): ``sum_i logloss_i + ||w||^2 / (2C)``, minimised with
    torch LBFGS (float64, zero initialisation -> deterministic). Samples whose label is
    neither 'E' nor 'D' are ignored. Returns the unit weight vector mapped back to the
    original (unstandardised) feature space and the training accuracy.
    """
    X = torch.as_tensor(acts).detach().double().cpu().reshape(-1, acts.shape[-1])
    lab = np.asarray(list(labels))
    keep = (lab == "D") | (lab == "E")
    X = X[torch.as_tensor(keep)]
    y = torch.as_tensor((lab[keep] == "D").astype(np.float64))
    if X.shape[0] == 0 or float(y.sum()) == 0.0 or float(y.sum()) == X.shape[0]:
        raise ValueError("probe_axis needs samples of both classes 'E' and 'D'")
    mu = X.mean(0, keepdim=True)
    sd = X.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    Z = (X - mu) / sd
    w = torch.zeros(Z.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, tolerance_grad=1e-9,
                            tolerance_change=1e-12, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = Z @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="sum")
        loss = loss + 0.5 / float(C) * (w @ w)
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        acc = float(((Z @ w + b > 0).double() == y).double().mean())
        w_orig = (w / sd.reshape(-1)).float()   # w.z = w.(x-mu)/sd = (w/sd).x + const
    return unit(w_orig), acc


# --------------------------------------------------------------------------------------
# rand / freq
# --------------------------------------------------------------------------------------
def random_directions(d: int, n: int, seed: int = 0) -> torch.Tensor:
    """``n`` isotropic random unit vectors in R^d (float32 CPU, seeded generator)."""
    g = torch.Generator().manual_seed(int(seed))
    X = torch.randn(int(n), int(d), generator=g)
    return X / X.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def frequency_axis(mb: ModelBundle, unigram_counts: np.ndarray, top_n: int = 100,
                   site: Optional[HookSite] = None) -> Direction:
    """``freq`` axis: mean unembedding row of the ``top_n`` most frequent tokens minus the
    mean row over all tokens, expressed in residual space via ``lm_head.weight``.

    This is a first-order proxy: the final LayerNorm (its scale, bias and mean-centring)
    is ignored, i.e. the axis is read directly off the unembedding matrix. ``site`` defaults
    to the input of the last block. meta: ``top_ids``, ``top_n``, ``top_mass`` (fraction of
    counts in the top-n), ``n_vocab`` (number of ids considered).
    """
    counts = np.asarray(unigram_counts, dtype=np.float64).reshape(-1)
    W = mb.lm_head.weight.detach().float().cpu()                         # [V,d]
    n_vocab = int(min(counts.shape[0], W.shape[0]))
    if n_vocab == 0:
        raise ValueError("empty unigram counts")
    counts = counts[:n_vocab]
    top_n = int(min(top_n, n_vocab))
    order = np.argsort(-counts, kind="stable")
    top_ids = order[:top_n]
    vec = W[torch.as_tensor(top_ids, dtype=torch.long)].mean(0) - W[:n_vocab].mean(0)
    total = float(counts.sum())
    meta = dict(top_ids=[int(i) for i in top_ids], top_n=top_n, n_vocab=n_vocab,
                top_mass=float(counts[top_ids].sum() / total) if total > 0 else float("nan"),
                note="first-order proxy; final LayerNorm ignored")
    if site is None:
        site = HookSite(mb.n_layers - 1, "block_input")
    return Direction("freq", vec, site, meta)


# --------------------------------------------------------------------------------------
# geometry report
# --------------------------------------------------------------------------------------
def cosine_table(dirs: Sequence[Direction]) -> pd.DataFrame:
    """Cosine-similarity matrix of a list of directions (names x names)."""
    dirs = list(dirs)
    names: list[str] = []
    seen: dict[str, int] = {}
    for d in dirs:   # make names unique so the frame is indexable
        n = d.name
        if n in seen:
            seen[n] += 1
            n = f"{n}#{seen[d.name]}"
        else:
            seen[n] = 0
        names.append(n)
    if not dirs:
        return pd.DataFrame(index=names, columns=names, dtype=float)
    M = torch.stack([d.vec for d in dirs], 0)
    M = M / M.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    cos = (M @ M.T).numpy().astype(np.float64)
    # exact symmetry and unit diagonal (float32 round-off otherwise)
    cos = 0.5 * (cos + cos.T)
    np.fill_diagonal(cos, 1.0)
    return pd.DataFrame(cos, index=names, columns=names)


def unembed_projection(mb: ModelBundle, direction: Union[Direction, torch.Tensor], top_k: int = 25,
                       h_rms: float = 1.0) -> dict:
    """Tokens promoted / suppressed by a direction through the final LN scale and the
    unembedding matrix.

    ``scores = lm_head.weight @ (final_ln.weight * vec / RMS(vec) * h_rms)``: the direction
    is rescaled to RMS ``h_rms`` (a residual-like scale), multiplied by the LayerNorm gain,
    and projected onto every unembedding row. LayerNorm mean-centring and bias are ignored
    (they add direction-independent terms). Only ids the tokenizer can decode are ranked.
    Returns ``dict(promoted=[(token_str, score)], suppressed=[...], scores=Tensor[V])``
    with ``promoted`` in descending and ``suppressed`` in ascending score order.
    """
    vec = _as_vec(direction)
    W = mb.lm_head.weight.detach().float().cpu()                         # [V,d]
    gain = mb.final_ln.weight.detach().float().cpu()
    z = gain * (vec / rms(vec).clamp_min(_EPS)) * float(h_rms)
    scores = W @ z                                                       # [V]
    n_tok = int(min(len(mb.tokenizer), scores.numel()))
    top_k = int(min(top_k, n_tok))
    s = scores[:n_tok]
    top_pos = torch.topk(s, top_k).indices
    top_neg = torch.topk(-s, top_k).indices

    def _dec(i: int) -> str:
        try:
            return mb.tokenizer.decode([int(i)])
        except Exception:   # pragma: no cover - defensive
            return f"<{int(i)}>"

    promoted = [(_dec(i), float(s[i])) for i in top_pos.tolist()]
    suppressed = [(_dec(i), float(s[i])) for i in top_neg.tolist()]
    return dict(promoted=promoted, suppressed=suppressed, scores=scores,
                promoted_ids=[int(i) for i in top_pos.tolist()],
                suppressed_ids=[int(i) for i in top_neg.tolist()])


def random_direction_set(d: int, n: int, seed: int, site: HookSite, name: str = "rand") -> list[Direction]:
    """Convenience: ``random_directions`` wrapped as Direction objects ``rand00, rand01, ...``."""
    R = random_directions(d, n, seed)
    return [Direction(f"{name}{i:02d}", R[i], site, dict(seed=int(seed), index=i)) for i in range(int(n))]


__all__ = [
    "Direction", "unit", "cosine", "entropy_direction", "load_axis_npz", "collect_activations",
    "within_class_pcs", "diff_of_means", "denoise_report", "DENOISE_GATE_COS", "language_axis",
    "probe_axis", "random_directions", "random_direction_set", "frequency_axis", "cosine_table",
    "unembed_projection",
]
