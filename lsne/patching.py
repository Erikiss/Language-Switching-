"""Counterfactual activation patching between a monolingual and a mixed-language variant
(DESIGN.md section 4.1, Im et al. 2026 style).

The question answered here: *which positions carry the effect of a language switch on the
target prediction?*  Residual activations of the monolingual variant ``x_mono`` are written
into the forward pass of the mixed variant ``x_mixed`` at a hook site, restricted to a
class of token positions (whitespace, punctuation, switch boundaries, word starts, ...),
and the recovered fraction of the text effect is reported together with a random-position
null model of the same size *drawn from the same candidate pool* and a gradient-times-
activation attribution.

Shared region.  Patching is only well defined where both variants carry the same token at
the same distance from the end.  Two regions are supported (``region`` argument of
:func:`counterfactual_attribution`):

* ``"suffix"`` (default): the maximal end-aligned run of identical token ids
  (:func:`align_shared_suffix`).  By construction of the stimulus bank this is the fixed
  English tail *plus* unit 10 (English in every variant) and, for C3 variants, the English
  predicate/adjunct of unit 9 - so the last switch boundary token and its context lie on
  matched positions and are measurable in every formulation.
* ``"tail"``: the fixed English tail only (:func:`align_tail`, the literal DESIGN 2.2
  region).  ``boundary`` / ``switch_context`` positions in the prefix are then added where
  the end-aligned token ids happen to agree (:func:`align_from_end`), as a separate,
  explicitly counted stratum (``n_nontail``).

Null model.  For a class with ``n_tail`` patched tail positions and ``n_nontail`` patched
prefix positions, every random draw takes ``n_tail`` positions from the tail pool and
``n_nontail`` positions from the prefix pool (stratified, without replacement).  The control
therefore never compares prefix patches against tail positions that sit causally closer to
the target.  When the class exhausts its pools (``tail``, ``all``) there is exactly one
possible draw - the measurement itself - and the control is reported as NaN with
``control_defined=False``: *excess over random is undefined for such classes*.

Token classes.  Under the GPT-NeoX BPE the space is attached to the following word, so the
``ws`` class (whitespace-only tokens) is empty for ordinary prose.  Two tokenizer-adequate
delimiter classes are therefore provided in addition: ``post_punct`` (the space-carrying
first token after a punctuation token) and ``delim`` (``ws`` + ``punct`` + ``boundary``,
the pre-registered union class R_ws+punct+boundary measured as one patch set).  Classes with
no patchable position are reported with ``measurable=False`` and a ``UserWarning``.

Variants are duck-typed: any object with ``.ids`` (list of token ids), ``.target_id`` and
``.annot`` (dict with per-token lists ``lang``, ``unit``, ``klass``,
``dist_to_last_switch`` and ints ``tail_start``, ``last_switch_pos``). ``lsne.stimuli`` is
deliberately *not* imported.
"""
from __future__ import annotations

import math
import warnings
import zlib
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .config import HookSite
from .model import Injector, ModelBundle, forward_logits, run_with_cache

#: Token classes read directly from ``annot['klass']``.
KLASS_NAMES = ("ws", "punct", "boundary", "word_start", "word_cont")
#: Classes derived in :func:`position_classes` on top of the ``klass`` labels.
DERIVED_CLASSES = ("post_punct", "delim", "tail", "all", "switch_context")
#: ``switch_context`` = the boundary token plus the next SWITCH_CONTEXT_WINDOW tokens.
SWITCH_CONTEXT_WINDOW = 3
#: |logp_mono - logp_mixed| below this value makes the recovery fraction undefined (NaN).
RECOVERY_DENOM_MIN = 1e-6
#: Shared regions accepted by :func:`counterfactual_attribution`.
REGIONS = ("suffix", "tail")
#: Classes that are extended to end-aligned prefix positions when ``region="tail"``.
PREFIX_EXTENDED_CLASSES = ("boundary", "switch_context")
#: Columns of the DataFrame returned by :func:`counterfactual_attribution`.
ATTRIBUTION_COLUMNS = (
    "site", "layer", "class", "region", "n_positions", "n_tail", "n_nontail", "n_pool_tail",
    "n_pool_nontail", "n_null_subsets", "measurable", "control_defined", "logp_mono", "logp_mixed",
    "logp_patched", "recovery", "recovery_random_mean", "recovery_random_sd", "dst_positions",
)


# --------------------------------------------------------------------------------------
# duck-typed variant access
# --------------------------------------------------------------------------------------
def _get(variant, name, default=None):
    """Attribute access with a dict fallback (variants may be dataclasses, namespaces or dicts)."""
    if isinstance(variant, dict):
        return variant.get(name, default)
    return getattr(variant, name, default)


def variant_ids(variant) -> list[int]:
    """Token ids of a variant as a plain list of ints."""
    ids = _get(variant, "ids")
    if ids is None:
        raise ValueError("variant has no 'ids'")
    if isinstance(ids, torch.Tensor):
        ids = ids.detach().reshape(-1).tolist()
    elif isinstance(ids, np.ndarray):
        ids = ids.reshape(-1).tolist()
    return [int(i) for i in ids]


def variant_annot(variant) -> dict:
    """The annotation dict of a variant (empty dict if absent)."""
    annot = _get(variant, "annot", None)
    return dict(annot) if annot is not None else {}


def tail_start_of(variant) -> int:
    """First tail position of a variant (``len(ids)`` if no tail is annotated), clipped to [0, T]."""
    n = len(variant_ids(variant))
    annot = variant_annot(variant)
    ts = annot.get("tail_start", n)
    ts = n if ts is None else int(ts)
    return max(0, min(ts, n))


def _to_ids(mb: ModelBundle, ids) -> torch.Tensor:
    """Anything id-like -> LongTensor[1, T] on the model device."""
    if isinstance(ids, torch.Tensor):
        t = ids.detach().long()
    else:
        t = torch.as_tensor([int(i) for i in np.asarray(ids).reshape(-1)], dtype=torch.long)
    if t.dim() == 1:
        t = t[None, :]
    if t.dim() != 2 or t.shape[0] != 1:
        raise ValueError(f"expected a single sequence, got shape {tuple(t.shape)}")
    return t.to(mb.device)


def _logp_last(logits: torch.Tensor, target_id: int) -> float:
    """log p(target) at the last position of logits [1, T, V]."""
    return float(torch.log_softmax(logits[0, -1].float(), -1)[int(target_id)])


# --------------------------------------------------------------------------------------
# position classes and alignment
# --------------------------------------------------------------------------------------
def position_classes(variant) -> dict[str, list[int]]:
    """Position sets of a variant, keyed by token class.

    Keys: ``ws``, ``punct``, ``boundary``, ``word_start``, ``word_cont`` (from
    ``annot['klass']``; these five partition the positions), ``tail`` (positions >=
    ``tail_start``), ``all`` (every position), ``switch_context`` (each boundary token plus
    the ``SWITCH_CONTEXT_WINDOW`` tokens after it), ``post_punct`` (every non-punctuation,
    non-whitespace token that directly follows a ``punct`` token - under the GPT-NeoX BPE
    this is the space-carrying sentence/clause starter, the tokenizer's stand-in for a
    whitespace token) and ``delim`` (``ws`` + ``punct`` + ``boundary``, the pre-registered
    union class). If ``klass`` carries no ``boundary`` label at all, boundaries are derived
    from changes of ``annot['lang']`` between consecutive tokens (empty for K=0 variants).
    Every list is sorted and duplicate free; the derived classes overlap by design.
    """
    ids = variant_ids(variant)
    n = len(ids)
    annot = variant_annot(variant)
    klass = list(annot.get("klass", []) or [])[:n]
    out: dict[str, list[int]] = {k: [] for k in KLASS_NAMES}
    for t, k in enumerate(klass):
        if k in out:
            out[k].append(t)
    if not out["boundary"]:
        lang = list(annot.get("lang", []) or [])[:n]
        out["boundary"] = [t for t in range(1, len(lang)) if lang[t] != lang[t - 1]]
    ts = tail_start_of(variant)
    out["tail"] = list(range(ts, n))
    out["all"] = list(range(n))
    ctx: set[int] = set()
    for b in out["boundary"]:
        ctx.update(range(b, min(n, b + SWITCH_CONTEXT_WINDOW + 1)))
    out["switch_context"] = sorted(ctx)
    out["post_punct"] = [t for t in range(1, len(klass))
                         if klass[t - 1] == "punct" and klass[t] not in ("punct", "ws")]
    out["delim"] = sorted(set(out["ws"]) | set(out["punct"]) | set(out["boundary"]))
    return out


def align_tail(v_src, v_dst) -> list[tuple[int, int]]:
    """``(src_pos, dst_pos)`` pairs for the shared English tail, aligned from the END.

    ``n = min(tail_len_src, tail_len_dst)`` positions are paired; the token ids of every
    pair must agree (``ValueError`` otherwise - the tails of a family must tokenize
    identically, see DESIGN section 2.2).
    """
    ids_s, ids_d = variant_ids(v_src), variant_ids(v_dst)
    n = min(len(ids_s) - tail_start_of(v_src), len(ids_d) - tail_start_of(v_dst))
    n = max(0, n)
    pairs = [(len(ids_s) - n + i, len(ids_d) - n + i) for i in range(n)]
    bad = [(s, d) for s, d in pairs if ids_s[s] != ids_d[d]]
    if bad:
        s, d = bad[0]
        raise ValueError(f"tail token ids differ at src {s} / dst {d}: {ids_s[s]} != {ids_d[d]} "
                         f"({len(bad)} mismatching pairs)")
    return pairs


def align_shared_suffix(v_src, v_dst) -> list[tuple[int, int]]:
    """``(src_pos, dst_pos)`` pairs of the maximal END-aligned run of identical token ids.

    The run always contains the shared tail (:func:`align_tail`, which validates it) and
    extends backwards through every earlier position whose token ids agree at the same
    distance from the end; it stops at the first mismatch. For the stimulus families this
    is unit 10 (English in every variant) plus the tail - for C3 variants also the English
    predicate/adjunct of unit 9 - so that the last switch boundary lies on matched
    positions. Pairs are sorted by destination position.
    """
    tail_pairs = align_tail(v_src, v_dst)
    ids_s, ids_d = variant_ids(v_src), variant_ids(v_dst)
    if tail_pairs:
        s, d = tail_pairs[0]
    else:
        s, d = len(ids_s), len(ids_d)
    extra: list[tuple[int, int]] = []
    s, d = s - 1, d - 1
    while s >= 0 and d >= 0 and ids_s[s] == ids_d[d]:
        extra.append((s, d))
        s, d = s - 1, d - 1
    return extra[::-1] + tail_pairs


def align_from_end(v_src, v_dst, dst_positions: Sequence[int]) -> list[tuple[int, int]]:
    """Pair arbitrary ``dst_positions`` with the src position at the same distance from the
    END, keeping only pairs whose token ids match (used for non-tail classes)."""
    ids_s, ids_d = variant_ids(v_src), variant_ids(v_dst)
    shift = len(ids_s) - len(ids_d)
    pairs = []
    for d in dst_positions:
        s = int(d) + shift
        if 0 <= s < len(ids_s) and ids_s[s] == ids_d[int(d)]:
            pairs.append((s, int(d)))
    return pairs


# --------------------------------------------------------------------------------------
# patched forward passes
# --------------------------------------------------------------------------------------
def _forward_patched(mb: ModelBundle, dst_ids: torch.Tensor, site: HookSite,
                     src_resid: torch.Tensor, pairs: Sequence[tuple[int, int]]) -> torch.Tensor:
    """Forward ``dst_ids`` with ``src_resid[0, src_pos]`` written at ``dst_pos`` for every
    pair (replace mode). Without pairs this is the plain forward."""
    pairs = list(pairs)
    if not pairs:
        return forward_logits(mb, dst_ids)
    src_pos = [int(p[0]) for p in pairs]
    dst_pos = [int(p[1]) for p in pairs]
    if len(set(dst_pos)) != len(dst_pos):
        raise ValueError("duplicate destination positions in pairs")
    delta = src_resid[0, src_pos, :].detach()                      # [P, d]
    with Injector(mb, site, delta, positions=dst_pos, mode="replace"):
        return forward_logits(mb, dst_ids)


def patch_run(mb: ModelBundle, src_ids, dst_ids, site: HookSite, pairs: Sequence[tuple[int, int]],
              target_id: int) -> dict:
    """Write the residuals of ``src`` at ``site`` into the run of ``dst`` at the given
    ``(src_pos, dst_pos)`` pairs and return ``log p(target)`` at the last position for the
    patched, the source and the destination run."""
    src_t, dst_t = _to_ids(mb, src_ids), _to_ids(mb, dst_ids)
    src_cache = run_with_cache(mb, src_t, [site])
    dst_logits = forward_logits(mb, dst_t)
    patched = _forward_patched(mb, dst_t, site, src_cache.resid[site.key], pairs)
    return dict(logp_patched=_logp_last(patched, target_id),
                logp_src=_logp_last(src_cache.logits, target_id),
                logp_dst=_logp_last(dst_logits, target_id),
                n_positions=len(list(pairs)))


def recovery_fraction(logp_patched: float, logp_mixed: float, logp_mono: float) -> float:
    """``(logp_patched - logp_mixed) / (logp_mono - logp_mixed)``; NaN when the text effect
    in the denominator is smaller than ``RECOVERY_DENOM_MIN`` in magnitude."""
    denom = logp_mono - logp_mixed
    if not np.isfinite(denom) or abs(denom) < RECOVERY_DENOM_MIN:
        return float("nan")
    return float((logp_patched - logp_mixed) / denom)


# --------------------------------------------------------------------------------------
# stratified random-position null
# --------------------------------------------------------------------------------------
def null_subset_count(n_pool_tail: int, n_tail: int, n_pool_nontail: int, n_nontail: int) -> int:
    """Number of distinct position sets the stratified null can draw: ``C(n_pool_tail,
    n_tail) * C(n_pool_nontail, n_nontail)`` (0 if a stratum asks for more than its pool
    holds, 1 if the only possible draw is the patched set itself)."""
    if n_tail > n_pool_tail or n_nontail > n_pool_nontail or min(n_tail, n_nontail) < 0:
        return 0
    return math.comb(n_pool_tail, n_tail) * math.comb(n_pool_nontail, n_nontail)


def draw_null_pairs(rng: np.random.Generator, tail_pool: Sequence[tuple[int, int]],
                    prefix_pool: Sequence[tuple[int, int]], n_tail: int, n_nontail: int) -> list[tuple[int, int]]:
    """One random position set of the null model: ``n_tail`` pairs drawn without
    replacement from ``tail_pool`` and ``n_nontail`` pairs from ``prefix_pool`` (the
    strata of the patched set), sorted by destination position."""
    out: list[tuple[int, int]] = []
    if n_tail > 0:
        idx = rng.choice(len(tail_pool), size=int(n_tail), replace=False)
        out += [tail_pool[i] for i in idx.tolist()]
    if n_nontail > 0:
        idx = rng.choice(len(prefix_pool), size=int(n_nontail), replace=False)
        out += [prefix_pool[i] for i in idx.tolist()]
    return sorted(out, key=lambda p: p[1])


def _null_rng(seed: int, site: HookSite, cls: str) -> np.random.Generator:
    """Generator seeded per (seed, site, class) so that a row does not depend on which
    other sites/classes were requested in the same call."""
    return np.random.default_rng([int(seed), zlib.crc32(f"{site.key}:{cls}".encode("utf-8"))])


def counterfactual_attribution(mb: ModelBundle, v_mono, v_mixed, sites: Sequence[HookSite],
                               classes: Sequence[str], n_random: int = 10, seed: int = 0,
                               region: str = "suffix") -> pd.DataFrame:
    """Position-class patching of ``v_mono`` activations into the ``v_mixed`` run.

    For every ``site`` and ``class`` the class positions of ``v_mixed`` that lie in the
    shared region are patched from the end-aligned positions of ``v_mono``. ``region``:

    * ``"suffix"`` (default): the maximal end-aligned run of identical token ids
      (:func:`align_shared_suffix`; tail + unit 10 for the stimulus families). All classes
      are restricted to it; positions before ``tail_start`` are counted in ``n_nontail``.
    * ``"tail"``: the fixed English tail (:func:`align_tail`). ``boundary`` and
      ``switch_context`` are additionally extended to prefix positions whose end-aligned
      token ids agree (:func:`align_from_end`), counted in ``n_nontail``.

    Recovery ``R = (logp_patched - logp_mixed) / (logp_mono - logp_mixed)``.

    Null model (stratified): ``n_random`` random position sets with the same number of
    tail positions drawn from the tail pool and the same number of prefix positions drawn
    from the prefix pool (the same candidate pools the patched set came from), reported as
    mean and sd of R. When only one such set exists - the class exhausts its pools, e.g.
    ``tail`` and ``all`` - the control is the measurement itself and is reported as NaN
    with ``control_defined=False`` (excess over random is undefined for these classes).
    ``n_null_subsets`` gives the number of distinct null sets, so saturated classes (a
    class covering most of its pool) can be recognised.

    Classes without any patchable position get ``n_positions=0``, NaN recovery,
    ``measurable=False`` and a ``UserWarning`` naming them (under the GPT-NeoX BPE this is
    the case for ``ws`` on ordinary prose; use ``post_punct`` / ``delim``).

    Returns a tidy DataFrame with the columns in :data:`ATTRIBUTION_COLUMNS`.
    """
    target_id = int(_get(v_mixed, "target_id"))
    if _get(v_mono, "target_id") is not None and int(_get(v_mono, "target_id")) != target_id:
        raise ValueError("v_mono and v_mixed have different target ids")
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of {REGIONS}")
    sites = list(sites)
    classes = list(classes)
    cls_mixed = position_classes(v_mixed)
    unknown = [c for c in classes if c not in cls_mixed]
    if unknown:
        raise ValueError(f"unknown position class {unknown[0]!r}; known: {sorted(cls_mixed)}")

    mono_t, mixed_t = _to_ids(mb, variant_ids(v_mono)), _to_ids(mb, variant_ids(v_mixed))
    mono_cache = run_with_cache(mb, mono_t, sites)
    logp_mono = _logp_last(mono_cache.logits, target_id)
    logp_mixed = _logp_last(forward_logits(mb, mixed_t), target_id)

    # shared region and the two candidate pools (strata) of the null model
    ts_mixed = tail_start_of(v_mixed)
    if region == "suffix":
        shared = align_shared_suffix(v_mono, v_mixed)
        prefix_pool = [p for p in shared if p[1] < ts_mixed]
    else:
        shared = align_tail(v_mono, v_mixed)
        prefix_pool = align_from_end(v_mono, v_mixed, range(ts_mixed))
    tail_pool = [p for p in shared if p[1] >= ts_mixed]
    shared_map = {d: s for s, d in shared}
    prefix_map = {d: s for s, d in prefix_pool}

    rows = []
    unmeasurable: list[str] = []
    for site in sites:
        src_resid = mono_cache.resid[site.key]
        for cls in classes:
            pos = cls_mixed[cls]
            pairs = {(shared_map[d], d) for d in pos if d in shared_map}
            if region == "tail" and cls in PREFIX_EXTENDED_CLASSES:
                pairs |= {(prefix_map[d], d) for d in pos if d < ts_mixed and d in prefix_map}
            pairs = sorted(pairs, key=lambda p: p[1])
            n_pos = len(pairs)
            n_tail = sum(1 for p in pairs if p[1] >= ts_mixed)
            n_nontail = n_pos - n_tail
            if n_pos > 0:
                logp_patched = _logp_last(_forward_patched(mb, mixed_t, site, src_resid, pairs), target_id)
                rec = recovery_fraction(logp_patched, logp_mixed, logp_mono)
            else:
                logp_patched, rec = logp_mixed, float("nan")
                if cls not in unmeasurable:
                    unmeasurable.append(cls)
            # stratified random control: same counts from the same pools
            n_subsets = null_subset_count(len(tail_pool), n_tail, len(prefix_pool), n_nontail)
            control_defined = n_pos > 0 and n_subsets > 1
            rand: list[float] = []
            if control_defined and int(n_random) > 0:
                rng = _null_rng(seed, site, cls)
                for _ in range(int(n_random)):
                    rp = draw_null_pairs(rng, tail_pool, prefix_pool, n_tail, n_nontail)
                    lp = _logp_last(_forward_patched(mb, mixed_t, site, src_resid, rp), target_id)
                    rand.append(recovery_fraction(lp, logp_mixed, logp_mono))
            rand_arr = np.asarray(rand, dtype=float)
            has_rand = bool(rand_arr.size) and bool(np.isfinite(rand_arr).any())
            rows.append(dict(
                site=site.key, layer=int(site.layer), **{"class": cls}, region=region,
                n_positions=n_pos, n_tail=n_tail, n_nontail=n_nontail,
                n_pool_tail=len(tail_pool), n_pool_nontail=len(prefix_pool),
                n_null_subsets=float(n_subsets) if n_subsets < 10 ** 300 else float("inf"),
                measurable=bool(n_pos > 0), control_defined=bool(control_defined),
                logp_mono=logp_mono, logp_mixed=logp_mixed, logp_patched=logp_patched,
                recovery=rec,
                recovery_random_mean=float(np.nanmean(rand_arr)) if has_rand else float("nan"),
                recovery_random_sd=float(np.nanstd(rand_arr)) if has_rand else float("nan"),
                dst_positions=";".join(str(p[1]) for p in pairs),
            ))
    if unmeasurable:
        warnings.warn(f"counterfactual_attribution: no patchable position in the shared {region} for "
                      f"class(es) {unmeasurable} (n_positions=0, recovery=NaN, measurable=False)",
                      UserWarning, stacklevel=2)
    return pd.DataFrame(rows, columns=list(ATTRIBUTION_COLUMNS))


# --------------------------------------------------------------------------------------
# gradient x activation
# --------------------------------------------------------------------------------------
def grad_x_act(mb: ModelBundle, ids, site: HookSite, target_id: int) -> torch.Tensor:
    """Gradient-times-activation attribution per position.

    ``log p(target)`` at the last position is differentiated with respect to the residual
    at ``site`` (the leaf-replacement trick of :func:`lsne.model.entropy_gradient`), the
    gradient is multiplied elementwise with the activation and summed over the model
    dimension. Returns a float32 CPU tensor of length T (first-order estimate of the
    effect of removing each position's residual at that site).
    """
    input_ids = _to_ids(mb, ids)
    leaf: dict[str, torch.Tensor] = {}

    def edit(h):
        t = h.detach().clone().requires_grad_(True)
        leaf["h"] = t
        return t

    with torch.enable_grad():
        with Injector(mb, site, edit):
            out = mb.model(input_ids=input_ids, use_cache=False)
        logp = torch.log_softmax(out.logits[0, -1].float(), -1)[int(target_id)]
        logp.backward()
    h = leaf["h"]
    if h.grad is None:
        raise RuntimeError("no gradient reached the hook site")
    gxa = (h.grad[0].float() * h.detach()[0].float()).sum(-1)
    return gxa.detach().cpu()
