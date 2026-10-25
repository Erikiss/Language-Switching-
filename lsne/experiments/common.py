"""Shared helpers for the experiment stages.

* :class:`Limits` - run-size knobs that are not part of :class:`lsne.config.RunConfig`
  (generation ramp length, number of random patch draws, TNCO steps, ...), with a
  ``smoke`` preset for a < 3 min CPU run on pythia-14m and a ``limited`` flag used by the
  scale sweep (stage F) for stages C and D.
* :func:`load_variants` - families from ``cfg.families_dir`` filtered by ``cfg.families`` /
  ``cfg.limit_families``, all K x C variants per family, consistency checks against the
  K=0 variant (exclusions are recorded, not raised), ``cfg.limit_variants``.
* :func:`load_original` - the State-2143 anchor (``lsne.anchor.load_anchor``) and the
  original-context variants; ``None`` with a warning when the sequence is not reachable.
* :func:`direction_bank` - the dictionary of :class:`~lsne.directions.Direction` objects
  used by stages B-E (``ent``, ``lang_dom``, ``lang_probe``, ``freq``, ``rand_00..``,
  optionally ``sae_D0..`` / ``sae_E0..``) plus the frequent-token ids for ``p_freq``.
* small accessors shared by the stages (dossier site, injection positions, tidy row
  prefixes, marker lexicon for the CA view, tqdm wrapper, stage bookkeeping).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .. import stimuli
from ..config import STATE2143, HookSite, RunConfig
from ..directions import (Direction, collect_activations, entropy_direction, frequency_axis,
                          language_axis, load_axis_npz, probe_axis, random_direction_set)
from ..io import RunDir, read_manifest, save_df, to_jsonable, write_manifest
from ..model import ModelBundle

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
#: Priority of (K, C) conditions when ``cfg.limit_variants`` truncates a family: the
#: core triple (monolingual, K=8/C1, K=28/seg) comes first so that a limited run still
#: contains the contrasts used by stages C-E.
PRIORITY: tuple[tuple[int, str], ...] = (
    (0, "C1"), (8, "C1"), (28, "seg"), (2, "C1"), (4, "C1"), (16, "seg"),
    (2, "C2"), (4, "C2"), (8, "C2"), (2, "C3"), (4, "C3"), (8, "C3"),
)
#: (K, C) conditions of the "core" subset used by the generation diagnostics (D3-D7),
#: the counterfactual pairs (C) and the tensor-network stage (E).
CORE_CONDITIONS: tuple[tuple[int, str], ...] = ((0, "C1"), (8, "C1"), (28, "seg"))
#: Frequent English / German function words used as marker tokens of the CA view (D9).
MARKER_WORDS = {
    "E": (" the", " and", " of", " is", " to", " in"),
    "D": (" der", " die", " und", " ist", " das", " nicht"),
}
#: Position classes evaluated by the counterfactual attribution (stage C).
PATCH_CLASSES = ("ws", "punct", "boundary", "word_start", "word_cont", "switch_context", "tail")
#: Standard id columns of every tidy table.
ID_COLS = ("tag", "family", "formulation", "K", "C", "kind", "n_tokens", "tail_start")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# --------------------------------------------------------------------------------------
# limits
# --------------------------------------------------------------------------------------
@dataclass
class Limits:
    """Run-size parameters outside :class:`~lsne.config.RunConfig`.

    ``smoke`` marks the < 3 min CPU preset (see :meth:`smoke`), ``limited`` the reduced
    stages C/D of the scale sweep (stage F): fewer random directions in D1, no random
    directions in the grid, generation diagnostics for ``lang_dom`` only.
    """
    smoke: bool = False
    limited: bool = False
    progress: bool = True
    # stage B
    n_unigram: int = 200            # Pile sequences for the unigram counts (cache/unigram)
    use_sae: bool = False
    n_sae_features: int = 5         # sae_D0..n-1 and sae_E0..n-1 directions
    # stage C
    n_random_patch: int = 10        # random position draws per (site, class)
    n_boot: int = 2000              # cluster-bootstrap resamples
    # stage D
    damage_doses: tuple = (0.01, 0.05, 0.1)
    gen_n_steps: int = 8            # D3 ramp steps
    gen_tokens_per_step: int = 8    # D3 tokens per plateau
    gen_t_on: int = 16              # D4 quench time
    n_boot_tau: int = 20            # D4 bootstrap resamples for tau
    ca_threshold: float = 0.5
    # stage E
    n_families_tn: int = 2          # family ids used in stage E
    tn_doses: tuple = (0.01, 0.1)
    tn_n_steps: int = 500           # TNCO annealing steps
    tn_n_runs: int = 4
    # stage F
    f_stages: tuple = ("B", "C", "D")

    @classmethod
    def smoke(cls, **kw) -> "Limits":
        """Preset for the CPU smoke run (pythia-14m, one family, three variants)."""
        base = dict(smoke=True, n_unigram=5, n_random_patch=2, n_boot=200, damage_doses=(0.01, 0.1),
                    gen_n_steps=2, gen_tokens_per_step=2, gen_t_on=2, n_boot_tau=5,
                    n_families_tn=1, tn_doses=(0.01, 0.1), tn_n_steps=50, tn_n_runs=1,
                    use_sae=False)
        base.update(kw)
        return cls(**base)

    def as_limited(self) -> "Limits":
        """Copy with the reduced-stage flag of the scale sweep set."""
        return replace(self, limited=True, n_random_patch=min(self.n_random_patch, 5))


def smoke_config(cfg: RunConfig) -> RunConfig:
    """Copy of ``cfg`` with every size limit set for the smoke run."""
    return replace(cfg, limit_families=cfg.limit_families or 1, limit_variants=cfg.limit_variants or 3,
                   n_random_dirs=2, doses=(0.0, 0.01, 0.1), doses_steer=(0.5,),
                   gen_n_new=8, gen_n_samples=2, tn_rank=16, tn_window=16)


def get_limits(cfg: RunConfig, kw: dict) -> Limits:
    """``kw['limits']`` or a fresh :class:`Limits` (smoke preset if ``kw['smoke']``)."""
    lim = kw.get("limits")
    if lim is None:
        lim = Limits.smoke() if kw.get("smoke") else Limits()
    return lim


# --------------------------------------------------------------------------------------
# paths / caches
# --------------------------------------------------------------------------------------
def cache_root() -> Path:
    """``$LSNE_CACHE`` or ``./cache`` (the anchor is cached under ``cache/state2143``)."""
    return Path(os.environ.get("LSNE_CACHE", "cache"))


def cache_dir(name: str) -> str:
    return str(cache_root() / name)


def resolve_families_dir(path: str | os.PathLike) -> Path:
    """``cfg.families_dir`` as given, or relative to the repository root if it does not
    exist relative to the working directory."""
    p = Path(path)
    if p.is_dir():
        return p
    alt = _REPO_ROOT / p
    return alt if alt.is_dir() else p


def offline() -> bool:
    return os.environ.get("LSNE_OFFLINE", "") not in ("", "0", "false", "False")


# --------------------------------------------------------------------------------------
# model sites / positions
# --------------------------------------------------------------------------------------
def dossier_site(mb: ModelBundle) -> HookSite:
    """The State-2143 intervention site generalised to any depth: block input of the last
    block (``layer 11`` for pythia-160m)."""
    return HookSite(mb.n_layers - 1, "block_input")


def depth_layer(mb: ModelBundle) -> int:
    """Block index of the early injection used for the depth growth exponent (D1b) and
    the tensor-network stage: ``n_layers // 3``."""
    return max(0, mb.n_layers // 3)


def depth_site(mb: ModelBundle) -> HookSite:
    return HookSite(depth_layer(mb), "block_input")


def variant_kind(v) -> str:
    return "original" if str(getattr(v, "C", "")) == "orig" else "standard"


def is_original(v) -> bool:
    return variant_kind(v) == "original"


def ids_tensor(mb: ModelBundle, v) -> torch.Tensor:
    """Context token ids of a variant as ``LongTensor[1, T]`` on the model device."""
    return torch.as_tensor([int(i) for i in v.ids], dtype=torch.long, device=mb.device)[None, :]


def tail_positions(v) -> list[int]:
    return list(range(int(v.tail_start), int(v.n_tokens)))


def unit_last_position(v, unit: Optional[int] = None) -> int:
    """Last token position of a unit (default: unit 9 of a standard family, the last
    switchable unit of the original family); ``tail_start - 1`` if the unit is absent."""
    units = list(v.annot.get("unit", []))
    if unit is None:
        if is_original(v):
            unit = max((u for u in units if u < max(units)), default=None) if units else None
        else:
            unit = 9
    pos = [t for t, u in enumerate(units) if unit is not None and int(u) == int(unit)]
    if not pos:
        return max(0, int(v.tail_start) - 1)
    return int(pos[-1])


def variant_row(v) -> dict:
    """Standard id columns of a variant for tidy tables."""
    return dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=int(v.K), C=str(v.C),
                kind=variant_kind(v), n_tokens=int(v.n_tokens), tail_start=int(v.tail_start))


def dir_family(name: str) -> str:
    """Direction family: ``rand`` for ``rand_XX``, ``sae`` for ``sae_D0``/``sae_E0``, else the name."""
    if name.startswith("rand_"):
        return "rand"
    if name.startswith("sae_"):
        return "sae"
    return name


def progress(iterable: Iterable, desc: str = "", total: Optional[int] = None, enabled: bool = True):
    """tqdm wrapper (``enabled=False`` or a non-TTY stderr disables the bar)."""
    try:
        from tqdm.auto import tqdm
    except ImportError:          # pragma: no cover - tqdm is a dependency
        return iterable
    return tqdm(iterable, desc=desc, total=total, disable=(None if enabled else True), leave=False)


# --------------------------------------------------------------------------------------
# families and variants
# --------------------------------------------------------------------------------------
def select_families(cfg: RunConfig, errors: Optional[dict] = None) -> list[stimuli.Family]:
    """Families of ``cfg.families_dir`` filtered by ``cfg.families`` (family ids such as
    ``F01`` or tags such as ``F01A``) and truncated to ``cfg.limit_families`` family ids."""
    fams = stimuli.load_families(resolve_families_dir(cfg.families_dir), strict=False, errors=errors)
    if cfg.families:
        wanted = {str(f).strip() for f in cfg.families if str(f).strip()}
        fams = [f for f in fams if f.family_id in wanted or f.tag in wanted]
    if cfg.limit_families:
        keep: list[str] = []
        for f in fams:
            if f.family_id not in keep:
                keep.append(f.family_id)
        keep = keep[: int(cfg.limit_families)]
        fams = [f for f in fams if f.family_id in keep]
    return fams


def priority_rank(v) -> int:
    try:
        return PRIORITY.index((int(v.K), str(v.C)))
    except ValueError:
        return len(PRIORITY) + int(v.K)


def group_key(v) -> tuple[str, str]:
    return (str(v.family_id), str(v.formulation))


def apply_variant_limit(variants: Sequence, limit: Optional[int]) -> list:
    """Keep at most ``limit`` variants: formulations in order, each in :data:`PRIORITY`
    order (so ``limit=3`` keeps the core triple of the first formulation)."""
    if not limit:
        return list(variants)
    groups: dict[tuple[str, str], list] = {}
    for v in variants:
        groups.setdefault(group_key(v), []).append(v)
    out: list = []
    for key, vs in groups.items():
        for v in sorted(vs, key=priority_rank):
            if len(out) >= int(limit):
                return out
            out.append(v)
    return out


def build_family_variants(fam: stimuli.Family, tokenizer, cfg: RunConfig) -> tuple[list, list[dict]]:
    """All variants of a family for ``cfg.k_levels`` x ``cfg.c_levels`` plus
    ``cfg.k_extreme`` with ``C="seg"``; build failures are returned as problem rows."""
    problems: list[dict] = []
    out: list = []
    conds: list[tuple[int, str]] = []
    if 0 in cfg.k_levels:
        conds.append((0, "C1"))
    for K in cfg.k_levels:
        if K == 0:
            continue
        conds.extend((int(K), str(C)) for C in cfg.c_levels)
    conds.extend((int(K), "seg") for K in (cfg.k_extreme or ()))
    for K, C in conds:
        try:
            mask = stimuli.segment_masks(fam.n_units, K, C)
            out.append(stimuli.make_variant(fam, mask, K, C, tokenizer))
        except Exception as e:  # noqa: BLE001 - record and continue
            problems.append(dict(tag=f"{fam.tag}:K{K}:{C}", family=fam.family_id, formulation=fam.formulation,
                                 K=K, C=C, step="build", problem=str(e)))
    return out, problems


def load_variants(cfg: RunConfig, mb: ModelBundle) -> tuple[list, pd.DataFrame]:
    """Variants of the selected families (see :func:`select_families`) with the K=0
    consistency checks applied.

    Returns ``(variants, checks)``: ``checks`` has one row per built or failed variant
    (``tag, family, formulation, K, C, step, problem, excluded``); variants with a
    problem are dropped (and a formulation without a K=0 reference is dropped entirely).
    ``cfg.limit_variants`` is applied last (:func:`apply_variant_limit`)."""
    errors: dict = {}
    fams = select_families(cfg, errors)
    rows: list[dict] = []
    for fname, probs in errors.items():
        rows.append(dict(tag=fname, family=fname.split(".")[0][:-1], formulation=fname.split(".")[0][-1:],
                         K=-1, C="", step="schema", problem="; ".join(probs), excluded=True))
    kept: list = []
    for fam in fams:
        vs, problems = build_family_variants(fam, mb.tokenizer, cfg)
        for p in problems:
            rows.append(dict(p, excluded=True))
        try:
            ref = stimuli.reference_variant(vs)
        except ValueError:
            for v in vs:
                rows.append(dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=v.K, C=v.C,
                                 step="check", problem="no K=0 reference variant", excluded=True))
            continue
        for v in vs:
            probs = stimuli.check_variant(v, ref)
            rows.append(dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=v.K, C=v.C,
                             step="check", problem="; ".join(probs), excluded=bool(probs)))
            if not probs:
                kept.append(v)
    checks = pd.DataFrame(rows, columns=["tag", "family", "formulation", "K", "C", "step", "problem", "excluded"])
    return apply_variant_limit(kept, cfg.limit_variants), checks


def core_subset(variants: Sequence) -> list:
    """The core variants: per standard formulation the conditions in
    :data:`CORE_CONDITIONS`; for the original family K=0 and the largest K."""
    out: list = []
    orig: dict[tuple[str, str], list] = {}
    for v in variants:
        if is_original(v):
            orig.setdefault(group_key(v), []).append(v)
        elif (int(v.K), str(v.C)) in CORE_CONDITIONS:
            out.append(v)
    for vs in orig.values():
        ks = sorted({int(v.K) for v in vs})
        out.extend(v for v in vs if int(v.K) in (ks[0], ks[-1]))
    return out


def reference_of(variants: Sequence) -> dict[tuple[str, str], object]:
    """``{(family, formulation): K=0 variant}``."""
    return {group_key(v): v for v in variants if int(v.K) == 0}


# --------------------------------------------------------------------------------------
# the original State 2143 context
# --------------------------------------------------------------------------------------
@dataclass
class Original:
    """The verified anchor sequence, its ``kind="original"`` family and variants."""
    anchor: object
    family: stimuli.Family
    variants: list

    @property
    def context_ids(self) -> list[int]:
        return list(self.anchor.context_ids)

    @property
    def target_id(self) -> int:
        return int(self.anchor.target_id)

    @property
    def position(self) -> int:
        """The dossier position (last context token)."""
        return len(self.context_ids) - 1

    def info(self) -> dict:
        return dict(sha256_context=self.anchor.sha256_context, sha256_sequence=self.anchor.sha256_sequence,
                    verified=bool(self.anchor.verified), n_context=len(self.context_ids),
                    target_id=self.target_id, n_variants=len(self.variants))


def load_original(cfg: RunConfig, mb: ModelBundle, limits: Optional[Limits] = None) -> Optional[Original]:
    """State-2143 anchor (``cache/state2143``, strict hash check) and the original-context
    variants (K in 0/2/4/5).  Returns ``None`` with a warning when the sequence cannot be
    fetched (offline) or the variants cannot be built with this tokenizer.  In smoke mode
    only K=0 and the largest K are kept."""
    from ..anchor import load_anchor
    try:
        a = load_anchor(cache_dir=cache_dir("state2143"), tokenizer=mb.tokenizer, strict=True)
    except Exception as e:  # noqa: BLE001 - offline or hash mismatch
        warnings.warn(f"State 2143 anchor not available ({e!r}); original-context variants skipped")
        return None
    try:
        fam = stimuli.original_2143_family(a, mb.tokenizer)
        vs = stimuli.build_original_variants(a, mb.tokenizer)
        ref = stimuli.reference_variant(vs)
        vs = [v for v in vs if not stimuli.check_variant(v, ref)]
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"original-context variants could not be built ({e!r})")
        return None
    if limits is not None and limits.smoke:
        vs = core_subset(vs)
    return Original(anchor=a, family=fam, variants=vs)


# --------------------------------------------------------------------------------------
# direction bank
# --------------------------------------------------------------------------------------
@dataclass
class DirectionBank:
    """Named directions at one site plus the frequent-token ids (dict-like access)."""
    site: HookSite
    dirs: dict[str, Direction] = field(default_factory=dict)
    freq_ids: list[int] = field(default_factory=list)
    unigram_counts: Optional[np.ndarray] = None
    sae: object = None
    sae_table: Optional[pd.DataFrame] = None
    meta: dict = field(default_factory=dict)

    def __getitem__(self, name: str) -> Direction:
        return self.dirs[name]

    def __contains__(self, name) -> bool:
        return name in self.dirs

    def __iter__(self):
        return iter(self.dirs)

    def __len__(self) -> int:
        return len(self.dirs)

    def keys(self):
        return self.dirs.keys()

    def items(self):
        return self.dirs.items()

    def values(self):
        return self.dirs.values()

    def get(self, name: str, default=None):
        return self.dirs.get(name, default)

    @property
    def rand_names(self) -> list[str]:
        return [n for n in self.dirs if n.startswith("rand_")]

    @property
    def sae_names(self) -> list[str]:
        return [n for n in self.dirs if n.startswith("sae_")]

    def names(self, families: Sequence[str]) -> list[str]:
        """Direction names whose :func:`dir_family` is in ``families`` (in bank order)."""
        fams = set(families)
        return [n for n in self.dirs if dir_family(n) in fams]

    def freq_tensor(self, device=None) -> Optional[torch.Tensor]:
        if not self.freq_ids:
            return None
        return torch.as_tensor(self.freq_ids, dtype=torch.long, device=device)


def bank_signature(variants: Sequence, site: HookSite, cfg: RunConfig, limits: Limits, original) -> str:
    """Hash identifying a direction bank (variants, site, seed, random count, SAE use)."""
    tags = sorted(v.tag for v in variants)
    payload = json.dumps(dict(tags=tags, site=site.key, seed=cfg.seed, n_random=cfg.n_random_dirs,
                              use_sae=bool(limits.use_sae), n_unigram=limits.n_unigram,
                              original=bool(original), model=cfg.model.name, revision=cfg.model.revision))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def unigram_counts(mb: ModelBundle, n_sequences: int, seed: int, variants: Sequence = ()) -> tuple[np.ndarray, dict]:
    """Unigram counts for the ``freq`` axis: ``n_sequences`` Pile sequences by range request
    (cached under ``cache/unigram``), or - offline / on failure - the token counts of the
    stimulus bank with add-one smoothing (``source`` says which)."""
    if not offline():
        try:
            from ..pile_fetch import sample_unigram_counts
            counts = sample_unigram_counts(int(n_sequences), int(seed), vocab=mb.vocab, cache_dir=cache_dir("unigram"))
            return np.asarray(counts, dtype=np.int64), dict(source="pile", n_sequences=int(n_sequences), seed=int(seed))
        except Exception as e:  # noqa: BLE001 - network problems
            warnings.warn(f"unigram counts could not be fetched ({e!r}); using stimulus-bank token counts")
    counts = np.ones(mb.vocab, dtype=np.int64)
    for v in variants:
        ids = np.asarray([int(i) for i in v.ids], dtype=np.int64)
        counts += np.bincount(ids[(ids >= 0) & (ids < mb.vocab)], minlength=mb.vocab)[: mb.vocab]
    return counts, dict(source="stimulus_bank", n_sequences=len(list(variants)), seed=int(seed))


def _sae_hookpoint(mb: ModelBundle, site: HookSite) -> Optional[tuple[str, str]]:
    """``(repo, hookpoint)`` of a published SAE matching ``site`` or ``None``."""
    from ..sae import sae_repo_for
    rep = sae_repo_for(mb.spec.name)
    if rep is None:
        return None
    repo, fmt = rep
    if fmt.endswith(".mlp"):
        if site.site != "mlp_output":
            return None
        return repo, fmt.format(n=site.layer)
    hp = site.sae_hookpoint
    if hp is None or site.site not in ("block_input", "block_output"):
        return None
    return repo, hp


def direction_bank(cfg: RunConfig, mb: ModelBundle, variants: Sequence, site: HookSite,
                   original: Optional[Original] = None, limits: Optional[Limits] = None,
                   sae=None, axis_npz: Optional[str] = None) -> DirectionBank:
    """Build the direction bank at ``site`` (DESIGN section 3).

    * ``ent``: entropy gradient at ``site`` and the dossier position of the original
      anchor (last context token) if available, else at the last token of the first K=0
      variant; optionally orthogonalised against the ARI axis in ``axis_npz``;
    * ``lang_dom``: denoised difference of means D - E over segment-final residuals;
    * ``lang_probe``: logistic-regression axis on the same activations;
    * ``freq``: frequency axis from unigram counts (``limits.n_unigram`` Pile sequences);
      ``freq_ids`` = the 100 most frequent ids (for ``p_freq``);
    * ``rand_00 .. rand_{n-1}``: ``cfg.n_random_dirs`` isotropic directions (``cfg.seed``);
    * ``sae_D0.., sae_E0..``: decoder rows of language-selective TopK-SAE features when
      ``limits.use_sae`` and an SAE exists for the model (or an ``sae`` object is given,
      e.g. a synthetic ``TopKSAE`` in tests).  FVU at the site is recorded; FVU > 0.4
      marks the SAE as not certified (meta) but the directions are still built.

    Directions whose construction fails (e.g. ``lang_dom`` without German segments) are
    skipped with a warning and listed in ``bank.meta['skipped']``."""
    limits = limits or Limits()
    bank = DirectionBank(site=site, meta=dict(skipped={}))
    variants = list(variants)
    std = [v for v in variants if not is_original(v)] or variants

    # ent
    try:
        if original is not None:
            ids = torch.as_tensor(original.context_ids, dtype=torch.long, device=mb.device)[None, :]
            pos, src = original.position, "original_2143"
        else:
            ref = next((v for v in std if int(v.K) == 0), std[0])
            ids, pos, src = ids_tensor(mb, ref), int(ref.n_tokens) - 1, ref.tag
        axis = load_axis_npz(axis_npz) if axis_npz else None
        ent = entropy_direction(mb, ids, site, pos, remove_axis=axis)
        ent.meta.update(source=src, dossier_position=STATE2143["hook_position"], axis_npz=axis_npz or "")
        bank.dirs["ent"] = ent
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"ent direction skipped: {e!r}")
        bank.meta["skipped"]["ent"] = repr(e)

    # lang_dom / lang_probe
    acts = labels = None
    try:
        bank.dirs["lang_dom"] = language_axis(mb, std, site, pool="segment_last", denoise_pcs=1)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"lang_dom direction skipped: {e!r}")
        bank.meta["skipped"]["lang_dom"] = repr(e)
    try:
        acts, labels, _meta = collect_activations(mb, std, site, which="segment_last")
        w, acc = probe_axis(acts, labels)
        bank.dirs["lang_probe"] = Direction("lang_probe", w, site,
                                            dict(train_acc=float(acc), n=int(acts.shape[0]),
                                                 n_E=int(sum(1 for l in labels if l == "E")),
                                                 n_D=int(sum(1 for l in labels if l == "D"))))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"lang_probe direction skipped: {e!r}")
        bank.meta["skipped"]["lang_probe"] = repr(e)

    # freq
    try:
        counts, cinfo = unigram_counts(mb, limits.n_unigram, cfg.seed, std)
        fq = frequency_axis(mb, counts, top_n=100, site=site)
        fq.meta.update(cinfo)
        bank.dirs["freq"] = fq
        bank.freq_ids = [int(i) for i in fq.meta["top_ids"]]
        bank.unigram_counts = counts
        bank.meta["unigram"] = cinfo
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"freq direction skipped: {e!r}")
        bank.meta["skipped"]["freq"] = repr(e)

    # rand
    for d in random_direction_set(mb.d_model, int(cfg.n_random_dirs), int(cfg.seed), site, name="rand_"):
        bank.dirs[d.name] = d

    # sae
    if sae is not None or limits.use_sae:
        try:
            from ..sae import TopKSAE, feature_direction, feature_table, language_selective_features, sae_fvu
            info: dict = {}
            if sae is None:
                hp = _sae_hookpoint(mb, site)
                if hp is None:
                    raise ValueError(f"no published SAE for {mb.spec.name} at {site.key}")
                repo, hookpoint = hp
                sae = TopKSAE.load_from_hub(repo, hookpoint, device=str(mb.device))
                info.update(repo=repo, hookpoint=hookpoint)
            acts_all, labels_all, _m = collect_activations(mb, std, site, which="all_tokens")
            fvu = float(sae_fvu(sae, acts_all))
            table = feature_table(sae, acts_all, labels_all)
            sel = language_selective_features(table, n=int(limits.n_sae_features))
            info.update(fvu=fvu, certified=bool(fvu <= 0.4), k=int(sae.k), num_latents=int(sae.num_latents),
                        n_acts=int(acts_all.shape[0]), features_D=sel["D"], features_E=sel["E"])
            for lang in ("D", "E"):
                for i, f in enumerate(sel[lang]):
                    row = table.iloc[int(f)]
                    bank.dirs[f"sae_{lang}{i}"] = Direction(
                        f"sae_{lang}{i}", feature_direction(sae, int(f)), site,
                        dict(feature=int(f), lang=lang, fvu=fvu, certified=bool(fvu <= 0.4),
                             selectivity=float(row["selectivity"]), rate_D=float(row["rate_D"]),
                             rate_E=float(row["rate_E"])))
            bank.sae, bank.sae_table = sae, table
            bank.meta["sae"] = info
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"SAE directions skipped: {e!r}")
            bank.meta["skipped"]["sae"] = repr(e)
    bank.meta["n_variants"] = len(std)
    return bank


# ---- persistence -----------------------------------------------------------------------
def save_bank(bank: DirectionBank, run_dir, name: str = "directions") -> Path:
    """``<run_dir>/<name>.npz`` (unit vectors keyed by name, ``freq_ids``) and
    ``<run_dir>/<name>_meta.json`` (site, per-direction meta)."""
    d = Path(os.fspath(run_dir))
    d.mkdir(parents=True, exist_ok=True)
    arrays = {n: dd.vec.numpy().astype(np.float32) for n, dd in bank.dirs.items()}
    arrays["freq_ids"] = np.asarray(bank.freq_ids, dtype=np.int64)
    np.savez(d / f"{name}.npz", **arrays)
    meta = dict(site=dict(layer=bank.site.layer, site=bank.site.site), names=list(bank.dirs),
                freq_ids=list(bank.freq_ids), bank_meta=to_jsonable(bank.meta),
                directions={n: to_jsonable({k: v for k, v in dd.meta.items() if k != "vec_plain"})
                            for n, dd in bank.dirs.items()})
    (d / f"{name}_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return d / f"{name}.npz"


def load_bank(run_dir, name: str = "directions") -> Optional[DirectionBank]:
    """Inverse of :func:`save_bank` (``None`` if the files are missing)."""
    d = Path(os.fspath(run_dir))
    npz, mj = d / f"{name}.npz", d / f"{name}_meta.json"
    if not (npz.exists() and mj.exists()):
        return None
    meta = json.loads(mj.read_text(encoding="utf-8"))
    site = HookSite(int(meta["site"]["layer"]), str(meta["site"]["site"]))
    bank = DirectionBank(site=site, meta=dict(meta.get("bank_meta", {})))
    with np.load(npz) as z:
        for n in meta["names"]:
            bank.dirs[n] = Direction(n, torch.as_tensor(np.asarray(z[n])), site, dict(meta["directions"].get(n, {})))
        bank.freq_ids = [int(i) for i in meta.get("freq_ids", [])]
    return bank


# --------------------------------------------------------------------------------------
# context shared between stages of one run
# --------------------------------------------------------------------------------------
def get_variants(cfg: RunConfig, mb: ModelBundle, ctx: Optional[dict]) -> tuple[list, pd.DataFrame]:
    """Variants + checks from ``ctx`` (filled on first use) or :func:`load_variants`."""
    if ctx is not None and "variants" in ctx:
        return ctx["variants"], ctx.get("checks", pd.DataFrame())
    variants, checks = load_variants(cfg, mb)
    if ctx is not None:
        ctx["variants"], ctx["checks"] = variants, checks
    return variants, checks


def get_original(cfg: RunConfig, mb: ModelBundle, ctx: Optional[dict], limits: Limits) -> Optional[Original]:
    if ctx is not None and "original" in ctx:
        return ctx["original"]
    orig = load_original(cfg, mb, limits)
    if ctx is not None:
        ctx["original"] = orig
    return orig


def all_variants(variants: Sequence, original: Optional[Original]) -> list:
    """Standard variants followed by the original-context variants (if any)."""
    return list(variants) + (list(original.variants) if original is not None else [])


def get_bank(cfg: RunConfig, mb: ModelBundle, variants: Sequence, site: HookSite, ctx: Optional[dict],
             limits: Limits, original: Optional[Original] = None, sae=None) -> DirectionBank:
    """Direction bank from ``ctx``, from a stage-B run directory of the same model under
    ``cfg.out_root`` (if its manifest carries the same :func:`bank_signature`), or freshly
    built with :func:`direction_bank`."""
    sig = bank_signature(variants, site, cfg, limits, original)
    if ctx is not None and isinstance(ctx.get("bank"), DirectionBank) and ctx.get("bank_signature") == sig:
        return ctx["bank"]
    bank = None
    try:
        rd = RunDir(cfg.out_root, "B", cfg.model, mkdir=False)
        if rd.manifest_path.exists() and read_manifest(rd).get("bank_signature") == sig:
            bank = load_bank(rd)
            if bank is not None:
                bank.meta["source"] = f"disk:{rd.path}"
    except Exception:  # noqa: BLE001 - any unreadable previous run: rebuild
        bank = None
    if bank is None:
        bank = direction_bank(cfg, mb, variants, site, original=original, limits=limits, sae=sae)
        bank.meta["source"] = "built"
    if ctx is not None:
        ctx["bank"], ctx["bank_signature"] = bank, sig
    return bank


# --------------------------------------------------------------------------------------
# marker lexicon (D9)
# --------------------------------------------------------------------------------------
def marker_lexicon(tokenizer, words: Optional[dict] = None) -> dict[str, list[int]]:
    """Single-token ids of frequent English / German function words (words that are not
    one token are skipped)."""
    words = words or MARKER_WORDS
    out: dict[str, list[int]] = {}
    for lang, ws in words.items():
        ids: list[int] = []
        for w in ws:
            t = tokenizer(w, add_special_tokens=False)["input_ids"]
            if len(t) == 1:
                ids.append(int(t[0]))
        out[lang] = ids
    return out


# --------------------------------------------------------------------------------------
# stage bookkeeping
# --------------------------------------------------------------------------------------
def finish_stage(run_dir, stage: str, cfg: RunConfig, mb: ModelBundle, tables: dict[str, pd.DataFrame],
                 limits: Optional[Limits] = None, t0: Optional[float] = None, **meta) -> dict[str, pd.DataFrame]:
    """Save every table (csv [+ parquet]) and write the manifest; returns ``tables``."""
    for name, df in tables.items():
        save_df(df, run_dir, name)
    write_manifest(run_dir, stage=stage, status="ok", cfg=cfg, model=mb.spec,
                   limits=(asdict(limits) if limits is not None else None),
                   n_layers=mb.n_layers, d_model=mb.d_model, n_heads=mb.n_heads, vocab=mb.vocab,
                   tables={n: int(len(df)) for n, df in tables.items()},
                   runtime_s=(time.perf_counter() - t0) if t0 is not None else None, **meta)
    return tables


def write_error(run_dir, stage: str, exc: BaseException, cfg: Optional[RunConfig] = None,
                mb: Optional[ModelBundle] = None) -> Path:
    """Write ``error.txt`` (traceback) and a manifest with ``status='error'``."""
    d = Path(os.fspath(run_dir))
    d.mkdir(parents=True, exist_ok=True)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    (d / "error.txt").write_text(tb, encoding="utf-8")
    write_manifest(d, stage=stage, status="error", error=repr(exc), cfg=cfg,
                   model=(mb.spec if mb is not None else None))
    return d / "error.txt"


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


__all__ = [
    "PRIORITY", "CORE_CONDITIONS", "MARKER_WORDS", "PATCH_CLASSES", "ID_COLS", "Limits", "smoke_config",
    "get_limits", "cache_root", "cache_dir", "resolve_families_dir", "offline", "dossier_site",
    "depth_layer", "depth_site", "variant_kind", "is_original", "ids_tensor", "tail_positions",
    "unit_last_position", "variant_row", "dir_family", "progress", "select_families", "priority_rank",
    "group_key", "apply_variant_limit", "build_family_variants", "load_variants", "core_subset",
    "reference_of", "Original", "load_original", "DirectionBank", "bank_signature", "unigram_counts",
    "direction_bank", "save_bank", "load_bank", "get_variants", "get_original", "all_variants",
    "get_bank", "marker_lexicon", "finish_stage", "write_error", "pearson",
]
