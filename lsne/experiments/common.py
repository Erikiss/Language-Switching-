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
import shutil
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
#: Position classes evaluated by the counterfactual attribution (stage C).  ``delim`` is
#: the pre-registered union class R_ws+punct+boundary measured as ONE patch set
#: (PREREGISTRATION.md section 3, contrast 3; DESIGN section 4.1), ``post_punct`` the
#: tokenizer-adequate stand-in for a whitespace token (the space-carrying token after a
#: punctuation token).  ``ws`` (whitespace-only tokens) is kept as a pre-registered
#: per-class row although it is empty by construction for the stimulus families under the
#: GPT-NeoX BPE (the space is attached to the following word); it is measurable on the
#: original-2143 context (newline tokens) and flagged ``measurable=False`` elsewhere.
PATCH_CLASSES = ("ws", "punct", "boundary", "delim", "post_punct", "word_start", "word_cont",
                 "switch_context", "tail")
#: The pre-registered confirmatory class of contrast 3 (R_delim - R_random).
PREREG_PATCH_CLASS = "delim"
#: Standard id columns of every tidy table.
ID_COLS = ("tag", "family", "formulation", "K", "C", "kind", "n_tokens", "tail_start")
#: D3-D7 generation core per (K, C) cell in the limited stage D of the scale sweep (stage F):
#: the full core (~96 formulations x 3 conditions x 2 directions of generation runs) takes
#: 1-2 A100-hours per checkpoint; a seeded subset of 24 formulations, each with all its core
#: conditions (so at most 24 variants per cell; the original-context variants always kept),
#: bounds it (:func:`limited_core`).  Full runs (not limited) use every core variant.
LIMITED_GEN_MAX_CORE = 24
#: Variants per checkpoint chunk of the resumable stage C / D substages (:class:`Resume`).
RESUME_CHUNK = 50

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: Pre-registered family domain groups (PREREGISTRATION.md section 3, contrast 13):
#: culture/politics F13-F16, law F17-F18, protein misfolding F19-F20 (DESIGN 4.8b),
#: everything else (F01-F12, DEV01-DEV04, the original context) = everyday / stable.
DOMAIN_GROUPS: dict[str, tuple[str, ...]] = {
    "culture_politics": ("F13", "F14", "F15", "F16"),
    "law": ("F17", "F18"),
    "protein_misfolding": ("F19", "F20"),
}
DEFAULT_DOMAIN_GROUP = "everyday"
#: Smoke-preset overrides of the stage G-L limits (see :class:`Limits`).
SMOKE_GL: dict = dict(
    g_k_levels=(0, 8), g_c_levels=("C1",), g_max_masks=3, g_deflate_k=(0, 5), g_r_mean=128, g_r_last=256,
    g_steer_eps=(0.05, 0.5), g_n_subsample=5, g_position_sites=2,
    h_lambdas_rel=(0.0, 0.3, 3.0, 30.0), h_n_steps=6, h_max_starts=1, h_knn=50, h_cov_rank=32,
    h_discriminants=("target_logit", "probe", "sae_feature"),
    i_doses=(0.1,), i_n_templates=1, i_max_words=6, i_head_eps=(0.5,), i_n_random_profile=4,
    j_n_pile=3, j_pile_max_len=64, j_n_random_bases=4, j_n_pca=4, j_max_variants_per_group=4, j_max_sites=2,
    k_seq_len=32, k_max_docs=4, k_max_snippets=2, k_mem_steps=3, k_mem_lr=1e-3, k_train_steps=3, k_train_lr=3e-4,
    k_train_batch=2, k_checkpoints=None, k_eps=0.1,
    l_source="rotated_self", l_n_texts=12, l_n_prompts=1, l_n_prefix=2, l_eps=1.0, l_n_null=3,
    synthetic_sae_if_missing=True, synthetic_sae_latents=64,
)


# --------------------------------------------------------------------------------------
# limits
# --------------------------------------------------------------------------------------
@dataclass
class Limits:
    """Run-size parameters outside :class:`~lsne.config.RunConfig`.

    ``is_smoke`` marks the < 3 min CPU preset (built by :meth:`smoke`; the field must not share
    the classmethod's name, or the bound method shadows the default and is truthy in every
    run), ``limited`` the reduced
    stages C/D of the scale sweep (stage F): fewer random directions in D1, no random
    directions in the grid, generation diagnostics for ``lang_dom`` only.
    """
    is_smoke: bool = False
    limited: bool = False
    progress: bool = True
    # stage B
    n_unigram: int = 2000           # Pile sequences for the unigram counts (DESIGN section 3: 2 000; cache/unigram)
    use_sae: bool = False
    n_sae_features: int = 5         # sae_D0..n-1 and sae_E0..n-1 directions
    # stage C
    n_random_patch: int = 10        # random position draws per (site, class)
    n_boot: int = 2000              # cluster-bootstrap resamples
    grid_batch_size: int = 64       # grid rows (direction x sign x dose) per batched forward
    # stage D
    damage_doses: tuple = (0.01, 0.05, 0.1)
    damage_batch_size: int = 64     # D1 rows per forward incl. the reference (lsne.dynamics.DAMAGE_BATCH_SIZE)
    gen_n_steps: int = 8            # D3 ramp steps
    gen_tokens_per_step: int = 8    # D3 tokens per plateau
    gen_t_on: int = 16              # D4 quench time
    n_boot_tau: int = 200           # D4 between-sample bootstrap resamples (gate on b, CI of tau)
    gen_max_core: Optional[int] = None  # D3-D7 core formulations when limited, i.e. <= n per (K, C) cell (None = all; as_limited: 24)
    ca_threshold: float = 0.5
    # stage E
    n_families_tn: int = 2          # family ids used in stage E
    tn_doses: tuple = (0.01, 0.1)
    tn_n_steps: int = 500           # TNCO annealing steps
    tn_n_runs: int = 4
    # stage F
    f_stages: tuple = ("B", "C", "D")
    # ---- stages G-L (docs/STAGES.md) ---------------------------------------------------
    # stage G (local geometry / trajectory)
    g_k_levels: tuple = (0, 2, 4, 8)      # K levels of the realization clouds
    g_c_levels: tuple = ("C1", "C2", "C3")
    g_max_masks: int = 8                  # unit masks per (K, C) cloud (seeded subset; K=2 and K=8 have exactly 8)
    g_deflate_k: tuple = (0, 5)           # deflation levels (PREREG 7/8: k=5)
    g_r_mean: int = 128                   # Mahalanobis rank, mean readout
    g_r_last: int = 256                   # Mahalanobis rank, last-token readout
    g_steer_eps: tuple = (0.05, 0.2, 0.5) # T3 doses (PREREG 9: 0.05)
    g_n_subsample: int = 20               # seeded draws of the n-matched contraction contrast
    g_position_sites: int = 3             # sites of the T2 position trajectory (first, middle, dossier)
    # stage H (lambda dynamics)
    h_lambdas_rel: tuple = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)   # DESIGN 4.6
    h_n_steps: int = 100                  # Alg. 1 steps per path
    h_step_rel: float = 0.02              # step as a fraction of ||h0||
    h_dmax_rel: float = 1.0               # d_max as a fraction of ||h0|| (l2 ball)
    h_cross_nats: float = 1.0             # boundary offset of the model-own discriminants (B >= 1 nat / +1 nat entropy)
    h_max_starts: int = 4                 # start states (K=0 formulations) per family
    h_knn: int = 50                       # k_nearest of the KDE (the paper's 50)
    h_cov_rank: int = 32                  # whitening rank of the target cloud
    h_discriminants: tuple = ("target_logit", "probe", "entropy", "sae_feature")
    # stage I (compass / heads)
    i_doses: tuple = (0.01, 0.05, 0.1)    # B(x) doses of the direction x domain table
    i_n_templates: Optional[int] = None   # template sentences per domain axis (None = all)
    i_max_words: Optional[int] = None     # domain words per side (None = all)
    i_head_eps: tuple = (0.1, 0.5)        # relative doses of the head arm
    i_n_random_profile: int = 32          # random directions of the domain-profile z (PREREG 12: 32)
    i_frozen_head: Optional[str] = None   # "L:H" frozen worst head from a development run (PREREG 14); None = select on a family half
    # stage J (RDX / pile / sharing)
    j_n_pile: int = 2000                  # Pile sequences of the difficulty ranking (PREREG 16: >= 2000)
    j_pile_max_len: int = 512
    j_n_random_bases: int = 32            # random bases of the RDX z (PREREG 15: 32)
    j_n_pca: int = 8                      # principal components of the "pca" basis
    j_max_variants_per_group: Optional[int] = None
    j_max_sites: Optional[int] = None     # keep only the last n grid sites (smoke)
    # stage K (test-time training)
    k_corpus: str = "data/ttt/dao_fork"
    k_seq_len: int = 256
    k_max_docs: Optional[int] = None
    k_max_snippets: int = 16              # snippets per arm of the memorisation table
    k_mem_steps: int = 50                 # memorisation_speed steps
    k_mem_lr: float = 1e-4
    k_threshold: float = 0.5              # pre-defined NLL threshold (nats) of the "fast phase"
    k_train_steps: int = 200              # test_time_train steps per arm
    k_train_lr: float = 1e-4
    k_train_batch: int = 4
    k_checkpoints: Optional[tuple] = None # extra revisions of the memorisation table (None = current only)
    k_eps: float = 0.05                   # dose of the before/after B(x)
    # stage L (transplant)
    # Source model of stage L. Full runs use a genuinely different model (pre-registration 18:
    # "Stand-in-Validierung"); "self" / "rotated_self" are plumbing checks without a verdict.
    l_source: str = "EleutherAI/pythia-70m@step143000"   # "model@revision" | "rotated_self" | "self"
    l_n_texts: int = 48                   # paired E/D unit sentences per language
    l_n_prompts: int = 3                  # prompts of the transplant experiment
    l_n_prefix: int = 8                   # virtual KV prefix length
    l_eps: float = 1.0                    # dose of the transferred feature
    l_n_null: int = 10                    # permuted-pair null draws of the functional test
    # shared by H / J / K: use a random TopK SAE when no published SAE is available
    # (smoke runs only: exercises the code paths; every row is marked sae_source)
    synthetic_sae_if_missing: bool = False
    synthetic_sae_latents: int = 64

    @classmethod
    def smoke(cls, **kw) -> "Limits":
        """Preset for the CPU smoke run (pythia-14m, one family, three variants)."""
        base = dict(is_smoke=True, n_unigram=5, n_random_patch=2, n_boot=200, damage_doses=(0.01, 0.1),
                    gen_n_steps=2, gen_tokens_per_step=2, gen_t_on=2, n_boot_tau=5,
                    n_families_tn=1, tn_doses=(0.01, 0.1), tn_n_steps=50, tn_n_runs=1,
                    use_sae=False)
        base.update(SMOKE_GL)
        base.update(kw)
        return cls(**base)

    def as_limited(self) -> "Limits":
        """Copy with the reduced-stage flag of the scale sweep set (also caps the D3-D7
        generation core at :data:`LIMITED_GEN_MAX_CORE` variants per (K, C) cell unless
        ``gen_max_core`` is already set)."""
        return replace(self, limited=True, n_random_patch=min(self.n_random_patch, 5),
                       gen_max_core=(self.gen_max_core if self.gen_max_core is not None else LIMITED_GEN_MAX_CORE))


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
    if limits is not None and limits.is_smoke:
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


class UnigramFetchError(RuntimeError):
    """The Pile unigram counts of a full run could not be fetched (no silent fallback)."""


def unigram_counts(mb: ModelBundle, n_sequences: int, seed: int, variants: Sequence = (),
                   strict: bool = False) -> tuple[np.ndarray, dict]:
    """Unigram counts for the ``freq`` axis: ``n_sequences`` Pile sequences by range request
    (cached under ``cache/unigram``), or - offline / on failure - the token counts of the
    stimulus bank with add-one smoothing (``source`` says which).

    ``strict`` (full runs; :func:`direction_bank` passes ``not limits.is_smoke``): a failed
    fetch raises :class:`UnigramFetchError` instead of falling back.  The stimulus-bank top
    100 contains German function words, which would make ``p_freq`` and the ``freq`` axis
    language-confounded; ``LSNE_OFFLINE=1`` selects the stimulus-bank counts explicitly."""
    if not offline():
        try:
            from ..pile_fetch import sample_unigram_counts
            counts = sample_unigram_counts(int(n_sequences), int(seed), vocab=mb.vocab, cache_dir=cache_dir("unigram"))
            return np.asarray(counts, dtype=np.int64), dict(source="pile", n_sequences=int(n_sequences), seed=int(seed))
        except Exception as e:  # noqa: BLE001 - network problems
            if strict:
                raise UnigramFetchError(f"Pile unigram fetch failed on a full run ({e!r}); set LSNE_OFFLINE=1 to "
                                        f"accept stimulus-bank counts explicitly") from e
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
    * ``lang_dom``: denoised difference of means D - E over the segment-final residuals of
      the paired unit segments (the shared English tail is excluded, ``include_tail=False``,
      DESIGN section 3);
    * ``lang_probe``: logistic-regression axis on the same activations;
    * ``freq``: frequency axis from unigram counts (``limits.n_unigram`` Pile sequences,
      2 000 by default as in DESIGN section 3; the smoke preset uses 5);
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
        bank.dirs["lang_dom"] = language_axis(mb, std, site, pool="segment_last", denoise_pcs=1, include_tail=False)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"lang_dom direction skipped: {e!r}")
        bank.meta["skipped"]["lang_dom"] = repr(e)
    try:
        acts, labels, _meta = collect_activations(mb, std, site, which="segment_last", include_tail=False)
        w, acc = probe_axis(acts, labels)
        bank.dirs["lang_probe"] = Direction("lang_probe", w, site,
                                            dict(train_acc=float(acc), n=int(acts.shape[0]),
                                                 n_E=int(sum(1 for l in labels if l == "E")),
                                                 n_D=int(sum(1 for l in labels if l == "D")),
                                                 include_tail=False))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"lang_probe direction skipped: {e!r}")
        bank.meta["skipped"]["lang_probe"] = repr(e)

    # freq
    try:
        counts, cinfo = unigram_counts(mb, limits.n_unigram, cfg.seed, std, strict=not limits.is_smoke)
        fq = frequency_axis(mb, counts, top_n=100, site=site)
        fq.meta.update(cinfo)
        bank.dirs["freq"] = fq
        bank.freq_ids = [int(i) for i in fq.meta["top_ids"]]
        bank.unigram_counts = counts
        bank.meta["unigram"] = cinfo
    except UnigramFetchError:
        raise                       # a full run must not go on with a confounded or missing freq axis
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
    if bank is not None:
        # the npz holds only the direction vectors: restore the SAE object of a stored bank
        # (stages H-J use bank.sae); a hub failure drops only the SAE, not the bank
        info = bank.meta.get("sae") or {}
        if sae is not None:
            bank.sae = sae
        elif limits.use_sae and isinstance(info, dict) and info.get("repo") and info.get("hookpoint"):
            try:
                from ..sae import TopKSAE
                bank.sae = TopKSAE.load_from_hub(info["repo"], info["hookpoint"], device=str(mb.device))
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"SAE of the stored bank could not be reloaded: {e!r}")
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


# --------------------------------------------------------------------------------------
# stages G-L: shared helpers
# --------------------------------------------------------------------------------------
def family_domain(family_id: str) -> str:
    """Pre-registered domain group of a family id (:data:`DOMAIN_GROUPS`; everything
    else, including ``DEV*`` and the original context, is ``"everyday"``)."""
    fid = str(family_id)
    for group, ids in DOMAIN_GROUPS.items():
        if fid in ids:
            return group
    return DEFAULT_DOMAIN_GROUP


def families_in_group(variants: Sequence, group: str) -> list[str]:
    """Sorted family ids of ``variants`` whose :func:`family_domain` is ``group``."""
    return sorted({str(v.family_id) for v in variants if family_domain(v.family_id) == group})


def k_subset(variants: Sequence, K: int, C: Optional[str] = None, standard_only: bool = True) -> list:
    """Variants with the given ``K`` (and ``C``), standard families only by default."""
    out = []
    for v in variants:
        if standard_only and is_original(v):
            continue
        if int(v.K) != int(K):
            continue
        if C is not None and str(v.C) != str(C):
            continue
        out.append(v)
    return out


def prereg_layer_grid(n_layers: int) -> tuple[int, ...]:
    """The pre-registered layer grid of stage C ({2, 5, 8, 11} for 12 blocks)."""
    from .stage_c_grid import attribution_layer_grid   # local import: stage C imports this module
    return attribution_layer_grid(int(n_layers))


def grid_sites(mb: ModelBundle) -> list[HookSite]:
    """Block inputs of the pre-registered layer grid."""
    return [HookSite(int(l), "block_input") for l in prereg_layer_grid(mb.n_layers)]


def last_state(mb: ModelBundle, v, site: HookSite) -> torch.Tensor:
    """Residual of variant ``v`` at ``site`` and its last context position (float32 CPU)."""
    from ..model import run_with_cache
    co = run_with_cache(mb, ids_tensor(mb, v), [site])
    return co.resid[site.key][0, -1].detach().float().cpu()


class Substages:
    """Per-substage error capture and timing for a stage runner.

    ``run(name, fn, default)`` calls ``fn()``; an exception is recorded under ``name``
    (``errors[name]`` = repr, ``tracebacks[name]`` = full traceback, a warning is emitted)
    and ``default`` is returned, so the remaining substages of the stage still run and
    the manifest lists what failed (``report()``).  ``seconds[name]`` holds the runtime.
    """

    def __init__(self, stage: str, verbose: bool = True):
        self.stage = str(stage)
        self.verbose = bool(verbose)
        self.errors: dict[str, str] = {}
        self.tracebacks: dict[str, str] = {}
        self.seconds: dict[str, float] = {}

    def run(self, name: str, fn, default=None):
        t0 = time.perf_counter()
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001 - recorded, the stage goes on
            self.errors[name] = repr(e)
            self.tracebacks[name] = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            warnings.warn(f"stage {self.stage} substage {name!r} failed: {e!r}")
            out = default() if callable(default) else default
        self.seconds[name] = time.perf_counter() - t0
        if self.verbose:
            status = "FAILED" if name in self.errors else "ok"
            print(f"[lsne]   {self.stage}/{name} {status} in {self.seconds[name]:.1f} s", flush=True)
        return out

    @property
    def ok(self) -> bool:
        return not self.errors

    def report(self) -> dict:
        return dict(errors=dict(self.errors), seconds={k: round(v, 3) for k, v in self.seconds.items()},
                    n_failed=len(self.errors))

    def write_tracebacks(self, run_dir) -> Optional[Path]:
        if not self.tracebacks:
            return None
        d = Path(os.fspath(run_dir))
        d.mkdir(parents=True, exist_ok=True)
        p = d / "substage_errors.txt"
        p.write_text("\n\n".join(f"=== {k} ===\n{v}" for k, v in self.tracebacks.items()), encoding="utf-8")
        return p


# --------------------------------------------------------------------------------------
# resumable stage runs (stages C and D)
# --------------------------------------------------------------------------------------
def resume_signature(**parts) -> str:
    """Hash of the inputs of a resumable stage run (JSON of ``parts``, sorted keys)."""
    payload = json.dumps(to_jsonable(parts), sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    full = [f for f in frames if len(f)]
    if not full:
        return frames[0]
    if len(full) == 1:
        return full[0].reset_index(drop=True)
    return pd.concat(full, ignore_index=True)


def concat_results(parts: Sequence):
    """Concatenate chunk results of the same shape: DataFrames, tuples of DataFrames or
    dicts of DataFrames (row order = chunk order)."""
    first = parts[0]
    if isinstance(first, pd.DataFrame):
        return _concat_frames(parts)
    if isinstance(first, tuple):
        return tuple(_concat_frames([p[i] for p in parts]) for i in range(len(first)))
    if isinstance(first, dict):
        return {k: _concat_frames([p[k] for p in parts]) for k in first}
    raise TypeError(f"cannot concatenate chunk results of type {type(first)}")


class Resume:
    """Checkpoints of one stage run in ``<run_dir>/_resume``: an interrupted or failed run
    (session limit, disconnect, an exception after hours of work) continues where it stopped
    when the stage is started again with the same inputs.

    Objects are pickled (exact dtypes; not CSV, so ``merge`` and the table loaders never see
    them), each written to a temporary file and moved into place with ``os.replace``.
    ``state.json`` holds the :func:`resume_signature` of the run's inputs; another signature
    discards the old checkpoints.  :meth:`clear` removes the directory once the stage
    manifest is written.  ``enabled=False`` computes everything and stores nothing."""

    DIRNAME = "_resume"

    def __init__(self, run_dir, signature: str, enabled: bool = True):
        self.dir = Path(os.fspath(run_dir)) / self.DIRNAME
        self.signature = str(signature)
        self.enabled = bool(enabled)
        self.loaded: list[str] = []
        if not self.enabled:
            return
        state = self.dir / "state.json"
        try:
            old = json.loads(state.read_text(encoding="utf-8")).get("signature") if state.exists() else None
        except Exception:  # noqa: BLE001 - unreadable state: start over
            old = None
        if old != self.signature:
            if self.dir.exists():
                shutil.rmtree(self.dir, ignore_errors=True)
            self.dir.mkdir(parents=True, exist_ok=True)
            self._atomic(state, lambda tmp: tmp.write_text(json.dumps(dict(signature=self.signature)), encoding="utf-8"))

    @staticmethod
    def _atomic(path: Path, write) -> None:
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        try:
            write(tmp)
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def _path(self, name: str) -> Path:
        return self.dir / f"{name}.pkl"

    def get(self, name: str):
        """The checkpoint ``name`` or None (missing, disabled or unreadable)."""
        if not self.enabled:
            return None
        p = self._path(name)
        if not p.exists():
            return None
        try:
            obj = pd.read_pickle(p)
        except Exception as e:  # noqa: BLE001 - truncated file etc.: recompute
            warnings.warn(f"checkpoint {p} unreadable ({e!r}); recomputing it")
            return None
        self.loaded.append(name)
        return obj

    def put(self, name: str, obj) -> None:
        if self.enabled:
            self._atomic(self._path(name), lambda tmp: pd.to_pickle(obj, tmp, compression=None))

    def step(self, name: str, fn):
        """``fn()`` once: the result is checkpointed under ``name`` and loaded on a rerun."""
        out = self.get(name)
        if out is None:
            out = fn()
            self.put(name, out)
        return out

    def chunked(self, name: str, items: Sequence, fn, chunk: int = RESUME_CHUNK):
        """``fn(items[i:i + chunk])`` for consecutive chunks, each checkpointed as one
        :meth:`step`; the results (DataFrame / tuple / dict of DataFrames) concatenated in
        item order (:func:`concat_results`)."""
        items = list(items)
        if not items:
            return fn(items)
        chunk = max(1, int(chunk))
        parts = [self.step(f"{name}_{k:05d}", lambda i=i: fn(items[i:i + chunk]))
                 for k, i in enumerate(range(0, len(items), chunk))]
        return concat_results(parts)

    def clear(self) -> None:
        if self.enabled:
            shutil.rmtree(self.dir, ignore_errors=True)


def limited_core(core: Sequence, max_per_cell: Optional[int], seed: int) -> list:
    """At most ``max_per_cell`` standard variants per (K, C) cell of ``core``: a seeded subset
    of the formulations, each kept with all its core conditions (so mixed and monolingual
    variants stay paired); the original-context variants are always kept, the order of
    ``core`` is preserved.  ``None`` returns ``core`` unchanged."""
    core = list(core)
    if max_per_cell is None:
        return core
    groups = sorted({group_key(v) for v in core if not is_original(v)})
    m = max(0, int(max_per_cell))
    if len(groups) > m:
        rng = np.random.default_rng(int(seed))
        groups = [groups[int(i)] for i in sorted(rng.choice(len(groups), size=m, replace=False))]
    keep = set(groups)
    return [v for v in core if is_original(v) or group_key(v) in keep]


def synthetic_sae(mb: ModelBundle, n_latents: int = 64, k: int = 8, seed: int = 0):
    """Random TopK SAE (``TopKSAE.from_tensors``) of the model width - a *placeholder*
    for code paths that need an SAE when no published one exists (smoke runs); results
    computed with it are function tests, not findings (rows carry ``sae_source``)."""
    from ..sae import TopKSAE
    g = torch.Generator().manual_seed(int(seed))
    d = mb.d_model
    F = int(n_latents)
    W_enc = torch.randn(F, d, generator=g) / float(np.sqrt(d))
    W_dec = torch.randn(F, d, generator=g) / float(np.sqrt(d))
    return TopKSAE.from_tensors(W_enc, torch.zeros(F), W_dec, torch.zeros(d), k=int(min(k, F)))


def get_sae(mb: ModelBundle, bank: Optional[DirectionBank], limits: Limits, kw: dict) -> tuple[object, str]:
    """``(sae, source)``: ``kw['sae']`` (``"given"``), the bank's SAE (``"hub:<repo>/<hookpoint>"``),
    a synthetic placeholder when ``limits.synthetic_sae_if_missing`` (``"synthetic_random"``),
    else ``(None, "none")``."""
    if kw.get("sae") is not None:
        return kw["sae"], "given"
    if bank is not None and bank.sae is not None:
        info = bank.meta.get("sae", {}) if isinstance(bank.meta, dict) else {}
        return bank.sae, f"hub:{info.get('repo', '?')}/{info.get('hookpoint', '?')}"
    if limits.synthetic_sae_if_missing:
        return synthetic_sae(mb, n_latents=int(limits.synthetic_sae_latents)), "synthetic_random"
    return None, "none"


def stage_run_dir(cfg: RunConfig, stage: str) -> Optional[Path]:
    """Run directory of ``stage`` for the same model under ``cfg.out_root`` (None if absent)."""
    try:
        rd = RunDir(cfg.out_root, stage, cfg.model, mkdir=False)
    except Exception:  # noqa: BLE001
        return None
    return rd.path if rd.path.is_dir() else None


def load_stage_table(cfg: RunConfig, stage: str, name: str) -> Optional[pd.DataFrame]:
    """``<out_root>/<stage>/<model>@<rev>/<name>.csv`` of an earlier stage, or None."""
    from ..io import load_df
    d = stage_run_dir(cfg, stage)
    if d is None:
        return None
    try:
        return load_df(d, name)
    except FileNotFoundError:
        return None


def family_text_pairs(fams: Sequence, max_per_lang: Optional[int] = None) -> tuple[list[str], list[str], list[str]]:
    """Paired English / German unit sentences of standard families (``subj pred adj.`` in
    both languages, unit by unit) plus the family tails as neutral sentences - the real
    translation pairs of the stimulus bank, used by stage L as paired source/target
    items.  Returns ``(texts_E, texts_D, texts_neutral)`` (aligned E/D lists)."""
    E, D, N = [], [], []
    for f in fams:
        if getattr(f, "kind", "standard") != "standard":
            continue
        for u in f.units:
            e = f"{u.E['subj']} {u.E['pred']} {u.E['adj']}."
            d = f"{u.D['subj']} {u.D['pred']} {u.D['adj']}."
            if e not in E:
                E.append(e)
                D.append(d)
        tail = str(f.tail_E).strip()
        if tail and tail not in N:
            N.append(tail)
    if max_per_lang is not None:
        E, D, N = E[: int(max_per_lang)], D[: int(max_per_lang)], N[: int(max_per_lang)]
    return E, D, N


def boot_row(df: pd.DataFrame, value_col: str, n_boot: int, seed: int, cluster_col: str = "family", **keys) -> dict:
    """One tidy row of a family cluster bootstrap (``stat_level='cluster'`` when the
    installed :func:`lsne.stats.cluster_bootstrap` supports it, else the row estimator):
    ``keys`` + ``mean, ci_low, ci_high, se, n_clusters, n_rows``."""
    from ..stats import cluster_bootstrap
    try:
        r = cluster_bootstrap(df, value_col, cluster_col, n_boot=int(n_boot), seed=int(seed), stat_level="cluster")
    except TypeError:            # older signature without stat_level
        r = cluster_bootstrap(df, value_col, cluster_col, n_boot=int(n_boot), seed=int(seed))
    return dict(keys, mean=float(r["mean"]), ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]),
                se=float(r.get("se", float("nan"))), n_clusters=int(r["n_clusters"]), n_rows=int(r.get("n_rows", len(df))))


__all__ = [
    "PRIORITY", "CORE_CONDITIONS", "MARKER_WORDS", "PATCH_CLASSES", "PREREG_PATCH_CLASS", "ID_COLS", "Limits",
    "smoke_config",
    "get_limits", "cache_root", "cache_dir", "resolve_families_dir", "offline", "dossier_site",
    "depth_layer", "depth_site", "variant_kind", "is_original", "ids_tensor", "tail_positions",
    "unit_last_position", "variant_row", "dir_family", "progress", "select_families", "priority_rank",
    "group_key", "apply_variant_limit", "build_family_variants", "load_variants", "core_subset",
    "reference_of", "Original", "load_original", "DirectionBank", "bank_signature", "unigram_counts",
    "direction_bank", "save_bank", "load_bank", "get_variants", "get_original", "all_variants",
    "get_bank", "marker_lexicon", "finish_stage", "write_error", "pearson",
    "DOMAIN_GROUPS", "DEFAULT_DOMAIN_GROUP", "SMOKE_GL", "family_domain", "families_in_group", "k_subset",
    "prereg_layer_grid", "grid_sites", "last_state", "Substages", "synthetic_sae", "get_sae", "stage_run_dir",
    "load_stage_table", "family_text_pairs", "boot_row", "UnigramFetchError", "LIMITED_GEN_MAX_CORE", "RESUME_CHUNK",
    "Resume", "resume_signature", "concat_results", "limited_core",
]
