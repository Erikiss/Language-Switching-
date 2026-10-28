"""Stage D control run: the language at the injection site (PREREGISTRATION section 6, amendment 11).

In C1 the ``depth`` injection of D1 (:mod:`lsne.experiments.stage_d_noneq`) hits the same token for
K = 0 and K > 0, the period "." that ends unit 9.  For K > 0 it closes a German clause, and the
final D -> E switch follows right after it, so the contrast K > 0 vs K = 0 may measure the language
at the injection site instead of the switching in the text.  The control asks whether the effect
on lambda survives at sites where a "." closes an English clause.

    python -m lsne.cli run --stage D --d-control token_language \\
        --d-main-run <LSNE>/runs/D/<model>@<revision> --out <LSNE>/runs_ctl [--model .. --revision ..]

:func:`lsne.experiments.stage_d_noneq.run` dispatches here when ``kw['d_control']`` is set; the run
directory is the usual ``<out_root>/D/<model>@<revision>``, so the control lives under its own
root (``runs_ctl``) and the evaluation keys it as ``D_ctl`` (manifest ``control``).

Inputs taken from the main stage-D run (``kw['d_main_run']``), never refitted:

* its manifest's ``lattice_directions`` and ``doses`` (both in their order) and
  ``limits.damage_batch_size``: every (variant, site) runs the (direction, dose) pairs of stage D
  in stage D's order and chunks, so lambda at the primary site replicates the main run;
* its depth bank ``directions_depth.npz`` (:func:`~lsne.experiments.common.load_bank`) with the
  vectors exactly as stored (:func:`load_main_bank`).

Variants: the regular builder (:func:`~lsne.experiments.common.get_variants`, the run's ``cfg``
on its test formulations) restricted to the standard variants of the test split (F01-F12 x {A, B})
at K = 0 and K = 2 / 4 / 8 x C1 / C2 / C3 (no ``seg``, no original-context variants), plus per test
formulation the variant ``u9D``: only unit 9 German (unit mask :data:`U9D_UNIT_MASK` realised at C1,
two switches), checked against the formulation's K = 0 variant like every variant
(PREREGISTRATION section 4; an excluded one is recorded in ``damage_ctl_checks``, not run).  A run
narrowed by cfg filters (``--families`` / ``--limit-families`` below the main run's selection,
``--limit-variants``, K / C levels) records them in the manifest (``design_filters``,
:func:`design_filters`); the evaluation never prefers it over a run of the whole design and judges
by its audits whether it covers the rows of the primary endpoint.

Sites (all at the depth site, block input of block ``n_layers // 3``; defined by the unit, never
by the language as :func:`lsne.stimuli.last_switched_position` is):

=========  ==========================  ======================================================
inject     position                    cells
=========  ==========================  ======================================================
``u9``     last token of unit 9        K = 0 and C1 (the primary site, replication audit),
                                       C2 / C3 (control 1), ``u9D`` (control 3)
``u10``    last token of unit 10       K = 0 and C1 (control 2; one token before the tail)
=========  ==========================  ======================================================

Tables:

``damage_ctl``            the ``damage`` table of :func:`~lsne.experiments.stage_d_noneq.damage_tables`
                          at the control sites plus ``inject_token_id`` (token id at
                          ``inject_pos``) and ``site_gap`` (``tail_start - inject_pos``)
``damage_ctl_profile``    its ``damage_profile`` table
``damage_ctl_checks``     one row per control variant built or excluded (``tag, family,
                          formulation, K, C, step, problem, excluded``)
``damage_ctl_contrasts``  descriptive: :func:`~lsne.experiments.stage_d_noneq.damage_contrasts` at
                          ``u9`` and at ``u10`` (the verdicts are made by ``lsne.prereg_eval``,
                          rows ``P.ctl*``)

Safety: the run refuses (:class:`ControlRunError`, raised before anything is written) to work in
the main run's directory or inside it, under a root that is a ``runs`` folder, lies in the main
run's root or holds the main run, or in any directory that holds a stage-D run other than this
control; to start without an ok main stage-D run of the same model, revision and dtype with its
depth bank; and to start without a control variant.  Its checkpoints (``_resume``, own signature)
stay in its own directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .. import stimuli
from ..config import HookSite, RunConfig
from ..dynamics import DAMAGE_BATCH_SIZE
from ..io import MANIFEST_NAME, read_manifest
from ..model import ModelBundle
from . import common as C
from . import stage_d_noneq as SD

#: The control of the injection-site language (``kw['d_control']`` / ``--d-control``).
CONTROL = "token_language"
#: Valid values of ``kw['d_control']``.
CONTROLS = (CONTROL,)
#: The amendment that fixes the control (PREREGISTRATION section 6, item 11).
AMENDMENT = "6.11"
#: Bump when the meaning of a checkpointed control result changes (invalidates ``_resume``).
RESUME_VERSION = 1
#: Name of the main run's depth bank (``<main run>/directions_depth.npz`` + ``_meta.json``).
BANK_NAME = "directions_depth"
#: K and C levels of the regular control variants (K = 0 is built with C1 only).
CTL_K = (0, 2, 4, 8)
CTL_C = ("C1", "C2", "C3")
#: Control 3: only unit 9 German, realised at C1 (two switches) and labelled ``u9D``.
U9D_UNIT_MASK = "EEEEEEEEDE"
U9D_K = 2
U9D_C = "u9D"
#: Injection sites of the control (``inject`` column) -> unit whose last token is injected.
SITE_UNITS = {"u9": 9, "u10": 10}
INJECT_NAMES = tuple(SITE_UNITS)
#: The site rule (manifest ``sites``; part of the resume signature).
SITE_RULES = {
    "u9": ("last token of unit 9, lsne.experiments.common.unit_last_position(v, 9), at the depth site; "
           "every control variant: K=0 and C1 (primary site), C2/C3 (control 1), u9D (control 3)"),
    "u10": ("last token of unit 10, lsne.experiments.common.unit_last_position(v, 10), at the depth site; "
            "K=0 and C1 (control 2)"),
}
#: Columns of ``damage_ctl_checks`` (as the checks of :func:`~lsne.experiments.common.load_variants`).
CHECK_COLS = ("tag", "family", "formulation", "K", "C", "step", "problem", "excluded")
#: Columns ``damage_ctl`` adds to the D1 damage table (placed after ``inject_klass``).
EXTRA_COLS = ("inject_token_id", "site_gap")


class ControlRunError(ValueError):
    """A stage-D control run that must not start.  Raised before anything is written into the
    run directory, which may hold a main stage-D run (the CLI then leaves it untouched)."""


# --------------------------------------------------------------------------------------
# guards and inputs
# --------------------------------------------------------------------------------------
def check_control(control) -> str:
    """``control`` if it is a known control (:data:`CONTROLS`), else :class:`ControlRunError`."""
    if control not in CONTROLS:
        raise ControlRunError(f"unknown stage-D control {control!r}; valid: {', '.join(CONTROLS)}")
    return str(control)


def _same_dir(a: Path, b: Path) -> bool:
    try:
        if a.exists() and b.exists():
            return os.path.samefile(a, b)
    except OSError:
        pass
    return a.resolve() == b.resolve()


def _inside(path: Path, root: Path) -> bool:
    """``path`` is ``root`` or lies below it (resolved paths)."""
    p, r = path.resolve(), root.resolve()
    return p == r or r in p.parents


def check_run_dir(run_dir, main_dir, out_root=None) -> None:
    """Refuse (:class:`ControlRunError`) a control-run directory that is the main run's directory,
    lies inside it or holds a stage-D run other than this control: a ``manifest.json`` without
    ``control == "token_language"`` (any status; never overwrite a main run), or the depth bank
    ``directions_depth.npz`` (a main stage-D run writes it before its checkpoints and its manifest,
    so an interrupted main run is recognised too).  With ``out_root`` (``--out``, the root of
    ``run_dir``) also a root that is a ``runs`` folder (a main-run root, of this account or
    another), lies inside the main run's root or holds the main run.  Reads only."""
    rd, md = Path(os.fspath(run_dir)), Path(os.fspath(main_dir))
    if _same_dir(rd, md):
        raise ControlRunError(f"the control run directory {rd} is the main stage-D run directory; "
                              f"give the control run its own root (--out .../runs_ctl)")
    if _inside(rd, md):
        raise ControlRunError(f"the control run directory {rd} lies inside the main stage-D run directory {md}; "
                              f"give the control run its own root (--out .../runs_ctl)")
    if out_root is not None:
        root = Path(os.fspath(out_root))
        main_root = md.parent.parent if md.parent.name == "D" else md          # <main root>/D/<model>@<revision>
        if root.name == "runs" or _inside(root, main_root) or _inside(md, root):
            raise ControlRunError(f"the control run root {root} is a runs folder, lies inside the root of the main stage-D "
                                  f"run {md} or holds it; give the control run its own root (--out .../runs_ctl)")
    if (rd / MANIFEST_NAME).exists():
        try:
            m = read_manifest(rd)
        except (OSError, ValueError) as e:
            raise ControlRunError(f"{rd / MANIFEST_NAME} is unreadable ({e!r}); it may belong to a main "
                                  f"stage-D run, which a control run never overwrites") from e
        if not (m.get("stage") == "D" and m.get("control") == CONTROL):
            raise ControlRunError(f"{rd} holds the manifest of a stage-{m.get('stage', '?')} run (status "
                                  f"{m.get('status', '?')!r}) that is not a {CONTROL} control run; a control "
                                  f"run never overwrites a main run")
    if (rd / f"{BANK_NAME}.npz").exists():
        raise ControlRunError(f"{rd} holds {BANK_NAME}.npz of a main stage-D run (possibly interrupted, with "
                              f"its checkpoints); a control run never writes there")


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_main_bank(main_dir) -> C.DirectionBank:
    """The main run's depth bank (:func:`~lsne.experiments.common.load_bank`, no refit) with every
    vector exactly as stored.  ``load_bank`` passes the vectors through
    :class:`~lsne.directions.Direction`, which normalises them again: a float32 unit vector whose
    computed norm is not exactly 1 changes in its last bits, and at the smallest dose that moves
    lambda beyond the 1e-3 replication bound (measured for a 1-ulp change at eps 0.01: up to
    5e-3 on pythia-160m, 0.06 for ``ent`` on pythia-14m).  The main run injected the stored
    vectors themselves (:func:`~lsne.experiments.common.save_bank` writes the float32 vectors
    unchanged), so these replicate its lambda exactly."""
    md = Path(os.fspath(main_dir))
    bank = C.load_bank(md, BANK_NAME)
    with np.load(md / f"{BANK_NAME}.npz") as z:
        for name, d in bank.dirs.items():
            d.vec = torch.from_numpy(np.array(z[name], dtype=np.float32).reshape(-1))
    return bank


def main_run_inputs(cfg: RunConfig, main_dir) -> dict:
    """The inputs of the control from the main stage-D run in ``main_dir``: ``manifest``,
    ``bank`` (``directions_depth``), ``bank_sha256``, ``lattice_dirs`` and ``doses`` (manifest
    order) and ``batch_size`` (``limits.damage_batch_size``).  :class:`ControlRunError` when the
    directory holds no ok stage-D run (or a control run), when its model name, revision or dtype
    differ from ``cfg.model``, or when one of these inputs or the bank files are missing."""
    md = Path(os.fspath(main_dir))
    if not (md / MANIFEST_NAME).exists():
        raise ControlRunError(f"no {MANIFEST_NAME} in the main stage-D run directory {md}")
    try:
        m = read_manifest(md)
    except (OSError, ValueError) as e:
        raise ControlRunError(f"the manifest of the main stage-D run {md} is unreadable ({e!r})") from e
    if m.get("stage") != "D" or m.get("status") != "ok":
        raise ControlRunError(f"{md} holds no ok stage-D run (stage {m.get('stage')!r}, status {m.get('status')!r})")
    if m.get("control"):
        raise ControlRunError(f"{md} is a {m.get('control')} control run, not the main stage-D run")
    model = m.get("model") if isinstance(m.get("model"), dict) else {}
    for key in ("name", "revision", "dtype"):
        want, got = str(getattr(cfg.model, key)), model.get(key)
        if str(got) != want:
            raise ControlRunError(f"the main stage-D run {md} has model {key} {got!r}, this run {want!r}")
    lim = m.get("limits") if isinstance(m.get("limits"), dict) else {}
    try:
        lattice = [str(n) for n in m.get("lattice_directions") or []]
        doses = [float(e) for e in m.get("doses") or []]
        batch = int(lim["damage_batch_size"])
    except (KeyError, TypeError, ValueError) as e:
        raise ControlRunError(f"the manifest of the main stage-D run {md} has no usable lattice_directions / "
                              f"doses / limits.damage_batch_size ({e!r})") from e
    if not lattice or not doses:
        raise ControlRunError(f"the manifest of the main stage-D run {md} lists no lattice_directions / doses")
    files = [md / f"{BANK_NAME}.npz", md / f"{BANK_NAME}_meta.json"]
    missing = [p.name for p in files if not p.exists()]
    if missing:
        raise ControlRunError(f"the main stage-D run {md} has no depth bank ({', '.join(missing)} missing)")
    return dict(dir=md, manifest=m, bank=load_main_bank(md), bank_sha256=file_sha256(files[0]),
                lattice_dirs=lattice, doses=doses, batch_size=batch)


def mark_error_manifest(run_dir, control: str = CONTROL) -> bool:
    """Add the control marks (``control``, ``amendment``) to the error manifest that the CLI writes
    for a failed control run (:func:`~lsne.experiments.common.write_error`), so that a restart,
    which resumes from the checkpoints, is not refused as a main stage-D run
    (:func:`check_run_dir`).  Returns True when the manifest was rewritten."""
    d = Path(os.fspath(run_dir))
    try:
        m = read_manifest(d)
    except (OSError, ValueError):
        return False
    if m.get("stage") != "D" or m.get("status") != "error":
        return False
    m.update(control=check_control(control), amendment=AMENDMENT)
    (d / MANIFEST_NAME).write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


# --------------------------------------------------------------------------------------
# variants and sites
# --------------------------------------------------------------------------------------
def in_control_set(family, formulation, K, Cc) -> bool:
    """A regular variant of the control: test split, K = 0 (C1) or K = 2/4/8 x C1/C2/C3."""
    K, Cc = int(K), str(Cc)
    cell = (K == 0 and Cc == "C1") or (K in CTL_K[1:] and Cc in CTL_C)
    return cell and C.prereg_split(family, formulation) == "test"


def control_formulations(cfg: RunConfig) -> list[str]:
    """Tags (``F01A``) of the test formulations (F01-F12 x {A, B}, kind standard) among the
    families that ``cfg.families`` selects, before ``cfg.limit_families``."""
    return [f.tag for f in C.select_families(replace(cfg, limit_families=None))
            if f.kind == "standard" and C.prereg_split(f.family_id, f.formulation) == "test"]


def design_filters(cfg: RunConfig, main_cfg: Optional[dict] = None) -> list[str]:
    """The settings of ``cfg`` that narrow the control design, every test formulation the main run
    selects x every cell (``main_cfg``: the ``cfg`` of the main run's manifest; None: the whole
    stimulus bank) - manifest ``design_filters``, [] for the whole design: ``families`` when
    ``cfg.families`` / ``cfg.limit_families`` drop a test formulation the main run has,
    ``limit_variants``, and ``k_levels`` / ``c_levels`` without a level of the design.  The
    evaluation never prefers such a run over one of the whole design
    (``lsne.prereg_eval.select_runs``); whether it covers the rows of the primary endpoint is
    decided by its audits there."""
    mc = main_cfg if isinstance(main_cfg, dict) else {}

    def formulations(c: RunConfig) -> set:          # the test formulations after cfg.families and cfg.limit_families
        sel = control_formulations(c)
        return {f.tag for f in C.select_families(replace(c, families=sel))} if sel else set()

    main = replace(cfg, families=mc.get("families") or None, limit_families=mc.get("limit_families") or None)
    on = dict(families=bool(formulations(main) - formulations(cfg)), limit_variants=bool(cfg.limit_variants),
              k_levels=not set(CTL_K) <= {int(k) for k in cfg.k_levels},
              c_levels=not set(CTL_C) <= {str(c) for c in cfg.c_levels})
    return [name for name, narrowed in on.items() if narrowed]


def u9d_variant(fam: stimuli.Family, tokenizer) -> stimuli.Variant:
    """Control 3: the variant of ``fam`` with only unit 9 German (:data:`U9D_UNIT_MASK` realised at
    C1, :data:`U9D_K` switches), tagged ``<family><formulation>:K2:u9D``."""
    mask = stimuli.segment_masks_from_unit_mask(U9D_UNIT_MASK, "C1")
    if stimuli.count_switches(mask) != U9D_K:          # pragma: no cover - fixed by the mask
        raise ValueError(f"unit mask {U9D_UNIT_MASK} does not have {U9D_K} switches")
    return stimuli.make_variant(fam, mask, U9D_K, U9D_C, tokenizer)


def control_variants(cfg: RunConfig, mb: ModelBundle, ctx: Optional[dict] = None) -> tuple[list, pd.DataFrame]:
    """``(variants, checks)`` of the control run.

    ``variants``: the regular variants (:func:`~lsne.experiments.common.get_variants`; the K = 0
    checks apply as in every stage) of the control set (:func:`in_control_set`), each test
    formulation followed by its ``u9D`` variant (:func:`u9d_variant`, kept only when it passes
    :func:`lsne.stimuli.check_variant` against the formulation's K = 0 variant), then
    ``cfg.limit_variants`` as in the regular builder.  The regular builder runs on the test
    formulations of ``cfg.families`` only (:func:`control_formulations`), so that its
    ``cfg.limit_families`` / ``cfg.limit_variants`` act on them (in the stimulus bank DEV01-DEV04
    come first: a limit applied before the split would leave nothing) and select the same C1
    variants as a main run limited alike.  ``checks`` (``damage_ctl_checks``,
    :data:`CHECK_COLS`): the regular check rows of the control set (as the regular builder's,
    also of variants its limit drops) and one row per ``u9D`` variant built or excluded.
    :class:`ControlRunError` when no test formulation is selected or no variant is left."""
    tags = control_formulations(cfg)
    if not tags:
        raise ControlRunError(f"no test formulation (F01-F12 x A, B) among the selected families "
                              f"({', '.join(map(str, cfg.families)) if cfg.families else 'all'}): no control variant")
    cfg = replace(cfg, families=tags)
    variants, checks = C.get_variants(cfg, mb, ctx)
    keep = [v for v in variants if not C.is_original(v) and in_control_set(v.family_id, v.formulation, v.K, v.C)]
    rows: list[dict] = []
    if len(checks):
        sel = [in_control_set(f, fo, k, c) for f, fo, k, c in
               zip(checks["family"], checks["formulation"], checks["K"], checks["C"])]
        rows = checks.loc[sel, list(CHECK_COLS)].to_dict("records")
    groups: dict[tuple[str, str], list] = {}
    for v in keep:
        groups.setdefault(C.group_key(v), []).append(v)
    refs = C.reference_of(keep)
    for fam in C.select_families(cfg):
        key = (str(fam.family_id), str(fam.formulation))
        if fam.kind != "standard" or key not in groups:
            continue                     # not a test formulation of this run (split, cfg.families, limits)
        ident = dict(tag=f"{fam.tag}:K{U9D_K}:{U9D_C}", family=fam.family_id, formulation=fam.formulation,
                     K=U9D_K, C=U9D_C)
        ref = refs.get(key)
        if ref is None:
            rows.append(dict(ident, step="check", problem="no K=0 reference variant", excluded=True))
            continue
        try:
            v = u9d_variant(fam, mb.tokenizer)
        except Exception as e:  # noqa: BLE001 - record and continue (as the regular builder)
            rows.append(dict(ident, step="build", problem=str(e), excluded=True))
            continue
        probs = stimuli.check_variant(v, ref)
        rows.append(dict(ident, step="check", problem="; ".join(probs), excluded=bool(probs)))
        if not probs:
            groups[key].append(v)
    out = C.apply_variant_limit([v for vs in groups.values() for v in vs], cfg.limit_variants)
    if not out:
        raise ControlRunError(f"no control variant left for the test formulations {', '.join(tags[:4])}"
                              f"{' ...' if len(tags) > 4 else ''} (K levels {cfg.k_levels}, C levels {cfg.c_levels}, "
                              f"limits {cfg.limit_families} families / {cfg.limit_variants} variants, section-4 exclusions)")
    return out, pd.DataFrame(rows, columns=list(CHECK_COLS))


def site_names(v) -> tuple[str, ...]:
    """Control sites of a variant: ``u9`` everywhere, ``u10`` also at K = 0 and C1."""
    return ("u9", "u10") if int(v.K) == 0 or str(v.C) == "C1" else ("u9",)


def control_injections(mb: ModelBundle, v) -> list[tuple[str, HookSite, int]]:
    """``(name, site, position)`` of the control injections of a variant (the ``injections_fn`` of
    :func:`~lsne.experiments.stage_d_noneq.damage_tables`): the depth site at the last token of
    unit 9 / 10 (:func:`~lsne.experiments.common.unit_last_position`) per :func:`site_names`."""
    site = C.depth_site(mb)
    return [(name, site, C.unit_last_position(v, SITE_UNITS[name])) for name in site_names(v)]


def control_cells(variants) -> dict[str, list[tuple[int, str]]]:
    """Per site the (K, C) cells of ``variants`` injected there (manifest ``cells``)."""
    return {name: sorted({(int(v.K), str(v.C)) for v in variants if name in site_names(v)}) for name in INJECT_NAMES}


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def control_damage_tables(mb: ModelBundle, variants, bank: C.DirectionBank, dir_names, doses, enabled=True,
                          batch_size: int = DAMAGE_BATCH_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(damage_ctl, damage_ctl_profile)``: :func:`~lsne.experiments.stage_d_noneq.damage_tables`
    at the control sites (:func:`control_injections`) with ``bank`` (the main run's depth bank),
    plus ``inject_token_id`` and ``site_gap`` (:data:`EXTRA_COLS`) after ``inject_klass``."""
    damage, profile = SD.damage_tables(mb, variants, {bank.site.key: bank}, dir_names, doses, enabled=enabled,
                                       batch_size=batch_size, injections_fn=control_injections)
    by_tag = {v.tag: v for v in variants}
    pos = damage["inject_pos"].astype(int).tolist()
    extra = dict(inject_token_id=[int(by_tag[t].ids[p]) for t, p in zip(damage["tag"], pos)],
                 site_gap=[int(ts) - p for ts, p in zip(damage["tail_start"].astype(int).tolist(), pos)])
    at = list(damage.columns).index("inject_klass") + 1
    for i, name in enumerate(EXTRA_COLS):
        damage.insert(at + i, name, pd.Series(extra[name], index=damage.index, dtype="int64"))
    return damage, profile


def platform_info(mb: ModelBundle, main_manifest: dict) -> dict:
    """Manifest ``replication_platform``: torch / CUDA build and device of this run next to the main
    run's torch version and OS (``main_platform``; this run's OS is the manifest's own ``platform``).
    The replication of the primary site is exact on the same platform; another build changes
    round-off, and lambda at the smallest dose moves by up to ~5e-3 under round-off-level input
    changes (pythia-160m, measured on CPU), which is above the 1e-3 bound of the audit."""
    gpu = ""
    if mb.device.type == "cuda" and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(mb.device)
    return dict(torch=str(torch.__version__), cuda=str(torch.version.cuda or ""), device=str(mb.device), gpu=gpu,
                main_torch=str(main_manifest.get("torch_version", "")), main_platform=str(main_manifest.get("platform", "")))


def control_contrasts(damage: pd.DataFrame) -> pd.DataFrame:
    """``damage_ctl_contrasts``: :func:`~lsne.experiments.stage_d_noneq.damage_contrasts` at every
    control site, concatenated (descriptive; ``position_deviation`` there means C != C1)."""
    parts = [p for p in (SD.damage_contrasts(damage, inject=name) for name in INJECT_NAMES) if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=SD.DAMAGE_CONTRAST_COLS)


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """The control run (see the module notes): ``kw['d_control']`` must be ``"token_language"``,
    ``kw['d_main_run']`` the main stage-D run directory.  Every guard (:func:`check_control`,
    :func:`check_run_dir`, :func:`main_run_inputs`, the bank's site) raises
    :class:`ControlRunError` before anything is written.  Resumable like stage D in
    ``<run_dir>/_resume`` (own signature, chunks of
    :data:`~lsne.experiments.common.RESUME_CHUNK` variants; ``kw['resume']=False`` disables it)."""
    t0 = time.perf_counter()
    control = check_control(kw.get("d_control"))
    main_dir = kw.get("d_main_run")
    if not main_dir:
        raise ControlRunError("the control run needs the main stage-D run directory (kw['d_main_run'], --d-main-run)")
    check_run_dir(run_dir, main_dir, cfg.out_root)
    main = main_run_inputs(cfg, main_dir)
    dsite = C.depth_site(mb)
    dbank = main["bank"]
    if dbank.site != dsite:
        raise ControlRunError(f"the depth bank of the main run sits at {dbank.site.key}, this model's depth site "
                              f"is {dsite.key}")
    lattice_dirs, doses, batch = main["lattice_dirs"], main["doses"], main["batch_size"]
    limits = replace(C.get_limits(cfg, kw), damage_batch_size=batch)     # the manifest records the layout used
    variants, checks = control_variants(cfg, mb, kw.get("ctx"))
    sig = C.resume_signature(stage="D_ctl", version=RESUME_VERSION, tags=[v.tag for v in variants],
                             model=mb.spec.tag(), dtype=mb.spec.dtype, depth_site=dsite.key, bank=main["bank_sha256"],
                             lattice_dirs=lattice_dirs, doses=doses, batch_size=batch, sites=SITE_RULES,
                             chunk=C.RESUME_CHUNK)
    res = C.Resume(run_dir, sig, enabled=bool(kw.get("resume", True)))
    damage, profile = res.chunked("damage_ctl", variants, lambda vs: control_damage_tables(
        mb, vs, dbank, lattice_dirs, doses, enabled=limits.progress, batch_size=batch))
    tables = {"damage_ctl": damage, "damage_ctl_profile": profile, "damage_ctl_checks": checks,
              "damage_ctl_contrasts": control_contrasts(damage)}
    mm = main["manifest"]
    mmodel = mm.get("model") or {}
    out = C.finish_stage(run_dir, "D", cfg, mb, tables, limits=limits, t0=t0, control=control, amendment=AMENDMENT,
                         main_run=dict(path=str(main["dir"].resolve()), git_sha=mm.get("git_sha", ""),
                                       timestamp=mm.get("timestamp", ""), model=mmodel.get("name", ""),
                                       revision=mmodel.get("revision", "")),
                         bank=dict(file=f"{BANK_NAME}.npz", sha256=main["bank_sha256"], names=list(dbank.dirs)),
                         depth_site=dsite, lattice_directions=lattice_dirs, doses=doses, damage_batch_size=batch,
                         sites=SITE_RULES, cells=control_cells(variants), u9D_unit_mask=U9D_UNIT_MASK,
                         n_variants=len(variants), n_formulations=len({C.group_key(v) for v in variants}),
                         design_filters=design_filters(cfg, mm.get("cfg")),
                         n_excluded=int(checks["excluded"].astype(bool).sum()) if len(checks) else 0,
                         resumed_checkpoints=len(res.loaded), audit=C.audit_summary(damage),
                         replication_platform=platform_info(mb, mm))
    res.clear()
    return out
