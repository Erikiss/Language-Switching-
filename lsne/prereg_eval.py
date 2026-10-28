"""Pre-registered evaluation of finished run directories (PREREGISTRATION.md sections 2-4).

    python -m lsne.prereg_eval --runs /content/drive/MyDrive/LSNE/runs \\
        [--runs /content/drive/MyDrive/LSNE_B/runs ...] --out DIR [--n-boot 10000] [--seed 0] \\
        [--model pythia-160m] [--revision step98000] [--frozen-dose EPS]

Computes the confirmatory endpoints from the tables the stages already wrote (parquet
preferred, csv otherwise): no model is loaded and no GPU stage is re-run, so a CPU runtime
is enough.  Every ``--runs`` root is searched for ``<stage>/<model>@<revision>/manifest.json``
with ``status: ok``; a parent folder such as ``MyDrive/LSNE`` contributes its ``runs``,
``runs_trace`` and ``runs_ctl`` children.  Several roots (the runs folders of two Google
accounts, one of them a Drive shortcut) are merged: when a stage has an ok run under more than
one root the newest manifest is used and the others are listed as superseded; stage L is split
into the stand-in run and the trace-source run (``manifest.source.kind == "trace"``), stage D
into the main run and the token-language control run (``manifest.control ==
"token_language"``, key ``D_ctl``, PREREGISTRATION 6.11), so a control run never replaces the
main run.  Only the stages in :data:`EVAL_STAGES` are opened: the scale sweep (stage F and its sub-runs
``F/<model>@<rev>/shard*/...``) is never touched, so a sweep that is still running is not
disturbed.

Confirmation set and statistic (PREREGISTRATION sections 1-2): test families F01-F12 x
formulations {A, B}; per family the mean of the per-formulation quantity, then the mean over
the families ("Familienmittel der Differenz"); 95 % percentile interval of a family cluster
bootstrap (:func:`lsne.stats.cluster_bootstrap` with ``stat_level='cluster'``, ``n_boot``
draws, both formulations of a family drawn together); two-sided bootstrap p (and a one-sided
p where a direction is pre-registered); Benjamini-Hochberg (:func:`lsne.stats.bh_correct`)
over the secondary group 1-6, 8 and 9 (when G0 passes), 10, 13 and 14, one test per sub-row
(the copied rows of 8, 9, 13 and 14 use the table's ``p_boot``; a table without it leaves
the row outside the group with its verdict capped at "open"); the grouping is written into the
report; ``bh_q_endpoint`` is the sensitivity grouping "Bonferroni within the endpoint, BH
across the endpoints".  Minimum effects (2: 0.05, 9: cos 0.3, 13: 0.05 nats) are judged by the
interval, never by the point estimate (:func:`min_effect_status`).

Endpoints (``id`` column)
-------------------------
``P``         primary (section 2) from stage D ``damage``: depth injection, kind standard,
              cell C1 only for K > 0 (the only cell whose injection position equals the
              pre-registered one, the last token of unit 9), K = 0 as reference; per family
              dlam(K) = mean lam(K, lang_dom) - mean lam(K0, lang_dom), the same for ``rand``
              averaged over every random direction present, DiD = dlam(lang_dom) - dlam(rand).
              Supported iff the CI of dlam(lang_dom, K=8) lies at or above the minimum effect
              (``ci_low >= 0.05``: the interval decides, never the point estimate, the rule of
              stage D's ``damage_boot`` and ``common.min_effect_verdict``), the DiD CI excludes 0
              (positive) and the rise in K is shown (``P.mono``: dlam(K_1) and every increment
              dlam(K_i) - dlam(K_i-1) have a CI above 0; PREREGISTRATION 6.10, reading (b) of
              28.10.2026; a CI below 0 refutes it, otherwise it is open; monotone point estimates are
              reported as reading (a)); "open" when a component is neither shown nor refuted.  Every
              pre-registered dose present is evaluated (``P@<eps>``); the main dose (``P``) is
              the DEV-frozen dose when one is recorded (``--frozen-dose`` or ``frozen_dose`` in
              the stage-D manifest).  C2 / C3 / seg rows are deviation rows
              (``P.dev.*``, ``not_interpretable``: injection position != the pre-registered one).
``P.ctl``     token-language control of P (PREREGISTRATION 6.11, fixed 28.10.2026 after viewing
              the data, before the control run; role ``primary_control``, outside the BH group)
              from ``damage_ctl`` of the control run (key ``D_ctl``): the statistic and rules of P
              at unit-defined sites that also hold "." but close an English clause, each against
              K = 0 at the same site in the same run.  Control 1 (``P.ctl1``): C2 and C3 pooled at
              the last token of unit 9; control 2 (``P.ctl2``): C1 at the last token of unit 10;
              K = 8 main, 2 and 4 reported (``.lang.K*`` / ``.did.K*``); control 3 (``P.ctl3``,
              only unit 9 German, ``u9D``) is reported only.  Per dose (suffix as P) the outcome is
              "Wechsel" (controls 1 and 2 supported), "lokal" (both have ci_high of dlam(lang_dom,
              K=8) < 0.05) or "offen".  The control can only weaken P (:func:`apply_token_control`):
              "lokal" makes a supported or open P ``not_interpretable``, "offen" or no valid
              control run caps it at "open".  Validity: ``audit.D1ctl`` (write audit),
              ``audit.D1ctl.rep`` (lam at the pre-registered site within 1e-3 of the main run),
              ``audit.D1ctl.site`` and ``audit.D1ctl.run`` (manifest marks, run newer than the
              amendment); a failed audit makes every ``P.ctl*`` row ``not_interpretable``.
``audit.*``   section 4 run validity: max / quantiles of ``audit_rel_err`` and the number of
              rows > 1e-4 in stage D ``damage`` (``audit.D1``) and stage C ``grid``
              (``audit.C``); a failed D1 audit makes the primary rows ``not_interpretable``.
``S1``-``S6`` stage C / D tables recomputed on the test split (S3 per pair and layer plus the
              exploratory nats-based excess ``S3x``; S6 per dose, paired per variant, pooled over
              every standard variant as stage C's pre-registered cell, one-sided in the State-2143
              sign +1; the reversed sign and the K = 0 cell are sensitivity rows).
``S7``-``S9`` stage G: the G0 gate (copied); 8 and 9 are ``not_interpretable`` unless it passes
              (their tables' ``split="test"`` rows when the run wrote them, :func:`boot_split`).
``S10``       stage H ``lambda_corr`` on the test split; a constant censored series (the
              boundary is never reached) enters as rho = 0, the as-run estimate (such starts
              dropped) is the sensitivity row ``*.as_run``.
``S11``-``S18`` copied from their tables, with the known caveats as deviation notes.

Outputs (``--out``): ``prereg_results.csv`` (one row per endpoint and sub-row, columns
:data:`RESULT_COLS` + :data:`EXTRA_COLS`), ``prereg_results.md`` (German headings) and
``prereg_runs.csv`` (every run directory found, used or not, :data:`RUN_COLS`).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from .config import DOSES
from .io import MANIFEST_NAME
from .stats import bh_correct, bootstrap_p_two_sided, cluster_bootstrap, paired_contrast

# --------------------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------------------
#: PREREGISTRATION section 1: confirmation set (test families x formulations) and the
#: development families (dose / layer / threshold choices only).
TEST_FAMILIES: tuple[str, ...] = tuple(f"F{i:02d}" for i in range(1, 13))
TEST_FORMULATIONS: tuple[str, ...] = ("A", "B")
DEV_FAMILIES: tuple[str, ...] = tuple(f"DEV{i:02d}" for i in range(1, 5))
#: Pre-registered doses (section 1, without 0); the steering doses 0.2 / 0.5 / 1.0 are exploratory.
PREREG_DOSES: tuple[float, ...] = tuple(float(e) for e in DOSES if float(e) > 0)
PRIMARY_K = 8
PRIMARY_C = "C1"
PRIMARY_MIN_EFFECT = 0.05          # minimum effect of section 2 (per layer)
N_RAND_PREREG = 32                 # random directions of the rand contrast (section 2)
AUDIT_MAX_REL_ERR = 1e-4           # section 4: written-vector relative error (FP32)
ALPHA = 0.05
DEFAULT_MODEL = "pythia-160m"
DEFAULT_REVISION = "step98000"
DEFAULT_N_BOOT = 10000
STAGE_LETTERS = tuple("ABCDEFGHIJKL")
#: Stages whose run directories the evaluation opens; the scale sweep F (which may still be
#: running) and the stages A, B, E are never opened, not even their manifests.
EVAL_STAGES = ("C", "D", "G", "H", "I", "J", "K", "L")
#: Run keys (stage L: the stand-in run and the trace-source run; stage D: the main run and the
#: token-language control run of PREREGISTRATION 6.11).
EVAL_KEYS = EVAL_STAGES + ("L_trace", "D_ctl")
#: Children of a parent folder (``MyDrive/LSNE``) that hold run roots.
ROOT_CHILDREN = ("runs", "runs_trace", "runs_ctl")
#: PREREGISTRATION 6.11 (token-language control of the primary endpoint): the commit that froze
#: the amendment and its time; a control run must be younger (``audit.D1ctl.run``).
AMENDMENT_611 = dict(id="6.11", commit="2ce9c71f4d0f902fdb504785c4d934b662e46f5b", utc="2026-10-28T08:05:51+00:00")
#: ``manifest.control`` of that control run (stage D, ``runs_ctl``); such a run is keyed ``D_ctl``.
CTL_CONTROL = "token_language"
#: Mirrors of the stage constants (tests/test_prereg_eval.py keeps them equal to the stage
#: modules; this module stays importable without torch).
PREREG_PATCH_CLASS = "delim"                                       # experiments.common
H_PREREG_DISCRIMINANT = "target_logit"                             # stage_h_lambda
H_PREREG_QUANTITIES = {"sae_fvu": -1, "steps_to_cross_censored": 0, "mahal_to_ref_mean": -1}
G_PREREG_STEER_EPS = 0.05                                          # stage_g_geometry
G_PREREG_MIN_COS = 0.3
I_PREREG_MIN_B = 0.05                                              # stage_i_compass
J_PREREG = ("culture_politics_vs_misfolding", "domain_axes", "last")   # stage_j_rdx
L_TRACE_MODES = ("residual", "kv", "both")                         # stage_l_transplant

VERDICTS = ("supported", "not_supported", "open", "gate_failed", "not_interpretable", "not_tested")
#: Columns of ``prereg_results.csv`` (fixed order) ...
RESULT_COLS = ["id", "name", "estimate", "ci_low", "ci_high", "p_two_sided", "p_one_sided", "n_families", "n_rows",
               "rule", "verdict", "bh_q", "deviation_note"]
#: ... followed by these: ``role`` (primary, primary_component, primary_deviation,
#: primary_control (PREREGISTRATION 6.11), audit, secondary, sensitivity, exploratory, gate,
#: conditional, copied), ``stage``, ``direction``
#: (+1 / -1 pre-registered, 0 two-sided), ``bh_group`` (row enters the BH family),
#: ``bh_q_endpoint`` (sensitivity grouping), ``source`` (stage/run/table) and ``min_effect``
#: (the pre-registered minimum effect the verdict is judged against by the interval, NaN for
#: none; added 27.10.2026 at the end so the earlier columns keep their order).
EXTRA_COLS = ["role", "stage", "direction", "bh_group", "bh_q_endpoint", "source", "min_effect"]
RUN_COLS = ["stage", "key", "root", "path", "status", "timestamp", "git_sha", "n_boot", "used", "note"]

#: Caveats the tables cannot show themselves (code-level properties of the stages and the
#: review of the step98000 runs, 2026-10-27); attached to the deviation note of every row
#: whose id starts with the key (longest key wins).
CAVEATS: dict[str, str] = {
    "P": ("the injected token is the same \".\" at K=0 and K>0 (C1, last token of unit 9); the K>0 vs K=0 contrast "
          "changes the language of the clause it closes (German at K>0) and puts the final D->E switch between the site "
          "and the shared English tail (control: PREREGISTRATION 6.11, rows P.ctl); lang_dom is fitted on all standard "
          "variants, test families included"),
    "P.dev": ("injection at the last German token (stimuli.last_switched_position), not at the pre-registered "
              "last token of unit 9; K=0 reference at the C1 site, so dlam mixes injection distance and condition"),
    "P.ctl": ("unit-defined sites (the \".\" ending unit 9 / unit 10, common.unit_last_position), not the last German "
              "token; control 2's site is 1 token before the shared tail instead of 6-14 (distance confound, so it decides "
              "only together with control 1); control 1 changes C (C2 / C3 move the final switch inside unit 9, before the "
              "site); directions from the bank of the main stage-D run (directions_depth.npz), no refit"),
    "S3": ("R = (logp_patched - logp_mixed) / (logp_mono - logp_mixed) is ill-conditioned for formulations with "
           "|T| ~ 0; the ws class is empty under the GPT-NeoX BPE, so delim = punct + boundary; direction from "
           "DESIGN 4.1 (R_ws+punct+boundary > R_random)"),
    "S4": ("the descending branch conditions on the German text the ascending branch generated while the restarts "
           "do not, so the area largely measures autoregressive context memory; area_z = area / null_area_sd"),
    "S6": ("tested one-sided in the State-2143 sign (+1, H1: ent above random; PREREGISTRATION 3 marks 8, 11, 14 and "
           "17(i) two-sided, not 6); the reversed sign (-1) is the control of DESIGN 6, not a test; pooled over every "
           "standard variant (stage C's pre-registered K=-1 / C=all cell; K = 0 alone as sensitivity); no dose was frozen "
           "on DEV01-04, so every pre-registered dose is its own BH row; ent comes from State 2143 (raw entropy gradient)"),
    "S7": ("G0 pools prefix means with the shared English tail excluded (PREREGISTRATION 7 says mean pool) over "
           "all selected families (DEV01-04 and F01-F20), not the test split"),
    "S8": ("n-matching of the K=8 cloud draws 4 of 32 realizations without stratifying by formulation (plausible "
           "confound); tables without a split column: bootstrap over all selected families as run; stage G's test "
           "rows are families F01-F12 whose clouds pool the formulations A-D, so the formulations C, D outside the "
           "confirmation set F01-F12 x A,B are inside these rows"),
    "S9": ("the 'layer of maximal discrete displacement' is chosen by disp_mahal_K, constant up to float noise at "
           "whitening rank ~ N-1 (review: the norm_K argmax is L11.block_output for every family); tables without a "
           "split column: all families; stage G's test rows are families F01-F12 whose clouds pool the formulations "
           "A-D, so the formulations C, D outside the confirmation set F01-F12 x A,B are inside these rows; the "
           "table's min_effect_met is the point estimate (the verdict here follows the interval)"),
    "S10": ("lambda_rel >= 0.3 paths saturate, so rho mostly contrasts lambda = 0 with lambda > 0; stage H pools DEV "
            "and test families (recomputed here on the test split)"),
    "S10iii": "partly by construction: lambda pulls the path towards the same K8/C1 cloud the distance is measured to",
    "S10f": ("run written before lambda_features_frozen: stage H selected candidates over all families, without the "
             "DEV-freeze / test-confirmation path"),
    "S11": ("single context, no inference; the target cloud is the generic K8/C1 cloud (few of its states come from "
            "State 2143); run the evaluation with --revision step99000 for the 99k half"),
    "S12": "documentation, no test; review: ent raises the lens mass of every domain, so the profile is not domain-selective",
    "S13": ("few culture/politics families (clusters); the counter-steer projection is uncentred and the cs arm is "
            "not dose-matched to lang_dom; the verdict here follows the interval (ci_low >= 0.05); stage-I tables "
            "written before the CI rule (no verdict / p_boot column) judge min_effect_met by the point estimate and "
            "flag every eps as pre-registered, later ones flag one dose (contrast_13_eps) and use the interval"),
    "S14": ("see the S14abs rows (same absolute norm) for the dose-matched comparison; "
            "the head arm writes a much smaller absolute RMS than the residual arm, so the difference is mostly "
            "residual-arm damage; no Soll/Ist audit is stored for either arm; several direction x dose cells "
            "without a pre-specified primary cell"),
    "S15": ("lenient null (isotropic random bases, explained ~ k/d); 'layers' = the grid sites; the heads basis uses "
            "the heads of the block after the site"),
    "S16": "the protein_misfolding bucket is small (keyword-classified Pile sequences)",
    "S17": ("exceeds_all_controls compares |delta| magnitudes without intervals; no DEV/test split; steps_to_threshold "
            "has little dynamic range at the default learning rate"),
    "S18": ("stand-in design of 2026-10-27: v_src at the source layer of each map (layer-consistent), dose "
            "Limits.l_eps = 0.3, 12 prompts round-robin over the families; a stand-in run that predates this design is "
            "rejected (read_standin); the cosine is in-sample"),
    "S18T": ("counts only with a healthy trace-LM corpus (lm_corpus_ok) and a validated stand-in; the first A100 trace "
             "run (2026-10-26) came from a broken pipeline (serialisation crashed, the hash split never ran, the "
             "vocabulary was built from 0 transactions, address OOV 0.967); a trace_verdict written before the "
             "tightened rule of 2026-10-27 (no lm_corpus_ok) is re-judged from the run's tables, never copied; "
             "transplant dose TRACE_EPS = 1.0; sharing / activity need an SAE"),
}


# --------------------------------------------------------------------------------------
# run discovery
# --------------------------------------------------------------------------------------
@dataclass
class StageRun:
    """One ``<root>/<stage>/<model>@<revision>/`` directory with its manifest."""
    stage: str
    path: Path
    root: Path
    manifest: dict = field(default_factory=dict)
    note: str = ""

    @property
    def kind(self) -> str:
        """Stage L: the source kind (``trace`` for the trace-source run), else ``""``."""
        src = self.manifest.get("source") if self.stage == "L" else None
        return str(src.get("kind", "")) if isinstance(src, dict) else ""

    @property
    def key(self) -> str:
        """Selection key: the stage letter, ``L_trace`` for the trace-source stage-L run,
        ``D_ctl`` for the token-language control run of stage D (``manifest.control ==
        "token_language"``, PREREGISTRATION 6.11), which therefore never replaces the main run."""
        if self.stage == "L" and self.kind == "trace":
            return "L_trace"
        if self.stage == "D" and self.manifest.get("control") == CTL_CONTROL:
            return "D_ctl"
        return self.stage

    @property
    def status(self) -> str:
        return str(self.manifest.get("status", ""))

    @property
    def timestamp(self) -> str:
        return str(self.manifest.get("timestamp", ""))

    @property
    def n_boot(self) -> Optional[int]:
        lim = self.manifest.get("limits") or {}
        v = lim.get("n_boot") if isinstance(lim, dict) else None
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    @property
    def reduced(self) -> bool:
        """A smoke or limited (scale-sweep) run (``limits.is_smoke`` / ``limits.limited``)."""
        lim = self.manifest.get("limits") or {}
        return bool(isinstance(lim, dict) and (lim.get("is_smoke") or lim.get("limited")))

    def label(self) -> str:
        return f"{self.stage}/{self.path.name}"


def model_tag(model: str) -> str:
    """``EleutherAI/pythia-160m`` -> ``pythia-160m`` (the run-directory tag)."""
    return str(model).split("/")[-1]


def _has_stage_dirs(p: Path) -> bool:
    try:
        return any(c.is_dir() and c.name in STAGE_LETTERS for c in p.iterdir())
    except OSError:
        return False


def resolve_roots(roots: Sequence) -> tuple[list[Path], list[str]]:
    """Run roots from the user's list: a folder with stage directories is used as is, a
    parent folder without them contributes its ``runs`` / ``runs_trace`` / ``runs_ctl`` children.  Returns
    ``(roots, notes)``; duplicates (same resolved path) are dropped."""
    out: list[Path] = []
    notes: list[str] = []
    seen: set[str] = set()

    def add(p: Path):
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)

    for r in roots:
        if r is None or str(r).strip() == "":
            continue
        p = Path(str(r)).expanduser()
        if not p.is_dir():
            notes.append(f"run root {p} does not exist (not mounted, or the Drive shortcut is missing)")
            continue
        if _has_stage_dirs(p):
            add(p)
            continue
        kids = [p / c for c in ROOT_CHILDREN if (p / c).is_dir() and _has_stage_dirs(p / c)]
        if kids:
            for k in kids:
                add(k)
            notes.append(f"{p} has no stage folders; using {', '.join(k.name for k in kids)} below it")
        else:
            notes.append(f"run root {p} holds no stage folders (A-L)")
    return out, notes


def discover_runs(roots: Sequence, model: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION) -> tuple[list[StageRun], list[str]]:
    """Every ``<root>/<stage>/<model>@<revision>/manifest.json`` of the stages in
    :data:`EVAL_STAGES` below the resolved roots (any status; an unreadable manifest is
    listed with status ``unreadable``)."""
    tag = f"{model_tag(model)}@{str(revision).replace('/', '-')}"
    resolved, notes = resolve_roots(roots)
    runs: list[StageRun] = []
    for root in resolved:
        for stage in EVAL_STAGES:
            d = root / stage / tag
            mp = d / MANIFEST_NAME
            if not mp.is_file():
                continue
            try:
                man = json.loads(mp.read_text(encoding="utf-8"))
                run = StageRun(stage=stage, path=d, root=root, manifest=man if isinstance(man, dict) else {})
                if run.status == "ok" and (d / "error.txt").exists():
                    run.note = "stale error.txt of an earlier attempt next to the ok manifest (ignored)"
                runs.append(run)
            except Exception as e:  # noqa: BLE001 - a broken manifest must not stop the evaluation
                runs.append(StageRun(stage=stage, path=d, root=root, manifest={"status": "unreadable"},
                                     note=f"manifest unreadable: {type(e).__name__}"))
    return runs, notes


def select_runs(runs: Sequence[StageRun]) -> dict[str, StageRun]:
    """The run used per key (stage letter, ``L_trace``): the newest ok manifest, a full run
    before a smoke / limited one; the other runs get a note (superseded / not ok)."""
    chosen: dict[str, StageRun] = {}
    by_key: dict[str, list[StageRun]] = {}
    for r in runs:
        by_key.setdefault(r.key, []).append(r)
    for key, rs in by_key.items():
        ok = sorted((r for r in rs if r.status == "ok"), key=lambda r: (not r.reduced, r.timestamp), reverse=True)
        if ok:
            chosen[key] = ok[0]
            for r in ok[1:]:
                why = "a full run" if r.reduced and not ok[0].reduced else "the newer run"
                r.note = (r.note + "; " if r.note else "") + f"superseded by {why} {ok[0].path}"
        for r in rs:
            if r.status != "ok":
                r.note = (r.note + "; " if r.note else "") + f"status {r.status!r}: not used"
    return chosen


def runs_table(runs: Sequence[StageRun], chosen: dict[str, StageRun], notes: Sequence[str] = ()) -> pd.DataFrame:
    rows = [dict(stage=r.stage, key=r.key, root=str(r.root), path=str(r.path), status=r.status, timestamp=r.timestamp,
                 git_sha=str(r.manifest.get("git_sha", "")), n_boot=r.n_boot,
                 used=bool(chosen.get(r.key) is r), note=r.note) for r in runs]
    for n in notes:
        rows.append(dict(stage="", key="", root="", path="", status="note", timestamp="", git_sha="", n_boot=None,
                         used=False, note=n))
    return pd.DataFrame(rows, columns=RUN_COLS)


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
def read_table(run_dir, name: str, columns: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
    """``<run_dir>/<name>.parquet`` (preferred) or ``.csv``; only ``columns`` that exist
    are read.  ``None`` when neither file exists; an empty csv gives an empty frame."""
    d = Path(run_dir)
    pq, csv = d / f"{name}.parquet", d / f"{name}.csv"
    cols = list(columns) if columns is not None else None
    if pq.exists():
        try:
            if cols is None:
                return pd.read_parquet(pq)
            try:
                return pd.read_parquet(pq, columns=cols)
            except Exception:  # a requested column is absent
                df = pd.read_parquet(pq)
                return df[[c for c in cols if c in df.columns]]
        except Exception:  # noqa: BLE001 - unreadable parquet: the csv is the record
            pass
    if not csv.exists():
        return None
    try:
        if cols is None:
            return pd.read_csv(csv)
        head = pd.read_csv(csv, nrows=0).columns
        return pd.read_csv(csv, usecols=[c for c in cols if c in head])
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=cols or [])


class Tables:
    """Tables of the selected runs, read once (``get``) and cached."""

    def __init__(self, chosen: dict[str, StageRun]):
        self.chosen = chosen
        self._cache: dict = {}

    def run(self, key: str) -> Optional[StageRun]:
        return self.chosen.get(key)

    def get(self, key: str, name: str, columns: Optional[Sequence[str]] = None) -> Optional[pd.DataFrame]:
        run = self.run(key)
        if run is None:
            return None
        ck = (key, name, tuple(columns) if columns is not None else None)
        if ck not in self._cache:
            self._cache[ck] = read_table(run.path, name, columns)
        return self._cache[ck]

    def source(self, key: str, name: str) -> str:
        run = self.run(key)
        return f"{run.label()}/{name}" if run is not None else ""


# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _finite(*xs) -> bool:
    return all(np.isfinite(_f(x)) for x in xs)


def _flag(v) -> Optional[bool]:
    """Boolean of a flag cell (bool in memory, 'True'/'False' after a csv reload); None if empty."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        return None
    try:
        if not np.isfinite(float(v)):
            return None
        return bool(v)
    except (TypeError, ValueError):
        return None


def _g(x, d: int = 4) -> str:
    x = _f(x)
    if not np.isfinite(x):
        return "nan"
    return f"{x:.{d}g}" if abs(x) < 1e-3 and x != 0 else f"{x:.{d}f}"


def _eps_key(eps: float) -> str:
    return f"{float(eps):g}"


def _standard(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["kind"].astype(str) == "standard"] if "kind" in df.columns else df


def split_rows(df: pd.DataFrame, families: Sequence[str] = TEST_FAMILIES,
               formulations: Optional[Sequence[str]] = TEST_FORMULATIONS) -> pd.DataFrame:
    """Rows of the confirmation set (``family`` in ``families`` and ``formulation`` in
    ``formulations``; ``formulations=None`` keeps every formulation)."""
    m = df["family"].astype(str).isin([str(f) for f in families])
    if formulations is not None and "formulation" in df.columns:
        m &= df["formulation"].astype(str).isin([str(f) for f in formulations])
    return df[m]


#: Note of the ``split="test"`` rows of a stage table (family means over F01-F12 x {A, B}) ...
SPLIT_TEST_NOTE = "split test of the table (F01-F12 x A,B, family means)"
#: ... and of stage G's, whose test rows are written per family (``formulation_col=None``):
#: its clouds pool the formulations A-D of a family.
G_SPLIT_TEST_NOTE = ("split test = families F01-F12, clouds pool formulations A-D (C, D are outside the confirmation set "
                     "F01-F12 x A,B)")


def boot_split(df: pd.DataFrame, test_note: str = SPLIT_TEST_NOTE) -> tuple[pd.DataFrame, str]:
    """``(rows, note)``: the confirmation-set rows of a bootstrap table copied from a stage.
    Tables written since the stages carry a ``split`` column hold the pooled ``split="all"``
    rows next to the family-mean ``split="test"`` rows: the test rows are used when present
    (note ``test_note``: :data:`SPLIT_TEST_NOTE`, :data:`G_SPLIT_TEST_NOTE` for stage G), else
    the pooled rows.  A table without the column (runs written before it existed) is returned
    unchanged with an empty note."""
    if "split" not in df.columns:
        return df, ""
    s = df["split"].astype(str)
    if (s == "test").any():
        return df[s == "test"], test_note
    return df[s == "all"], "split all of the table (no test rows)"


def _missing_cols(df: Optional[pd.DataFrame], cols: Sequence[str]) -> list[str]:
    return list(cols) if df is None else [c for c in cols if c not in df.columns]


def p_one_sided(boot, direction: int) -> float:
    """One-sided percentile-bootstrap p for H1 ``sign(stat) == direction``:
    ``(#{boot on the H0 side of 0} + 1) / (n + 1)``; NaN for ``direction == 0``."""
    if not direction:
        return float("nan")
    b = np.asarray(boot, dtype=np.float64).reshape(-1)
    b = b[np.isfinite(b)]
    if b.size == 0:
        return float("nan")
    k = np.sum(b <= 0.0) if direction > 0 else np.sum(b >= 0.0)
    return float(min(1.0, (k + 1.0) / (b.size + 1.0)))


def family_boot(df: pd.DataFrame, value_col: str, n_boot: int, seed: int, direction: int = 0,
                n_rows: Optional[int] = None) -> dict:
    """Family cluster bootstrap of the mean of family means of ``value_col`` (``family``
    column); returns the row fields estimate / ci / p / n plus ``boot`` and the per-family
    values."""
    r = cluster_bootstrap(df, value_col, "family", n_boot=int(n_boot), seed=int(seed), stat_level="cluster")
    return dict(estimate=float(r["mean"]), ci_low=float(r["ci_low"]), ci_high=float(r["ci_high"]),
                p_two_sided=bootstrap_p_two_sided(r["boot"], 0.0) if r["n_clusters"] else float("nan"),
                p_one_sided=p_one_sided(r["boot"], direction) if r["n_clusters"] else float("nan"),
                n_families=int(r["n_clusters"]), n_rows=int(r["n_rows"] if n_rows is None else n_rows),
                boot=r["boot"], keys=list(r["cluster_keys"]), values=r["cluster_values"])


def _stat_fields(b: dict) -> dict:
    return {k: b[k] for k in ("estimate", "ci_low", "ci_high", "p_two_sided", "p_one_sided", "n_families", "n_rows")}


def make_row(id: str, name: str, *, role: str, stage: str, rule: str = "", verdict: str = "not_tested", note: str = "",
             estimate=np.nan, ci_low=np.nan, ci_high=np.nan, p_two_sided=np.nan, p_one_sided=np.nan, n_families=0,
             n_rows=0, direction: int = 0, bh_group: bool = False, source: str = "", min_effect=np.nan) -> dict:
    if verdict not in VERDICTS:
        raise ValueError(f"unknown verdict {verdict!r}")
    return dict(id=id, name=name, estimate=_f(estimate), ci_low=_f(ci_low), ci_high=_f(ci_high),
                p_two_sided=_f(p_two_sided), p_one_sided=_f(p_one_sided), n_families=int(n_families or 0),
                n_rows=int(n_rows or 0), rule=rule, verdict=verdict, bh_q=np.nan, deviation_note=note, role=role,
                stage=stage, direction=int(direction), bh_group=bool(bh_group), bh_q_endpoint=np.nan, source=source,
                min_effect=_f(min_effect))


def not_tested(id: str, name: str, stage: str, reason: str, role: str = "secondary", rule: str = "") -> dict:
    return make_row(id, name, role=role, stage=stage, rule=rule, verdict="not_tested", note=reason)


def _no_run(id: str, name: str, stage: str, key: str, ctx: "EvalContext", role: str = "secondary", rule: str = "") -> dict:
    which = {"L_trace": "the trace-source stage-L run",
             "D_ctl": "the token-language control run of stage D (manifest control = 'token_language', runs_ctl)"
             }.get(key, f"stage {stage}")
    return not_tested(id, name, stage, f"no run of {which} ({ctx.tag}, status ok) under the given roots", role, rule)


def _no_table(id: str, name: str, stage: str, table: str, ctx: "EvalContext", key: Optional[str] = None,
              cols: Sequence[str] = (), role: str = "secondary", rule: str = "") -> dict:
    run = ctx.tables.run(key or stage)
    where = str(run.path) if run is not None else "?"
    exists = run is not None and any((run.path / f"{t.strip()}.{ext}").exists()
                                     for t in table.split("/") for ext in ("csv", "parquet"))
    if cols:
        what = f"columns {list(cols)} missing in {table}"
    elif exists:
        what = f"table {table} empty or without the required columns"
    else:
        what = f"table {table} missing"
    return not_tested(id, name, stage, f"{what} ({where})", role, rule)


def min_effect_status(est, lo, hi, min_effect: float, direction: int = 1) -> str:
    """Minimum-effect rule of PREREGISTRATION section 2, judged by the 95 % interval alone and
    never by the point estimate (the rule of the stage tables, ``common.min_effect_verdict``
    and docs/STAGES.md): ``pass`` iff the lower bound reaches ``min_effect`` (in ``direction``),
    ``fail`` iff the upper bound lies below it (the minimum effect is excluded), ``open``
    otherwise (the minimum effect neither established nor excluded, whether or not the interval
    includes 0); ``nan`` when not computable.  ``common.min_effect_verdict`` maps as supported
    -> pass, open -> open, null / negative -> fail, positive_below_min -> open or fail (by the
    upper bound)."""
    if not _finite(est, lo, hi):
        return "nan"
    s = 1.0 if direction >= 0 else -1.0
    a, b = sorted((s * _f(lo), s * _f(hi)))
    if a >= min_effect:
        return "pass"
    if b < min_effect:
        return "fail"
    return "open"


#: :func:`min_effect_status` -> verdict
MIN_EFFECT_VERDICT = {"pass": "supported", "open": "open", "fail": "not_supported", "nan": "not_tested"}


def secondary_verdict(est, lo, hi, q=np.nan, direction: int = 0) -> str:
    """Secondary rule: supported iff the CI excludes 0 (in the pre-registered direction, if
    any) and the BH q <= 0.05 (rows outside the BH family: the CI alone); not_supported iff
    the CI excludes 0 against a pre-registered direction; otherwise open (no minimum effect
    is pre-registered, so an interval that includes 0 is not a null finding)."""
    if not _finite(est, lo, hi):
        return "not_tested"
    lo, hi = _f(lo), _f(hi)
    sig = (not np.isfinite(_f(q))) or _f(q) <= ALPHA
    if direction == 0:
        return "supported" if (lo > 0 or hi < 0) and sig else "open"
    right = lo > 0 if direction > 0 else hi < 0
    wrong = hi < 0 if direction > 0 else lo > 0
    if right and sig:
        return "supported"
    if wrong:
        return "not_supported"
    return "open"


def bh_verdict(est, lo, hi, q=np.nan, direction: int = 0, min_effect=np.nan) -> str:
    """Verdict of a row of the BH family: :func:`secondary_verdict`, or for a pre-registered
    minimum effect (``min_effect`` finite: contrasts 9 and 13) the interval rule of
    :func:`min_effect_status` with BH q <= 0.05 as a further condition of ``supported``."""
    if not np.isfinite(_f(min_effect)):
        return secondary_verdict(est, lo, hi, q, direction)
    v = MIN_EFFECT_VERDICT[min_effect_status(est, lo, hi, _f(min_effect), direction or 1)]
    sig = (not np.isfinite(_f(q))) or _f(q) <= ALPHA
    return "open" if v == "supported" and not sig else v


#: Note of a copied row whose table carries no ``p_boot`` (runs written before the column).
NO_P_NOTE = "no p in the table: not BH-corrected (verdict capped at open)"


def table_p(r, direction: int = 0) -> tuple[float, float]:
    """``(p_two_sided, p_one_sided)`` of a copied bootstrap row from its ``p_boot`` column (the
    two-sided bootstrap p of a zero mean, ``common.boot_row(with_p=True)``): for H1
    ``sign == direction`` the one-sided p is ``p_boot / 2`` when the estimate lies on the H1
    side, else ``1 - p_boot / 2`` (NaN for ``direction == 0``); both NaN without a p."""
    p = _f(r.get("p_boot")) if hasattr(r, "get") else float("nan")
    if not np.isfinite(p):
        return float("nan"), float("nan")
    if not direction:
        return p, float("nan")
    est = _f(r.get("mean"))
    h1_side = np.isfinite(est) and (est > 0 if direction > 0 else est < 0)
    return p, (p / 2.0 if h1_side else 1.0 - p / 2.0)


def _cap_open(verdict: str) -> str:
    """A verdict that could not be BH-corrected: ``supported`` becomes ``open``."""
    return "open" if verdict == "supported" else verdict


def _rule_secondary(direction: int) -> str:
    side = {1: "one-sided, H1 > 0", -1: "one-sided, H1 < 0"}.get(direction, "two-sided")
    return (f"family-mean cluster bootstrap on the test split ({side}); supported iff the 95% CI excludes 0 "
            "and BH q <= 0.05; open otherwise (no minimum effect)")


def endpoint_key(row_id: str) -> str:
    """``S3.K0_vs_K8C1.L2`` -> ``S3``; ``S10ii`` -> ``S10``; ``P@0.05`` -> ``P``; ``P.ctl1@0.05`` -> ``P.ctl``."""
    m = re.match(r"(S\d+|P\.ctl|P|audit)", str(row_id))
    return m.group(1) if m else str(row_id)


def caveat_keys(row_id: str) -> list[str]:
    """Keys of :data:`CAVEATS` that apply to a row: its sub-endpoint (``P.dev``, ``P.ctl``,
    ``S10iii``, ``S18T``, ...) or else its endpoint (``P``, ``S3``); the roman sub-endpoints
    (``S10ii``, ``S17iii``) carry both.  Exploratory ``S3x`` rows carry none."""
    m = re.match(r"(P|S\d+)(\.dev|\.ctl|i{1,3}|f|T|x)?", str(row_id))
    if not m:
        return []
    base, sub = m.group(1), m.group(2) or ""
    keys = [base + sub] if sub else [base]
    if sub and set(sub) <= {"i"}:
        keys.append(base)
    return [k for k in dict.fromkeys(keys) if k in CAVEATS]


def caveat_for(row_id: str) -> str:
    """The joined :data:`CAVEATS` of a row (:func:`caveat_keys`)."""
    return _join(*(CAVEATS[k] for k in caveat_keys(row_id)))


def _join(*parts) -> str:
    return "; ".join(str(p) for p in parts if p)


def _nboot_note(run: Optional[StageRun]) -> str:
    if run is None or run.n_boot is None:
        return ""
    return (f"copied interval from n_boot = {run.n_boot} as run (pre-registered 10000)"
            if run.n_boot < DEFAULT_N_BOOT else f"n_boot = {run.n_boot} as run")


# --------------------------------------------------------------------------------------
# evaluation context
# --------------------------------------------------------------------------------------
@dataclass
class EvalContext:
    tables: Tables
    model: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    n_boot: int = DEFAULT_N_BOOT
    seed: int = 0
    frozen_dose: Optional[float] = None
    families: tuple = TEST_FAMILIES
    formulations: tuple = TEST_FORMULATIONS

    @property
    def tag(self) -> str:
        return f"{model_tag(self.model)}@{self.revision}"

    @property
    def split_note(self) -> str:
        if tuple(self.families) == TEST_FAMILIES and tuple(self.formulations) == TEST_FORMULATIONS:
            return ""
        return (f"confirmation set overridden: families {','.join(self.families)} x formulations "
                f"{','.join(self.formulations)} (pre-registered: F01-F12 x A,B)")

    def split(self, df: pd.DataFrame) -> pd.DataFrame:
        return split_rows(df, self.families, self.formulations)


# --------------------------------------------------------------------------------------
# audits (section 4)
# --------------------------------------------------------------------------------------
def audit_summary(values) -> dict:
    """``audit_rel_err`` summary: n, n_finite, n_nan, max, q50 / q99 / q999, n_over (> 1e-4)."""
    v = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(v)
    f = v[ok]
    q = np.quantile(f, [0.5, 0.99, 0.999]) if f.size else [np.nan] * 3
    return dict(n=int(v.size), n_finite=int(f.size), n_nan=int((~ok).sum()), max=float(f.max()) if f.size else np.nan,
                q50=float(q[0]), q99=float(q[1]), q999=float(q[2]), n_over=int((f > AUDIT_MAX_REL_ERR).sum()))


def audit_row(id: str, name: str, stage: str, s: dict, source: str, extra: str = "") -> dict:
    if s["n_finite"] == 0:
        verdict = "not_tested"
    else:
        verdict = "supported" if s["n_over"] == 0 else "not_supported"
    note = _join(f"q50 {_g(s['q50'], 3)}, q99 {_g(s['q99'], 3)}, q99.9 {_g(s['q999'], 3)}, max {_g(s['max'], 3)}",
                 f"rows > 1e-4: {s['n_over']} of {s['n_finite']}" + (f" (NaN: {s['n_nan']})" if s["n_nan"] else ""),
                 "run INVALID (PREREGISTRATION section 4)" if s["n_over"] else "", extra)
    return make_row(id, name, role="audit", stage=stage, estimate=s["max"], n_rows=s["n_finite"], verdict=verdict,
                    rule="run valid iff no written-vector relative error > 1e-4 (FP32, PREREGISTRATION section 4)",
                    note=note, source=source)


# --------------------------------------------------------------------------------------
# primary endpoint (section 2, stage D)
# --------------------------------------------------------------------------------------
DAMAGE_COLS = ["tag", "family", "formulation", "K", "C", "kind", "inject", "direction", "dir_family", "eps", "lam",
               "audit_rel_err"]
#: Columns read from the main run's ``damage`` (``inject_pos`` only for the replication audit of
#: the control run, ``audit.D1ctl.rep``; one cached read serves both).
DAMAGE_READ_COLS = DAMAGE_COLS + ["inject_pos"]
PRIMARY_NAME = "PRIMARY dlam(K, lang_dom) = lam(K>0, C1) - lam(K0), depth injection"
PRIMARY_RULE = ("supported iff the 95% CI of dlam(lang_dom, K=8, C1) lies at or above the minimum effect (ci_low >= 0.05; "
                "the interval decides, never the point estimate, as in damage_boot) AND the CI of DiD (lang_dom - rand) "
                "> 0 AND monotone in K (the CI of dlam at the first K and of every increment between successive K lies "
                "above 0; PREREGISTRATION 6.10); not_supported iff the dlam CI lies below 0.05, the DiD CI lies below 0 "
                "or includes 0 below 0.05, or a step in K has a CI below 0 (monotonicity refuted); open otherwise "
                "(a component neither established nor excluded; PREREGISTRATION section 2)")
PRIMARY_MONO_RULE = ("monotone in K (PREREGISTRATION 6.10, reading (b)): supported iff the 95% CI of dlam at the first K > 0 "
                     "and of every increment dlam(K_i) - dlam(K_i-1) lies above 0; not_supported iff one of these CIs lies "
                     "below 0; open otherwise; monotone point estimates (reading (a)) and the bootstrap P(monotone) are "
                     "reported in the note")
#: Rules of the component rows (dlam(lang_dom) per K, DiD per K) of P and of its control.
COMPONENT_LANG_RULE = "component: ci_low >= 0.05 (interval rule of section 2); not_supported iff ci_high < 0.05; open otherwise"
COMPONENT_DID_RULE = "component: CI of the DiD > 0 (H1: larger than the rand contrast)"


def _variant_lam(d: pd.DataFrame, fam: str) -> pd.DataFrame:
    """Per variant and dose: mean ``lam`` over the directions of ``fam`` and their number."""
    sub = d[d["dir_family"].astype(str) == fam]
    keys = ["family", "formulation", "tag", "K", "C", "eps"]
    if sub.empty:
        return pd.DataFrame(columns=keys + ["lam", "n_dirs"])
    g = sub.groupby(keys, sort=True)
    return pd.DataFrame({"lam": g["lam"].mean(), "n_dirs": g["direction"].nunique()}).reset_index()


def family_delta(var: pd.DataFrame, K: int, Cc) -> pd.DataFrame:
    """Per family: mean lam over the (K, C) variants minus mean lam over the K=0 variants
    (``value``), with the number of variant rows on both sides (``n_rows``).  ``Cc`` is one C
    level or a tuple / list of them, pooled in one family mean (control 1 of PREREGISTRATION
    6.11: C2 and C3 together)."""
    cs = [str(c) for c in Cc] if isinstance(Cc, (tuple, list)) else [str(Cc)]
    mixed = var[(var["K"] == int(K)) & var["C"].astype(str).isin(cs)].groupby("family")["lam"].agg(["mean", "size"])
    ref = var[var["K"] == 0].groupby("family")["lam"].agg(["mean", "size"])
    j = mixed.join(ref, lsuffix="_k", rsuffix="_0", how="inner")
    return pd.DataFrame({"family": j.index.astype(str), "value": (j["mean_k"] - j["mean_0"]).to_numpy(dtype=float),
                         "n_rows": (j["size_k"] + j["size_0"]).to_numpy(dtype=int)})


def _did(fl: pd.DataFrame, fr: pd.DataFrame) -> pd.DataFrame:
    m = fl.merge(fr, on="family", suffixes=("_l", "_r"))
    return pd.DataFrame({"family": m["family"], "value": m["value_l"] - m["value_r"], "n_rows": m["n_rows_l"] + m["n_rows_r"]})


def _did_status(b: dict) -> str:
    """DiD component: ``pass`` (CI > 0), ``open`` (CI includes 0 and reaches 0.05), ``fail``."""
    if not _finite(b["estimate"], b["ci_low"], b["ci_high"]):
        return "nan"
    if b["ci_low"] > 0:
        return "pass"
    return "open" if b["ci_low"] <= 0 <= b["ci_high"] and b["ci_high"] >= PRIMARY_MIN_EFFECT else "fail"


def primary_verdict(lang_status: str, did_status: str, mono_status) -> str:
    """Section 2: supported iff the lang_dom contrast, the DiD and the rise in K pass
    (``mono_status``: ``pass`` / ``open`` / ``fail``, PREREGISTRATION 6.10; a bool is read as
    pass / fail); not_supported iff a component fails; open iff none fails and one is open;
    not_tested when the lang_dom contrast (or, with it passing, the rand contrast) is missing."""
    if isinstance(mono_status, (bool, np.bool_)):
        mono_status = "pass" if mono_status else "fail"
    if lang_status == "nan":
        return "not_tested"
    if "fail" in (lang_status, did_status, mono_status):
        return "not_supported"
    if did_status == "nan":
        return "not_tested"
    if "open" in (lang_status, did_status, mono_status):
        return "open"
    return "supported"


def _boot_fam(fam: pd.DataFrame, ctx: EvalContext, direction: int = 1) -> dict:
    return family_boot(fam, "value", ctx.n_boot, ctx.seed, direction=direction,
                       n_rows=int(fam["n_rows"].sum()) if len(fam) else 0)


def _rand_note(rand: pd.DataFrame) -> str:
    """How many random directions the rand contrast averages per variant (:func:`_variant_lam` rows)."""
    if rand.empty:
        return "no rand rows: DiD not computable"
    lo_n, hi_n = int(rand["n_dirs"].min()), int(rand["n_dirs"].max())
    return (f"rand = mean over {lo_n if lo_n == hi_n else f'{lo_n}-{hi_n}'} random directions per variant"
            + (f" (fewer than the pre-registered {N_RAND_PREREG})" if hi_n < N_RAND_PREREG else ""))


def _ci_txt(b: dict) -> str:
    return f"{_g(b['estimate'])} [{_g(b['ci_low'])}, {_g(b['ci_high'])}]"


def _depth_gate_note(run: StageRun) -> str:
    p = run.path / "directions_depth_meta.json"
    if not p.is_file():
        return ""
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        m = (meta.get("directions") or {}).get("lang_dom") or {}
    except Exception:  # noqa: BLE001
        return ""
    ok = _flag(m.get("gate_ok"))
    if ok is False:
        return (f"depth-site lang_dom failed its denoising gate (cos_denoised_plain {_g(m.get('cos_denoised_plain'), 3)} "
                f"< {_g(m.get('gate_cos', 0.7), 2)}): the axis keeps little of the plain D-E mean difference at "
                f"{(meta.get('site') or {}).get('layer', '?')}")
    return ""


def _frozen_dose(ctx: EvalContext, run: StageRun) -> Optional[float]:
    if ctx.frozen_dose is not None:
        return float(ctx.frozen_dose)
    for k in ("frozen_dose", "prereg_dose"):
        v = run.manifest.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def primary_rows(ctx: EvalContext) -> list[dict]:
    run = ctx.tables.run("D")
    if run is None:
        return [_no_run("P", PRIMARY_NAME, "D", "D", ctx, role="primary", rule=PRIMARY_RULE),
                not_tested("audit.D1", "D1 injection audit (audit_rel_err)", "D", "no stage-D run", role="audit")]
    dmg = ctx.tables.get("D", "damage", DAMAGE_READ_COLS)
    miss = _missing_cols(dmg, DAMAGE_COLS)
    src = ctx.tables.source("D", "damage")
    if miss:
        return [_no_table("P", PRIMARY_NAME, "D", "damage", ctx, cols=miss if dmg is not None else (), role="primary",
                          rule=PRIMARY_RULE),
                not_tested("audit.D1", "D1 injection audit (audit_rel_err)", "D", "damage table unusable", role="audit")]
    rows: list[dict] = []
    # ---- section 4 audit over the whole damage table (both injections, every dose)
    a_all = audit_summary(dmg["audit_rel_err"])
    by_inj = "; ".join(f"{inj}: max {_g(audit_summary(g['audit_rel_err'])['max'], 3)}"
                       for inj, g in dmg.groupby("inject", sort=True))
    rows.append(audit_row("audit.D1", "D1 injection audit: written-vector relative error (stage D damage)", "D", a_all,
                          src, by_inj))
    invalid = a_all["n_over"] > 0
    # ---- confirmation set, depth injection
    d = dmg[(dmg["inject"].astype(str) == "depth")]
    d = ctx.split(_standard(d)).copy()
    d = d[np.isfinite(pd.to_numeric(d["lam"], errors="coerce"))]
    d["eps"] = d["eps"].astype(float).round(10)
    d["K"] = d["K"].astype(int)
    lang, rand = _variant_lam(d, "lang_dom"), _variant_lam(d, "rand")
    if lang.empty:
        rows.insert(0, not_tested("P", PRIMARY_NAME, "D",
                                  _join("no depth-injection lang_dom rows of the confirmation set in damage", ctx.split_note),
                                  role="primary", rule=PRIMARY_RULE))
        return rows
    doses = sorted(set(lang["eps"]))
    prereg = [e for e in doses if any(np.isclose(e, p) for p in PREREG_DOSES)]
    explo = [e for e in doses if e not in prereg]
    missing = [p for p in PREREG_DOSES if not any(np.isclose(p, e) for e in doses)]
    frozen = _frozen_dose(ctx, run)
    main = next((e for e in prereg if frozen is not None and np.isclose(e, frozen)), None)
    dose_note = _join(f"pre-registered doses evaluated: {', '.join(_eps_key(e) for e in prereg) or 'none'}",
                      f"missing: {', '.join(_eps_key(e) for e in missing)}" if missing else "",
                      f"exploratory doses present (not evaluated): {', '.join(_eps_key(e) for e in explo)}" if explo else "")
    if frozen is None:
        dose_note = _join(dose_note, "no DEV-frozen dose recorded: every pre-registered dose is reported, none is the main "
                                     f"dose (multiplicity over {len(prereg)} doses not controlled)")
    elif main is None:
        dose_note = _join(dose_note, f"frozen dose {_eps_key(frozen)} not in the run: every dose reported, no main dose")
    gate_note = _depth_gate_note(run)
    for eps in prereg:
        rows.extend(_primary_dose(ctx, lang[lang["eps"] == eps], rand[rand["eps"] == eps], eps, main, dose_note, gate_note,
                                  invalid, a_all, src, len(prereg)))
    if not prereg:
        rows.insert(0, not_tested("P", PRIMARY_NAME, "D", _join("no pre-registered dose in damage", dose_note),
                                  role="primary", rule=PRIMARY_RULE))
    return rows


def _primary_dose(ctx: EvalContext, lang: pd.DataFrame, rand: pd.DataFrame, eps: float, main: Optional[float],
                  dose_note: str, gate_note: str, invalid: bool, audit: dict, src: str, n_doses: int = 1) -> list[dict]:
    sfx = "" if main is not None and np.isclose(eps, main) else f"@{_eps_key(eps)}"
    rows: list[dict] = []
    ks = sorted(int(k) for k in set(lang[(lang["K"] > 0) & (lang["C"].astype(str) == PRIMARY_C)]["K"]))
    if not ks or lang[lang["K"] == 0].empty:
        return [not_tested(f"P{sfx}", f"{PRIMARY_NAME}, eps={_eps_key(eps)}", "D",
                           "no K=0 reference or no K>0 C1 rows at this dose", role="primary", rule=PRIMARY_RULE)]
    rand_note = _rand_note(rand)
    bl, bd, br, fls = {}, {}, {}, {}
    for K in ks:
        fl = family_delta(lang, K, PRIMARY_C)
        fls[K] = fl
        bl[K] = _boot_fam(fl, ctx, 1)
        if len(rand):
            fr = family_delta(rand, K, PRIMARY_C)
            br[K] = _boot_fam(fr, ctx, 1)
            bd[K] = _boot_fam(_did(fl, fr), ctx, 1)
    kmain = PRIMARY_K if PRIMARY_K in ks else max(ks)
    deltas = [bl[K]["estimate"] for K in ks]
    mono_point = bool(np.all(np.diff([0.0] + deltas) > 0))      # reading (a), reported only
    # monotonicity under the bootstrap: the draws align when every K has the same families
    same = len({tuple(bl[K]["keys"]) for K in ks}) == 1
    p_mono = np.nan
    if same and ctx.n_boot > 0:
        arr = np.vstack([np.zeros(len(bl[ks[0]]["boot"]))] + [bl[K]["boot"] for K in ks])
        p_mono = float(np.mean(np.all(np.diff(arr, axis=0) > 0, axis=0)))
    inc_notes, inc_lows, inc_highs = [], [bl[ks[0]]["ci_low"]], [bl[ks[0]]["ci_high"]]
    for k0, k1 in zip(ks[:-1], ks[1:]):
        m = fls[k1].merge(fls[k0], on="family", suffixes=("_1", "_0"))
        inc = pd.DataFrame({"family": m["family"], "value": m["value_1"] - m["value_0"], "n_rows": m["n_rows_1"]})
        b = _boot_fam(inc, ctx, 1)
        inc_lows.append(b["ci_low"])
        inc_highs.append(b["ci_high"])
        inc_notes.append(f"K{k0}->K{k1} {_g(b['estimate'])} [{_g(b['ci_low'])}, {_g(b['ci_high'])}]")
    # reading (b) of PREREGISTRATION 6.10: the rise from K=0 to the first K and every later step has a CI above 0
    # (pass); a step with a CI below 0 refutes it (fail); otherwise neither shown nor refuted (open, section 2)
    if all(np.isfinite(lo) and lo > 0 for lo in inc_lows):
        mono = "pass"
    elif any(np.isfinite(hi) and hi < 0 for hi in inc_highs):
        mono = "fail"
    else:
        mono = "open"
    lang_status = min_effect_status(bl[kmain]["estimate"], bl[kmain]["ci_low"], bl[kmain]["ci_high"], PRIMARY_MIN_EFFECT, 1)
    did_status = _did_status(bd[kmain]) if kmain in bd else "nan"
    verdict = primary_verdict(lang_status, did_status, mono)
    audit_note = ""
    if invalid:
        audit_note = (f"run INVALID by the D1 audit ({audit['n_over']} rows > 1e-4, max {_g(audit['max'], 3)}); "
                      f"rule outcome ignoring the audit: {verdict}")
        verdict = "not_interpretable"
    did_main = bd.get(kmain)
    k_note = (f"dlam(lang_dom, K={kmain}) {lang_status}, DiD {did_status}, monotone {mono} (increments' CIs vs 0, "
              f"PREREGISTRATION 6.10); point estimates monotone {mono_point}"
              + (f" (bootstrap P(monotone) {_g(p_mono, 3)})" if np.isfinite(p_mono) else ""))
    did_txt = (f"DiD(K={kmain}) {_g(did_main['estimate'])} [{_g(did_main['ci_low'])}, {_g(did_main['ci_high'])}]"
               if did_main else "")
    bonf = ""
    if main is None and n_doses > 1 and np.isfinite(bl[kmain]["p_one_sided"]):
        bonf = (f"Bonferroni over {n_doses} doses: one-sided p of dlam {_g(min(1.0, n_doses * bl[kmain]['p_one_sided']), 3)}"
                + (f", of DiD {_g(min(1.0, n_doses * did_main['p_one_sided']), 3)}" if did_main else ""))
    note = _join(audit_note, k_note, did_txt, rand_note, "C1 only (position-matched to K=0); K=0 reference = C1 site",
                 dose_note, bonf, gate_note, ctx.split_note, caveat_for("P"))
    rows.append(make_row(f"P{sfx}", f"{PRIMARY_NAME}, K={kmain}, eps={_eps_key(eps)}", role="primary", stage="D",
                         rule=PRIMARY_RULE, verdict=verdict, note=note, direction=1, source=src, min_effect=PRIMARY_MIN_EFFECT,
                         **_stat_fields(bl[kmain])))
    comp = "not_interpretable" if invalid else None
    labels = MIN_EFFECT_VERDICT
    for K in ks:
        st = min_effect_status(bl[K]["estimate"], bl[K]["ci_low"], bl[K]["ci_high"], PRIMARY_MIN_EFFECT, 1)
        v = comp or labels[st]
        rows.append(make_row(f"P.lang.K{K}{sfx}", f"dlam(lang_dom) K={K} vs K=0, C1, eps={_eps_key(eps)}",
                             role="primary_component", stage="D", direction=1, verdict=v, source=src,
                             rule=COMPONENT_LANG_RULE, min_effect=PRIMARY_MIN_EFFECT,
                             note=_join("not the main K" if K != kmain else "", ctx.split_note), **_stat_fields(bl[K])))
        if K in bd:
            b = bd[K]
            v = comp or labels[_did_status(b)]
            r = br[K]
            rows.append(make_row(f"P.did.K{K}{sfx}", f"DiD dlam(lang_dom) - dlam(rand) K={K}, C1, eps={_eps_key(eps)}",
                                 role="primary_component", stage="D", direction=1, verdict=v, source=src,
                                 rule=COMPONENT_DID_RULE,
                                 note=_join(f"dlam(rand) {_g(r['estimate'])} [{_g(r['ci_low'])}, {_g(r['ci_high'])}]", rand_note),
                                 **_stat_fields(b)))
    rows.append(make_row(f"P.mono{sfx}", f"monotonicity of lam(lang_dom) in K (C1), eps={_eps_key(eps)}",
                         role="primary_component", stage="D", estimate=float(mono == "pass"),
                         verdict=comp or {"pass": "supported", "fail": "not_supported", "open": "open"}[mono],
                         rule=PRIMARY_MONO_RULE, source=src,
                         n_families=bl[kmain]["n_families"], n_rows=sum(bl[K]["n_rows"] for K in ks),
                         note=_join("dlam by K: " + ", ".join(f"K{K} {_g(bl[K]['estimate'])}" for K in ks),
                                    f"point estimates monotone (reading (a), not the rule): {mono_point}",
                                    "increments " + "; ".join(inc_notes) if inc_notes else "",
                                    f"bootstrap P(monotone) {_g(p_mono, 3)}" if np.isfinite(p_mono) else "")))
    # ---- deviation rows: C2 / C3 / seg (injection position != pre-registered position)
    for Cc in ("C2", "C3", "seg"):
        kc = sorted(int(k) for k in set(lang[(lang["K"] > 0) & (lang["C"].astype(str) == Cc)]["K"]))
        if not kc:
            continue
        K = max(kc)
        fl = family_delta(lang, K, Cc)
        b = _boot_fam(fl, ctx, 1)
        rows.append(make_row(f"P.dev.{Cc}.lang.K{K}{sfx}", f"deviation: dlam(lang_dom) K={K}, {Cc}, eps={_eps_key(eps)}",
                             role="primary_deviation", stage="D", direction=1, verdict="not_interpretable", source=src,
                             rule="reported only (injection position differs from the pre-registered one)",
                             note=caveat_for("P.dev"), **_stat_fields(b)))
        if len(rand):
            fr = family_delta(rand, K, Cc)
            b = _boot_fam(_did(fl, fr), ctx, 1)
            r = _boot_fam(fr, ctx, 1)
            rows.append(make_row(f"P.dev.{Cc}.did.K{K}{sfx}", f"deviation: DiD lang_dom - rand K={K}, {Cc}, eps={_eps_key(eps)}",
                                 role="primary_deviation", stage="D", direction=1, verdict="not_interpretable", source=src,
                                 rule="reported only (injection position differs from the pre-registered one)",
                                 note=_join(f"dlam(rand) {_g(r['estimate'])} [{_g(r['ci_low'])}, {_g(r['ci_high'])}]",
                                            caveat_for("P.dev")), **_stat_fields(b)))
    return rows


# --------------------------------------------------------------------------------------
# token-language control of the primary endpoint (PREREGISTRATION 6.11, stage-D control run)
# --------------------------------------------------------------------------------------
#: Columns of the control run's ``damage_ctl``: the damage columns plus the site bookkeeping.
DAMAGE_CTL_COLS = DAMAGE_COLS + ["inject_pos", "inject_lang", "inject_unit", "inject_token", "inject_token_id", "site_gap"]
CTL_SITE_COLS = ["inject_lang", "inject_unit", "inject_token", "site_gap"]
#: Injections of the control run -> the unit whose last token (".") is the site.
CTL_SITES = {"u9": 9, "u10": 10}
CTL_TOKEN = "."
CTL1_C = ("C2", "C3")       # control 1: C2 and C3 pooled (unit 9 is DDE / DEE: the "." closes an English clause)
CTL3_C = "u9D"              # control 3: only unit 9 German (unit mask EEEEEEEEDE, K = 2)
CTL_KS = (2, 4, 8)
CTL_REP_TOL = 1e-3          # replication of the pre-registered site: |lam_ctl - lam_main| <= 1e-3
#: The three controls: (site, C level(s), K levels, label); controls 1 and 2 decide, 3 is reported.
CTL_ARMS = {1: ("u9", CTL1_C, CTL_KS, "C2 + C3 pooled at u9 (end of unit 9)"),
            2: ("u10", PRIMARY_C, CTL_KS, "C1 at u10 (end of unit 10)"),
            3: ("u9", CTL3_C, (2,), "u9D (only unit 9 German) at u9 (end of unit 9)")}
#: Outcome per dose <-> verdict of ``P.ctl``.
CTL_OUTCOME_VERDICT = {"Wechsel": "supported", "lokal": "not_supported", "offen": "open"}
CTL_VERDICT_OUTCOME = {v: k for k, v in CTL_OUTCOME_VERDICT.items()}
CTL_NAME = "token-language control of P (PREREGISTRATION 6.11): switching effect at '.' sites that close an English clause"
CTL_AUDIT_NAME = "D1 control-run injection audit: written-vector relative error (damage_ctl)"
CTL_RULE = ("per dose (PREREGISTRATION 6.11): 'Wechsel' (supported) iff control 1 (C2 + C3, last token of unit 9) and control 2 "
            "(C1, last token of unit 10) both show the effect at K=8 (ci_low of dlam(lang_dom) >= 0.05 and the CI of the DiD "
            "against rand at the same site > 0); 'lokal' (not_supported) iff ci_high of dlam(lang_dom, K=8) < 0.05 in both; "
            "'offen' (open) otherwise; control 3 (u9D) is reported only; not_interpretable when an audit of the control run "
            "fails; outside the BH group.  On P at the same dose: 'lokal' makes a supported or open P not_interpretable, 'offen' "
            "or no valid control run caps it at open, 'Wechsel' leaves it unchanged")
CTL_ARM_RULE = ("control of PREREGISTRATION 6.11: shows the effect (supported) iff the 95% CI of dlam(lang_dom, K=8) vs K=0 at "
                "the same site lies at or above the minimum effect (ci_low >= 0.05) AND the CI of the DiD (lang_dom - rand) "
                "> 0; excludes it iff ci_high of dlam(lang_dom) < 0.05; not_supported iff a component fails; open otherwise; "
                "monotonicity in K is not required (K=2 and 4 reported)")
CTL3_RULE = ("reported only, never in the outcome (PREREGISTRATION 6.11): the control rule at K=2 (u9D: only unit 9 German, two "
             "switches) vs K=0 at the last token of unit 9")


def _parse_utc(ts) -> Optional[_dt.datetime]:
    """An ISO-8601 manifest timestamp as an aware UTC datetime (``Z`` accepted, a naive time
    read as UTC); None when unreadable."""
    s = str(ts or "").strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        t = _dt.datetime.fromisoformat(s)
    except ValueError:
        return None
    return t.replace(tzinfo=_dt.timezone.utc) if t.tzinfo is None else t.astimezone(_dt.timezone.utc)


def token_control_rows(ctx: EvalContext) -> list[dict]:
    """Token-language control of the primary endpoint (PREREGISTRATION 6.11) from ``damage_ctl``
    of the control run (key ``D_ctl``): the statistic and the rules of P (confirmation set,
    family means, family cluster bootstrap, minimum effect 0.05 by the interval, DiD against
    rand) at the unit-defined sites ``u9`` / ``u10`` (the "." ending unit 9 / unit 10), each
    against K = 0 at the same site of the same run.  Control 1 = C2 and C3 pooled at u9,
    control 2 = C1 at u10 (K = 8 main, 2 and 4 reported), control 3 = ``u9D`` at u9 (reported
    only).  Per pre-registered dose (suffix as P's) ``P.ctl`` holds the outcome "Wechsel" /
    "lokal" / "offen".  ``audit.D1ctl`` (write audit), ``.rep`` (replication of the main run at
    the pre-registered site), ``.site`` and ``.run`` decide the run's validity: an audit that
    does not pass makes every ``P.ctl*`` row ``not_interpretable``.  No row enters the BH group;
    :func:`apply_token_control` applies the outcome to P."""
    run = ctx.tables.run("D_ctl")
    if run is None:
        return [_no_run("P.ctl", CTL_NAME, "D", "D_ctl", ctx, role="primary_control", rule=CTL_RULE)]
    dmg = ctx.tables.get("D_ctl", "damage_ctl", DAMAGE_CTL_COLS)
    src = ctx.tables.source("D_ctl", "damage_ctl")
    run_audit = _ctl_run_audit(ctx, run)
    miss = _missing_cols(dmg, DAMAGE_COLS)
    if miss:
        return [_no_table("P.ctl", CTL_NAME, "D", "damage_ctl", ctx, key="D_ctl", cols=miss if dmg is not None else (),
                          role="primary_control", rule=CTL_RULE),
                not_tested("audit.D1ctl", CTL_AUDIT_NAME, "D", "damage_ctl table unusable", role="audit"), run_audit]
    # ---- validity of the control run (PREREGISTRATION 6.11 (a)-(d)), over the whole table
    a_all = audit_summary(dmg["audit_rel_err"])
    by_inj = "; ".join(f"{inj}: max {_g(audit_summary(g['audit_rel_err'])['max'], 3)}"
                       for inj, g in dmg.groupby("inject", sort=True))
    audits = [audit_row("audit.D1ctl", CTL_AUDIT_NAME, "D", a_all, src, by_inj), _ctl_replication_audit(ctx, dmg, src),
              _ctl_site_audit(dmg, src), run_audit]
    # ---- confirmation set, per site
    d = ctx.split(_standard(dmg)).copy()
    d = d[np.isfinite(pd.to_numeric(d["lam"], errors="coerce"))]
    d["eps"] = d["eps"].astype(float).round(10)
    d["K"] = d["K"].astype(int)
    inj = d["inject"].astype(str)
    lang = {s: _variant_lam(d[inj == s], "lang_dom") for s in CTL_SITES}
    rand = {s: _variant_lam(d[inj == s], "rand") for s in CTL_SITES}
    doses = sorted(set(lang["u9"]["eps"]) | set(lang["u10"]["eps"]))
    prereg = [e for e in doses if any(np.isclose(e, p) for p in PREREG_DOSES)]
    explo = [e for e in doses if e not in prereg]
    missing = [p for p in PREREG_DOSES if not any(np.isclose(p, e) for e in doses)]
    drun = ctx.tables.run("D")
    frozen = _frozen_dose(ctx, drun) if drun is not None else ctx.frozen_dose     # the suffix rule of P
    main = next((e for e in prereg if frozen is not None and np.isclose(e, frozen)), None)
    dose_note = _join(f"pre-registered doses evaluated: {', '.join(_eps_key(e) for e in prereg) or 'none'}",
                      f"missing: {', '.join(_eps_key(e) for e in missing)}" if missing else "",
                      f"exploratory doses present (not evaluated): {', '.join(_eps_key(e) for e in explo)}" if explo else "",
                      "no DEV-frozen dose recorded: every pre-registered dose is reported, none is the main dose"
                      if frozen is None else
                      f"frozen dose {_eps_key(frozen)} not in the control run: every dose reported, no main dose"
                      if main is None else "")
    u9d_note = _u9d_note(ctx)
    rows: list[dict] = []
    for eps in prereg:
        rows.extend(_token_control_dose(ctx, {s: v[v["eps"] == eps] for s, v in lang.items()},
                                        {s: v[v["eps"] == eps] for s, v in rand.items()}, eps, main, dose_note, u9d_note,
                                        src))
    if not prereg:
        rows.append(not_tested("P.ctl", CTL_NAME, "D",
                               _join("no lang_dom rows of the confirmation set at a pre-registered dose (u9 / u10) in "
                                     "damage_ctl", dose_note, ctx.split_note), role="primary_control", rule=CTL_RULE))
    failed = [a for a in audits if a["verdict"] != "supported"]
    if failed:
        why = ", ".join(f"{a['id']} {a['verdict']}" for a in failed)
        for r in rows:
            lit = r["verdict"]
            if re.fullmatch(r"P\.ctl(@.+)?", r["id"]) and lit in CTL_VERDICT_OUTCOME:
                lit = f"{CTL_VERDICT_OUTCOME[lit]} ({lit})"
            r.update(verdict="not_interpretable",
                     deviation_note=_join(f"control run INVALID by its audits ({why}): no valid control (PREREGISTRATION "
                                          f"6.11); rule outcome ignoring the audits: {lit}", r["deviation_note"]))
    return audits + rows


def _token_control_dose(ctx: EvalContext, lang: dict, rand: dict, eps: float, main: Optional[float], dose_note: str,
                        u9d_note: str, src: str) -> list[dict]:
    """Rows of one dose: the outcome ``P.ctl{sfx}`` and the three controls with their components."""
    sfx = "" if main is not None and np.isclose(eps, main) else f"@{_eps_key(eps)}"
    arms = {j: _ctl_arm(ctx, j, lang[CTL_ARMS[j][0]], rand[CTL_ARMS[j][0]], eps, sfx, src, u9d_note if j == 3 else "")
            for j in CTL_ARMS}
    a1, a2 = arms[1], arms[2]
    if a1["verdict"] == "supported" and a2["verdict"] == "supported":
        outcome, why = "Wechsel", "controls 1 and 2 show the effect at K=8"
    elif a1["lang"] == "fail" and a2["lang"] == "fail":
        outcome, why = "lokal", "controls 1 and 2 exclude it (ci_high of dlam(lang_dom, K=8) < 0.05)"
    else:
        outcome, why = "offen", "the controls neither both show the effect nor both exclude it"
    summary = make_row(f"P.ctl{sfx}", f"{CTL_NAME}, eps={_eps_key(eps)}", role="primary_control", stage="D", rule=CTL_RULE,
                       verdict=CTL_OUTCOME_VERDICT[outcome], source=src,
                       n_families=min(a1["n_families"], a2["n_families"]), n_rows=a1["n_rows"] + a2["n_rows"],
                       note=_join(f"outcome {outcome}: {why}", a1["summary"], a2["summary"],
                                  arms[3]["summary"] + " (reported only)", dose_note, ctx.split_note, caveat_for("P.ctl")))
    return [summary] + a1["rows"] + a2["rows"] + arms[3]["rows"]


def _ctl_arm(ctx: EvalContext, j: int, lang: pd.DataFrame, rand: pd.DataFrame, eps: float, sfx: str, src: str,
             note: str = "") -> dict:
    """Control ``j`` (:data:`CTL_ARMS`) at one dose: dlam(lang_dom), dlam(rand) and their DiD per K
    against K = 0 at the same site (P's statistic, C levels pooled), the row ``P.ctl{j}`` with
    ``primary_verdict(lang, did, "pass")`` at the main K (8; 2 for control 3) and the component
    rows; returns the rows, the verdict, the main-K lang / DiD status, the counts and a summary."""
    site, Cc, ks, label = CTL_ARMS[j]
    kmain = PRIMARY_K if PRIMARY_K in ks else ks[-1]
    e = _eps_key(eps)
    bl, br, bd = {}, {}, {}
    for K in ks:
        fl = family_delta(lang, K, Cc)
        bl[K] = _boot_fam(fl, ctx, 1)
        if len(rand):
            fr = family_delta(rand, K, Cc)
            br[K] = _boot_fam(fr, ctx, 1)
            bd[K] = _boot_fam(_did(fl, fr), ctx, 1)
    bd = {K: b for K, b in bd.items() if b["n_families"]}            # a DiD needs families on both sides
    st = {K: min_effect_status(bl[K]["estimate"], bl[K]["ci_low"], bl[K]["ci_high"], PRIMARY_MIN_EFFECT, 1) for K in ks}
    dst = {K: _did_status(bd[K]) if K in bd else "nan" for K in ks}
    verdict = primary_verdict(st[kmain], dst[kmain], "pass")
    rand_note = _rand_note(rand)
    rtxt = {K: f"dlam(rand) {_ci_txt(br[K])}" if K in br and br[K]["n_families"] else "" for K in ks}
    did_main = f"DiD(K={kmain}) {_ci_txt(bd[kmain])}" if kmain in bd else ""
    only = "reported only: never in the outcome (PREREGISTRATION 6.11)" if j == 3 else ""
    ref = f"K=0 reference at the same site ({site}, last token of unit {CTL_SITES[site]}) of the same run"
    empty = {K: "" if bl[K]["n_families"] else f"no K={K} variants of these cells or no K=0 reference at {site} in the "
                                               "confirmation set" for K in ks}
    rid = f"P.ctl{j}"
    rows = [make_row(f"{rid}{sfx}", f"control {j}: dlam(lang_dom) K={kmain} vs K=0, {label}, eps={e}", role="primary_control",
                     stage="D", rule=CTL3_RULE if j == 3 else CTL_ARM_RULE, verdict=verdict, direction=1, source=src,
                     min_effect=PRIMARY_MIN_EFFECT,
                     note=_join(only, empty[kmain],
                                f"dlam(lang_dom, K={kmain}) {st[kmain]}, DiD {dst[kmain]} (no monotonicity required)",
                                did_main, rtxt[kmain], rand_note, ref, note, ctx.split_note, caveat_for("P.ctl")),
                     **_stat_fields(bl[kmain]))]
    for K in ks:
        kid = "" if j == 3 else f".K{K}"
        main_k = "" if K == kmain else "not the main K"
        rows.append(make_row(f"{rid}.lang{kid}{sfx}", f"control {j}: dlam(lang_dom) K={K} vs K=0, {label}, eps={e}",
                             role="primary_control", stage="D", direction=1, verdict=MIN_EFFECT_VERDICT[st[K]], source=src,
                             rule=COMPONENT_LANG_RULE, min_effect=PRIMARY_MIN_EFFECT,
                             note=_join(only, empty[K], main_k, rtxt[K], ctx.split_note), **_stat_fields(bl[K])))
        rows.append(make_row(f"{rid}.did{kid}{sfx}", f"control {j}: DiD dlam(lang_dom) - dlam(rand) K={K}, {label}, eps={e}",
                             role="primary_control", stage="D", direction=1, verdict=MIN_EFFECT_VERDICT[dst[K]], source=src,
                             rule=COMPONENT_DID_RULE, note=_join(only, empty[K], main_k, rtxt[K], rand_note),
                             **(_stat_fields(bd[K]) if K in bd else {})))
    lang_txt = f"{_ci_txt(bl[kmain])} {st[kmain]}" if bl[kmain]["n_families"] else "not computable"
    did_txt = f"{_ci_txt(bd[kmain])} {dst[kmain]}" if kmain in bd else "not computable"
    summary = f"control {j}, {label}: dlam(lang_dom, K={kmain}) {lang_txt}, DiD {did_txt} -> {verdict}"
    return dict(rows=rows, verdict=verdict, lang=st[kmain], did=dst[kmain], n_families=bl[kmain]["n_families"],
                n_rows=bl[kmain]["n_rows"], summary=summary)


def _u9d_note(ctx: EvalContext) -> str:
    """Section-4 exclusions of the u9D variants of the confirmation set (``damage_ctl_checks``)."""
    chk = ctx.tables.get("D_ctl", "damage_ctl_checks")
    if _missing_cols(chk, ["family", "formulation", "C", "excluded"]):
        return ""
    c = ctx.split(chk[chk["C"].astype(str) == CTL3_C])
    if c.empty:
        return ""
    n_ex = int((c["excluded"].map(_flag) == True).sum())  # noqa: E712
    return f"u9D variants excluded by section 4 (damage_ctl_checks): {n_ex} of {len(c)}"


def _ctl_replication_audit(ctx: EvalContext, dmg: pd.DataFrame, src: str) -> dict:
    """``audit.D1ctl.rep`` (PREREGISTRATION 6.11 (b)): every control row at the pre-registered
    site (u9, C1, K 0/2/4/8) has the main run's ``depth`` row of the same tag, direction and eps
    at the same position, and its lam lies within :data:`CTL_REP_TOL` of it; the estimate is
    the largest absolute difference."""
    name = "D1 control run: replication of the pre-registered site (u9, C1) against the main stage-D run"
    rule = ("control run valid iff every control row at u9 / C1 (K 0/2/4/8) has the main run's depth row (same tag, direction, "
            "eps) at the same inject_pos and |lam_ctl - lam_main| <= 1e-3 (PREREGISTRATION 6.11 (b))")

    def row(verdict: str, note: str, **kw) -> dict:
        invalid = "" if verdict == "supported" else "control run not validated (PREREGISTRATION 6.11 (b))"
        return make_row("audit.D1ctl.rep", name, role="audit", stage="D", rule=rule, verdict=verdict, source=src,
                        note=_join(note, invalid), **kw)

    c = dmg[(dmg["inject"].astype(str) == "u9") & (dmg["C"].astype(str) == PRIMARY_C)]
    if c.empty:
        return row("not_tested", "no u9 / C1 rows in damage_ctl")
    if "inject_pos" not in c.columns:
        return row("not_tested", "inject_pos missing in damage_ctl")
    drun = ctx.tables.run("D")
    if drun is None:
        return row("not_tested", "no main stage-D run under the given roots")
    need = ["tag", "inject", "direction", "eps", "lam", "inject_pos"]
    m = ctx.tables.get("D", "damage", DAMAGE_READ_COLS)
    miss = _missing_cols(m, need)
    if miss:
        return row("not_tested", f"main run {drun.path}: "
                                 + ("damage table missing" if m is None else f"columns {miss} missing in damage"))
    key = ["tag", "direction", "eps"]
    c = c[key + ["K", "lam", "inject_pos"]].copy()
    m = m[(m["inject"].astype(str) == "depth") & m["tag"].astype(str).isin(set(c["tag"].astype(str)))]
    m = m[key + ["lam", "inject_pos"]].copy()
    for x in (c, m):
        x["tag"], x["direction"] = x["tag"].astype(str), x["direction"].astype(str)
        x["eps"] = x["eps"].astype(float).round(10)
    j = c.merge(m, on=key, how="left", suffixes=("_ctl", "_main"), indicator=True)
    num = lambda col: pd.to_numeric(j[col], errors="coerce").to_numpy(dtype=float)  # noqa: E731
    found = (j["_merge"] == "both").to_numpy()
    diff = np.abs(num("lam_ctl") - num("lam_main"))
    same_pos = num("inject_pos_ctl") == num("inject_pos_main")
    n_missing = int((~found).sum())
    n_pos = int((found & ~same_pos).sum())
    n_over = int((found & ~(diff <= CTL_REP_TOL)).sum())          # a NaN difference is a mismatch
    fin = diff[found & np.isfinite(diff)]
    est = float(fin.max()) if fin.size else np.nan
    ok = n_missing == n_pos == n_over == 0
    ks = ", ".join(str(k) for k in sorted(set(c["K"].astype(int))))
    note = _join(f"{len(c)} control rows at u9 / C1 (K {ks}) vs the main run's depth rows: max |lam_ctl - lam_main| "
                 f"{_g(est, 3)} (tolerance 1e-3)",
                 f"missing in the main run: {n_missing}" if n_missing else "",
                 f"different inject_pos: {n_pos}" if n_pos else "",
                 f"|difference| > 1e-3 or not finite: {n_over}" if n_over else "",
                 f"main run {drun.path}")
    return row("supported" if ok else "not_supported", note, estimate=est, n_rows=len(c))


def _ctl_site_audit(dmg: pd.DataFrame, src: str) -> dict:
    """``audit.D1ctl.site`` (PREREGISTRATION 6.11 (c)): every row injects at ".", its language
    mark is E at u10 and at u9 for K = 0 and C2 / C3, D at u9 for C1 (K > 0) and u9D, its unit
    is 9 at u9 and 10 at u10, and ``site_gap`` (tail_start - inject_pos) is the formulation's
    K = 0 gap at u9 and 1 at u10.  The estimate is the number of rows that break a rule."""
    name = "D1 control run: injection sites ('.' token, language mark, unit, distance to the tail)"
    rule = ("control run valid iff every row injects at the token '.', inject_lang is E at u10 and at u9 for K=0 and C2 / C3 "
            "and D at u9 for C1 (K>0) and u9D, inject_unit is 9 at u9 and 10 at u10, and site_gap (tail_start - inject_pos) "
            "equals the formulation's K=0 gap at u9 and 1 at u10 (PREREGISTRATION 6.11 (c))")
    need = ["family", "formulation", "K", "C", "inject"] + CTL_SITE_COLS
    miss = _missing_cols(dmg, need)
    if miss or dmg.empty:
        why = f"columns {miss} missing in damage_ctl" if miss else "damage_ctl is empty"
        return make_row("audit.D1ctl.site", name, role="audit", stage="D", rule=rule, verdict="not_tested", source=src,
                        note=_join(why, "control run not validated (PREREGISTRATION 6.11 (c))"))
    x = dmg.assign(family=dmg["family"].astype(str), formulation=dmg["formulation"].astype(str))
    inj, Cc = x["inject"].astype(str).to_numpy(), x["C"].astype(str).to_numpy()
    K = pd.to_numeric(x["K"], errors="coerce").to_numpy(dtype=float)
    u9, u10 = inj == "u9", inj == "u10"
    english = u10 | (u9 & ((K == 0) | np.isin(Cc, CTL1_C)))
    german = u9 & (((Cc == PRIMARY_C) & (K > 0)) | (Cc == CTL3_C))
    exp_lang = np.where(english, "E", np.where(german, "D", ""))
    cell = exp_lang != ""                                       # a row inside the control design
    bad_tok = x["inject_token"].astype(str).to_numpy() != CTL_TOKEN
    bad_lang = cell & (x["inject_lang"].astype(str).to_numpy() != exp_lang)
    unit = pd.to_numeric(x["inject_unit"], errors="coerce").to_numpy(dtype=float)
    bad_unit = ~(unit == np.where(u9, 9.0, np.where(u10, 10.0, np.nan)))
    gap = pd.to_numeric(x["site_gap"], errors="coerce").to_numpy(dtype=float)
    ref = pd.Series(gap[u9 & (K == 0)]).groupby([x["family"].to_numpy()[u9 & (K == 0)],
                                                  x["formulation"].to_numpy()[u9 & (K == 0)]]).first().to_dict()
    exp_gap = np.array([1.0 if b10 else ref.get((f, fo), np.nan) if b9 else np.nan
                        for b9, b10, f, fo in zip(u9, u10, x["family"], x["formulation"])], dtype=float)
    bad_gap = ~(gap == exp_gap)
    bad = bad_tok | bad_lang | bad_unit | bad_gap | ~cell
    ids = ""
    if "inject_token_id" in x.columns:
        tid = sorted({int(t) for t in pd.to_numeric(x["inject_token_id"], errors="coerce").dropna()})
        ids = f"token ids at the sites: {', '.join(str(t) for t in tid[:5])}" + (" ..." if len(tid) > 5 else "")
    note = _join(f"{len(x)} rows; rows breaking a rule: {int(bad.sum())} (token != '.': {int(bad_tok.sum())}, language mark: "
                 f"{int(bad_lang.sum())}, unit: {int(bad_unit.sum())}, site_gap: {int(bad_gap.sum())}, outside the control "
                 f"cells: {int((~cell).sum())})", ids,
                 "control run not validated (PREREGISTRATION 6.11 (c))" if bad.any() else "")
    return make_row("audit.D1ctl.site", name, role="audit", stage="D", rule=rule, estimate=float(bad.sum()), n_rows=len(x),
                    verdict="not_supported" if bad.any() else "supported", note=note, source=src)


def _ctl_run_audit(ctx: EvalContext, run: StageRun) -> dict:
    """``audit.D1ctl.run`` (PREREGISTRATION 6.11 (d)): the manifest carries ``control =
    "token_language"`` and ``amendment = "6.11"``, the run is younger than the amendment commit
    (:data:`AMENDMENT_611`) and is no smoke / limited run."""
    name = "D1 control run: manifest marks and age"
    rule = (f"control run valid iff its manifest says control = '{CTL_CONTROL}' and amendment = '{AMENDMENT_611['id']}', it "
            f"is younger than the commit of amendment 6.11 ({AMENDMENT_611['commit'][:7]}, {AMENDMENT_611['utc']}) and it is "
            "no smoke / limited run (PREREGISTRATION 6.11 (d))")
    m = run.manifest
    bad = []
    if m.get("control") != CTL_CONTROL:
        bad.append(f"control {m.get('control')!r} instead of {CTL_CONTROL!r}")
    if str(m.get("amendment", "")) != AMENDMENT_611["id"]:
        bad.append(f"amendment {m.get('amendment')!r} instead of {AMENDMENT_611['id']!r}")
    t, t611 = _parse_utc(run.timestamp), _parse_utc(AMENDMENT_611["utc"])
    if t is None:
        bad.append(f"timestamp {run.timestamp!r} unreadable")
    elif t <= t611:
        bad.append(f"the run ({run.timestamp}) predates the commit of amendment 6.11 ({AMENDMENT_611['commit'][:7]}, "
                   f"{AMENDMENT_611['utc']})")
    if run.reduced:
        bad.append("smoke / limited run")
    mr = m.get("main_run") if isinstance(m.get("main_run"), dict) else {}
    drun = ctx.tables.run("D")
    note = _join("; ".join(bad), "control run not validated (PREREGISTRATION 6.11 (d))" if bad else "",
                 f"run {run.timestamp or '?'}, git {str(m.get('git_sha') or '?')[:7]}",
                 (f"main run in the manifest: {mr.get('path', '?')} ({mr.get('timestamp', '?')}, git "
                  f"{str(mr.get('git_sha') or '?')[:7]})") if mr else "the manifest names no main run",
                 f"main stage-D run of this evaluation: {drun.path}" if drun is not None else "no main stage-D run")
    return make_row("audit.D1ctl.run", name, role="audit", stage="D", rule=rule, verdict="not_supported" if bad else "supported",
                    n_rows=1, note=note, source=f"{run.label()}/{MANIFEST_NAME}")


def apply_token_control(df: pd.DataFrame) -> pd.DataFrame:
    """PREREGISTRATION 6.11 on the primary rows ``P`` / ``P@<eps>`` (before :func:`apply_bh`):
    the outcome ``P.ctl{sfx}`` of the same dose leaves P unchanged ("Wechsel", supported), makes
    a supported or open P ``not_interpretable`` ("lokal", not_supported; any other verdict
    stays) or caps it at "open" (:func:`_cap_open`: "offen", or no valid control run, i.e.
    ``P.ctl{sfx}`` missing, not_tested or not_interpretable).  The gate never improves a
    verdict; P's note gets the outcome and the rule outcome without the control."""
    df = df.copy()
    ids = df["id"].astype(str)
    ctl = {i: str(v) for i, v in zip(ids, df["verdict"]) if i.startswith("P.ctl")}
    vcol, ncol = df.columns.get_loc("verdict"), df.columns.get_loc("deviation_note")
    for i in np.flatnonzero(ids.str.fullmatch(r"P(@.+)?").to_numpy(dtype=bool)):
        prev = str(df.iat[i, vcol])
        c = ctl.get(f"P.ctl{ids.iloc[i][1:]}")
        if c == "supported":
            new = prev
        elif c == "not_supported":
            new = "not_interpretable" if prev in ("supported", "open") else prev
        else:
            new = _cap_open(prev)
        note = df.iat[i, ncol]
        df.iat[i, vcol] = new
        df.iat[i, ncol] = _join(note if isinstance(note, str) else "",
                                f"token-language control (PREREGISTRATION 6.11): "
                                f"{CTL_VERDICT_OUTCOME.get(c, 'no valid control run')}; rule outcome without the control: {prev}")
    return df


def c_audit_row(ctx: EvalContext) -> tuple[dict, bool]:
    """``audit.C`` from stage C ``grid`` (returns the row and whether the run is invalid)."""
    name = "C grid injection audit: written-vector relative error (stage C grid)"
    if ctx.tables.run("C") is None:
        return _no_run("audit.C", name, "C", "C", ctx, role="audit"), False
    g = ctx.tables.get("C", "grid", ["eps", "audit_rel_err"])
    if _missing_cols(g, ["audit_rel_err"]):
        return _no_table("audit.C", name, "C", "grid", ctx, role="audit"), False
    s = audit_summary(g["audit_rel_err"])
    extra = ""
    if "eps" in g.columns and s["n_over"]:
        over = g[pd.to_numeric(g["audit_rel_err"], errors="coerce") > AUDIT_MAX_REL_ERR]
        extra = "doses with rows > 1e-4: " + ", ".join(_eps_key(e) for e in sorted(set(over["eps"].astype(float))))
    return audit_row("audit.C", name, "C", s, ctx.tables.source("C", "grid"), extra), s["n_over"] > 0


# --------------------------------------------------------------------------------------
# secondary 1-6 (stages C, D)
# --------------------------------------------------------------------------------------
def _boot_row(id: str, name: str, df: pd.DataFrame, value_col: str, ctx: EvalContext, *, stage: str, direction: int = 0,
              role: str = "secondary", bh_group: bool = True, note: str = "", source: str = "", rule: str = "") -> dict:
    sub = df[np.isfinite(pd.to_numeric(df[value_col], errors="coerce"))] if len(df) else df
    if sub.empty:
        return not_tested(id, name, stage, _join("no rows of the confirmation set", note), role=role,
                          rule=rule or _rule_secondary(direction))
    b = family_boot(sub, value_col, ctx.n_boot, ctx.seed, direction=direction)
    # provisional verdict from the uncorrected interval; apply_bh() finalises the BH rows
    v = secondary_verdict(b["estimate"], b["ci_low"], b["ci_high"], np.nan, direction)
    return make_row(id, name, role=role, stage=stage, direction=direction, bh_group=bh_group, source=source, verdict=v,
                    rule=rule or _rule_secondary(direction), note=_join(note, ctx.split_note, caveat_for(id)), **_stat_fields(b))


def _paired_row(id: str, name: str, df: pd.DataFrame, value_col: str, cond_col: str, a, b, ctx: EvalContext, *, stage: str,
                direction: int = 0, role: str = "secondary", bh_group: bool = True, note: str = "", source: str = "") -> dict:
    pr = df[df[cond_col].isin([a, b])]
    if pr.empty:
        return not_tested(id, name, stage, _join("no rows of the confirmation set", note), role=role, rule=_rule_secondary(direction))
    r = paired_contrast(pr, value_col, cond_col, a, b, ["family", "formulation"], "family", n_boot=ctx.n_boot,
                        seed=ctx.seed, stat_level="cluster")
    if r["n_clusters"] == 0:
        return not_tested(id, name, stage, _join("no matched pairs on the confirmation set", note), role=role,
                          rule=_rule_secondary(direction))
    return make_row(id, name, role=role, stage=stage, direction=direction, bh_group=bh_group, source=source,
                    rule=_rule_secondary(direction),
                    verdict=secondary_verdict(r["mean"], r["ci_low"], r["ci_high"], np.nan, direction),
                    note=_join(note, f"mean {a} {_g(r['mean_a'])}, {b} {_g(r['mean_b'])}, {r['n_pairs']} pairs",
                               ctx.split_note, caveat_for(id)),
                    estimate=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"], p_two_sided=r["p_boot"],
                    p_one_sided=p_one_sided(r["boot"], direction), n_families=r["n_clusters"], n_rows=r["n_pairs"])


def _contrast_T(ctx: EvalContext) -> Optional[pd.DataFrame]:
    con = ctx.tables.get("C", "contrasts", ["family", "formulation", "tag", "K", "C", "kind", "sign", "eps", "T"])
    if _missing_cols(con, ["family", "formulation", "tag", "K", "C", "eps", "T"]):
        return None
    t = _standard(con)
    t = t[np.isclose(t["eps"].astype(float), 0.0)]
    if "sign" in t.columns:
        t = t[t["sign"].astype(int) == 1]
    return ctx.split(t.drop_duplicates(["tag"])).assign(K=lambda x: x["K"].astype(int), C=lambda x: x["C"].astype(str))


def s1_s2_rows(ctx: EvalContext) -> list[dict]:
    n1, n2 = "S1 T(K=8, C1): logp0(K8C1) - logp0(K0)", "S2 K x C: T(8, C3) - T(8, C1) (paired)"
    if ctx.tables.run("C") is None:
        return [_no_run("S1", n1, "C", "C", ctx), _no_run("S2", n2, "C", "C", ctx)]
    t = _contrast_T(ctx)
    if t is None:
        return [_no_table("S1", n1, "C", "contrasts", ctx), _no_table("S2", n2, "C", "contrasts", ctx)]
    src = ctx.tables.source("C", "contrasts")
    rows = [_boot_row("S1", n1, t[(t["K"] == 8) & (t["C"] == "C1")], "T", ctx, stage="C", source=src,
                      note="T uses the unperturbed forward (no injection)")]
    k8 = t[(t["K"] == 8) & (t["C"].isin(["C1", "C3"]))]
    rows.append(_paired_row("S2", n2, k8, "T", "C", "C3", "C1", ctx, stage="C", source=src))
    return rows


def s3_rows(ctx: EvalContext) -> list[dict]:
    name = "S3 R_excess = R_ws+punct+boundary - R_random"
    if ctx.tables.run("C") is None:
        return [_no_run("S3", name, "C", "C", ctx, rule=_rule_secondary(1))]
    ac = ctx.tables.get("C", "attribution_contrasts")
    need = ["pair", "family", "formulation", "tag_mixed", "layer", "class", "R_excess"]
    if _missing_cols(ac, need):
        return [_no_table("S3", name, "C", "attribution_contrasts", ctx, rule=_rule_secondary(1))]
    src = ctx.tables.source("C", "attribution_contrasts")
    a = ctx.split(_standard(ac))
    a = a[a["class"].astype(str) == PREREG_PATCH_CLASS]
    if a.empty:
        return [not_tested("S3", name, "C", _join("no delim rows of the confirmation set", ctx.split_note),
                           rule=_rule_secondary(1))]
    att = ctx.tables.get("C", "attribution", ["pair", "tag_mixed", "layer", "class", "n_positions", "logp_mono",
                                              "logp_mixed"])
    denom = None
    small = ""
    if not _missing_cols(att, ["pair", "tag_mixed", "layer", "class", "logp_mono", "logp_mixed"]):
        x = att.copy()
        x["denom"] = x["logp_mono"].astype(float) - x["logp_mixed"].astype(float)
        denom = x[["pair", "tag_mixed", "layer", "class", "denom"]]
        dd = x[x["class"].astype(str) == PREREG_PATCH_CLASS].drop_duplicates(["pair", "tag_mixed"])
        dd = dd[dd["tag_mixed"].isin(set(a["tag_mixed"]))]
        if len(dd):
            small = (f"{int((dd['denom'].abs() < 0.05).sum())} of {len(dd)} test pairs have |logp_mono - logp_mixed| "
                     f"< 0.05 nats, {int((dd['denom'] < 0).sum())} have T > 0")
        if "n_positions" in x.columns:
            ws = x[x["class"].astype(str) == "ws"]
            if len(ws) and (pd.to_numeric(ws["n_positions"], errors="coerce").fillna(0) == 0).all():
                small = _join(small, "ws class empty in every pair")
    rows = []
    for (pair, layer), g in a.groupby(["pair", "layer"], sort=True):
        rows.append(_boot_row(f"S3.{pair}.L{int(layer)}", f"{name}, {pair}, layer {int(layer)}", g, "R_excess", ctx,
                              stage="C", direction=1, source=src, note=small))
    if denom is not None:
        m = a.merge(denom, on=["pair", "tag_mixed", "layer", "class"], how="left")
        # R_excess x |logp_mono - logp_mixed| = how many nats the delim patch moves log p(target) towards the
        # monolingual value beyond the random patch: no ratio, so a near-zero text effect cannot blow it up
        m["excess_nats"] = m["R_excess"].astype(float) * m["denom"].abs()
        for (pair, layer), g in m.groupby(["pair", "layer"], sort=True):
            rows.append(_boot_row(f"S3x.nats.{pair}.L{int(layer)}",
                                  f"exploratory: nats-based excess R_excess x |logp_mono - logp_mixed|, {pair}, layer {int(layer)}",
                                  g, "excess_nats", ctx, stage="C", direction=1, role="exploratory", bh_group=False,
                                  source=ctx.tables.source("C", "attribution"),
                                  note="exploratory (not pre-registered, not in the BH group): recovery towards the "
                                       "monolingual log p beyond the random patch, in nats (no ratio)"))
    return rows


def s4_rows(ctx: EvalContext) -> list[dict]:
    name = "S4 hysteresis area A_hyst(lang_dom) - A_hyst(restart null)"
    if ctx.tables.run("D") is None:
        return [_no_run("S4", name, "D", "D", ctx)]
    hy = ctx.tables.get("D", "hysteresis", ["family", "formulation", "tag", "K", "C", "kind", "direction", "area_minus_null"])
    if _missing_cols(hy, ["family", "formulation", "K", "direction", "area_minus_null"]):
        return [_no_table("S4", name, "D", "hysteresis", ctx)]
    src = ctx.tables.source("D", "hysteresis")
    h = ctx.split(_standard(hy))
    h = h[h["direction"].astype(str) == "lang_dom"]
    rows = [_boot_row("S4", f"{name} (core variants pooled)", h, "area_minus_null", ctx, stage="D", source=src)]
    for K, g in h.groupby(h["K"].astype(int), sort=True):
        rows.append(_boot_row(f"S4.K{K}", f"{name}, K={K} only", g, "area_minus_null", ctx, stage="D", role="sensitivity",
                              bh_group=False, source=src, note="sensitivity by K (uncorrected)"))
    return rows


def s5_rows(ctx: EvalContext) -> list[dict]:
    name = "S5 non-monotonicity index D2, K=8/C1 - K=0 (paired)"
    if ctx.tables.run("D") is None:
        return [_no_run("S5", name, "D", "D", ctx)]
    ls = ctx.tables.get("D", "lens_summary", ["family", "formulation", "tag", "K", "C", "kind", "nmi_last", "nmi_tail"])
    if _missing_cols(ls, ["family", "formulation", "K", "C", "nmi_last", "nmi_tail"]):
        return [_no_table("S5", name, "D", "lens_summary", ctx)]
    src = ctx.tables.source("D", "lens_summary")
    x = ctx.split(_standard(ls)).copy()
    x["cond"] = np.where((x["K"].astype(int) == 8) & (x["C"].astype(str) == "C1"), "K8C1",
                         np.where(x["K"].astype(int) == 0, "K0", ""))
    return [_paired_row(f"S5.{q}", f"{name}, {q}", x, q, "cond", "K8C1", "K0", ctx, stage="D", source=src,
                        note="readout not fixed by PREREGISTRATION 5: both readouts enter the BH group")
            for q in ("nmi_last", "nmi_tail")]


#: Test arm of contrast 6: ent in its State-2143 sign (the sign that gives the benefit there).
S6_TEST_SIGN = 1


def s6_rows(ctx: EvalContext, c_invalid: bool) -> list[dict]:
    """Contrast 6 on the cell stage C marks ``preregistered`` (``contrasts_boot``
    ``quantity="B_ent_minus_rand"``, ``K=-1``, ``C="all"``): B(ent) - mean B(rand) per variant,
    pooled over every standard variant of the confirmation set.  The test arm is ent in its
    State-2143 sign (+1), one-sided (H1: ent above random), one row per pre-registered dose in
    the BH group (no dose is frozen on DEV01-04 for 6: every dose is reported).  The reversed
    sign (-1) is the sign-reversal control of DESIGN 6 and the K = 0 cell (the cell evaluated
    before 27.10.2026) a sensitivity row; neither enters the BH group."""
    name = "S6 transfer: B(ent) - mean B(rand), every standard variant (paired per variant)"
    rule6 = ("ent in the State-2143 sign (+1), pooled over every standard variant (stage C's pre-registered cell); "
             + _rule_secondary(1) + "; every pre-registered dose is its own BH row")
    if ctx.tables.run("C") is None:
        return [_no_run("S6", name, "C", "C", ctx, rule=rule6)]
    gs = ctx.tables.get("C", "grid_summary", ["tag", "family", "formulation", "K", "C", "kind", "dir_family", "sign", "eps", "B",
                                              "n_dirs"])
    if _missing_cols(gs, ["tag", "family", "formulation", "K", "dir_family", "sign", "eps", "B"]):
        return [_no_table("S6", name, "C", "grid_summary", ctx, rule=rule6)]
    src = ctx.tables.source("C", "grid_summary")
    g = ctx.split(_standard(gs))
    g = g[g["dir_family"].astype(str).isin(["ent", "rand"])].copy()
    g["eps"] = g["eps"].astype(float).round(10)
    g["K"] = g["K"].astype(int)
    g = g[g["eps"] > 0]
    if g.empty:
        return [not_tested("S6", name, "C", _join("no ent / rand rows of the confirmation set", ctx.split_note), rule=rule6)]
    piv = g.pivot_table(index=["family", "formulation", "tag", "K", "sign", "eps"], columns="dir_family", values="B").reset_index()
    if "ent" not in piv.columns or "rand" not in piv.columns:
        return [not_tested("S6", name, "C", "ent or rand missing in grid_summary", rule=rule6)]
    piv["dB"] = piv["ent"] - piv["rand"]
    nd = g[g["dir_family"] == "rand"]["n_dirs"] if "n_dirs" in g.columns else pd.Series(dtype=float)
    rand_note = (f"rand = mean over {int(nd.min())} random directions" + (
        f" (fewer than {N_RAND_PREREG})" if nd.max() < N_RAND_PREREG else "")) if len(nd) else ""
    pool_note = f"pooled over {piv['tag'].nunique()} standard variants (K levels {', '.join(str(k) for k in sorted(set(piv['K'])))})"
    rows = []
    for (sign, eps), sub in piv.groupby(["sign", "eps"], sort=True):
        if not any(np.isclose(eps, p) for p in PREREG_DOSES):
            continue
        sg, ek = int(sign), _eps_key(eps)
        if sg == S6_TEST_SIGN:
            out = [_boot_row(f"S6.sign{sg:+d}.eps{ek}", f"{name}, sign {sg:+d}, eps={ek}", sub, "dB", ctx, stage="C",
                             direction=1, source=src, rule=rule6, note=_join(rand_note, pool_note)),
                   _boot_row(f"S6.K0.sign{sg:+d}.eps{ek}", f"{name}, K = 0 variants only, sign {sg:+d}, eps={ek}",
                             sub[sub["K"] == 0], "dB", ctx, stage="C", direction=1, role="sensitivity", bh_group=False,
                             source=src, note=_join("sensitivity: K = 0 variants only (uncorrected)", rand_note))]
        else:
            out = [_boot_row(f"S6.sign{sg:+d}.eps{ek}", f"control: {name}, sign {sg:+d} (reversed ent), eps={ek}", sub, "dB",
                             ctx, stage="C", direction=0, role="sensitivity", bh_group=False, source=src,
                             rule="sign-reversal control (DESIGN 6): reported two-sided, not a test of 6 (not in the BH group)",
                             note=_join("control: ent with the sign reversed (uncorrected)", rand_note, pool_note))]
        for r in out:
            if c_invalid and r["verdict"] != "not_tested":
                r.update(verdict="not_interpretable", bh_group=False,
                         deviation_note=_join("stage C run INVALID by the grid audit (section 4)", r["deviation_note"]))
        rows.extend(out)
    return rows or [not_tested("S6", name, "C", "no pre-registered dose in grid_summary", rule=rule6)]


# --------------------------------------------------------------------------------------
# stage G: gate 7, conditional 8-9
# --------------------------------------------------------------------------------------
def g_rows(ctx: EvalContext) -> list[dict]:
    n7 = "S7 G0 gate: d(A_E,A_D) < d(A_E,B_E) < d(other family) in >= 50% of the middle layers"
    rule7 = "gate (no test): passes iff core_order_ok in >= 50% of the middle snapshots; degenerate metric -> no verdict"
    run = ctx.tables.run("G")
    if run is None:
        return [_no_run("S7", n7, "G", "G", ctx, role="gate", rule=rule7),
                not_tested("S8", "S8 G1 contraction (conditional on G0)", "G", "no stage-G run", role="conditional"),
                not_tested("S9", "S9 T3 steering validity (conditional on G0)", "G", "no stage-G run", role="conditional")]
    gate = ctx.tables.get("G", "g0_gate")
    rows = []
    gate_state = None      # True passed, False failed, None no verdict
    if gate is None or gate.empty or "pass_middle" not in gate.columns:
        rows.append(_no_table("S7", n7, "G", "g0_gate", ctx, role="gate", rule=rule7))
    else:
        r = gate.iloc[0]
        deg = _flag(r.get("degenerate"))
        pm, pa = _flag(r.get("pass_middle")), _flag(r.get("pass_all"))
        if deg:
            verdict = "not_interpretable"
        elif pm is True:
            verdict, gate_state = "supported", True
        elif pm is False:
            verdict, gate_state = "gate_failed", False
        else:
            verdict = "not_interpretable"
        note = _join(f"middle {int(_f(r.get('n_ok_middle')) or 0)}/{int(_f(r.get('n_middle')) or 0)} ok, all layers "
                     f"{int(_f(r.get('n_ok')) or 0)}/{int(_f(r.get('n_layers')) or 0)} (pass_all {pa})",
                     f"degenerate {deg}; rank r {r.get('r')} (requested {r.get('r_requested')}), N {r.get('n_states')}",
                     caveat_for("S7"))
        rows.append(make_row("S7", n7, role="gate", stage="G", rule=rule7, verdict=verdict, note=note,
                             estimate=r.get("frac_ok_middle"), n_families=int(_f(r.get("n_families")) or 0),
                             n_rows=int(_f(r.get("n_states")) or 0), source=ctx.tables.source("G", "g0_gate")))
    cond_note = {True: "", False: "G0 failed: not interpreted (PREREGISTRATION 7)",
                 None: "G0 gave no verdict: not interpreted"}[gate_state]
    # ---- 8: G1 contraction (pre-registered rows), two-sided, BH group when G0 passes
    cb = ctx.tables.get("G", "contraction_boot")
    n8 = "S8 G1 contraction dlog-Vol(K=8,C1) - dlog-Vol(K=0) (mean) / dPR (last), k=5, n-matched"
    rule8 = "two-sided, sign open; supported iff the CI excludes 0 and BH q <= 0.05 (secondary group); interpreted only if G0 passes"
    if cb is None or _missing_cols(cb, ["quantity", "site", "preregistered", "mean", "ci_low", "ci_high"]):
        rows.append(_no_table("S8", n8, "G", "contraction_boot", ctx, role="conditional", rule=rule8))
    else:
        cb, snote = boot_split(cb, G_SPLIT_TEST_NOTE)
        pre = cb[cb["preregistered"].map(_flag) == True]  # noqa: E712
        for _, r in pre.iterrows():
            p2, p1 = table_p(r, 0)
            has_p = bool(np.isfinite(p2))
            lit = secondary_verdict(r["mean"], r["ci_low"], r["ci_high"], np.nan, 0)
            lit = lit if has_p else _cap_open(lit)
            v = lit if gate_state else "not_interpretable"
            rows.append(make_row(f"S8.{r['quantity']}.{r['site']}", f"{n8}: {r['quantity']} at {r['site']}", role="conditional",
                                 stage="G", rule=rule8, verdict=v, bh_group=bool(gate_state and has_p and lit != "not_tested"),
                                 estimate=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"], p_two_sided=p2, p_one_sided=p1,
                                 n_families=int(_f(r.get("n_clusters")) or 0), n_rows=int(_f(r.get("n_rows")) or 0),
                                 note=_join(cond_note, f"rule outcome ignoring the gate: {lit}" if not gate_state else "",
                                            "" if has_p else NO_P_NOTE, snote, _nboot_note(run), caveat_for("S8")),
                                 source=ctx.tables.source("G", "contraction_boot")))
        if pre.empty:
            rows.append(not_tested("S8", n8, "G", "no preregistered rows in contraction_boot", role="conditional", rule=rule8))
    # ---- 9: T3 steering validity at eps = 0.05, minimum effect by the interval, BH group when G0 passes
    sb = ctx.tables.get("G", "steering_boot")
    n9 = "S9 T3 steering validity: cos(steer lang_dom, mean(K8)-mean(K0)) at the max-displacement layer, eps=0.05"
    rule9 = ("one-sided, minimum effect cos 0.3 (interval rule): supported iff ci_low >= 0.3 and BH q <= 0.05 (secondary "
             "group); not_supported iff ci_high < 0.3; open otherwise; interpreted only if G0 passes")
    if sb is None or _missing_cols(sb, ["quantity", "inject_site", "eps", "mean", "ci_low", "ci_high"]):
        rows.append(_no_table("S9", n9, "G", "steering_boot", ctx, role="conditional", rule=rule9))
    else:
        sb, snote = boot_split(sb, G_SPLIT_TEST_NOTE)
        s = sb[(sb["quantity"].astype(str) == "cos_at_max_disp_layer") & np.isclose(sb["eps"].astype(float), G_PREREG_STEER_EPS)]
        for _, r in s.iterrows():
            est, lo, hi = _f(r["mean"]), _f(r["ci_low"]), _f(r["ci_high"])
            p2, p1 = table_p(r, 1)
            has_p = bool(np.isfinite(p2))
            lit = MIN_EFFECT_VERDICT[min_effect_status(est, lo, hi, G_PREREG_MIN_COS, 1)]
            lit = lit if has_p else _cap_open(lit)
            v = lit if gate_state else "not_interpretable"
            rows.append(make_row(f"S9.{r['inject_site']}", f"{n9}, injection {r['inject_site']}", role="conditional", stage="G",
                                 rule=rule9, verdict=v, estimate=est, ci_low=lo, ci_high=hi, direction=1,
                                 p_two_sided=p2, p_one_sided=p1, min_effect=G_PREREG_MIN_COS,
                                 bh_group=bool(gate_state and has_p and lit != "not_tested"),
                                 n_families=int(_f(r.get("n_clusters")) or 0), n_rows=int(_f(r.get("n_rows")) or 0),
                                 note=_join(cond_note, f"rule outcome ignoring the gate: {lit}" if not gate_state else "",
                                            "" if has_p else NO_P_NOTE, snote, _nboot_note(run), caveat_for("S9")),
                                 source=ctx.tables.source("G", "steering_boot")))
        if s.empty:
            rows.append(not_tested("S9", n9, "G", "no cos_at_max_disp_layer row at eps 0.05", role="conditional", rule=rule9))
    return rows


# --------------------------------------------------------------------------------------
# stage H: 10 (recomputed), 11 (copied)
# --------------------------------------------------------------------------------------
def _spearman(x, y) -> tuple[float, int]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 2:
        return float("nan"), n
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan"), n
    return float(np.corrcoef(rx, ry)[0, 1]), n


def _lambda_corr(ctx: EvalContext) -> Optional[pd.DataFrame]:
    """Per-start Spearman rho of the pre-registered quantities: ``lambda_corr`` as run, or
    recomputed from ``lambda_paths`` when the correlation table is missing."""
    corr = ctx.tables.get("H", "lambda_corr")
    if not _missing_cols(corr, ["family", "formulation", "discriminant", "quantity", "rho", "n"]):
        return corr[corr["discriminant"].astype(str) == H_PREREG_DISCRIMINANT]
    paths = ctx.tables.get("H", "lambda_paths")
    if _missing_cols(paths, ["tag", "family", "formulation", "discriminant", "lam"]):
        return None
    rows = []
    p = paths[paths["discriminant"].astype(str) == H_PREREG_DISCRIMINANT]
    for (tag, fam, form), g in p.groupby(["tag", "family", "formulation"], sort=True):
        for q in H_PREREG_QUANTITIES:
            if q in g.columns:
                rho, n = _spearman(g["lam"], pd.to_numeric(g[q], errors="coerce"))
                rows.append(dict(tag=tag, family=fam, formulation=form, discriminant=H_PREREG_DISCRIMINANT, quantity=q,
                                 rho=rho, n=n))
    return pd.DataFrame(rows, columns=["tag", "family", "formulation", "discriminant", "quantity", "rho", "n"])


def h_rows(ctx: EvalContext) -> list[dict]:
    names = {"sae_fvu": ("S10i", "S10 (i) Spearman rho(lambda, SAE FVU at the endpoint)"),
             "steps_to_cross_censored": ("S10ii", "S10 (ii) Spearman rho(lambda, censored steps to crossing)"),
             "mahal_to_ref_mean": ("S10iii", "S10 (iii) Spearman rho(lambda, Mahalanobis distance to the target cloud)")}
    run = ctx.tables.run("H")
    if run is None:
        rows = [_no_run(sid, nm, "H", "H", ctx, rule=_rule_secondary(H_PREREG_QUANTITIES[q])) for q, (sid, nm) in names.items()]
        rows.append(not_tested("S11", "S11 B(x)(lambda) State 2143: lambda_rel=3 vs 0", "H", "no stage-H run", role="copied"))
        return rows
    rows = []
    corr = _lambda_corr(ctx)
    src = ctx.tables.source("H", "lambda_corr")
    unreach = ""
    paths = ctx.tables.get("H", "lambda_paths", ["tag", "family", "formulation", "discriminant", "lam_rel",
                                                 "logp_target_unperturbed"])
    if not _missing_cols(paths, ["tag", "family", "formulation", "discriminant", "logp_target_unperturbed"]):
        cross = _f(((run.manifest.get("path") or {}) if isinstance(run.manifest.get("path"), dict) else {}).get("cross_nats", 1.0))
        st = ctx.split(paths[paths["discriminant"].astype(str) == H_PREREG_DISCRIMINANT]).drop_duplicates(["tag"])
        if len(st) and np.isfinite(cross):
            k = int((st["logp_target_unperturbed"].astype(float) > -cross).sum())
            unreach = f"{k} of {len(st)} test starts have log p0(target) > -{float(cross):g}: boundary unreachable"
    for q, (sid, nm) in names.items():
        dirn = H_PREREG_QUANTITIES[q]
        if corr is None:
            rows.append(_no_table(sid, nm, "H", "lambda_corr / lambda_paths", ctx, rule=_rule_secondary(dirn)))
            continue
        g = ctx.split(corr[corr["quantity"].astype(str) == q]).copy()
        g["rho"] = pd.to_numeric(g["rho"], errors="coerce")
        n = pd.to_numeric(g["n"], errors="coerce").fillna(0)
        const = g["rho"].isna() & (n >= 2)
        if "constant_series" in g.columns:     # stage H since the rho = 0 rule: such starts are written with rho = 0
            const |= g["constant_series"].map(lambda v: bool(_flag(v)))
        g["rho_main"] = g["rho"].where(~const, 0.0)
        g["rho_as_run"] = g["rho"].where(~const)
        if not np.isfinite(g["rho_main"]).any():
            why = "no finite rho on the test split" + (" (no SAE: sae_source none)" if q == "sae_fvu" else "")
            rows.append(not_tested(sid, nm, "H", _join(why, ctx.split_note), rule=_rule_secondary(dirn)))
            continue
        cnote = f"{int(const.sum())} of {len(g)} starts have a constant series: entered as rho = 0"
        r = _boot_row(sid, nm, g, "rho_main", ctx, stage="H", direction=dirn, source=src,
                      note=_join(cnote, unreach if q == "steps_to_cross_censored" else ""))
        sae_src = str(run.manifest.get("sae_source", ""))
        if q == "sae_fvu" and sae_src == "synthetic_random":
            r.update(verdict="not_interpretable", bh_group=False,
                     deviation_note=_join("synthetic random SAE (sae_source synthetic_random): function test, not a finding",
                                          r["deviation_note"]))
        rows.append(r)
        rows.append(_boot_row(f"{sid}.as_run", f"{nm}, as run (constant series dropped)", g, "rho_as_run", ctx, stage="H",
                              direction=dirn, role="sensitivity", bh_group=False, source=src,
                              note=f"sensitivity: {int(const.sum())} constant-series starts dropped as in lambda_boot"))
    # ---- 10f: features frozen on DEV, confirmed on the test families
    rows.append(_s10f_row(ctx, run))
    # ---- 11: State 2143 lambda sweep (copied, documentation)
    lo = ctx.tables.get("H", "lambda_original")
    n11 = "S11 B(x)(lambda) State 2143: sign of B at lambda_rel = 3 vs lambda = 0"
    if lo is None or lo.empty or _missing_cols(lo, ["B_lambda0", "B_lambda3"]):
        rows.append(_no_table("S11", n11, "H", "lambda_original", ctx, role="copied"))
    else:
        r = lo.iloc[0]
        b0, b3 = _f(r["B_lambda0"]), _f(r["B_lambda3"])
        same = _flag(r.get("same_sign"))
        rows.append(make_row("S11", f"{n11} ({ctx.revision})", role="copied", stage="H", verdict="open", estimate=b3, n_rows=1,
                             rule="documentation (two-sided, expectation open): sign of B at lambda_rel = 3 vs lambda = 0",
                             note=_join(f"B(lambda=0) {_g(b0)}, B(lambda_rel=3) {_g(b3)}, difference {_g(b3 - b0)}, same sign {same}",
                                        caveat_for("S11")),
                             source=ctx.tables.source("H", "lambda_original")))
    return rows


RULE_10F = ("candidates frozen on DEV01-04 (|mean rho| >= 0.7), confirmed on F01-F12 x A,B: supported iff at least one "
            "candidate is confirmed (test CI excludes 0 on the DEV side) with BH q <= 0.05 over the candidates' p_boot_test; "
            "not_supported iff candidates exist and none is; not_tested without a candidate")


def _s10f_row(ctx: EvalContext, run: StageRun) -> dict:
    """10f from stage H's ``lambda_features_frozen`` (candidates frozen on DEV01-04 and tested
    on F01-F12 x {A, B}, ``target_logit`` discriminant), BH over the candidates' two-sided
    ``p_boot_test``; runs that predate the table fall back to the count of
    ``lambda_features_summary`` (``not_interpretable`` with :data:`CAVEATS` ``S10f``)."""
    nf = "S10f features with |rho| >= 0.7 frozen on DEV, tested on the test families"
    sae = f"sae_source {run.manifest.get('sae_source', '?')}"
    ff = ctx.tables.get("H", "lambda_features_frozen")
    if ff is None:
        fs = ctx.tables.get("H", "lambda_features_summary")
        if fs is None or fs.empty:
            return not_tested("S10f", nf, "H", _join("lambda_features_summary empty (no SAE)", sae), role="copied", rule=RULE_10F)
        nc = int(fs["candidate_all"].map(lambda v: bool(_flag(v))).sum()) if "candidate_all" in fs.columns else 0
        return make_row("S10f", nf, role="copied", stage="H", verdict="not_interpretable", estimate=nc, n_rows=len(fs),
                        rule=RULE_10F, note=_join(f"{nc} candidate features (|rho| >= 0.7 at every start, all families)",
                                                  "run predates lambda_features_frozen", caveat_for("S10f")),
                        source=ctx.tables.source("H", "lambda_features_summary"))
    src = ctx.tables.source("H", "lambda_features_frozen")
    if "discriminant" in ff.columns:
        ff = ff[ff["discriminant"].astype(str) == H_PREREG_DISCRIMINANT]
    if ff.empty:
        return not_tested("S10f", nf, "H", _join("no candidate on DEV (lambda_features_frozen empty)", sae), role="copied",
                          rule=RULE_10F)
    p = pd.to_numeric(ff["p_boot_test"], errors="coerce").to_numpy(dtype=float) if "p_boot_test" in ff.columns \
        else np.full(len(ff), np.nan)
    conf = ff["confirmed"].map(lambda v: bool(_flag(v))).to_numpy(dtype=bool) if "confirmed" in ff.columns \
        else np.zeros(len(ff), dtype=bool)
    n_test = int(np.isfinite(p).sum())
    if n_test == 0:
        return not_tested("S10f", nf, "H", _join(f"{len(ff)} DEV-frozen candidates, none with a test start", sae),
                          role="copied", rule=RULE_10F)
    q = bh_correct(p)
    ok = conf & np.isfinite(q) & (q <= ALPHA)
    k, kc = int(ok.sum()), int(conf.sum())
    feats = ", ".join(str(f) for f in ff.loc[ok, "feature"].tolist()[:10]) if "feature" in ff.columns else ""
    ncl = pd.to_numeric(ff["n_clusters_test"], errors="coerce").max() if "n_clusters_test" in ff.columns else np.nan
    return make_row("S10f", nf, role="copied", stage="H", verdict="supported" if k else "not_supported", estimate=k,
                    n_rows=len(ff), n_families=int(ncl) if np.isfinite(_f(ncl)) else 0, rule=RULE_10F, source=src,
                    note=_join(f"{len(ff)} DEV-frozen candidates ({n_test} with test starts), {kc} confirmed by the test CI, "
                               f"{k} after BH over the candidates' p_boot_test (min q {_g(np.nanmin(q), 3)})",
                               f"confirmed features: {feats}" if feats else "", sae))


# --------------------------------------------------------------------------------------
# stages I, J, K, L: 12-18 (copied)
# --------------------------------------------------------------------------------------
def _ci_verdict(est, lo, hi, direction: int = 0) -> str:
    return secondary_verdict(est, lo, hi, np.nan, direction)


def i_rows(ctx: EvalContext) -> list[dict]:
    n12 = "S12 domain compass: profile of ent at h207, is law the top domain?"
    n13 = "S13 B(dom_culture_politics after counter-steering dom_law) - B(lang_dom), culture/politics, K=8/C1"
    n14 = "S14 head arm: B(worst head) - B(residual), same relative dose"
    run = ctx.tables.run("I")
    if run is None:
        return [_no_run(i, n, "I", "I", ctx, role="copied") for i, n in (("S12", n12), ("S13", n13), ("S14", n14))]
    rows = []
    cs = ctx.tables.get("I", "compass_summary")
    if _missing_cols(cs, ["state_source", "direction", "lang", "law_is_top"]):
        rows.append(_no_table("S12", n12, "I", "compass_summary", ctx, role="copied"))
    else:
        sel = cs[(cs["state_source"].astype(str) == "original_2143") & (cs["direction"].astype(str) == "ent")]
        for _, r in sel.iterrows():
            top = _flag(r.get("law_is_top"))
            rows.append(make_row(f"S12.{r['lang']}", f"{n12} (lexicon {r['lang']})", role="copied", stage="I",
                                 verdict={True: "supported", False: "not_supported"}.get(top, "not_tested"),
                                 estimate=r.get("law_z"), n_rows=int(_f(r.get("n_domains")) or 0),
                                 rule="documentation (no test): supported iff law has the highest z (atlas expectation)",
                                 note=_join(f"top domain {r.get('top_domain')} (z {_g(r.get('top_z'), 3)}), law z {_g(r.get('law_z'), 3)} "
                                            f"rank {r.get('law_rank')} of {r.get('n_domains')}", caveat_for("S12")),
                                 source=ctx.tables.source("I", "compass_summary")))
        if sel.empty:
            rows.append(not_tested("S12", n12, "I", "no original_2143 / ent rows (anchor unavailable?)", role="copied"))
    db = ctx.tables.get("I", "direction_domain_boot")
    rule13 = ("one-sided, minimum effect 0.05 nats (interval rule): supported iff ci_low >= 0.05 and BH q <= 0.05 "
              "(secondary group); not_supported iff ci_high < 0.05; open otherwise")
    if _missing_cols(db, ["quantity", "family_domain", "K", "C", "eps", "mean", "ci_low", "ci_high"]):
        rows.append(_no_table("S13", n13, "I", "direction_domain_boot", ctx, role="copied", rule=rule13))
    else:
        s = db[(db["quantity"].astype(str) == "B_cs_minus_lang_dom") & (db["family_domain"].astype(str) == "culture_politics")
               & (db["K"].astype(int) == 8) & (db["C"].astype(str) == "C1")]
        dn = _dd_norm_note(ctx)
        ci_table = "verdict" in db.columns        # stage I since the CI rule (min_effect_met = ci_low >= 0.05)
        for _, r in s.iterrows():
            p2, p1 = table_p(r, 1)
            has_p = bool(np.isfinite(p2))
            # tables with p_boot flag one dose (contrast_13_eps): the other doses are sensitivity rows
            main = not (has_p and _flag(r.get("preregistered")) is False)
            lit = MIN_EFFECT_VERDICT[min_effect_status(r["mean"], r["ci_low"], r["ci_high"], I_PREREG_MIN_B, 1)]
            lit = lit if (has_p or not main) else _cap_open(lit)
            tab = (f"table verdict {r.get('verdict')} (min_effect_met {r.get('min_effect_met')}, CI rule)" if ci_table
                   else f"min_effect_met (table, point estimate) {r.get('min_effect_met')}")
            rows.append(make_row(f"S13.eps{_eps_key(r['eps'])}", f"{n13}, eps={_eps_key(r['eps'])}",
                                 role="copied" if main else "sensitivity", stage="I", rule=rule13, verdict=lit,
                                 estimate=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"], direction=1,
                                 p_two_sided=p2, p_one_sided=p1, min_effect=I_PREREG_MIN_B,
                                 bh_group=bool(main and has_p and lit != "not_tested"),
                                 n_families=int(_f(r.get("n_clusters")) or 0), n_rows=int(_f(r.get("n_rows")) or 0),
                                 note=_join("" if main else "sensitivity: not the pre-registered contrast-13 dose (uncorrected)",
                                            "" if has_p else NO_P_NOTE, tab, dn, _nboot_note(run), caveat_for("S13")),
                                 source=ctx.tables.source("I", "direction_domain_boot")))
        if s.empty:
            rows.append(not_tested("S13", n13, "I", "no B_cs_minus_lang_dom culture_politics K=8/C1 rows (no culture/politics "
                                                    "families selected?)", role="copied", rule=rule13))
    hb = ctx.tables.get("I", "head_steering_boot")
    rule14 = "two-sided: supported iff the CI excludes 0 and BH q <= 0.05 (secondary group)"
    if _missing_cols(hb, ["quantity", "direction", "eps", "mean", "ci_low", "ci_high"]):
        rows.append(_no_table("S14", n14, "I", "head_steering_boot", ctx, role="copied", rule=rule14))
    else:
        s = hb[hb["quantity"].astype(str) == "B_head_minus_residual"]
        hn = _head_note(ctx)
        for _, r in s.iterrows():
            p2, p1 = table_p(r, 0)
            has_p = bool(np.isfinite(p2))
            lit = _ci_verdict(r["mean"], r["ci_low"], r["ci_high"], 0)
            lit = lit if has_p else _cap_open(lit)
            # PREREGISTRATION 1: steering doses above 0.1 (0.2 / 0.5 / 1.0) are exploratory -> outside the BH group
            prereg_dose = any(np.isclose(float(r["eps"]), d) for d in PREREG_DOSES)
            rows.append(make_row(f"S14.{r['direction']}.eps{_eps_key(r['eps'])}", f"{n14}, {r['direction']}, eps={_eps_key(r['eps'])}",
                                 role="copied" if prereg_dose else "exploratory", stage="I", rule=rule14, verdict=lit,
                                 estimate=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"], p_two_sided=p2, p_one_sided=p1,
                                 bh_group=bool(prereg_dose and has_p and lit != "not_tested"),
                                 n_families=int(_f(r.get("n_clusters")) or 0), n_rows=int(_f(r.get("n_rows")) or 0),
                                 note=_join("" if prereg_dose else "exploratory dose (PREREGISTRATION 1: steering doses "
                                                                   "0.2 / 0.5 / 1.0 are exploratory; not in the BH group)",
                                            "" if has_p else NO_P_NOTE, hn, _nboot_note(run), caveat_for("S14")),
                                 source=ctx.tables.source("I", "head_steering_boot")))
        # sensitivity: the residual arm at the head's absolute written norm (dose_match "absolute")
        for _, r in hb[hb["quantity"].astype(str) == "B_head_minus_residual_abs"].iterrows():
            p2, p1 = table_p(r, 0)
            lit = _ci_verdict(r["mean"], r["ci_low"], r["ci_high"], 0)
            rows.append(make_row(f"S14abs.{r['direction']}.eps{_eps_key(r['eps'])}",
                                 f"S14 head arm: B(worst head) - B(residual), same absolute norm, {r['direction']}, "
                                 f"eps={_eps_key(r['eps'])}", role="sensitivity", stage="I", rule=rule14,
                                 verdict=lit if np.isfinite(p2) else _cap_open(lit), estimate=r["mean"], ci_low=r["ci_low"],
                                 ci_high=r["ci_high"], p_two_sided=p2, p_one_sided=p1, bh_group=False,
                                 n_families=int(_f(r.get("n_clusters")) or 0), n_rows=int(_f(r.get("n_rows")) or 0),
                                 note=_join("sensitivity: residual arm scaled to the head arm's absolute written norm "
                                            "(uncorrected)", hn, _nboot_note(run)),
                                 source=ctx.tables.source("I", "head_steering_boot")))
        if s.empty:
            rows.append(not_tested("S14", n14, "I", "head_steering_boot has no rows (no head selected?)", role="copied",
                                   rule=rule14))
    return rows


def _dd_norm_note(ctx: EvalContext) -> str:
    dd = ctx.tables.get("I", "direction_domain", ["family_domain", "direction", "K", "C", "eps", "delta_norm"])
    if _missing_cols(dd, ["family_domain", "direction", "K", "C", "eps", "delta_norm"]):
        return ""
    s = dd[(dd["family_domain"].astype(str) == "culture_politics") & (dd["K"].astype(int) == 8) & (dd["C"].astype(str) == "C1")
           & dd["direction"].astype(str).isin(["dom_culture_politics_cs", "lang_dom"])]
    if s.empty:
        return ""
    m = s.groupby(["eps", "direction"])["delta_norm"].mean().unstack("direction")
    if "dom_culture_politics_cs" not in m.columns or "lang_dom" not in m.columns:
        return ""
    return "mean ||delta|| cs vs lang_dom: " + ", ".join(
        f"eps {_eps_key(e)} {_g(r['dom_culture_politics_cs'], 3)} vs {_g(r['lang_dom'], 3)}" for e, r in m.iterrows())


def _head_note(ctx: EvalContext) -> str:
    parts = []
    wh = ctx.tables.get("I", "worst_head")
    if not _missing_cols(wh, ["layer", "head", "selected"]):
        sel = wh[wh["selected"].map(_flag) == True]  # noqa: E712
        if len(sel):
            r = sel.iloc[0]
            parts.append(f"worst head L{int(r['layer'])}H{int(r['head'])} (selection {r.get('selection', '?')})")
    hs = ctx.tables.get("I", "head_steering", ["tag", "direction", "eps", "arm", "rms_delta"])
    if not _missing_cols(hs, ["tag", "direction", "eps", "arm", "rms_delta"]):
        p = hs.pivot_table(index=["tag", "direction", "eps"], columns="arm", values="rms_delta")
        if "head" in p.columns and "residual" in p.columns:
            ratio = (p["residual"] / p["head"]).replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio):
                parts.append(f"residual / head written RMS: median {_g(ratio.median(), 3)} (range {_g(ratio.min(), 3)}-{_g(ratio.max(), 3)})")
    man = ctx.tables.run("I").manifest if ctx.tables.run("I") is not None else {}
    w = man.get("worst_head") if isinstance(man.get("worst_head"), dict) else {}
    if w.get("evaluation_families"):
        parts.append(f"evaluated on {len(w['evaluation_families'])} families disjoint from the selection half")
    return _join(*parts)


def j_rows(ctx: EvalContext) -> list[dict]:
    n15 = "S15 RDX: share of ||Delta||^2 (culture/politics vs counterpole) explained by the domain axes, z >= 2 in >= half of the layers"
    n16 = "S16 Pile difficulty ranking: is protein_misfolding in the hardest third? (replacement rule)"
    run = ctx.tables.run("J")
    if run is None:
        return [_no_run("S15", n15, "J", "J", ctx, role="copied"), _no_run("S16", n16, "J", "J", ctx, role="copied")]
    chk = ctx.tables.get("J", "pile_rank_check")
    counterpole = None
    if chk is None or chk.empty or "in_hardest_third" not in chk.columns:
        row16 = _no_table("S16", n16, "J", "pile_rank_check", ctx, role="copied")
    else:
        r = chk.iloc[0]
        inh = _flag(r.get("in_hardest_third"))
        counterpole = str(r.get("counterpole", "") or "")
        nseq = int(_f(r.get("n_sequences")) or 0)
        row16 = make_row("S16", n16, role="copied", stage="J", verdict={True: "supported", False: "not_supported"}.get(inh, "not_tested"),
                         estimate=r.get("rank"), n_rows=nseq,
                         rule="report of the rank; not in the hardest third -> the counterpole is replaced by the hardest subfield",
                         note=_join(f"rank {r.get('rank')} of {r.get('n_subfields')}, hardest {r.get('hardest')}, counterpole "
                                    f"{counterpole or '?'}", f"N = {nseq} sequences (< 2000 pre-registered)" if nseq < 2000 else "",
                                    caveat_for("S16")),
                         source=ctx.tables.source("J", "pile_rank_check"))
    rows = _s15_rows(ctx, n15, counterpole)
    rows.append(row16)
    return rows


def _s15_rows(ctx: EvalContext, n15: str, counterpole: Optional[str]) -> list[dict]:
    rows: list[dict] = []
    rs = ctx.tables.get("J", "rdx_summary")
    rule15 = "supported iff z >= 2 in at least half of the layers (min_effect_met)"
    if _missing_cols(rs, ["contrast", "basis", "readout", "frac_z_ge_2", "min_effect_met"]):
        rows.append(_no_table("S15", n15, "J", "rdx_summary", ctx, role="copied", rule=rule15))
        return rows
    c, b, ro = J_PREREG
    s = rs[(rs["contrast"].astype(str) == c) & (rs["basis"].astype(str) == b) & (rs["readout"].astype(str) == ro)]
    stab = ""
    st = ctx.tables.get("J", "rdx_stability")
    if st is not None and len(st) and {"contrast", "basis"} <= set(st.columns):
        ss = st[(st["contrast"].astype(str) == c) & (st["basis"].astype(str) == b)]
        if "readout" in ss.columns:
            ss = ss[ss["readout"].astype(str) == ro]
        fcol = next((x for x in ("frac_z_ge_2", "frac_units_z_ge_2") if x in ss.columns), None)
        if fcol and len(ss):
            stab = "stability over family pairs (frac z >= 2 per site): " + ", ".join(
                f"{r.get('site', r.get('layer', '?'))} {_g(r[fcol], 2)}" for _, r in ss.iterrows())
    replaced = bool(counterpole) and counterpole != "protein_misfolding"
    if s.empty:
        rows.append(not_tested("S15", n15, "J", "no pre-registered rdx_summary row (culture/politics or misfolding families "
                                               "not selected?)", role="copied", rule=rule15))
        return rows
    r = s.iloc[0]
    met = _flag(r.get("min_effect_met"))
    lit = {True: "supported", False: "not_supported"}.get(met, "not_tested")
    fields = dict(estimate=r.get("frac_z_ge_2"), n_rows=int(_f(r.get("n_sites")) or 0),
                  source=ctx.tables.source("J", "rdx_summary"))
    detail = (f"{int(_f(r.get('n_z_ge_2')) or 0)} of {int(_f(r.get('n_sites')) or 0)} sites z >= 2; z mean "
              f"{_g(r.get('z_mean'), 3)}, min {_g(r.get('z_min'), 3)}; explained mean {_g(r.get('explained_mean'), 3)}")
    if replaced:
        rows.append(make_row("S15", n15, role="copied", stage="J", rule=rule15, verdict="not_tested",
                             note=_join(f"replacement rule 16 fired: the pre-registered contrast is culture/politics vs "
                                        f"{counterpole}, which stage J did not compute", caveat_for("S15")), n_rows=0,
                             source=fields["source"]))
        rows.append(make_row("S15.misfolding", f"deviation: {n15} with the original misfolding counterpole", role="copied",
                             stage="J", rule=rule15, verdict=lit, note=_join("deviation row (counterpole replaced by rule 16)",
                                                                             detail, stab), **fields))
    else:
        rows.append(make_row("S15", n15, role="copied", stage="J", rule=rule15, verdict=lit,
                             note=_join(detail, stab, caveat_for("S15")), **fields))
    return rows


def k_rows(ctx: EvalContext) -> list[dict]:
    n17 = "S17 test-time training"
    run = ctx.tables.run("K")
    if run is None:
        return [_no_run("S17", n17, "K", "K", ctx, role="copied")]
    rows = []
    corp = ctx.tables.get("K", "ttt_corpora")
    placeholder = bool(corp is not None and "is_placeholder" in corp.columns
                       and corp["is_placeholder"].map(lambda v: bool(_flag(v))).any())
    if not placeholder and _flag(run.manifest.get("is_placeholder")):
        placeholder = True
    ph_note = "placeholder corpus (is_placeholder): function test, not a finding" if placeholder else ""
    mb = ctx.tables.get("K", "ttt_memorisation_boot")
    if _missing_cols(mb, ["checkpoint", "arm", "quantity", "mean", "ci_low", "ci_high"]):
        rows.append(_no_table("S17i", f"{n17} (i) memorisation speed", "K", "ttt_memorisation_boot", ctx, role="copied"))
    else:
        s = mb[mb["arm"].astype(str).str.startswith("injection-") & (mb["quantity"].astype(str) == "steps_to_threshold")]
        for _, r in s.iterrows():
            lit = _ci_verdict(r["mean"], r["ci_low"], r["ci_high"], 0)
            ctrl = str(r["arm"]).split("-", 1)[1]
            rows.append(make_row(f"S17i.{ctrl}.{r['checkpoint']}", f"{n17} (i) steps to threshold, injection - {ctrl}, {r['checkpoint']}",
                                 role="copied", stage="K", rule="two-sided: supported iff the CI (bootstrap over snippets) excludes 0",
                                 verdict="not_interpretable" if placeholder else lit, estimate=r["mean"], ci_low=r["ci_low"],
                                 ci_high=r["ci_high"], n_rows=int(_f(r.get("n")) or 0),
                                 note=_join(ph_note, f"rule outcome: {lit}" if placeholder else "", caveat_for("S17")),
                                 source=ctx.tables.source("K", "ttt_memorisation_boot")))
    eff = ctx.tables.get("K", "ttt_effects")
    if _missing_cols(eff, ["arm", "measure", "delta", "exceeds_all_controls"]):
        rows.append(_no_table("S17ii", f"{n17} (ii)/(iii) effects vs controls", "K", "ttt_effects", ctx, role="copied"))
        return rows
    inj = eff[eff["arm"].astype(str) == "injection"]
    groups = [("S17ii", "(ii) d sharing index / network activity", lambda m: m == "sharing_jaccard" or m.startswith("activity_")),
              ("S17iii", "(iii) d T(K=8), d B(x)", lambda m: m in ("T_K8", "B_lang_dom"))]
    for sid, label, pick in groups:
        ms = sorted(m for m in set(inj["measure"].astype(str)) if pick(m))
        if not ms:
            rows.append(not_tested(sid, f"{n17} {label}", "K", "measure not in ttt_effects" + (
                " (sharing / activity need an SAE)" if sid == "S17ii" else ""), role="copied"))
            continue
        for m in ms:
            g = inj[inj["measure"].astype(str) == m]
            ex = g["exceeds_all_controls"].map(_flag).dropna()
            frac = float(ex.mean()) if len(ex) else np.nan
            lit = ("supported" if frac == 1.0 else "not_supported" if frac == 0.0 else "open") if np.isfinite(frac) else "not_tested"
            ctrl = eff[(eff["arm"].astype(str) != "injection") & (eff["measure"].astype(str) == m)]
            cn = ", ".join(f"{a} {_g(x)}" for a, x in ctrl.groupby("arm")["delta"].mean().items())
            rows.append(make_row(f"{sid}.{m}", f"{n17} {label}: {m}", role="copied", stage="K",
                                 rule="the injection arm must exceed all three controls (exceeds_all_controls on every row)",
                                 verdict="not_interpretable" if placeholder else lit, estimate=g["delta"].astype(float).mean(),
                                 n_rows=len(g),
                                 note=_join(ph_note, f"rule outcome: {lit}" if placeholder else "",
                                            f"exceeds_all_controls on {_g(frac, 3)} of rows; mean delta controls: {cn}",
                                            caveat_for("S17")),
                                 source=ctx.tables.source("K", "ttt_effects")))
    return rows


def l_rows(ctx: EvalContext) -> list[dict]:
    n18 = "S18 transplant stand-in validation (R2 gap >= 0.2, cos >= 0.3, sign agreement residual and kv)"
    rows = []
    run = ctx.tables.run("L")
    if run is None:
        rows.append(_no_run("S18", n18, "L", "L", ctx, role="copied"))
    else:
        v = ctx.tables.get("L", "verdict")
        if v is None or v.empty or "validated" not in v.columns:
            rows.append(_no_table("S18", n18, "L", "verdict", ctx, role="copied"))
        else:
            r = v.iloc[0]
            selfc = _flag(r.get("self_consistency"))
            val, counts = _flag(r.get("validated")), _flag(r.get("counts_for_prereg"))
            if selfc or val is None:
                verdict = "not_interpretable"
            elif val and counts is not False:
                verdict = "supported"
            else:
                verdict = "not_supported"
            ts = ctx.tables.get("L", "transfer_summary")
            est, extra = np.nan, ""
            if ts is not None and len(ts):
                t = ts.iloc[0]
                est = _f(t.get("cos_transferred_vs_native"))
                extra = (f"best layer {t.get('best_layer')}: cos {_g(est, 3)}, r2_gap {_g(t.get('r2_gap'), 3)}, "
                         f"{t.get('n_validated_layers')} of {t.get('n_layers')} layers validated")
            src = run.manifest.get("source") if isinstance(run.manifest.get("source"), dict) else {}
            rows.append(make_row("S18", n18, role="copied", stage="L", verdict=verdict, estimate=est, n_rows=1,
                                 rule="validated = R2 gap, cosine and steering sign at the best layer AND sign agreement for "
                                      "residual and kv; self-consistency sources give no verdict",
                                 note=_join(f"source {src.get('kind', '?')} {src.get('name', '')}".strip(),
                                            f"r2_gap_ok {r.get('r2_gap_ok')}, cos_ok {r.get('cos_ok')}, sign residual "
                                            f"{r.get('sign_residual')}, kv {r.get('sign_kv')}, counts_for_prereg {counts}",
                                            extra, caveat_for("S18")),
                                 source=ctx.tables.source("L", "verdict")))
    n18t = "S18T trace source: sharing effect of the trace LM's direction transplanted into Pythia"
    tr = ctx.tables.run("L_trace")
    if tr is None:
        rows.append(_no_run("S18T", n18t, "L", "L_trace", ctx, role="copied"))
        return rows
    tv = ctx.tables.get("L_trace", "trace_verdict")
    if tv is None or tv.empty or "sharing_effect_claimed" not in tv.columns:
        rows.append(_no_table("S18T", n18t, "L", "trace_verdict", ctx, key="L_trace", role="copied", rule=RULE_18T))
        return rows
    tv, rj_note, rj_error = _trace_verdict_current(tv, tr)
    src = ctx.tables.source("L_trace", "trace_verdict")
    for _, r in tv.iterrows():
        claimed, counts = _flag(r.get("sharing_effect_claimed")), _flag(r.get("counts_for_prereg"))
        if rj_error:
            verdict = "not_interpretable"
        elif claimed and counts:
            verdict = "supported"
        elif claimed:
            verdict = "not_interpretable"
        else:
            verdict = "not_supported"
        rows.append(make_row(f"S18T.{r.get('mode', '?')}", f"{n18t}, mode {r.get('mode', '?')}", role="copied", stage="L",
                             verdict=verdict, estimate=r.get("B_vs_random"), n_families=int(_f(r.get("n_clusters")) or 0),
                             rule=RULE_18T,
                             note=_join(rj_error, rj_note,
                                        f"probe_ok {r.get('probe_ok')}, align_ok {r.get('align_ok')}, align_beats_untrained "
                                        f"{r.get('align_beats_untrained')}, transplant_effect {r.get('transplant_effect')}, "
                                        f"lm_corpus_ok {r.get('lm_corpus_ok')}, counts_for_prereg {counts}",
                                        "claimed but does not count for the pre-registration" if claimed and not counts else "",
                                        caveat_for("S18T")),
                             source=src))
    return rows


RULE_18T = ("claim = probe_ok (frozen probe criterion at the frozen layer AND trained AUC above the surface baseline AND "
            "v_src from that probe) AND align_ok AND align_beats_untrained AND transplant_effect (B or T vs plain, random "
            "and permuted_alignment, CIs exclude 0 in the pre-specified direction, >= 3 families); counts only after a "
            "passed stand-in validation and with a healthy trace-LM corpus (counts_for_prereg)")


def _trace_verdict_current(tv: pd.DataFrame, run: StageRun) -> tuple[pd.DataFrame, str, str]:
    """``(table, note, error)``: a ``trace_verdict`` written before the tightened rule of
    27.10.2026 (no ``lm_corpus_ok`` column) is recomputed with
    :func:`lsne.experiments.stage_l_transplant.rejudge_trace` (CPU, read-only, reads the trace
    LM's corpus health from ``manifest.source.model_dir``); the table's own claim is not used.
    When the rejudge fails, the old table is returned with the reason as ``error`` (the rows
    are then ``not_interpretable``)."""
    if "lm_corpus_ok" in tv.columns:
        return tv, "", ""
    old = "trace_verdict predates the tightened rule of 27.10.2026 (no lm_corpus_ok)"
    try:
        from .experiments.stage_l_transplant import rejudge_trace
        new = rejudge_trace(run.path)["trace_verdict"]
    except Exception as e:  # noqa: BLE001 - recorded on the row
        return tv, "", f"{old} and rejudge_trace failed ({type(e).__name__}: {e}): the table's own claim is not interpretable"
    if new is None or new.empty:
        return tv, "", f"{old} and rejudge_trace returned no rows: the table's own claim is not interpretable"
    return new, f"{old}: re-judged from the run's tables (stage_l_transplant.rejudge_trace)", ""


# --------------------------------------------------------------------------------------
# BH and the driver
# --------------------------------------------------------------------------------------
def apply_bh(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """BH over the rows with ``bh_group`` and a finite p (one-sided where ``direction`` is
    set, else two-sided); finalises their verdicts with :func:`bh_verdict` (the interval rule
    of the row's ``min_effect`` where one is pre-registered).  Also fills ``bh_q_endpoint``
    (Bonferroni within the endpoint, BH across endpoints) and returns a description of both
    groupings."""
    df = df.copy()
    if "min_effect" not in df.columns:
        df["min_effect"] = np.nan
    p_used = np.where(df["direction"].astype(int) != 0, df["p_one_sided"].astype(float), df["p_two_sided"].astype(float))
    mask = df["bh_group"].astype(bool).to_numpy() & np.isfinite(p_used)
    info = dict(m=int(mask.sum()), ids=df.loc[mask, "id"].tolist(), m_endpoints=0, endpoints={})
    if not mask.any():
        return df, info
    q = bh_correct(p_used[mask])
    df.loc[mask, "bh_q"] = q
    ep = df.loc[mask, "id"].map(endpoint_key).to_numpy()
    p_m = p_used[mask]
    keys = sorted(set(ep), key=lambda k: (len(k), k))
    p_ep = np.array([min(1.0, float(np.sum(ep == k)) * float(np.min(p_m[ep == k]))) for k in keys])
    q_ep = bh_correct(p_ep)
    qmap = dict(zip(keys, q_ep))
    df.loc[mask, "bh_q_endpoint"] = [qmap[k] for k in ep]
    info.update(m_endpoints=len(keys), endpoints={k: dict(n_rows=int(np.sum(ep == k)), p_bonferroni=float(pe), q=float(qq))
                                                  for k, pe, qq in zip(keys, p_ep, q_ep)})
    for i in np.flatnonzero(mask):
        r = df.iloc[i]
        df.iat[i, df.columns.get_loc("verdict")] = bh_verdict(r["estimate"], r["ci_low"], r["ci_high"], r["bh_q"],
                                                              int(r["direction"]), r["min_effect"])
    return df, info


def _guard(fn: Callable[[], list[dict]], id: str, name: str, stage: str, role: str = "secondary") -> list[dict]:
    """Run one endpoint block; an unexpected error becomes a ``not_tested`` row with the
    exception (the other endpoints still run)."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc(limit=3).strip().splitlines()[-1]
        return [not_tested(id, name, stage, f"evaluation error: {type(e).__name__}: {e} ({tb})", role=role)]


def evaluate(roots: Sequence, model: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION, n_boot: int = DEFAULT_N_BOOT,
             seed: int = 0, frozen_dose: Optional[float] = None, families: Sequence[str] = TEST_FAMILIES,
             formulations: Sequence[str] = TEST_FORMULATIONS) -> dict:
    """Evaluate every endpoint from the run directories below ``roots``.

    Returns ``dict(results, runs, bh, meta)``: the results table (:data:`RESULT_COLS` +
    :data:`EXTRA_COLS`), the run table (:data:`RUN_COLS`), the BH grouping and the
    evaluation settings.  ``families`` / ``formulations`` override the confirmation set
    (tests and smoke runs only; the override is noted on every recomputed row)."""
    runs, notes = discover_runs(roots, model, revision)
    chosen = select_runs(runs)
    ctx = EvalContext(tables=Tables(chosen), model=model, revision=revision, n_boot=int(n_boot), seed=int(seed),
                      frozen_dose=(None if frozen_dose is None else float(frozen_dose)), families=tuple(families),
                      formulations=tuple(formulations))
    rows: list[dict] = []
    rows += _guard(lambda: primary_rows(ctx), "P", PRIMARY_NAME, "D", role="primary")
    rows += _guard(lambda: token_control_rows(ctx), "P.ctl", CTL_NAME, "D", role="primary_control")
    c_audit, c_invalid = c_audit_row(ctx)
    rows.append(c_audit)
    rows += _guard(lambda: s1_s2_rows(ctx), "S1", "S1/S2 text effect", "C")
    rows += _guard(lambda: s3_rows(ctx), "S3", "S3 attribution", "C")
    rows += _guard(lambda: s4_rows(ctx), "S4", "S4 hysteresis", "D")
    rows += _guard(lambda: s5_rows(ctx), "S5", "S5 non-monotonicity", "D")
    rows += _guard(lambda: s6_rows(ctx, c_invalid), "S6", "S6 transfer", "C")
    rows += _guard(lambda: g_rows(ctx), "S7", "S7-S9 geometry", "G", role="gate")
    rows += _guard(lambda: h_rows(ctx), "S10", "S10-S11 lambda dynamics", "H")
    rows += _guard(lambda: i_rows(ctx), "S12", "S12-S14 compass / heads", "I", role="copied")
    rows += _guard(lambda: j_rows(ctx), "S15", "S15-S16 RDX / Pile", "J", role="copied")
    rows += _guard(lambda: k_rows(ctx), "S17", "S17 TTT", "K", role="copied")
    rows += _guard(lambda: l_rows(ctx), "S18", "S18 transplant", "L", role="copied")
    df = pd.DataFrame(rows, columns=RESULT_COLS + EXTRA_COLS)
    df = apply_token_control(df)            # PREREGISTRATION 6.11: the control can only weaken P (P is outside BH)
    df, bh = apply_bh(df)
    drun = ctx.tables.run("D")
    frozen = _frozen_dose(ctx, drun) if drun is not None else ctx.frozen_dose
    meta = dict(model=model_tag(model), revision=revision, n_boot=int(n_boot), seed=int(seed), frozen_dose=frozen,
                frozen_dose_source=("--frozen-dose" if ctx.frozen_dose is not None else
                                    "stage-D manifest" if frozen is not None else None),
                families=list(ctx.families), formulations=list(ctx.formulations),
                roots=[str(r) for r in roots], timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                notes=list(notes))
    return dict(results=df, runs=runs_table(runs, chosen, notes), bh=bh, meta=meta)


# --------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------
SECTIONS = (
    ("Primärer Endpunkt (§2), Kontrolle der Sprache an der Injektionsstelle (Nachtrag 6.11) und Laufvalidität (§4)",
     ("primary", "primary_component", "primary_deviation", "primary_control", "audit")),
    ("Sekundäre Endpunkte 1–6 und 10 (BH-Gruppe)", ("secondary",)),
    ("Gate G0 (7) und bedingte Endpunkte 8–9 (BH-Gruppe, wenn G0 besteht)", ("gate", "conditional")),
    ("Übernommene Endpunkte 10f–18 (13 und 14 in der BH-Gruppe)", ("copied",)),
    ("Sensitivität und Exploration (nicht konfirmatorisch)", ("sensitivity", "exploratory")),
)
VERDICT_DE = {"supported": "gestützt", "not_supported": "nicht gestützt", "open": "offen", "gate_failed": "Gate verfehlt",
              "not_interpretable": "nicht interpretierbar", "not_tested": "nicht getestet"}


def _md_cell(x) -> str:
    return str(x).replace("|", "\\|").replace("\n", " ")


def _fmt_p(p) -> str:
    p = _f(p)
    return "–" if not np.isfinite(p) else (f"{p:.1e}" if p < 1e-3 else f"{p:.4f}")


def _md_note(row) -> str:
    """The deviation note without the endpoint's :data:`CAVEATS` (listed once per endpoint)."""
    note = str(row["deviation_note"] if isinstance(row["deviation_note"], str) else "")
    cav = caveat_for(row["id"])
    if cav:
        note = note.replace(cav, "")
    return "; ".join(p.strip() for p in note.split("; ") if p.strip())


def markdown_report(res: dict) -> str:
    df, runs, bh, meta = res["results"], res["runs"], res["bh"], res["meta"]
    dose = (f"eingefroren {_eps_key(meta['frozen_dose'])} ({meta.get('frozen_dose_source')})" if meta["frozen_dose"] is not None
            else "keine auf DEV eingefrorene Dosis vermerkt: alle präregistrierten Dosen berichtet")
    out = [f"# Präregistrierte Auswertung LS-NEQ: {meta['model']}@{meta['revision']}", ""]
    out += [f"Erzeugt {meta['timestamp']} von `python -m lsne.prereg_eval` aus den vorhandenen Laufverzeichnissen "
            "(keine GPU-Stufe neu gerechnet).", "",
            f"- Bestätigungsmenge: Familien {', '.join(meta['families'])} × Formulierungen {', '.join(meta['formulations'])}",
            "- Statistik: Mittel der Familienmittel; 95 %-Perzentilintervall eines Familien-Cluster-Bootstraps "
            f"({meta['n_boot']} Ziehungen, Seed {meta['seed']}); p zweiseitig (einseitig, wo eine Richtung präregistriert ist)",
            f"- Dosis des Primärendpunkts: {dose}",
            f"- BH-Gruppe: {bh['m']} Tests (eine Zeile je Untertest der Endpunkte 1–6, 8–9 (wenn G0 besteht), 10, 13 und 14; "
            "8, 9, 13, 14 mit dem `p_boot` ihrer Tabelle, ohne p nicht korrigiert und höchstens „offen“); Sensitivität "
            f"`bh_q_endpoint`: Bonferroni innerhalb des Endpunkts, BH über {bh['m_endpoints']} Endpunkte",
            "- Mindesteffekte (2: 0,05; 9: cos 0,3; 13: 0,05 nats) entscheidet das Intervall (untere Grenze ≥ Mindesteffekt), "
            "nie der Punktschätzer",
            "- Kontrolle der Sprache an der Injektionsstelle (Nachtrag 6.11, Zeilen `P.ctl…`, außerhalb der BH-Gruppe): "
            "„Wechsel“ lässt P unverändert, „lokal“ macht ein gestütztes oder offenes P nicht interpretierbar, „offen“ oder "
            "kein gültiger Kontrolllauf begrenzen P auf höchstens „offen“", ""]
    out += ["## Laufverzeichnisse", "", "| Stufe | verwendet | Status | Zeitstempel | n_boot | Pfad | Hinweis |", "|---|---|---|---|---|---|---|"]
    for _, r in runs.iterrows():
        out.append(f"| {r['key'] or '–'} | {'ja' if r['used'] else 'nein'} | {r['status']} | {r['timestamp']} | "
                   f"{'' if pd.isna(r['n_boot']) else int(r['n_boot'])} | {_md_cell(r['path'])} | {_md_cell(r['note'])} |")
    out.append("")
    head = "| id | Endpunkt | Schätzer [95 %-KI] | p (2s / 1s) | q (BH) | Familien / Zeilen | Urteil | Hinweis |"
    for title, roles in SECTIONS:
        sub = df[df["role"].isin(roles)]
        if sub.empty:
            continue
        out += [f"## {title}", "", head, "|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            est = "–" if not np.isfinite(_f(r["estimate"])) else _g(r["estimate"])
            if np.isfinite(_f(r["ci_low"])):
                est += f" [{_g(r['ci_low'])}, {_g(r['ci_high'])}]"
            out.append(f"| {r['id']} | {_md_cell(r['name'])} | {est} | {_fmt_p(r['p_two_sided'])} / {_fmt_p(r['p_one_sided'])} | "
                       f"{_fmt_p(r['bh_q'])} | {r['n_families']} / {r['n_rows']} | **{VERDICT_DE.get(r['verdict'], r['verdict'])}** | "
                       f"{_md_cell(_md_note(r))} |")
        out.append("")
    out += ["## BH-Gruppierung", "", f"BH (`lsne.stats.bh_correct`) über m = {bh['m']} Zeilen: {', '.join(bh['ids']) or '–'}.", ""]
    if bh["endpoints"]:
        out += ["| Endpunkt | Zeilen | p (Bonferroni im Endpunkt) | q (BH über Endpunkte) |", "|---|---|---|---|"]
        for k, v in bh["endpoints"].items():
            out.append(f"| {k} | {v['n_rows']} | {_fmt_p(v['p_bonferroni'])} | {_fmt_p(v['q'])} |")
        out.append("")
    out += ["## Entscheidungsregeln", "", "| Endpunkte | Regel |", "|---|---|"]
    rules: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        if r["rule"]:
            ks = rules.setdefault(r["rule"], [])
            if endpoint_key(r["id"]) not in ks:
                ks.append(endpoint_key(r["id"]))
    out += [f"| {', '.join(ks)} | {_md_cell(rule)} |" for rule, ks in rules.items()]
    notes = df["deviation_note"].astype(str)
    keys = [k for k in dict.fromkeys(k for rid in df["id"] for k in caveat_keys(rid))
            if notes.str.contains(CAVEATS[k], regex=False).any()]           # only caveats a row actually carries
    if keys:
        out += ["", "## Bekannte Einschränkungen (in `deviation_note` jeder Zeile)", ""]
        out += [f"- **{k}**: {_md_cell(CAVEATS[k])}" for k in keys]
    if meta["notes"]:
        out += ["", "## Hinweise zu den Wurzelverzeichnissen", ""] + [f"- {n}" for n in meta["notes"]]
    out.append("")
    return "\n".join(out)


def write_outputs(res: dict, out_dir) -> dict[str, Path]:
    """``prereg_results.csv``, ``prereg_results.md``, ``prereg_runs.csv`` and ``prereg_meta.json`` in ``out_dir``."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = dict(results=d / "prereg_results.csv", markdown=d / "prereg_results.md", runs=d / "prereg_runs.csv",
                 meta=d / "prereg_meta.json")
    res["results"].to_csv(paths["results"], index=False)
    res["runs"].to_csv(paths["runs"], index=False)
    paths["markdown"].write_text(markdown_report(res), encoding="utf-8")
    paths["meta"].write_text(json.dumps(dict(res["meta"], bh=res["bh"]), indent=1, ensure_ascii=False, default=str),
                             encoding="utf-8")
    return paths


# --------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lsne.prereg_eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", action="append", required=True,
                   help="run root (<root>/<stage>/<model>@<revision>/); repeat for several roots / accounts")
    p.add_argument("--out", required=True, help="output directory of prereg_results.csv / .md")
    p.add_argument("--model", default=DEFAULT_MODEL, help="model tag of the run directories (default pythia-160m)")
    p.add_argument("--revision", default=DEFAULT_REVISION)
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT, help="family bootstrap draws (pre-registered: 10000)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frozen-dose", type=float, default=None,
                   help="damage dose frozen on DEV01-DEV04 before the confirmatory run (main dose of the primary endpoint)")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    res = evaluate(args.runs, model=args.model, revision=args.revision, n_boot=args.n_boot, seed=args.seed,
                   frozen_dose=args.frozen_dose)
    paths = write_outputs(res, args.out)
    df = res["results"]
    used = res["runs"][res["runs"]["used"].astype(bool)]
    print(f"runs used: {', '.join(f'{k} ({p})' for k, p in zip(used['key'], used['path'])) or 'none'}")
    for n in res["meta"]["notes"]:
        print("note:", n)
    counts = df["verdict"].value_counts()
    print("verdicts:", ", ".join(f"{v} {int(counts.get(v, 0))}" for v in VERDICTS))
    main_rows = df[df["role"].isin(["primary", "primary_control", "secondary", "gate"])]
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(main_rows[["id", "estimate", "ci_low", "ci_high", "p_two_sided", "p_one_sided", "bh_q", "verdict"]].to_string(index=False))
    for k, p in paths.items():
        print(f"{k}: {p}")
    return 0


__all__ = [
    "TEST_FAMILIES", "TEST_FORMULATIONS", "DEV_FAMILIES", "PREREG_DOSES", "PRIMARY_K", "PRIMARY_MIN_EFFECT", "N_RAND_PREREG",
    "AUDIT_MAX_REL_ERR", "EVAL_STAGES", "EVAL_KEYS", "ROOT_CHILDREN", "AMENDMENT_611", "CTL_CONTROL", "DAMAGE_CTL_COLS",
    "token_control_rows", "apply_token_control", "VERDICTS", "RESULT_COLS", "EXTRA_COLS", "RUN_COLS", "CAVEATS", "StageRun",
    "resolve_roots", "discover_runs", "select_runs", "runs_table", "read_table", "split_rows", "boot_split", "p_one_sided",
    "SPLIT_TEST_NOTE", "G_SPLIT_TEST_NOTE", "NO_P_NOTE", "MIN_EFFECT_VERDICT", "table_p", "bh_verdict", "S6_TEST_SIGN",
    "family_boot", "family_delta", "min_effect_status", "primary_verdict", "secondary_verdict", "caveat_keys", "caveat_for",
    "audit_summary", "make_row", "apply_bh", "evaluate", "markdown_report", "write_outputs", "build_parser", "main",
]


if __name__ == "__main__":
    sys.exit(main())
