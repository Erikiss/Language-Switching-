"""Command-line entry point.

    python -m lsne.cli run --stage A|B|...|L|all[,...] --model EleutherAI/pythia-160m \\
        --revision step98000 --dtype float32 --device auto --families F01,F02 --out runs \\
        [--limit-variants N] [--limit-families N] [--smoke] [--use-sae] [--seed S] \\
        [--pairs model@rev,...|trace|self] [--f-stages B,C,D] [--shard-index I --shard-count N] \\
        [--ttt-corpus DIR] [--ttt-checkpoints step4000,step98000,step143000] \\
        [--transplant-source rotated_self|self|model@rev] [--atlas-dir DIR]
    python -m lsne.cli run --stage D --d-control token_language --d-main-run <root>/D/<model>@<rev> \\
        --out <LSNE>/runs_ctl [--model ... --revision ...]
    python -m lsne.cli merge --root runs --out merged/

``run`` loads the model once, runs the requested stages in order (``all`` = A B C D E F
G H I J K L, see docs/STAGES.md) and shares the stimulus bank / direction bank between
them.  Every stage is wrapped in ``try/except``: a failing stage writes ``error.txt`` and
a manifest with ``status='error'`` into its run directory and the next stage still runs,
so a Colab run always yields partial results; stages G-L additionally capture failures
per substage (the run report then says ``ok_with_substage_errors`` and lists them, and a
stale ``error.txt`` of an earlier failed run is removed once a stage ends ok).  Such a stage
ran and wrote its tables: the final line counts it as run (naming the substage errors), and
the exit code is 1 only when every requested stage failed outright (``status='error'``).
``--smoke`` sets every limit for a short CPU run with pythia-14m (one family, three
variants, tiny generation / path / training budgets; stage F re-runs the same
model/revision only); the per-stage durations are printed at the end.

``--d-control token_language --d-main-run DIR`` turns stage D into the control run of the
injection-site language (PREREGISTRATION section 6, amendment 11;
:mod:`lsne.experiments.stage_d_control`): stage D alone, with the main stage-D run's directions,
doses and depth bank, written to ``<out>/D/<model>@<rev>`` under its own root (``runs_ctl``).  The
run is refused before the model is loaded when that directory is the main run's or holds a stage-D
run other than this control; a refused control run writes nothing, and the error manifest of a
failed one carries the control marks, so a restart resumes it.

``merge`` concatenates all same-named csv files below ``--root`` (any depth, including
the stage-F sub-run directories of all shards) into ``<out>/<name>.csv`` with the
columns ``run_id`` (directory relative to the root), ``run_stage``, ``run_model`` and
``run_revision`` prepended, and collects every manifest into ``<out>/manifests.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from .config import ModelSpec, RunConfig
from .io import RunDir, read_manifest
from .model import load_model
from .experiments import STAGES, STAGE_ORDER, STAGE_TITLES
from .experiments import common as C
from .experiments import stage_d_control as SDC
from .experiments.stage_f_scale import parse_pairs


# --------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lsne.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run experiment stages")
    r.add_argument("--stage", default="all", help="A|B|C|D|E|F|G|H|I|J|K|L|all or a comma-separated list (default: all)")
    r.add_argument("--model", default=RunConfig().model.name)
    r.add_argument("--revision", default=RunConfig().model.revision)
    r.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    r.add_argument("--device", default="auto")
    r.add_argument("--families", default=None, help="comma-separated family ids (F01) or tags (F01A); default: all")
    r.add_argument("--families-dir", default=RunConfig().families_dir)
    r.add_argument("--out", default=None, help="output root (default: lsne.io.drive_root(), i.e. Drive on Colab or ./runs)")
    r.add_argument("--limit-variants", type=int, default=None,
                   help="keep at most N variants (formulation by formulation, core triple first)")
    r.add_argument("--limit-families", type=int, default=None, help="keep at most N family ids")
    r.add_argument("--k-levels", default=None, help="comma-separated K levels of the unit-level grid (default 0,2,4,8)")
    r.add_argument("--c-levels", default=None, help="comma-separated C levels (default C1,C2,C3)")
    r.add_argument("--k-extreme", default=None, help="comma-separated segment-level K levels (default 16,28; '' = none)")
    r.add_argument("--n-random-dirs", type=int, default=None)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--smoke", action="store_true", help="tiny CPU run (pythia-14m, < 3 min)")
    r.add_argument("--use-sae", action="store_true", help="SAE feature directions (stage B/C/D) if an SAE exists")
    r.add_argument("--n-unigram", type=int, default=None, help="Pile sequences for the unigram counts (default 200; smoke 5)")
    r.add_argument("--axis-npz", default=None, help="optional axes.npz with the ARI axis to orthogonalise 'ent' against")
    r.add_argument("--gen-n-new", type=int, default=None)
    r.add_argument("--gen-n-samples", type=int, default=None)
    r.add_argument("--tn-rank", type=int, default=None)
    r.add_argument("--tn-window", type=int, default=None)
    r.add_argument("--pairs", default=None, help="stage F: 'model@rev,...', bare revisions, 'trace' or 'self'")
    r.add_argument("--f-stages", default=None, help="stage F: stages per pair (default B,C,D)")
    r.add_argument("--shard-index", type=int, default=0)
    r.add_argument("--shard-count", type=int, default=1)
    r.add_argument("--ttt-corpus", default=None, help="stage K: corpus directory/file (default data/ttt/dao_fork; placeholder if missing)")
    r.add_argument("--ttt-checkpoints", default=None,
                   help="stage K: extra revisions of the memorisation table, e.g. step4000,step98000,step143000 (default: current only)")
    r.add_argument("--transplant-source", default=None,
                   help="stage L: rotated_self (default) | self | model@revision (HF causal LM, e.g. EleutherAI/pythia-70m@step143000)")
    r.add_argument("--atlas-dir", default=None, help="stage I: directory with the stored atlas extraction (default data/atlas)")
    r.add_argument("--worst-head", default=None, help="stage I: frozen worst head 'L:H' from a development run (pre-registration 14)")
    r.add_argument("--d-control", default=None, choices=SDC.CONTROLS,
                   help="stage D alone as a control run of the main stage-D run (token_language: PREREGISTRATION 6.11); "
                        "needs --d-main-run and its own --out root, e.g. <LSNE>/runs_ctl")
    r.add_argument("--d-main-run", default=None,
                   help="with --d-control: directory of the main stage-D run (<root>/D/<model>@<revision>)")
    r.add_argument("--no-progress", action="store_true")

    m = sub.add_parser("merge", help="merge csv tables of several run directories")
    m.add_argument("--root", required=True)
    m.add_argument("--out", required=True)
    return p


def parse_stages(spec: str) -> list[str]:
    spec = (spec or "all").strip()
    if spec.lower() == "all":
        return list(STAGE_ORDER)
    out = []
    for s in spec.replace(";", ",").split(","):
        s = s.strip().upper()
        if not s:
            continue
        if s not in STAGES:
            raise SystemExit(f"unknown stage {s!r}; choose from {', '.join(STAGE_ORDER)} or all")
        out.append(s)
    return out


def make_config(args: argparse.Namespace) -> tuple[RunConfig, C.Limits]:
    """RunConfig + Limits from the parsed arguments (smoke preset applied last)."""
    from .io import drive_root
    spec = ModelSpec(name=args.model, revision=args.revision, dtype=args.dtype, device=args.device)
    fams = [f.strip() for f in args.families.split(",") if f.strip()] if args.families else None
    cfg = RunConfig(model=spec, families_dir=args.families_dir, families=fams, seed=int(args.seed),
                    out_root=args.out or drive_root(), limit_variants=args.limit_variants,
                    limit_families=args.limit_families)
    if args.n_random_dirs is not None:
        cfg.n_random_dirs = int(args.n_random_dirs)
    if args.k_levels is not None:
        cfg.k_levels = tuple(int(k) for k in args.k_levels.split(",") if k.strip())
    if args.c_levels is not None:
        cfg.c_levels = tuple(c.strip() for c in args.c_levels.split(",") if c.strip())
    if args.k_extreme is not None:
        cfg.k_extreme = tuple(int(k) for k in args.k_extreme.split(",") if k.strip())
    for name in ("gen_n_new", "gen_n_samples", "tn_rank", "tn_window"):
        val = getattr(args, name, None)
        if val is not None:
            setattr(cfg, name, int(val))
    if args.smoke:
        cfg = C.smoke_config(cfg)
        limits = C.Limits.smoke()
    else:
        limits = C.Limits()
    limits.use_sae = bool(args.use_sae) and not args.smoke
    limits.progress = not args.no_progress
    if args.n_unigram is not None:
        limits.n_unigram = int(args.n_unigram)
    if args.f_stages:
        limits.f_stages = tuple(parse_stages(args.f_stages))
    return cfg, limits


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run_stages(cfg: RunConfig, limits: C.Limits, stages: Sequence[str], mb=None, **kw) -> dict[str, dict]:
    """Run ``stages`` in order on one loaded model; returns ``{stage: {status, run_dir, seconds, error}}``."""
    if mb is None:
        mb = load_model(cfg.model)
    ctx: dict = {}
    report: dict[str, dict] = {}
    for st in stages:
        rd = RunDir(cfg.out_root, st, cfg.model)
        t0 = time.perf_counter()
        print(f"[lsne] stage {st} ({STAGE_TITLES[st]}) -> {rd.path}", flush=True)
        try:
            tables = STAGES[st].run(cfg, mb, rd, ctx=ctx, limits=limits, **kw)
            sub = C.substage_status(rd.path)       # "ok_with_substage_errors" when a substage failed
            report[st] = dict(status=sub["status"], run_dir=str(rd.path), seconds=time.perf_counter() - t0,
                              tables=sorted(tables), error="", n_failed_substages=sub["n_failed_substages"],
                              substage_errors=sub["substage_errors"])
            print(f"[lsne] stage {st} {sub['status']} in {report[st]['seconds']:.1f} s: {', '.join(sorted(tables))}", flush=True)
            for name, err in sub["substage_errors"].items():
                print(f"[lsne]   substage {st}/{name} FAILED: {err}", file=sys.stderr, flush=True)
        except Exception as e:  # noqa: BLE001 - keep the other stages running
            refused = isinstance(e, SDC.ControlRunError)
            if not refused:         # a refused control run wrote nothing; its directory may hold a main run
                C.write_error(rd, st, e, cfg, mb)
                if st == "D" and kw.get("d_control"):
                    SDC.mark_error_manifest(rd, kw["d_control"])
            report[st] = dict(status="error", run_dir=str(rd.path), seconds=time.perf_counter() - t0, tables=[],
                              error=repr(e))
            where = "" if refused else f" (see {rd.file('error.txt')})"
            print(f"[lsne] stage {st} {'REFUSED' if refused else 'FAILED'}: {e!r}{where}", file=sys.stderr, flush=True)
    print(duration_summary(report), flush=True)
    return report


def duration_summary(report: dict[str, dict]) -> str:
    """One line per stage with its wall time and status, plus the total."""
    lines = ["[lsne] durations:"]
    total = 0.0
    for st, r in report.items():
        total += float(r.get("seconds", 0.0))
        lines.append(f"[lsne]   {st}  {r.get('seconds', 0.0):7.1f} s  {r.get('status', '')}")
    lines.append(f"[lsne]   total {total:5.1f} s")
    return "\n".join(lines)


def check_d_control(cfg: RunConfig, stages: Sequence[str], control: Optional[str], main_run: Optional[str]) -> None:
    """``--d-control`` / ``--d-main-run``: both or neither, stage D alone, a run directory
    ``<out>/D/<model>@<rev>`` that is neither the main run's nor one holding another stage-D run, and
    an ok main stage-D run of the same model with its depth bank (the guards of
    :mod:`lsne.experiments.stage_d_control`); SystemExit before the model is loaded."""
    if not (control and main_run):
        raise SystemExit("[lsne] --d-control and --d-main-run go together: a stage-D control run needs the "
                         "directory of the main stage-D run")
    if list(stages) != ["D"]:
        raise SystemExit(f"[lsne] --d-control runs stage D alone (--stage D), got --stage {','.join(stages)}")
    try:
        SDC.check_control(control)
        SDC.check_run_dir(RunDir(cfg.out_root, "D", cfg.model, mkdir=False).path, main_run)
        SDC.main_run_inputs(cfg, main_run)
    except SDC.ControlRunError as e:
        raise SystemExit(f"[lsne] control run refused: {e}") from e


def cmd_run(args: argparse.Namespace) -> int:
    cfg, limits = make_config(args)
    stages = parse_stages(args.stage)
    if args.d_control or args.d_main_run:
        check_d_control(cfg, stages, args.d_control, args.d_main_run)
    kw = dict(axis_npz=args.axis_npz, shard_index=int(args.shard_index), shard_count=int(args.shard_count),
              ttt_corpus=args.ttt_corpus, transplant_source=args.transplant_source, atlas_dir=args.atlas_dir, worst_head=args.worst_head,
              ttt_checkpoints=([s.strip() for s in args.ttt_checkpoints.split(",") if s.strip()] if args.ttt_checkpoints else None),
              d_control=args.d_control, d_main_run=args.d_main_run)
    if args.pairs:
        kw["pairs"] = parse_pairs(args.pairs, cfg)
    elif args.smoke:
        kw["pairs"] = [(cfg.model.name, cfg.model.revision)]
    kw["f_stages"] = list(limits.f_stages)
    print(f"[lsne] model {cfg.model.tag()} device={cfg.model.device} stages={''.join(stages)} out={cfg.out_root}"
          f"{' SMOKE' if limits.is_smoke else ''}"
          f"{f' CONTROL {args.d_control} of {args.d_main_run}' if args.d_control else ''}", flush=True)
    report = run_stages(cfg, limits, stages, **kw)
    summary_path = Path(cfg.out_root) / f"run_report_{cfg.model.tag().replace(':', '_').replace('@', '_')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(dict(cfg=C.to_jsonable(cfg), limits=C.to_jsonable(limits), stages=report),
                                       indent=1), encoding="utf-8")
    # a stage that finished with a failed optional substage wrote its tables: it ran, it is no failure
    n_err = sum(1 for r in report.values() if r["status"] == "error")
    n_sub = sum(1 for r in report.values() if str(r["status"]).endswith("_with_substage_errors"))
    print(f"[lsne] done: {len(report) - n_err}/{len(report)} stages ran ({n_sub} with substage errors); "
          f"report {summary_path}", flush=True)
    return 1 if n_err and n_err == len(report) else 0


# --------------------------------------------------------------------------------------
# merge
# --------------------------------------------------------------------------------------
def _run_info(rel_dir: Path) -> tuple[str, str, str]:
    """``(stage, model_tag, revision)`` from a run directory ``.../<stage>/<model>@<rev>``."""
    parts = rel_dir.parts
    stage = parts[-2] if len(parts) >= 2 else ""
    leaf = parts[-1] if parts else ""
    model, _, rev = leaf.partition("@")
    return stage, model, rev


def merge_runs(root: str | os.PathLike, out: str | os.PathLike) -> dict[str, int]:
    """Concatenate same-named csv files below ``root``; returns ``{name: n_rows}``."""
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[pd.DataFrame]] = {}
    manifests: list[dict] = []
    for p in sorted(root.rglob("*.csv")):
        try:
            if out.resolve() in p.resolve().parents:
                continue
        except OSError:
            pass
        rel_dir = p.parent.relative_to(root)
        stage, model, rev = _run_info(rel_dir)
        try:
            df = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        df.insert(0, "run_revision", rev)
        df.insert(0, "run_model", model)
        df.insert(0, "run_stage", stage)
        df.insert(0, "run_id", str(rel_dir).replace(os.sep, "/"))
        groups.setdefault(p.stem, []).append(df)
    for p in sorted(root.rglob("manifest.json")):
        try:
            m = read_manifest(p.parent)
            m["run_id"] = str(p.parent.relative_to(root)).replace(os.sep, "/")
            manifests.append(m)
        except Exception:  # noqa: BLE001
            continue
    counts: dict[str, int] = {}
    for name, frames in groups.items():
        merged = pd.concat(frames, ignore_index=True, sort=False)
        merged.to_csv(out / f"{name}.csv", index=False)
        counts[name] = int(len(merged))
    (out / "manifests.json").write_text(json.dumps(manifests, indent=1, ensure_ascii=False), encoding="utf-8")
    index = pd.DataFrame([dict(table=k, n_rows=v, n_runs=len(groups[k])) for k, v in sorted(counts.items())],
                         columns=["table", "n_rows", "n_runs"])
    index.to_csv(out / "merge_index.csv", index=False)
    return counts


def cmd_merge(args: argparse.Namespace) -> int:
    counts = merge_runs(args.root, args.out)
    print(f"[lsne] merged {len(counts)} tables into {args.out}", flush=True)
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v} rows")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "merge":
        return cmd_merge(args)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
