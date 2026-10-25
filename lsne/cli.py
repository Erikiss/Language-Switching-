"""Command-line entry point.

    python -m lsne.cli run --stage A|B|C|D|E|F|all[,...] --model EleutherAI/pythia-160m \\
        --revision step98000 --dtype float32 --device auto --families F01,F02 --out runs \\
        [--limit-variants N] [--limit-families N] [--smoke] [--use-sae] [--seed S] \\
        [--pairs model@rev,...|trace|self] [--f-stages B,C,D] [--shard-index I --shard-count N]
    python -m lsne.cli merge --root runs --out merged/

``run`` loads the model once, runs the requested stages in order (``all`` = A B C D E F)
and shares the stimulus bank / direction bank between them.  Every stage is wrapped in
``try/except``: a failing stage writes ``error.txt`` and a manifest with
``status='error'`` into its run directory and the next stage still runs, so a Colab run
always yields partial results.  ``--smoke`` sets every limit for a < 3 min CPU run with
pythia-14m (one family, three variants, tiny generation budgets; stage F re-runs the
same model/revision only).

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
from .experiments.stage_f_scale import parse_pairs


# --------------------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lsne.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run experiment stages")
    r.add_argument("--stage", default="all", help="A|B|C|D|E|F|all or a comma-separated list (default: all)")
    r.add_argument("--model", default=RunConfig().model.name)
    r.add_argument("--revision", default=RunConfig().model.revision)
    r.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    r.add_argument("--device", default="auto")
    r.add_argument("--families", default=None, help="comma-separated family ids (F01) or tags (F01A); default: all")
    r.add_argument("--families-dir", default=RunConfig().families_dir)
    r.add_argument("--out", default=None, help="output root (default: lsne.io.drive_root(), i.e. Drive on Colab or ./runs)")
    r.add_argument("--limit-variants", type=int, default=None)
    r.add_argument("--limit-families", type=int, default=None)
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
            report[st] = dict(status="ok", run_dir=str(rd.path), seconds=time.perf_counter() - t0,
                              tables=sorted(tables), error="")
            print(f"[lsne] stage {st} ok in {report[st]['seconds']:.1f} s: {', '.join(sorted(tables))}", flush=True)
        except Exception as e:  # noqa: BLE001 - keep the other stages running
            C.write_error(rd, st, e, cfg, mb)
            report[st] = dict(status="error", run_dir=str(rd.path), seconds=time.perf_counter() - t0, tables=[],
                              error=repr(e))
            print(f"[lsne] stage {st} FAILED: {e!r} (see {rd.file('error.txt')})", file=sys.stderr, flush=True)
    return report


def cmd_run(args: argparse.Namespace) -> int:
    cfg, limits = make_config(args)
    stages = parse_stages(args.stage)
    kw = dict(axis_npz=args.axis_npz, shard_index=int(args.shard_index), shard_count=int(args.shard_count))
    if args.pairs:
        kw["pairs"] = parse_pairs(args.pairs, cfg)
    elif args.smoke:
        kw["pairs"] = [(cfg.model.name, cfg.model.revision)]
    kw["f_stages"] = list(limits.f_stages)
    print(f"[lsne] model {cfg.model.tag()} device={cfg.model.device} stages={''.join(stages)} out={cfg.out_root}"
          f"{' SMOKE' if limits.smoke else ''}", flush=True)
    report = run_stages(cfg, limits, stages, **kw)
    summary_path = Path(cfg.out_root) / f"run_report_{cfg.model.tag().replace(':', '_').replace('@', '_')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(dict(cfg=C.to_jsonable(cfg), limits=C.to_jsonable(limits), stages=report),
                                       indent=1), encoding="utf-8")
    n_err = sum(1 for r in report.values() if r["status"] != "ok")
    print(f"[lsne] done: {len(report) - n_err}/{len(report)} stages ok; report {summary_path}", flush=True)
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
