"""Stage F - scale and checkpoint sweep (DESIGN section 7).

Loops over ``(model, revision)`` pairs (``kw['pairs']``; default: the State-2143
intervention checkpoints of ``cfg.model.name``) and runs the stages in
``Limits.f_stages`` (default B, C, D with the *limited* flag) with a **fresh**
``load_model`` per pair.  Every pair gets its own sub-run directory

    <F run dir>/shard<i>of<n>/<stage>/<model>@<revision>/

The work unit is the triple ``(model, revision, family_id)``; ``shard_index`` /
``shard_count`` (``kw`` or ``--shard-index/--shard-count``) partition the sorted triples
deterministically (``index % shard_count == shard_index``), so the shards of
``jobs/shards.yaml`` never overlap and ``lsne.cli merge`` joins them.

Tables
------
plan   every triple with its shard assignment and whether this process ran it
runs   one row per (pair, stage): status, runtime, tables written, error message
"""
from __future__ import annotations

import gc
import os
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

from ..config import CHECKPOINT_TRACE, STATE2143, RunConfig
from ..io import RunDir
from ..model import ModelBundle, load_model
from . import common as C


def default_pairs(cfg: RunConfig) -> list[tuple[str, str]]:
    """The State-2143 intervention checkpoints (step98000, step99000) of ``cfg.model.name``."""
    return [(cfg.model.name, str(r)) for r in STATE2143["checkpoints_intervention"]]


def parse_pairs(spec: Optional[str], cfg: RunConfig) -> list[tuple[str, str]]:
    """``"model@rev,model@rev"`` (a bare ``rev`` uses ``cfg.model.name``), ``"trace"`` for
    the 27-checkpoint trace of pythia-160m, ``"self"`` for ``cfg.model`` itself, or
    ``None`` for :func:`default_pairs`."""
    if not spec:
        return default_pairs(cfg)
    if spec == "trace":
        return [(STATE2143["model"], r) for r in CHECKPOINT_TRACE]
    if spec == "self":
        return [(cfg.model.name, cfg.model.revision)]
    out = []
    for item in str(spec).split(","):
        item = item.strip()
        if not item:
            continue
        if "@" in item:
            m, r = item.rsplit("@", 1)
            out.append((m.strip(), r.strip()))
        else:
            out.append((cfg.model.name, item))
    return out


def plan_triples(pairs: Sequence[tuple[str, str]], families: Sequence[str], shard_index: int = 0,
                 shard_count: int = 1) -> pd.DataFrame:
    """Sorted ``(model, revision, family)`` triples with a deterministic shard assignment."""
    triples = sorted({(m, r, f) for (m, r) in pairs for f in families})
    rows = []
    for i, (m, r, f) in enumerate(triples):
        rows.append(dict(index=i, model=m, revision=r, family=f, shard=i % int(shard_count),
                         mine=(i % int(shard_count)) == int(shard_index)))
    return pd.DataFrame(rows, columns=["index", "model", "revision", "family", "shard", "mine"])


def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Sweep over model checkpoints / sizes; see the module docstring."""
    from . import STAGES   # local import: the package imports this module

    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    sub_limits = limits.as_limited()
    pairs = kw.get("pairs")
    if pairs is None:
        pairs = default_pairs(cfg) if not limits.smoke else [(cfg.model.name, cfg.model.revision)]
    pairs = [(str(m), str(r)) for m, r in pairs]
    stages = tuple(kw.get("f_stages") or limits.f_stages)
    shard_index, shard_count = int(kw.get("shard_index", 0)), int(kw.get("shard_count", 1))
    families = sorted({f.family_id for f in C.select_families(cfg)})
    plan = plan_triples(pairs, families, shard_index, shard_count)
    mine = plan[plan["mine"]]
    sub_root = Path(os.fspath(run_dir)) / f"shard{shard_index}of{shard_count}"

    rows = []
    done_pairs = list(dict.fromkeys((m, r) for m, r in zip(mine["model"], mine["revision"])))
    for (model, revision) in C.progress(done_pairs, "F pairs", enabled=limits.progress):
        fams = sorted(mine[(mine["model"] == model) & (mine["revision"] == revision)]["family"].tolist())
        spec = replace(cfg.model, name=model, revision=revision)
        sub_cfg = replace(cfg, model=spec, families=fams)
        t_load = time.perf_counter()
        try:
            mb2 = load_model(spec)
            load_s = time.perf_counter() - t_load
        except Exception as e:  # noqa: BLE001 - e.g. a checkpoint that does not exist
            rows.append(dict(model=model, revision=revision, stage="load", status="error", seconds=time.perf_counter() - t_load,
                             n_tables=0, families=",".join(fams), error=repr(e), run_dir=""))
            continue
        ctx: dict = {}
        for st in stages:
            rd = RunDir(sub_root, st, spec)
            t_st = time.perf_counter()
            try:
                tables = STAGES[st].run(sub_cfg, mb2, rd, ctx=ctx, limits=sub_limits, sae=kw.get("sae"))
                rows.append(dict(model=model, revision=revision, stage=st, status="ok", seconds=time.perf_counter() - t_st,
                                 n_tables=len(tables), families=",".join(fams), error="", run_dir=str(rd.path)))
            except Exception as e:  # noqa: BLE001 - keep going with the next stage / pair
                C.write_error(rd, st, e, sub_cfg, mb2)
                rows.append(dict(model=model, revision=revision, stage=st, status="error", seconds=time.perf_counter() - t_st,
                                 n_tables=0, families=",".join(fams), error=traceback.format_exception_only(type(e), e)[-1].strip(),
                                 run_dir=str(rd.path)))
        rows.append(dict(model=model, revision=revision, stage="load", status="ok", seconds=load_s, n_tables=0,
                         families=",".join(fams), error="", run_dir=""))
        del mb2, ctx
        gc.collect()
    runs = pd.DataFrame(rows, columns=["model", "revision", "stage", "status", "seconds", "n_tables", "families", "error", "run_dir"])
    tables = {"plan": plan, "runs": runs}
    return C.finish_stage(run_dir, "F", cfg, mb, tables, limits=limits, t0=t0, pairs=pairs, stages=list(stages),
                          shard_index=shard_index, shard_count=shard_count, n_triples=int(len(plan)),
                          n_mine=int(len(mine)), sub_root=str(sub_root))
