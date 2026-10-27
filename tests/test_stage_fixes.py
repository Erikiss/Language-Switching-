"""Stage-level fixes after the review of the pythia-160m@step98000 runs (CPU, pythia-14m).

* common: ``Limits.n_boot`` = 10 000 (smoke 200, scale sweep 2 000), a stale ``error.txt``
  is removed when a stage ends ok, failed substages reach the run report and the notebook
  summary json, the pre-registered split (DEV01-DEV04 / F01-F12 x {A, B});
* stage C: family-mean bootstrap rows per split, paired T(8, C3) - T(8, C1) and
  B(ent) - mean B(rand);
* stage D: ``damage_boot`` (primary endpoint per family and pooled) and the audit summary;
* stage G: T3 layer choice, stratified n-matching, NaN contraction layer, split bootstraps;
* stage H: constant censored series, ``sign_as_expected`` on NaN means, DEV / test split;
* stage I: endpoint 14 at equal absolute ||delta||, endpoint 13 flag on one dose and the CI
  rule;
* stage J: the rule-16 counterpole vs the evaluated pole;
* stage L: layer-consistent transplant directions, round-robin stand-in prompts, defaults.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import RunConfig
from lsne.experiments import common as C

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------
# common
# --------------------------------------------------------------------------------------
def test_limits_n_boot_full_smoke_and_sweep():
    """PREREGISTRATION section 2: 10 000 bootstrap draws in full runs; the smoke preset keeps
    a small value and the scale sweep (stage F) keeps the 2 000 draws it was started with."""
    assert C.Limits().n_boot == 10000
    assert C.Limits.smoke().n_boot == 200
    assert C.Limits().as_limited().n_boot == C.LIMITED_N_BOOT == 2000
    assert C.Limits.smoke().as_limited().n_boot == 200


def test_prereg_split_and_mask():
    assert C.prereg_split("DEV03", "D") == "dev" and C.prereg_split("F01", "A") == "test"
    assert C.prereg_split("F12", "B") == "test" and C.prereg_split("F12", "C") == "other"
    assert C.prereg_split("F13", "A") == "other" and C.prereg_split("orig2143", "orig") == "other"
    assert C.prereg_split("F05") == "test"
    df = pd.DataFrame(dict(family=["F01", "F01", "F01", "DEV01", "F14"], formulation=["A", "B", "C", "D", "A"]))
    assert C.split_mask(df, "all").tolist() == [True] * 5
    assert C.split_mask(df, "test").tolist() == [True, True, False, False, False]
    assert C.split_mask(df, "dev").tolist() == [False, False, False, True, False]
    assert C.split_mask(df[["family"]], "test").tolist() == [True, True, True, False, False]
    with pytest.raises(ValueError):
        C.split_mask(df, "train")
    rows = C.split_boot_rows(df.assign(x=[1.0, 3.0, 100.0, 7.0, 9.0]), "x", 200, 0, splits=C.SPLITS, quantity="x")
    by = {r["split"]: r for r in rows}
    assert set(by) == {"all", "test", "dev"} and by["test"]["mean"] == pytest.approx(2.0)
    assert by["test"]["n_clusters"] == 1 and by["test"]["n_rows"] == 2 and by["dev"]["mean"] == pytest.approx(7.0)
    assert C.split_boot_rows(df.assign(x=1.0)[df["family"] == "F14"], "x", 50, 0, splits=("test",)) == []
    # the pooled row stays even when the statistic is undefined (no cluster), the other splits do not
    undefined = C.split_boot_rows(df.assign(x=np.nan), "x", 50, 0, splits=C.SPLITS)
    assert [r["split"] for r in undefined] == ["all"] and undefined[0]["n_clusters"] == 0


def _stage_dir(tmp_path, mb, stage="X"):
    from lsne.io import RunDir
    return RunDir(tmp_path, stage, mb.spec)


def test_finish_stage_removes_stale_error_txt(mb, tmp_path):
    rd = _stage_dir(tmp_path, mb)
    cfg = RunConfig(model=mb.spec, out_root=str(tmp_path))
    try:
        raise RuntimeError("old OOM")
    except RuntimeError as e:
        C.write_error(rd, "X", e, cfg, mb)
    err = rd.path / "error.txt"
    assert err.exists() and not C.clear_stale_error(rd.path)      # manifest says error: kept
    C.finish_stage(rd, "X", cfg, mb, {"t": pd.DataFrame(dict(a=[1]))}, limits=C.Limits.smoke())
    assert not err.exists()
    assert json.loads((rd.path / "manifest.json").read_text(encoding="utf-8"))["status"] == "ok"
    assert not C.clear_stale_error(rd.path)                        # nothing left to remove


def test_substage_status_and_stale_substage_file(tmp_path):
    from lsne.io import write_manifest
    d = tmp_path / "J"
    sub = C.Substages("J", verbose=False)
    with pytest.warns(UserWarning):
        sub.run("sharing", lambda: (_ for _ in ()).throw(RuntimeError("no SAE available")), default=None)
    sub.write_tracebacks(d)
    write_manifest(d, stage="J", status="ok", substages=sub.report())
    st = C.substage_status(d)
    assert st["status"] == "ok_with_substage_errors" and st["n_failed_substages"] == 1
    assert "no SAE available" in st["substage_errors"]["sharing"]
    assert C.substage_status(d, "reused")["status"] == "reused_with_substage_errors"
    assert C.substage_status(d, "error")["status"] == "error"
    # a clean rerun removes the stale traceback file and reads plain ok
    assert (d / "substage_errors.txt").exists()
    clean = C.Substages("J", verbose=False)
    clean.run("sharing", lambda: 1)
    assert clean.write_tracebacks(d) is None and not (d / "substage_errors.txt").exists()
    write_manifest(d, stage="J", status="ok", substages=clean.report())
    assert C.substage_status(d) == dict(status="ok", n_failed_substages=0, substage_errors={})
    assert C.substage_status(tmp_path / "missing")["status"] == "ok"


class HalfBroken:
    """Stage A stub that finishes ok with one failed optional substage (like stage J without an SAE)."""
    @staticmethod
    def run(cfg, mb, run_dir, **kw):
        sub = C.Substages("A", verbose=False)
        sub.run("good", lambda: 1)
        sub.run("bad", lambda: 1 / 0)
        sub.write_tracebacks(run_dir)
        return C.finish_stage(run_dir, "A", cfg, mb, {"t": pd.DataFrame(dict(a=[1]))}, substages=sub.report())


class Broken:
    @staticmethod
    def run(cfg, mb, run_dir, **kw):
        raise RuntimeError("stage failed")


def test_cli_report_names_failed_substages(mb, tmp_path, monkeypatch):
    from lsne import cli

    monkeypatch.setitem(cli.STAGES, "A", HalfBroken)
    cfg = C.smoke_config(RunConfig(model=mb.spec, families=["DEV01"], out_root=str(tmp_path)))
    with pytest.warns(UserWarning):
        report = cli.run_stages(cfg, C.Limits.smoke(progress=False), ["A"], mb=mb)
    assert report["A"]["status"] == "ok_with_substage_errors" and report["A"]["n_failed_substages"] == 1
    assert "ZeroDivisionError" in report["A"]["substage_errors"]["bad"]


def test_cli_exit_code_counts_hard_failures_only(mb, tmp_path, monkeypatch, capsys):
    """``python -m lsne.cli run``: a stage that wrote its tables with one failed optional
    substage ran (exit 0, the summary names the substage errors); only stages that failed
    outright count as failures (exit 1 when every requested stage failed)."""
    from lsne import cli

    monkeypatch.setattr(cli, "load_model", lambda spec: mb)
    argv = ["run", "--stage", "A", "--model", mb.spec.name, "--revision", mb.spec.revision, "--device", "cpu",
            "--families", "DEV01", "--smoke", "--no-progress"]
    monkeypatch.setitem(cli.STAGES, "A", HalfBroken)
    with pytest.warns(UserWarning):
        rc = cli.main(argv + ["--out", str(tmp_path / "half")])
    out = capsys.readouterr().out
    assert rc == 0 and "done: 1/1 stages ran (1 with substage errors)" in out
    rep = json.loads(next((tmp_path / "half").glob("run_report_*.json")).read_text(encoding="utf-8"))
    assert rep["stages"]["A"]["status"] == "ok_with_substage_errors"
    monkeypatch.setitem(cli.STAGES, "A", Broken)
    rc = cli.main(argv + ["--out", str(tmp_path / "broken")])
    assert rc == 1 and "done: 0/1 stages ran (0 with substage errors)" in capsys.readouterr().out


def _builder():
    spec = importlib.util.spec_from_file_location("build_notebooks", REPO / "scripts" / "build_notebooks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cell_source(cell) -> str:
    return cell[1] if isinstance(cell, tuple) else cell.source


def test_notebook_summary_reports_substage_errors(mb, tmp_path):
    """The summary cell of notebooks 00-11 reads the substage report of every stage it ran or
    reused: a failed substage turns the stage status and ``status_overall`` away from ok."""
    import contextlib
    import os
    from lsne.io import write_manifest
    builder = _builder()
    src = _cell_source(builder.summary_cell())
    run_dir = tmp_path / "runs" / "J"
    write_manifest(run_dir, stage="J", status="ok",
                   substages=dict(errors={"sharing": "RuntimeError('no SAE available')"}, seconds={}, n_failed=1))
    fig_dir = tmp_path / "fig"
    fig_dir.mkdir()
    ns = dict(guarded=lambda title: contextlib.nullcontext(), NOTEBOOK="09_rdx_sharing", STAGE="J", spec=mb.spec, mb=mb,
              SMOKE=True, FAMILIES=["DEV01"], SEED=0, OUT_ROOT=str(tmp_path), RUNS_ROOT=str(tmp_path / "runs"),
              CLI_COMMAND="python -m lsne.cli run --stage J", report={"J": dict(status="ok", run_dir=str(run_dir),
                                                                                 seconds=1.0, error="")},
              results={"J": {"sharing": pd.DataFrame()}}, FIG_DIR=str(fig_dir), C=C, json=json, os=os)
    exec(compile(src, "summary_cell", "exec"), ns)
    s = json.loads(next(tmp_path.glob("summary_09_rdx_sharing_*.json")).read_text(encoding="utf-8"))
    assert s["status_overall"] == "substage_errors" and "sharing" in s["substage_errors"]["J"]
    assert s["report"]["J"]["status"] == "ok_with_substage_errors" and s["report"]["J"]["n_failed_substages"] == 1


def test_notebook_runner_clears_stale_error_and_lists_substages():
    builder = _builder()
    src = _cell_source(builder.runner_cell())
    assert "C.clear_stale_error(rd.path)" in src and "C.substage_status(rd.path, status)" in src
    assert 'stage_status(stage, rd, "reused")' in src and 'stage_status(stage, rd, "ok")' in src


# --------------------------------------------------------------------------------------
# stage C
# --------------------------------------------------------------------------------------
def _grid_summary(seed: int = 0) -> pd.DataFrame:
    """Synthetic ``grid_summary``: 3 test families, one family outside both splits, DEV01;
    formulations A-D; K=0, K=8/C1, K=8/C3; directions ent / rand; eps 0 and 0.05."""
    rng = np.random.default_rng(seed)
    rows = []
    for fam in ("F01", "F02", "F03", "F13", "DEV01"):
        for form in "ABCD":
            for K, Cc in ((0, "C1"), (8, "C1"), (8, "C3")):
                lp0 = float(rng.normal(-2.0 - 0.5 * K / 8 - (0.3 if Cc == "C3" else 0.0), 0.2))
                for dfam in ("ent", "rand"):
                    for sign in (1, -1):
                        for eps in (0.0, 0.05):
                            B = 0.0 if eps == 0 else float(rng.normal(0.3 if dfam == "ent" else 0.0, 0.1))
                            rows.append(dict(tag=f"{fam}{form}:K{K}:{Cc}", family=fam, formulation=form, K=K, C=Cc,
                                             kind="standard", n_tokens=100, tail_start=80, dir_family=dfam, sign=sign,
                                             eps=eps, B=B, B_sd=0.0, n_dirs=1, logp_0=lp0, logp_delta=lp0 + B,
                                             entropy=1.0, p_freq=0.1))
    return pd.DataFrame(rows)


def _family_mean(df: pd.DataFrame, col: str) -> float:
    return float(df.groupby("family")[col].mean().mean())


def _test_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["family"].isin(["F01", "F02", "F03"]) & df["formulation"].isin(["A", "B"])]


def test_stage_c_boot_splits_and_paired_rows():
    from lsne.experiments.stage_c_grid import BOOT_COLS, bootstrap_contrasts, bootstrap_paired, contrast_rows
    summ = _grid_summary()
    con = contrast_rows(summ)
    boot = pd.concat([bootstrap_contrasts(con, 500, 0), bootstrap_paired(con, summ, 500, 0)], ignore_index=True)
    assert list(boot.columns) == BOOT_COLS
    t = con[(con["eps"] == 0) & (con["sign"] == 1)].drop_duplicates("tag")
    t8 = t[(t["K"] == 8) & (t["C"] == "C1")]

    T = boot[(boot["quantity"] == "T") & (boot["K"] == 8) & (boot["C"] == "C1")].set_index("split")
    assert set(T.index) == {"all", "test", "dev"}
    # the pooled row is unchanged: every standard family, row level
    assert T.loc["all", "stat_level"] == "row" and T.loc["all", "mean"] == pytest.approx(float(t8["T"].mean()))
    assert T.loc["all", "n_clusters"] == 5 and T.loc["all", "n_rows"] == 20
    # the confirmation set: F01-F12 x {A, B}, mean of family means (F13 and formulations C / D left out)
    assert T.loc["test", "stat_level"] == "cluster" and T.loc["test", "n_clusters"] == 3 and T.loc["test", "n_rows"] == 6
    assert T.loc["test", "mean"] == pytest.approx(_family_mean(_test_rows(t8), "T"))
    assert T.loc["test", "ci_low"] <= T.loc["test", "mean"] <= T.loc["test", "ci_high"]
    assert T.loc["dev", "n_clusters"] == 1 and T.loc["dev", "n_rows"] == 4
    # preregistered = the cell (every split), confirmatory = the cell on the confirmation set
    assert T["preregistered"].astype(bool).all()
    assert bool(T.loc["test", "confirmatory"]) and not T.loc[["all", "dev"], "confirmatory"].astype(bool).any()
    assert not boot[(boot["quantity"] == "T") & (boot["C"] == "C3")]["preregistered"].astype(bool).any()
    assert (boot["confirmatory"] == (boot["preregistered"] & (boot["split"] == "test"))).all()
    assert T["p_boot"].between(0, 1).all()

    # contrast 2: T(8, C3) - T(8, C1) paired per formulation
    d2 = boot[boot["quantity"] == "T_C3_minus_C1"].set_index("split")
    assert set(d2.index) == {"all", "test", "dev"} and (d2["K"] == 8).all() and (d2["C"] == "C3-C1").all()
    p = t[t["K"] == 8].pivot_table(index=["family", "formulation"], columns="C", values="T").reset_index()
    p["d"] = p["C3"] - p["C1"]
    assert d2.loc["test", "mean"] == pytest.approx(_family_mean(_test_rows(p), "d"))
    assert d2.loc["all", "mean"] == pytest.approx(_family_mean(p, "d")) and d2.loc["all", "stat_level"] == "cluster"
    assert d2.loc["test", "n_rows"] == 6 and bool(d2.loc["test", "confirmatory"]) and not bool(d2.loc["all", "confirmatory"])

    # contrast 6: B(ent) - mean B(rand) of the same variant, per (K, C, sign, eps) and pooled
    b6 = boot[boot["quantity"] == "B_ent_minus_rand"]
    assert set(b6["eps"]) == {0.05} and set(b6["sign"]) == {1, -1} and set(b6["dir_family"]) == {"ent"}
    pooled = b6[(b6["K"] == -1) & (b6["C"] == "all") & (b6["sign"] == 1)].set_index("split")
    s = summ[(summ["eps"] == 0.05) & (summ["sign"] == 1)]
    ent, rnd = s[s["dir_family"] == "ent"].set_index("tag"), s[s["dir_family"] == "rand"].set_index("tag")
    dd = ent[["family", "formulation"]].assign(dB=ent["B"] - rnd["B"]).reset_index()
    assert pooled.loc["test", "mean"] == pytest.approx(_family_mean(_test_rows(dd), "dB"))
    assert pooled.loc["test", "n_rows"] == 18 and bool(pooled.loc["test", "confirmatory"])
    cell = b6[(b6["K"] == 0) & (b6["C"] == "C1") & (b6["sign"] == 1) & (b6["split"] == "test")]
    assert len(cell) == 1 and not bool(cell["preregistered"].iloc[0]) and cell["n_rows"].iloc[0] == 6


def test_stage_c_attribution_boot_splits():
    """Contrast 3: R_excess rows per split, ``confirmatory`` on the delim test rows only."""
    from lsne.experiments.stage_c_grid import bootstrap_attribution
    keys = [(f, fo, l, c) for f in ("F01", "F14", "DEV02") for fo in "AC" for l in (2, 5) for c in ("delim", "punct")]
    att = pd.DataFrame([dict(family=f, formulation=fo, K=8, C="C1", kind="standard", layer=l, **{"class": c},
                             R_excess=float(i)) for i, (f, fo, l, c) in enumerate(keys)])
    ra = bootstrap_attribution(att, 200, 0)
    assert set(ra["split"]) == {"all", "test", "dev"} and len(ra) == 3 * 4
    tr = ra[ra["split"] == "test"]
    assert (tr["n_rows"] == 1).all() and (tr["preregistered"] == (tr["class"] == "delim")).all()
    assert not ra[ra["split"] != "test"]["confirmatory"].any() and (tr["confirmatory"] == tr["preregistered"]).all()
    assert (ra[ra["split"] == "all"]["n_rows"] == 6).all() and (ra[ra["split"] == "dev"]["n_rows"] == 2).all()


def test_min_effect_verdict_uses_the_interval_only():
    assert C.min_effect_verdict(0.06, 0.2, 0.05) == (True, "supported")
    assert C.min_effect_verdict(0.01, 0.2, 0.05) == (False, "positive_below_min")   # point estimate may exceed 0.05
    assert C.min_effect_verdict(-0.01, 0.2, 0.05) == (False, "open")
    assert C.min_effect_verdict(-0.08, 0.003, 0.05) == (False, "null")
    assert C.min_effect_verdict(-0.08, -0.01, 0.05) == (False, "negative")
    assert C.min_effect_verdict(float("nan"), 0.1, 0.05) == (None, "")
    assert set(C.VERDICTS) == {"supported", "positive_below_min", "open", "null", "negative"}


def test_audit_summary():
    a = C.audit_summary(pd.DataFrame(dict(audit_rel_err=[0.0, 1e-7, 2e-4, float("nan")])))
    assert a["n_rows"] == 4 and a["n_finite"] == 3 and a["max"] == pytest.approx(2e-4)
    assert a["n_over_1e-4"] == 1 and a["valid"] is False and a["q50"] == pytest.approx(1e-7)
    e = C.audit_summary(pd.DataFrame())
    assert e["n_rows"] == 0 and e["valid"] is True and np.isnan(e["max"])


# --------------------------------------------------------------------------------------
# stage D
# --------------------------------------------------------------------------------------
def _damage(seed: int = 0) -> pd.DataFrame:
    """Synthetic D1 table: lambda(lang_dom) rises by 0.1 * K / 8 at C1 (0.02 at C3), the random
    directions by 0.01; families F01/F02 (test), F15 (neither split), DEV01 (dev)."""
    rng = np.random.default_rng(seed)
    rows = []
    for fam in ("F01", "F02", "F15", "DEV01"):
        for form in "ABC":
            base = float(rng.normal(0.05, 0.01))
            for K, Cc in ((0, "C1"), (4, "C1"), (8, "C1"), (8, "C3")):
                for inject in ("depth", "dossier"):
                    for name in ("lang_dom", "rand_00", "rand_01", "ent"):
                        lift = (0.1 if Cc == "C1" else 0.02) * K / 8 if name == "lang_dom" else 0.01 * (K > 0)
                        rows.append(dict(tag=f"{fam}{form}:K{K}:{Cc}", family=fam, formulation=form, K=K, C=Cc,
                                         kind="standard", inject=inject, direction=name, dir_family=C.dir_family(name),
                                         eps=0.05, lam=base + lift + float(rng.normal(0, 0.002)), audit_rel_err=0.0))
    return pd.DataFrame(rows)


def test_stage_d_damage_contrasts_and_boot():
    from lsne.experiments.stage_d_noneq import DAMAGE_BOOT_COLS, damage_boot, damage_contrasts
    dmg = _damage()
    con = damage_contrasts(dmg)
    assert set(con["inject"]) == {"depth"} and len(con) == 4 * 3 * 3 and (con["n_rand"] == 2).all()
    r = con[con["tag"] == "F01A:K8:C1"].iloc[0]
    d = dmg[(dmg["inject"] == "depth") & (dmg["family"] == "F01") & (dmg["formulation"] == "A")].set_index(["tag", "direction"])["lam"]
    assert r["dlam_lang_dom"] == pytest.approx(d[("F01A:K8:C1", "lang_dom")] - d[("F01A:K0:C1", "lang_dom")])
    rand = lambda t: (d[(t, "rand_00")] + d[(t, "rand_01")]) / 2   # noqa: E731
    assert r["dlam_rand"] == pytest.approx(rand("F01A:K8:C1") - rand("F01A:K0:C1"))
    assert r["did_lang_dom_minus_rand"] == pytest.approx(r["dlam_lang_dom"] - r["dlam_rand"])
    assert r["split"] == "test" and not r["position_deviation"]
    assert con[con["C"] == "C3"]["position_deviation"].all()
    assert set(con[con["family"] == "F01"]["split"]) == {"test", "other"}       # formulation C is not in the test set

    boot = damage_boot(con, 500, 0)
    assert list(boot.columns) == DAMAGE_BOOT_COLS and (boot["stat_level"] == "cluster").all()
    t = boot[(boot["split"] == "test") & (boot["quantity"] == "dlam_lang_dom") & (boot["K"] == 8) & (boot["C"] == "C1")]
    assert len(t) == 1 and t["n_clusters"].iloc[0] == 2 and t["n_rows"].iloc[0] == 4
    test_rows = con[(con["split"] == "test") & (con["K"] == 8) & (con["C"] == "C1")]
    assert t["mean"].iloc[0] == pytest.approx(test_rows.groupby("family")["dlam_lang_dom"].mean().mean())
    assert bool(t["confirmatory"].iloc[0]) and t["verdict"].iloc[0] in C.VERDICTS
    assert (boot["preregistered"] == ((boot["C"] == "C1") & (boot["quantity"] != "dlam_rand"))).all()
    pre = boot[boot["confirmatory"].astype(bool)]
    assert set(pre["split"]) == {"test"} and set(pre["C"]) == {"C1"}
    assert set(pre["quantity"]) == {"dlam_lang_dom", "did_lang_dom_minus_rand"}
    assert (boot[boot["C"] == "C3"]["position_deviation"]).all() and not boot[boot["C"] == "C1"]["position_deviation"].any()
    # the CI rule, not the point estimate: min_effect_met only with ci_low >= 0.05
    met = boot[boot["quantity"] != "dlam_rand"]
    assert (met["min_effect_met"].astype(bool) == (met["ci_low"] >= 0.05)).all()
    assert boot[boot["quantity"] == "dlam_rand"]["min_effect_met"].isna().all()
    # lambda(lang_dom) rises with K at C1: monotone over K = 4, 8 on every split
    mono = boot[(boot["quantity"] == "dlam_lang_dom") & (boot["C"] == "C1")]
    assert mono["monotone_in_K"].astype(bool).all() and boot[boot["C"] == "C3"]["monotone_in_K"].isna().all()
    assert set(boot["split"]) == {"all", "test", "dev"}
    assert damage_boot(damage_contrasts(pd.DataFrame()), 10, 0).empty


def test_stage_d_resume_signature_ignores_later_default_changes():
    """Changing the defaults of n_boot / l_n_prompts / l_eps must not discard the checkpoints of
    a stage-D run started before (the limits enter the resume signature)."""
    from dataclasses import asdict, replace
    from lsne.experiments.stage_d_noneq import RESUME_PINNED_LIMITS

    def sig_limits(lim):
        return {k: RESUME_PINNED_LIMITS.get(k, v) for k, v in asdict(lim).items() if k not in ("progress", "damage_batch_size")}

    new = C.Limits().as_limited()
    old = replace(new, n_boot=2000, l_n_prompts=3, l_eps=1.0)          # the defaults the running sweep was started with
    assert sig_limits(new) == sig_limits(old) == {k: v for k, v in asdict(old).items()
                                                  if k not in ("progress", "damage_batch_size")}


# --------------------------------------------------------------------------------------
# stage G
# --------------------------------------------------------------------------------------
def test_depth_trajectory_layer_of_max_contraction_needs_a_negative_increment():
    from lsne.trajectory import depth_trajectory
    rows = [dict(site=f"L{l}.block_input", readout="mean", k=5, family_id="F01", K=K, C="C1", log_volume=lv)
            for K, series in ((8, [0.0, 1.0, 3.0, 3.5]), (0, [0.0, -1.0, 2.0, 1.5])) for l, lv in enumerate(series)]
    dt = depth_trajectory(pd.DataFrame(rows))
    grow, mixed = dt[dt["K"] == 8], dt[dt["K"] == 0]
    assert grow["layer_of_max_contraction"].isna().all()                  # nothing contracts: no layer, not the smallest rise
    assert (grow["layer_of_max_expansion"] == 2).all()
    assert (mixed["layer_of_max_contraction"] == 1).all() and (mixed["layer_of_max_expansion"] == 2).all()
    shrink = depth_trajectory(pd.DataFrame([dict(r, log_volume=-r["log_volume"]) for r in rows if r["K"] == 8]))
    assert shrink["layer_of_max_expansion"].isna().all() and (shrink["layer_of_max_contraction"] == 2).all()


def _cloud_groups(rng, fams=("F01",), n_forms=4, n_masks=2, d=6):
    """K=0 clouds of one state per formulation; K=8 clouds whose realizations of one
    formulation coincide (so any stratified draw gives the same matched cloud)."""
    from types import SimpleNamespace
    groups, by_group = {}, {}
    forms = "ABCD"[:n_forms]
    for fam in fams:
        groups[(fam, 0, "C1")] = [SimpleNamespace(formulation=f) for f in forms]
        by_group[(fam, 0, "C1")] = rng.normal(size=(n_forms, d))
        P = rng.normal(size=(n_forms, d)) * 3.0
        groups[(fam, 8, "C1")] = [SimpleNamespace(formulation=f) for f in forms for _ in range(n_masks)]
        by_group[(fam, 8, "C1")] = np.repeat(P, n_masks, axis=0)
    return groups, by_group


def test_contraction_n_matching_is_stratified_by_formulation():
    from lsne.experiments.stage_g_geometry import contraction_tables, stratified_draw
    from lsne.geometry import log_volume
    rng = np.random.default_rng(0)
    groups, by_group = _cloud_groups(rng)
    states = {("L1.block_input", "mean"): by_group}
    con, boot = contraction_tables(pd.DataFrame(), states, groups, (0,), n_subsample=25, seed=0, n_boot=100)
    r = con.iloc[0]
    P = by_group[("F01", 8, "C1")][::2]                                    # one realization per formulation
    assert r["n_match"] == "stratified_formulation" and r["n_strata"] == 4 and r["n_ref"] == 4 and r["n_mixed"] == 8
    assert r["n_subsample"] == 25
    # every one of the 25 draws holds one realization per formulation: the matched cloud is exactly P
    assert r["log_volume_mixed"] == pytest.approx(log_volume(P))
    assert r["d_log_volume"] == pytest.approx(log_volume(P) - log_volume(by_group[("F01", 0, "C1")]))
    # the draw itself: one row per reference formulation, from that formulation's realizations
    mix = [v.formulation for v in groups[("F01", 8, "C1")]]
    for _ in range(20):
        idx = stratified_draw(list("ABCD"), mix, rng)
        assert sorted(mix[i] for i in idx) == list("ABCD")
    assert stratified_draw(list("ABCE"), mix, rng) is None
    plain = stratified_draw([""] * 3, [""] * 8, rng)                        # no labels: 3 distinct rows of 8
    assert len(set(plain.tolist())) == 3


def test_contraction_boot_has_test_split():
    from lsne.experiments.stage_g_geometry import contraction_tables
    rng = np.random.default_rng(1)
    groups, by_group = _cloud_groups(rng, fams=("F01", "F02", "F15", "DEV01"))
    con, boot = contraction_tables(pd.DataFrame(), {("L1.block_input", "mean"): by_group}, groups, (0, 5),
                                   n_subsample=3, seed=0, n_boot=100)
    assert {"split", "confirmatory", "p_boot"} <= set(boot.columns) and set(boot["split"]) == {"all", "test"}
    lv = boot[(boot["quantity"] == "d_log_volume") & (boot["k"] == 0)].set_index("split")
    assert lv.loc["all", "n_clusters"] == 4 and lv.loc["test", "n_clusters"] == 2
    test = con[(con["k"] == 0) & con["family"].isin(["F01", "F02"])]
    assert lv.loc["test", "mean"] == pytest.approx(test["d_log_volume"].mean())
    assert (boot["confirmatory"] == (boot["preregistered"] & (boot["split"] == "test"))).all()
    pre = boot[boot["preregistered"]]
    assert set(pre["k"]) == {5} and set(zip(pre["quantity"], pre["readout"])) == {("d_log_volume", "mean")}


def _paths(seed: int = 0) -> pd.DataFrame:
    """Synthetic lambda_paths of the target_logit discriminant: starts of F01/F02 (test), F03C
    (neither split) and DEV01; four lambdas.  F01A never crosses (constant censored series)."""
    rng = np.random.default_rng(seed)
    rows = []
    starts = [("F01", "A"), ("F01", "B"), ("F02", "A"), ("F03", "C"), ("DEV01", "A"), ("DEV01", "B")]
    for fam, form in starts:
        for i, lam in enumerate((0.0, 0.1, 1.0, 10.0)):
            steps = 7.0 if (fam, form) == ("F01", "A") else float(6 - i + (fam == "DEV01"))
            rows.append(dict(tag=f"{fam}{form}:K0:C1", family=fam, formulation=form, discriminant="target_logit", lam=lam,
                             crossed=steps < 7, steps_to_cross_censored=steps, mahal_to_ref_mean=5.0 - i + rng.normal(0, 0.01),
                             sae_fvu=np.nan))
    return pd.DataFrame(rows)


def test_stage_h_constant_censored_series_counts_as_rho_zero():
    from lsne.experiments.stage_h_lambda import CENSORED_STEPS, CENSORED_STEPS_AS_RUN, correlation_tables
    corr, boot = correlation_tables(_paths(), 300, 0)
    c = corr[corr["quantity"] == CENSORED_STEPS].set_index("tag")
    assert c.loc["F01A:K0:C1", "rho"] == 0.0 and bool(c.loc["F01A:K0:C1", "constant_series"])
    assert c.loc["F02A:K0:C1", "rho"] == pytest.approx(-1.0) and not bool(c.loc["F02A:K0:C1", "constant_series"])
    assert c["preregistered"].all() and not c["as_run"].any()
    asrun = corr[corr["quantity"] == CENSORED_STEPS_AS_RUN].set_index("tag")
    assert np.isnan(asrun.loc["F01A:K0:C1", "rho"]) and asrun["as_run"].all() and not asrun["preregistered"].any()
    b = boot[(boot["quantity"] == CENSORED_STEPS) & (boot["split"] == "all")].iloc[0]
    assert b["n_rows"] == 6 and b["n_constant_as_zero"] == 1                     # the start is censored, not discarded
    old = boot[(boot["quantity"] == CENSORED_STEPS_AS_RUN) & (boot["split"] == "all")].iloc[0]
    assert old["n_rows"] == 5 and bool(old["as_run"])                            # the as-run statistic, flagged
    assert old["mean"] == pytest.approx(-1.0)


def test_stage_h_sign_as_expected_is_empty_for_an_undefined_mean():
    from lsne.experiments.stage_h_lambda import correlation_tables, sign_as_expected
    assert sign_as_expected(float("nan"), -1) is None and sign_as_expected(-0.5, -1) is True
    assert sign_as_expected(0.5, -1) is False and sign_as_expected(0.5, 0) is None
    corr, boot = correlation_tables(_paths(), 300, 0)
    fvu = boot[boot["quantity"] == "sae_fvu"]                                    # no SAE: every rho is NaN
    assert len(fvu) == 1 and fvu["sign_as_expected"].isna().all() and fvu["mean"].isna().all()
    assert bool(fvu["preregistered"].iloc[0]) and fvu["n_clusters"].iloc[0] == 0
    mahal = boot[(boot["quantity"] == "mahal_to_ref_mean") & (boot["split"] == "all")]
    assert bool(mahal["sign_as_expected"].iloc[0])


def test_stage_h_dev_test_split_rows():
    from lsne.experiments.stage_h_lambda import correlation_tables
    corr, boot = correlation_tables(_paths(), 300, 0)
    assert dict(corr.drop_duplicates("tag")["split"].value_counts()) == {"test": 3, "dev": 2, "other": 1}
    m = boot[boot["quantity"] == "mahal_to_ref_mean"].set_index("split")
    assert set(m.index) == {"all", "test", "dev"} and m.loc["test", "n_rows"] == 3 and m.loc["test", "n_clusters"] == 2
    assert m.loc["dev", "n_rows"] == 2 and bool(m.loc["test", "confirmatory"]) and not m.loc[["all", "dev"], "confirmatory"].any()


def test_stage_h_features_frozen_on_dev_and_tested_on_test():
    from lsne.experiments.stage_h_lambda import FROZEN_COLS, frozen_feature_table
    rows = []
    rng = np.random.default_rng(0)
    for fam, form in [("DEV01", "A"), ("DEV02", "B"), ("F01", "A"), ("F01", "B"), ("F02", "A"), ("F05", "A"), ("F03", "C")]:
        split = C.prereg_split(fam, form)
        for feat, rho in ((1, 0.9), (2, -0.8), (3, 0.2)):
            if split == "test" and feat == 2:
                rho = 0.5                                                          # sign flips on test: not confirmed
            rows.append(dict(tag=f"{fam}{form}", family=fam, formulation=form, discriminant="target_logit", feature=feat,
                             rho=rho + rng.normal(0, 0.01), candidate=abs(rho) >= 0.7, split=split))
    fr = frozen_feature_table(pd.DataFrame(rows), 300, 0)
    assert list(fr.columns) == FROZEN_COLS
    fr = fr.set_index("feature")
    assert set(fr.index) == {1, 2} and fr["candidate_dev"].all()                 # feature 3 not frozen on DEV
    assert fr.loc[1, "n_starts_dev"] == 2 and fr.loc[1, "n_clusters_test"] == 3 and fr.loc[1, "n_starts_test"] == 4
    assert bool(fr.loc[1, "confirmed"]) and fr.loc[1, "frac_candidate_test"] == 1.0
    assert fr.loc[2, "sign_dev"] == -1 and not bool(fr.loc[2, "confirmed"]) and fr.loc[2, "frac_candidate_test"] == 0.0
    assert frozen_feature_table(pd.DataFrame(rows)[lambda d: d["split"] != "dev"], 10, 0).empty


def test_t3_layer_is_the_euclidean_argmax():
    from lsne.experiments.stage_g_geometry import max_disp_snapshots
    df = pd.DataFrame(dict(snapshot=["L1.block_input", "L2.block_input", "L2.block_output"] * 2,
                           norm_K=[1.0, 3.0, 2.0] * 2, disp_mahal_K=[3.13748, 3.13740, 3.13749] * 2))
    assert max_disp_snapshots(df) == ("L2.block_input", "L2.block_output")


# --------------------------------------------------------------------------------------
# stage I
# --------------------------------------------------------------------------------------
def _dd(seed: int = 0) -> pd.DataFrame:
    """Synthetic direction_domain rows: culture/politics F13-F16 (counter-steer arm 0.2 nats above
    lang_dom with a small spread) and F01 (everyday); K=8/C1; three doses."""
    rng = np.random.default_rng(seed)
    rows = []
    for fam in ("F13", "F14", "F15", "F16", "F01"):
        for form in "AB":
            for eps in (0.01, 0.05, 0.1):
                for name, B in (("dom_culture_politics_cs", 0.2), ("lang_dom", 0.0)):
                    rows.append(dict(tag=f"{fam}{form}:K8:C1", family=fam, formulation=form, K=8, C="C1", kind="standard",
                                     family_domain=C.family_domain(fam), direction=name, eps=eps,
                                     B=B + (0.04 if fam == "F16" else 0.0) + float(rng.normal(0, 0.005))))
    return pd.DataFrame(rows)


def test_stage_i_contrast13_one_preregistered_dose_and_ci_rule():
    from lsne.experiments.stage_i_compass import PREREG_EPS_13, contrast13_eps, direction_domain_boot
    assert contrast13_eps(None, (0.01, 0.05, 0.1)) == (0.05, "default_0.05") and PREREG_EPS_13 == 0.05
    assert contrast13_eps(0.1, (0.01, 0.05, 0.1)) == (0.1, "frozen_input")
    eps, src = contrast13_eps(None, (0.1,))
    assert eps is None and "not in doses" in src
    db = direction_domain_boot(_dd(), 500, 0, prereg_eps=0.05)
    q = db[db["quantity"] == "B_cs_minus_lang_dom"]
    pre = q[q["preregistered"]]
    assert len(pre) == 1 and pre["eps"].iloc[0] == 0.05 and pre["family_domain"].iloc[0] == "culture_politics"
    assert (q["min_effect_met"] == (q["ci_low"] >= 0.05)).all() and set(q["verdict"]) <= set(C.VERDICTS)
    cp = q[q["family_domain"] == "culture_politics"]
    assert (cp["verdict"] == "supported").all() and cp["min_effect_met"].all()
    # a point estimate above 0.05 whose interval reaches below it is not "met" (it was before)
    dd2 = _dd()
    dd2.loc[(dd2["family"] == "F13") & (dd2["direction"] == "dom_culture_politics_cs"), "B"] = -0.3
    q2 = direction_domain_boot(dd2, 500, 0)
    r = q2[(q2["quantity"] == "B_cs_minus_lang_dom") & (q2["family_domain"] == "culture_politics") & (q2["eps"] == 0.05)].iloc[0]
    assert r["mean"] >= 0.05 and r["ci_low"] < 0.05 and not r["min_effect_met"] and r["verdict"] in ("open", "positive_below_min")
    assert db[db["quantity"] == "B"]["min_effect_met"].isna().all()
    assert len(direction_domain_boot(_dd(), 50, 0, prereg_eps=None).query("preregistered")) == 0


def test_stage_i_head_arm_at_equal_absolute_norm(mb):
    from lsne.experiments.stage_i_compass import head_steering_tables
    from lsne.directions import random_directions
    cfg = C.smoke_config(RunConfig(model=mb.spec, families=["DEV01"]))
    variants, _ = C.load_variants(cfg, mb)
    k0 = C.k_subset(variants, 0)[:1]
    vec = random_directions(mb.d_model, 1, 3)[0]
    hs, hb = head_steering_tables(mb, k0, dict(layer=1, head=0), {"lang_dom": vec}, (0.5,), 200, 0, enabled=False)
    assert set(hs["arm"]) == {"head", "residual", "residual_abs"}
    by = hs.set_index("arm")
    assert by.loc["residual_abs", "delta_norm"] == pytest.approx(by.loc["head", "delta_norm"], rel=1e-5)
    assert by.loc["residual", "dose_match"] == "relative" and by.loc["residual_abs", "dose_match"] == "absolute"
    assert (hs["audit_rel_err"] < 1e-4).all() and np.isfinite(hs["audit_rel_err"]).all()
    assert set(hb["quantity"]) == {"B_head_minus_residual", "B_head_minus_residual_abs"}
    absrow = hb[hb["quantity"] == "B_head_minus_residual_abs"].iloc[0]
    assert absrow["dose_match"] == "absolute" and absrow["delta_norm_ratio"] == pytest.approx(1.0, rel=1e-5)
    assert absrow["mean"] == pytest.approx(by.loc["head", "B"] - by.loc["residual_abs", "B"])
    rel = hb[hb["quantity"] == "B_head_minus_residual"].iloc[0]
    assert rel["delta_norm_ratio"] == pytest.approx(by.loc["residual", "delta_norm"] / by.loc["head", "delta_norm"])


# --------------------------------------------------------------------------------------
# stage J
# --------------------------------------------------------------------------------------
def test_stage_j_rule16_counterpole_vs_evaluated_pole():
    from lsne.experiments.stage_j_rdx import rdx_summary, rule16_status
    fired = rule16_status("books")
    assert fired["preregistered_form"] is False and fired["evaluated_pole"] == "protein_misfolding"
    assert fired["counterpole_rule16"] == "books" and "NOT the pre-registered form" in fired["note"]
    assert rule16_status("protein_misfolding")["preregistered_form"] is True
    assert rule16_status(None)["preregistered_form"] is None and rule16_status(float("nan"))["counterpole_rule16"] is None
    rdx = pd.DataFrame([dict(contrast=c, unit="pooled", basis="domain_axes", readout="last", site=f"L{l}.block_input",
                             z=3.0, explained=0.1, preregistered=(c == "culture_politics_vs_misfolding"))
                        for c in ("culture_politics_vs_misfolding", "E_vs_D") for l in (2, 5)])
    s = rdx_summary(rdx, "books").set_index("contrast")
    assert s.loc["culture_politics_vs_misfolding", "counterpole_rule16"] == "books"
    assert s.loc["culture_politics_vs_misfolding", "evaluated_pole"] == "protein_misfolding"
    assert s.loc["culture_politics_vs_misfolding", "preregistered_form"] is False
    assert bool(s.loc["culture_politics_vs_misfolding", "min_effect_met"])           # the rule is met, but not in its form
    assert s.loc["E_vs_D", "evaluated_pole"] == "" and s.loc["E_vs_D", "preregistered_form"] is None
    assert rdx_summary(rdx, "protein_misfolding").set_index("contrast").loc["culture_politics_vs_misfolding", "preregistered_form"]


# --------------------------------------------------------------------------------------
# stage L
# --------------------------------------------------------------------------------------
def test_limits_stage_l_defaults():
    assert C.Limits().l_n_prompts == 12 and C.Limits().l_eps == 0.3
    assert C.Limits.smoke().l_n_prompts == C.SMOKE_GL["l_n_prompts"]


def test_standin_prompts_round_robin_over_families():
    from types import SimpleNamespace
    from lsne.experiments.stage_l_transplant import round_robin_families, trace_prompt_variants
    vs = [SimpleNamespace(family_id=f, formulation=fo, tag=f"{f}{fo}") for f in ("F01", "F02", "F03") for fo in "ABCD"]
    assert [v.tag for v in round_robin_families(vs, 5)] == ["F01A", "F02A", "F03A", "F01B", "F02B"]
    assert len(round_robin_families(vs, 99)) == 12 and round_robin_families(vs, 0) == []
    k8 = [SimpleNamespace(family_id="F02", formulation="C")]
    tr = trace_prompt_variants(vs, k8, 4)
    assert [v.tag for v in tr] == ["F01A", "F02C", "F03A", "F01B"]                  # partnered formulation first
    # the trace default (one per family, at least 3) does not follow the stand-in's l_n_prompts = 12
    assert len(trace_prompt_variants(vs, k8, None, C.Limits())) == 3
    assert len(trace_prompt_variants(vs[:8], k8, None, C.Limits())) == 3


def test_transplant_doses_each_map_with_its_own_source_layer(mb, monkeypatch):
    import lsne.transplant as tp
    from lsne.config import HookSite
    from lsne.directions import cosine, random_directions
    from lsne.experiments.stage_l_transplant import layer_consistent_directions
    stub = tp.TraceSourceStub(d_model=48, n_layers=3, vocab=512, seed=1)
    texts = ["The children played in the garden.", "She poured a cup of tea.", "The train arrived late.",
             "We walked along the river.", "Die Kinder spielten im Garten.", "Sie goss eine Tasse Tee ein.",
             "Der Zug kam spät an.", "Wir gingen am Fluss entlang."]
    pairs = [(t, t) for t in texts]
    a0 = tp.layer_alignments(stub, mb, pairs, layers_src=[0], alpha=0.1)
    a3 = tp.layer_alignments(stub, mb, pairs, layers_src=[3], alpha=0.1)
    aligns = {l: (a0[l] if l % 2 == 0 else a3[l]) for l in a0}                      # maps from two source layers
    assert {a.layer_src for a in aligns.values()} == {0, 3}
    v = layer_consistent_directions(stub, texts[:4], texts[4:], [0, 3, 3])
    assert set(v) == {0, 3} and all(abs(float(x.norm()) - 1.0) < 1e-5 for x in v.values())
    assert cosine(v[0], v[3]) < 0.999                                                 # the layers' directions differ
    prompt = "The weather was"
    H_S = stub.residuals(stub.encode(prompt), [0, 3])
    used = []
    real = tp._dose_rows

    def spy(H, vec, eps):
        layer = next((l for l, h in H_S.items() if h.shape == H.shape and torch.allclose(h, H)), None)
        used.append((layer, vec.clone()))
        return real(H, vec, eps)
    monkeypatch.setattr(tp, "_dose_rows", spy)
    site = HookSite(mb.n_layers // 2, "block_input")
    native = random_directions(mb.d_model, 1, 5)[0]
    df = tp.transplant_experiment(mb, stub, aligns, v, [(prompt, " cold")], site, eps=0.3, n_prefix=2, modes=("residual",),
                                  controls=("identity",), gen_kw=dict(n_new=2, n_samples=1), native_axis=native, seed=0)
    assert used and all(layer in (0, 3) for layer, _ in used)
    assert all(torch.allclose(vec, v[layer]) for layer, vec in used)                  # each map dosed at its own layer
    a_site = aligns[site.layer]
    assert df["v_src_per_layer"].all() and (df["v_src_layer"] == a_site.layer_src).all()
    exp = cosine(tp.transfer_direction(a_site, v[a_site.layer_src]), native)
    assert np.allclose(df["cos_native"], exp)
    with pytest.raises(KeyError):
        tp.transplant_experiment(mb, stub, aligns, {0: v[0]}, [(prompt, " cold")], site, eps=0.3, modes=("residual",),
                                 controls=(), gen_kw=dict(n_new=2, n_samples=1))
    # a single vector keeps the old behaviour
    one = tp.transplant_experiment(mb, stub, aligns, v[3], [(prompt, " cold")], site, eps=0.3, modes=("residual",),
                                   controls=(), gen_kw=dict(n_new=2, n_samples=1))
    assert not one["v_src_per_layer"].any() and (one["v_src_layer"] == -1).all()
