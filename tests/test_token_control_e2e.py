"""End to end: the token-language control of the primary endpoint (PREREGISTRATION section 6,
amendment 11) from the control run to the evaluation (CPU, pythia-14m, test families F01 / F02
x {A, B}).

* A main stage-D run is built under a MyDrive-like tree, ``MyDrive/LSNE/runs/D/<model>@<rev>``,
  with the stage helpers exactly as stage D writes its D1 part: the dossier and depth banks
  (``get_bank`` / ``depth_bank``, the depth bank saved as ``directions_depth.npz`` +
  ``_meta.json``), the ``damage`` table of both injections and an ok manifest with the lattice
  directions, the doses (two pre-registered, one exploratory) and the batch size (two forwards
  per injection).
* The control run goes through the command line, ``python -m lsne.cli run --stage D --d-control
  token_language --d-main-run <main run> --out MyDrive/LSNE/runs_ctl``, into
  ``runs_ctl/D/<model>@<rev>``.
* ``lsne.prereg_eval.evaluate`` on the parent folder ``MyDrive/LSNE`` selects the control run
  (key ``D_ctl``) next to the main run (key ``D``); the four audits of the control run pass (the
  pre-registered site replicates the main run's lambda, bit for bit on one platform); every
  pre-registered dose gets its ``P.ctl`` rows outside the BH group, and the outcome gates the
  primary rows.

Every manifest written here is dated after the commit of amendment 6.11 (the real time on any
clock past 2026-10-28 08:05:51 UTC), so the control run passes the age rule of
``audit.D1ctl.run``.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lsne import io as lsne_io
from lsne import prereg_eval as PE
from lsne.cli import main as cli_main
from lsne.config import RunConfig
from lsne.experiments import common as C
from lsne.experiments import stage_d_control as SDC
from lsne.experiments import stage_d_noneq as SD
from lsne.io import RunDir, read_manifest

#: Two test families, formulations A and B (``cfg.families`` takes formulation tags).
FAMILIES = ["F01A", "F01B", "F02A", "F02B"]
#: D1 doses of the main run: two pre-registered ones (evaluated) and one steering dose (exploratory).
DAMAGE_DOSES = (0.05, 0.1)
STEER_DOSES = (0.5,)
#: ent, lang_dom, rand_00, rand_01 x 3 doses = 12 pairs per injection in forwards of 6 + the reference.
BATCH = 7
N_BOOT = 200
#: Control variants per formulation: K=0, K=2/4/8 x C1/C2/C3 and u9D; (variant, site) pairs: K=0 and C1 at u9
#: and u10, C2 / C3 / u9D at u9.
N_CTL_VARIANTS = 11
N_CTL_SITES = 15


def _snapshot(d: Path) -> dict:
    """Every file below ``d`` with its bytes (to show that nothing in a directory changed)."""
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}


def _after_amendment(env_info):
    """``lsne.io.environment_info`` dated after the commit of amendment 6.11: the real time on a
    clock past it (the runs are then genuinely newer), one hour after it on a clock that lags."""
    t611 = datetime.fromisoformat(PE.AMENDMENT_611["utc"])

    def info() -> dict:
        out = env_info()
        if datetime.fromisoformat(out["timestamp"]) <= t611:
            out["timestamp"] = (t611 + timedelta(hours=1)).isoformat()
        return out

    return info


def _main_run(mb, runs_root: Path) -> tuple[Path, list[str], list[float]]:
    """The D1 part of a main stage-D run as stage D writes it (``stage_d_noneq.run``): banks at the
    dossier and the depth site, the depth bank saved, ``damage`` of both injections, ok manifest."""
    t0 = time.perf_counter()
    cfg = RunConfig(model=mb.spec, families=FAMILIES, n_random_dirs=2, doses_steer=STEER_DOSES, out_root=str(runs_root))
    limits = C.Limits(progress=False, damage_doses=DAMAGE_DOSES, damage_batch_size=BATCH, n_boot=N_BOOT)
    rd = RunDir(runs_root, "D", mb.spec)
    ctx: dict = {}
    variants, _checks = C.get_variants(cfg, mb, ctx)
    site, dsite = C.dossier_site(mb), C.depth_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits)
    dbank = SD.depth_bank(cfg, mb, variants, dsite, bank, ctx=ctx, limits=limits)
    C.save_bank(dbank, rd, name="directions_depth")
    lattice = bank.names(SD.LATTICE_FAMILIES)
    doses = [float(e) for e in limits.damage_doses] + [float(e) for e in cfg.doses_steer]
    damage, profile = SD.damage_tables(mb, variants, {site.key: bank, dsite.key: dbank}, lattice, doses, enabled=False,
                                       batch_size=BATCH)
    tables = {"damage": damage, "damage_profile": profile, "damage_contrasts": SD.damage_contrasts(damage)}
    C.finish_stage(rd, "D", cfg, mb, tables, limits=limits, t0=t0, site=site, depth_site=dsite,
                   depth_position="lsne.stimuli.last_switched_position", lattice_directions=lattice, doses=doses,
                   n_variants=len(variants), audit=C.audit_summary(damage))
    return rd.path, lattice, doses


@pytest.fixture(scope="module")
def e2e(mb, tmp_path_factory):
    """``MyDrive/LSNE`` with the main run (``runs``) and the CLI control run (``runs_ctl``), the
    evaluation of the parent folder and, for comparison, of the main run alone."""
    lsne_root = tmp_path_factory.mktemp("MyDrive") / "LSNE"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LSNE_OFFLINE", "1")                    # freq axis from stimulus-bank counts (no Pile fetch)
        mp.setattr(lsne_io, "environment_info", _after_amendment(lsne_io.environment_info))
        main, lattice, doses = _main_run(mb, lsne_root / "runs")
        before = _snapshot(main)
        argv = ["run", "--stage", "D", "--model", mb.spec.name, "--revision", mb.spec.revision, "--dtype", mb.spec.dtype,
                "--device", "cpu", "--families", ",".join(FAMILIES), "--out", str(lsne_root / "runs_ctl"),
                "--d-control", "token_language", "--d-main-run", str(main), "--no-progress"]
        rc = cli_main(argv)
    ctl = RunDir(lsne_root / "runs_ctl", "D", mb.spec, mkdir=False).path
    res = PE.evaluate([lsne_root], model=mb.spec.name, revision=mb.spec.revision, n_boot=N_BOOT, seed=0)
    alone = PE.evaluate([lsne_root / "runs"], model=mb.spec.name, revision=mb.spec.revision, n_boot=N_BOOT, seed=0)
    return dict(root=lsne_root, main=main, ctl=ctl, rc=rc, before=before, lattice=lattice, doses=doses, res=res,
                df=res["results"].set_index("id"), alone=alone["results"].set_index("id"))


# --------------------------------------------------------------------------------------
# the control run
# --------------------------------------------------------------------------------------
def test_control_run_through_the_cli_takes_over_the_main_run(e2e):
    """The CLI run lands in runs_ctl/D/<model>@<rev> with the manifest contract of 6.11 and the main
    run's directions, doses and batch size; the main run (manifest, tables, bank) is untouched."""
    assert e2e["rc"] == 0
    ctl, main = e2e["ctl"], e2e["main"]
    assert ctl.parent.parent.name == "runs_ctl" and ctl.name == main.name
    m, mm = read_manifest(ctl), read_manifest(main)
    assert m["stage"] == "D" and m["status"] == "ok" and m["control"] == "token_language" and m["amendment"] == "6.11"
    assert m["main_run"] == dict(path=str(main.resolve()), git_sha=mm["git_sha"], timestamp=mm["timestamp"],
                                 model=mm["model"]["name"], revision=mm["model"]["revision"])
    assert m["bank"]["file"] == "directions_depth.npz"
    assert m["bank"]["sha256"] == SDC.file_sha256(main / "directions_depth.npz")
    assert m["lattice_directions"] == mm["lattice_directions"] == e2e["lattice"]
    assert m["doses"] == mm["doses"] == e2e["doses"]
    assert m["damage_batch_size"] == m["limits"]["damage_batch_size"] == mm["limits"]["damage_batch_size"] == BATCH
    assert not (m["limits"]["is_smoke"] or m["limits"]["limited"])
    assert m["n_variants"] == len(FAMILIES) * N_CTL_VARIANTS and m["n_excluded"] == 0 and m["audit"]["valid"]
    # --families selects the main run's formulations: the whole design relative to the main run (no design filter)
    assert m["n_formulations"] == len(FAMILIES) and m["design_filters"] == []
    assert datetime.fromisoformat(m["timestamp"]) > datetime.fromisoformat(PE.AMENDMENT_611["utc"])
    assert _snapshot(main) == e2e["before"]
    assert not (ctl / "_resume").exists() and not (ctl / "directions_depth.npz").exists()

    d = pd.read_csv(ctl / "damage_ctl.csv")
    assert set(d["family"] + d["formulation"]) == set(FAMILIES)
    assert d.groupby(["tag", "inject"]).ngroups == len(FAMILIES) * N_CTL_SITES
    assert len(d) == len(FAMILIES) * N_CTL_SITES * len(e2e["lattice"]) * len(e2e["doses"])
    assert set(d["inject"]) == {"u9", "u10"} and set(d["C"]) == {"C1", "C2", "C3", "u9D"}


# --------------------------------------------------------------------------------------
# the evaluation of the parent folder
# --------------------------------------------------------------------------------------
def test_evaluation_selects_the_control_run_next_to_the_main_run(e2e):
    runs = e2e["res"]["runs"]
    used = runs[runs["used"].astype(bool)].set_index("key")
    assert set(used.index) == {"D", "D_ctl"} and set(used["stage"]) == {"D"}
    assert Path(used.loc["D", "path"]) == e2e["main"] and Path(used.loc["D_ctl", "path"]) == e2e["ctl"]
    assert "using runs, runs_ctl below it" in " ".join(runs["note"].astype(str))
    md = PE.markdown_report(e2e["res"])
    assert "| D_ctl | ja |" in md and "| P.ctl@0.05 |" in md and "| audit.D1ctl.rep |" in md


def test_control_run_passes_its_audits(e2e):
    """6.11 (a)-(d) on the real tables: write audit, replication of the pre-registered site (u9 at
    K = 0 and C1) within 1e-3 of the main run (here within 1e-6: bit for bit on one platform), sites
    and manifest marks."""
    df = e2e["df"]
    for a in ("audit.D1", "audit.D1ctl", "audit.D1ctl.rep", "audit.D1ctl.site", "audit.D1ctl.run"):
        assert df.loc[a, "verdict"] == "supported" and df.loc[a, "role"] == "audit", (a, df.loc[a, "deviation_note"])
    rep = df.loc["audit.D1ctl.rep"]
    n_c1 = len(FAMILIES) * 4 * len(e2e["lattice"]) * len(e2e["doses"])          # K 0/2/4/8 at C1
    assert rep["n_rows"] == n_c1 and 0.0 <= rep["estimate"] <= 1e-6
    site = df.loc["audit.D1ctl.site"]
    assert site["estimate"] == 0.0 and site["n_rows"] == len(FAMILIES) * N_CTL_SITES * len(e2e["lattice"]) * len(e2e["doses"])


def test_control_rows_per_evaluated_dose_outside_bh(e2e):
    df = e2e["df"]
    ctl = df[df.index.str.startswith("P.ctl")]
    # every pre-registered dose of the main run is evaluated (no frozen dose: suffix @<eps>); 0.5 is exploratory
    for e in ("0.05", "0.1"):
        ids = {f"P.ctl@{e}", f"P.ctl1@{e}", f"P.ctl2@{e}", f"P.ctl3@{e}", f"P.ctl3.lang@{e}", f"P.ctl3.did@{e}"}
        ids |= {f"P.ctl{j}.{q}.K{K}@{e}" for j in (1, 2) for q in ("lang", "did") for K in (2, 4, 8)}
        assert ids <= set(ctl.index), e
        assert df.loc[f"P.ctl@{e}", "verdict"] in ("supported", "not_supported", "open")
        outcome = PE.CTL_VERDICT_OUTCOME[df.loc[f"P.ctl@{e}", "verdict"]]
        assert df.loc[f"P.ctl@{e}", "deviation_note"].startswith(f"outcome {outcome}")
        assert df.loc[f"P.ctl1@{e}", "n_families"] == df.loc[f"P.ctl2@{e}", "n_families"] == 2
    assert len(ctl) == 2 * 18 and not any(i.endswith("@0.5") for i in ctl.index)
    assert (ctl["role"] == "primary_control").all() and not ctl["bh_group"].astype(bool).any()
    assert ctl["bh_q"].isna().all() and not any(i.startswith("P") for i in e2e["res"]["bh"]["ids"])
    assert not ctl["verdict"].eq("not_interpretable").any()               # a valid control run


def test_primary_rows_carry_the_gate(e2e):
    """Each P row reads the outcome of P.ctl of its own dose (PREREGISTRATION 6.11): the note names
    the outcome and the rule outcome without the control (the verdict the evaluation without the
    control run gives as well), and the verdict is that outcome's gate on it."""
    df, alone = e2e["df"], e2e["alone"]

    def gate(prev: str, ctl: str) -> str:
        if ctl == "supported":
            return prev
        if ctl == "not_supported":
            return "not_interpretable" if prev in ("supported", "open") else prev
        return "open" if prev == "supported" else prev

    for e in ("0.05", "0.1"):
        p, c = df.loc[f"P@{e}"], df.loc[f"P.ctl@{e}", "verdict"]
        m = re.search(r"token-language control \(PREREGISTRATION 6\.11\): (\w+); rule outcome without the control: (\w+)",
                      p["deviation_note"])
        assert m is not None, p["deviation_note"]
        outcome, prev = m.groups()
        assert outcome == PE.CTL_VERDICT_OUTCOME[c] and p["verdict"] == gate(prev, c)
        # the same rule outcome without the control run, where the gate finds no valid control
        assert (f"token-language control (PREREGISTRATION 6.11): no valid control run; rule outcome without the control: "
                f"{prev}") in alone.loc[f"P@{e}", "deviation_note"]
        assert alone.loc[f"P@{e}", "verdict"] == ("open" if prev == "supported" else prev)
        stat = ["estimate", "ci_low", "ci_high"]
        np.testing.assert_array_equal(p[stat].to_numpy(dtype=float), alone.loc[f"P@{e}", stat].to_numpy(dtype=float))
    assert [i for i in alone.index if i.startswith("P.ctl")] == ["P.ctl"] and alone.loc["P.ctl", "verdict"] == "not_tested"
    assert np.isfinite(df.loc["P@0.05", "estimate"])
