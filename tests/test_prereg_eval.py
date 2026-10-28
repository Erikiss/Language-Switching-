"""Pre-registered evaluation of existing run directories (``lsne.prereg_eval``).

Tiny synthetic run directories with the real table schemas of the stages (column lists
taken from the stage code) check the primary statistic (C1 restriction, K=0 reference,
rand DiD, monotonicity, the D1 audit), the family-mean cluster bootstrap, the test split
(F01-F12 x {A, B}), the constant-series rule of contrast 10 (ii), the BH grouping over the
secondary group, the G0 gate, the selection of runs under several roots (two accounts, a
parent folder, superseded and failed runs) and the handling of missing stages and tables.
Everything runs on CPU without a model.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lsne import prereg_eval as PE
from lsne.io import save_df
from lsne.stats import bh_correct, bootstrap_p_two_sided, cluster_bootstrap

TAG = "pythia-160m@step98000"
ID_COLS = ["tag", "family", "formulation", "K", "C", "kind", "n_tokens", "tail_start"]
#: stage D damage columns (lsne/experiments/stage_d_noneq.py damage_tables)
DAMAGE_COLS = ID_COLS + ["inject", "site", "dir_site", "inject_layer", "inject_pos", "inject_lang", "inject_unit",
                         "inject_token", "inject_klass", "direction", "dir_family", "eps", "lam", "intercept", "r2",
                         "n_fit_layers", "D_final_tail", "D_final_last", "D_max", "cos_final_last", "front01_final",
                         "front1_final", "front01_max", "front1_max", "n_spread01_final", "n_spread1_final", "n_after",
                         "logp_target_ref", "logp_target_pert", "B", "delta_norm", "audit_rel_err"]
#: stage C contrasts (stage_c_grid.contrast_rows)
CONTRAST_COLS = ["family", "formulation", "tag", "K", "C", "kind", "dir_family", "sign", "eps", "T", "B_mixed", "B_mono", "I"]
#: stage H lambda_corr (stage_h_lambda.correlation_tables)
CORR_COLS = ["tag", "family", "formulation", "discriminant", "quantity", "rho", "pvalue", "n", "preregistered", "expected_sign"]
TEST = [f"F{i:02d}" for i in range(1, 13)]
OUTSIDE = ["DEV01", "F13"]          # families outside the confirmation set (must never leak in)


def _variant(fam, form, K, C):
    return dict(tag=f"{fam}{form}:K{K}:{C}", family=fam, formulation=form, K=int(K), C=C, kind="standard", n_tokens=90,
                tail_start=82)


def write_run(root: Path, stage: str, tables: dict, timestamp: str = "2026-10-27T00:00:00+00:00", status: str = "ok",
              **manifest) -> Path:
    d = Path(root) / stage / TAG
    d.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        save_df(df, d, name)
    m = dict(stage=stage, status=status, timestamp=timestamp, limits=dict(n_boot=2000), tables={n: len(t) for n, t in tables.items()})
    m.update(manifest)
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return d


def _noise(key: str, salt: int, scale: float = 0.01) -> float:
    """Deterministic noise in [-scale, scale] (stable across processes)."""
    return float(np.random.default_rng(zlib.crc32(f"{key}|{salt}".encode())).uniform(-scale, scale))


def damage_table(effect=(0.10, 0.12, 0.14), rand_effect=0.0, n_rand=4, audit=1e-7, eps_list=(0.05, 0.5)) -> pd.DataFrame:
    """D1 rows: lang_dom lam = base + effect[K] (+ family noise) at C1; C2 K=8 at +1.0 (must
    stay out of the C1 estimate), rand lam = base + rand_effect; the K>0 rows of families
    outside the split and of formulation C carry +5 (must never leak in); dossier rows lam NaN."""
    fx = dict(zip((2, 4, 8), effect))
    rows = []
    fams = [(f, form) for f in TEST + OUTSIDE for form in ("A", "B")] + [(f, "C") for f in TEST[:3]]
    for fam, form in fams:
        out = fam in OUTSIDE or form == "C"
        base = 0.05 + _noise(fam, 0)
        cells = [(0, "C1"), (2, "C1"), (4, "C1"), (8, "C1"), (8, "C2")]
        for K, Cc in cells:
            v = _variant(fam, form, K, Cc)
            leak = 5.0 if out and K > 0 else 0.0
            for eps in eps_list:
                for inj in ("depth", "dossier"):
                    for name in ["lang_dom", "ent"] + [f"rand_{i:02d}" for i in range(n_rand)]:
                        fam_d = "rand" if name.startswith("rand_") else name
                        if inj == "dossier":
                            lam = np.nan
                        elif fam_d == "lang_dom":
                            lam = base + (0.0 if K == 0 else (1.0 if Cc == "C2" else fx[K] + _noise(fam, K))) + leak
                        elif fam_d == "rand":
                            lam = base + (0.0 if K == 0 else rand_effect) + 0.001 * (int(name[-2:]) - n_rand / 2) + leak
                        else:
                            lam = 0.3
                        rows.append(dict(v, inject=inj, direction=name, dir_family=fam_d, eps=float(eps), lam=lam,
                                         audit_rel_err=audit))
    return pd.DataFrame(rows).reindex(columns=DAMAGE_COLS)


def contrasts_table() -> pd.DataFrame:
    """T rows: test families T = -0.3 + noise (C1) and -0.2 + noise (C3); outside rows +5."""
    rows = []
    for fam in TEST + OUTSIDE:
        for form in ("A", "B", "C"):
            leak = 5.0 if (fam in OUTSIDE or form == "C") else 0.0
            for Cc, base in (("C1", -0.3), ("C3", -0.2)):
                t = base + _noise(fam + form, 7 if Cc == "C1" else 8) + leak
                v = _variant(fam, form, 8, Cc)
                for eps in (0.0, 0.05):
                    rows.append(dict(family=fam, formulation=form, tag=v["tag"], K=8, C=Cc, kind="standard", dir_family="ent",
                                     sign=1, eps=eps, T=t, B_mixed=0.0, B_mono=0.0, I=0.0))
    return pd.DataFrame(rows, columns=CONTRAST_COLS)


@pytest.fixture()
def two_roots(tmp_path):
    """Account A: C and D (+ an older failed G); account B (a parent folder with runs/):
    G with a failed gate, H, and an older duplicate of C."""
    a = tmp_path / "LSNE" / "runs"
    b = tmp_path / "LSNE_B"
    d = write_run(a, "D", {"damage": damage_table()})
    (d / "error.txt").write_text("CUDA out of memory (earlier attempt)", encoding="utf-8")
    write_run(a, "C", {"contrasts": contrasts_table()}, timestamp="2026-10-27T05:00:00+00:00")
    write_run(a, "G", {}, status="error", timestamp="2026-10-26T00:00:00+00:00")
    write_run(a, "F", {}, status="running")                                # the sweep: never opened
    write_run(b / "runs", "C", {"contrasts": contrasts_table().assign(T=9.0)}, timestamp="2026-10-26T05:00:00+00:00")
    write_run(b / "runs", "D", {"damage": damage_table(effect=(0.0, 0.0, 0.0))}, timestamp="2026-10-28T00:00:00+00:00",
              limits=dict(n_boot=200, is_smoke=True))                  # newer, but a smoke run
    gate = pd.DataFrame([dict(n_layers=13, n_ok=0, frac_ok=0.0, pass_all=False, n_middle=5, n_ok_middle=0, frac_ok_middle=0.0,
                              pass_middle=False, degenerate=False, n_states=192, n_families=24, k=5, r=47, r_requested=128)])
    cb = pd.DataFrame([dict(quantity="d_log_volume", K=8, C="C1", site="L10.block_input", layer=10, readout="mean", k=5,
                            preregistered=True, mean=-1.8, ci_low=-2.0, ci_high=-1.6, se=0.1, n_clusters=24, n_rows=24)])
    write_run(b / "runs", "G", {"g0_gate": gate, "contraction_boot": cb}, timestamp="2026-10-27T01:00:00+00:00")
    return a, b


# --------------------------------------------------------------------------------------
# bootstrap and verdict helpers
# --------------------------------------------------------------------------------------
def test_family_boot_is_mean_of_family_means():
    """Unbalanced families: the statistic weighs every family the same (not every row) and
    the interval is the percentile interval of resampled family means."""
    df = pd.DataFrame(dict(family=["a", "a", "a", "b", "c"], v=[1.0, 1.0, 1.0, 4.0, 7.0]))
    b = PE.family_boot(df, "v", n_boot=2000, seed=3, direction=1)
    assert b["estimate"] == pytest.approx(4.0)          # (1 + 4 + 7) / 3, the row mean would be 2.8
    assert b["n_families"] == 3 and b["n_rows"] == 5
    fam = np.array([1.0, 4.0, 7.0])
    idx = np.random.default_rng(3).integers(0, 3, size=(2000, 3))
    boot = fam[idx].mean(axis=1)
    assert b["ci_low"] == pytest.approx(np.percentile(boot, 2.5))
    assert b["ci_high"] == pytest.approx(np.percentile(boot, 97.5))
    assert b["p_two_sided"] == pytest.approx(bootstrap_p_two_sided(boot, 0.0))
    assert b["p_one_sided"] == pytest.approx(1.0 / 2001.0)   # no draw <= 0
    assert PE.p_one_sided(-boot, -1) == pytest.approx(1.0 / 2001.0) and np.isnan(PE.p_one_sided(boot, 0))


def test_min_effect_and_secondary_rules():
    # section 2, by the interval alone: pass (ci_low >= min) / fail (ci_high < min) / open (min not excluded)
    assert PE.min_effect_status(0.10, 0.06, 0.14, 0.05) == "pass"
    assert PE.min_effect_status(0.10, 0.03, 0.17, 0.05) == "open"      # point estimate above, lower bound below: open
    assert PE.min_effect_status(0.04, 0.01, 0.07, 0.05) == "open"      # excludes 0, the minimum effect not excluded
    assert PE.min_effect_status(0.03, 0.01, 0.045, 0.05) == "fail"     # excludes 0 and the minimum effect
    assert PE.min_effect_status(0.02, -0.01, 0.06, 0.05) == "open"
    assert PE.min_effect_status(0.01, -0.01, 0.03, 0.05) == "fail"     # minimum effect excluded
    assert PE.min_effect_status(-0.10, -0.14, -0.06, 0.05, direction=-1) == "pass"
    # the same rule finalises BH rows with a minimum effect (9, 13): pass needs q <= 0.05 as well
    assert PE.bh_verdict(0.5, 0.35, 0.6, q=0.01, direction=1, min_effect=0.3) == "supported"
    assert PE.bh_verdict(0.5, 0.35, 0.6, q=0.2, direction=1, min_effect=0.3) == "open"
    assert PE.bh_verdict(0.5, 0.2, 0.6, q=0.01, direction=1, min_effect=0.3) == "open"
    assert PE.bh_verdict(0.2, 0.1, 0.25, q=0.01, direction=1, min_effect=0.3) == "not_supported"
    assert PE.bh_verdict(0.3, 0.1, 0.5, q=0.01) == PE.secondary_verdict(0.3, 0.1, 0.5, q=0.01)
    # p of a copied row from the table's two-sided p_boot
    assert PE.table_p(dict(p_boot=0.04, mean=0.2), 1) == (0.04, pytest.approx(0.02))
    assert PE.table_p(dict(p_boot=0.04, mean=-0.2), 1) == (0.04, pytest.approx(0.98))
    assert PE.table_p(dict(p_boot=0.04, mean=-0.2), 0)[0] == 0.04 and np.isnan(PE.table_p(dict(mean=0.2), 0)[0])
    assert PE.primary_verdict("pass", "pass", True) == "supported"
    assert PE.primary_verdict("pass", "pass", False) == "not_supported"
    assert PE.primary_verdict("pass", "pass", "pass") == "supported"
    assert PE.primary_verdict("pass", "pass", "open") == "open"           # rise in K neither shown nor refuted
    assert PE.primary_verdict("pass", "pass", "fail") == "not_supported"   # a step in K with a CI below 0
    assert PE.primary_verdict("open", "pass", "fail") == "not_supported"
    assert PE.primary_verdict("pass", "open", True) == "open"
    assert PE.primary_verdict("open", "fail", True) == "not_supported"
    assert PE.primary_verdict("pass", "nan", True) == "not_tested"
    # secondary: BH q decides, directional rows can be contradicted
    assert PE.secondary_verdict(0.3, 0.1, 0.5, q=0.01) == "supported"
    assert PE.secondary_verdict(0.3, 0.1, 0.5, q=0.2) == "open"
    assert PE.secondary_verdict(0.3, -0.1, 0.5, q=0.01) == "open"
    assert PE.secondary_verdict(-0.3, -0.5, -0.1, q=0.01, direction=1) == "not_supported"
    assert PE.secondary_verdict(-0.3, -0.5, -0.1, q=0.01, direction=-1) == "supported"
    assert PE.secondary_verdict(np.nan, np.nan, np.nan) == "not_tested"


def test_stage_constants_mirror_the_stage_modules():
    """The evaluation keeps copies of a few stage constants (importable without torch)."""
    from lsne.config import DOSES
    from lsne.experiments import common as C
    from lsne.experiments import stage_g_geometry as SG
    from lsne.experiments import stage_h_lambda as SH
    from lsne.experiments import stage_i_compass as SI
    from lsne.experiments import stage_j_rdx as SJ
    from lsne.experiments import stage_l_transplant as SL

    assert PE.PREREG_PATCH_CLASS == C.PREREG_PATCH_CLASS
    assert PE.H_PREREG_DISCRIMINANT == SH.PREREG_DISCRIMINANT and PE.H_PREREG_QUANTITIES == SH.PREREG_QUANTITIES
    assert (PE.G_PREREG_STEER_EPS, PE.G_PREREG_MIN_COS) == (SG.PREREG_STEER_EPS, SG.PREREG_MIN_COS)
    assert PE.I_PREREG_MIN_B == SI.PREREG_MIN_B
    assert PE.J_PREREG == (SJ.PREREG_CONTRAST, SJ.PREREG_BASIS, SJ.PREREG_READOUT)
    assert PE.L_TRACE_MODES == SL.TRACE_MODES
    assert PE.PREREG_DOSES == tuple(d for d in DOSES if d > 0)
    assert PE.TEST_FAMILIES == tuple(TEST) and PE.TEST_FORMULATIONS == ("A", "B")


# --------------------------------------------------------------------------------------
# primary endpoint
# --------------------------------------------------------------------------------------
def _primary_expected(dmg: pd.DataFrame, K: int, fam_filter=TEST) -> pd.Series:
    d = dmg[(dmg["inject"] == "depth") & (dmg["dir_family"] == "lang_dom") & (dmg["eps"] == 0.05)
            & dmg["family"].isin(fam_filter) & dmg["formulation"].isin(["A", "B"])]
    mixed = d[(d["K"] == K) & (d["C"] == "C1")].groupby("family")["lam"].mean()
    ref = d[d["K"] == 0].groupby("family")["lam"].mean()
    return mixed - ref


def test_primary_statistic_c1_restriction_and_test_split(tmp_path):
    dmg = damage_table()
    write_run(tmp_path, "D", {"damage": dmg})
    res = PE.evaluate([tmp_path], n_boot=500, seed=1)
    df = res["results"].set_index("id")
    # no frozen dose: every pre-registered dose present is reported with its own suffix, 0.5 is exploratory
    assert "P" not in df.index and "P@0.05" in df.index and not any(i.endswith("@0.5") for i in df.index)
    row = df.loc["P@0.05"]
    exp = _primary_expected(dmg, 8)
    assert len(exp) == 12
    assert row["estimate"] == pytest.approx(exp.mean())                 # mean of family means, C1 only, split only
    assert row["n_families"] == 12 and row["n_rows"] == 12 * 4           # 2 K8/C1 + 2 K0 variants per family
    r = cluster_bootstrap(pd.DataFrame(dict(family=exp.index, v=exp.to_numpy())), "v", "family", n_boot=500, seed=1,
                          stat_level="cluster")
    assert (row["ci_low"], row["ci_high"]) == (pytest.approx(r["ci_low"]), pytest.approx(r["ci_high"]))
    assert row["verdict"] == "supported" and row["direction"] == 1 and row["role"] == "primary"
    note = row["deviation_note"]
    assert "no DEV-frozen dose" in note and "exploratory doses present (not evaluated): 0.5" in note
    assert "fewer than the pre-registered 32" in note and "missing: 0.001" in note
    # components: lang per K, DiD against the rand contrast, monotonicity
    for K in (2, 4, 8):
        assert df.loc[f"P.lang.K{K}@0.05", "estimate"] == pytest.approx(_primary_expected(dmg, K).mean())
    did = df.loc["P.did.K8@0.05"]
    assert did["estimate"] == pytest.approx(exp.mean() - 0.0, abs=1e-12)   # rand contrast is 0 by construction
    assert did["verdict"] == "supported"
    assert df.loc["P.mono@0.05", "verdict"] == "supported" and df.loc["P.mono@0.05", "estimate"] == 1.0
    # C2 never enters the C1 estimate; it is reported as a deviation row
    dev = df.loc["P.dev.C2.lang.K8@0.05"]
    assert dev["verdict"] == "not_interpretable" and dev["role"] == "primary_deviation"
    assert dev["estimate"] == pytest.approx(1.0, abs=0.03)
    # the D1 audit passes
    audit = df.loc["audit.D1"]
    assert audit["verdict"] == "supported" and audit["estimate"] == pytest.approx(1e-7)


def test_primary_rule_outcomes_and_frozen_dose(tmp_path):
    # not monotone in K: the K=4 step lies above K=8 -> not supported although the contrasts pass
    write_run(tmp_path / "a", "D", {"damage": damage_table(effect=(0.11, 0.20, 0.14))})
    df = PE.evaluate([tmp_path / "a"], n_boot=300)["results"].set_index("id")
    assert df.loc["P@0.05", "verdict"] == "not_supported" and df.loc["P.mono@0.05", "verdict"] == "not_supported"
    # rand rises more than lang_dom: the DiD CI lies below 0 -> not supported although dlam(lang_dom) passes
    write_run(tmp_path / "b", "D", {"damage": damage_table(rand_effect=0.20)})
    df = PE.evaluate([tmp_path / "b"], n_boot=300)["results"].set_index("id")
    assert df.loc["P.lang.K8@0.05", "verdict"] == "supported"
    assert df.loc["P@0.05", "verdict"] == "not_supported" and df.loc["P.did.K8@0.05", "verdict"] == "not_supported"
    assert df.loc["P.did.K8@0.05", "ci_high"] < 0
    # a frozen dose (recorded in the stage-D manifest) is the main dose "P"
    write_run(tmp_path / "c", "D", {"damage": damage_table(eps_list=(0.01, 0.05))}, frozen_dose=0.05)
    res = PE.evaluate([tmp_path / "c"], n_boot=300)
    df = res["results"].set_index("id")
    assert "P" in df.index and "P@0.01" in df.index and "P@0.05" not in df.index
    assert "Bonferroni" not in df.loc["P", "deviation_note"]
    assert res["meta"]["frozen_dose"] == 0.05 and res["meta"]["frozen_dose_source"] == "stage-D manifest"
    assert "eingefroren 0.05 (stage-D manifest)" in PE.markdown_report(res)
    # --frozen-dose overrides
    df = PE.evaluate([tmp_path / "c"], n_boot=300, frozen_dose=0.01)["results"].set_index("id")
    assert "P" in df.index and "P@0.05" in df.index


def test_primary_audit_failure_invalidates_the_run(tmp_path):
    dmg = damage_table()
    dmg.loc[dmg.index[5], "audit_rel_err"] = 2e-4
    write_run(tmp_path, "D", {"damage": dmg})
    df = PE.evaluate([tmp_path], n_boot=200)["results"].set_index("id")
    assert df.loc["audit.D1", "verdict"] == "not_supported"
    assert "rows > 1e-4: 1 of" in df.loc["audit.D1", "deviation_note"]
    row = df.loc["P@0.05"]
    assert row["verdict"] == "not_interpretable"
    assert "run INVALID" in row["deviation_note"] and "rule outcome ignoring the audit: supported" in row["deviation_note"]
    assert (df.loc[[i for i in df.index if i.startswith("P.lang")], "verdict"] == "not_interpretable").all()


def _primary_damage(dl8) -> pd.DataFrame:
    """Damage rows of the confirmation set with per-family dlam(K) = (0.85, 0.93, 1) x ``dl8[i]``
    (the same in A and B) and rand = 0 (four directions)."""
    rows = []
    for i, fam in enumerate(TEST):
        for form in ("A", "B"):
            for K, dl in ((0, 0.0), (2, dl8[i] * 0.85), (4, dl8[i] * 0.93), (8, dl8[i])):
                for name, fd in [("lang_dom", "lang_dom")] + [(f"rand_{j:02d}", "rand") for j in range(4)]:
                    rows.append(dict(tag=f"{fam}{form}:K{K}:C1", family=fam, formulation=form, K=K, C="C1", kind="standard",
                                     inject="depth", direction=name, dir_family=fd, eps=0.05,
                                     lam=0.05 + (dl if fd == "lang_dom" else 0.0), audit_rel_err=1e-7))
    return pd.DataFrame(rows)


def test_primary_monotonicity_needs_increments_with_a_ci_above_zero(tmp_path):
    """PREREGISTRATION 6.10, reading (b): monotone point estimates are not enough; the rise
    from K = 0 to K = 2 and every later step needs a CI above 0.  Here dlam(K=2) is clearly
    positive and the point estimates rise by 0.002 per step, but the per-family steps scatter
    around 0: monotonicity is neither shown nor refuted, so the primary endpoint is open
    (section 2), and the note keeps reading (a)."""
    spread = (-0.12, 0.10, -0.06, 0.09, 0.0, 0.05, -0.08, 0.11, -0.03, 0.02, -0.09, 0.01)   # mean 0
    rows = []
    for i, fam in enumerate(TEST):
        step = 0.002 + 0.2 * spread[i]
        for form in ("A", "B"):
            for K, dl in ((0, 0.0), (2, 0.15), (4, 0.15 + step), (8, 0.15 + 2 * step)):
                for name, fd in [("lang_dom", "lang_dom")] + [(f"rand_{j:02d}", "rand") for j in range(4)]:
                    rows.append(dict(tag=f"{fam}{form}:K{K}:C1", family=fam, formulation=form, K=K, C="C1", kind="standard",
                                     inject="depth", direction=name, dir_family=fd, eps=0.05,
                                     lam=0.05 + (dl if fd == "lang_dom" else 0.0), audit_rel_err=1e-7))
    write_run(tmp_path, "D", {"damage": pd.DataFrame(rows)}, frozen_dose=0.05)
    df = PE.evaluate([tmp_path], n_boot=2000, seed=0)["results"].set_index("id")
    assert df.loc["P.lang.K8", "verdict"] == "supported" and df.loc["P.did.K8", "verdict"] == "supported"
    assert df.loc["P.mono", "verdict"] == "open" and df.loc["P.mono", "estimate"] == 0.0
    assert "point estimates monotone (reading (a), not the rule): True" in df.loc["P.mono", "deviation_note"]
    assert df.loc["P", "verdict"] == "open"
    assert "monotone open" in df.loc["P", "deviation_note"] and "point estimates monotone True" in df.loc["P", "deviation_note"]
    assert "6.10" in df.loc["P", "rule"]


def test_primary_verdict_follows_the_interval_like_damage_boot(tmp_path):
    """One minimum-effect rule: the primary verdict of prereg_eval and the confirmatory rows of
    stage D's damage_boot judge the same interval the same way (ci_low >= 0.05), never the
    point estimate.  The review case: dlam(lang_dom, K=8) = 0.0987 with CI [0.032, 0.165]."""
    from lsne.experiments import stage_d_noneq as SD

    spread = (-0.12, 0.10, -0.06, 0.09, 0.0, 0.05, -0.08, 0.11, -0.03, 0.02, -0.09, 0.01)
    cases = {"below": np.array([0.0987 + 1.6 * x for x in spread]),          # point 0.0987, ci_low ~ 0.032
             "above": np.array([0.15 + 0.2 * x for x in spread])}            # ci_low well above 0.05
    for case, dl8 in cases.items():
        dmg = _primary_damage(dl8)
        write_run(tmp_path / case, "D", {"damage": dmg}, frozen_dose=0.05)
        df = PE.evaluate([tmp_path / case], n_boot=10000, seed=0)["results"].set_index("id")
        boot = SD.damage_boot(SD.damage_contrasts(dmg), n_boot=10000, seed=0)
        conf = boot[boot["confirmatory"].astype(bool) & (boot["K"] == 8)].set_index("quantity")
        lang = conf.loc["dlam_lang_dom"]
        # the same statistic and interval on both paths
        assert df.loc["P", "estimate"] == pytest.approx(lang["mean"])
        assert (df.loc["P", "ci_low"], df.loc["P", "ci_high"]) == (pytest.approx(lang["ci_low"]), pytest.approx(lang["ci_high"]))
        assert df.loc["P.mono", "verdict"] == "supported" and df.loc["P.did.K8", "verdict"] == "supported"
        assert df.loc["P", "min_effect"] == pytest.approx(0.05)
        # ... and the same verdict: supported iff the table's min_effect_met
        assert (df.loc["P", "verdict"] == "supported") == bool(lang["min_effect_met"])
        assert (df.loc["P.lang.K8", "verdict"] == "supported") == bool(lang["min_effect_met"])
        if case == "below":
            assert lang["mean"] == pytest.approx(0.0987) and 0.0 < lang["ci_low"] < 0.05 < lang["ci_high"]
            assert lang["verdict"] == "positive_below_min" and not lang["min_effect_met"]
            assert df.loc["P", "verdict"] == "open" and df.loc["P.lang.K8", "verdict"] == "open"
        else:
            assert lang["verdict"] == "supported" and df.loc["P", "verdict"] == "supported"


# --------------------------------------------------------------------------------------
# secondary endpoints, test split, constant-series rule, BH
# --------------------------------------------------------------------------------------
def grid_summary_table(dB: dict, eps_list=(0.05, 0.1), K_list=(0, 8)) -> pd.DataFrame:
    """grid_summary rows (stage_c_grid, columns used by contrast 6): rand B = 0 (mean over 32
    directions), ent B = ``dB[sign]`` (a callable of family index and K, or a number); the rows
    outside the confirmation set carry +5 (must never leak in)."""
    rows = []
    for i, fam in enumerate(TEST + OUTSIDE):
        for form in ("A", "B", "C"):
            leak = 5.0 if (fam in OUTSIDE or form == "C") else 0.0
            for K in K_list:
                Cc = "C1"
                tag = f"{fam}{form}:K{K}:{Cc}"
                for sign in (1, -1):
                    for eps in eps_list:
                        d = dB[sign](i, K) if callable(dB[sign]) else dB[sign]
                        base = dict(tag=tag, family=fam, formulation=form, K=K, C=Cc, kind="standard", sign=sign, eps=eps)
                        rows.append(dict(base, dir_family="rand", B=0.0, n_dirs=32))
                        rows.append(dict(base, dir_family="ent", B=d + leak, n_dirs=1))
    return pd.DataFrame(rows)


def test_s6_is_one_sided_in_the_state_2143_sign_on_the_pooled_cell(tmp_path):
    """Contrast 6: ent in its State-2143 sign (+1) against random, one-sided (H1: above), pooled
    over every standard variant (stage C's preregistered K=-1 / C="all" cell); a significantly
    reversed transfer is not_supported, the sign -1 rows are a control outside the BH group and
    the K = 0 cell is a sensitivity row."""
    gs = grid_summary_table({1: lambda i, K: -0.19 + (0.02 if K == 8 else 0.0) + _noise(str(i), 6),
                             -1: lambda i, K: 0.14 + _noise(str(i), 5)})
    write_run(tmp_path, "C", {"grid_summary": gs})
    res = PE.evaluate([tmp_path], n_boot=500, seed=0)
    df = res["results"].set_index("id")
    for eps in ("0.05", "0.1"):
        r = df.loc[f"S6.sign+1.eps{eps}"]
        assert r["direction"] == 1 and r["role"] == "secondary" and r["bh_group"]
        assert r["ci_high"] < 0 and r["verdict"] == "not_supported"           # reversed benefit: not "supported"
        assert r["p_one_sided"] > 0.99 and r["bh_q"] > 0.05
        t = gs[(gs["sign"] == 1) & (gs["eps"] == float(eps)) & gs["family"].isin(TEST) & gs["formulation"].isin(["A", "B"])]
        dB = t.pivot_table(index=["family", "tag"], columns="dir_family", values="B")
        assert r["estimate"] == pytest.approx((dB["ent"] - dB["rand"]).groupby("family").mean().mean())   # K 0 and 8 pooled
        k0 = df.loc[f"S6.K0.sign+1.eps{eps}"]
        assert k0["role"] == "sensitivity" and not k0["bh_group"] and k0["estimate"] < r["estimate"]
        ctrl = df.loc[f"S6.sign-1.eps{eps}"]
        assert ctrl["role"] == "sensitivity" and not ctrl["bh_group"] and np.isnan(ctrl["bh_q"]) and ctrl["direction"] == 0
    assert not any(i.startswith("S6.sign-1") for i in res["bh"]["ids"])
    assert {"S6.sign+1.eps0.05", "S6.sign+1.eps0.1"} <= set(res["bh"]["ids"])
    # a benefit in the State-2143 sign is supported
    write_run(tmp_path / "pos", "C", {"grid_summary": grid_summary_table({1: lambda i, K: 0.2 + _noise(str(i), 6), -1: -0.1})})
    df = PE.evaluate([tmp_path / "pos"], n_boot=500, seed=0)["results"].set_index("id")
    assert df.loc["S6.sign+1.eps0.05", "verdict"] == "supported"


def head_steering_boot_table(p_boot=0.049, doses=(0.05, 0.1, 0.2)) -> pd.DataFrame:
    """stage_i_compass head_steering_boot rows of ``B_head_minus_residual`` (CI [0.001, 0.2])."""
    rows = [dict(quantity="B_head_minus_residual", direction="lang_dom", eps=e, dose_match="relative", mean=0.1, ci_low=0.001,
                 ci_high=0.2, se=0.05, n_clusters=12, n_rows=24, delta_norm_ratio=3.0) for e in doses]
    df = pd.DataFrame(rows)
    if p_boot is not None:
        df["p_boot"] = p_boot
    return df


def test_copied_secondaries_enter_the_bh_group(tmp_path):
    """8, 9 (with a passed gate), 13 and 14 are BH-corrected with the secondary group from the
    table's p_boot: three S14 rows with p 0.049 are 'supported' by their uncorrected CIs but end
    'open' next to the S1-S6 rows; tables without p_boot (runs on Drive) are capped at 'open'."""
    write_run(tmp_path / "new", "C", {"contrasts": contrasts_table(),
                                      "grid_summary": grid_summary_table({1: lambda i, K: 0.05 * (1 if i % 2 else -1), -1: 0.0})})
    dd = pd.DataFrame([dict(quantity="B_cs_minus_lang_dom", family_domain="culture_politics", direction="dom_culture_politics_cs-lang_dom",
                            K=8, C="C1", eps=e, preregistered=e == 0.05, min_effect_met=False, mean=0.06, ci_low=0.02, ci_high=0.10,
                            se=0.02, n_clusters=4, n_rows=8, verdict="positive_below_min", p_boot=0.004) for e in (0.05, 0.1)])
    write_run(tmp_path / "new", "I", {"head_steering_boot": head_steering_boot_table(), "direction_domain_boot": dd})
    res = PE.evaluate([tmp_path / "new"], n_boot=500, seed=0)
    df = res["results"].set_index("id")
    s14 = df.loc[[f"S14.lang_dom.eps{e}" for e in ("0.05", "0.1")]]
    assert s14["bh_group"].all() and (s14["p_two_sided"] == 0.049).all()
    assert (s14["bh_q"] > 0.05).all() and (s14["verdict"] == "open").all()
    # eps 0.2 is an exploratory steering dose (PREREGISTRATION 1): reported, outside the BH group
    x14 = df.loc["S14.lang_dom.eps0.2"]
    assert x14["role"] == "exploratory" and not x14["bh_group"] and "exploratory dose" in x14["deviation_note"]
    assert "S14.lang_dom.eps0.2" not in set(res["bh"]["ids"])
    assert {"S1", "S2", "S6.sign+1.eps0.05", "S14.lang_dom.eps0.05", "S13.eps0.05"} <= set(res["bh"]["ids"])
    # 13: the pre-registered dose is a BH row judged by the interval (CI [0.02, 0.10]: minimum effect not
    # established -> open, like the table's positive_below_min); the other dose is a sensitivity row
    s13 = df.loc["S13.eps0.05"]
    assert s13["bh_group"] and s13["role"] == "copied" and s13["verdict"] == "open" and s13["p_one_sided"] == pytest.approx(0.002)
    assert "table verdict positive_below_min" in s13["deviation_note"]
    assert "min_effect_met (table, point estimate)" not in s13["deviation_note"]
    assert df.loc["S13.eps0.1", "role"] == "sensitivity" and not df.loc["S13.eps0.1", "bh_group"]
    # alone, the S14 rows would pass BH (q = 0.049): the group decides
    alone, info = PE.apply_bh(df.reset_index().loc[lambda x: x["id"].str.startswith("S14")])
    assert info["m"] == 2 and (alone.loc[alone["bh_group"].astype(bool), "verdict"] == "supported").all()
    # tables without p_boot: verdict capped at open, outside the BH group
    write_run(tmp_path / "old", "I", {"head_steering_boot": head_steering_boot_table(p_boot=None),
                                      "direction_domain_boot": dd.drop(columns=["verdict", "p_boot"]).assign(preregistered=True,
                                                                                                             ci_low=0.06)})
    df = PE.evaluate([tmp_path / "old"], n_boot=100)["results"].set_index("id")
    for rid in ("S14.lang_dom.eps0.05", "S13.eps0.05", "S13.eps0.1"):
        assert df.loc[rid, "verdict"] == "open" and not df.loc[rid, "bh_group"] and PE.NO_P_NOTE in df.loc[rid, "deviation_note"]
    assert "min_effect_met (table, point estimate)" in df.loc["S13.eps0.05", "deviation_note"]


def test_s14_absolute_norm_rows_are_a_sensitivity(tmp_path):
    """14: the B_head_minus_residual_abs rows of stage I (residual arm at the head arm's absolute
    written norm) are reported as S14abs sensitivity rows, never in the BH group."""
    hb = pd.concat([head_steering_boot_table(doses=(0.1,)),
                    pd.DataFrame([dict(quantity="B_head_minus_residual_abs", direction="lang_dom", eps=0.1,
                                       dose_match="absolute", mean=0.0008, ci_low=-0.0013, ci_high=0.003, se=0.001,
                                       n_clusters=12, n_rows=96, delta_norm_ratio=1.0, p_boot=0.49)])], ignore_index=True)
    write_run(tmp_path, "I", {"head_steering_boot": hb})
    res = PE.evaluate([tmp_path], n_boot=100)
    df = res["results"].set_index("id")
    row = df.loc["S14abs.lang_dom.eps0.1"]
    assert row["role"] == "sensitivity" and not row["bh_group"] and row["verdict"] == "open"
    assert "same absolute norm" in row["name"] and "S14abs.lang_dom.eps0.1" not in set(res["bh"]["ids"])
    assert df.loc["S14.lang_dom.eps0.1", "bh_group"]


def test_s10f_reads_the_dev_frozen_feature_table(tmp_path):
    """10f: candidates frozen on DEV01-04 and tested on F01-F12 x A,B (stage H
    lambda_features_frozen): supported iff a candidate is confirmed after BH over the
    candidates' p_boot_test; the 'no DEV-freeze path' caveat only for runs without the table."""
    from lsne.experiments import stage_h_lambda as SH

    def frozen(rows):
        base = dict(discriminant="target_logit", n_starts_dev=8, n_families_dev=4, rho_mean_dev=0.8, sign_dev=1, candidate_dev=True,
                    n_starts_test=24, n_clusters_test=12, frac_candidate_test=0.5)
        return pd.DataFrame([dict(base, **r) for r in rows], columns=SH.FROZEN_COLS)
    cands = frozen([dict(feature=1, rho_mean_test=0.6, ci_low_test=0.4, ci_high_test=0.8, p_boot_test=0.001, confirmed=True),
                    dict(feature=2, rho_mean_test=0.2, ci_low_test=0.01, ci_high_test=0.4, p_boot_test=0.04, confirmed=True),
                    dict(feature=3, rho_mean_test=0.0, ci_low_test=-0.2, ci_high_test=0.2, p_boot_test=0.6, confirmed=False),
                    dict(feature=4, discriminant="probe", rho_mean_test=0.9, ci_low_test=0.8, ci_high_test=1.0, p_boot_test=1e-4,
                         confirmed=True)])
    write_run(tmp_path / "a", "H", {"lambda_features_frozen": cands}, sae_source="hub:x/y")
    r = PE.evaluate([tmp_path / "a"], n_boot=50)["results"].set_index("id").loc["S10f"]
    q = bh_correct([0.001, 0.04, 0.6])                                      # the probe discriminant is not contrast 10
    assert q[1] > 0.05
    assert r["verdict"] == "supported" and r["estimate"] == 1 and r["n_rows"] == 3 and r["n_families"] == 12
    assert "3 DEV-frozen candidates" in r["deviation_note"] and "2 confirmed by the test CI, 1 after BH" in r["deviation_note"]
    assert PE.CAVEATS["S10f"] not in r["deviation_note"]
    write_run(tmp_path / "b", "H", {"lambda_features_frozen": cands.assign(confirmed=False)}, sae_source="hub:x/y")
    assert PE.evaluate([tmp_path / "b"], n_boot=50)["results"].set_index("id").loc["S10f", "verdict"] == "not_supported"
    write_run(tmp_path / "c", "H", {"lambda_features_frozen": cands.iloc[:0]}, sae_source="hub:x/y")
    r = PE.evaluate([tmp_path / "c"], n_boot=50)["results"].set_index("id").loc["S10f"]
    assert r["verdict"] == "not_tested" and "no candidate on DEV" in r["deviation_note"]
    # a run written before the table: the old count, not interpretable, with the caveat
    fs = pd.DataFrame([dict(discriminant="target_logit", feature=1, n_starts=24, rho_mean=0.8, candidate_all=True)])
    write_run(tmp_path / "d", "H", {"lambda_features_summary": fs}, sae_source="hub:x/y")
    r = PE.evaluate([tmp_path / "d"], n_boot=50)["results"].set_index("id").loc["S10f"]
    assert r["verdict"] == "not_interpretable" and PE.CAVEATS["S10f"] in r["deviation_note"]
def test_s1_s2_use_the_test_split_only(tmp_path):
    con = contrasts_table()
    write_run(tmp_path, "C", {"contrasts": con})
    df = PE.evaluate([tmp_path], n_boot=400, seed=2)["results"].set_index("id")
    t = con[(con["eps"] == 0) & con["family"].isin(TEST) & con["formulation"].isin(["A", "B"])]
    exp1 = t[t["C"] == "C1"].groupby("family")["T"].mean().mean()
    assert df.loc["S1", "estimate"] == pytest.approx(exp1)
    assert df.loc["S1", "n_families"] == 12 and df.loc["S1", "n_rows"] == 24
    p = t.pivot_table(index=["family", "formulation"], columns="C", values="T")
    exp2 = (p["C3"] - p["C1"]).groupby("family").mean().mean()
    assert df.loc["S2", "estimate"] == pytest.approx(exp2)
    assert df.loc["S1", "bh_group"] and df.loc["S2", "bh_group"]


#: stage C attribution_contrasts (stage_c_grid.ATT_CONTRAST_COLS) and a subset of attribution
ATT_CONTRAST_COLS = ["pair", "family", "formulation", "tag_mixed", "K", "C", "kind", "site", "layer", "class", "preregistered",
                     "n_positions", "recovery", "recovery_random_mean", "recovery_random_sd", "R_excess"]
ATT_COLS = ["pair", "family", "formulation", "tag_mixed", "K_mixed", "C_mixed", "kind", "site", "layer", "class", "region",
            "n_positions", "measurable", "control_defined", "logp_mono", "logp_mixed", "logp_patched", "recovery",
            "recovery_random_mean", "recovery_random_sd"]


def test_s3_delim_rows_per_pair_and_layer_and_nats_excess(tmp_path):
    from lsne.experiments import stage_c_grid as SC

    assert ATT_CONTRAST_COLS == SC.ATT_CONTRAST_COLS
    con, att = [], []
    for fam in TEST + OUTSIDE:
        for form in ("A", "B"):
            for pair, K, Cc in (("K0_vs_K8C1", 8, "C1"), ("K0_vs_K28seg", 28, "seg")):
                tag = f"{fam}{form}:K{K}:{Cc}"
                denom = (-0.5 if form == "A" else 0.2) * (1 + int(fam[-2:]) / 10)      # T > 0 for the A formulations
                for layer in (2, 5):
                    for cls, rex in (("delim", 0.1 + 0.01 * layer + _noise(tag, layer)), ("punct", 9.0)):
                        rex = rex + (5.0 if fam in OUTSIDE else 0.0)
                        con.append(dict(pair=pair, family=fam, formulation=form, tag_mixed=tag, K=K, C=Cc, kind="standard",
                                        site=f"L{layer}.block_input", layer=layer, **{"class": cls}, preregistered=cls == "delim",
                                        n_positions=3, recovery=rex, recovery_random_mean=0.0, recovery_random_sd=0.1,
                                        R_excess=rex))
                        att.append(dict(pair=pair, family=fam, formulation=form, tag_mixed=tag, K_mixed=K, C_mixed=Cc,
                                        kind="standard", site=f"L{layer}.block_input", layer=layer, **{"class": cls},
                                        region="suffix", n_positions=3, measurable=True, control_defined=True,
                                        logp_mono=-1.0, logp_mixed=-1.0 - denom, logp_patched=np.nan, recovery=rex,
                                        recovery_random_mean=0.0, recovery_random_sd=0.1))
    con, att = pd.DataFrame(con, columns=ATT_CONTRAST_COLS), pd.DataFrame(att, columns=ATT_COLS)
    write_run(tmp_path, "C", {"attribution_contrasts": con, "attribution": att})
    df = PE.evaluate([tmp_path], n_boot=300)["results"].set_index("id")
    t = con[con["family"].isin(TEST) & (con["class"] == "delim")]
    for pair in ("K0_vs_K8C1", "K0_vs_K28seg"):
        for layer in (2, 5):
            g = t[(t["pair"] == pair) & (t["layer"] == layer)]
            row = df.loc[f"S3.{pair}.L{layer}"]
            assert row["estimate"] == pytest.approx(g.groupby("family")["R_excess"].mean().mean())
            assert row["direction"] == 1 and row["bh_group"] and row["n_families"] == 12
            m = g.merge(att[["pair", "tag_mixed", "layer", "class", "logp_mono", "logp_mixed"]],
                        on=["pair", "tag_mixed", "layer", "class"])
            nats = m["R_excess"] * (m["logp_mono"] - m["logp_mixed"]).abs()
            x = df.loc[f"S3x.nats.{pair}.L{layer}"]
            assert x["estimate"] == pytest.approx(nats.groupby(m["family"]).mean().mean())
            assert x["role"] == "exploratory" and not x["bh_group"] and np.isnan(x["bh_q"])
    note = df.loc["S3.K0_vs_K8C1.L2", "deviation_note"]
    assert "0 of 48 test pairs have |logp_mono - logp_mixed| < 0.05 nats, 24 have T > 0" in note


def test_split_rows_and_formulation_filter():
    df = pd.DataFrame(dict(family=["F01", "F01", "F12", "F13", "DEV01"], formulation=["A", "C", "B", "A", "A"]))
    assert PE.split_rows(df).index.tolist() == [0, 2]
    assert PE.split_rows(df, ("DEV01",), None).index.tolist() == [4]


def _lambda_corr() -> pd.DataFrame:
    """Per-start rho of steps_to_cross_censored: half of the starts have a constant
    censored series (rho NaN with n = 7), one start has no data (n = 0); mahal finite."""
    rows = []
    for i, fam in enumerate(TEST + OUTSIDE):
        for form in ("A", "B"):
            tag = f"{fam}{form}:K0:C1"
            const = form == "B"
            rho_s = np.nan if const else 0.6
            n_s = 0 if (fam == "F01" and form == "B") else 7
            rows.append(dict(tag=tag, family=fam, formulation=form, discriminant="target_logit", quantity="steps_to_cross_censored",
                             rho=rho_s, pvalue=np.nan, n=n_s, preregistered=True, expected_sign=0))
            rows.append(dict(tag=tag, family=fam, formulation=form, discriminant="target_logit", quantity="mahal_to_ref_mean",
                             rho=-0.5 - 0.01 * i, pvalue=0.1, n=7, preregistered=True, expected_sign=-1))
            rows.append(dict(tag=tag, family=fam, formulation=form, discriminant="target_logit", quantity="sae_fvu",
                             rho=np.nan, pvalue=np.nan, n=0, preregistered=True, expected_sign=-1))
            rows.append(dict(tag=tag, family=fam, formulation=form, discriminant="probe", quantity="mahal_to_ref_mean",
                             rho=0.9, pvalue=0.1, n=7, preregistered=False, expected_sign=0))
    return pd.DataFrame(rows, columns=CORR_COLS)


def test_constant_censored_series_counts_as_rho_zero(tmp_path):
    corr = _lambda_corr()
    write_run(tmp_path, "H", {"lambda_corr": corr}, sae_source="none")
    df = PE.evaluate([tmp_path], n_boot=300)["results"].set_index("id")
    # main rule: constant series (n >= 2, rho NaN) -> 0; F01B (n = 0) stays missing
    main = df.loc["S10ii"]
    fam_means = [0.6 if f == "F01" else (0.6 + 0.0) / 2 for f in TEST]     # F01: only the A start is defined
    assert main["estimate"] == pytest.approx(np.mean(fam_means))
    assert main["n_rows"] == 23 and main["n_families"] == 12 and main["bh_group"]
    assert "11 of 24 starts have a constant series" in main["deviation_note"]
    # as run: constant series dropped -> only the A starts (rho 0.6) remain
    asrun = df.loc["S10ii.as_run"]
    assert asrun["estimate"] == pytest.approx(0.6) and asrun["n_rows"] == 12 and not asrun["bh_group"]
    assert asrun["role"] == "sensitivity"
    # (iii): directional, test families only (probe discriminant and outside families ignored)
    iii = df.loc["S10iii"]
    exp = np.mean([-0.5 - 0.01 * i for i in range(12)])
    assert iii["estimate"] == pytest.approx(exp) and iii["direction"] == -1 and iii["verdict"] == "supported"
    assert np.isfinite(iii["p_one_sided"]) and iii["p_one_sided"] < 0.01
    # (i): no SAE -> not tested, not in the BH group
    assert df.loc["S10i", "verdict"] == "not_tested" and not df.loc["S10i", "bh_group"]


def test_constant_series_flag_of_the_current_stage_h(tmp_path):
    """Stage H since the rho = 0 rule writes the constant censored series with rho = 0 and
    ``constant_series=True`` (plus ``*_as_run`` rows): the main row is unchanged and the
    as-run sensitivity row still drops those starts."""
    old = _lambda_corr()
    new = old.assign(constant_series=False, as_run=False, split="test")
    const = (new["quantity"] == "steps_to_cross_censored") & new["rho"].isna() & (new["n"] >= 2)
    new.loc[const, ["rho", "pvalue", "constant_series"]] = [0.0, 1.0, True]
    as_run = old[old["quantity"] == "steps_to_cross_censored"].assign(quantity="steps_to_cross_censored_as_run",
                                                                       preregistered=False, as_run=True)
    new = pd.concat([new, as_run], ignore_index=True)
    write_run(tmp_path / "old", "H", {"lambda_corr": old}, sae_source="none")
    write_run(tmp_path / "new", "H", {"lambda_corr": new}, sae_source="none")
    a = PE.evaluate([tmp_path / "old"], n_boot=300)["results"].set_index("id")
    b = PE.evaluate([tmp_path / "new"], n_boot=300)["results"].set_index("id")
    for rid in ("S10ii", "S10ii.as_run"):
        assert b.loc[rid, "estimate"] == pytest.approx(a.loc[rid, "estimate"])
        assert b.loc[rid, "n_rows"] == a.loc[rid, "n_rows"]
    assert b.loc["S10ii.as_run", "estimate"] == pytest.approx(0.6) and b.loc["S10ii.as_run", "n_rows"] == 12
    assert "11 of 24 starts have a constant series" in b.loc["S10ii", "deviation_note"]


def test_lambda_corr_fallback_from_paths(tmp_path):
    """Without lambda_corr the per-start Spearman rho is recomputed from lambda_paths
    (a constant censored series again counts as rho = 0)."""
    rows = []
    lam = [0.0, 0.1, 0.3, 1.0, 3.0]
    for fam in TEST:
        for form in ("A", "B"):
            for j, lv in enumerate(lam):
                rows.append(dict(context="standard", tag=f"{fam}{form}:K0:C1", family=fam, formulation=form,
                                 discriminant="target_logit", lam=lv, lam_rel=lv,
                                 steps_to_cross_censored=(101 if form == "B" else [1, 2, 3, 101, 101][j]),
                                 mahal_to_ref_mean=5.0 - j, sae_fvu=np.nan, logp_target_unperturbed=(-0.5 if form == "B" else -3.0)))
    write_run(tmp_path, "H", {"lambda_paths": pd.DataFrame(rows)}, path=dict(cross_nats=1.0))
    df = PE.evaluate([tmp_path], n_boot=200)["results"].set_index("id")
    rho_a = pd.Series(lam).rank().corr(pd.Series([1, 2, 3, 101, 101]).rank())
    assert df.loc["S10ii", "estimate"] == pytest.approx(rho_a / 2)
    assert df.loc["S10iii", "estimate"] == pytest.approx(-1.0)
    assert "12 of 24 test starts have log p0(target) > -1" in df.loc["S10ii", "deviation_note"]


def test_bh_over_the_secondary_group():
    rows = [PE.make_row("S1", "a", role="secondary", stage="C", estimate=0.3, ci_low=0.1, ci_high=0.5, p_two_sided=0.01,
                        bh_group=True),
            PE.make_row("S2", "b", role="secondary", stage="C", estimate=0.1, ci_low=0.01, ci_high=0.2, p_two_sided=0.04,
                        bh_group=True),
            PE.make_row("S3.x.L2", "c", role="secondary", stage="C", estimate=0.1, ci_low=0.02, ci_high=0.2, p_two_sided=0.03,
                        p_one_sided=0.015, direction=1, bh_group=True),
            PE.make_row("S3.x.L5", "d", role="secondary", stage="C", estimate=-0.1, ci_low=-0.2, ci_high=0.1, p_two_sided=0.5,
                        p_one_sided=0.75, direction=1, bh_group=True),
            PE.make_row("S4.K0", "e", role="sensitivity", stage="D", estimate=0.3, ci_low=0.1, ci_high=0.5, p_two_sided=0.001),
            PE.make_row("S6", "f", role="secondary", stage="C", bh_group=True)]          # not tested: no p
    df, info = PE.apply_bh(pd.DataFrame(rows, columns=PE.RESULT_COLS + PE.EXTRA_COLS))
    df = df.set_index("id")
    p_used = [0.01, 0.04, 0.015, 0.75]                       # one-sided where a direction is set
    q = bh_correct(p_used)
    assert info["m"] == 4 and info["ids"] == ["S1", "S2", "S3.x.L2", "S3.x.L5"]
    assert df.loc[["S1", "S2", "S3.x.L2", "S3.x.L5"], "bh_q"].to_numpy() == pytest.approx(q)
    assert np.isnan(df.loc["S4.K0", "bh_q"]) and np.isnan(df.loc["S6", "bh_q"])
    assert df.loc["S1", "verdict"] == "supported" and df.loc["S3.x.L2", "verdict"] == "supported"
    assert q[1] > 0.05 and df.loc["S2", "verdict"] == "open"   # CI excludes 0 but BH q > 0.05
    assert df.loc["S6", "verdict"] == "not_tested"
    # endpoint grouping: Bonferroni within S3 (2 x 0.015), BH over S1, S2, S3
    qe = bh_correct([0.01, 0.04, min(1.0, 2 * 0.015)])
    assert df.loc["S3.x.L2", "bh_q_endpoint"] == pytest.approx(qe[2]) and info["m_endpoints"] == 3


# --------------------------------------------------------------------------------------
# runs, roots, gate, missing stages, outputs
# --------------------------------------------------------------------------------------
def test_two_account_roots_selection_and_gate(two_roots, tmp_path):
    a, b = two_roots
    res = PE.evaluate([a, b, tmp_path / "missing"], n_boot=200)
    runs = res["runs"]
    used = runs[runs["used"]].set_index("key")
    assert set(used.index) == {"C", "D", "G"}
    assert str(used.loc["C", "path"]).startswith(str(a))                 # newer C wins over account B's older copy
    assert str(used.loc["G", "path"]).startswith(str(b / "runs"))        # the failed G of account A is not used
    assert str(used.loc["D", "path"]).startswith(str(a))                 # a full run wins over a newer smoke run
    assert "F" not in set(runs["stage"])
    notes = " ".join(runs["note"].astype(str))
    assert "superseded by the newer run" in notes and "status 'error': not used" in notes
    assert "superseded by a full run" in notes
    assert "stale error.txt" in str(used.loc["D", "note"])                # an ok run with a leftover error.txt is used
    assert "does not exist" in notes and "using runs below it" in notes
    df = res["results"].set_index("id")
    assert df.loc["S1", "estimate"] < 0                                   # account B's copy (T = 9) is not used
    assert df.loc["S7", "verdict"] == "gate_failed" and df.loc["S7", "role"] == "gate"
    s8 = df.loc["S8.d_log_volume.L10.block_input"]
    # a table without p_boot cannot enter the BH group: the rule outcome is capped at open
    assert s8["verdict"] == "not_interpretable" and "rule outcome ignoring the gate: open" in s8["deviation_note"]
    assert PE.NO_P_NOTE in s8["deviation_note"] and not s8["bh_group"]
    assert "n_boot = 2000 as run" in s8["deviation_note"]


def test_gate_pass_makes_8_and_9_interpretable(tmp_path):
    gate = pd.DataFrame([dict(n_layers=13, n_ok=9, frac_ok=0.69, pass_all=True, n_middle=5, n_ok_middle=4, frac_ok_middle=0.8,
                              pass_middle=True, degenerate=False, n_states=192, n_families=24, k=5, r=47, r_requested=128)])
    sb = pd.DataFrame([dict(quantity="cos_at_max_disp_layer", inject_site="L11.block_input", snapshot="per_family", layer=-1,
                            eps=0.05, preregistered=True, min_effect_met=False, mean=0.18, ci_low=0.15, ci_high=0.21, se=0.01,
                            n_clusters=24, n_rows=24)])
    write_run(tmp_path, "G", {"g0_gate": gate, "steering_boot": sb})
    df = PE.evaluate([tmp_path], n_boot=100)["results"].set_index("id")
    assert df.loc["S7", "verdict"] == "supported"
    assert df.loc["S9.L11.block_input", "verdict"] == "not_supported"     # CI below the 0.3 minimum
    assert df.loc["S8", "verdict"] == "not_tested"                        # table missing -> not tested with a reason
    assert "contraction_boot missing" in df.loc["S8", "deviation_note"]
    # a degenerate gate gives no verdict
    write_run(tmp_path, "G", {"g0_gate": gate.assign(degenerate=True, pass_middle=None, pass_all=None), "steering_boot": sb})
    df = PE.evaluate([tmp_path], n_boot=100)["results"].set_index("id")
    assert df.loc["S7", "verdict"] == "not_interpretable"
    assert df.loc["S9.L11.block_input", "verdict"] == "not_interpretable"


def test_split_boot_tables_use_the_test_rows(tmp_path):
    """Stage-G tables written since the ``split`` column exists hold the pooled ``all`` rows
    next to the ``test`` rows: 8 and 9 take the test rows, one row per endpoint cell (no
    duplicate ids); without test rows the pooled rows are used and the note says so."""
    gate = pd.DataFrame([dict(n_layers=13, n_ok=9, frac_ok=0.69, pass_all=True, n_middle=5, n_ok_middle=4, frac_ok_middle=0.8,
                              pass_middle=True, degenerate=False, n_states=192, n_families=24, k=5, r=47, r_requested=128)])
    cb = pd.DataFrame([dict(quantity="d_log_volume", K=8, C="C1", site="L10.block_input", layer=10, readout="mean", k=5,
                            preregistered=True, mean=m, ci_low=m - 0.2, ci_high=m + 0.2, se=0.1, n_clusters=n, n_rows=n,
                            split=split, confirmatory=split == "test", p_boot=0.001)
                       for split, m, n in (("all", -1.8, 24), ("test", 0.5, 12))])
    sb = pd.DataFrame([dict(quantity="cos_at_max_disp_layer", inject_site="L11.block_input", snapshot="per_family", layer=-1,
                            eps=0.05, preregistered=True, min_effect_met=m >= 0.3, mean=m, ci_low=m - 0.03, ci_high=m + 0.03,
                            se=0.01, n_clusters=n, n_rows=n, split=split, confirmatory=split == "test", p_boot=0.001)
                       for split, m, n in (("all", 0.18, 24), ("test", 0.45, 12))])
    write_run(tmp_path, "G", {"g0_gate": gate, "contraction_boot": cb, "steering_boot": sb})
    df = PE.evaluate([tmp_path], n_boot=100)["results"]
    assert df["id"].is_unique
    df = df.set_index("id")
    s8, s9 = df.loc["S8.d_log_volume.L10.block_input"], df.loc["S9.L11.block_input"]
    assert s8["estimate"] == pytest.approx(0.5) and s8["n_families"] == 12 and "split test" in s8["deviation_note"]
    assert s9["estimate"] == pytest.approx(0.45) and s9["verdict"] == "supported" and "split test" in s9["deviation_note"]
    # stage G writes its test rows per family: the clouds pool the formulations A-D, and the note says so
    for r in (s8, s9):
        assert PE.G_SPLIT_TEST_NOTE in r["deviation_note"] and "F01-F12 x A,B, family means" not in r["deviation_note"]
        assert "clouds pool the formulations A-D" in r["deviation_note"]           # the caveat as well
    # with the gate passed and p_boot in the table, 8 and 9 are in the BH group (9 one-sided, minimum effect 0.3)
    assert s8["bh_group"] and s9["bh_group"] and np.isfinite(s8["bh_q"]) and np.isfinite(s9["bh_q"])
    assert s8["p_two_sided"] == pytest.approx(0.001) and s9["p_one_sided"] == pytest.approx(0.0005)
    assert s9["min_effect"] == pytest.approx(0.3)
    # a run without the confirmation set: the pooled rows, flagged
    write_run(tmp_path, "G", {"g0_gate": gate, "contraction_boot": cb[cb["split"] == "all"], "steering_boot": sb[sb["split"] == "all"]})
    df = PE.evaluate([tmp_path], n_boot=100)["results"].set_index("id")
    assert df.loc["S9.L11.block_input", "estimate"] == pytest.approx(0.18)
    assert "split all of the table (no test rows)" in df.loc["S9.L11.block_input", "deviation_note"]
    rows, note = PE.boot_split(cb.drop(columns=["split"]))
    assert len(rows) == 2 and note == ""                                  # runs written before the column: unchanged


def test_missing_stages_are_not_tested_and_outputs_are_written(tmp_path):
    out = tmp_path / "out"
    rc = PE.main(["--runs", str(tmp_path / "empty"), "--out", str(out), "--n-boot", "50"])
    assert rc == 0
    df = pd.read_csv(out / "prereg_results.csv")
    assert list(df.columns) == PE.RESULT_COLS + PE.EXTRA_COLS
    assert set(df["verdict"]) == {"not_tested"}
    assert set(df["verdict"]) <= set(PE.VERDICTS)
    reasons = dict(zip(df["id"], df["deviation_note"]))
    assert "no run of stage D" in reasons["P"] and "no run of stage G" in reasons["S7"]
    assert "trace-source stage-L run" in reasons["S18T"]
    assert {"P", "audit.D1", "audit.C", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10i", "S10ii", "S10iii", "S11",
            "S12", "S13", "S14", "S15", "S16", "S17", "S18", "S18T"} <= set(df["id"])
    md = (out / "prereg_results.md").read_text(encoding="utf-8")
    assert md.startswith("# Präregistrierte Auswertung") and "## Primärer Endpunkt" in md and "## BH-Gruppierung" in md
    assert (out / "prereg_runs.csv").exists() and json.loads((out / "prereg_meta.json").read_text())["n_boot"] == 50


def test_missing_table_and_error_rows(tmp_path):
    # a stage-D run without the damage table, and a hysteresis table without its value column
    write_run(tmp_path, "D", {"hysteresis": pd.DataFrame(dict(family=["F01"], formulation=["A"], K=[0], direction=["lang_dom"]))})
    df = PE.evaluate([tmp_path], n_boot=50)["results"].set_index("id")
    assert df.loc["P", "verdict"] == "not_tested" and "table damage missing" in df.loc["P", "deviation_note"]
    assert df.loc["S4", "verdict"] == "not_tested"
    assert "table hysteresis empty or without the required columns" in df.loc["S4", "deviation_note"]


def test_read_table_prefers_parquet_and_falls_back_to_csv(tmp_path):
    pytest.importorskip("pyarrow")
    pd.DataFrame(dict(a=[1, 2], b=[3, 4])).to_parquet(tmp_path / "t.parquet", index=False)
    pd.DataFrame(dict(a=[9, 9], b=[9, 9])).to_csv(tmp_path / "t.csv", index=False)
    assert PE.read_table(tmp_path, "t")["a"].tolist() == [1, 2]
    assert list(PE.read_table(tmp_path, "t", columns=["b", "zz"]).columns) == ["b"]
    (tmp_path / "t.parquet").write_bytes(b"not a parquet file")
    assert PE.read_table(tmp_path, "t")["a"].tolist() == [9, 9]
    assert list(PE.read_table(tmp_path, "t", columns=["b", "zz"]).columns) == ["b"]
    (tmp_path / "e.csv").write_text("\n", encoding="utf-8")
    assert PE.read_table(tmp_path, "e").empty and PE.read_table(tmp_path, "nothing") is None


def test_trace_and_standin_stage_l_runs_are_separate(tmp_path):
    verdict = pd.DataFrame([dict(source_kind="hf", self_consistency=False, best_layer=5, r2_gap_ok=True, cos_ok=True,
                                 sign_agreement_steering=True, readout_alive=True, sign_residual=True, sign_kv=True,
                                 transfer_ok=True, validated=True, counts_for_prereg=True)])
    # a trace_verdict of the tightened rule (lm_corpus_ok and the plain / surface columns): copied as written
    tv = pd.DataFrame([dict(mode=m, probe_ok=True, align_ok=True, align_beats_untrained=True, transplant_effect=(m == "kv"),
                            sharing_effect_claimed=(m == "kv"), counts_for_prereg=True, B_vs_random=0.03, n_clusters=24,
                            probe_beats_surface=True, effect_B_vs_plain=(m == "kv"), lm_corpus_ok=True)
                       for m in ("residual", "kv", "both")])
    write_run(tmp_path / "LSNE" / "runs", "L", {"verdict": verdict}, source=dict(kind="hf", name="EleutherAI/pythia-70m"))
    write_run(tmp_path / "LSNE" / "runs_trace", "L", {"trace_verdict": tv}, source=dict(kind="trace", name="trace_lm"))
    res = PE.evaluate([tmp_path / "LSNE"], n_boot=50)
    df = res["results"].set_index("id")
    assert set(res["runs"][res["runs"]["used"]]["key"]) == {"L", "L_trace"}
    assert df.loc["S18", "verdict"] == "supported"
    assert df.loc["S18T.kv", "verdict"] == "supported" and df.loc["S18T.residual", "verdict"] == "not_supported"
    assert "broken pipeline" in df.loc["S18T.kv", "deviation_note"]       # the review caveat travels with the row
    assert "predates" not in df.loc["S18T.kv", "deviation_note"]


def test_old_trace_verdict_is_rejudged_never_copied(tmp_path, monkeypatch):
    """A trace_verdict written before the tightened rule (no lm_corpus_ok: the first A100 run on
    Drive) is never copied: prereg_eval re-judges it with rejudge_trace, and when that is not
    possible the rows are not_interpretable -- never 'supported' from the old claim."""
    from lsne.experiments import stage_l_transplant as SL

    old = pd.DataFrame([dict(mode=m, probe_ok=True, align_ok=True, transplant_effect=(m == "kv"),
                             sharing_effect_claimed=(m == "kv"), counts_for_prereg=True,
                             B_vs_random=0.0275 if m == "kv" else 0.0, n_clusters=24) for m in ("residual", "kv", "both")])
    write_run(tmp_path / "a" / "runs_trace", "L", {"trace_verdict": old}, source=dict(kind="trace", name="trace_lm"))
    df = PE.evaluate([tmp_path / "a"], n_boot=50)["results"].set_index("id")
    kv = df.loc["S18T.kv"]
    assert kv["verdict"] == "not_interpretable"                            # was "supported" when copied verbatim
    assert "predates the tightened rule" in kv["deviation_note"] and "rejudge_trace failed" in kv["deviation_note"]
    assert (df.loc[[i for i in df.index if i.startswith("S18T")], "verdict"] != "supported").all()
    # rejudge possible: its verdict replaces the table's claim
    rejudged = old.assign(sharing_effect_claimed=False, counts_for_prereg=False, lm_corpus_ok=False,
                          claim="no claim: trace LM corpus: address OOV rate 0.967")
    calls = []

    def fake_rejudge(run_dir, *a, **k):
        calls.append(Path(run_dir))
        return dict(trace_verdict=rejudged, trace_damage=pd.DataFrame())
    monkeypatch.setattr(SL, "rejudge_trace", fake_rejudge)
    df = PE.evaluate([tmp_path / "a"], n_boot=50)["results"].set_index("id")
    assert calls and calls[0].name == TAG
    assert df.loc["S18T.kv", "verdict"] == "not_supported"
    assert "re-judged from the run's tables" in df.loc["S18T.kv", "deviation_note"]
    assert "lm_corpus_ok False" in df.loc["S18T.kv", "deviation_note"]
    assert "align_beats_untrained" in df.loc["S18T.kv", "rule"]


def test_copied_endpoints_placeholder_and_replacement_rules(tmp_path):
    corp = pd.DataFrame([dict(arm=a, requested="data/ttt/dao_fork", source="placeholder", is_placeholder=True, n_docs=30,
                              n_snippets=16, n_tokens=1963) for a in ("injection", "neutral", "nofork", "permuted")])
    eff = pd.DataFrame([dict(arm=a, measure="T_K8", condition="F01A", before=-1.0, after=-1.1, delta=-0.1,
                             delta_injection_minus_control=0.0, exceeds_all_controls=(False if a == "injection" else None))
                        for a in ("injection", "neutral", "nofork", "permuted")])
    write_run(tmp_path, "K", {"ttt_corpora": corp, "ttt_effects": eff})
    chk = pd.DataFrame([dict(subfield="protein_misfolding", rank=9, n_subfields=11, present=True, in_hardest_third=False,
                             hardest="books", counterpole="books", n_sequences=2000)])
    rs = pd.DataFrame([dict(contrast="culture_politics_vs_misfolding", basis="domain_axes", readout="last", n_sites=4, n_z_ge_2=4,
                            frac_z_ge_2=1.0, z_mean=25.0, z_min=9.0, explained_mean=0.09, preregistered=True, min_effect_met=True)])
    write_run(tmp_path, "J", {"pile_rank_check": chk, "rdx_summary": rs})
    df = PE.evaluate([tmp_path], n_boot=50)["results"].set_index("id")
    assert df.loc["S17iii.T_K8", "verdict"] == "not_interpretable" and "placeholder" in df.loc["S17iii.T_K8", "deviation_note"]
    assert df.loc["S17ii", "verdict"] == "not_tested"
    assert df.loc["S16", "verdict"] == "not_supported"
    assert df.loc["S15", "verdict"] == "not_tested" and "books" in df.loc["S15", "deviation_note"]
    assert df.loc["S15.misfolding", "verdict"] == "supported"
