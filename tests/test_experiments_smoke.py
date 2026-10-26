"""End-to-end CPU smoke test of the experiment stages A-L and the CLI (pythia-14m, DEV01).

``python -m lsne.cli run --smoke --stage all`` on the DEV01 family must finish in well under
ten minutes and leave at least one csv table plus a manifest.json in every stage directory
(stage F re-runs B/C/D for the same model/revision in its own sub-run directories; stages
G-L must report no failed substage).  The ``merge`` command is exercised on the result.  A
few fast unit tests cover the helpers of ``lsne.experiments.common`` and the stage modules
(variant limiting, shard partition, pair parsing, direction bank with a synthetic SAE, bank
persistence, realization clouds, substage error capture, domain groups, text pairs).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lsne import stimuli
from lsne.cli import duration_summary, main, merge_runs, parse_stages
from lsne.config import HookSite, RunConfig
from lsne.experiments import STAGE_ORDER, STAGE_TITLES, STAGES
from lsne.experiments import common as C
from lsne.experiments.stage_f_scale import parse_pairs, plan_triples
from lsne.sae import TopKSAE

REPO = Path(__file__).resolve().parent.parent
# Wall-clock budget for the whole --smoke run (A-L) on a shared 4-core CPU box. The run
# takes ~2 min unloaded but several times longer when other processes compete for the
# cores (observed 730 s for A-F alone at load average 7), so the budget guards against
# runaway runs, not against load. It is applied to the process CPU time summed over threads
# (OpenMP spin-waiting under contention still inflates it, hence the generous value).
SMOKE_BUDGET_S = 3600.0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _stage_dir(root: Path, stage: str) -> Path:
    d = root / stage / "pythia-14m@step98000"
    assert d.is_dir(), f"missing stage directory {d}"
    return d


def _assert_run_dir(d: Path):
    csvs = sorted(d.glob("*.csv"))
    assert csvs, f"no csv in {d}"
    assert (d / "manifest.json").exists(), f"no manifest in {d}"
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert m.get("status") == "ok", f"{d}: {m.get('status')} {m.get('error')}"
    assert not (d / "error.txt").exists()
    return csvs, m


# --------------------------------------------------------------------------------------
# the smoke run
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    t0, c0 = time.perf_counter(), time.process_time()
    rc = main(["run", "--stage", "all", "--model", "EleutherAI/pythia-14m", "--revision", "step98000",
               "--dtype", "float32", "--device", "cpu", "--families", "DEV01", "--out", str(out), "--smoke",
               "--no-progress"])
    return dict(out=out, rc=rc, seconds=time.perf_counter() - t0, cpu_seconds=time.process_time() - c0)


def test_smoke_all_stages_produce_tables(smoke_run):
    out, rc = smoke_run["out"], smoke_run["rc"]
    assert rc == 0
    # CPU time (all threads of this process), not wall-clock time: on a shared box other jobs
    # stretch the wall clock several-fold (observed 1 843 s wall while a model trained alongside)
    assert smoke_run["cpu_seconds"] < SMOKE_BUDGET_S, (f"smoke run used {smoke_run['cpu_seconds']:.0f} CPU-s "
                                                       f"({smoke_run['seconds']:.0f} s wall)")
    assert tuple(STAGE_ORDER) == tuple("ABCDEFGHIJKL") and set(STAGE_TITLES) == set(STAGE_ORDER)
    for stage in STAGE_ORDER:
        _csvs, m = _assert_run_dir(_stage_dir(out, stage))
        if stage in "GHIJKL":   # per-substage error capture: nothing may have failed in the smoke run
            assert m["substages"]["n_failed"] == 0, (stage, m["substages"]["errors"])
            assert not (_stage_dir(out, stage) / "substage_errors.txt").exists()
            assert m["prereg"]   # the pre-registration map of the stage (docs/STAGES.md)
    # stage F: one sub-run per (pair, stage) with its own tables and manifest
    f_dir = _stage_dir(out, "F")
    runs = pd.read_csv(f_dir / "runs.csv")
    assert set(runs[runs["stage"] != "load"]["stage"]) == {"B", "C", "D"}
    assert (runs["status"] == "ok").all(), runs.to_string()
    for st in ("B", "C", "D"):
        _assert_run_dir(f_dir / "shard0of1" / st / "pythia-14m@step98000")
    plan = pd.read_csv(f_dir / "plan.csv")
    assert plan["family"].tolist() == ["DEV01"] and plan["mine"].all()
    report = json.loads(next(out.glob("run_report_*.json")).read_text(encoding="utf-8"))
    assert all(r["status"] == "ok" for r in report["stages"].values())


def test_smoke_stage_contents(smoke_run):
    out = smoke_run["out"]
    # A: DEV01A core triple (limit_variants=3) + the two original-context variants
    var = pd.read_csv(_stage_dir(out, "A") / "variants.csv")
    std = var[var["kind"] == "standard"]
    assert sorted(zip(std["K"], std["C"])) == [(0, "C1"), (8, "C1"), (28, "seg")]
    assert (std["formulation"] == "A").all() and (std["target_id"] == std["target_id"].iloc[0]).all()
    orig = var[var["kind"] == "original"]
    if len(orig):   # anchor available (cached or fetched)
        assert set(orig["K"]) == {0, 5} and (orig["target_id"] == 22905).all()
    assert (pd.read_csv(_stage_dir(out, "A") / "exclusions.csv")).empty
    # B: baseline and direction bank
    base = pd.read_csv(_stage_dir(out, "B") / "baseline.csv")
    assert len(base) == len(var) and np.isfinite(base[["logp_target", "entropy", "p_freq"]].to_numpy()).all()
    assert (base["p_target"] > 0).all() and (base["p_freq"] <= 1).all()
    meta = pd.read_csv(_stage_dir(out, "B") / "direction_meta.csv")
    assert {"ent", "lang_dom", "lang_probe", "freq", "rand_00", "rand_01"} <= set(meta["name"])
    assert (meta["site"] == "L5.block_input").all()
    bank = C.load_bank(_stage_dir(out, "B"))
    assert bank is not None and set(bank.keys()) == set(meta["name"])
    for d in bank.values():
        assert d.vec.shape == (128,) and abs(float(d.vec.norm()) - 1.0) < 1e-5
    assert len(bank.freq_ids) == 100
    cos = pd.read_csv(_stage_dir(out, "B") / "cosines.csv")
    assert np.allclose(np.diag(cos.set_index("name").to_numpy()), 1.0)
    # C: grid with the variants' own baselines, contrasts relative to K=0
    grid = pd.read_csv(_stage_dir(out, "C") / "grid.csv")
    assert set(grid["dir_family"]) == {"ent", "lang_dom", "freq", "rand"}
    assert set(grid["sign"]) == {1, -1} and set(grid["eps"]) == {0.0, 0.01, 0.1}
    zero = grid[grid["eps"] == 0.0]
    assert (zero["B"] == 0).all() and np.allclose(zero["logp_delta"], zero["logp_0"])
    assert np.isfinite(grid["B"]).all() and (grid["audit_rel_err"].fillna(0) < 1e-3).all()
    con = pd.read_csv(_stage_dir(out, "C") / "contrasts.csv")
    k8 = con[(con["tag"] == "DEV01A:K8:C1")]
    assert len(k8) and np.allclose(k8["T"], k8["T"].iloc[0])
    assert np.allclose(k8["I"], k8["B_mixed"] - k8["B_mono"])
    att = pd.read_csv(_stage_dir(out, "C") / "attribution.csv")
    # pre-registered grid = last block of each quarter of the depth ({2,5,8,11} for 160m,
    # {1,2,4,5} for the 6 blocks of pythia-14m); block 0 (token embedding) is never patched
    assert {"K0_vs_K8C1", "K0_vs_K28seg"} <= set(att["pair"]) and set(att["layer"]) == {1, 2, 4, 5}
    assert set(att["class"]) == set(C.PATCH_CLASSES) >= {"delim", "post_punct", "ws", "punct", "boundary"}
    # the pre-registered union class R_ws+punct+boundary (`delim`, contrast 3) is measurable with a
    # defined stratified control on every pair and layer; `ws` alone is empty by construction
    # (BPE attaches the space to the word) and flagged, never silently NaN
    delim = att[att["class"] == C.PREREG_PATCH_CLASS]
    assert len(delim) == att["pair"].nunique() * 4
    assert delim["measurable"].all() and delim["control_defined"].all() and (delim["n_positions"] >= 2).all()
    assert np.isfinite(delim[["recovery", "recovery_random_mean", "recovery_random_sd"]].to_numpy()).all()
    assert (delim["n_positions"] >= att[att["class"] == "punct"].set_index(["pair", "layer"]).loc[
        list(zip(delim["pair"], delim["layer"])), "n_positions"].to_numpy()).all()
    ws = att[(att["class"] == "ws") & (att["kind"] == "standard")]
    assert len(ws) and (~ws["measurable"]).all() and ws["recovery"].isna().all() and (ws["n_positions"] == 0).all()
    assert (att["measurable"] == (att["n_positions"] > 0)).all()
    # contrast 3 per layer: per-formulation excess rows and the family cluster bootstrap next to T / I
    acon = pd.read_csv(_stage_dir(out, "C") / "attribution_contrasts.csv")
    assert set(acon["class"]) <= set(C.PATCH_CLASSES) and not {"tail", "all"} & set(acon["class"])
    ad = acon[acon["class"] == C.PREREG_PATCH_CLASS]
    assert len(ad) == len(delim) and ad["preregistered"].all() and (~acon[acon["class"] != "delim"]["preregistered"]).all()
    assert np.allclose(ad["R_excess"], ad["recovery"] - ad["recovery_random_mean"])
    assert set(ad["layer"]) == {1, 2, 4, 5} and set(zip(ad["K"], ad["C"])) >= {(8, "C1"), (28, "seg")}
    boot = pd.read_csv(_stage_dir(out, "C") / "contrasts_boot.csv")
    assert {"T", "I", "R_excess"} <= set(boot["quantity"]) and {"layer", "class"} <= set(boot.columns)
    assert (boot[boot["quantity"] != "R_excess"]["layer"] == -1).all()
    r3 = boot[(boot["quantity"] == "R_excess") & (boot["class"] == C.PREREG_PATCH_CLASS)]
    assert sorted(zip(r3["K"], r3["C"], r3["layer"])) == sorted((K, Cc, l) for (K, Cc) in [(8, "C1"), (28, "seg")]
                                                                 for l in (1, 2, 4, 5))
    assert np.isfinite(r3[["mean", "ci_low", "ci_high"]].to_numpy()).all() and (r3["n_clusters"] == 1).all()
    std_ad = ad[ad["kind"] == "standard"].set_index(["K", "C", "layer"])
    for _, r in r3.iterrows():
        assert r["mean"] == pytest.approx(std_ad.loc[(r["K"], r["C"], r["layer"]), "R_excess"])
    cman = json.loads((_stage_dir(out, "C") / "manifest.json").read_text(encoding="utf-8"))
    assert cman["layer_grid"] == [1, 2, 4, 5] and cman["prereg_patch_class"] == "delim"
    assert cman["patch_classes"] == list(C.PATCH_CLASSES)
    gxa = pd.read_csv(_stage_dir(out, "C") / "grad_x_act_by_class.csv")
    assert gxa["abs_share"].between(0, 1 + 1e-9).all() and set(gxa["layer"]) == {1, 2, 4, 5}
    # D: damage spreading and generation diagnostics
    dmg = pd.read_csv(_stage_dir(out, "D") / "damage.csv")
    assert set(dmg["inject"]) == {"dossier", "depth"} and set(dmg["dir_family"]) == {"ent", "lang_dom", "rand"}
    assert set(dmg["eps"]) == {0.01, 0.1, 0.5}
    depth = dmg[dmg["inject"] == "depth"]
    assert np.isfinite(depth["lam"]).all() and (depth["n_fit_layers"] == 4).all()
    assert (dmg[dmg["inject"] == "dossier"]["n_fit_layers"] == 1).all()
    assert (dmg["D_final_tail"] > 0).all() and (dmg["audit_rel_err"] < 1e-3).all()
    # each injection uses the directions fitted at its own site (rand shared, isotropic)
    assert (dmg["dir_site"] == dmg["site"]).all()
    assert set(depth["dir_site"]) == {"L2.block_input"} and set(dmg[dmg["inject"] == "dossier"]["dir_site"]) == {"L5.block_input"}
    dbank = C.load_bank(_stage_dir(out, "D"), name="directions_depth")
    assert dbank is not None and dbank.site == HookSite(2, "block_input") and {"ent", "lang_dom", "rand_00", "rand_01"} <= set(dbank.keys())
    bank_b = C.load_bank(_stage_dir(out, "B"))
    assert torch.allclose(dbank["rand_00"].vec, bank_b["rand_00"].vec)
    assert not torch.allclose(dbank["lang_dom"].vec, bank_b["lang_dom"].vec)
    dman = json.loads((_stage_dir(out, "D") / "manifest.json").read_text(encoding="utf-8"))
    assert set(dman["cos_dossier_vs_depth"]) >= {"ent", "lang_dom", "rand_00"} and dman["cos_dossier_vs_depth"]["rand_00"] == pytest.approx(1.0)
    assert dman["depth_position"].endswith("last_switched_position") and dman["gen"]["n_null"] == 4
    # the depth injection sits on a German token for every mixed variant, on unit 9 for K=0
    mixed = depth[depth["K"] > 0]
    assert (mixed["inject_lang"] == "D").all() and (depth[(depth["K"] == 0) & (depth["kind"] == "standard")]["inject_unit"] == 9).all()
    assert (depth["inject_pos"] < depth["tail_start"]).all() and (dmg[dmg["inject"] == "dossier"]["inject_pos"] == dmg[dmg["inject"] == "dossier"]["n_tokens"] - 1).all()
    phase = pd.read_csv(_stage_dir(out, "D") / "damage_phase.csv")
    assert (phase[phase["dir_family"] == "rand"]["n"] == 2).all()
    lens = pd.read_csv(_stage_dir(out, "D") / "lens.csv")
    assert set(lens["layer"]) == set(range(7)) and np.allclose(lens[lens["layer"] == 6]["kl_last"], 0.0, atol=1e-5)
    summ = pd.read_csv(_stage_dir(out, "D") / "lens_summary.csv")
    assert summ["nmi_last"].between(0, 1).all()
    ca = pd.read_csv(_stage_dir(out, "D") / "ca.csv")
    assert np.isfinite(ca[["spatial_entropy", "temporal_entropy", "activity", "lz76"]].to_numpy()).all()
    # D9: the N symbol exists (absolute marker-mass floor), E/D still dominate late layers
    assert (ca["mass_floor"] > 0).all() and (ca["frac_N"] > 0).all() and (ca["frac_N"] == ca["frac_below_floor"]).all()
    cal = pd.read_csv(_stage_dir(out, "D") / "ca_layers.csv")
    assert (cal[cal["layer"] == 0]["frac_N"].mean() > cal[cal["layer"] == 6]["frac_N"].mean())
    hy = pd.read_csv(_stage_dir(out, "D") / "hysteresis.csv")
    assert set(hy["direction"]) == {"ent", "lang_dom"}
    # NaN-aware areas (ROUND2 G3.2): a plateau without E/D-labelled tokens is missing data, so with
    # the 2-token smoke plateaus an area may be NaN; the standard variants must still be finite
    hy_std = hy[hy["kind"] == "standard"]
    assert len(hy_std) and np.isfinite(hy_std[["area", "null_area", "null_area_sd"]].to_numpy()).all()
    assert (hy["kind"][hy[["area", "null_area"]].isna().any(axis=1)] == "original").all()
    assert set(hy["K"]) >= {0, 8, 28} and (hy["n_null"] == 4).all() and (hy["n_null_pairs"] == 6).all()
    curves = pd.read_csv(_stage_dir(out, "D") / "hysteresis_curves.csv")
    assert set(curves["branch"]) == {"up", "down", "restart_a", "restart_b", "restart_c", "restart_d"}
    rs = curves[curves["replicate"] >= 0]
    # ramp levels 0, eps/2, eps (zero-dose plateaus at both ends, ROUND2 G3.1); one restart seed per
    # (replicate, level): 4 replicates x 3 levels
    n_levels = int(rs["level"].nunique())
    assert n_levels == 3 and rs["seed"].min() >= 10_000
    assert rs.groupby(["tag", "direction"])["seed"].nunique().eq(4 * n_levels).all()
    up = curves[curves["branch"] == "up"].sort_values(["tag", "direction", "step"])
    assert (up.groupby(["tag", "direction"])["eps"].first() == 0).all() and (up.groupby(["tag", "direction"])["eps"].last() == 0.5).all()
    rx = pd.read_csv(_stage_dir(out, "D") / "relaxation.csv")
    assert (rx["eps"] == 0.5).all() and (rx["n_new"] == 8).all()
    assert {"tau_raw", "tau_censored", "identifiable", "reason", "b_se", "r2", "f_pvalue", "b_lo", "b_hi"} <= set(rx.columns)
    assert (rx["tau"].notna() <= rx["identifiable"].astype(bool)).all()
    # the gate no longer shrinks the residual df by the label window: for fitted rows
    # n_eff = n / vif >= n / window (unfitted rows - constant or too short - keep n_eff = 0)
    fitted = rx["tau_raw"].notna()
    assert (rx.loc[fitted, "n_eff"] >= rx.loc[fitted, "n_fit"] / 8.0 - 1e-9).all() and (rx.loc[fitted, "n_eff"] > 0).all()
    assert (rx.loc[~fitted, "n_eff"] == 0).all()
    assert (rx["identifiable"].astype(bool) == (rx["reason"].fillna("") == "")).all()
    su = pd.read_csv(_stage_dir(out, "D") / "susceptibility.csv")
    assert set(su["order_parameter"]) == {"fD", "proj_lang_dom"} and set(su["eps"]) == {0.0, 0.5}
    fdt = pd.read_csv(_stage_dir(out, "D") / "fdt.csv")
    assert np.isfinite(fdt["var0_proj"]).all() and (fdt["n_samples"] == 2).all()
    # E: tensor network on K=0 and K=8/C1 of DEV01A
    tn = pd.read_csv(_stage_dir(out, "E") / "tn_summary.csv")
    assert sorted(set(tn["tag"])) == ["DEV01A:K0:C1", "DEV01A:K8:C1"]
    assert set(tn["direction"]) == {"ent", "lang_dom"} and set(tn["eps"]) == {0.01, 0.1}
    assert np.isfinite(tn[["nonlinearity_last", "factorization_error_last", "linear_norm_last",
                           "nonlinearity_odd_wmean", "nonlinearity_even_wmean", "factorization_error_inspan_wmean",
                           "factorization_error_full_wmean", "truncation_error_wmean", "linear_captured_wmean",
                           "linear_captured_frob", "noise_floor_f32_wmean", "snr_f32_last"]].to_numpy()).all()
    assert (tn["linear_norm_last"] > 0).all() and tn["source_captured"].between(0, 1 + 1e-6).all()
    # the measurement runs in float64 with a float64 softmax; the float32 floor is recorded
    assert (tn["measured_dtype"] == "float64").all() and tn["softmax_float64"].all()
    assert (tn["noise_floor_f32_wmean"] > 0).all() and (tn["n_below_floor_f32"] >= 0).all()
    # response-fitted basis: the source lies in the span and the output response is largely captured
    assert (tn["source_captured"] > 0.9).all()
    assert tn["linear_captured_wmean"].between(0.5, 1 + 1e-6).all() and tn["linear_captured_frob"].between(0.5, 1 + 1e-6).all()
    # the factorisation error is no longer the truncation floor (~1.0 with the state PCA)
    assert (tn["factorization_error_wmean"] < 0.95).all() and (tn["full_rank_reference"] == "computed").all()
    assert tn["factorization_error_inspan_wmean"].between(0, 1.5).all()
    # D8: linear regime at eps = 0.01: nonlinearity ~ eps, second order ~ eps, odd part ~ 0
    small = tn[tn["eps"] == 0.01]
    assert (small["nonlinearity_odd_wmean"] < 0.01).all() and (small["nonlinearity_wmean"] < 0.05).all()
    for (tag, dr), g in tn.groupby(["tag", "direction"]):
        g = g.set_index("eps")
        for col in ("nonlinearity_wmean", "nonlinearity_even_wmean", "nonlinearity_last"):
            ratio = g.loc[0.1, col] / g.loc[0.01, col]
            assert 5.0 < ratio < 20.0, (tag, dr, col, ratio)
    cmp_ = pd.read_csv(_stage_dir(out, "E") / "tn_compare.csv")
    assert {"nonlinearity_odd", "nonlinearity_even", "linear_captured", "factorization_error_inspan",
            "factorization_error_full", "truncation_error", "noise_floor_f32", "snr_f32", "above_floor_f32",
            "measured_norm_minus"} <= set(cmp_.columns)
    assert cmp_["above_floor_f32"].dtype == bool and (cmp_["snr_f32"] > 0).all()
    paths = pd.read_csv(_stage_dir(out, "E") / "tn_paths.csv")
    assert {"tnco", "opt_einsum", "reference_greedy", "reference_dp"} == set(paths["backend"])
    assert set(paths["basis"]) == {"response_svd", "identity"} and set(paths["r"]) == {16, 128}
    assert (paths["n_tensors"] == 9).all() and np.isfinite(paths["cost_log10_opt_einsum"]).all()
    basis = pd.read_csv(_stage_dir(out, "E") / "tn_basis.csv")
    assert len(basis) == 16 and basis["explained_variance_cum"].is_monotonic_increasing
    assert (basis["source"] == "linear_response_svd").all() and (basis["n_rows"] == 2 * 2 * 5 * 16).all()
    assert basis["explained_variance_cum"].iloc[-1] > 0.8 and basis["explained_variance_states_cum"].is_monotonic_increasing
    eman = json.loads((_stage_dir(out, "E") / "manifest.json").read_text(encoding="utf-8"))
    assert eman["measured_dtype"] == "float64" and eman["softmax_float64"] and eman["full_rank_reference"] == "computed"


def _csv(out, stage, name):
    return pd.read_csv(_stage_dir(out, stage) / f"{name}.csv")


def _manifest(out, stage):
    return json.loads((_stage_dir(out, stage) / "manifest.json").read_text(encoding="utf-8"))


def test_smoke_stage_g_geometry(smoke_run):
    out = smoke_run["out"]
    snaps = {f"L{l}.block_input" for l in range(6)} | {"L5.block_output"}
    clouds = _csv(out, "G", "clouds")
    n_forms = clouds[clouds["K"] == 0]["formulation"].nunique()
    assert n_forms >= 1 and set(clouds["K"]) == {0, 8} and set(clouds["C"]) == {"C1"} and not clouds["excluded"].any()
    assert len(clouds[clouds["K"] == 8]) == 3 * n_forms and clouds[clouds["K"] == 8]["unit_mask"].nunique() == 3
    for m in clouds[clouds["K"] == 8]["unit_mask"]:      # unit 1 / 10 English, unit 9 German, exactly 8 switches
        assert m[0] == "E" and m[-1] == "E" and m[-2] == "D" and sum(a != b for a, b in zip(m, m[1:])) == 8
    geom = _csv(out, "G", "geometry")
    assert set(geom["site"]) == snaps and set(geom["readout"]) == {"mean", "last"} and set(geom["k"]) == {0, 5}
    assert set(geom["group"]) == {"DEV01|K0|C1", "DEV01|K8|C1"} and np.isfinite(geom["log_volume"]).all()
    k0 = geom[geom["K"] == 0]
    # the K=0 cloud is its own reference; the Mahalanobis displacement is undefined only where the pooled
    # states have no variance at all (block-0 last token: the identical tail token embedding, r = 0)
    assert (k0["disp_euclid"] == 0).all() and ((k0["disp_mahal"] == 0) | (k0["disp_mahal"].isna() & (k0["r"] == 0))).all()
    assert (geom[(geom["readout"] == "last") & (geom["site"] == "L0.block_input")]["r"] == 0).all()
    gate = _csv(out, "G", "g0_gate")
    assert len(gate) == 1 and gate["n_layers"].iloc[0] == 7 and gate["k"].iloc[0] == 5
    # the gate rank is capped at (N-1)//4 so the whitened metric cannot collapse onto a simplex;
    # a gate that is still degenerate gives no verdict (pass flags empty, never True)
    N = int(gate["n_states"].iloc[0])
    assert N == 2 * n_forms and int(gate["r_requested"].iloc[0]) == 128 and int(gate["r"].iloc[0]) == max(1, (N - 1) // 4)
    if bool(gate["degenerate"].iloc[0]):
        assert gate["pass_all"].isna().all() and gate["pass_middle"].isna().all()
    else:
        assert gate["pass_all"].iloc[0] in (True, False) and gate["pass_middle"].iloc[0] in (True, False)
    po = _csv(out, "G", "pair_ordering")
    assert set(po["category"]) >= {"exact_translation", "same_meaning_same_lang", "other_family"} and set(po["site"]) == snaps
    depth = _csv(out, "G", "depth_trajectory")
    assert (depth["n_layers"] == 7).all() and set(depth["layer"]) == set(range(7)) and depth["breathing_index"].between(0, 5).all()
    con = _csv(out, "G", "contraction")
    pre = con[con["preregistered"]]
    assert len(pre) == 7 * 2 * 1 and (pre["k"] == 5).all() and (pre["n_ref"] == n_forms).all() and (pre["n_mixed"] == 3 * n_forms).all()
    assert (con["n_subsample"] >= 1).all() and np.isfinite(con.loc[con["readout"] == "mean", "d_log_volume"]).all()
    # the raw contrast carries the n-dependence of the log-volume (12 vs 4 eigenvalues); the matched one does not
    assert (con.loc[con["readout"] == "mean", "d_log_volume_raw"].abs() > con.loc[con["readout"] == "mean", "d_log_volume"].abs()).all()
    boot = _csv(out, "G", "contraction_boot")
    bp = boot[boot["preregistered"]]
    assert sorted(set(zip(bp["quantity"], bp["readout"]))) == [("d_log_volume", "mean"), ("d_participation_ratio", "last")]
    # one family -> one cluster per row; the participation ratio is undefined (0/0) where all pooled
    # states coincide (block-0 last token = the identical tail-token embedding): NaN, no cluster
    undefined = bp["mean"].isna()
    assert len(bp) == 14 and (bp.loc[~undefined, "n_clusters"] == 1).all() and (bp.loc[undefined, "n_clusters"] == 0).all()
    assert set(bp.loc[undefined, "site"]) <= {"L0.block_input"} and (bp.loc[undefined, "readout"] == "last").all()
    pos = _csv(out, "G", "position_traj")
    assert set(pos["region"]) == {"tail", "prefix_bin", "from_switch"} and set(pos["site"]) == {"L5.block_input", "L3.block_input"}
    assert set(pos["group"]) == {"DEV01|K0|C1", "DEV01|K8|C1"}
    st = _csv(out, "G", "steering")
    assert set(st["inject_site"]) == {"L5.block_input", "L2.block_input"} and set(st["eps"]) == {0.05, 0.5}
    assert (st["direction"] == "lang_dom").all() and (st["n_random"] == 2).all() and st["cos_steer_vs_K"].between(-1, 1).all()
    assert (st[st["inject_site"] == "L2.block_input"]["snapshot"].nunique() == 5) and (st[st["inject_site"] == "L5.block_input"]["snapshot"].nunique() == 2)
    assert st.groupby(["inject_site", "eps"])["at_max_disp_layer"].sum().eq(1).all()
    sb = _csv(out, "G", "steering_boot")
    top = sb[sb["quantity"] == "cos_at_max_disp_layer"]
    assert len(top) == 4 and top["preregistered"].sum() == 2 and top["min_effect_met"].isin([True, False]).all()
    # T5 coupling with stage D of the same run: both partners joined on (family, K, C, layer)
    cp = _csv(out, "G", "coupling")
    assert set(cp["partner"]) == {"lambda", "kl"} and (cp["n"] >= 3).all() and cp["rho"].dropna().between(-1, 1).all()
    gm = _manifest(out, "G")
    # the manifest mirrors the gate row (rank capped at (N-1)//4; a degenerate gate has no verdict)
    g0 = gm["g0_gate"]
    assert gm["stage_d_available"] and set(gm["steering_sites"]) == {"L5.block_input", "L2.block_input"}
    assert int(g0["r"]) == max(1, (int(g0["n_states"]) - 1) // 4) and (g0["pass_all"] is None) == bool(g0["degenerate"])


def test_smoke_stage_h_lambda(smoke_run):
    out = smoke_run["out"]
    p = _csv(out, "H", "lambda_paths")
    assert set(p["discriminant"]) == {"target_logit", "probe", "sae_feature"} and set(p["lam_rel"]) == {0.0, 0.3, 3.0, 30.0}
    assert (p["sae_source"] == "synthetic_random").all() and (p["context"] == "standard").all() and (p["site"] == "L5.block_input").all()
    assert p["tag"].nunique() == 1 and (p["n_steps"] <= 6).all() and (p["n_target"] == 3 * _csv(out, "G", "clouds")[lambda d: d["K"] == 0]["formulation"].nunique()).all()
    assert np.isfinite(p[["B", "entropy", "kl_from_base", "mahal_to_ref_mean", "sae_fvu", "path_length", "initial_g"]].to_numpy()).all()
    assert (p.loc[np.isclose(p["lam_rel"], 0.0), "final_density"] <= 1.0).all() and (p["lambda_unit"] > 0).all()
    z = p[np.isclose(p["lam_rel"], 0.0)]
    assert (z["lam"] == 0).all() and (z["g_drop"] >= -1e-5).all()      # lambda = 0: pure descent of g
    steps = _csv(out, "H", "lambda_steps")
    assert set(steps["discriminant"]) == set(p["discriminant"]) and (steps.groupby(["discriminant", "lam_rel"])["step"].min() == 0).all()
    corr = _csv(out, "H", "lambda_corr")
    pre = corr[corr["preregistered"]]
    assert set(pre["quantity"]) == {"sae_fvu", "steps_to_cross_censored", "mahal_to_ref_mean"} and (pre["discriminant"] == "target_logit").all()
    boot = _csv(out, "H", "lambda_boot")
    bp = boot[boot["preregistered"]]
    assert len(bp) == 3 and set(bp["expected_sign"]) == {-1, 0}
    feats = _csv(out, "H", "lambda_features")
    assert len(feats) and feats["candidate"].dtype == bool and feats["rho"].dropna().between(-1, 1).all()
    orig = _csv(out, "H", "lambda_original")
    if len(orig):   # anchor available (cached or fetched)
        assert (orig["context"] == "original_2143").all() and orig["target_id"].iloc[0] == 22905 and len(orig) == 4
        assert orig["same_sign"].dtype == bool and np.isfinite(orig[["B_lambda0", "B_lambda3"]].to_numpy()).all()
        assert (orig["target_cloud"] == "orig_K>0 + std_K8C1").all()
    hm = _manifest(out, "H")
    assert hm["kde"]["metric"] == "mahalanobis" and hm["path"]["ball_metric"] == "l2" and hm["n_starts"] == 1


def test_smoke_stage_i_compass(smoke_run):
    out = smoke_run["out"]
    comp = _csv(out, "I", "compass")
    dirs = {"state", "ent", "lang_dom", "freq", "dom_law", "dom_culture_politics", "dom_protein_misfolding"}
    assert set(comp["direction"]) == dirs and set(comp["lang"]) == {"E", "D"}
    assert comp.groupby(["direction", "lang"])["domain"].nunique().eq(11).all() and comp.groupby(["direction", "lang"])["is_top"].sum().eq(1).all()
    assert (comp[comp["direction"] != "state"]["eps"] == 0.05).all() and np.isfinite(comp["z"]).all()
    summ = _csv(out, "I", "compass_summary")
    ent = summ[(summ["direction"] == "ent") & (summ["lang"] == "E")]
    assert len(ent) == 1 and ent["law_rank"].between(1, 11).all() and ent["law_is_top"].dtype == bool
    assert _csv(out, "I", "atlas_overlap").empty           # no atlas extraction in the repository
    axes = _csv(out, "I", "domain_axes")
    assert set(axes["name"]) == {"dom_law", "dom_culture_politics", "dom_protein_misfolding"} and (axes["site"] == "L5.block_input").all()
    assert axes["cos_denoised_plain"].between(-1, 1).all() and np.allclose(np.diag(axes[[f"cos_{n}" for n in axes["name"]]].to_numpy()), 1.0)
    assert C.load_bank(_stage_dir(out, "I"), name="directions_domain") is not None
    dd = _csv(out, "I", "direction_domain")
    assert set(dd["direction"]) == {"dom_culture_politics_cs", "dom_culture_politics", "dom_law", "lang_dom", "rand"}
    assert set(dd["eps"]) == {0.1} and (dd["family_domain"] == "everyday").all() and sorted(set(dd["K"])) == [0, 8]
    cs = dd[dd["direction"] == "dom_culture_politics_cs"]
    assert (cs["arm"] == "counter_steer_replace").all() and (cs["cos_after_remove"].abs() < 1e-5).all()
    assert (dd[dd["direction"] == "rand"]["n_dirs"] == 2).all() and (dd["audit_rel_err"].dropna() < 1e-3).all()
    assert np.allclose(dd["B"], dd["logp_delta"] - dd["logp_0"])
    T = _csv(out, "I", "direction_domain_T")
    assert len(T) == 1 and np.isfinite(T["T"]).all() and T["family_domain"].iloc[0] == "everyday"
    db = _csv(out, "I", "direction_domain_boot")
    q = db[db["quantity"] == "B_cs_minus_lang_dom"]
    assert len(q) == 2 and (~q["preregistered"]).all() and q["min_effect_met"].isin([True, False]).all()
    eff = _csv(out, "I", "head_effects")
    assert set(eff["layer"]) == {1, 2, 4, 5} and set(eff["head"]) == {0, 1, 2, 3} and eff["tag"].nunique() == 2
    assert np.allclose(eff["delta"], eff["logp_ablated"] - eff["logp_base"])
    direct = _csv(out, "I", "head_direct")
    assert len(direct) == 2 * 4 * 4 and np.isfinite(direct["direct_effect"]).all()
    wh = _csv(out, "I", "worst_head")
    assert len(wh) == 16 and wh["selected"].sum() == 1 and (wh["selection"] == "fallback_all_families_vs_none").all()
    assert wh.iloc[0]["selected"] and wh["score"].is_monotonic_increasing
    hs = _csv(out, "I", "head_steering")
    assert set(hs["arm"]) == {"head", "residual"} and set(hs["direction"]) == {"lang_dom", "dom_culture_politics"} and set(hs["eps"]) == {0.5}
    assert (hs[hs["arm"] == "residual"]["cos_realised_direction"] == 1.0).all() and hs[hs["arm"] == "head"]["cos_realised_direction"].between(-1, 1).all()
    assert (hs["layer"] == wh.iloc[0]["layer"]).all() and (hs["head"] == wh.iloc[0]["head"]).all()
    hb = _csv(out, "I", "head_steering_boot")
    assert len(hb) == 2 and (hb["quantity"] == "B_head_minus_residual").all() and np.isfinite(hb["mean"]).all()
    im = _manifest(out, "I")
    assert im["state_source"] in ("original_2143", "DEV01A:K0:C1") and im["worst_head"]["selection"] == "fallback_all_families_vs_none"


def test_smoke_stage_j_rdx(smoke_run):
    out = smoke_run["out"]
    jm = _manifest(out, "J")
    pile = _csv(out, "J", "pile_difficulty")
    assert len(pile) == 3 and (pile["n_tokens"] == 63).all() and np.isfinite(pile["mean_nll"]).all()
    assert set(pile["subfield_lexicon"]) <= set(["other"] + list(json.loads((REPO / "data/lexicon/domains.json").read_text())["domains"]))
    rank = _csv(out, "J", "pile_ranking")
    assert rank["rank"].tolist() == list(range(1, len(rank) + 1)) and rank["mean_nll"].is_monotonic_decreasing
    chk = _csv(out, "J", "pile_rank_check")
    assert len(chk) == 1 and chk["subfield"].iloc[0] == "protein_misfolding" and chk["n_sequences"].iloc[0] == 3
    assert chk["counterpole"].iloc[0] == (chk["hardest"].iloc[0] if not chk["in_hardest_third"].iloc[0] else "protein_misfolding")
    rdx = _csv(out, "J", "rdx")
    assert set(rdx["contrast"]) == {"E_vs_D", "K0_vs_K8"} and set(rdx["basis"]) == {"domain_axes", "heads", "sae", "pca"}
    assert set(rdx["site"]) == {"L4.block_input", "L5.block_input"} and set(rdx["readout"]) == {"last", "mean"}
    assert rdx["explained"].between(0, 1).all() and (rdx["delta_norm"] > 0).all() and (~rdx["preregistered"]).all()
    assert (rdx.loc[rdx["basis"] == "sae", "sae_source"] == "synthetic_random").all() and (rdx.loc[rdx["basis"] == "heads", "k"] == 4).all()
    assert set(rdx["unit"]) == {"pooled", "DEV01"}
    assert (rdx.loc[rdx["basis"] == "pca", "explained"] > 0.99).all()   # a one-row cloud per side: the PC spans the difference
    summ = _csv(out, "J", "rdx_summary")
    assert (summ["n_sites"] == 2).all() and summ["frac_z_ge_2"].between(0, 1).all() and summ["min_effect_met"].dtype == bool
    stab = _csv(out, "J", "rdx_stability")
    assert len(stab) and (stab["n_families"] == 1).all()
    sh = _csv(out, "J", "sharing")
    assert set(sh["kind"]) == {"sharing", "activity", "heads"} and (sh["sae_source"] == "synthetic_random").all()
    assert sh[sh["kind"] == "sharing"]["jaccard"].between(0, 1).all() and set(sh[sh["kind"] == "heads"]["layer"]) == {1, 2, 4, 5}
    assert set(sh[sh["kind"] == "activity"]["a"]) == {"everyday", "K0", "K8C1"}
    assert jm["contrasts"] == {"E_vs_D": [1, 1], "K0_vs_K8": [1, 1]} and jm["pile"]["n"] == 3


def test_smoke_stage_k_ttt(smoke_run):
    out = smoke_run["out"]
    corp = _csv(out, "K", "ttt_corpora")
    assert corp["arm"].tolist() == ["injection", "neutral", "nofork", "permuted"] and corp["is_placeholder"].all()
    assert (corp["n_snippets"] == corp["n_snippets"].iloc[0]).all() and (corp["n_tokens"] == corp["n_tokens"].iloc[0]).all()
    mem = _csv(out, "K", "ttt_memorisation")
    assert len(mem) == 8 and (mem["checkpoint"] == "step98000").all() and set(mem["arm"]) == set(corp["arm"])
    assert np.isfinite(mem[["nll_start", "final_nll", "auc_drop"]].to_numpy()).all() and mem["is_placeholder"].all()
    assert (mem["reached_threshold"] == mem["steps_to_threshold"].notna()).all()
    # the permuted arm starts from a higher NLL than the readable text
    assert mem[mem["arm"] == "permuted"]["nll_start"].mean() > mem[mem["arm"] == "injection"]["nll_start"].mean()
    mb_ = _csv(out, "K", "ttt_memorisation_boot")
    assert {"injection", "injection-neutral", "injection-nofork", "injection-permuted"} <= set(mb_["arm"])
    assert set(mb_["quantity"]) == {"steps_to_threshold", "auc_drop", "final_nll"}
    tr = _csv(out, "K", "ttt_training")
    assert len(tr) == 4 * 3 and set(tr["step"]) == {1, 2, 3} and np.isfinite(tr["loss"]).all()
    ba = _csv(out, "K", "ttt_before_after")
    assert set(ba["phase"]) == {"before", "after"} and set(ba[ba["phase"] == "after"]["arm"]) == set(corp["arm"])
    measures = {"logp_target", "B_lang_dom", "T_K8", "sharing_jaccard", "activity_frac_active", "activity_mean_mass", "head_delta",
                "head_delta_mean", "log_volume", "participation_ratio"}
    assert measures <= set(ba["measure"]) and (ba["sae_source"] == "synthetic_random").all()
    before = ba[ba["phase"] == "before"]
    assert (before["arm"].isna() | (before["arm"] == "")).all() and before["measure"].value_counts()["logp_target"] == 2
    eff = _csv(out, "K", "ttt_effects")
    assert set(eff["arm"]) == set(corp["arm"]) and np.allclose(eff["delta"], eff["after"] - eff["before"])
    inj = eff[eff["arm"] == "injection"]
    # identical placeholder corpora in the injection / neutral / nofork arms: no effect can exceed all controls
    assert (~inj["exceeds_all_controls"].astype(bool)).all() and inj["delta_injection_minus_control"].isna().all()
    assert np.allclose(eff[eff["arm"] == "neutral"]["delta_injection_minus_control"], 0.0)
    km = _manifest(out, "K")
    assert km["is_placeholder"] and km["memorisation"]["threshold"] == 0.5 and km["training"]["steps"] == 3


def test_smoke_stage_l_transplant(smoke_run):
    out = smoke_run["out"]
    tt = _csv(out, "L", "transfer_test")
    assert tt["layer_tgt"].tolist() == [1, 2, 4, 5] and (tt["layer_src"] == tt["layer_tgt"]).all() and (tt["n_null"] == 3).all()
    # the rotated-self stand-in: the alignment has to undo a known isometry; it beats the permuted-pairs null clearly
    assert (tt["cos_transferred_vs_native"] > 0.9).all() and (tt["cos_transferred_vs_native"] > tt["cos_null_max"]).all()
    assert (tt["r2_holdout"] > tt["r2_permuted"]).all() and (tt["r2_gap"] >= 0.2).all() and tt["r2_gap_ok"].all() and tt["cos_ok"].all()
    assert (tt["n_pairs"] == 28).all() and tt["readout_alive"].all() and (tt["n_ED_baseline"] > 0).all()
    assert (tt["validated"] == (tt["r2_gap_ok"] & tt["cos_ok"] & tt["sign_agreement"])).all()
    summ = _csv(out, "L", "transfer_summary")
    assert len(summ) == 1 and summ["best_layer"].iloc[0] in tt["layer_tgt"].tolist() and summ["n_layers"].iloc[0] == 4
    if tt["validated"].any():
        assert bool(summ["validated"].iloc[0]) and summ["n_validated_layers"].iloc[0] == int(tt["validated"].sum())
    lm = _csv(out, "L", "layer_matching")
    assert set(lm["layer_tgt"]) == set(lm["layer_src"]) == set(range(7)) and len(lm) == 49 and (lm["method"] == "procrustes").all()
    assert (lm["gap"] > 0).all()
    al = _csv(out, "L", "alignments")
    assert al["layer_tgt"].tolist() == list(range(7)) and (al["r2_holdout"] > al["r2_permuted"]).all()
    tp = _csv(out, "L", "transplant")
    assert len(tp) == 12 and set(tp["mode"]) == {"residual", "kv", "both"}
    assert set(tp["control"]) == {"none", "random", "identity", "permuted_alignment"}
    assert (tp.loc[tp["mode"] != "residual", "kv_layers"] == "0,1,2,3,4,5,6").all() and (tp.loc[tp["mode"] != "residual", "n_prefix"] == 2).all()
    ident = tp[(tp["mode"] == "residual") & (tp["control"] == "identity")]
    assert (ident["written_norm"] == 0).all() and (tp[(tp["mode"] == "residual") & (tp["control"] == "none")]["written_norm"] > 0).all()
    assert np.isfinite(tp[["entropy", "logp_target", "fD"]].to_numpy()).all() and tp["prompt_idx"].nunique() == 1
    ts = _csv(out, "L", "transplant_summary")
    assert set(ts["mode"]) == {"residual", "kv", "both"} and set(ts["sign_fD_vs_identity"]) <= {-1, 0, 1}
    assert ts["sign_agrees_with_native"].dtype == bool
    lman = _manifest(out, "L")
    assert lman["source"]["kind"] == "rotated_self" and not lman.get("stub")
    assert lman["n_texts"]["E"] == lman["n_texts"]["D"] == 12 and 1 <= lman["n_texts"]["neutral"] <= 12


def test_merge_concatenates_runs(smoke_run, tmp_path):
    out = smoke_run["out"]
    merged = tmp_path / "merged"
    counts = merge_runs(out, merged)
    assert counts["baseline"] > 0 and (merged / "baseline.csv").exists()
    df = pd.read_csv(merged / "baseline.csv")
    # the stage-B run and the stage-F sub-run of B
    assert list(df.columns[:4]) == ["run_id", "run_stage", "run_model", "run_revision"]
    assert set(df["run_id"]) == {"B/pythia-14m@step98000", "F/pythia-14m@step98000/shard0of1/B/pythia-14m@step98000"}
    assert (df["run_stage"] == "B").all() and (df["run_model"] == "pythia-14m").all()
    assert (df["run_revision"] == "step98000").all()
    manifests = json.loads((merged / "manifests.json").read_text(encoding="utf-8"))
    assert len(manifests) >= 15 and all("run_id" in m for m in manifests)
    assert counts["lambda_paths"] > 0 and counts["transfer_test"] == 4 and counts["ttt_effects"] > 0
    index = pd.read_csv(merged / "merge_index.csv")
    assert set(index["table"]) == set(counts)
    # the merge command through the CLI
    assert main(["merge", "--root", str(out), "--out", str(tmp_path / "merged2")]) == 0
    assert (tmp_path / "merged2" / "damage.csv").exists()


# --------------------------------------------------------------------------------------
# fast unit tests of the helpers
# --------------------------------------------------------------------------------------
def _dev01a(tok):
    fam = next(f for f in stimuli.load_families() if f.tag == "DEV01A")
    return stimuli.build_variants(fam, tok)


def test_apply_variant_limit_core_first(tok):
    vs = _dev01a(tok)
    assert len(vs) == 12
    three = C.apply_variant_limit(vs, 3)
    assert [(v.K, v.C) for v in three] == [(0, "C1"), (8, "C1"), (28, "seg")]
    assert len(C.apply_variant_limit(vs, None)) == 12 and len(C.apply_variant_limit(vs, 100)) == 12
    core = C.core_subset(vs)
    assert sorted((v.K, v.C) for v in core) == [(0, "C1"), (8, "C1"), (28, "seg")]
    assert C.unit_last_position(vs[0]) < vs[0].tail_start
    assert vs[0].annot["unit"][C.unit_last_position(vs[0])] == 9


def test_load_variants_and_smoke_config(mb):
    cfg = C.smoke_config(RunConfig(model=mb.spec, families=["DEV01"]))
    assert cfg.limit_variants == 3 and cfg.limit_families == 1 and cfg.n_random_dirs == 2
    variants, checks = C.load_variants(cfg, mb)
    assert [v.tag for v in variants] == ["DEV01A:K0:C1", "DEV01A:K8:C1", "DEV01A:K28:seg"]
    n_forms = len([f for f in stimuli.load_families() if f.family_id == "DEV01"])     # A/B (+ C/D once shipped)
    assert n_forms >= 2 and len(checks) == 12 * n_forms and not checks["excluded"].any()
    cfg2 = RunConfig(model=mb.spec, families=["DEV01B", "DEV02"], limit_families=1)
    fams = C.select_families(cfg2)
    assert [f.tag for f in fams] == ["DEV01B"]


def test_damage_tables_batched_and_nothing_retained(mb, tok, monkeypatch):
    """D1 runs all (direction, dose) pairs of an injection through damage_spreading_batch and
    keeps nothing per variant (field report: a full reference cache with [1, T, V] logits per
    variant on the GPU ran a 40 GB A100 out of memory after ~450 pythia-160m variants).
    The tables equal those of the sequential per-pair path."""
    import gc
    import warnings
    from lsne.directions import Direction
    from lsne.dynamics import damage_spreading
    from lsne.experiments import stage_d_noneq as SD

    vs = C.apply_variant_limit(_dev01a(tok), 3)
    site, dsite = C.dossier_site(mb), C.depth_site(mb)
    g = torch.Generator().manual_seed(0)
    banks = {s.key: C.DirectionBank(site=s, dirs={n: Direction(n, torch.randn(mb.d_model, generator=g), s) for n in names})
             for s, names in ((site, ("lang_dom", "rand_00", "rand_01")), (dsite, ("lang_dom", "rand_00")))}
    names, doses = ["lang_dom", "rand_00", "rand_01"], (0.01, 0.5)
    calls = []
    real_rwc = SD.run_with_cache
    monkeypatch.setattr(SD, "run_with_cache", lambda *a, **k: calls.append(1) or real_rwc(*a, **k))

    def n_tensors():
        gc.collect()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sum(1 for o in gc.get_objects() if isinstance(o, torch.Tensor))

    with pytest.warns(UserWarning, match="rand_01"):     # missing at the depth site: skipped there
        SD.damage_tables(mb, vs[:1], banks, names, doses, enabled=False, batch_size=4)    # warm-up
    n0 = n_tensors()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = SD.damage_tables(mb, vs, banks, names, doses, enabled=False, batch_size=4)
    assert isinstance(out, tuple) and len(out) == 2          # (damage, damage_profile): no cache dict
    assert all(isinstance(t, pd.DataFrame) for t in out)
    assert n_tensors() == n0 and not calls                     # no tensor survives, no reference cache built
    dmg, prof = out
    assert len(dmg) == len(vs) * (3 + 2) * len(doses) and len(prof) == len(dmg) * (mb.n_layers + 1)
    assert (dmg["audit_rel_err"] < 1e-4).all() and (dmg[dmg["inject"] == "dossier"]["direction"] == "rand_01").any()
    # same tables as the sequential path (one damage_spreading per pair)
    def sequential(mb_, ids, site_, pos, dirs, eps, target_id=None, batch_size=None):
        return [damage_spreading(mb_, ids, site_, pos, d, e, target_id=target_id) for d, e in zip(dirs, eps)]
    monkeypatch.setattr(SD, "damage_spreading_batch", sequential)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dmg_s, prof_s = SD.damage_tables(mb, vs, banks, names, doses, enabled=False)
    for a, b in ((dmg, dmg_s), (prof, prof_s)):
        assert list(a.columns) == list(b.columns) and a.shape == b.shape
        num = [c for c in a.columns if a[c].dtype.kind == "f"]
        rest = [c for c in a.columns if c not in num]
        assert a[rest].equals(b[rest])
        assert np.allclose(a[num].to_numpy(), b[num].to_numpy(), rtol=1e-4, atol=1e-5, equal_nan=True)


def _sequential_grid_rows(mb, variants, bank, site, doses, dir_names):
    """The grid as it was computed before batching: one full forward per dosed row."""
    from lsne.model import Injector, forward_logits, make_dose_delta, next_token_stats, run_with_cache
    freq_t = bank.freq_tensor(mb.device)
    rows = []
    for v in variants:
        ids = C.ids_tensor(mb, v)
        T = int(ids.shape[1])
        with torch.no_grad():
            cache = run_with_cache(mb, ids, [site])
            base = next_token_stats(cache.logits[:, -1], int(v.target_id), freq_t)
            h_ref = cache.resid[site.key][0, T - 1].detach()
        b = {k: float(val[0]) for k, val in base.items()}
        for name in dir_names:
            vec = bank[name].vec.to(h_ref.device)
            for sign in (1, -1):
                for eps in doses:
                    eps = float(eps)
                    if eps == 0.0:
                        st, dn, err = b, 0.0, 0.0
                    else:
                        delta = make_dose_delta(h_ref, sign * vec, eps).to(h_ref.dtype)
                        inj = Injector(mb, site, delta, positions=[T - 1], mode="add")
                        with torch.no_grad(), inj:
                            logits = forward_logits(mb, ids)
                        st = {k: float(val[0]) for k, val in next_token_stats(logits[:, -1], int(v.target_id), freq_t).items()}
                        dn, err = float(delta.float().norm()), float(inj.audit.get("rel_err", float("nan")))
                    rows.append(dict(C.variant_row(v), direction=name, dir_family=C.dir_family(name), sign=sign, eps=eps,
                                     logp_0=b["logp_target"], logp_delta=st["logp_target"],
                                     B=st["logp_target"] - b["logp_target"], p_target=st["p_target"],
                                     entropy_0=b["entropy"], entropy=st["entropy"],
                                     p_freq_0=b.get("p_freq", float("nan")), p_freq=st.get("p_freq", float("nan")),
                                     delta_norm=dn, audit_rel_err=err))
    return rows


@pytest.mark.parametrize("batch_size", [5, 64, 1])
def test_grid_rows_batched_matches_sequential(mb, tok, batch_size):
    """Stage C grid: the batched forwards (one delta per row, LM head on the last position)
    give the rows of one forward per dose, in the same order (finding C1)."""
    from lsne.directions import Direction
    from lsne.experiments import stage_c_grid as SC
    vs = C.apply_variant_limit(_dev01a(tok), 3)
    site = C.dossier_site(mb)
    g = torch.Generator().manual_seed(1)
    names = ["ent", "lang_dom", "rand_00"]
    bank = C.DirectionBank(site=site, dirs={n: Direction(n, torch.randn(mb.d_model, generator=g), site) for n in names},
                           freq_ids=[11, 13, 17, 19, 23])
    doses = (0.0, 0.01, 0.1, 0.5)
    new = pd.DataFrame(SC.grid_rows(mb, vs, bank, site, doses, names, enabled=False, batch_size=batch_size),
                       columns=SC.GRID_COLS)
    old = pd.DataFrame(_sequential_grid_rows(mb, vs, bank, site, doses, names), columns=SC.GRID_COLS)
    assert new.shape == old.shape == (len(vs) * len(names) * 2 * len(doses), len(SC.GRID_COLS))
    num = [c for c in SC.GRID_COLS if new[c].dtype.kind == "f"]
    rest = [c for c in SC.GRID_COLS if c not in num]
    assert new[rest].equals(old[rest])
    assert np.allclose(new[num].to_numpy(), old[num].to_numpy(), rtol=1e-4, atol=1e-5, equal_nan=True)
    assert (new.loc[new["eps"] == 0, ["B", "delta_norm", "audit_rel_err"]].to_numpy() == 0).all()
    dosed = new[new["eps"] > 0]
    assert (dosed["audit_rel_err"] < 1e-4).all() and (dosed["delta_norm"] > 0).all()
    assert np.abs(dosed["B"].to_numpy()).max() > 1e-4          # the doses do something


def test_resume_checkpoints_chunks_and_signature(tmp_path):
    """C2: chunk results are checkpointed, a rerun with the same signature loads them instead
    of recomputing, another signature discards them, clear() removes the directory."""
    calls = []

    def fn(xs):
        calls.append(list(xs))
        if xs and xs[-1] == "boom":
            raise RuntimeError("boom")
        return (pd.DataFrame(dict(x=list(xs), n=[len(x) for x in xs])), pd.DataFrame(dict(y=[len(xs)])))

    items = ["a", "bb", "ccc", "dddd", "boom"]
    r = C.Resume(tmp_path, "sig1")
    with pytest.raises(RuntimeError, match="boom"):
        r.chunked("t", items, fn, chunk=2)
    assert calls == [["a", "bb"], ["ccc", "dddd"], ["boom"]]
    assert sorted(p.name for p in (tmp_path / "_resume").iterdir()) == ["state.json", "t_00000.pkl", "t_00001.pkl"]
    calls.clear()
    items[-1] = "e"                                  # the failure is gone; the signature is unchanged
    a, b = C.Resume(tmp_path, "sig1").chunked("t", items, fn, chunk=2)
    assert calls == [["e"]]                          # only the missing chunk is computed
    assert a["x"].tolist() == items and a.index.tolist() == list(range(5)) and b["y"].tolist() == [2, 2, 1]
    calls.clear()
    r2 = C.Resume(tmp_path, "sig2")                  # other inputs: everything is recomputed
    assert r2.chunked("t", items, fn, chunk=2)[0]["x"].tolist() == items and len(calls) == 3 and not r2.loaded
    assert r2.step("s", lambda: {"k": pd.DataFrame(dict(z=[1]))})["k"]["z"].tolist() == [1]
    assert C.Resume(tmp_path, "sig2").step("s", lambda: 1 / 0)["k"]["z"].tolist() == [1]
    r2.clear()
    assert not (tmp_path / "_resume").exists()
    off = C.Resume(tmp_path, "sig3", enabled=False)
    assert off.chunked("t", [], lambda xs: pd.DataFrame(dict(x=xs))).empty and not (tmp_path / "_resume").exists()


def test_limited_core_keeps_pairs_and_originals(tok):
    """F2: the limited generation core is a seeded subset of formulations with all their core
    conditions; originals stay; None keeps everything."""
    import types
    vs = []
    for f in range(30):
        for K, Cc in C.CORE_CONDITIONS:
            vs.append(types.SimpleNamespace(tag=f"F{f:02d}A:K{K}:{Cc}", family_id=f"F{f:02d}", formulation="A",
                                            K=K, C=Cc, kind="standard"))
    vs += [types.SimpleNamespace(tag=f"orig:K{K}", family_id="ORIG", formulation="", K=K, C="orig", kind="original")
           for K in (0, 9)]
    assert C.limited_core(vs, None, 0) == vs
    sub = C.limited_core(vs, 4, seed=1)
    assert sub == C.limited_core(vs, 4, seed=1) and sub != C.limited_core(vs, 4, seed=2)
    std = [v for v in sub if v.kind == "standard"]
    assert len(std) == 4 * len(C.CORE_CONDITIONS) and [v for v in sub if v.kind == "original"] == vs[-2:]
    fams = {v.family_id for v in std}
    assert all(sum(1 for v in std if v.family_id == f) == len(C.CORE_CONDITIONS) for f in fams)
    assert [v for v in vs if v in sub] == sub                                    # order preserved
    assert C.Limits().gen_max_core is None and C.Limits().as_limited().gen_max_core == C.LIMITED_GEN_MAX_CORE
    assert C.Limits(gen_max_core=7).as_limited().gen_max_core == 7


def test_unigram_counts_strict_on_full_runs(mb, tok, monkeypatch):
    """B-2: a failed Pile fetch falls back to stimulus-bank counts only in smoke runs; a full
    run raises (and direction_bank does not swallow it)."""
    import lsne.pile_fetch as PF

    def boom(*a, **k):
        raise RuntimeError("HTTP 503")

    monkeypatch.delenv("LSNE_OFFLINE", raising=False)
    monkeypatch.setattr(PF, "sample_unigram_counts", boom)
    vs = C.apply_variant_limit(_dev01a(tok), 3)
    with pytest.warns(UserWarning, match="stimulus-bank"):
        counts, info = C.unigram_counts(mb, 5, 0, vs)
    assert info["source"] == "stimulus_bank" and counts.shape == (mb.vocab,)
    with pytest.raises(C.UnigramFetchError, match="LSNE_OFFLINE=1"):
        C.unigram_counts(mb, 5, 0, vs, strict=True)
    cfg = RunConfig(model=mb.spec, families=["DEV01"], n_random_dirs=1, seed=1)
    with pytest.raises(C.UnigramFetchError):
        C.direction_bank(cfg, mb, vs, C.dossier_site(mb), limits=C.Limits(n_unigram=5))
    monkeypatch.setenv("LSNE_OFFLINE", "1")                                     # explicit opt-in
    assert C.unigram_counts(mb, 5, 0, vs, strict=True)[1]["source"] == "stimulus_bank"


def test_get_bank_from_disk_restores_the_sae(mb, tok, tmp_path, monkeypatch):
    """IJ-1: a bank reloaded from stage B's directory gets its SAE back (published SAE
    reloaded from the hub; here a synthetic one), so stages H-J see it."""
    from lsne import sae as SAE
    from lsne.io import write_manifest
    vs = C.apply_variant_limit(_dev01a(tok), 3)
    cfg = RunConfig(model=mb.spec, families=["DEV01"], n_random_dirs=2, seed=1, out_root=str(tmp_path))
    limits = C.Limits.smoke(use_sae=True, n_sae_features=2, synthetic_sae_if_missing=False)
    site = C.dossier_site(mb)
    sae = C.synthetic_sae(mb, n_latents=32)
    monkeypatch.setenv("LSNE_OFFLINE", "1")
    bank = C.direction_bank(cfg, mb, vs, site, limits=limits, sae=sae)
    bank.meta["sae"].update(repo="org/sae-repo", hookpoint=f"layers.{site.layer}")
    rd = tmp_path / "B" / f"{mb.spec.name.split('/')[-1]}@{mb.spec.revision}"
    C.save_bank(bank, rd)
    write_manifest(rd, stage="B", status="ok", bank_signature=C.bank_signature(vs, site, cfg, limits, None))
    loads = []
    monkeypatch.setattr(SAE.TopKSAE, "load_from_hub",
                        classmethod(lambda cls, repo, hp, device="cpu": loads.append((repo, hp)) or sae))
    back = C.get_bank(cfg, mb, vs, site, {}, limits)
    assert back.meta["source"].startswith("disk:") and loads == [("org/sae-repo", f"layers.{site.layer}")]
    assert back.sae is sae and C.get_sae(mb, back, limits, {})[1] == f"hub:org/sae-repo/layers.{site.layer}"
    # a hub failure drops only the SAE (warning), the stored bank is still used
    monkeypatch.setattr(SAE.TopKSAE, "load_from_hub", classmethod(lambda cls, *a, **k: 1 / 0))
    with pytest.warns(UserWarning, match="could not be reloaded"):
        back = C.get_bank(cfg, mb, vs, site, {}, limits)
    assert back.sae is None and back.meta["source"].startswith("disk:") and C.get_sae(mb, back, limits, {})[1] == "none"


def _boom(msg):
    def f(*a, **k):
        raise RuntimeError(msg)
    return f


def _assert_tables_close(a: dict, b: dict):
    assert sorted(a) == sorted(b)
    for name in a:
        x, y = a[name].reset_index(drop=True), b[name].reset_index(drop=True)
        assert list(x.columns) == list(y.columns) and x.shape == y.shape, name
        num = [c for c in x.columns if x[c].dtype.kind == "f"]
        rest = [c for c in x.columns if c not in num]
        assert x[rest].equals(y[rest]), name
        assert np.allclose(x[num].to_numpy(dtype=float), y[num].to_numpy(dtype=float), rtol=1e-6, atol=1e-8,
                           equal_nan=True), name


def test_stage_c_resumes_after_a_late_failure(mb, tmp_path, monkeypatch):
    """C2: a failure after the grid (here in grad x act) keeps the grid and attribution
    checkpoints; the rerun loads them instead of recomputing and gives the tables of an
    uninterrupted run; the checkpoints are gone afterwards."""
    from lsne.experiments import stage_c_grid as SC
    monkeypatch.setenv("LSNE_OFFLINE", "1")
    cfg = C.smoke_config(RunConfig(model=mb.spec, families=["DEV01"], out_root=str(tmp_path)))
    limits = C.Limits.smoke(progress=False)
    ref = SC.run(cfg, mb, tmp_path / "fresh", ctx={}, limits=limits, resume=False)
    assert not (tmp_path / "fresh" / "_resume").exists()
    rd = tmp_path / "C"
    real = dict(grid_rows=SC.grid_rows, attribution_table=SC.attribution_table, grad_x_act_tables=SC.grad_x_act_tables)
    monkeypatch.setattr(SC, "grad_x_act_tables", _boom("late failure"))
    with pytest.raises(RuntimeError, match="late failure"):
        SC.run(cfg, mb, rd, ctx={}, limits=limits)
    ck = sorted(p.name for p in (rd / "_resume").iterdir())
    assert "grid_00000.pkl" in ck and "attribution_00000.pkl" in ck and not any(n.startswith("grad_x_act") for n in ck)
    monkeypatch.setattr(SC, "grad_x_act_tables", real["grad_x_act_tables"])
    monkeypatch.setattr(SC, "grid_rows", _boom("grid recomputed"))
    monkeypatch.setattr(SC, "attribution_table", _boom("attribution recomputed"))
    out = SC.run(cfg, mb, rd, ctx={}, limits=limits)
    _assert_tables_close(out, ref)
    m = json.loads((rd / "manifest.json").read_text())
    assert m["status"] == "ok" and m["resumed_checkpoints"] == len(ck) - 1 and m["unigram"]["source"] == "stimulus_bank"
    assert not (rd / "_resume").exists()
    # other inputs (another dose grid) do not reuse the old checkpoints
    monkeypatch.setattr(SC, "grad_x_act_tables", _boom("late failure"))
    monkeypatch.setattr(SC, "grid_rows", real["grid_rows"])
    monkeypatch.setattr(SC, "attribution_table", real["attribution_table"])
    with pytest.raises(RuntimeError):
        SC.run(cfg, mb, rd, ctx={}, limits=limits)
    from dataclasses import replace
    cfg2 = replace(cfg, doses=(0.0, 0.05))
    monkeypatch.setattr(SC, "grad_x_act_tables", real["grad_x_act_tables"])
    out2 = SC.run(cfg2, mb, rd, ctx={}, limits=limits)
    assert sorted(out2["grid"]["eps"].unique()) == [0.0, 0.05]
    assert json.loads((rd / "manifest.json").read_text())["resumed_checkpoints"] == 0


def test_stage_d_resumes_after_a_late_failure(mb, tmp_path, monkeypatch):
    """C2 for stage D: D1 and D2/D9 survive a failure in the generation diagnostics."""
    from lsne.experiments import stage_d_noneq as SD
    monkeypatch.setenv("LSNE_OFFLINE", "1")
    cfg = C.smoke_config(RunConfig(model=mb.spec, families=["DEV01"], out_root=str(tmp_path)))
    limits = C.Limits.smoke(progress=False)
    rd = tmp_path / "D"
    real_gen = SD.generation_tables
    monkeypatch.setattr(SD, "generation_tables", _boom("generation failed"))
    with pytest.raises(RuntimeError, match="generation failed"):
        SD.run(cfg, mb, rd, ctx={}, limits=limits)
    ck = sorted(p.name for p in (rd / "_resume").iterdir())
    assert "damage_00000.pkl" in ck and "lens_00000.pkl" in ck
    monkeypatch.setattr(SD, "generation_tables", real_gen)
    monkeypatch.setattr(SD, "damage_tables", _boom("damage recomputed"))
    monkeypatch.setattr(SD, "lens_tables", _boom("lens recomputed"))
    out = SD.run(cfg, mb, rd, ctx={}, limits=limits)
    assert len(out["damage"]) and len(out["lens"]) and len(out["hysteresis"])
    m = json.loads((rd / "manifest.json").read_text())
    assert m["status"] == "ok" and m["resumed_checkpoints"] == 2 and m["gen_max_core"] is None
    assert m["n_core"] == m["n_core_all"] == len(m["core_tags"]) and m["unigram"]["source"] == "stimulus_bank"
    assert not (rd / "_resume").exists()


def test_plan_triples_partition():
    pairs = [("m", "r1"), ("m", "r2"), ("n", "r1")]
    fams = ["F01", "F02", "F03", "DEV01"]
    full = plan_triples(pairs, fams, 0, 1)
    assert len(full) == 12 and full["mine"].all() and full["index"].tolist() == list(range(12))
    parts = [plan_triples(pairs, fams, i, 5) for i in range(5)]
    mine = [set(map(tuple, p[p["mine"]][["model", "revision", "family"]].to_numpy())) for p in parts]
    assert sum(len(m) for m in mine) == 12 and set.union(*mine) == set(map(tuple, full[["model", "revision", "family"]].to_numpy()))
    assert all(not (a & b) for i, a in enumerate(mine) for b in mine[i + 1:])
    # deterministic: the assignment is a function of the sorted triples only
    assert plan_triples(pairs[::-1], fams[::-1], 2, 5).equals(parts[2])


def test_parse_pairs_and_stages():
    cfg = RunConfig()
    assert parse_pairs(None, cfg) == [("EleutherAI/pythia-160m", "step98000"), ("EleutherAI/pythia-160m", "step99000")]
    assert parse_pairs("self", cfg) == [("EleutherAI/pythia-160m", "step98000")]
    assert len(parse_pairs("trace", cfg)) == 27
    assert parse_pairs("EleutherAI/pythia-1b@step143000, step5000", cfg) == [("EleutherAI/pythia-1b", "step143000"),
                                                                             ("EleutherAI/pythia-160m", "step5000")]
    assert parse_stages("all") == list(STAGE_ORDER) and parse_stages("b, d") == ["B", "D"] and parse_stages("g,L") == ["G", "L"]
    with pytest.raises(SystemExit):
        parse_stages("Z")
    assert set(STAGES) == set(STAGE_ORDER) == set("ABCDEFGHIJKL") and all(hasattr(m, "run") for m in STAGES.values())
    text = duration_summary({"A": dict(seconds=1.5, status="ok"), "G": dict(seconds=2.0, status="error")})
    assert "total   3.5 s" in text and "G" in text and "error" in text


def test_direction_bank_with_synthetic_sae_and_roundtrip(mb, tok, tmp_path):
    torch.manual_seed(0)
    vs = C.apply_variant_limit(_dev01a(tok), 3)
    cfg = RunConfig(model=mb.spec, families=["DEV01"], n_random_dirs=3, seed=1)
    limits = C.Limits.smoke(n_sae_features=2)
    d, F = mb.d_model, 64
    g = torch.Generator().manual_seed(0)
    sae = TopKSAE.from_tensors(torch.randn(F, d, generator=g), torch.zeros(F), torch.randn(F, d, generator=g),
                               torch.zeros(d), k=8)
    site = HookSite(mb.n_layers - 1, "block_input")
    import os
    os.environ["LSNE_OFFLINE"] = "1"       # freq axis from the stimulus bank: no network in this test
    try:
        bank = C.direction_bank(cfg, mb, vs, site, original=None, limits=limits, sae=sae)
    finally:
        os.environ.pop("LSNE_OFFLINE", None)
    names = set(bank.keys())
    assert {"ent", "lang_dom", "lang_probe", "freq", "rand_00", "rand_01", "rand_02", "sae_D0", "sae_D1", "sae_E0", "sae_E1"} == names
    assert bank["ent"].meta["source"] == "DEV01A:K0:C1" and bank["ent"].meta["position"] == vs[0].n_tokens - 1
    assert bank["lang_probe"].meta["train_acc"] > 0.9
    assert bank.meta["unigram"]["source"] == "stimulus_bank" and len(bank.freq_ids) == 100
    assert bank.meta["sae"]["k"] == 8 and "fvu" in bank.meta["sae"] and bank.sae_table is not None
    assert bank.names(("rand",)) == ["rand_00", "rand_01", "rand_02"] and bank.names(("sae",)) == ["sae_D0", "sae_D1", "sae_E0", "sae_E1"]
    assert C.dir_family("sae_E1") == "sae" and C.dir_family("rand_07") == "rand" and C.dir_family("ent") == "ent"
    # persistence round trip
    p = C.save_bank(bank, tmp_path)
    assert p.exists() and (tmp_path / "directions_meta.json").exists()
    back = C.load_bank(tmp_path)
    assert set(back.keys()) == names and back.site == site and back.freq_ids == bank.freq_ids
    for n in names:
        assert torch.allclose(back[n].vec, bank[n].vec)
    assert back["sae_D0"].meta["feature"] == bank["sae_D0"].meta["feature"]
    assert C.load_bank(tmp_path / "nowhere") is None


def test_stage_error_is_recorded_and_run_continues(mb, tmp_path, monkeypatch):
    """A failing stage writes error.txt + a manifest with status='error'; later stages still run."""
    from lsne import cli

    class Boom:
        @staticmethod
        def run(cfg, mb, run_dir, **kw):
            raise RuntimeError("boom")

    monkeypatch.setitem(cli.STAGES, "A", Boom)
    monkeypatch.setenv("LSNE_OFFLINE", "1")
    cfg = C.smoke_config(RunConfig(model=mb.spec, families=["DEV01"], out_root=str(tmp_path)))
    report = cli.run_stages(cfg, C.Limits.smoke(progress=False), ["A", "B"], mb=mb)
    assert report["A"]["status"] == "error" and "boom" in report["A"]["error"]
    a_dir = tmp_path / "A" / "pythia-14m@step98000"
    assert (a_dir / "error.txt").exists() and "RuntimeError: boom" in (a_dir / "error.txt").read_text()
    assert json.loads((a_dir / "manifest.json").read_text())["status"] == "error"
    assert report["B"]["status"] == "ok" and (tmp_path / "B" / "pythia-14m@step98000" / "baseline.csv").exists()


def test_marker_lexicon_and_sites(mb, tok):
    lex = C.marker_lexicon(tok)
    assert len(lex["E"]) == 6 and len(lex["D"]) == 6 and not set(lex["E"]) & set(lex["D"])
    assert C.dossier_site(mb) == HookSite(mb.n_layers - 1, "block_input")
    assert C.depth_site(mb) == HookSite(mb.n_layers // 3, "block_input")


# --------------------------------------------------------------------------------------
# stages G-L: shared helpers
# --------------------------------------------------------------------------------------
def test_limits_full_run_is_not_smoke():
    """Regression: a dataclass field named like the ``smoke`` classmethod was shadowed by
    the bound method and evaluated truthy in every full run (trimming the original-2143
    variants and collapsing the scale sweep of stage F to one model)."""
    from dataclasses import asdict
    full = C.Limits()
    assert full.is_smoke is False and asdict(full)["is_smoke"] is False
    assert not any(callable(v) for v in asdict(full).values())
    assert C.Limits.smoke().is_smoke is True


def test_limits_smoke_preset_covers_stages_g_to_l():
    full, smoke = C.Limits(), C.Limits.smoke()
    assert full.g_k_levels == (0, 2, 4, 8) and full.j_n_pile == 2000 and full.h_n_steps == 100 and not full.synthetic_sae_if_missing
    assert smoke.is_smoke and smoke.g_k_levels == (0, 8) and smoke.g_max_masks == 3 and smoke.j_n_pile == 3 and smoke.synthetic_sae_if_missing
    assert set(C.SMOKE_GL) <= {f.name for f in C.Limits.__dataclass_fields__.values()}
    assert C.Limits.smoke(h_n_steps=2).h_n_steps == 2 and C.Limits.smoke().n_unigram == 5


def test_family_domain_groups():
    assert [C.family_domain(f) for f in ("F13", "F16", "F17", "F18", "F19", "F20")] == \
        ["culture_politics", "culture_politics", "law", "law", "protein_misfolding", "protein_misfolding"]
    assert C.family_domain("F01") == C.family_domain("F12") == C.family_domain("DEV01") == C.family_domain("orig2143") == "everyday"
    assert sum(len(v) for v in C.DOMAIN_GROUPS.values()) == 8
    assert C.prereg_layer_grid(12) == (2, 5, 8, 11) and C.prereg_layer_grid(6) == (1, 2, 4, 5)


def test_realization_clouds(tok):
    from lsne.experiments.stage_g_geometry import realization_clouds, select_masks
    from lsne.geometry import count_switches
    m8 = select_masks(8, 10, None, 0)
    assert len(m8) == 8 and all(count_switches(m) == 8 and m[0] == "E" and m[-1] == "E" and m[-2] == "D" for m in m8)
    assert len(select_masks(2, 10, None, 0)) == 8 and len(select_masks(4, 10, None, 0)) > 8
    sub = select_masks(4, 10, 3, 0)
    assert len(sub) == 3 and sub == select_masks(4, 10, 3, 0) and sub != select_masks(4, 10, 3, 1) and set(sub) <= set(select_masks(4, 10, None, 0))
    fams = [f for f in stimuli.load_families() if f.family_id == "DEV01"]
    groups, mono_D, clouds = realization_clouds(fams, tok, (0, 2), ("C1", "C3"), 2, 0)
    n_forms = len(fams)
    assert set(groups) == {("DEV01", 0, "C1"), ("DEV01", 2, "C1"), ("DEV01", 2, "C3")}
    assert len(groups[("DEV01", 0, "C1")]) == n_forms and len(groups[("DEV01", 2, "C1")]) == 2 * n_forms
    assert set(mono_D) == {("DEV01", f.formulation) for f in fams}
    for (fid, form), v in mono_D.items():
        assert set(v.mask) == {"D"} and v.annot["n_tokens_D"] > 0 and v.target_id == groups[("DEV01", 0, "C1")][0].target_id
    assert not clouds["excluded"].any() and len(clouds) == n_forms * (1 + 2 + 2)
    for v in groups[("DEV01", 2, "C3")]:
        assert v.K == 2 and v.C == "C3" and count_switches(v.mask) == 2


def test_substages_capture_errors_and_defaults(tmp_path):
    sub = C.Substages("X", verbose=False)
    assert sub.run("ok", lambda: 3) == 3 and sub.ok
    with pytest.warns(UserWarning, match="substage 'bad' failed"):
        out = sub.run("bad", lambda: 1 / 0, default=lambda: "dflt")
    assert out == "dflt" and not sub.ok and "ZeroDivisionError" in sub.errors["bad"]
    assert sub.run("bad2", lambda: [][1], default=None) is None
    rep = sub.report()
    assert rep["n_failed"] == 2 and set(rep["seconds"]) == {"ok", "bad", "bad2"}
    p = sub.write_tracebacks(tmp_path)
    assert p is not None and "=== bad ===" in p.read_text() and "ZeroDivisionError" in p.read_text()
    assert C.Substages("Y", verbose=False).write_tracebacks(tmp_path / "none") is None


def test_synthetic_sae_and_get_sae(mb):
    sae = C.synthetic_sae(mb, n_latents=32, k=4, seed=1)
    assert sae.d_in == mb.d_model and sae.num_latents == 32 and sae.k == 4
    assert torch.equal(sae.W_enc, C.synthetic_sae(mb, n_latents=32, k=4, seed=1).W_enc)
    bank = C.DirectionBank(site=HookSite(0, "block_input"))
    assert C.get_sae(mb, bank, C.Limits(), {}) == (None, "none")
    s2, src = C.get_sae(mb, bank, C.Limits.smoke(), {})
    assert src == "synthetic_random" and s2.num_latents == 64
    assert C.get_sae(mb, bank, C.Limits(), {"sae": sae}) == (sae, "given")
    bank.sae, bank.meta["sae"] = sae, dict(repo="r", hookpoint="layers.0")
    assert C.get_sae(mb, bank, C.Limits(), {}) == (sae, "hub:r/layers.0")


def test_family_text_pairs_and_stage_tables(tmp_path):
    fams = [f for f in stimuli.load_families() if f.family_id in ("DEV01", "DEV02")]
    E, D, N = C.family_text_pairs(fams, 15)
    assert len(E) == len(D) == 15 and len(N) <= 15 and all(e.endswith(".") and d.endswith(".") for e, d in zip(E, D))
    assert E[0].startswith(fams[0].units[0].E["subj"]) and D[0].startswith(fams[0].units[0].D["subj"]) and len(set(E)) == 15
    cfg = RunConfig(out_root=str(tmp_path))
    assert C.stage_run_dir(cfg, "D") is None and C.load_stage_table(cfg, "D", "lens") is None
    row = C.boot_row(pd.DataFrame(dict(family=["a", "a", "b"], x=[1.0, 3.0, 5.0])), "x", 50, 0, quantity="x")
    assert row["n_clusters"] == 2 and row["n_rows"] == 3 and row["ci_low"] <= row["mean"] <= row["ci_high"] and row["quantity"] == "x"


# --------------------------------------------------------------------------------------
# stage C: pre-registered layer grid
# --------------------------------------------------------------------------------------
def test_attribution_layer_grid():
    from lsne.config import LAYER_GRID_160M
    from lsne.experiments.stage_c_grid import attribution_layer_grid

    assert attribution_layer_grid(12) == tuple(LAYER_GRID_160M) == (2, 5, 8, 11)
    assert attribution_layer_grid(6) == (1, 2, 4, 5)        # pythia-14m / 70m
    assert attribution_layer_grid(16) == (3, 7, 11, 15)     # pythia-1b
    assert attribution_layer_grid(24) == (5, 11, 17, 23)    # pythia-410m / 1.4b
    assert attribution_layer_grid(4) == (0, 1, 2, 3) and attribution_layer_grid(0) == ()
    for n in range(5, 40):
        g = attribution_layer_grid(n)
        assert len(g) == 4 and 0 not in g and g[-1] == n - 1 and list(g) == sorted(set(g))


# --------------------------------------------------------------------------------------
# stage E: response basis, float64 measurement
# --------------------------------------------------------------------------------------
def test_response_basis_recovers_subspace():
    from lsne.experiments.stage_e_tn import response_basis

    g = torch.Generator().manual_seed(0)
    d, k, N = 16, 3, 50
    B = torch.randn(k, d, generator=g, dtype=torch.float64)
    rows = (torch.randn(N, k, generator=g, dtype=torch.float64) @ B).numpy()
    P, ev = response_basis(rows, 8)
    assert P.shape == (d, 8) and P.dtype == torch.float32 and ev.shape == (8,)
    assert np.allclose((P.T @ P).numpy(), np.eye(8), atol=1e-5)          # orthonormal incl. completion
    assert ev[k - 1] == pytest.approx(1.0, abs=1e-9) and np.all(np.diff(ev) >= -1e-12)
    proj = (rows @ P[:, :k].numpy().astype(np.float64)) @ P[:, :k].numpy().astype(np.float64).T
    assert np.allclose(proj, rows, atol=1e-5)                               # rows lie in the span
    # uncentred: the mean response stays in the span (a centred PCA would drop it)
    mean = rows.mean(0, keepdims=True)
    assert np.allclose((mean @ P[:, :k].numpy()) @ P[:, :k].numpy().T, mean, atol=1e-5)
    # deterministic, and N < r completes the basis
    P2, _ = response_basis(rows, 8)
    assert torch.equal(P, P2)
    P3, ev3 = response_basis(rows[:2], 5)
    assert P3.shape == (d, 5) and np.allclose((P3.T @ P3).numpy(), np.eye(5), atol=1e-5) and ev3[-1] == pytest.approx(1.0)


def test_float64_bundle_and_attention_floor(mb, sample_ids):
    """The float64 copy reproduces the model; GPT-NeoX's float32 softmax leaves a round-off
    floor in the even part of a +/- perturbation that the float64-softmax patch removes."""
    from lsne.model import make_dose_delta, run_with_cache
    from lsne.experiments.stage_e_tn import float64_attention, float64_bundle, measured_response
    from transformers.models.gpt_neox import modeling_gpt_neox as mg

    L = mb.n_layers
    site_from, out_site = HookSite(L // 3, "block_input"), HookSite(L - 1, "block_output")
    ids = sample_ids
    T = int(ids.shape[1])
    window, position = slice(T - 6, T), T - 4
    mb64 = float64_bundle(mb)
    assert mb64 is not None and mb64.dtype == torch.float64 and mb64.spec.dtype == "float64" and mb.dtype == torch.float32
    c32 = run_with_cache(mb, ids, [site_from, out_site])
    c64 = run_with_cache(mb64, ids, [out_site])
    assert c64.resid[out_site.key].dtype == torch.float64
    assert float((c32.logits - c64.logits).abs().max()) < 5e-2
    h_ref = c32.resid[site_from.key][0, position]
    g = torch.Generator().manual_seed(0)
    u = torch.randn(mb.d_model, generator=g)
    delta = make_dose_delta(h_ref, u / u.norm(), 1e-4)
    orig = mg.eager_attention_forward

    def even_part(bundle, ref):
        mp = measured_response(bundle, ref, ids, site_from, position, delta, out_site, window)
        mm = measured_response(bundle, ref, ids, site_from, position, -delta, out_site, window)
        assert mp.shape == (6, mb.d_model) and mp.dtype == np.float64
        return float(np.linalg.norm(mp + mm) / 2.0)

    ref64 = c64.resid[out_site.key][0]
    floor_unpatched = even_part(mb64, ref64)
    with float64_attention() as ok:
        assert ok and mg.eager_attention_forward is not orig
        c64p = run_with_cache(mb64, ids, [out_site])
        even_patched = even_part(mb64, c64p.resid[out_site.key][0])
        # float32 models are bitwise unaffected by the patch
        c32p = run_with_cache(mb, ids, [out_site])
        assert torch.equal(c32p.resid[out_site.key], c32.resid[out_site.key])
    assert mg.eager_attention_forward is orig
    # true second order at eps = 1e-4 is ~1e-8; the float32 softmax floor is ~1e-4
    assert even_patched < 1e-6 and floor_unpatched > 1e-5 and floor_unpatched > 100 * even_patched
