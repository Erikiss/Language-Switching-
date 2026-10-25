"""End-to-end CPU smoke test of the experiment stages and the CLI (pythia-14m, DEV01).

``python -m lsne.cli run --smoke --stage all`` on the DEV01 family must finish in well under
four minutes and leave at least one csv table plus a manifest.json in every stage directory
(stage F re-runs B/C/D for the same model/revision in its own sub-run directories).  The
``merge`` command is exercised on the result.  A few fast unit tests cover the helpers of
``lsne.experiments.common`` and ``stage_f_scale`` (variant limiting, shard partition, pair
parsing, direction bank with a synthetic SAE, bank persistence).
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lsne import stimuli
from lsne.cli import main, merge_runs, parse_stages
from lsne.config import HookSite, RunConfig
from lsne.experiments import STAGE_ORDER, STAGES
from lsne.experiments import common as C
from lsne.experiments.stage_f_scale import parse_pairs, plan_triples
from lsne.sae import TopKSAE

REPO = Path(__file__).resolve().parent.parent
SMOKE_BUDGET_S = 240.0


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
    t0 = time.perf_counter()
    rc = main(["run", "--stage", "all", "--model", "EleutherAI/pythia-14m", "--revision", "step98000",
               "--dtype", "float32", "--device", "cpu", "--families", "DEV01", "--out", str(out), "--smoke",
               "--no-progress"])
    return dict(out=out, rc=rc, seconds=time.perf_counter() - t0)


def test_smoke_all_stages_produce_tables(smoke_run):
    out, rc = smoke_run["out"], smoke_run["rc"]
    assert rc == 0
    assert smoke_run["seconds"] < SMOKE_BUDGET_S, f"smoke run took {smoke_run['seconds']:.0f} s"
    for stage in STAGE_ORDER:
        _assert_run_dir(_stage_dir(out, stage))
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
    assert set(att["class"]) == set(C.PATCH_CLASSES)
    cman = json.loads((_stage_dir(out, "C") / "manifest.json").read_text(encoding="utf-8"))
    assert cman["layer_grid"] == [1, 2, 4, 5]
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
    phase = pd.read_csv(_stage_dir(out, "D") / "damage_phase.csv")
    assert (phase[phase["dir_family"] == "rand"]["n"] == 2).all()
    lens = pd.read_csv(_stage_dir(out, "D") / "lens.csv")
    assert set(lens["layer"]) == set(range(7)) and np.allclose(lens[lens["layer"] == 6]["kl_last"], 0.0, atol=1e-5)
    summ = pd.read_csv(_stage_dir(out, "D") / "lens_summary.csv")
    assert summ["nmi_last"].between(0, 1).all()
    ca = pd.read_csv(_stage_dir(out, "D") / "ca.csv")
    assert np.isfinite(ca[["spatial_entropy", "temporal_entropy", "activity", "lz76"]].to_numpy()).all()
    hy = pd.read_csv(_stage_dir(out, "D") / "hysteresis.csv")
    assert set(hy["direction"]) == {"ent", "lang_dom"} and np.isfinite(hy[["area", "null_area"]].to_numpy()).all()
    assert set(hy["K"]) >= {0, 8, 28}
    curves = pd.read_csv(_stage_dir(out, "D") / "hysteresis_curves.csv")
    assert set(curves["branch"]) == {"up", "down", "restart_a", "restart_b"}
    rx = pd.read_csv(_stage_dir(out, "D") / "relaxation.csv")
    assert (rx["eps"] == 0.5).all() and (rx["n_new"] == 8).all()
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
    assert len(manifests) >= 9 and all("run_id" in m for m in manifests)
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
    assert len(checks) == 24 and not checks["excluded"].any()
    cfg2 = RunConfig(model=mb.spec, families=["DEV01B", "DEV02"], limit_families=1)
    fams = C.select_families(cfg2)
    assert [f.tag for f in fams] == ["DEV01B"]


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
    assert parse_stages("all") == list(STAGE_ORDER) and parse_stages("b, d") == ["B", "D"]
    with pytest.raises(SystemExit):
        parse_stages("Z")
    assert set(STAGES) == set(STAGE_ORDER) and all(hasattr(m, "run") for m in STAGES.values())


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
