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
    assert {"K0_vs_K8C1", "K0_vs_K28seg"} <= set(att["pair"]) and set(att["layer"]) == {0, 2, 3, 5}
    assert set(att["class"]) == set(C.PATCH_CLASSES)
    gxa = pd.read_csv(_stage_dir(out, "C") / "grad_x_act_by_class.csv")
    assert gxa["abs_share"].between(0, 1 + 1e-9).all()
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
    assert np.isfinite(tn[["nonlinearity_last", "factorization_error_last", "linear_norm_last"]].to_numpy()).all()
    assert (tn["linear_norm_last"] > 0).all() and tn["source_captured"].between(0, 1 + 1e-6).all()
    paths = pd.read_csv(_stage_dir(out, "E") / "tn_paths.csv")
    assert {"tnco", "opt_einsum", "reference_greedy", "reference_dp"} == set(paths["backend"])
    assert (paths["n_tensors"] == 9).all() and np.isfinite(paths["cost_log10_opt_einsum"]).all()
    basis = pd.read_csv(_stage_dir(out, "E") / "tn_basis.csv")
    assert len(basis) == 16 and basis["explained_variance_cum"].is_monotonic_increasing


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
