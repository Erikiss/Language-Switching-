"""Stage-D control run of the injection-site language (PREREGISTRATION section 6, amendment 11;
CPU, pythia-14m, family F01).

* sites: ``u9`` (last token of unit 9) for every control variant, ``u10`` (last token of unit 10)
  for K = 0 and C1, both at the depth site and defined by the unit, never by the language;
* ``u9D`` (control 3): K = 2, C ``u9D``, unit mask EEEEEEEEDE at C1, checked against K = 0;
* the primary site (``u9`` at K = 0 and C1) reproduces stage D's ``depth`` rows of the main run
  bit for bit, from the main run's bank, directions, doses and batch layout;
* the guards refuse before anything is written, and never touch the main run; other torch /
  transformers versions than the main run's are refused unless allowed, and then recorded;
* a restart resumes only checkpoints of the same platform (torch, transformers, attention
  implementation) and commit (git sha; uncommitted edits and a non-git install are not detected);
* the manifest carries the fields the evaluation (``lsne.prereg_eval``, key ``D_ctl``) reads, and
  the SHA-256 of the evaluation code;
* stage D's own injections are unchanged.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import transformers

import lsne
from lsne import stimuli
from lsne.cli import main as cli_main
from lsne.cli import run_stages
from lsne.config import RunConfig
from lsne.experiments import common as C
from lsne.experiments import stage_d_control as SDC
from lsne.experiments import stage_d_noneq as SD
from lsne.io import RunDir, read_manifest, write_manifest
from lsne.model import load_model

FAMILY = "F01"
#: Directions of the fake main run in its manifest order (``ent``: its float32 norm is not exactly
#: 1, so a re-normalised reload would not replicate the main run, see ``load_main_bank``).
LATTICE = ["ent", "lang_dom", "rand_00", "rand_01"]
DOSES = [0.01, 0.1]
#: 8 (direction, dose) pairs per site in forwards of 5 rows: two chunks, as the main run's layout.
BATCH = 5
#: The D1 damage columns of stage D (the default injections must leave them as they were).
STAGE_D_COLS = list(C.ID_COLS) + [
    "inject", "site", "dir_site", "inject_layer", "inject_pos", "inject_lang", "inject_unit", "inject_token",
    "inject_klass", "direction", "dir_family", "eps", "lam", "intercept", "r2", "n_fit_layers", "D_final_tail",
    "D_final_last", "D_max", "cos_final_last", "front01_final", "front1_final", "front01_max", "front1_max",
    "n_spread01_final", "n_spread1_final", "n_after", "logp_target_ref", "logp_target_pert", "B", "delta_norm",
    "audit_rel_err"]
CONTROL_CELLS = [(0, "C1")] + [(K, c) for K in (2, 4, 8) for c in ("C1", "C2", "C3")] + [(2, "u9D")]


# --------------------------------------------------------------------------------------
# helpers and fixtures
# --------------------------------------------------------------------------------------
def _family(tag: str) -> stimuli.Family:
    return next(f for f in stimuli.load_families() if f.tag == tag)


def _snapshot(d: Path) -> dict:
    """Every file below ``d`` with its bytes (to show that nothing in a directory changed)."""
    return {str(p.relative_to(d)): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}


def _eval_code_sha256() -> str:
    """SHA-256 of ``lsne/prereg_eval.py`` of the package under test."""
    return hashlib.sha256((Path(lsne.__file__).resolve().parent / "prereg_eval.py").read_bytes()).hexdigest()


def _write_main_run(path, mb, bank, **over) -> Path:
    """A fake main stage-D run: the depth bank (``C.save_bank``) and an ok manifest with the
    inputs the control takes over (``lattice_directions``, ``doses``, ``limits.damage_batch_size``)."""
    C.save_bank(bank, path, name="directions_depth")
    meta = dict(stage="D", status="ok", cfg=RunConfig(model=mb.spec), model=mb.spec,
                limits=asdict(replace(C.Limits.smoke(), damage_batch_size=BATCH)),
                lattice_directions=LATTICE, doses=DOSES)
    meta.update(over)
    write_manifest(path, **meta)
    return Path(path)


@pytest.fixture(scope="module")
def depth_bank(mb):
    """A depth-site bank built by the stage helpers (stimulus-bank unigram counts, two rand)."""
    vs = stimuli.build_variants(_family(f"{FAMILY}A"), mb.tokenizer)[:4]
    cfg = RunConfig(model=mb.spec, families=[FAMILY], n_random_dirs=2, seed=0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LSNE_OFFLINE", "1")
        return C.direction_bank(cfg, mb, vs, C.depth_site(mb), limits=C.Limits.smoke(progress=False))


@pytest.fixture(scope="module")
def ctl_run(mb, depth_bank, tmp_path_factory):
    """The control run of F01 (formulations A and B) against a fake main run that also holds
    checkpoints of its own."""
    root = tmp_path_factory.mktemp("ctl")
    main = _write_main_run(RunDir(root / "runs", "D", mb.spec).path, mb, depth_bank)
    (main / "_resume").mkdir()
    (main / "_resume" / "state.json").write_text(json.dumps(dict(signature="main-run")), encoding="utf-8")
    before = _snapshot(main)
    cfg = RunConfig(model=mb.spec, families=[FAMILY], out_root=str(root / "runs_ctl"))
    rd = RunDir(cfg.out_root, "D", cfg.model)
    out = SD.run(cfg, mb, rd, ctx={}, limits=C.Limits.smoke(progress=False), d_control="token_language",
                 d_main_run=str(main))
    return dict(main=main, before=before, cfg=cfg, rd=rd.path, out=out, manifest=read_manifest(rd))


# --------------------------------------------------------------------------------------
# sites and variants
# --------------------------------------------------------------------------------------
def test_sites_per_cell_are_unit_defined(mb, tok):
    fam = _family(f"{FAMILY}A")
    vs = [v for v in stimuli.build_variants(fam, tok) if (v.K, v.C) in CONTROL_CELLS] + [SDC.u9d_variant(fam, tok)]
    assert sorted((v.K, v.C) for v in vs) == sorted(CONTROL_CELLS)
    dsite = C.depth_site(mb)
    for v in vs:
        inj = SDC.control_injections(mb, v)
        names = [n for n, _, _ in inj]
        assert names == (["u9", "u10"] if v.K == 0 or v.C == "C1" else ["u9"]), v.tag
        assert all(site == dsite for _, site, _ in inj)
        pos = {n: p for n, _, p in inj}
        a = v.annot
        assert a["unit"][pos["u9"]] == 9 and a["unit"][pos["u9"] + 1] == 10 and a["tokens"][pos["u9"]] == "."
        lsp = stimuli.last_switched_position(v)
        if v.C in ("C2", "C3"):           # an English-closing "." where stage D injects at the last German token
            assert a["lang"][pos["u9"]] == "E" and pos["u9"] != lsp
        else:                             # K = 0, C1 and u9D: the same "." as stage D's depth site
            assert a["lang"][pos["u9"]] == ("E" if v.K == 0 else "D") and pos["u9"] == lsp
        if v.K == 0 or v.C == "C1":
            assert [p for n, _, p in SD.injections(mb, v) if n == "depth"] == [pos["u9"]]     # the primary site
            assert pos["u10"] == v.tail_start - 1 and a["unit"][pos["u10"]] == 10
            assert a["tokens"][pos["u10"]] == "." and a["lang"][pos["u10"]] == "E"
    assert SDC.control_cells(vs) == {"u9": sorted(CONTROL_CELLS), "u10": [(0, "C1"), (2, "C1"), (4, "C1"), (8, "C1")]}


def test_u9d_variant_and_control_variants(mb, tok, monkeypatch):
    fam = _family(f"{FAMILY}A")
    v = SDC.u9d_variant(fam, tok)
    assert (v.K, v.C, v.tag) == (2, "u9D", f"{FAMILY}A:K2:u9D")
    assert v.mask == stimuli.segment_masks_from_unit_mask("EEEEEEEEDE", "C1") and stimuli.count_switches(v.mask) == 2
    assert stimuli.unit_composition(v.mask) == ["EEE"] * 8 + ["DDD", "EEE"]
    assert stimuli.check_variant(v, stimuli.reference_variant(stimuli.build_variants(fam, tok))) == []

    # the test split only (no DEV01, no formulations C / D), no seg; each formulation ends with its u9D
    variants, checks = SDC.control_variants(RunConfig(model=mb.spec, families=[FAMILY, "DEV01"]), mb, {})
    assert len(variants) == 2 * len(CONTROL_CELLS)
    assert sorted({(x.family_id, x.formulation) for x in variants}) == [(FAMILY, "A"), (FAMILY, "B")]
    for form in "AB":
        cells = [(x.K, x.C) for x in variants if x.formulation == form]
        assert sorted(cells) == sorted(CONTROL_CELLS) and cells[-1] == (2, "u9D")
    assert list(checks.columns) == list(SDC.CHECK_COLS) and len(checks) == len(variants) and not checks["excluded"].any()
    assert set(checks.loc[checks["C"] == "u9D", "tag"]) == {f"{FAMILY}A:K2:u9D", f"{FAMILY}B:K2:u9D"}

    # an excluded u9D (PREREGISTRATION section 4) is recorded, not run; so is a build failure
    real = stimuli.check_variant
    monkeypatch.setattr(stimuli, "check_variant", lambda x, ref: ["tail token ids differ"] if x.C == "u9D" else real(x, ref))
    cfg = RunConfig(model=mb.spec, families=[f"{FAMILY}A"])
    variants, checks = SDC.control_variants(cfg, mb, {})
    assert "u9D" not in {x.C for x in variants} and len(variants) == len(CONTROL_CELLS) - 1
    row = checks[checks["C"] == "u9D"].iloc[0]
    assert bool(row["excluded"]) and row["step"] == "check" and "tail token ids" in row["problem"]
    def unbuildable(*a):
        raise ValueError("no 10 units")

    monkeypatch.setattr(SDC, "u9d_variant", unbuildable)
    checks = SDC.control_variants(cfg, mb, {})[1]
    row = checks[checks["C"] == "u9D"].iloc[0]
    assert bool(row["excluded"]) and row["step"] == "build" and row["problem"] == "no 10 units"


def test_variant_limits_act_on_the_test_formulations(mb):
    """The regular builder runs on the test formulations only, so --limit-variants without --families
    keeps control variants (DEV01-DEV04 come first in the stimulus bank: a limit applied before the
    split left none) and selects the C1 variants of a main run limited alike; a run without a test
    formulation or without a variant is refused before it writes anything.  The manifest's
    design_filters name what narrows the design below the main run's selection."""
    assert SDC.control_formulations(RunConfig(model=mb.spec))[:3] == ["F01A", "F01B", "F02A"]
    variants, _ = SDC.control_variants(RunConfig(model=mb.spec, limit_variants=3), mb, {})
    assert [v.tag for v in variants] == ["F01A:K0:C1", "F01A:K8:C1", "F01A:K2:u9D"]
    with pytest.raises(SDC.ControlRunError, match="no test formulation"):
        SDC.control_variants(RunConfig(model=mb.spec, families=["DEV01", "F13A", "F01C"]), mb, {})
    with pytest.raises(SDC.ControlRunError, match="no control variant left"):
        SDC.control_variants(RunConfig(model=mb.spec, families=["F01A"], k_levels=(16,)), mb, {})
    tags = ["F01A", "F01B", "F02A", "F02B"]
    for kw, main_cfg, want in (({}, None, []), ({"families": [f"F{i:02d}" for i in range(1, 13)]}, {}, []),
                               ({"families": tags}, {"families": tags}, []), ({"families": ["F01"]}, {}, ["families"]),
                               ({"limit_families": 2}, {}, ["families"]), ({"limit_variants": 3}, {"families": tags},
                                                                            ["limit_variants"]),
                               ({"k_levels": (0, 8), "c_levels": ("C1", "C2")}, {}, ["k_levels", "c_levels"])):
        assert SDC.design_filters(RunConfig(model=mb.spec, **kw), main_cfg) == want, kw


# --------------------------------------------------------------------------------------
# the control run
# --------------------------------------------------------------------------------------
def test_control_run_tables_sites_and_manifest(ctl_run, mb, tok, depth_bank):
    out, m, rd, main = ctl_run["out"], ctl_run["manifest"], ctl_run["rd"], ctl_run["main"]
    assert set(out) == {"damage_ctl", "damage_ctl_profile", "damage_ctl_checks", "damage_ctl_contrasts"}
    assert all((rd / f"{name}.csv").exists() for name in out)
    d = out["damage_ctl"]
    cols = list(d.columns)
    k = cols.index("inject_klass")
    assert cols[:k + 1] + cols[k + 3:] == STAGE_D_COLS and cols[k + 1:k + 3] == ["inject_token_id", "site_gap"]

    # every (variant, site): all pairs of the main run in stage D's order, at the depth site
    for _, g in d.groupby(["tag", "inject"], sort=False):
        assert list(zip(g["direction"], g["eps"])) == [(n, e) for n in LATTICE for e in DOSES]
    assert set(d["site"]) == set(d["dir_site"]) == {C.depth_site(mb).key}
    cells = {inj: sorted(set(zip(g["K"].astype(int), g["C"]))) for inj, g in d.groupby("inject")}
    assert cells == {"u9": sorted(CONTROL_CELLS), "u10": [(0, "C1"), (2, "C1"), (4, "C1"), (8, "C1")]}
    assert set(zip(d["family"], d["formulation"])) == {(FAMILY, "A"), (FAMILY, "B")}

    # the site checks of amendment 6.11 (c): "." everywhere, unit 9 / 10, language, distance to the tail
    assert (d["inject_token"] == ".").all() and (d["inject_unit"] == d["inject"].map(SDC.SITE_UNITS)).all()
    assert d["inject_token_id"].nunique() == 1 and tok.decode([int(d["inject_token_id"].iloc[0])]) == "."
    german = (d["inject"] == "u9") & (((d["C"] == "C1") & (d["K"] > 0)) | (d["C"] == "u9D"))
    assert (d.loc[german, "inject_lang"] == "D").all() and (d.loc[~german, "inject_lang"] == "E").all()
    assert (d["site_gap"] == d["tail_start"] - d["inject_pos"]).all()
    assert (d.loc[d["inject"] == "u10", "site_gap"] == 1).all()
    u9 = d[d["inject"] == "u9"]
    gap0 = u9[u9["K"] == 0].groupby("formulation")["site_gap"].first()
    assert (u9["site_gap"].to_numpy() == gap0.loc[u9["formulation"]].to_numpy()).all()
    assert (d["audit_rel_err"] < C.AUDIT_MAX_REL_ERR).all() and np.isfinite(d["lam"]).all()

    prof = out["damage_ctl_profile"]
    assert len(prof) == len(d) * (mb.n_layers + 1) and set(prof["inject"]) == {"u9", "u10"}
    checks = out["damage_ctl_checks"]
    assert len(checks) == 2 * len(CONTROL_CELLS) and not checks["excluded"].any()
    con = out["damage_ctl_contrasts"]
    assert set(con["inject"]) == {"u9", "u10"} and (con["K"] > 0).all() and set(con.loc[con["inject"] == "u10", "C"]) == {"C1"}
    assert len(con) == 2 * len(DOSES) * (len(CONTROL_CELLS) - 1 + 3)

    # the manifest: the contract of the evaluation (key D_ctl, audits) and the main run's inputs
    assert m["stage"] == "D" and m["status"] == "ok" and m["control"] == "token_language" and m["amendment"] == "6.11"
    mm = read_manifest(main)
    assert m["main_run"] == dict(path=str(main.resolve()), git_sha=mm["git_sha"], timestamp=mm["timestamp"],
                                 model=mb.spec.name, revision=mb.spec.revision)
    assert m["bank"] == dict(file="directions_depth.npz", sha256=SDC.file_sha256(main / "directions_depth.npz"),
                             names=list(depth_bank.dirs))
    assert m["lattice_directions"] == LATTICE and m["doses"] == DOSES
    assert m["damage_batch_size"] == m["limits"]["damage_batch_size"] == BATCH
    assert m["sites"] == SDC.SITE_RULES and m["u9D_unit_mask"] == "EEEEEEEEDE"
    assert m["cells"] == {k: [list(c) for c in v] for k, v in cells.items()}
    assert m["n_variants"] == 2 * len(CONTROL_CELLS) and m["n_excluded"] == 0 and m["n_formulations"] == 2
    assert m["design_filters"] == ["families"]                     # F01 only: never preferred over the whole design
    assert m["audit"]["valid"] and m["audit"]["n_rows"] == len(d) and m["tables"]["damage_ctl"] == len(d)
    assert m["depth_site"] == dict(layer=C.depth_site(mb).layer, site="block_input")
    # the replication platform next to the environment's own OS string (which it must not replace)
    rp = m["replication_platform"]
    assert rp["device"] == "cpu" and rp["main_torch"] == mm["torch_version"] and rp["main_platform"] == mm["platform"]
    assert rp["torch"] == str(torch.__version__) and rp["gpu"] == "" and rp["cuda"] == str(torch.version.cuda or "")
    assert rp["transformers"] == str(transformers.__version__) and rp["main_transformers"] == mm["transformers_version"]
    assert isinstance(m["platform"], str) and m["platform"] and m["platform"] == mm["platform"]
    # same software as the main run; the evaluation code it was run with (lsne/prereg_eval.py)
    assert m["platform_mismatch"] == [] and m["eval_code_sha256"] == _eval_code_sha256()

    # its own checkpoints are gone; the main run (manifest, bank, its checkpoints) is untouched
    assert not (rd / "_resume").exists() and not (rd / "directions_depth.npz").exists()
    assert _snapshot(main) == ctl_run["before"]


def test_primary_site_replicates_stage_d(ctl_run, mb, depth_bank):
    """``u9`` at K = 0 and C1 is stage D's ``depth`` site: the main run's computation (default
    injections, the bank's vectors in memory, same directions / doses / batch layout) gives the
    same positions and bit-identical lambda."""
    variants, _ = SDC.control_variants(ctl_run["cfg"], mb, {})
    prim = [v for v in variants if v.C == "C1"]
    assert sorted({v.K for v in prim}) == [0, 2, 4, 8]
    main_dmg, _ = SD.damage_tables(mb, prim, depth_bank, LATTICE, DOSES, enabled=False, batch_size=BATCH)
    key = ["tag", "direction", "eps"]
    a = main_dmg[main_dmg["inject"] == "depth"].set_index(key).sort_index()
    d = ctl_run["out"]["damage_ctl"]
    b = d[(d["inject"] == "u9") & (d["C"] == "C1")].set_index(key).sort_index()
    assert a.index.equals(b.index) and len(a) == len(prim) * len(LATTICE) * len(DOSES)
    assert (a["inject_pos"] == b["inject_pos"]).all()
    np.testing.assert_allclose(b["lam"].to_numpy(), a["lam"].to_numpy(), rtol=0, atol=1e-9)
    # the bank is the main run's, vector for vector (no refit, no re-normalisation)
    loaded = SDC.load_main_bank(ctl_run["main"])
    assert list(loaded.dirs) == list(depth_bank.dirs) and loaded.site == depth_bank.site
    assert all(np.array_equal(loaded[n].vec.numpy(), depth_bank[n].vec.numpy()) for n in depth_bank.dirs)


def test_guards_refuse_before_writing(mb, depth_bank, tmp_path):
    main = _write_main_run(RunDir(tmp_path / "runs", "D", mb.spec).path, mb, depth_bank)
    before = _snapshot(main)
    cfg = RunConfig(model=mb.spec, families=[f"{FAMILY}A"], out_root=str(tmp_path / "runs_ctl"))
    rd = RunDir(cfg.out_root, "D", cfg.model).path
    kw = dict(ctx={}, limits=C.Limits.smoke(progress=False), d_control="token_language", d_main_run=str(main))

    def refused(run_dir, match, **over):
        with pytest.raises(SDC.ControlRunError, match=match):
            SD.run(cfg, mb, run_dir, **dict(kw, **over))

    with pytest.raises(ValueError, match="unknown stage-D control"):
        SD.run(cfg, mb, rd, **dict(kw, d_control="token_lang"))
    refused(rd, "needs the main stage-D run directory", d_main_run=None)
    # the run directory: the main run's (also through a link), another stage-D run (any status), or
    # an interrupted main run (its bank, no manifest yet)
    refused(main, "is the main stage-D run directory")
    link = tmp_path / "link"
    link.symlink_to(main, target_is_directory=True)
    refused(link, "is the main stage-D run directory")
    refused(RunDir(main, "D", mb.spec, mkdir=False).path, "lies inside the main stage-D run directory")     # --out <main>
    # the root (--out): a runs folder (here of another account; also through a link or '..'), or one
    # that holds the main run
    (tmp_path / "LSNE_C" / "runs").mkdir(parents=True)
    alias = tmp_path / "ctl_alias"
    alias.symlink_to(tmp_path / "LSNE_C" / "runs", target_is_directory=True)
    for root in (tmp_path / "LSNE_B" / "runs", tmp_path, alias, tmp_path / "LSNE_C" / "runs" / "x" / ".."):
        with pytest.raises(SDC.ControlRunError, match="is a runs folder, lies inside the root of the main"):
            SD.run(replace(cfg, out_root=str(root)), mb, RunDir(root, "D", mb.spec, mkdir=False).path, **kw)
        assert not (root / "D").exists()
    assert not any((tmp_path / "LSNE_C" / "runs").iterdir())
    # a root of its own, also through a link, is accepted
    (tmp_path / "LSNE_C" / "runs_ctl").mkdir()
    (tmp_path / "ctl_ok").symlink_to(tmp_path / "LSNE_C" / "runs_ctl", target_is_directory=True)
    for root in (tmp_path / "LSNE_C" / "runs_ctl", tmp_path / "ctl_ok", tmp_path / "LSNE_C" / "runs" / ".." / "runs_ctl"):
        SDC.check_run_dir(RunDir(root, "D", mb.spec, mkdir=False).path, main, root)
    for status in ("ok", "error"):
        other = tmp_path / f"other_{status}"
        write_manifest(other, stage="D", status=status)
        refused(other, "not a token_language control run")
    interrupted = tmp_path / "interrupted"
    C.save_bank(depth_bank, interrupted, name="directions_depth")
    refused(interrupted, "directions_depth.npz")
    # the main run: missing, not ok, a control run, another model revision / dtype, no bank
    refused(rd, "no manifest.json", d_main_run=str(tmp_path / "nowhere"))
    bad = {"error": dict(status="error"), "ctl": dict(control="token_language"),
           "rev": dict(model=replace(mb.spec, revision="step99000")), "dtype": dict(model=replace(mb.spec, dtype="bfloat16"))}
    match = {"error": "no ok stage-D run", "ctl": "control run, not the main", "rev": "model revision", "dtype": "model dtype"}
    for name, over in bad.items():
        refused(rd, match[name], d_main_run=str(_write_main_run(tmp_path / f"main_{name}", mb, depth_bank, **over)))
    nobank = tmp_path / "main_nobank"
    shutil.copytree(main, nobank)
    (nobank / "directions_depth_meta.json").unlink()
    refused(rd, "no depth bank", d_main_run=str(nobank))
    assert _snapshot(main) == before and not any(rd.iterdir())


# --------------------------------------------------------------------------------------
# platform, provenance and resume
# --------------------------------------------------------------------------------------
def test_platform_mismatches_and_their_guard(monkeypatch):
    """torch / transformers of this run against the main run's manifest: one readable string per
    difference, nothing for equal versions or for a field the main manifest lacks; refused unless
    allowed (the message names the differences and the flag)."""
    here = dict(torch_version=str(torch.__version__), transformers_version=str(transformers.__version__))
    assert SDC.platform_mismatches(here) == [] and SDC.check_platform(here, allow=False) == []
    for main in ({}, dict(torch_version=None, transformers_version=""), dict(platform="Linux"), None):
        assert SDC.platform_mismatches(main) == [] and SDC.check_platform(main, allow=False) == []
    monkeypatch.setattr(torch, "__version__", "2.12.0+cu130")
    monkeypatch.setattr(transformers, "__version__", "5.21.0")
    main = dict(torch_version="2.11.0+cu130", transformers_version="5.20.2", platform="Linux")
    want = ["torch 2.12.0+cu130 instead of 2.11.0+cu130 (main run)", "transformers 5.21.0 instead of 5.20.2 (main run)"]
    assert SDC.platform_mismatches(main) == want
    assert SDC.platform_mismatches(dict(transformers_version="5.21.0", torch_version="2.11.0+cu130")) == want[:1]
    assert SDC.platform_mismatches(dict(transformers_version="5.20.2")) == want[1:]        # no torch_version: not compared
    with pytest.raises(SDC.ControlRunError) as e:
        SDC.check_platform(main, allow=False)
    assert all(w in str(e.value) for w in want) and "--d-allow-platform-mismatch" in str(e.value)
    assert SDC.check_platform(main, allow=True) == want


def test_platform_mismatch_refused_in_the_run_and_recorded_when_allowed(mb, depth_bank, tmp_path):
    """A main run on other transformers: the run refuses before it writes anything; with
    ``d_allow_platform_mismatch`` it runs and records the difference (``platform_mismatch``, the
    main run's version in ``replication_platform``) and the evaluation code's SHA-256."""
    main = _write_main_run(RunDir(tmp_path / "runs", "D", mb.spec).path, mb, depth_bank, transformers_version="4.0.0")
    before = _snapshot(main)
    cfg = RunConfig(model=mb.spec, families=[f"{FAMILY}A"], out_root=str(tmp_path / "runs_ctl"), limit_variants=3)
    rd = RunDir(cfg.out_root, "D", cfg.model, mkdir=False).path
    kw = dict(ctx={}, limits=C.Limits.smoke(progress=False), d_control="token_language", d_main_run=str(main))
    want = [f"transformers {transformers.__version__} instead of 4.0.0 (main run)"]
    with pytest.raises(SDC.ControlRunError, match=r"instead of 4\.0\.0 \(main run\).*--d-allow-platform-mismatch"):
        SD.run(cfg, mb, rd, **kw)
    assert not Path(cfg.out_root).exists()
    SD.run(cfg, mb, rd, **dict(kw, d_allow_platform_mismatch=True))
    m = read_manifest(rd)
    assert m["status"] == "ok" and m["platform_mismatch"] == want and m["eval_code_sha256"] == _eval_code_sha256()
    assert m["replication_platform"]["main_transformers"] == "4.0.0"
    assert m["replication_platform"]["transformers"] == m["transformers_version"] == str(transformers.__version__)
    assert _snapshot(main) == before


def test_eval_code_sha256_of_the_evaluation_module(monkeypatch, tmp_path):
    """``eval_code_sha256``: the SHA-256 of lsne/prereg_eval.py next to the package's modules, ''
    when it cannot be read."""
    assert SDC.EVAL_CODE == Path(lsne.__file__).resolve().parent / "prereg_eval.py"
    assert SDC.eval_code_sha256() == _eval_code_sha256() == SDC.file_sha256(SDC.EVAL_CODE)
    monkeypatch.setattr(SDC, "EVAL_CODE", tmp_path / "missing" / "prereg_eval.py")
    assert SDC.eval_code_sha256() == ""


def test_resume_only_from_the_same_platform_and_code(mb, depth_bank, tmp_path, monkeypatch):
    """Checkpoints of an interrupted control run are resumed by a restart on the same platform and
    code; another torch version, attention implementation (sdpa vs eager: another kernel) or git
    sha changes the resume signature, and the restart discards them instead of mixing them with
    its own chunks.  Every restart ends with the tables of an uninterrupted run."""
    main = _write_main_run(RunDir(tmp_path / "runs", "D", mb.spec).path, mb, depth_bank)
    monkeypatch.setattr(C, "RESUME_CHUNK", 1)                   # one variant per checkpoint: 3 chunks
    cfg = RunConfig(model=mb.spec, families=[f"{FAMILY}A"], out_root=str(tmp_path / "runs_ctl"), limit_variants=3)
    limits = C.Limits.smoke(progress=False)
    kw = dict(d_control="token_language", d_main_run=str(main))
    ref_cfg = replace(cfg, out_root=str(tmp_path / "ref_ctl"))
    assert run_stages(ref_cfg, limits, ["D"], mb=mb, **kw)["D"]["status"] == "ok"
    ref = pd.read_csv(RunDir(ref_cfg.out_root, "D", mb.spec, mkdir=False).path / "damage_ctl.csv")
    assert read_manifest(RunDir(ref_cfg.out_root, "D", mb.spec, mkdir=False))["resumed_checkpoints"] == 0
    rd = RunDir(cfg.out_root, "D", cfg.model, mkdir=False).path
    real = SDC.control_damage_tables

    def interrupted(mb_):
        calls = []

        def flaky(*a, **k):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("session lost after the first checkpoint")
            return real(*a, **k)

        with monkeypatch.context() as mp:
            mp.setattr(SDC, "control_damage_tables", flaky)
            assert run_stages(cfg, limits, ["D"], mb=mb_, **kw)["D"]["status"] == "error"
        assert sorted(p.name for p in (rd / "_resume").glob("*.pkl")) == ["damage_ctl_00000.pkl"]

    def restart(mb_, **extra) -> dict:
        assert run_stages(cfg, limits, ["D"], mb=mb_, **kw, **extra)["D"]["status"] == "ok"
        pd.testing.assert_frame_equal(pd.read_csv(rd / "damage_ctl.csv"), ref, check_exact=True)
        assert not (rd / "_resume").exists()
        return read_manifest(rd)

    interrupted(mb)
    assert restart(mb)["resumed_checkpoints"] == 1                # same platform and code: resumed
    interrupted(mb)
    with monkeypatch.context() as mp:                            # another torch: refused unless allowed, then fresh
        mp.setattr(torch, "__version__", "0.0.1+other")
        assert SDC.run_platform(mb)["torch"] == "0.0.1+other"
        m = restart(mb, d_allow_platform_mismatch=True)
    assert m["resumed_checkpoints"] == 0
    assert m["platform_mismatch"] == [f"torch 0.0.1+other instead of {torch.__version__} (main run)"]
    sdpa = load_model(replace(mb.spec, attn_implementation="sdpa"))
    assert SDC.run_platform(sdpa)["attn"] == "sdpa" and SDC.run_platform(mb)["attn"] == "eager"
    assert SDC.attn_implementation(sdpa) == "sdpa" and SDC.run_platform(mb)["device"] == "cpu" and SDC.gpu_name(mb) == ""
    interrupted(sdpa)                                            # started with another attention kernel
    assert restart(mb)["resumed_checkpoints"] == 0
    interrupted(mb)
    with monkeypatch.context() as mp:                            # another commit (git sha)
        mp.setattr(SDC, "git_sha", lambda *a, **k: "f" * 40)
        assert restart(mb)["resumed_checkpoints"] == 0


def test_default_stage_d_injections_unchanged(mb, tok, depth_bank):
    assert inspect.signature(SD.damage_tables).parameters["injections_fn"].default is None
    vs = [v for v in stimuli.build_variants(_family(f"{FAMILY}A"), tok) if (v.K, v.C) in ((0, "C1"), (8, "C2"))]
    args = (mb, vs, depth_bank, ["lang_dom", "rand_00"], (0.05,))
    default = SD.damage_tables(*args, enabled=False, batch_size=3)
    explicit = SD.damage_tables(*args, enabled=False, batch_size=3, injections_fn=SD.injections)
    for a, b in zip(default, explicit):
        pd.testing.assert_frame_equal(a, b)
    dmg = default[0]
    assert list(dmg.columns) == STAGE_D_COLS
    for v in vs:
        rows = dmg[dmg["tag"] == v.tag]
        assert list(dict.fromkeys(rows["inject"])) == ["dossier", "depth"]
        assert set(rows.loc[rows["inject"] == "dossier", "inject_pos"]) == {v.n_tokens - 1}
        assert set(rows.loc[rows["inject"] == "depth", "inject_pos"]) == {stimuli.last_switched_position(v)}
    only = SD.damage_tables(*args, enabled=False, batch_size=3, injections_fn=lambda mb_, v: [("x", C.depth_site(mb_), 3)])[0]
    assert set(only["inject"]) == {"x"} and set(only["inject_pos"]) == {3} and len(only) == len(vs) * 2


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def test_cli_control_run_and_refusals(mb, depth_bank, tmp_path):
    """``--d-control`` / ``--d-main-run`` reach the control run (``<out>/D/<model>@<rev>``); wrong
    combinations, the main run's root as ``--out``, a selection without a test formulation and a
    main run on other torch / transformers versions stop before the model is loaded and before
    anything is created; ``--d-allow-platform-mismatch`` (only with ``--d-control``) runs anyway
    and records the differences."""
    main = _write_main_run(RunDir(tmp_path / "LSNE" / "runs", "D", mb.spec).path, mb, depth_bank)
    before = _snapshot(main)
    ctl_root = tmp_path / "LSNE" / "runs_ctl"
    base = ["run", "--model", mb.spec.name, "--revision", mb.spec.revision, "--dtype", mb.spec.dtype, "--device", "cpu",
            "--families", f"{FAMILY}A", "--limit-variants", "3", "--no-progress"]
    ctl = ["--d-control", "token_language", "--d-main-run", str(main)]
    for argv, match in ((["--stage", "D", "--out", str(ctl_root), "--d-control", "token_language"], "go together"),
                        (["--stage", "D", "--out", str(ctl_root), "--d-main-run", str(main)], "go together"),
                        (["--stage", "D", "--out", str(ctl_root), "--d-allow-platform-mismatch"],
                         "belongs to a stage-D control"),
                        (["--stage", "D", "--out", str(ctl_root), "--d-main-run", str(main), "--d-allow-platform-mismatch"],
                         "belongs to a stage-D control"),
                        (["--stage", "D,E", "--out", str(ctl_root)] + ctl, "stage D alone"),
                        (["--stage", "D", "--out", str(tmp_path / "LSNE" / "runs")] + ctl, "main stage-D run directory"),
                        (["--stage", "D", "--out", str(main)] + ctl, "lies inside the main stage-D run directory"),
                        (["--stage", "D", "--out", str(tmp_path / "LSNE")] + ctl, "holds it"),
                        (["--stage", "D", "--out", str(tmp_path / "LSNE_B" / "runs")] + ctl, "is a runs folder"),
                        (["--stage", "D", "--out", str(ctl_root), "--families", "DEV01"] + ctl, "no test formulation")):
        with pytest.raises(SystemExit, match=match):
            cli_main(base + argv)
    with pytest.raises(SystemExit):                                          # argparse: not a known control
        cli_main(base + ["--stage", "D", "--out", str(ctl_root), "--d-control", "tokens", "--d-main-run", str(main)])
    assert not ctl_root.exists() and not (tmp_path / "LSNE_B").exists() and _snapshot(main) == before

    assert cli_main(base + ["--stage", "D", "--out", str(ctl_root)] + ctl) == 0
    rd = ctl_root / "D" / main.name
    m = read_manifest(rd)
    assert m["status"] == "ok" and m["control"] == "token_language" and m["main_run"]["path"] == str(main.resolve())
    assert m["platform_mismatch"] == [] and m["eval_code_sha256"] == _eval_code_sha256()
    d = pd.read_csv(rd / "damage_ctl.csv")
    assert set(d["tag"]) == {f"{FAMILY}A:K0:C1", f"{FAMILY}A:K8:C1", f"{FAMILY}A:K2:u9D"}
    assert set(d["inject"]) == {"u9", "u10"} and (d["inject_token"] == ".").all()
    assert _snapshot(main) == before

    # a main run on other transformers: refused before anything is created; run with the flag and recorded
    other = _write_main_run(RunDir(tmp_path / "LSNE_old" / "runs", "D", mb.spec).path, mb, depth_bank,
                            transformers_version="4.0.0")
    other_root = tmp_path / "LSNE_old" / "runs_ctl"
    argv = base + ["--stage", "D", "--out", str(other_root), "--d-control", "token_language", "--d-main-run", str(other)]
    with pytest.raises(SystemExit, match=r"refused: .*transformers \S+ instead of 4\.0\.0 \(main run\).*"
                                         r"--d-allow-platform-mismatch"):
        cli_main(argv)
    assert not other_root.exists()
    assert cli_main(argv + ["--d-allow-platform-mismatch"]) == 0
    m = read_manifest(other_root / "D" / other.name)
    assert m["status"] == "ok" and m["replication_platform"]["main_transformers"] == "4.0.0"
    assert m["platform_mismatch"] == [f"transformers {transformers.__version__} instead of 4.0.0 (main run)"]


def test_refusal_in_the_run_leaves_no_empty_run_directory(mb, depth_bank, tmp_path):
    """A guard that can refuse only after the model is loaded (here: no control variant left at
    K = 16) leaves no run directory behind: the control run gets it only when it writes there, and
    an empty one is removed; a directory with anything in it, its parents, links and files stay."""
    main = _write_main_run(RunDir(tmp_path / "runs", "D", mb.spec).path, mb, depth_bank)
    cfg = RunConfig(model=mb.spec, families=[f"{FAMILY}A"], out_root=str(tmp_path / "runs_ctl"), k_levels=(16,))
    limits = C.Limits.smoke(progress=False)
    kw = dict(d_control="token_language", d_main_run=str(main))
    rd = RunDir(cfg.out_root, "D", cfg.model, mkdir=False).path
    rep = run_stages(cfg, limits, ["D"], mb=mb, **kw)
    assert rep["D"]["status"] == "error" and "no control variant left" in rep["D"]["error"]
    assert not Path(cfg.out_root).exists()
    rd.mkdir(parents=True)                                          # an empty run directory from before
    assert run_stages(cfg, limits, ["D"], mb=mb, **kw)["D"]["status"] == "error"
    assert not rd.exists() and rd.parent.is_dir()                   # only the run directory itself goes
    rd.mkdir()
    (rd / "notes.txt").write_text("keep", encoding="utf-8")
    assert run_stages(cfg, limits, ["D"], mb=mb, **kw)["D"]["status"] == "error"
    assert _snapshot(rd) == {"notes.txt": b"keep"}
    # remove_empty_run_dir: an empty directory only
    empty, link, file = tmp_path / "empty", tmp_path / "link", tmp_path / "file"
    empty.mkdir()
    link.symlink_to(empty, target_is_directory=True)
    file.write_text("x", encoding="utf-8")
    assert not SDC.remove_empty_run_dir(link) and link.is_symlink() and empty.is_dir()
    assert not SDC.remove_empty_run_dir(file) and file.exists()
    assert not SDC.remove_empty_run_dir(rd) and not SDC.remove_empty_run_dir(tmp_path / "missing")
    assert SDC.remove_empty_run_dir(empty) and not empty.exists()


def test_failed_control_run_restarts_and_refusal_writes_nothing(mb, depth_bank, tmp_path, monkeypatch):
    """A control run that fails after its guards leaves an error manifest with the control marks, so
    the restart is not refused as a main run; a refused run writes no error manifest at all and
    creates no run directory (not even an empty one inside the main run's directory)."""
    main = _write_main_run(RunDir(tmp_path / "runs", "D", mb.spec).path, mb, depth_bank)
    before = _snapshot(main)
    cfg = RunConfig(model=mb.spec, families=[f"{FAMILY}A"], out_root=str(tmp_path / "runs_ctl"), limit_variants=3)
    limits = C.Limits.smoke(progress=False)
    kw = dict(d_control="token_language", d_main_run=str(main))
    rd = RunDir(cfg.out_root, "D", cfg.model, mkdir=False).path
    real = SDC.control_damage_tables

    def boom(*a, **k):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(SDC, "control_damage_tables", boom)
    assert run_stages(cfg, limits, ["D"], mb=mb, **kw)["D"]["status"] == "error"
    m = read_manifest(rd)
    assert m["status"] == "error" and m["control"] == "token_language" and m["amendment"] == "6.11"
    assert (rd / "error.txt").exists()
    SDC.check_run_dir(rd, main)                                     # a restart is allowed
    monkeypatch.setattr(SDC, "control_damage_tables", real)
    assert run_stages(cfg, limits, ["D"], mb=mb, **kw)["D"]["status"] == "ok"
    assert read_manifest(rd)["status"] == "ok" and not (rd / "error.txt").exists()
    assert not SDC.mark_error_manifest(rd)                         # only an error manifest gets the marks
    # refused (the main run's root as the output root): no error.txt, no error manifest in the main run;
    # the main run's directory as the output root: no run directory is created inside it
    rep = run_stages(replace(cfg, out_root=str(tmp_path / "runs")), limits, ["D"], mb=mb, **kw)
    assert rep["D"]["status"] == "error" and "main stage-D run directory" in rep["D"]["error"]
    rep = run_stages(replace(cfg, out_root=str(main)), limits, ["D"], mb=mb, **kw)
    assert rep["D"]["status"] == "error" and "lies inside the main stage-D run directory" in rep["D"]["error"]
    assert _snapshot(main) == before and not (main / "D").exists()
