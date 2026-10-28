"""Stage-D control run of the injection-site language (PREREGISTRATION section 6, amendment 11;
CPU, pythia-14m, family F01).

* sites: ``u9`` (last token of unit 9) for every control variant, ``u10`` (last token of unit 10)
  for K = 0 and C1, both at the depth site and defined by the unit, never by the language;
* ``u9D`` (control 3): K = 2, C ``u9D``, unit mask EEEEEEEEDE at C1, checked against K = 0;
* the primary site (``u9`` at K = 0 and C1) reproduces stage D's ``depth`` rows of the main run
  bit for bit, from the main run's bank, directions, doses and batch layout;
* the guards refuse before anything is written, and never touch the main run;
* the manifest carries the fields the evaluation (``lsne.prereg_eval``, key ``D_ctl``) reads;
* stage D's own injections are unchanged.
"""
from __future__ import annotations

import inspect
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lsne import stimuli
from lsne.cli import main as cli_main
from lsne.cli import run_stages
from lsne.config import RunConfig
from lsne.experiments import common as C
from lsne.experiments import stage_d_control as SDC
from lsne.experiments import stage_d_noneq as SD
from lsne.io import RunDir, read_manifest, write_manifest

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
    assert m["n_variants"] == 2 * len(CONTROL_CELLS) and m["n_excluded"] == 0
    assert m["audit"]["valid"] and m["audit"]["n_rows"] == len(d) and m["tables"]["damage_ctl"] == len(d)
    assert m["depth_site"] == dict(layer=C.depth_site(mb).layer, site="block_input")
    assert m["platform"]["device"] == "cpu" and m["platform"]["main_torch"] == mm["torch_version"]

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
    combinations and the main run's root as ``--out`` stop before the model is loaded."""
    main = _write_main_run(RunDir(tmp_path / "LSNE" / "runs", "D", mb.spec).path, mb, depth_bank)
    before = _snapshot(main)
    ctl_root = tmp_path / "LSNE" / "runs_ctl"
    base = ["run", "--model", mb.spec.name, "--revision", mb.spec.revision, "--dtype", mb.spec.dtype, "--device", "cpu",
            "--families", f"{FAMILY}A", "--limit-variants", "3", "--no-progress"]
    ctl = ["--d-control", "token_language", "--d-main-run", str(main)]
    for argv, match in ((["--stage", "D", "--out", str(ctl_root), "--d-control", "token_language"], "go together"),
                        (["--stage", "D", "--out", str(ctl_root), "--d-main-run", str(main)], "go together"),
                        (["--stage", "D,E", "--out", str(ctl_root)] + ctl, "stage D alone"),
                        (["--stage", "D", "--out", str(tmp_path / "LSNE" / "runs")] + ctl, "main stage-D run directory")):
        with pytest.raises(SystemExit, match=match):
            cli_main(base + argv)
    with pytest.raises(SystemExit):                                          # argparse: not a known control
        cli_main(base + ["--stage", "D", "--out", str(ctl_root), "--d-control", "tokens", "--d-main-run", str(main)])
    assert not ctl_root.exists() and _snapshot(main) == before

    assert cli_main(base + ["--stage", "D", "--out", str(ctl_root)] + ctl) == 0
    rd = ctl_root / "D" / main.name
    m = read_manifest(rd)
    assert m["status"] == "ok" and m["control"] == "token_language" and m["main_run"]["path"] == str(main.resolve())
    d = pd.read_csv(rd / "damage_ctl.csv")
    assert set(d["tag"]) == {f"{FAMILY}A:K0:C1", f"{FAMILY}A:K8:C1", f"{FAMILY}A:K2:u9D"}
    assert set(d["inject"]) == {"u9", "u10"} and (d["inject_token"] == ".").all()
    assert _snapshot(main) == before


def test_failed_control_run_restarts_and_refusal_writes_nothing(mb, depth_bank, tmp_path, monkeypatch):
    """A control run that fails after its guards leaves an error manifest with the control marks, so
    the restart is not refused as a main run; a refused run writes no error manifest at all."""
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
    # refused (the main run's root as the output root): no error.txt, no error manifest in the main run
    rep = run_stages(replace(cfg, out_root=str(tmp_path / "runs")), limits, ["D"], mb=mb, **kw)
    assert rep["D"]["status"] == "error" and "main stage-D run directory" in rep["D"]["error"]
    assert _snapshot(main) == before
