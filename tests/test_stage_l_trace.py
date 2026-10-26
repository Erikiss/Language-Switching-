"""Stage L with a trace source (DESIGN 4.9, ``--transplant-source trace:<model_dir>``).

A tiny trace LM (``lsne.chain.train``, tiny config, a few steps) is trained on the Xatu
fixture (tests/fixtures/xatu, ethPandaOps Xatu, CC BY 4.0) in a temporary directory; the
target is pythia-14m.  The end-to-end test runs ``python -m lsne.cli run --stage L --smoke
--transplant-source trace:<dir>`` and checks that every table is written, that the
functional transfer test is skipped (manifest), that the transplant prompts are real pairs
(trace on the source side, stimulus text on the target side) and that the verdict claims
nothing (the 14-transaction fixture has no fan-out >= 5, so the probe falls back and the
probe criterion cannot hold).  Unit tests cover the verdict rule (never a claim without the
probe criterion), the residualised alignment, the probe fallback flags, the shard
resolution and the saved-direction file.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.chain import cli as chain_cli
from lsne.chain import pairs as TPP
from lsne.chain import train as T
from lsne.chain import vocab as V
from lsne.chain.source import TraceLMSource
from lsne.cli import main
from lsne.experiments import common as C
from lsne.experiments import stage_l_transplant as SL
from lsne.transplant import SourceModel

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "xatu")
TRACE_TABLES = {"transfer_test", "transfer_summary", "layer_matching", "alignments", "transplant", "transplant_summary",
                "verdict", "trace_probe", "trace_probe_match", "trace_alignment_controls", "trace_bt", "trace_T",
                "trace_effects", "trace_sharing", "trace_verdict"}


@pytest.fixture(scope="module")
def trace_lm(tmp_path_factory):
    """ITR shards of the fixture, vocabulary and a tiny trace LM trained 8 steps (cached per module)."""
    d = tmp_path_factory.mktemp("trace_lm")
    itr_dir, voc_dir, lm_dir = str(d / "itr"), str(d / "vocab"), str(d / "lm")
    assert chain_cli.main(["serialize", "--cache", FIX, "--from", "1718000", "--to", "1921000", "--out", itr_dir]) == 0
    assert chain_cli.main(["vocab", "--in", itr_dir, "--out", voc_dir]) == 0
    vocab, tok = V.TraceVocab.load(voc_dir), V.load_tokenizer(voc_dir)
    tr, va, _ = chain_cli.corpus_blocks(itr_dir, vocab, block_size=32, max_tx_len=512, val_frac=0.2, log_fn=lambda m: None)
    model = T.make_model(len(vocab), "tiny", seed=0)
    T.train(model, tr, va, lm_dir, lr=3e-3, batch_size=8, steps=8, warmup=2, eval_every=4, save_every=0, seed=0,
            log_fn=None, tokenizer=tok, vocab=vocab, device="cpu", manifest_extra=dict(size="tiny", data=itr_dir))
    return dict(root=d, itr=itr_dir, vocab=voc_dir, lm=lm_dir)


@pytest.fixture(scope="module")
def tsrc(trace_lm):
    return TraceLMSource(trace_lm["lm"], device="cpu")


@pytest.fixture(scope="module")
def stage_run(trace_lm, tmp_path_factory):
    out = tmp_path_factory.mktemp("stage_l_trace")
    rc = main(["run", "--stage", "L", "--model", "EleutherAI/pythia-14m", "--revision", "step98000", "--dtype", "float32",
               "--device", "cpu", "--families", "DEV01", "--out", str(out), "--smoke", "--no-progress",
               "--transplant-source", f"trace:{trace_lm['lm']},k=5,n_probe=50"])      # options on the spec string
    return dict(rc=rc, dir=Path(out) / "L" / "pythia-14m@step98000")


# --------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------
def test_stage_l_trace_end_to_end(stage_run, trace_lm):
    assert stage_run["rc"] == 0
    d = stage_run["dir"]
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert m["status"] == "ok" and m["substages"]["n_failed"] == 0, m["substages"]["errors"]
    assert {p.stem for p in d.glob("*.csv")} == TRACE_TABLES and set(m["tables"]) == TRACE_TABLES
    # the source and the skipped functional transfer test
    assert m["source"]["kind"] == "trace" and m["source"]["n_layers"] == 2 and m["source"]["d_model"] == 64
    assert m["functional_transfer"]["skipped"] is True and "paired language texts" in m["functional_transfer"]["reason"]
    assert m["validated"] is None and m["self_consistency"] is False
    tr = m["trace"]
    assert tr["shards_from"] == "model_manifest" and os.path.samefile(tr["shards"], trace_lm["itr"])
    assert tr["n_transactions"] == 15 and tr["frozen_layer"] == 2 and tr["sae_source"] == "synthetic_random"
    assert tr["standin"]["ok"] is False and "no stand-in" in tr["standin"]["reason"] and tr["split"] == "all"
    assert tr["probe"]["feature_frozen"] is False and tr["probe"]["k_requested"] == 5      # no fan-out >= 5 in the fixture
    assert tr["direction_source"].startswith("probe:fanout")
    assert pd.read_csv(d / "transfer_test.csv").empty and pd.read_csv(d / "transfer_summary.csv").empty
    v18 = pd.read_csv(d / "verdict.csv")
    assert len(v18) == 1 and not bool(v18["counts_for_prereg"].iloc[0]) and v18["functional_transfer"].iloc[0] == "skipped"
    # probe: per source snapshot, untrained twin, frozen layer marked
    pr = pd.read_csv(d / "trace_probe.csv")
    assert pr["layer"].tolist() == [0, 1, 2] and pr["frozen"].tolist() == [False, False, True]
    assert {"auc_trained", "auc_untrained", "delta_ci_lo", "delta_ci_hi", "counts", "feature_frozen"} <= set(pr.columns)
    assert not pr["counts"].any()
    # residualised alignments: one map per target layer 0..6 from the frozen source layer
    al = pd.read_csv(d / "alignments.csv")
    assert al["layer_tgt"].tolist() == list(range(7)) and (al["layer_src"] == 2).all()
    assert al["method"].str.endswith("+len_resid").all()
    lm = pd.read_csv(d / "layer_matching.csv")
    assert len(lm) == 3 * 7 and np.isfinite(lm[["r2_holdout", "r2_holdout_raw"]].to_numpy()).all()
    ctr = pd.read_csv(d / "trace_alignment_controls.csv")
    assert {"r2_raw", "r2_resid", "r2_resid_untrained", "r2_length_only", "gain_resid"} <= set(ctr.columns)
    # transplant: real pairs -- a carrier transaction on the source side, the stimulus on the target side
    tp = pd.read_csv(d / "transplant.csv")
    assert set(tp["mode"]) == {"residual", "kv", "both"}
    assert set(tp["control"]) == {"none", "random", "identity", "permuted_alignment"}
    assert tp["paired"].all() and tp["source_item"].str.startswith("0x").all()
    assert (tp["source_item"] == tp["carrier"]).all() and (tp["n_src_tokens"] != tp["n_tgt_tokens"]).any()
    assert set(tp["K"]) == {0, 8} and (tp.loc[tp["mode"] != "residual", "n_kv_layers"] == 7).all()
    assert (tp.loc[tp["mode"] != "kv", "n_substituted"] == 2).all()        # smoke: l_n_prefix positions
    assert tp.groupby("formulation")["carrier"].nunique().eq(1).all()      # one carrier per formulation (K0 and K8)
    ident = tp[(tp["mode"] == "residual") & (tp["control"] == "identity")]
    assert (ident["written_norm"] == 0).all()
    assert np.isfinite(tp[["entropy", "logp_target", "fD", "cos_native"]].to_numpy()).all()
    # B / T readouts and effects
    bt = pd.read_csv(d / "trace_bt.csv")
    assert np.allclose(bt["B"], bt["logp_target"] - bt["logp_plain"])
    assert np.allclose(bt.loc[(bt["mode"] == "residual") & (bt["control"] == "identity"), "B"], 0.0, atol=1e-4)
    tt = pd.read_csv(d / "trace_T.csv")
    assert len(tt) == 3 * 4 and np.allclose(tt["dT"], tt["T"] - tt["T_plain"])
    eff = pd.read_csv(d / "trace_effects.csv")
    assert {"B_all", "B_K0", "B_K8C1", "T", "fD", "entropy"} <= set(eff["quantity"])
    assert set(eff["control"]) == {"random", "identity", "permuted_alignment", "plain"}
    assert not eff["enough_clusters"].any() and not eff["ci_excludes_0"].any()   # one family: never an effect
    # sharing index / network activity per arm (+ the plain model), synthetic SAE marked
    sh = pd.read_csv(d / "trace_sharing.csv")
    assert (sh["sae_source"] == "synthetic_random").all() and (sh["site"] == "L5.block_input").all()
    jac = sh[sh["measure"] == "sharing_jaccard"]
    assert len(jac) == 3 * 4 + 1 and jac["value"].between(0, 1).all()
    assert {"activity_frac_active", "activity_participation_ratio"} <= set(sh["measure"])
    # the verdict claims nothing
    vt = pd.read_csv(d / "trace_verdict.csv")
    assert vt["mode"].tolist() == ["residual", "kv", "both"]
    assert not vt["sharing_effect_claimed"].any() and not vt["probe_ok"].any() and not vt["counts_for_prereg"].any()
    assert vt["claim"].str.startswith("no claim").all() and vt["claim"].str.contains("probe").all()
    assert vt["claim"].str.contains(r"only 1 families \(< 3\)").all()          # one family: the reason is named
    assert vt["claim"].str.contains("PREREGISTRATION 18").all()
    assert (vt.loc[vt["mode"] == "residual", "n_planted_layers"] == 0).all()
    assert (vt.loc[vt["mode"] != "residual", "n_planted_layers"] == 7).all()


# --------------------------------------------------------------------------------------
# source / options
# --------------------------------------------------------------------------------------
def test_make_source_trace(mb, trace_lm):
    src, kind = SL.make_source(mb, f"trace:{trace_lm['lm']}", 0)
    assert kind == "trace" and isinstance(src, TraceLMSource) and isinstance(src, SourceModel)
    assert src.model_dir == trace_lm["lm"] and (src.d_model, src.n_layers) == (64, 2)
    with pytest.raises(ValueError):
        SL.make_source(mb, "trace:", 0)
    assert SL.make_source(mb, "self", 0)[1] == "self"                     # the other kinds unchanged


def test_trace_options_and_shards(trace_lm, tmp_path):
    lim = C.Limits.smoke()
    assert SL._trace_opt({}, lim, "k", 5) == 5 and SL._trace_opt({"trace_k": 3}, lim, "k", 5) == 3
    lim.l_trace_n_pairs = 7                                               # a Limits field when present
    assert SL._trace_opt({}, lim, "n_pairs", 1) == 7
    p, how = TPP.resolve_shards(trace_lm["lm"])
    assert how == "model_manifest" and os.path.samefile(p, trace_lm["itr"])
    assert TPP.resolve_shards(trace_lm["lm"], trace_lm["itr"]) == (trace_lm["itr"], "explicit")
    assert TPP.resolve_shards(trace_lm["lm"], str(tmp_path / "nope"))[0] is None
    assert TPP.resolve_shards(str(tmp_path))[0] is None                   # no manifest, no shards
    idx = TPP.shard_index(trace_lm["itr"])
    assert len(idx) == 15 and {"tx_hash", "era", "fan_out_eth", "n_tokens", "n_tokens_visible", "label"} <= set(idx.columns)
    assert (idx["n_tokens_visible"] <= 512).all() and idx["n_tokens"].max() == 1409
    hs = TPP.sample_pair_hashes(idx, 10, seed=0)
    assert len(hs) == 10 == len(set(hs)) and set(hs) <= set(idx["tx_hash"])
    # pairs fit the window: the drain (1 409 tokens) is never an alignment pair with max_len=512
    short = TPP.sample_pair_hashes(idx, 15, seed=0, max_len=512)
    long_ = set(idx.loc[idx["n_tokens"] > 512, "tx_hash"])
    assert long_ and not (set(short) & long_) and len(short) == 15 - len(long_)
    assert set(TPP.fetch_rows(trace_lm["itr"], hs[:3])) == set(hs[:3])


def test_fanout_probe_fallback_is_flagged(trace_lm):
    idx = TPP.shard_index(trace_lm["itr"])
    pos, neg, mt, info = TPP.fanout_probe(idx, n=50, seed=0, k=5)
    # the fixture has no fan-out >= 5 and no length-compatible pair: era-only matching with k = 2
    assert [a["n_pairs"] for a in info["attempts"]][:2] == [0, 0]
    assert info["k_used"] == 2 and info["match"] == "era_only" and info["feature_frozen"] is False
    assert len(pos) == len(neg) == info["n_pairs"] >= 2
    f = idx.set_index("tx_hash")
    assert (f.loc[pos, "fan_out_eth"] >= 2).all() and (f.loc[neg, "fan_out_eth"] == 1).all()
    assert (f.loc[pos, "era"].to_numpy() == f.loc[neg, "era"].to_numpy()).all()
    # a frozen probe is reported as such: length matching with a generous caliper on a synthetic index
    syn = pd.DataFrame(dict(tx_hash=[f"0x{i:064x}" for i in range(8)], era="post_fork", label="x",
                            fan_out_eth=[6, 6, 1, 1, 7, 1, 1, 1], fan_in_eth=1, gas_used=21000, dao_involved=False,
                            n_tokens=[40, 80, 42, 79, 300, 600, 12, 13], n_call_targets=1))
    p2, q2, mt2, info2 = TPP.fanout_probe(syn, n=10, seed=0, k=5, length_ratio=1.25)
    assert info2["feature_frozen"] is True and info2["match"] == "length" and len(p2) == 2
    lens = syn.set_index("tx_hash")["n_tokens"]
    assert all(abs(np.log(lens[a] / lens[b])) <= np.log(1.25) for a, b in zip(p2, q2))


def test_residualised_alignment_beyond_length():
    """The map is fitted on length-residualised states (R² beyond length), the intercept on
    the raw means; a target that is pure length gives no residualised R²."""
    g = np.random.default_rng(0)
    N, dS, dT = 80, 6, 5
    Z = g.normal(size=(N, 2))
    X = g.normal(size=(N, dS)) + Z @ g.normal(size=(2, dS)) * 3
    Wt = g.normal(size=(dS, dT))
    content = X - np.column_stack([np.ones(N), Z]) @ np.linalg.lstsq(np.column_stack([np.ones(N), Z]), X, rcond=None)[0]
    Y = content @ Wt + Z @ g.normal(size=(2, dT)) * 5 + 0.05 * g.normal(size=(N, dT)) + 2.0
    a = TPP.residualised_alignment(X, Y, Z, 1, 2, method="ridge", alpha=1e-3, seed=0, covariate_names=["a", "b"])
    assert a.method == "ridge+len_resid" and (a.layer_src, a.layer_tgt) == (1, 2) and a.r2_holdout > 0.9
    assert a.meta["residualised_on"] == ["a", "b"] and np.isfinite(a.meta["r2_holdout_raw"])
    hold, train = TPP.alignment_split(N, 0.2, 0)
    assert torch.allclose(a.b.double(), torch.from_numpy(Y[train].mean(0) - X[train].mean(0) @ a.W.double().numpy()),
                          atol=1e-4)
    only_len = Z @ g.normal(size=(2, dT)) + 0.01 * g.normal(size=(N, dT))
    b = TPP.residualised_alignment(X, only_len, Z, 0, 0, method="ridge", alpha=1e-3, seed=0)
    assert b.r2_holdout < 0.2 and b.meta["r2_holdout_raw"] > 0.5      # raw R² is length, residualised R² is not


def test_direction_file(trace_lm, tsrc, tmp_path):
    """A sharing_directions.pt of `lsne.chain.cli probe` replaces the in-stage direction; a file
    of another width / model is refused."""
    probe_dir = str(tmp_path / "probe")
    assert chain_cli.main(["probe", "--model", trace_lm["lm"], "--in", trace_lm["itr"], "--kind", "dao", "--layer", "all",
                           "--out", probe_dir, "--gas-bins", "1", "--device", "cpu", "--n", "6"]) == 0
    path = TPP.find_direction_file(trace_lm["lm"], probe_dir)
    assert path == os.path.join(probe_dir, "sharing_directions.pt")
    assert TPP.find_direction_file(str(tmp_path / "nothing")) is None
    vec, info = TPP.load_direction(path, 2, 64, trace_lm["lm"])
    assert vec.shape == (64,) and float(vec.norm()) == pytest.approx(1.0, abs=1e-5) and info["kind"] == "dao"
    assert TPP.load_direction(path, 2, 128, trace_lm["lm"])[0] is None
    assert "refused" in TPP.load_direction(path, 2, 64, str(tmp_path / "other_model"))[1]
    assert TPP.load_direction(path, 7, 64, trace_lm["lm"])[0] is None
    # another model directory with the same basename is refused; a copy of the same model is accepted
    import shutil
    other = tmp_path / "elsewhere" / os.path.basename(trace_lm["lm"])
    other.mkdir(parents=True)
    (other / "config.json").write_text('{"other": true}', encoding="utf-8")
    assert "refused" in TPP.load_direction(path, 2, 64, str(other))[1]
    copy = tmp_path / "copied_lm"
    shutil.copytree(trace_lm["lm"], copy)
    assert TPP.load_direction(path, 2, 64, str(copy))[0] is not None
    ok_spec = dict(kind="fanout", match="length", mask="none", readout="mean", k=5, length_ratio=1.25, max_len=512)
    assert TPP.direction_spec_ok(ok_spec, 5, 1.25, 512) == (True, [])
    for bad in (dict(kind="dao"), dict(k=3), dict(mask="all_addresses"), dict(readout="last"), dict(length_ratio=None)):
        assert TPP.direction_spec_ok({**ok_spec, **bad}, 5, 1.25, 512)[0] is False, bad
    # in the stage: the file wins, its cosine with the probe direction is recorded
    idx = TPP.shard_index(trace_lm["itr"])
    twin = TPP.untrained_twin(tsrc, seed=0)
    lim = C.Limits.smoke(n_boot=20)
    res = SL.trace_probe(tsrc, twin, idx, trace_lm["itr"], 2, lim, dict(trace_direction=probe_dir), seed=0)
    assert res["direction_source"] == f"file:{path}" and torch.allclose(res["v_src"], vec)
    assert -1.0 <= res["file_info"]["cos_with_probe_direction"] <= 1.0
    assert (res["table"]["direction_source"] == res["direction_source"]).all()
    # ... but a DAO direction is not the frozen fan-out probe: it can never support a claim
    assert res["direction_spec_ok"] is False and not res["table"]["direction_spec_ok"].any()
    assert any("kind='dao'" in m for m in res["file_info"]["spec_mismatch"])
    assert info["k"] == 5 and info["length_ratio"] is None and info["model_fingerprint"]
    res2 = SL.trace_probe(tsrc, twin, idx, trace_lm["itr"], 2, lim, {}, seed=0)
    assert res2["direction_source"].startswith("probe:fanout") and res2["v_src"].shape == (64,)


# --------------------------------------------------------------------------------------
# the verdict rule
# --------------------------------------------------------------------------------------
STANDIN_OK = dict(ok=True, path="runs/L/pythia-160m@step98000/verdict.csv", reason=None)


def _verdict_inputs(probe_counts=True, ci_lo=0.05, feature_frozen=True, n_pairs=40, gap=0.5, r2=0.3, effect=True,
                    spec_ok=True, planted=None):
    probe = dict(info=dict(feature_frozen=feature_frozen, k_requested=5, k_used=5, match="length"),
                 direction_source="probe:fanout(k=5, match=length)", direction_spec_ok=spec_ok,
                 table=pd.DataFrame([dict(layer=2, n_pairs=n_pairs, auc_trained=0.8, auc_untrained=0.6, delta=0.2,
                                          delta_ci_lo=ci_lo, delta_ci_hi=0.3, counts=probe_counts)]))
    al_rows = [dict(layer_tgt=2, layer_src=2, r2_holdout=r2, r2_permuted=r2 - gap, gap=gap)]
    for l, (r2_l, gap_l) in (planted or {}).items():        # further planted target layers of the KV prefix
        al_rows.append(dict(layer_tgt=l, layer_src=2, r2_holdout=r2_l, r2_permuted=r2_l - gap_l, gap=gap_l))
    align = dict(alignments=pd.DataFrame(al_rows),
                 controls=pd.DataFrame([dict(layer_src=2, layer_tgt=2, gain_resid=0.1, gain_resid_ci_lo=0.02,
                                             r2_length_only=0.4)]))
    rows = []
    for mode in SL.TRACE_MODES:
        for control in ("random", "permuted_alignment", "identity"):
            for q in ("B_all", "T"):
                rows.append(dict(mode=mode, control=control, quantity=q, mean=-0.5, ci_low=-0.8 if effect else -0.8,
                                 ci_high=-0.2 if effect else 0.1, se=0.1, n_clusters=4, n_rows=8, enough_clusters=True,
                                 ci_excludes_0=bool(effect)))
    return probe, align, pd.DataFrame(rows, columns=SL.EFFECT_COLS)


def test_trace_verdict_never_claims_without_the_probe_criterion():
    probe, align, eff = _verdict_inputs()
    v = SL.trace_verdict(probe, align, eff, pd.DataFrame(), 2, 2, "hub:x/y", False, standin=STANDIN_OK)
    assert v["sharing_effect_claimed"].all() and v["probe_ok"].all() and v["align_ok"].all()
    assert v["claim"].str.startswith("sharing effect").all() and v["counts_for_prereg"].all()
    assert v["standin_validated"].all() and not v["claim"].str.contains("PREREGISTRATION").any()
    # without a passed stand-in validation the claim stands, but nothing counts for PREREGISTRATION 18
    v0 = SL.trace_verdict(probe, align, eff, pd.DataFrame(), 2, 2, "hub:x/y", False)
    assert v0["sharing_effect_claimed"].all() and not v0["counts_for_prereg"].any() and not v0["standin_validated"].any()
    assert v0["claim"].str.contains("does not count for PREREGISTRATION 18: no stand-in run given").all()
    # a direction file of another probe specification never supports a claim (and never counts)
    p, a, e = _verdict_inputs(spec_ok=False)
    p["direction_source"] = "file:/x/sharing_directions.pt"
    p["file_info"] = dict(spec_mismatch=["kind='dao' (needs 'fanout')"])
    v1 = SL.trace_verdict(p, a, e, pd.DataFrame(), 2, 2, "hub:x/y", False, standin=STANDIN_OK)
    assert not v1["probe_ok"].any() and not v1["sharing_effect_claimed"].any() and not v1["counts_for_prereg"].any()
    assert v1["claim"].str.contains("not the frozen probe's direction").all() and v1["claim"].str.contains("kind='dao'").all()
    # kv / both plant the prefix through every target layer's map: one failing map blocks them, not residual
    p, a, e = _verdict_inputs(planted={0: (0.4, 0.5), 3: (0.3, 0.0035), 4: (-0.1, 0.193)})
    v2 = SL.trace_verdict(p, a, e, pd.DataFrame(), 2, 2, "hub:x/y", False, standin=STANDIN_OK).set_index("mode")
    assert v2.loc["residual", "sharing_effect_claimed"] and v2.loc["residual", "align_ok"]
    assert not v2.loc["kv", "align_ok"] and not v2.loc["kv", "sharing_effect_claimed"]
    assert not v2.loc["both", "align_ok"] and not v2.loc["both", "sharing_effect_claimed"]
    assert v2.loc["kv", "n_planted_layers"] == 4 and v2.loc["kv", "n_planted_layers_ok"] == 2
    assert v2.loc["kv", "planted_layers_failing"] == "3,4" and "target layers [3, 4]" in v2.loc["kv", "claim"]
    # too few families: the reason says so
    p, a, e = _verdict_inputs(effect=False)
    e = e.assign(n_clusters=1, enough_clusters=False, ci_excludes_0=False)
    v3 = SL.trace_verdict(p, a, e, pd.DataFrame(), 2, 2, "hub:x/y", False, standin=STANDIN_OK)
    assert v3["claim"].str.contains(r"only 1 families \(< 3\)").all() and (v3["n_clusters"] == 1).all()
    # every single missing ingredient blocks the claim; the probe criterion above all
    for kw in (dict(probe_counts=False), dict(ci_lo=-0.01), dict(feature_frozen=False), dict(n_pairs=3), dict(spec_ok=False)):
        p, a, e = _verdict_inputs(**kw)
        v = SL.trace_verdict(p, a, e, pd.DataFrame(), 2, 2, "hub:x/y", False, standin=STANDIN_OK)
        assert not v["sharing_effect_claimed"].any() and not v["probe_ok"].any(), kw
        assert v["claim"].str.contains("probe").all()
    for kw in (dict(gap=0.1), dict(r2=-0.2), dict(effect=False)):
        p, a, e = _verdict_inputs(**kw)
        v = SL.trace_verdict(p, a, e, pd.DataFrame(), 2, 2, "hub:x/y", False)
        assert v["probe_ok"].all() and not v["sharing_effect_claimed"].any(), kw
    # nothing at all (failed substages): rows for every mode, no claim
    v = SL.trace_verdict({}, {}, pd.DataFrame(columns=SL.EFFECT_COLS), pd.DataFrame(), 2, 2, "none", True)
    assert len(v) == 3 and not v["sharing_effect_claimed"].any() and not v["counts_for_prereg"].any()


def test_parse_trace_spec_and_prompt_spread():
    assert SL.parse_trace_spec("trace:runs/lm") == ("runs/lm", {})
    d, o = SL.parse_trace_spec("trace:runs/lm,shards=data/itr,k=3,n-prompts=12,standin=runs/L/x@y")
    assert d == "runs/lm" and o == dict(shards="data/itr", k="3", n_prompts="12", standin="runs/L/x@y")
    with pytest.raises(ValueError):
        SL.parse_trace_spec("trace:runs/lm,nonsense=1")
    with pytest.raises(ValueError):
        SL.parse_trace_spec("trace:runs/lm,shards")
    # one K=0 variant per family (round robin), formulations with a K=8/C1 partner first
    from types import SimpleNamespace as NS
    k0 = [NS(family_id=f, formulation=fo) for f in ("F1", "F2", "F3", "F4") for fo in ("a", "b")]
    k8 = [NS(family_id=f, formulation="b") for f in ("F1", "F2", "F3", "F4")]
    lim = C.Limits()
    got = SL.trace_prompt_variants(k0, k8, None, lim)
    assert [(v.family_id, v.formulation) for v in got] == [("F1", "b"), ("F2", "b"), ("F3", "b"), ("F4", "b")]
    assert len({v.family_id for v in got}) == 4 >= SL.TRACE_MIN_CLUSTERS
    six = SL.trace_prompt_variants(k0, k8, 6, lim)
    assert [v.family_id for v in six] == ["F1", "F2", "F3", "F4", "F1", "F2"]
    assert len(SL.trace_prompt_variants(k0, k8, None, C.Limits.smoke())) == C.Limits.smoke().l_n_prompts


def test_read_standin(tmp_path):
    """PREREGISTRATION 18: only a validated language-source stand-in counts (verdict.csv, not transfer_summary)."""
    def write(name, **row):
        d = tmp_path / name
        d.mkdir()
        pd.DataFrame([row]).to_csv(d / "verdict.csv", index=False)
        return str(d)
    ok = SL.read_standin(write("ok", source_kind="hf:EleutherAI/pythia-70m@step143000", self_consistency=False,
                               validated=True, counts_for_prereg=True))
    assert ok["ok"] is True and ok["path"].endswith("verdict.csv")
    rot = SL.read_standin(write("rot", source_kind="rotated_self", self_consistency=True, validated=np.nan,
                                counts_for_prereg=False))
    assert rot["ok"] is False and "self-consistency" in rot["reason"]
    failed = SL.read_standin(write("failed", source_kind="hf:x@y", self_consistency=False, validated=False,
                                   counts_for_prereg=True))
    assert failed["ok"] is False and "not validated" in failed["reason"]
    assert SL.read_standin(None)["ok"] is False and SL.read_standin(str(tmp_path / "missing"))["ok"] is False
