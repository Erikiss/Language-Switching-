"""Tests for lsne.trajectory (DESIGN section 4.5: depth, position, dose, coupling)."""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from lsne import trajectory as TJ
from lsne.config import HookSite
from lsne.model import all_block_inputs


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
TAIL = " The dog barks"
F01 = {("A", "E"): "The cat sleeps on the mat.", ("A", "D"): "Die Katze schläft auf der Matte.",
       ("B", "E"): "A cat is sleeping on a mat.", ("B", "D"): "Eine Katze schläft auf einer Matte."}


def make_variant(tok, text, family_id, formulation, K, C, lang, with_switch=True):
    ids = tok(text + TAIL, add_special_tokens=False)["input_ids"]
    T = len(ids)
    tail_start = T - 3
    annot = dict(lang=[lang] * tail_start + ["E"] * (T - tail_start),
                 unit=[1] * tail_start + [2] * (T - tail_start),
                 klass=["word_start"] * T, tail_start=tail_start)
    if with_switch:
        annot["last_switch_pos"] = max(tail_start - 2, 0)
    return types.SimpleNamespace(ids=ids, target_id=ids[-1], annot=annot, family_id=family_id,
                                 formulation=formulation, K=K, C=C, lang=lang)


@pytest.fixture(scope="module")
def k0_variants(tok):
    return [make_variant(tok, F01[("A", "E")], "F01", "A", 0, "C1", "E"),
            make_variant(tok, F01[("B", "E")], "F01", "B", 0, "C1", "E")]


@pytest.fixture(scope="module")
def k_variants(tok):
    return [make_variant(tok, F01[("A", "D")], "F01", "A", 2, "C1", "D"),
            make_variant(tok, F01[("B", "D")], "F01", "B", 2, "C1", "D")]


def synthetic_geom_table(series_by_cond: dict, readout="mean", k=0) -> pd.DataFrame:
    """series_by_cond: (family_id, K, C) -> list of log-volumes over layers 0..n-1."""
    rows = []
    for (fam, K, C), series in series_by_cond.items():
        for l, lv in enumerate(series):
            site = f"L{l}.block_input"
            rows.append(dict(site=site, readout=readout, k=k, family_id=fam, K=K, C=C,
                             group=f"{fam}|K{K}|{C}", n=16, log_volume=lv,
                             participation_ratio=1.0 + l, disp_euclid=0.1 * l, disp_mahal=0.2 * l))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# layer parsing / depth trajectory
# --------------------------------------------------------------------------------------
def test_layer_index():
    assert TJ.layer_index("L0.block_input") == 0
    assert TJ.layer_index("L11.block_input") == 11
    assert TJ.layer_index("L11.block_output") == 12
    assert TJ.layer_index("L3.mlp_output") == 3
    with pytest.raises(ValueError):
        TJ.layer_index("block_input")


def test_breathing_index():
    assert TJ.breathing_index([np.nan, -1, -2, -0.5]) == 0
    assert TJ.breathing_index([np.nan, 1, -1, 1, -1]) == 3
    assert TJ.breathing_index([np.nan, 1, 0, -1]) == 1          # zeros ignored
    assert TJ.breathing_index([]) == 0


def test_depth_trajectory_synthetic():
    n = 7
    mono = list(-np.arange(n, dtype=float))                      # strictly decreasing
    alt = [float((-1) ** l) for l in range(n)]                   # alternating
    gt = synthetic_geom_table({("F01", 0, "C1"): mono, ("F01", 2, "C1"): alt})
    dt = TJ.depth_trajectory(gt)
    assert len(dt) == 2 * n
    assert list(dt.columns[:5]) == ["family_id", "K", "C", "readout", "k"]
    m = dt[dt.K == 0].sort_values("layer")
    a = dt[dt.K == 2].sort_values("layer")
    assert m.layer.tolist() == list(range(n)) and a.layer.tolist() == list(range(n))
    assert np.isnan(m.delta_log_volume.iloc[0]) and np.allclose(m.delta_log_volume.iloc[1:], -1.0)
    assert (m.breathing_index == 0).all()
    assert m.breathing_amplitude.iloc[0] == pytest.approx(1.0)
    assert (a.breathing_index == n - 2).all()
    assert a.breathing_amplitude.iloc[0] == pytest.approx(2.0)
    assert a.layer_of_max_contraction.iloc[0] == 1              # first drop (+1 -> -1) lands on layer 1
    assert a.layer_of_max_expansion.iloc[0] == 2
    assert m.layer_of_max_contraction.iloc[0] in range(1, n)
    assert m.net_change.iloc[0] == pytest.approx(-(n - 1))
    assert (dt.n_layers == n).all()
    assert np.allclose(m.delta_participation_ratio.iloc[1:], 1.0)
    assert np.allclose(m.delta_disp_mahal.iloc[1:], 0.2)
    # the depth-ordering follows the numeric layer, not the row order
    shuffled = gt.sample(frac=1.0, random_state=0)
    dt2 = TJ.depth_trajectory(shuffled)
    assert dt2[dt2.K == 2].sort_values("layer").log_volume.tolist() == alt
    assert (dt2[dt2.K == 2].breathing_index == n - 2).all()


def test_depth_trajectory_handles_block_output_and_multiple_readouts():
    rows = []
    for ro in ("mean", "last"):
        for site, lv in (("L0.block_input", 0.0), ("L1.block_input", -1.0), ("L1.block_output", -3.0)):
            rows.append(dict(site=site, readout=ro, k=0, family_id="F", K=0, C="C1", log_volume=lv))
    dt = TJ.depth_trajectory(pd.DataFrame(rows))
    assert sorted(dt.layer.unique()) == [0, 1, 2]
    assert set(dt.readout) == {"mean", "last"}
    assert (dt.layer_of_max_contraction == 2).all()
    assert (dt.breathing_index == 0).all()


# --------------------------------------------------------------------------------------
# coupling with dynamics (synthetic)
# --------------------------------------------------------------------------------------
def test_couple_with_dynamics_synthetic():
    n = 8
    rng = np.random.default_rng(0)
    series = {("F01", 0, "C1"): list(np.cumsum(rng.standard_normal(n))),
              ("F01", 2, "C1"): list(np.cumsum(rng.standard_normal(n)))}
    dt = TJ.depth_trajectory(synthetic_geom_table(series))
    # lambda perfectly rank-correlated with the increments (per condition), KL anti-correlated,
    # given per condition and layer
    lam_rows, kl_rows = [], []
    for (fam, K, C), g in dt.groupby(["family_id", "K", "C"]):
        for _, r in g.iterrows():
            d = r.delta_log_volume if np.isfinite(r.delta_log_volume) else 0.0
            lam_rows.append(dict(family_id=fam, K=K, C=C, layer=r.layer, **{"lambda": 2.0 * d + 1.0}))
            kl_rows.append(dict(family_id=fam, K=K, C=C, layer=r.layer, kl=-d))
    out = TJ.couple_with_dynamics(dt, pd.DataFrame(lam_rows), pd.DataFrame(kl_rows))
    assert set(out.partner) == {"lambda", "kl"} and len(out) == 4
    assert np.isfinite(out.rho).all() and np.isfinite(out.pvalue).all()
    assert (out.n == n - 1).all()                               # first layer has no increment
    assert np.allclose(out[out.partner == "lambda"].rho, 1.0)
    assert np.allclose(out[out.partner == "kl"].rho, -1.0)
    # join on layer only when the partner table has no condition columns
    lam_only = pd.DataFrame(dict(layer=range(n), lam=rng.standard_normal(n)))
    out2 = TJ.couple_with_dynamics(dt, lam_only, None)
    assert len(out2) == 2 and np.isfinite(out2.rho).all() and (out2.partner == "lambda").all()
    assert out2.rho.between(-1, 1).all()


# --------------------------------------------------------------------------------------
# pythia-14m: steering tracks switching (T3), position trajectory (T2)
# --------------------------------------------------------------------------------------
def test_steering_tracks_switching_pythia(mb, k0_variants, k_variants):
    site = HookSite(mb.n_layers // 2, "block_input")
    direction = torch.randn(mb.d_model, generator=torch.Generator().manual_seed(1))
    eps_list = (0.0, 0.05)
    df = TJ.steering_tracks_switching(mb, k0_variants, k_variants, direction, site, eps_list,
                                      n_random=6, seed=0)
    L = mb.n_layers
    snaps = [s.key for s in all_block_inputs(mb)][site.layer:]
    assert list(df.columns[:10]) == ["snapshot", "layer", "eps", "cos_steer_vs_K", "cos_rand_mean",
                                     "cos_rand_sd", "norm_steer", "norm_K", "disp_mahal_steer",
                                     "disp_mahal_K"]
    assert len(df) == len(eps_list) * len(snaps)
    assert set(df.snapshot) == set(snaps) and df.layer.min() == site.layer and df.layer.max() == L
    # zero dose: no displacement, undefined cosine
    z = df[df.eps == 0.0]
    assert np.allclose(z.norm_steer, 0.0) and np.allclose(z.disp_mahal_steer, 0.0)
    assert z.cos_steer_vs_K.isna().all()
    # positive dose: cosines in [-1, 1], finite norms and displacements
    p = df[df.eps == 0.05]
    for c in ("cos_steer_vs_K", "cos_rand_mean", "cos_dir_vs_K"):
        assert p[c].between(-1, 1).all(), c
    assert (p.norm_steer > 0).all() and (p.cos_rand_sd >= 0).all()
    assert np.isfinite(p[["norm_K", "disp_mahal_steer", "disp_mahal_K"]].to_numpy()).all()
    assert (p.disp_mahal_K > 0).all() and (p.disp_mahal_steer > 0).all()
    # at the injection snapshot the steering displacement IS the dose direction
    inj = p[p.layer == site.layer].iloc[0]
    assert inj.cos_steer_vs_K == pytest.approx(inj.cos_dir_vs_K, abs=1e-4)
    assert (df.n_K0 == 2).all() and (df.n_K == 2).all() and (df.n_random == 6).all()
    # the Direction duck type (object with .vec) is accepted and gives the same result
    dir_obj = types.SimpleNamespace(vec=direction / direction.norm(), name="rand")
    df2 = TJ.steering_tracks_switching(mb, k0_variants, k_variants, dir_obj, site, (0.05,),
                                       n_random=2, seed=0)
    assert np.allclose(df2.cos_steer_vs_K.to_numpy(), p.cos_steer_vs_K.to_numpy(), atol=1e-5)
    # block_output injection site: first active snapshot is the next block input
    site_out = HookSite(1, "block_output")
    df3 = TJ.steering_tracks_switching(mb, k0_variants, k_variants, direction, site_out, (0.05,),
                                       n_random=1, seed=0)
    assert df3.layer.min() == 2 and df3.cos_steer_vs_K.between(-1, 1).all()


def test_position_trajectory_pythia(mb, k0_variants, k_variants):
    site = HookSite(1, "block_input")
    variants = k0_variants + k_variants
    df = TJ.position_trajectory(mb, variants, site, n_prefix_bins=4)
    assert list(df.columns) == ["site", "region", "index", "rel_pos", "pos_from_end",
                                "rel_to_last_switch", "n", "log_volume", "participation_ratio",
                                "delta_log_volume"]
    tail = df[df.region == "tail"]
    assert len(tail) == 3 and tail["index"].tolist() == [0, 1, 2]
    assert tail.pos_from_end.tolist() == [-3.0, -2.0, -1.0]
    assert (tail.n == 4).all() and np.isfinite(tail.log_volume).all()
    assert np.isnan(tail.delta_log_volume.iloc[0]) and np.isfinite(tail.delta_log_volume.iloc[1:]).all()
    assert tail.delta_log_volume.iloc[1] == pytest.approx(tail.log_volume.iloc[1] - tail.log_volume.iloc[0])
    assert tail.rel_to_last_switch.tolist() == [2.0, 3.0, 4.0]
    pb = df[df.region == "prefix_bin"]
    assert len(pb) == 4 and pb.rel_pos.tolist() == [0.125, 0.375, 0.625, 0.875]
    assert (pb.n == 4).all() and np.isfinite(pb.log_volume).all()
    sw = df[df.region == "from_switch"]
    assert len(sw) == 5 and sw.rel_to_last_switch.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    # from-switch rows overlap the tail rows at the same absolute positions
    assert sw.log_volume.iloc[-3:].tolist() == pytest.approx(tail.log_volume.tolist())
    # aligning the tail from its start gives the same cloud here (equal tail lengths)
    df_s = TJ.position_trajectory(mb, variants, site, align="tail_from_start", n_prefix_bins=4)
    assert df_s[df_s.region == "tail"].log_volume.tolist() == pytest.approx(tail.log_volume.tolist())
    # without last_switch_pos: no from_switch rows, NaN relative positions
    no_sw = [make_variant(mb.tokenizer, F01[("A", "E")], "F01", "A", 0, "C1", "E", with_switch=False),
             make_variant(mb.tokenizer, F01[("B", "E")], "F01", "B", 0, "C1", "E", with_switch=False)]
    df_n = TJ.position_trajectory(mb, no_sw, site, n_prefix_bins=2)
    assert (df_n.region != "from_switch").all() and df_n.rel_to_last_switch.isna().all()
    assert len(df_n[df_n.region == "tail"]) == 3
    with pytest.raises(ValueError):
        TJ.position_trajectory(mb, variants, site, align="middle")
