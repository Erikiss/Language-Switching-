import numpy as np
import pandas as pd
import pytest
import torch
from scipy import stats as sps

from lsne.config import HookSite
from lsne import generation as G

EN = "The man reads the report in the morning and the woman writes a letter."
DE = "Der Mann liest den Bericht am Morgen und die Frau schreibt einen Brief."


# --------------------------------------------------------------------------------------
# schedules
# --------------------------------------------------------------------------------------
def test_schedule_ramp_values_and_length():
    eps_max, n_steps, tpt = 1.0, 4, 2
    f = G.schedule_ramp(eps_max, n_steps, tpt)
    total = 2 * n_steps * tpt
    vals = [f(k) for k in range(total)]
    expected = []
    for level in [1, 2, 3, 4, 4, 3, 2, 1]:
        expected += [eps_max * level / n_steps] * tpt
    assert vals == pytest.approx(expected)
    assert f(-1) == 0.0 and f(total) == 0.0 and f(total + 5) == 0.0
    assert sum(1 for k in range(-5, total + 5) if f(k) > 0) == total     # ramp length
    # plateau per step, palindromic (same level on the way up and down)
    for k in range(total):
        assert f(k) == pytest.approx(f(total - 1 - k))
    assert f.n_new == total and f.n_steps == n_steps and f.tokens_per_step == tpt
    assert np.allclose(f.levels, G.ramp_levels(eps_max, n_steps))
    assert np.allclose(G.ramp_levels(0.5, 2), [0.25, 0.5])
    slices = G.ramp_step_slices(n_steps, tpt)
    assert len(slices) == 2 * n_steps
    assert [s[2] for s in slices] == [0, 1, 2, 3, 3, 2, 1, 0]
    assert [s[0] for s in slices] == ["up"] * 4 + ["down"] * 4


def test_schedule_quench_and_const():
    q = G.schedule_quench(0.3, 5)
    assert [q(k) for k in range(8)] == pytest.approx([0, 0, 0, 0, 0, 0.3, 0.3, 0.3])
    c = G.schedule_const(0.7)
    assert [c(k) for k in (0, 3, 100)] == pytest.approx([0.7, 0.7, 0.7])


# --------------------------------------------------------------------------------------
# language trajectory
# --------------------------------------------------------------------------------------
def _en_de_tokens(tok):
    en = tok(EN, add_special_tokens=False)["input_ids"]
    de = tok(DE, add_special_tokens=False)["input_ids"]
    n = min(len(en), len(de))
    return np.array([en[:n], de[:n]], dtype=np.int64)


@pytest.mark.parametrize("labeller", ["auto", "fallback"])
def test_language_trajectory_en_vs_de(tok, labeller):
    toks = _en_de_tokens(tok)
    B, n = toks.shape
    out = G.language_trajectory(tok, toks, labeller=labeller)
    assert out["labels"].shape == (B, n) and out["fD_window"].shape == (B, n)
    assert set(np.unique(out["labels"])) <= {"E", "D", "N"}
    fD_en = np.nanmean(out["fD_window"][0])
    fD_de = np.nanmean(out["fD_window"][1])
    assert fD_de > 0.8 and fD_en < 0.2
    assert out["n_switches"].shape == (B,) and list(out["n_switches"]) == [0, 0]
    assert out["switches"] == [[], []]
    assert out["labeller"] == ("lang_id" if labeller == "auto" else "fallback")
    # torch input and a single 1-D sample are accepted
    out_t = G.language_trajectory(tok, torch.as_tensor(toks), labeller=labeller)
    assert np.array_equal(out_t["labels"], out["labels"])
    out_1 = G.language_trajectory(tok, toks[1], labeller=labeller)
    assert out_1["labels"].shape == (1, n)


def test_language_trajectory_switches_and_window(tok):
    toks = _en_de_tokens(tok)
    mixed = np.concatenate([toks[0], toks[1]])[None, :]        # English then German
    out = G.language_trajectory(tok, mixed, window=8)
    assert out["n_switches"][0] == 1
    sw = out["switches"][0][0]
    assert toks.shape[1] - 2 <= sw <= toks.shape[1] + 1       # switch at the sentence boundary
    # trailing window: at the end of the German part fD -> 1, at the end of the English part 0
    fD = out["fD_window"][0]
    assert fD[-1] == pytest.approx(1.0)
    assert np.nanmax(fD[: sw]) == pytest.approx(0.0)
    # window_fraction on hand-made labels
    wf = G.window_fraction(list("EEDDNNDD"), window=4)
    assert not np.isnan(wf[0]) and wf[0] == 0.0
    assert wf[3] == pytest.approx(0.5) and wf[7] == pytest.approx(1.0)
    assert np.isnan(G.window_fraction(list("NNN"), window=2)).all()
    assert G.block_fraction(np.array(list("EDDN"))) == (pytest.approx(2 / 3), 3)
    assert G.block_fraction(np.array(list("NN"))) == (0.0, 0)


def test_fallback_labeller_word_rules():
    assert G._fallback_word_label("der") == "D"
    assert G._fallback_word_label("Über") == "D"           # umlaut
    assert G._fallback_word_label("the") == "E"
    assert G._fallback_word_label("xylophone") == "N"
    assert G._fallback_word_label("...") == "N"
    assert G.switch_positions(list("NEEDNDEN")) == [3, 6]


# --------------------------------------------------------------------------------------
# D3 hysteresis, D4 relaxation (pythia-14m)
# --------------------------------------------------------------------------------------
def _direction(mb):
    return torch.randn(mb.d_model, generator=torch.Generator().manual_seed(3))


def test_hysteresis_small(mb, sample_ids):
    site = HookSite(mb.n_layers // 2, "block_input")
    n_steps, tpt, n_samples = 2, 3, 2
    h = G.hysteresis(mb, sample_ids, site, _direction(mb), eps_max=0.5, n_steps=n_steps,
                     tokens_per_step=tpt, n_samples=n_samples, seed=0)
    assert h["up"].shape == (n_steps,) and h["down"].shape == (n_steps,)
    assert h["restart"].shape == (n_steps,) and h["restart_b"].shape == (n_steps,)
    assert np.allclose(h["eps_levels"], [0.25, 0.5])
    for key in ("up", "down", "restart", "restart_b"):
        assert np.isfinite(h[key]).all() and (0 <= h[key]).all() and (h[key] <= 1).all()
    assert np.isfinite(h["area"]) and np.isfinite(h["null_area"])
    assert h["area"] == pytest.approx(G.hysteresis_area(h["up"], h["down"], h["eps_levels"]))
    assert h["tokens"].shape == (n_samples, 2 * n_steps * tpt)
    assert h["labels"].shape == h["tokens"].shape and h["fD_window"].shape == h["tokens"].shape
    assert np.allclose(h["eps"], [0.25] * 3 + [0.5] * 6 + [0.25] * 3)
    curves = h["curves"]
    assert isinstance(curves, pd.DataFrame)
    assert set(curves.columns) >= {"branch", "step", "level", "eps", "fD", "fD_window", "entropy", "n_ED"}
    assert sorted(curves["branch"].unique()) == ["down", "restart_a", "restart_b", "up"]
    assert (curves["branch"] == "up").sum() == n_steps and (curves["branch"] == "down").sum() == n_steps
    # area formula: trapezoid of (down - up) over the dose levels
    assert G.hysteresis_area([0.0, 0.0], [0.4, 0.8], [0.25, 0.5]) == pytest.approx(0.25 * 0.6)
    assert G.hysteresis_area([0.1], [0.5], [0.5]) == 0.0


def test_relaxation_small(mb, sample_ids):
    site = HookSite(mb.n_layers // 2, "block_input")
    n_new, t_on, n_samples = 10, 3, 2
    r = G.relaxation(mb, sample_ids, site, _direction(mb), eps=0.5, t_on=t_on, n_new=n_new,
                     n_samples=n_samples, seed=0)
    assert r["fD_t"].shape == (n_new,) and r["entropy_t"].shape == (n_new,)
    assert np.isfinite(r["entropy_t"]).all() and (r["entropy_t"] > 0).all()
    assert np.isfinite(r["tau"]) or np.isnan(r["tau"])
    assert len(r["tau_ci"]) == 2 and r["tau_boot"].shape == (20,)
    assert np.allclose(r["eps_t"], [0.0] * t_on + [0.5] * (n_new - t_on))
    assert r["tokens"].shape == (n_samples, n_new) and r["labels"].shape == (n_samples, n_new)


def test_fit_relaxation_recovers_tau():
    t = np.arange(64)
    t_on, tau_true = 16, 6.0
    y = np.where(t >= t_on, 0.1 + 0.6 * (1 - np.exp(-(t - t_on) / tau_true)), 0.1)
    fit = G.fit_relaxation(y, t_on, t)
    assert fit["tau"] == pytest.approx(tau_true, rel=1e-3)
    assert fit["a"] == pytest.approx(0.1, abs=1e-4) and fit["b"] == pytest.approx(0.6, abs=1e-4)
    assert np.isnan(G.fit_relaxation(np.full(20, 0.3), 5)["tau"])          # constant -> nan
    assert np.isnan(G.fit_relaxation(np.array([0.1, 0.2, np.nan]), 0)["tau"])   # too few points


# --------------------------------------------------------------------------------------
# D5 switch statistics
# --------------------------------------------------------------------------------------
def test_switch_interval_stats_synthetic():
    labels = list("EEEEEDDDEEEEEEEDDEEEEDDDDDDE")
    st = G.switch_interval_stats(labels)
    assert st["intervals"].tolist() == [3, 7, 2, 4, 6]
    assert st["n_switches"] == 6 and st["n_intervals"] == 5
    assert st["geometric_p"] == pytest.approx(1 / np.mean([3, 7, 2, 4, 6]))
    assert 0 <= st["ks_stat"] <= 1 and 0 <= st["ks_p"] <= 1
    assert st["tail_index"] > 0                                   # 5 intervals -> Hill defined
    # 'N' is transparent; fewer than 5 intervals -> tail index nan
    st2 = G.switch_interval_stats(list("EENNDDNEEDD"))       # switches at 4, 7, 9
    assert st2["intervals"].tolist() == [3, 2] and np.isnan(st2["tail_index"])
    # no switches at all
    st0 = G.switch_interval_stats(list("EEEE"))
    assert st0["n_intervals"] == 0 and np.isnan(st0["geometric_p"]) and np.isnan(st0["ks_stat"])
    # list of samples and 2-D arrays pool the intervals
    st3 = G.switch_interval_stats([list("EEDDEE"), list("DDEEDD")])
    assert st3["intervals"].tolist() == [2, 2]
    st4 = G.switch_interval_stats(np.array([list("EEDDEE"), list("DDEEDD")]))
    assert st4["intervals"].tolist() == [2, 2]


def test_switch_interval_stats_geometric_sample():
    rng = np.random.default_rng(0)
    iv = rng.geometric(0.25, size=300)
    labels, cur = [], "E"
    for L in iv:
        labels += [cur] * int(L)
        cur = "D" if cur == "E" else "E"
    st = G.switch_interval_stats(labels)
    assert st["n_intervals"] == 298
    assert st["geometric_p"] == pytest.approx(0.25, abs=0.05)
    # exact discrete KS distance equals the brute-force sup over the integer support
    ivs, p = st["intervals"], st["geometric_p"]
    support = np.arange(1, ivs.max() + 1)
    ecdf = np.searchsorted(np.sort(ivs), support, side="right") / ivs.size
    brute = np.max(np.abs(ecdf - sps.geom.cdf(support, p)))
    assert st["ks_stat"] == pytest.approx(brute)
    assert st["ks_stat"] < 0.1 and st["ks_p"] > 0.05               # geometric data fits
    assert np.isfinite(st["tail_index"]) and st["tail_index"] > 0 and st["hill_k"] == 60


# --------------------------------------------------------------------------------------
# D6 susceptibility, D7 FDT
# --------------------------------------------------------------------------------------
def test_susceptibility_and_fdt():
    fD = {0.0: [0.10, 0.20], 0.2: [0.5, 0.6, 0.4], 0.1: [0.30, 0.40]}
    df = G.susceptibility(fD)
    assert list(df.columns) == ["eps", "n", "mean", "var", "sd", "chi"]
    assert df["eps"].tolist() == [0.0, 0.1, 0.2]
    assert df["n"].tolist() == [2, 2, 3]
    assert df["mean"].to_numpy() == pytest.approx([0.15, 0.35, 0.5])
    assert df["var"].iloc[0] == pytest.approx(np.var([0.1, 0.2], ddof=1))
    # linear mean -> chi equals the slope everywhere (central + one-sided differences)
    lin = {e: [2.0 * e + 0.1, 2.0 * e - 0.1] for e in (0.0, 0.05, 0.1, 0.2)}
    dfl = G.susceptibility(lin)
    assert dfl["chi"].to_numpy() == pytest.approx([2.0] * 4)
    assert np.isnan(G.susceptibility({0.1: [0.2, 0.3]})["chi"].iloc[0])
    m0 = np.array([0.1, 0.2, 0.3, 0.4])
    assert G.fdt_ratio(m0, 2.0) == pytest.approx(np.var(m0, ddof=1) / 2.0)
    assert np.isnan(G.fdt_ratio(m0, 0.0)) and np.isnan(G.fdt_ratio(m0, np.nan))
    assert np.isnan(G.fdt_ratio([0.1], 1.0))
