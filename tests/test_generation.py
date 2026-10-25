import warnings

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
    """DESIGN D3: ramp 0 -> eps_max -> 0 in n_steps steps, i.e. the zero-dose plateaus at
    both ends are part of the cycle (finding: they were missing, the ramp started at
    eps_max / n_steps)."""
    eps_max, n_steps, tpt = 1.0, 4, 2
    f = G.schedule_ramp(eps_max, n_steps, tpt)
    n_lev = n_steps + 1
    total = 2 * n_lev * tpt
    vals = [f(k) for k in range(total)]
    expected = []
    for level in [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]:
        expected += [eps_max * level / n_steps] * tpt
    assert vals == pytest.approx(expected)
    assert vals[:tpt] == [0.0] * tpt and vals[-tpt:] == [0.0] * tpt       # zero-dose plateaus inside the ramp
    assert f(-1) == 0.0 and f(total) == 0.0 and f(total + 5) == 0.0
    assert sum(1 for k in range(-5, total + 5) if f(k) > 0) == total - 2 * tpt     # dosed length
    # plateau per step, palindromic (same level on the way up and down)
    for k in range(total):
        assert f(k) == pytest.approx(f(total - 1 - k))
    assert f.n_new == total and f.n_steps == n_steps and f.tokens_per_step == tpt and f.n_levels == n_lev
    assert np.allclose(f.levels, G.ramp_levels(eps_max, n_steps)) and f.levels.shape == (n_lev,)
    assert np.allclose(G.ramp_levels(0.5, 2), [0.0, 0.25, 0.5])
    slices = G.ramp_step_slices(n_steps, tpt)
    assert len(slices) == 2 * n_lev
    assert [s[2] for s in slices] == [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]
    assert [s[0] for s in slices] == ["up"] * n_lev + ["down"] * n_lev
    assert [s[1] for s in slices] == list(range(2 * n_lev))
    assert all(s[3] == slice(s[1] * tpt, (s[1] + 1) * tpt) for s in slices)
    # production settings: 9 levels, 144 tokens
    assert G.schedule_ramp(0.1, 8, 8).n_new == 144 and G.ramp_levels(0.1, 8).shape == (9,)
    with pytest.raises(ValueError):
        G.schedule_ramp(0.1, 0, 8)


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
    assert G.block_fraction(np.array(list("NN"))) == (0.0, 0)                 # lang_id convention
    f_empty, n_empty = G.block_fraction(np.array(list("NN")), empty=np.nan)   # hysteresis: missing data
    assert np.isnan(f_empty) and n_empty == 0
    # per-token indicator: 1 for D, 0 for E, NaN for N (any shape)
    ind = G.token_indicator(np.array([list("EDN"), list("DDE")]))
    assert np.array_equal(ind, [[0.0, 1.0, np.nan], [1.0, 1.0, 0.0]], equal_nan=True)
    assert np.array_equal(G.token_indicator(list("EDN"), lang="E"), [1.0, 0.0, np.nan], equal_nan=True)


def test_hysteresis_area_nan_aware():
    """Finding: a plateau without E/D-labelled tokens entered the branches as f_D = 0 and
    faked an area.  Such plateaus are NaN now and skipped by the trapezoid; the area is NaN
    (not 0) when fewer than two levels carry data on both branches."""
    levels = [0.0, 0.25, 0.5]
    assert G.hysteresis_area([0.0, 0.0, 0.0], [0.4, 0.6, 0.8], levels) == pytest.approx(0.25 * 0.5 + 0.25 * 0.7)
    # the middle level is missing on one branch: integrate over the two remaining levels
    assert G.hysteresis_area([0.0, np.nan, 0.0], [0.4, 0.8, 0.9], levels) == pytest.approx(0.5 * (0.4 + 0.9) / 2)
    assert G.hysteresis_area([0.0, 0.0, 0.0], [0.4, np.nan, 0.9], levels) == pytest.approx(0.5 * (0.4 + 0.9) / 2)
    # fewer than two levels with data on both branches: undefined
    assert np.isnan(G.hysteresis_area([0.1], [0.5], [0.5]))
    assert np.isnan(G.hysteresis_area([np.nan, 0.0, 0.0], [0.4, np.nan, 0.9], levels))
    assert np.isnan(G.hysteresis_area([np.nan] * 3, [0.4, 0.5, 0.9], levels))
    assert G.levels_with_data([0.0, np.nan, 0.0], [0.4, 0.8, 0.9]) == 2
    assert G.levels_with_data([np.nan, 0.0, 0.0], [0.4, np.nan, 0.9]) == 1
    assert G.trapezoid([1.0, np.nan, 3.0], [0.0, 1.0, 2.0]) == pytest.approx(4.0)
    assert np.isnan(G.trapezoid([1.0], [0.0])) and np.isnan(G.trapezoid([], []))
    # the old behaviour, for the record: a fake 0 at the empty level produced a non-zero area
    assert G.trapezoid(np.array([0.5, 0.5]) - np.array([0.0, 0.5]), [0.25, 0.5]) == pytest.approx(0.0625)


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


def _finite_in_unit(x):
    x = np.asarray(x, dtype=np.float64)
    fin = x[np.isfinite(x)]
    return bool(((0 <= fin) & (fin <= 1)).all())


def test_hysteresis_small(mb, sample_ids):
    site = HookSite(mb.n_layers // 2, "block_input")
    n_steps, tpt, n_samples, n_null, seed = 2, 3, 2, 3, 0
    n_lev = n_steps + 1
    h = G.hysteresis(mb, sample_ids, site, _direction(mb), eps_max=0.5, n_steps=n_steps,
                     tokens_per_step=tpt, n_samples=n_samples, seed=seed, n_null=n_null)
    # n_steps + 1 levels including the zero-dose plateaus at both ends (DESIGN D3)
    assert h["n_levels"] == n_lev and h["up"].shape == (n_lev,) and h["down"].shape == (n_lev,)
    assert h["restart"].shape == (n_lev,) and h["restart_b"].shape == (n_lev,)
    assert np.allclose(h["eps_levels"], [0.0, 0.25, 0.5])
    # plateaus without E/D tokens are NaN (missing data), everything else is a fraction
    for key in ("up", "down", "restart", "restart_b"):
        assert _finite_in_unit(h[key])
    assert 0 <= h["n_levels_with_data"] <= n_lev
    assert h["n_levels_with_data"] == G.levels_with_data(h["up"], h["down"])
    assert np.isfinite(h["area"]) == (h["n_levels_with_data"] >= 2)
    assert h["area"] == pytest.approx(G.hysteresis_area(h["up"], h["down"], h["eps_levels"]), nan_ok=True)
    assert h["tokens"].shape == (n_samples, 2 * n_lev * tpt)
    assert h["labels"].shape == h["tokens"].shape and h["fD_window"].shape == h["tokens"].shape
    assert np.allclose(h["eps"], [0.0] * 3 + [0.25] * 3 + [0.5] * 6 + [0.25] * 3 + [0.0] * 3)
    curves = h["curves"]
    assert isinstance(curves, pd.DataFrame)
    assert set(curves.columns) >= {"branch", "replicate", "seed", "step", "level", "eps", "fD", "fD_window", "entropy", "n_ED"}
    assert sorted(curves["branch"].unique()) == ["down", "restart_a", "restart_b", "restart_c", "up"]
    assert (curves["branch"] == "up").sum() == n_lev and (curves["branch"] == "down").sum() == n_lev
    assert (curves.loc[curves["branch"].isin(["up", "down"]), "replicate"] == -1).all()
    assert curves.loc[curves["branch"] == "up", "level"].tolist() == list(range(n_lev))
    assert curves.loc[curves["branch"] == "down", "level"].tolist() == list(range(n_lev))[::-1]
    # the curve rows agree with the branch arrays; fD is NaN exactly where n_ED == 0
    assert curves["fD"].isna().to_numpy().tolist() == (curves["n_ED"] == 0).tolist()
    for branch, arr in (("up", h["up"]), ("down", h["down"])):
        sub = curves[curves["branch"] == branch].sort_values("level")
        assert np.allclose(sub["fD"].to_numpy(), arr, equal_nan=True)
    # null model: n_null independent restarts per level, seeds disjoint from the ramp and from
    # each other (no common random numbers across levels, restart[0] != up[0] by construction)
    assert h["n_null"] == n_null and h["restarts"].shape == (n_null, n_lev)
    assert np.array_equal(h["restart"], h["restarts"][0], equal_nan=True)
    assert np.array_equal(h["restart_b"], h["restarts"][1], equal_nan=True)
    seeds = h["restart_seeds"]
    assert seeds.shape == (n_null, n_lev) and len(set(seeds.reshape(-1).tolist())) == n_null * n_lev
    assert seed not in set(seeds.reshape(-1).tolist())
    assert seeds.tolist() == [[G.restart_seed(seed, r, l) for l in range(n_lev)] for r in range(n_null)]
    assert G.restart_seed(0, 0, 0) == 10_000 and G.restart_seed(3, 1, 2) == 3 + 20_000 + 2
    for r in range(n_null):
        sub = curves[curves["replicate"] == r]
        assert len(sub) == n_lev and sub["seed"].tolist() == seeds[r].tolist()
        assert sub["branch"].unique().tolist() == [f"restart_{'abc'[r]}"]
        assert sub["eps"].tolist() == pytest.approx([0.0, 0.25, 0.5])
    # null distribution: one signed area per replicate pair, NaN-aware symmetric summaries
    pairs = [(a, b) for a in range(n_null) for b in range(a + 1, n_null)]
    assert h["null_areas"].shape == (len(pairs),)
    assert np.allclose(h["null_areas"], [G.hysteresis_area(h["restarts"][a], h["restarts"][b], h["eps_levels"]) for a, b in pairs],
                       equal_nan=True)
    fin = h["null_areas"][np.isfinite(h["null_areas"])]
    if fin.size:
        assert h["null_area"] == pytest.approx(float(np.mean(fin)))
        assert h["null_area_sd"] == pytest.approx(float(np.sqrt(np.mean(fin ** 2))))
    else:
        assert np.isnan(h["null_area"]) and np.isnan(h["null_area_sd"])
    if np.isfinite(h["null_area_sd"]) and h["null_area_sd"] > 0 and np.isfinite(h["area"]):
        assert h["area_z"] == pytest.approx(h["area"] / h["null_area_sd"])
    else:
        assert np.isnan(h["area_z"])
    with pytest.raises(ValueError):
        G.hysteresis(mb, sample_ids, site, _direction(mb), eps_max=0.5, n_steps=1, tokens_per_step=1, n_samples=1, n_null=1)
    # area formula: trapezoid of (down - up) over the dose levels
    assert G.hysteresis_area([0.0, 0.0], [0.4, 0.8], [0.25, 0.5]) == pytest.approx(0.25 * 0.6)
    assert np.isnan(G.hysteresis_area([0.1], [0.5], [0.5]))


def test_hysteresis_restart_is_independent_of_ramp(mb, sample_ids):
    """Replicate 0 at level 0 no longer reproduces the ramp's first plateau (it used to share
    the ramp seed), and two replicates at the same level differ (own seeds)."""
    site = HookSite(mb.n_layers // 2, "block_input")
    vec = _direction(mb)
    n_steps, tpt, n_samples, seed = 2, 4, 4, 0
    h = G.hysteresis(mb, sample_ids, site, vec, eps_max=0.5, n_steps=n_steps, tokens_per_step=tpt,
                     n_samples=n_samples, seed=seed, n_null=2)
    from lsne.model import generate_steered
    eps0 = float(h["eps_levels"][0])                   # level 0 is the zero-dose plateau
    ramp_first = h["tokens"][:, :tpt]
    # what the old implementation did: restart with the ramp seed reproduces the first plateau
    same_seed = generate_steered(mb, sample_ids, site, vec, G.schedule_const(eps0), n_new=tpt, n_samples=n_samples,
                                 temperature=1.0, seed=seed)["tokens"].numpy()
    assert np.array_equal(same_seed, ramp_first)
    # the restart replicates use their own seeds
    rep0 = generate_steered(mb, sample_ids, site, vec, G.schedule_const(eps0), n_new=tpt, n_samples=n_samples,
                            temperature=1.0, seed=int(h["restart_seeds"][0, 0]))["tokens"].numpy()
    rep1 = generate_steered(mb, sample_ids, site, vec, G.schedule_const(eps0), n_new=tpt, n_samples=n_samples,
                            temperature=1.0, seed=int(h["restart_seeds"][1, 0]))["tokens"].numpy()
    assert not np.array_equal(rep0, ramp_first) and not np.array_equal(rep0, rep1)


def test_relaxation_small(mb, sample_ids):
    site = HookSite(mb.n_layers // 2, "block_input")
    n_new, t_on, n_samples = 16, 4, 2
    r = G.relaxation(mb, sample_ids, site, _direction(mb), eps=0.5, t_on=t_on, n_new=n_new,
                     n_samples=n_samples, seed=0, n_boot=20)
    assert r["fD_t"].shape == (n_new,) and r["fD_token_t"].shape == (n_new,) and r["entropy_t"].shape == (n_new,)
    assert np.isfinite(r["entropy_t"]).all() and (r["entropy_t"] > 0).all()
    assert np.isfinite(r["tau"]) or np.isnan(r["tau"])
    assert len(r["tau_ci"]) == 2 and r["tau_boot"].shape == (20,) and r["b_boot"].shape == (20,)
    assert len(r["b_ci"]) == 2 and isinstance(r["b_ci_excludes_zero"], bool)
    assert set(r["fit"]) == set(G._FIT_KEYS)
    assert (np.isnan(r["tau"]) and np.isnan(r["fit"]["tau"])) or r["tau"] == r["fit"]["tau"]
    assert np.isnan(r["tau"]) or r["fit"]["identifiable"]
    # the combined gate: series test AND between-sample bootstrap (two samples -> informative
    # unless too few resamples yield a fit on this short series)
    assert r["criterion"] == r["fit"]["criterion"] in ("series+bootstrap", "series")
    assert r["n_tau_boot"] == int(np.isfinite(r["tau_boot"]).sum())
    assert r["fit"]["identifiable"] == (r["fit"]["reason"] == "")
    if r["criterion"] == "series+bootstrap" and not r["b_ci_excludes_zero"]:
        assert "bootstrap" in r["fit"]["reason"] and np.isnan(r["tau"])
    assert np.allclose(r["eps_t"], [0.0] * t_on + [0.5] * (n_new - t_on))
    assert r["tokens"].shape == (n_samples, n_new) and r["labels"].shape == (n_samples, n_new)
    # tau is fitted on the unwindowed per-token D indicator (window=1, iid F-test, no MA null);
    # the running-window series fD_t is kept for plots only
    assert r["fit_window"] == 1 and r["fit"]["n_null"] == 0
    if np.isfinite(r["fit"]["vif"]):
        assert r["fit"]["vif"] == pytest.approx(1.0)
    assert r["fD_token"].shape == (n_samples, n_new) and r["fD_window"].shape == (n_samples, n_new)
    assert np.array_equal(r["fD_token"], G.token_indicator(r["labels"]), equal_nan=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        assert np.allclose(r["fD_token_t"], np.nanmean(r["fD_token"], axis=0), equal_nan=True)
        assert np.allclose(r["fD_t"], np.nanmean(r["fD_window"], axis=0), equal_nan=True)
    # the gate is a pure function of the per-sample indicator (reproducible without a model)
    g = G.fit_relaxation_samples(r["fD_token"], t_on, r["t"], window=1, n_boot=20, seed=0)
    assert np.allclose(g["fD_t"], r["fD_token_t"], equal_nan=True) and g["criterion"] == r["criterion"]
    assert np.allclose(g["b_boot"], r["b_boot"], equal_nan=True) and g["fit"]["reason"] == r["fit"]["reason"]
    assert g["window"] == 1 and G.fit_relaxation_samples(r["fD_window"], t_on, r["t"], window=8, n_boot=5)["window"] == 8


def _bernoulli_labels(rng, p_t, n_samples):
    lab = rng.random((n_samples, p_t.shape[0])) < p_t[None, :]
    return np.where(lab, "D", "E")


def test_relaxation_tau_floor_of_windowed_series():
    """Finding: tau was fitted on the running-window fraction (window 8).  The trailing box
    window turns a step into a rise over 8 tokens, so the fitted tau could not fall below
    ~window/2 and slower responses were biased by ~(window-1)/2.  The per-token indicator
    (window=1) has no such floor: a step is censored at TAU_BOUNDS[0], tau = 4 is recovered."""
    rng = np.random.default_rng(7)
    t = np.arange(64)
    t_on, w, B = 16, 8, 400                                    # many samples: a nearly noise-free mean

    def both_fits(p_t):
        labels = _bernoulli_labels(rng, p_t, B)
        fw = np.stack([G.window_fraction(labels[b], w) for b in range(B)])
        f_win = G.fit_relaxation(np.nanmean(fw, axis=0), t_on, t, window=w)
        f_ind = G.fit_relaxation(np.nanmean(G.token_indicator(labels), axis=0), t_on, t, window=1)
        return f_win, f_ind

    # instantaneous response (step one token after t_on)
    f_win, f_ind = both_fits(np.where(t >= t_on + 1, 0.7, 0.1))
    assert f_win["identifiable"] and f_win["tau_raw"] > 2.5                       # the floor: ~w/2
    assert f_ind["identifiable"] and f_ind["tau_censored"] and f_ind["tau_raw"] == pytest.approx(G.TAU_BOUNDS[0], rel=0.2)
    # tau = 4: the indicator recovers it, the windowed series overestimates it by several tokens
    tau_true = 4.0
    f_win, f_ind = both_fits(np.where(t >= t_on, 0.1 + 0.6 * (1 - np.exp(-(t - t_on) / tau_true)), 0.1))
    assert f_ind["identifiable"] and not f_ind["tau_censored"] and abs(f_ind["tau"] - tau_true) < 1.0
    assert f_win["tau_raw"] > f_ind["tau"] + 2.0
    # the full gate on per-sample indicators at the production sample size (8) still identifies
    # a strong response and reports its tau with the iid test (no Monte-Carlo null)
    p = np.where(t >= t_on, 0.1 + 0.6 * (1 - np.exp(-(t - t_on) / tau_true)), 0.1)
    g = G.fit_relaxation_samples(G.token_indicator(_bernoulli_labels(rng, p, 8)), t_on, t, n_boot=50, seed=0)
    assert g["window"] == 1 and g["fit"]["n_null"] == 0 and g["fit"]["identifiable"] and 1.0 < g["tau"] < 12.0


def test_fit_relaxation_recovers_tau():
    t = np.arange(64)
    t_on, tau_true = 16, 6.0
    y = np.where(t >= t_on, 0.1 + 0.6 * (1 - np.exp(-(t - t_on) / tau_true)), 0.1)
    fit = G.fit_relaxation(y, t_on, t)
    assert fit["tau"] == pytest.approx(tau_true, rel=1e-3)
    assert fit["a"] == pytest.approx(0.1, abs=1e-4) and fit["b"] == pytest.approx(0.6, abs=1e-4)
    assert fit["b_window"] == pytest.approx(0.6 * (1 - np.exp(-47 / tau_true)), abs=1e-4)
    assert fit["identifiable"] and not fit["tau_censored"] and fit["reason"] == ""
    assert fit["r2"] == pytest.approx(1.0, abs=1e-6) and fit["f_pvalue"] < 1e-6 and fit["n"] == 48
    assert fit["tau_raw"] == fit["tau"] and np.isfinite(fit["b_se"]) and fit["b_se"] < 1e-3
    assert set(fit) == set(G._FIT_KEYS)
    # window=1: plain iid F-test, no variance inflation, no Monte-Carlo null
    assert fit["f_pvalue"] == fit["f_pvalue_iid"] and fit["vif"] == pytest.approx(1.0) and fit["n_null"] == 0
    assert fit["n_eff"] == pytest.approx(48.0) and fit["criterion"] == "series"
    const = G.fit_relaxation(np.full(20, 0.3), 5)
    assert np.isnan(const["tau"]) and const["reason"] == "constant series" and not const["identifiable"]
    few = G.fit_relaxation(np.array([0.1, 0.2, np.nan]), 0)
    assert np.isnan(few["tau"]) and "four" in few["reason"]
    # the same clean response read through the label window is identified; the F statistic
    # keeps its plain degrees of freedom and its p-value is calibrated against the MA(window)
    # Monte-Carlo null (resolution 1 / (N_NULL_F + 1)); the response directions carry a
    # variance inflation of up to `window` -> n_eff = n / vif is reported, not used as df
    f8 = G.fit_relaxation(y, t_on, t, window=8)
    assert f8["identifiable"] and f8["f_stat"] == fit["f_stat"] and f8["f_pvalue_iid"] == fit["f_pvalue_iid"]
    assert f8["f_pvalue"] == pytest.approx(1.0 / (G.N_NULL_F + 1)) and f8["n_null"] == G.N_NULL_F
    assert 1.0 < f8["vif"] <= 8.0 and f8["n_eff"] == pytest.approx(48.0 / f8["vif"])
    assert f8["b_se_hac"] >= f8["b_se"] > 0
    # the MA(window) correlation and the cached null are deterministic
    R = G.ma_correlation(np.arange(5), 3)
    assert np.allclose(R, [[1, 2 / 3, 1 / 3, 0, 0], [2 / 3, 1, 2 / 3, 1 / 3, 0], [1 / 3, 2 / 3, 1, 2 / 3, 1 / 3],
                           [0, 1 / 3, 2 / 3, 1, 2 / 3], [0, 0, 1 / 3, 2 / 3, 1]])
    assert np.array_equal(G.ma_correlation(np.arange(4), 1), np.eye(4))
    tt = np.arange(t_on, 64, dtype=float)
    null = G.f_null_distribution(tt, t_on, 8)
    assert null.shape == (G.N_NULL_F,) and np.all(np.diff(null) >= 0) and np.isfinite(null).all()
    assert G.f_null_distribution(tt, t_on, 8) is null                       # cached
    assert np.array_equal(G.f_null_distribution(tt + 5, t_on + 5, 8), null)  # depends on the relative grid only


def _window_mean_noise(rng, p, n_samples=8, window=8, n=64):
    """Mean over samples of the trailing-window fraction of iid Bernoulli(p) labels: the
    autocorrelated noise structure of a non-responding relaxation() series."""
    lab = rng.random((n_samples, n)) < p
    fw = np.stack([G.window_fraction(np.where(lab[b], "D", "E"), window) for b in range(n_samples)])
    return np.nanmean(fw, axis=0)


def test_fit_relaxation_rejects_non_responses():
    """Finding: a 3-parameter fit of any noisy flat series returned a finite, often
    bound-saturated tau.  Now tau is NaN unless |b_window| >= B_MIN and the fit beats the
    constant model (F-test calibrated for the window autocorrelation); the raw value is kept."""
    rng = np.random.default_rng(0)
    t = np.arange(64)
    t_on = 16
    # iid noise around a constant
    fits = [G.fit_relaxation(0.3 + rng.normal(0, 0.05, 64), t_on, t) for _ in range(200)]
    assert all(np.isfinite(f["tau_raw"]) for f in fits)                       # the fit itself always converges
    assert np.mean([np.isfinite(f["tau"]) for f in fits]) <= 0.06              # ~alpha false positives
    assert all(f["reason"] for f in fits if np.isnan(f["tau"]))
    # window-smoothed label noise (what relaxation() produces at eps=0): the iid test is
    # anti-conservative, the window-calibrated one rejects at (about) the nominal alpha
    # (200 seeded series: 0.05 +/- 2 binomial sd = 0.08)
    series = [_window_mean_noise(rng, 0.3) for _ in range(200)]
    rate_iid = np.mean([np.isfinite(G.fit_relaxation(s, t_on, t, window=1)["tau"]) for s in series])
    rate_win = np.mean([np.isfinite(G.fit_relaxation(s, t_on, t, window=8)["tau"]) for s in series])
    assert rate_win <= 0.08 and rate_iid > 0.3, (rate_win, rate_iid)
    # a single bump of 0.01 on a flat series is not a response
    bump = np.full(64, 0.3)
    bump[20] = 0.31
    fb = G.fit_relaxation(bump, t_on, t)
    assert np.isnan(fb["tau"]) and np.isfinite(fb["tau_raw"]) and "b_min" in fb["reason"]
    # a near-linear fit (tau close to the upper bound) with a large asymptotic b but a rise of
    # 0.02 inside the window is gated on the realised amplitude b_window, not on b
    ramp_small = np.where(t >= t_on, 0.3 + 0.0004 * (t - t_on), 0.3)
    fr0 = G.fit_relaxation(ramp_small, t_on, t)
    assert fr0["tau_raw"] > 500 and abs(fr0["b"]) > G.B_MIN > abs(fr0["b_window"]) and "b_window" in fr0["reason"]
    assert np.isnan(fr0["tau"]) and fr0["f_pvalue"] < 1e-6         # the shape is real, the amplitude is not
    # a real response of the same noise level is still found (seeded)
    sig = np.where(t >= t_on, 0.1 + 0.5 * (1 - np.exp(-(t - t_on) / 6.0)), 0.1) + rng.normal(0, 0.05, 64)
    fs = G.fit_relaxation(sig, t_on, t)
    assert fs["identifiable"] and 3.0 < fs["tau"] < 12.0 and fs["b"] == pytest.approx(0.5, abs=0.1)
    # bound saturation is flagged as censored (linear ramp -> upper bound, step -> lower bound)
    ramp = np.where(t >= t_on, 0.1 + 0.01 * (t - t_on), 0.1)
    fr = G.fit_relaxation(ramp, t_on, t)
    assert fr["tau_censored"] and fr["tau_raw"] == pytest.approx(G.TAU_BOUNDS[1], rel=1e-3)
    step = np.where(t >= t_on + 1, 0.7, 0.1)
    fstep = G.fit_relaxation(step, t_on, t)
    assert fstep["tau_censored"] and fstep["tau_raw"] == pytest.approx(G.TAU_BOUNDS[0], rel=1e-3)
    assert not G.fit_relaxation(np.where(t >= t_on, 0.1 + 0.6 * (1 - np.exp(-(t - t_on) / 6.0)), 0.1), t_on, t)["tau_censored"]


def test_fit_relaxation_power_at_production_window():
    """Finding (fix round 2): the gate shrank the residual df by the label window
    (n=48, window=8 -> 3 df), so real, strong responses were rejected (b=0.3, sd=0.05:
    0 of 200 identified).  The F statistic now keeps its df and only its null is
    window-calibrated: such responses must be identified at the production settings
    (n_new=64, t_on=16, window=8)."""
    rng = np.random.default_rng(1)
    t = np.arange(64)
    t_on = 16
    n_ok = 0
    for _ in range(50):
        y = np.where(t >= t_on, 0.1 + 0.3 * (1 - np.exp(-(t - t_on) / 8.0)), 0.1) + rng.normal(0, 0.05, 64)
        f = G.fit_relaxation(y, t_on, t, window=8)
        n_ok += bool(f["identifiable"])
    assert n_ok >= 45, n_ok
    # one explicit case: identifiable, tau recovered, not censored
    y = np.where(t >= t_on, 0.1 + 0.3 * (1 - np.exp(-(t - t_on) / 8.0)), 0.1) + rng.normal(0, 0.05, 64)
    f = G.fit_relaxation(y, t_on, t, window=8)
    assert f["identifiable"] and 3.0 < f["tau"] < 20.0 and not f["tau_censored"] and f["reason"] == ""
    assert f["f_pvalue"] <= G.FIT_ALPHA and f["f_pvalue_iid"] <= f["f_pvalue"]


def _label_fractions(rng, p_t, n_samples=8, window=8):
    """Per-sample trailing-window fractions of Bernoulli(p_t) labels (relaxation() structure)."""
    n = p_t.shape[0]
    lab = rng.random((n_samples, n)) < p_t[None, :]
    return np.stack([G.window_fraction(np.where(lab[b], "D", "E"), window) for b in range(n_samples)])


def test_fit_relaxation_samples_gate():
    """The relaxation() gate on synthetic per-sample labels: a real response through the
    label window is identified (series test AND between-sample bootstrap), a flat one is
    not, and the bootstrap is skipped for a single sample."""
    rng = np.random.default_rng(2)
    t = np.arange(64)
    t_on = 16
    p_resp = np.where(t >= t_on, 0.1 + 0.5 * (1 - np.exp(-(t - t_on) / 8.0)), 0.1)
    g = G.fit_relaxation_samples(_label_fractions(rng, p_resp), t_on, t, window=8, n_boot=100, seed=0)
    assert g["criterion"] == "series+bootstrap" and g["fit"]["identifiable"] and np.isfinite(g["tau"])
    assert g["b_ci_excludes_zero"] and g["b_ci"][0] > 0 and 2.0 < g["tau"] < 30.0
    assert g["n_tau_boot"] >= 50 and np.isfinite(g["tau_ci"]).all() and g["tau_ci"][0] <= g["tau"] <= g["tau_ci"][1]
    assert g["tau_boot"].shape == (100,) and g["b_boot"].shape == (100,)
    # flat labels: mostly not identifiable (nominal 5 % for the series test alone, fewer combined)
    n_fp = 0
    for i in range(40):
        g0 = G.fit_relaxation_samples(_label_fractions(rng, np.full(64, 0.3)), t_on, t, window=8, n_boot=50, seed=i)
        n_fp += np.isfinite(g0["tau"])
        assert np.isfinite(g0["tau"]) == g0["fit"]["identifiable"]
        assert g0["fit"]["identifiable"] or g0["fit"]["reason"]
    assert n_fp <= 4, n_fp
    # a single sample: the bootstrap is degenerate and skipped, the series test decides
    g1 = G.fit_relaxation_samples(_label_fractions(rng, p_resp, n_samples=1), t_on, t, window=8, n_boot=20, seed=0)
    assert g1["criterion"] == "series" and g1["fit"]["criterion"] == "series"
    assert g1["fit"]["identifiable"] == (g1["fit"]["reason"] == "") and not g1["b_ci_excludes_zero"]
    assert np.isnan(g1["b_ci"]).all()


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
