"""Generation-time non-equilibrium diagnostics (DESIGN §4.2, D3-D7).

The model is steered during sampling along a residual direction with a dose schedule
``eps(k)`` (relative RMS dose at the step that *produces* generated token ``k``, see
:func:`lsne.model.generate_steered`).  The order parameter is the German fraction
``f_D`` of the generated text (word-level labels 'E'/'D'/'N' from :mod:`lsne.lang_id`,
with a small self-contained fallback labeller if that module is unavailable).

Diagnostics
-----------
D3 hysteresis      :func:`hysteresis`      dose ramp 0 -> eps_max -> 0, f_D up vs. down, area
D4 relaxation      :func:`relaxation`      dose quench 0 -> eps at token t_on, f_D(t), tau
D5 switch statistics :func:`switch_interval_stats`  inter-switch intervals vs. geometric
D6 susceptibility  :func:`susceptibility`  chi(eps) = d f_D / d eps and Var(f_D)
D7 fluctuation-response :func:`fdt_ratio`  T_eff = Var_0(m) / chi
"""
from __future__ import annotations

import math
import re
import warnings
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy import optimize
from scipy import stats as sps

from .config import HookSite
from .model import ModelBundle, generate_steered

WINDOW = 8                 # trailing window (tokens) for the running German fraction
LABELS = ("E", "D", "N")


# ======================================================================================
# dose schedules
# ======================================================================================
def ramp_levels(eps_max: float, n_steps: int) -> np.ndarray:
    """Dose levels visited by :func:`schedule_ramp`: ``eps_max * (1..n_steps) / n_steps``."""
    n_steps = int(n_steps)
    return float(eps_max) * np.arange(1, n_steps + 1, dtype=np.float64) / n_steps


def ramp_step_slices(n_steps: int, tokens_per_step: int) -> list[tuple[str, int, int, slice]]:
    """``(branch, step, level, slice)`` for every plateau of the ramp.

    ``branch`` is ``"up"`` for steps ``0..n_steps-1`` and ``"down"`` for steps
    ``n_steps..2*n_steps-1``; ``level`` indexes :func:`ramp_levels` (the descending branch
    mirrors the ascending one); ``slice`` selects the generated token indices ``k``."""
    n_steps, tokens_per_step = int(n_steps), int(tokens_per_step)
    out = []
    for s in range(2 * n_steps):
        branch = "up" if s < n_steps else "down"
        level = s if s < n_steps else 2 * n_steps - 1 - s
        out.append((branch, s, level, slice(s * tokens_per_step, (s + 1) * tokens_per_step)))
    return out


def schedule_ramp(eps_max: float, n_steps: int = 8, tokens_per_step: int = 8) -> Callable[[int], float]:
    """Piecewise-constant dose ramp 0 -> eps_max -> 0 (D3).

    Generated tokens are grouped into plateaus of ``tokens_per_step`` tokens.  Plateaus
    ``0..n_steps-1`` ramp linearly up from ``eps_max/n_steps`` to ``eps_max``; plateaus
    ``n_steps..2*n_steps-1`` ramp back down symmetrically (``eps_max`` ... ``eps_max/n_steps``),
    so every level is visited exactly once on each branch.  Outside the ramp
    (``k < 0`` or ``k >= 2*n_steps*tokens_per_step``) the dose is 0 -- these are the "0"
    end points of the cycle.  The returned callable carries the attributes ``n_steps``,
    ``tokens_per_step``, ``n_new`` (= total ramp length) and ``levels``."""
    n_steps, tokens_per_step = int(n_steps), int(tokens_per_step)
    if n_steps < 1 or tokens_per_step < 1:
        raise ValueError("n_steps and tokens_per_step must be >= 1")
    eps_max = float(eps_max)
    total = 2 * n_steps * tokens_per_step

    def eps(k: int) -> float:
        k = int(k)
        if k < 0 or k >= total:
            return 0.0
        s = k // tokens_per_step
        level = s if s < n_steps else 2 * n_steps - 1 - s
        return eps_max * (level + 1) / n_steps

    eps.n_steps = n_steps                      # type: ignore[attr-defined]
    eps.tokens_per_step = tokens_per_step      # type: ignore[attr-defined]
    eps.n_new = total                          # type: ignore[attr-defined]
    eps.levels = ramp_levels(eps_max, n_steps)  # type: ignore[attr-defined]
    return eps


def schedule_quench(eps: float, t_on: int) -> Callable[[int], float]:
    """Dose jump (D4): 0 for generated tokens ``k < t_on``, ``eps`` afterwards."""
    eps, t_on = float(eps), int(t_on)

    def f(k: int) -> float:
        return eps if int(k) >= t_on else 0.0

    f.eps = eps          # type: ignore[attr-defined]
    f.t_on = t_on        # type: ignore[attr-defined]
    return f


def schedule_const(eps: float) -> Callable[[int], float]:
    """Constant dose ``eps`` at every generated token."""
    eps = float(eps)

    def f(k: int) -> float:
        return eps

    f.eps = eps          # type: ignore[attr-defined]
    return f


# ======================================================================================
# language labelling of generated tokens
# ======================================================================================
# Fallback labeller (used only if lsne.lang_id cannot be imported): German function
# words + umlaut/sharp-s check -> 'D'; English function words -> 'E'; everything else 'N'.
# Words that are common in both languages ("in", "an", "man", "so", "hat", "was", ...)
# are deliberately left out of both lists.
_FALLBACK_DE = frozenset("""
der die das den dem des ein eine einen einem einer eines und oder aber nicht ist sind
war waren wird werden hat haben auch noch schon sehr wie als wenn dass ich du er sie es
wir ihr sich mich mir dich dir ihm ihn ihnen uns mit nach von zu aus bei auf für über
unter vor durch gegen ohne um am im zum zur beim vom ins kein keine keinen diese dieser
dieses jetzt hier dort dann doch nur mehr immer wieder kann muss soll nichts etwas
""".split())
_FALLBACK_EN = frozenset("""
the a and or but not is are was were be been being has have had do does did will would
can could should may might must of to on at by for with from about into over under
through between after before during without within this that these those it its he she
they them his her their we our you your i my me us who what which when where why how
there here then than if as
""".split())
_UMLAUT = re.compile(r"[äöüÄÖÜß]")
_WORD_CORE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)


def _fallback_word_label(word: str) -> str:
    core_m = _WORD_CORE.search(word)
    if core_m is None:
        return "N"
    if _UMLAUT.search(word):
        return "D"
    core = core_m.group().lower()
    if core in _FALLBACK_DE:
        return "D"
    if core in _FALLBACK_EN:
        return "E"
    return "N"


def _fallback_token_words(tokenizer, ids: Sequence[int]) -> list[tuple[int, int, str]]:
    """Group tokens into words ``(start, end, text)``: a word starts at the first token,
    at a token whose decoded string begins with whitespace, and after (or at) a
    whitespace-only / punctuation-only token.  Groups are decoded jointly so multi-byte
    characters split across tokens (umlauts) are recovered."""
    strs = [tokenizer.decode([int(i)]) for i in ids]
    groups: list[list[int]] = []
    prev_break = True
    for t, s in enumerate(strs):
        is_ws = s != "" and s.strip() == ""
        is_punct = (not is_ws) and not any(ch.isalnum() for ch in s) and "�" not in s
        if t == 0 or prev_break or is_ws or is_punct or s[:1].isspace():
            groups.append([t])
        else:
            groups[-1].append(t)
        prev_break = is_ws or is_punct
    return [(g[0], g[-1] + 1, tokenizer.decode([int(ids[t]) for t in g]).strip()) for g in groups]


def fallback_label_tokens(tokenizer, ids: Sequence[int]) -> list[str]:
    """Self-contained per-token labeller ('E'/'D'/'N'), see module notes.  Used when
    :mod:`lsne.lang_id` is not importable."""
    ids = [int(i) for i in ids]
    out = ["N"] * len(ids)
    for s, e, text in _fallback_token_words(tokenizer, ids):
        lab = _fallback_word_label(text)
        for t in range(s, e):
            out[t] = lab
    return out


def _resolve_labeller(labeller: str = "auto") -> tuple[str, Callable]:
    """Return ``(name, fn)`` with ``fn(tokenizer, ids) -> list[str]``."""
    if labeller in ("auto", "lang_id"):
        try:
            from .lang_id import label_tokens  # noqa: WPS433 (optional dependency inside the package)
            return "lang_id", label_tokens
        except ImportError:
            if labeller == "lang_id":
                raise
    if labeller in ("auto", "fallback"):
        return "fallback", fallback_label_tokens
    raise ValueError(f"unknown labeller {labeller!r} (use 'auto', 'lang_id' or 'fallback')")


def switch_positions(labels: Sequence[str]) -> list[int]:
    """Indices where the E/D label differs from the previous E/D label ('N' is transparent)."""
    out: list[int] = []
    prev: Optional[str] = None
    for i, x in enumerate(labels):
        if x not in ("E", "D"):
            continue
        if prev is not None and x != prev:
            out.append(i)
        prev = x
    return out


def window_fraction(labels: Sequence[str], window: int = WINDOW, lang: str = "D") -> np.ndarray:
    """Running fraction of ``lang`` among the E/D labels in the trailing window
    ``[t-window+1, t]`` (NaN where the window holds no E/D label)."""
    lab = np.asarray(labels, dtype="<U1")
    n = lab.shape[0]
    is_lang = (lab == lang).astype(np.int64)
    is_ed = ((lab == "E") | (lab == "D")).astype(np.int64)
    cl = np.concatenate([[0], np.cumsum(is_lang)])
    ce = np.concatenate([[0], np.cumsum(is_ed)])
    t = np.arange(n)
    lo = np.maximum(0, t + 1 - int(window))
    n_lang = cl[t + 1] - cl[lo]
    n_ed = ce[t + 1] - ce[lo]
    out = np.full(n, np.nan, dtype=np.float64)
    ok = n_ed > 0
    out[ok] = n_lang[ok] / n_ed[ok]
    return out


def _to_int_array(tokens) -> np.ndarray:
    if isinstance(tokens, torch.Tensor):
        arr = tokens.detach().cpu().numpy()
    else:
        arr = np.asarray(tokens)
    arr = arr.astype(np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"tokens must be [B,n] (or [n]), got shape {arr.shape}")
    return arr


def language_trajectory(tokenizer, tokens, window: int = WINDOW, labeller: str = "auto") -> dict:
    """Word-level language labels of generated token arrays and derived trajectories.

    Parameters
    ----------
    tokens : array-like ``[B, n]`` of token ids (torch or numpy; a 1-D array is one sample).
    window : trailing window for the running German fraction (default 8).
    labeller : ``"auto"`` (lsne.lang_id if importable, else the fallback), ``"lang_id"``
        or ``"fallback"``.

    Returns
    -------
    dict with ``labels`` ``[B, n]`` (numpy ``<U1`` array of 'E'/'D'/'N'), ``fD_window``
    ``[B, n]`` (fraction of 'D' among E/D labels in the trailing window; NaN if the
    window holds no E/D token), ``switches`` (per sample: token indices where the E/D
    label changes, 'N' transparent), ``n_switches`` ``[B]``, ``window`` and ``labeller``.
    """
    toks = _to_int_array(tokens)
    B, n = toks.shape
    name, fn = _resolve_labeller(labeller)
    labels = np.full((B, n), "N", dtype="<U1")
    for b in range(B):
        if n:
            labels[b] = np.asarray(fn(tokenizer, toks[b].tolist()), dtype="<U1")
    fD = np.stack([window_fraction(labels[b], window) for b in range(B)]) if B else np.zeros((0, n))
    switches = [switch_positions(labels[b]) for b in range(B)]
    n_switches = np.array([len(s) for s in switches], dtype=np.int64)
    return dict(labels=labels, fD_window=fD, switches=switches, n_switches=n_switches,
                window=int(window), labeller=name)


def block_fraction(labels: np.ndarray, lang: str = "D") -> tuple[float, int]:
    """Pooled fraction of ``lang`` among the E/D labels of a label block (any shape).
    Returns ``(fraction, n_ED)``; the fraction is 0.0 when the block has no E/D label
    (the :func:`lsne.lang_id.language_fraction` convention)."""
    lab = np.asarray(labels, dtype="<U1")
    n_ed = int(((lab == "E") | (lab == "D")).sum())
    if n_ed == 0:
        return 0.0, 0
    return float((lab == lang).sum() / n_ed), n_ed


def _as_vector(direction, mb: Optional[ModelBundle] = None) -> torch.Tensor:
    """Accept a Tensor[d], an array, or an object with a ``.vec`` attribute (Direction)."""
    if hasattr(direction, "vec"):
        direction = direction.vec
    vec = torch.as_tensor(np.asarray(direction) if not isinstance(direction, torch.Tensor) else direction)
    vec = vec.detach().float().reshape(-1)
    if mb is not None and vec.numel() != mb.d_model:
        raise ValueError(f"direction has {vec.numel()} components, model d_model={mb.d_model}")
    return vec


def trapezoid(y, x) -> float:
    """Trapezoidal integral of ``y`` over ``x`` (0.0 for fewer than two points)."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.shape[0] < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# ======================================================================================
# D3 hysteresis
# ======================================================================================
def hysteresis_area(up, down, eps_levels) -> float:
    """Signed area between the descending and ascending branch,
    ``sum_levels (down - up) * d eps`` (trapezoid rule over the dose levels)."""
    return trapezoid(np.asarray(down, dtype=np.float64) - np.asarray(up, dtype=np.float64), eps_levels)


def hysteresis(mb: ModelBundle, input_ids: torch.Tensor, site: HookSite, direction, eps_max: float,
               n_steps: int = 8, tokens_per_step: int = 8, n_samples: int = 8, seed: int = 0,
               temperature: float = 1.0, window: int = WINDOW) -> dict:
    """D3: German fraction under a dose ramp 0 -> eps_max -> 0 (ascending vs. descending).

    One steered generation of ``2*n_steps*tokens_per_step`` tokens follows
    :func:`schedule_ramp`.  ``up[s]`` is the pooled fraction of 'D' among the E/D-labelled
    tokens generated on plateau ``s`` of the ascending branch (all samples pooled), and
    ``down[s]`` the same on the descending branch at the same dose level
    ``eps_levels[s]``.  ``area = sum_s (down - up) d eps`` (trapezoid).

    Null model (independent restarts, no memory): for each level an independent
    generation of ``tokens_per_step`` tokens from the prompt at constant dose
    (:func:`schedule_const`) is run twice (seeds ``seed`` and ``seed + 1000``);
    ``restart`` holds the first run's fractions, ``restart_b`` the second's, and
    ``null_area`` is the same area formula applied to ``(restart_b - restart)``.  Note
    that the ramp run and the first restart run share ``seed``, so ``up[0]`` and
    ``restart[0]`` are measured on the same first tokens.

    Returns
    -------
    dict(up, down, eps_levels, area, null_area, restart, restart_b, curves, tokens,
    labels, fD_window, entropies, eps, n_steps, tokens_per_step, eps_max, site).
    ``curves`` is a long DataFrame with one row per (branch, step): columns
    ``branch`` ('up', 'down', 'restart_a', 'restart_b'), ``step``, ``level``, ``eps``,
    ``k_start``, ``k_end``, ``fD`` (pooled fraction), ``fD_window`` (mean running
    fraction), ``entropy`` (mean next-token entropy), ``n_ED`` (E/D tokens pooled).
    """
    vec = _as_vector(direction, mb)
    n_steps, tokens_per_step = int(n_steps), int(tokens_per_step)
    sched = schedule_ramp(eps_max, n_steps, tokens_per_step)
    n_new = sched.n_new
    res = generate_steered(mb, input_ids, site, vec, sched, n_new=n_new, n_samples=n_samples,
                           temperature=temperature, seed=seed)
    traj = language_trajectory(mb.tokenizer, res["tokens"], window=window)
    labels, fDw = traj["labels"], traj["fD_window"]
    ent = res["entropies"].numpy()
    levels = ramp_levels(eps_max, n_steps)

    up = np.full(n_steps, np.nan)
    down = np.full(n_steps, np.nan)
    rows: list[dict] = []
    for branch, step, level, sl in ramp_step_slices(n_steps, tokens_per_step):
        f, n_ed = block_fraction(labels[:, sl])
        (up if branch == "up" else down)[level] = f
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fw = float(np.nanmean(fDw[:, sl])) if fDw[:, sl].size else np.nan
        rows.append(dict(branch=branch, step=step, level=level, eps=float(levels[level]),
                         k_start=sl.start, k_end=sl.stop, fD=f, fD_window=fw,
                         entropy=float(ent[:, sl].mean()), n_ED=n_ed))

    # null model: independent restarts at every level (two independent runs)
    restart_a = np.full(n_steps, np.nan)
    restart_b = np.full(n_steps, np.nan)
    for level, eps in enumerate(levels):
        for tag, s, store in (("restart_a", int(seed), restart_a), ("restart_b", int(seed) + 1000, restart_b)):
            r = generate_steered(mb, input_ids, site, vec, schedule_const(float(eps)), n_new=tokens_per_step,
                                 n_samples=n_samples, temperature=temperature, seed=s)
            t = language_trajectory(mb.tokenizer, r["tokens"], window=window)
            f, n_ed = block_fraction(t["labels"])
            store[level] = f
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                fw = float(np.nanmean(t["fD_window"])) if t["fD_window"].size else np.nan
            rows.append(dict(branch=tag, step=level, level=level, eps=float(eps), k_start=0,
                             k_end=tokens_per_step, fD=f, fD_window=fw,
                             entropy=float(r["entropies"].numpy().mean()), n_ED=n_ed))

    area = hysteresis_area(up, down, levels)
    null_area = hysteresis_area(restart_a, restart_b, levels)
    curves = pd.DataFrame(rows, columns=["branch", "step", "level", "eps", "k_start", "k_end",
                                         "fD", "fD_window", "entropy", "n_ED"])
    return dict(up=up, down=down, eps_levels=levels, area=area, null_area=null_area,
                restart=restart_a, restart_b=restart_b, curves=curves,
                tokens=res["tokens"].numpy(), labels=labels, fD_window=fDw, entropies=ent,
                eps=res["eps"].numpy(), n_steps=n_steps, tokens_per_step=tokens_per_step,
                eps_max=float(eps_max), site=site.key, n_samples=int(n_samples), seed=int(seed))


# ======================================================================================
# D4 relaxation after a dose quench
# ======================================================================================
TAU_BOUNDS = (0.5, 1000.0)


def _relax_model(t, a, b, tau, t_on):
    return a + b * (1.0 - np.exp(-(t - t_on) / tau))


def fit_relaxation(fD_t: np.ndarray, t_on: int, t: Optional[np.ndarray] = None) -> dict:
    """Fit ``fD_t[t >= t_on]`` to ``a + b * (1 - exp(-(t - t_on)/tau))`` with
    :func:`scipy.optimize.curve_fit` (``tau`` bounded to ``TAU_BOUNDS``).  NaN entries are
    ignored.  Returns ``dict(a, b, tau)`` with ``tau = nan`` (and ``a``, ``b`` = nan) if the
    series is constant, has fewer than four finite points, or the fit fails."""
    y = np.asarray(fD_t, dtype=np.float64)
    tt = np.arange(y.shape[0], dtype=np.float64) if t is None else np.asarray(t, dtype=np.float64)
    m = (tt >= t_on) & np.isfinite(y)
    y, tt = y[m], tt[m]
    nan = dict(a=np.nan, b=np.nan, tau=np.nan)
    if y.shape[0] < 4 or not np.isfinite(y).all() or np.ptp(y) == 0.0:
        return nan
    tau0 = float(np.clip(max(1.0, (tt[-1] - tt[0]) / 4.0), *TAU_BOUNDS))
    p0 = [float(y[0]), float(y[-1] - y[0]), tau0]
    lo = [-np.inf, -np.inf, TAU_BOUNDS[0]]
    hi = [np.inf, np.inf, TAU_BOUNDS[1]]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = optimize.curve_fit(lambda x, a, b, tau: _relax_model(x, a, b, tau, t_on),
                                         tt, y, p0=p0, bounds=(lo, hi), maxfev=2000)
    except (RuntimeError, ValueError, TypeError, optimize.OptimizeWarning):
        return nan
    a, b, tau = (float(v) for v in popt)
    if not np.isfinite(tau):
        return nan
    return dict(a=a, b=b, tau=tau)


def relaxation(mb: ModelBundle, input_ids: torch.Tensor, site: HookSite, direction, eps: float,
               t_on: int = 16, n_new: int = 64, n_samples: int = 8, seed: int = 0,
               temperature: float = 1.0, window: int = WINDOW, n_boot: int = 20) -> dict:
    """D4: response of the German fraction to a dose quench 0 -> eps at generated token
    ``t_on`` (:func:`schedule_quench`).

    ``fD_t[t]`` is the mean over samples of the running German fraction (trailing
    window) at generated position ``t`` (NaN-aware); ``entropy_t`` the mean next-token
    entropy.  ``tau`` is the time constant of an exponential fit of ``fD_t[t_on:]`` to
    ``a + b (1 - exp(-(t - t_on)/tau))`` (NaN if the fit fails or the series is constant);
    ``tau_ci`` is a (2.5, 97.5) percentile interval from ``n_boot`` bootstrap resamples
    of the samples (NaN pair if fewer than two resamples yield a finite tau).

    Returns dict(fD_t, entropy_t, tau, tau_ci, tau_boot, fit (a, b, tau), t, t_on, eps,
    tokens, labels, fD_window, entropies, eps_t, site).
    """
    vec = _as_vector(direction, mb)
    n_new, t_on = int(n_new), int(t_on)
    sched = schedule_quench(eps, t_on)
    res = generate_steered(mb, input_ids, site, vec, sched, n_new=n_new, n_samples=n_samples,
                           temperature=temperature, seed=seed)
    traj = language_trajectory(mb.tokenizer, res["tokens"], window=window)
    fDw = traj["fD_window"]                            # [B, n_new]
    ent = res["entropies"].numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fD_t = np.nanmean(fDw, axis=0)
    entropy_t = ent.mean(axis=0)
    t = np.arange(n_new)
    fit = fit_relaxation(fD_t, t_on, t)
    rng = np.random.default_rng(int(seed))
    B = fDw.shape[0]
    taus = np.full(int(n_boot), np.nan)
    for i in range(int(n_boot)):
        idx = rng.integers(0, B, size=B)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            fb = np.nanmean(fDw[idx], axis=0)
        taus[i] = fit_relaxation(fb, t_on, t)["tau"]
    finite = taus[np.isfinite(taus)]
    tau_ci = (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))) if finite.size >= 2 \
        else (np.nan, np.nan)
    return dict(fD_t=fD_t, entropy_t=entropy_t, tau=fit["tau"], tau_ci=tau_ci, tau_boot=taus, fit=fit,
                t=t, t_on=t_on, eps=float(eps), tokens=res["tokens"].numpy(), labels=traj["labels"],
                fD_window=fDw, entropies=ent, eps_t=res["eps"].numpy(), site=site.key,
                n_samples=int(n_samples), seed=int(seed))


# ======================================================================================
# D5 switch statistics
# ======================================================================================
def _label_rows(labels) -> list[list[str]]:
    """Normalise ``labels`` (one sequence, a list of sequences, or a 2-D array) to rows."""
    if isinstance(labels, np.ndarray):
        if labels.ndim == 2:
            return [list(map(str, row)) for row in labels]
        return [list(map(str, labels))]
    labels = list(labels)
    if not labels:
        return [[]]
    if isinstance(labels[0], str):
        return [labels]
    return [list(map(str, row)) for row in labels]


def inter_switch_intervals(labels: Sequence[str]) -> list[int]:
    """Distances (in tokens) between consecutive switch positions of one sequence."""
    pos = switch_positions(labels)
    return [b - a for a, b in zip(pos, pos[1:])]


def hill_tail_index(x, top_frac: float = 0.2, min_n: int = 5) -> tuple[float, float, int]:
    """Hill estimator on the largest ``top_frac`` of the values.

    Returns ``(alpha, gamma, k)``: ``gamma = mean(log x_(1..k)) - log x_(k+1)`` (order
    statistics descending, ``k = ceil(top_frac * n)``) and the tail index ``alpha = 1/gamma``
    (``inf`` if the top values are tied, NaN if fewer than ``min_n`` values)."""
    x = np.sort(np.asarray(x, dtype=np.float64))[::-1]
    n = x.shape[0]
    if n < min_n or np.any(x <= 0):
        return np.nan, np.nan, 0
    k = int(math.ceil(top_frac * n))
    k = max(1, min(k, n - 1))
    gamma = float(np.mean(np.log(x[:k])) - np.log(x[k]))
    alpha = float(1.0 / gamma) if gamma > 0 else float("inf")
    return alpha, gamma, k


def switch_interval_stats(labels) -> dict:
    """D5: statistics of the intervals between consecutive language switches.

    ``labels`` is one label sequence (list/array of 'E'/'D'/'N'), a list of such
    sequences, or a 2-D array ``[B, n]``; intervals are computed per sequence and pooled.

    Returns
    -------
    dict(intervals, n_intervals, n_switches, mean_interval, geometric_p, ks_stat, ks_plus,
    ks_minus, ks_p, tail_index, hill_gamma, hill_k).  ``geometric_p = 1/mean(intervals)``
    is the MLE of a geometric (memoryless Markov) interval law on ``{1, 2, ...}``.
    ``ks_stat`` is the Kolmogorov-Smirnov sup distance between the empirical CDF and the
    fitted geometric CDF, exact for the discrete support: ``ks_plus`` is
    :func:`scipy.stats.kstest` (``alternative="greater"``) against ``F(x)`` and
    ``ks_minus`` the ``"less"`` statistic against the left limit ``F(x-1)``, so that both
    step functions are compared at the same points (the two-sided ``kstest`` would compare
    the left limit of the empirical CDF with the right limit of ``F``).  ``ks_p`` is the
    Kolmogorov (continuous-case) p-value ``kstwo.sf(ks_stat, n)``, conservative for
    discrete data and with ``p`` estimated from the same sample -- indicative only.
    ``tail_index`` is the Hill estimator (alpha) on the top 20 % of the intervals (NaN if
    fewer than 5 intervals; heavy tails -> small alpha, ``inf`` if the top values are tied).
    """
    rows = _label_rows(labels)
    intervals: list[int] = []
    n_switches = 0
    for row in rows:
        n_switches += len(switch_positions(row))
        intervals.extend(inter_switch_intervals(row))
    iv = np.asarray(intervals, dtype=np.float64)
    out = dict(intervals=iv.astype(np.int64), n_intervals=int(iv.size), n_switches=int(n_switches),
               mean_interval=np.nan, geometric_p=np.nan, ks_stat=np.nan, ks_plus=np.nan,
               ks_minus=np.nan, ks_p=np.nan, tail_index=np.nan, hill_gamma=np.nan, hill_k=0)
    if iv.size == 0:
        return out
    mean = float(iv.mean())
    p = float(1.0 / mean)
    out["mean_interval"] = mean
    out["geometric_p"] = p
    ks_plus = float(sps.kstest(iv, lambda x: sps.geom.cdf(x, p), alternative="greater").statistic)
    ks_minus = float(sps.kstest(iv, lambda x: sps.geom.cdf(np.asarray(x) - 1, p), alternative="less").statistic)
    ks_stat = max(ks_plus, ks_minus, 0.0)
    out["ks_plus"], out["ks_minus"], out["ks_stat"] = ks_plus, ks_minus, ks_stat
    out["ks_p"] = float(sps.kstwo.sf(ks_stat, int(iv.size)))
    alpha, gamma, k = hill_tail_index(iv, 0.2, 5)
    out["tail_index"], out["hill_gamma"], out["hill_k"] = alpha, gamma, int(k)
    return out


# ======================================================================================
# D6 susceptibility, D7 fluctuation-response
# ======================================================================================
def susceptibility(fD_by_eps: dict) -> pd.DataFrame:
    """D6: mean, variance (over samples) and susceptibility ``chi = d mean / d eps`` of the
    German fraction per dose.  ``fD_by_eps`` maps ``eps`` to an array of per-sample
    ``f_D`` values.  ``chi`` uses central finite differences where possible (one-sided
    at the ends, :func:`numpy.gradient`; NaN with a single dose).  Columns:
    ``eps, n, mean, var, sd, chi`` sorted by ``eps``."""
    eps = np.array(sorted(float(e) for e in fD_by_eps), dtype=np.float64)
    rows = []
    for e in eps:
        key = next(k for k in fD_by_eps if float(k) == e)
        v = np.asarray(fD_by_eps[key], dtype=np.float64).reshape(-1)
        v = v[np.isfinite(v)]
        n = int(v.size)
        mean = float(v.mean()) if n else np.nan
        var = float(v.var(ddof=1)) if n > 1 else (0.0 if n == 1 else np.nan)
        rows.append(dict(eps=float(e), n=n, mean=mean, var=var, sd=float(np.sqrt(var)) if n else np.nan))
    df = pd.DataFrame(rows, columns=["eps", "n", "mean", "var", "sd"])
    if len(df) >= 2 and np.all(np.diff(eps) > 0):
        df["chi"] = np.gradient(df["mean"].to_numpy(dtype=np.float64), eps)
    else:
        df["chi"] = np.nan
    return df


def fdt_ratio(m0_samples, chi: float) -> float:
    """D7: effective temperature ``T_eff = Var_0(m) / chi`` of an order parameter ``m``
    (sample variance at zero dose over its susceptibility).  NaN if ``chi`` is 0 or not
    finite, or with fewer than two samples."""
    m = np.asarray(m0_samples, dtype=np.float64).reshape(-1)
    m = m[np.isfinite(m)]
    chi = float(chi)
    if m.size < 2 or not np.isfinite(chi) or chi == 0.0:
        return float("nan")
    return float(m.var(ddof=1) / chi)
