"""Stage G - local geometry under language switching and its trajectory
(DESIGN sections 4.4 / 4.5; PREREGISTRATION section 3, contrasts 7-9).

Realization clouds (DESIGN 4.4): per condition ``(family, K, C)`` every formulation of the
family (A-D) x up to ``Limits.g_max_masks`` admissible unit masks with exactly ``K``
switches (:func:`lsne.geometry.enumerate_unit_masks`; unit 1 / 10 English, unit 9 German)
realised at structure level ``C`` (:func:`lsne.stimuli.segment_masks_from_unit_mask`).
``K = 0`` has the formulations only.  Every realization passes the K=0 consistency check
of stage A (same target id and tail ids as the formulation's K=0 variant).

Tables
------
clouds            one row per realization: group, family, formulation, K, C, unit mask,
                  n_tokens, n_tokens_D
geometry          :func:`lsne.geometry.geometry_table` at every residual snapshot
                  (block inputs 0..L-1 + output of the last block), readouts ``mean`` /
                  ``last``, deflation levels ``Limits.g_deflate_k``: log_volume,
                  participation_ratio, disp_euclid / disp_mahal from the family's K=0 cloud
pair_ordering     G0 gate (contrast 7) per snapshot: the paper's core-and-spread
                  categories (exact_translation = formulation rendered all-E vs all-D,
                  same_meaning_same_lang = formulations A vs B .., other_family) under the
                  truncated Mahalanobis distance of the *prefix-mean* pooled states
                  (the shared English tail is excluded from the pooling), deflation k=5
g0_gate           one row: n_layers, n_ok (core_order_ok), frac_ok, pass_all (>= 50 % of
                  all snapshots), pass_middle (>= 50 % of the middle third),
                  degenerate (r_eff >= N-1 somewhere: the gate is then uninformative, see
                  :func:`lsne.geometry.pair_ordering`)
depth_trajectory  T1 (:func:`lsne.trajectory.depth_trajectory`): per condition and layer
                  the curves, increments, breathing index / amplitude, layer of maximal
                  contraction
contraction       contrast 8 (G1) per family x mixed condition x site x readout x k:
                  ``d_log_volume = log_volume(K, C) - log_volume(K=0)`` and
                  ``d_participation_ratio``, both *raw* (the geometry-table values; the
                  clouds differ in n, and the log-volume depends on n) and *n-matched*
                  (``Limits.g_n_subsample`` seeded subsets of the mixed cloud of the K=0
                  cloud's size, averaged; the confirmatory quantity)
contraction_boot  family cluster bootstrap of the n-matched contrasts per
                  (K, C, site, readout, k, quantity); the pre-registered rows are
                  K=8 / C1 / k=5 with readout ``mean`` (log-volume) and ``last`` (PR)
position_traj     T2 (:func:`lsne.trajectory.position_trajectory`) per group at the
                  first, middle and dossier snapshots (``Limits.g_position_sites``)
steering          T3 / contrast 9 (:func:`lsne.trajectory.steering_tracks_switching`) per
                  family: ``lang_dom`` injected at the dossier site (block input of the last
                  block, the bank of stage B) and at the depth site (``n_layers // 3``, the
                  axis fitted there), doses ``Limits.g_steer_eps`` (0.05 pre-registered),
                  cosine of the steering displacement with ``mean(K=8) - mean(K=0)`` per
                  snapshot vs ``cfg.n_random_dirs`` random directions
steering_boot     family bootstrap of ``cos_steer_vs_K`` and of its excess over the random
                  mean per (inject site, snapshot, eps), plus ``quantity =
                  "cos_at_max_disp_layer"``: the cosine at the snapshot of maximal discrete
                  displacement (the pre-registered layer; minimum effect cos >= 0.3)
coupling          T5 (:func:`lsne.trajectory.couple_with_dynamics`): Spearman correlation
                  over layers between ``delta_log_volume`` and the per-layer damage growth
                  rate ``lambda_l = ln D(l+1) - ln D(l)`` (stage D ``damage_profile``, depth
                  injection, ``lang_dom``) and the lens KL (stage D ``lens``); empty when
                  stage D has not run for this model under ``cfg.out_root``
"""
from __future__ import annotations

import time
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .. import stimuli
from ..config import RunConfig, HookSite
from ..directions import language_axis
from ..geometry import (deflate, enumerate_unit_masks, geometry_table, group_label, log_volume,
                        pair_ordering, participation_ratio, pool_states)
from ..model import ModelBundle, all_block_inputs, run_with_cache
from ..trajectory import (couple_with_dynamics, depth_trajectory, layer_index, position_trajectory,
                          steering_tracks_switching)
from . import common as C

PREREG_K, PREREG_C = 8, "C1"
PREREG_DEFLATE_K = 5
PREREG_STEER_EPS = 0.05
PREREG_MIN_COS = 0.3


# --------------------------------------------------------------------------------------
# realization clouds
# --------------------------------------------------------------------------------------
def select_masks(K: int, n_units: int, max_masks: Optional[int], seed: int) -> list[str]:
    """Admissible unit masks of ``K`` switches; a seeded subset of ``max_masks`` when
    there are more (K=4 has 56 for ten units, K=2 and K=8 exactly 8)."""
    masks = enumerate_unit_masks(int(K), int(n_units))
    if max_masks is not None and len(masks) > int(max_masks):
        rng = np.random.default_rng(int(seed) + 7919 * int(K))
        idx = np.sort(rng.choice(len(masks), size=int(max_masks), replace=False))
        masks = [masks[i] for i in idx]
    return masks


def realization_clouds(fams: Sequence, tokenizer, k_levels: Sequence[int], c_levels: Sequence[str],
                       max_masks: Optional[int], seed: int) -> tuple[dict, dict, pd.DataFrame]:
    """``(groups, mono_D, clouds)``: ``groups[(family_id, K, C)]`` = list of realization
    variants, ``mono_D[(family_id, formulation)]`` = the all-German rendering (G0 gate),
    ``clouds`` the tidy table (one row per realization; failed checks are listed with
    ``excluded=True``)."""
    groups: dict = {}
    mono_D: dict = {}
    rows: list[dict] = []
    by_id: dict[str, list] = {}
    for f in fams:
        if getattr(f, "kind", "standard") == "standard":
            by_id.setdefault(str(f.family_id), []).append(f)
    for fid, forms in by_id.items():
        for fam in forms:
            n = fam.n_units
            try:
                ref = stimuli.make_variant(fam, stimuli.segment_masks_from_unit_mask("E" * n, "C1"), 0, "C1", tokenizer)
            except Exception as e:  # noqa: BLE001
                rows.append(dict(group=group_label((fid, 0, "C1")), family=fid, formulation=fam.formulation, K=0, C="C1",
                                 unit_mask="E" * n, realization=0, n_tokens=np.nan, n_tokens_D=np.nan,
                                 excluded=True, problem=f"build: {e!r}"))
                continue
            try:
                vD = stimuli.make_variant(fam, stimuli.segment_masks_from_unit_mask("D" * n, "C1"), 0, "C1", tokenizer)
                probs = stimuli.check_variant(vD, ref)
                if not probs:
                    mono_D[(fid, str(fam.formulation))] = vD
            except Exception as e:  # noqa: BLE001
                warnings.warn(f"{fam.tag}: all-German rendering failed ({e!r}); G0 pair skipped")
            for K in k_levels:
                K = int(K)
                conds = [("C1", ["E" * n])] if K == 0 else [(str(Cc), select_masks(K, n, max_masks, seed)) for Cc in c_levels
                                                             if str(Cc) != "seg"]
                for Cc, masks in conds:
                    key = (fid, K, Cc)
                    for mi, um in enumerate(masks):
                        try:
                            seg = stimuli.segment_masks_from_unit_mask(um, Cc)
                            v = ref if K == 0 else stimuli.make_variant(fam, seg, K, Cc, tokenizer)
                            probs = stimuli.check_variant(v, ref)
                        except Exception as e:  # noqa: BLE001
                            probs = [f"build: {e!r}"]
                            v = None
                        rows.append(dict(group=group_label(key), family=fid, formulation=fam.formulation, K=K, C=Cc,
                                         unit_mask=um, realization=len(groups.get(key, [])),
                                         n_tokens=(int(v.n_tokens) if v is not None else np.nan),
                                         n_tokens_D=(int(v.annot.get("n_tokens_D", 0)) if v is not None else np.nan),
                                         excluded=bool(probs), problem="; ".join(probs)))
                        if v is not None and not probs:
                            groups.setdefault(key, []).append(v)
    cols = ["group", "family", "formulation", "K", "C", "unit_mask", "realization", "n_tokens", "n_tokens_D",
            "excluded", "problem"]
    return groups, mono_D, pd.DataFrame(rows, columns=cols)


def pooled_states(mb: ModelBundle, groups: dict, sites: Sequence[HookSite], readouts: Sequence[str]
                  ) -> dict[tuple[str, str], dict]:
    """``{(site.key, readout): {group key: [n, d]}}`` with one forward per realization.
    Readout ``mean_prefix`` pools the positions before ``tail_start`` only."""
    out: dict[tuple[str, str], dict] = {(s.key, ro): {} for s in sites for ro in readouts}
    for key, vs in groups.items():
        for v in vs:
            co = run_with_cache(mb, C.ids_tensor(mb, v), list(sites))
            for s in sites:
                h = co.resid[s.key][0].detach().float().cpu()
                for ro in readouts:
                    if ro == "mean_prefix":
                        pos = list(range(max(1, int(v.tail_start))))
                        x = pool_states(h, "mean", positions=pos)[0]
                    else:
                        x = pool_states(h, ro)[0]
                    out[(s.key, ro)].setdefault(key, []).append(x)
    return {k: {g: np.stack(v, 0) for g, v in d.items()} for k, d in out.items()}


# --------------------------------------------------------------------------------------
# G0 gate
# --------------------------------------------------------------------------------------
def g0_tables(mb: ModelBundle, groups: dict, mono_D: dict, sites: Sequence[HookSite], r: int, k: int
              ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pair ordering per snapshot on the prefix-mean states of every formulation's all-E
    (the K=0 realization) and all-D rendering; returns (pair_ordering rows, gate row)."""
    items: list[tuple] = []           # (family, formulation, lang, variant)
    for (fid, K, Cc), vs in groups.items():
        if K == 0:
            for v in vs:
                items.append((fid, str(v.formulation), "E", v))
    for (fid, form), v in mono_D.items():
        items.append((fid, form, "D", v))
    cols = ["site", "layer", "category", "n_pairs", "mean_mahal", "sd_mahal", "mean_cosine", "sd_cosine",
            "core_order_ok", "r_eff", "n_states", "degenerate"]
    gate_cols = ["n_layers", "n_ok", "frac_ok", "pass_all", "n_middle", "n_ok_middle", "frac_ok_middle",
                 "pass_middle", "degenerate", "n_states", "n_families", "k", "r", "r_requested"]
    if len(items) < 3:
        return pd.DataFrame(columns=cols), pd.DataFrame([dict(n_layers=0, n_ok=0, frac_ok=np.nan, pass_all=False,
                                                           n_middle=0, n_ok_middle=0, frac_ok_middle=np.nan,
                                                           pass_middle=False, degenerate=True, n_states=len(items),
                                                           n_families=0, k=k, r=r, r_requested=r)], columns=gate_cols)
    tmp = {("g0", i): [it[3]] for i, it in enumerate(items)}
    states = pooled_states(mb, tmp, sites, ("mean_prefix",))
    # A whitened (Mahalanobis) metric with rank r close to N-1 maps N points onto a regular
    # simplex: all pairwise distances become equal and any ordering is a floating-point accident.
    # The gate therefore caps the rank at (N-1)//4 (pre-registration item 7, sample-size note).
    r_requested = int(r)
    r = int(max(1, min(r_requested, (len(items) - 1) // 4)))
    labels = pd.DataFrame([dict(family=it[0], formulation=it[1], lang=it[2]) for it in items])
    rows: list[dict] = []
    oks: list[tuple[int, bool]] = []
    degenerate = False
    for s in sites:
        X = np.stack([states[(s.key, "mean_prefix")][("g0", i)][0] for i in range(len(items))], 0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            po = pair_ordering(X, labels, r=int(r), k=int(k))
        r_eff = int(po.attrs.get("r", 0))
        deg = bool(r_eff >= X.shape[0] - 1) or any("regular simplex" in str(x.message) for x in w)
        degenerate = degenerate or deg
        ok = bool(po["core_order_ok"].iloc[0]) if len(po) else False
        oks.append((layer_index(s.key), ok))
        for _, pr in po.iterrows():
            rows.append(dict(site=s.key, layer=layer_index(s.key), category=pr["category"], n_pairs=int(pr["n_pairs"]),
                             mean_mahal=float(pr["mean_mahal"]), sd_mahal=float(pr["sd_mahal"]),
                             mean_cosine=float(pr["mean_cosine"]), sd_cosine=float(pr["sd_cosine"]),
                             core_order_ok=(ok if pr["category"] == po["category"].iloc[0] else None),
                             r_eff=r_eff, n_states=int(X.shape[0]), degenerate=deg))
    layers = sorted(oks)
    n = len(layers)
    n_ok = sum(1 for _, ok in layers if ok)
    lo, hi = n // 3, max(n // 3 + 1, (2 * n + 2) // 3)
    middle = layers[lo:hi] if n >= 3 else layers
    n_ok_m = sum(1 for _, ok in middle if ok)
    # a degenerate metric makes the gate uninformative: no pass/fail verdict (None), never True
    pass_all = None if degenerate else bool(n and n_ok / n >= 0.5)
    pass_middle = None if degenerate else bool(middle and n_ok_m / len(middle) >= 0.5)
    gate = dict(n_layers=n, n_ok=n_ok, frac_ok=(n_ok / n) if n else np.nan, pass_all=pass_all,
                n_middle=len(middle), n_ok_middle=n_ok_m, frac_ok_middle=(n_ok_m / len(middle)) if middle else np.nan,
                pass_middle=pass_middle, degenerate=degenerate,
                n_states=len(items), n_families=int(labels["family"].nunique()), k=int(k), r=int(r), r_requested=r_requested)
    return pd.DataFrame(rows, columns=cols), pd.DataFrame([gate], columns=gate_cols)


# --------------------------------------------------------------------------------------
# contraction contrast (G1, contrast 8)
# --------------------------------------------------------------------------------------
def contraction_tables(geom: pd.DataFrame, states: dict, groups: dict, deflate_k: Sequence[int],
                       n_subsample: int, seed: int, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw and n-matched log-volume / PR contrasts of every mixed cloud against the K=0
    cloud of its family (see module notes) and their family bootstrap."""
    cols = ["family", "K", "C", "site", "layer", "readout", "k", "n_mixed", "n_ref", "log_volume_mixed", "log_volume_ref",
            "d_log_volume_raw", "d_log_volume", "pr_mixed", "pr_ref", "d_participation_ratio_raw",
            "d_participation_ratio", "n_subsample", "preregistered"]
    rows: list[dict] = []
    keys = list(groups.keys())
    rng = np.random.default_rng(int(seed))
    gidx = geom.set_index(["site", "readout", "k", "group"]) if len(geom) else None
    for (site_key, ro), by_group in states.items():
        if ro not in ("mean", "last"):
            continue
        order = [g for g in keys if g in by_group]
        X = np.concatenate([by_group[g] for g in order], 0)
        slices, start = {}, 0
        for g in order:
            n = by_group[g].shape[0]
            slices[g] = slice(start, start + n)
            start += n
        for k in deflate_k:
            Xk, _ = deflate(X, int(k))
            for g in order:
                fid, K, Cc = g
                if int(K) == 0:
                    continue
                ref = (fid, 0, "C1")
                if ref not in slices:
                    continue
                G0, GK = Xk[slices[ref]], Xk[slices[g]]
                n0, nK = G0.shape[0], GK.shape[0]
                lv0, pr0 = log_volume(G0), participation_ratio(G0)
                raw_lv = raw_pr = np.nan
                if gidx is not None:
                    try:
                        rm = gidx.loc[(site_key, ro, int(k), group_label(g))]
                        r0 = gidx.loc[(site_key, ro, int(k), group_label(ref))]
                        raw_lv = float(rm["log_volume"] - r0["log_volume"])
                        raw_pr = float(rm["participation_ratio"] - r0["participation_ratio"])
                    except KeyError:
                        pass
                if n0 < 2 or nK < n0:
                    lvK, prK, nsub = np.nan, np.nan, 0
                else:
                    nsub = 1 if nK == n0 else int(max(1, n_subsample))
                    lvs, prs = [], []
                    for _ in range(nsub):
                        idx = np.arange(nK) if nK == n0 else rng.choice(nK, size=n0, replace=False)
                        lvs.append(log_volume(GK[idx]))
                        prs.append(participation_ratio(GK[idx]))
                    with warnings.catch_warnings():          # a zero-variance cloud (block-0 last token) has PR = NaN
                        warnings.simplefilter("ignore", RuntimeWarning)
                        lvK, prK = float(np.nanmean(lvs)), float(np.nanmean(prs))
                rows.append(dict(family=fid, K=int(K), C=str(Cc), site=site_key, layer=layer_index(site_key), readout=ro,
                                 k=int(k), n_mixed=int(nK), n_ref=int(n0), log_volume_mixed=lvK, log_volume_ref=lv0,
                                 d_log_volume_raw=raw_lv, d_log_volume=lvK - lv0, pr_mixed=prK, pr_ref=pr0,
                                 d_participation_ratio_raw=raw_pr, d_participation_ratio=prK - pr0, n_subsample=nsub,
                                 preregistered=bool(int(K) == PREREG_K and str(Cc) == PREREG_C and int(k) == PREREG_DEFLATE_K)))
    con = pd.DataFrame(rows, columns=cols)
    bcols = ["quantity", "K", "C", "site", "layer", "readout", "k", "preregistered", "mean", "ci_low", "ci_high", "se",
             "n_clusters", "n_rows"]
    brows: list[dict] = []
    if len(con):
        for (K, Cc, site_key, ro, k), g in con.groupby(["K", "C", "site", "readout", "k"], sort=True):
            for q, ro_pre in (("d_log_volume", "mean"), ("d_participation_ratio", "last")):
                pre = bool(int(K) == PREREG_K and str(Cc) == PREREG_C and int(k) == PREREG_DEFLATE_K and ro == ro_pre)
                brows.append(C.boot_row(g, q, n_boot, seed, quantity=q, K=int(K), C=str(Cc), site=site_key,
                                        layer=layer_index(site_key), readout=ro, k=int(k), preregistered=pre))
    return con, pd.DataFrame(brows, columns=bcols)


# --------------------------------------------------------------------------------------
# T2 / T3 / T5
# --------------------------------------------------------------------------------------
def position_sites(mb: ModelBundle, n: int) -> list[HookSite]:
    """Dossier site, middle block input, block 0 input (the first ``n``)."""
    cands = [C.dossier_site(mb), HookSite(mb.n_layers // 2, "block_input"), HookSite(0, "block_input")]
    out: list[HookSite] = []
    for s in cands:
        if s.key not in {x.key for x in out}:
            out.append(s)
    return out[: max(1, int(n))]


def position_tables(mb: ModelBundle, groups: dict, sites: Sequence[HookSite], enabled: bool = True) -> pd.DataFrame:
    frames = []
    todo = [(key, s) for key in groups for s in sites]
    for key, s in C.progress(todo, "G T2 position", enabled=enabled):
        df = position_trajectory(mb, groups[key], s, n_prefix_bins=8)
        df.insert(0, "C", str(key[2]))
        df.insert(0, "K", int(key[1]))
        df.insert(0, "family", str(key[0]))
        df.insert(0, "group", group_label(key))
        df["layer"] = layer_index(s.key)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["group", "family", "K", "C", "site", "region", "index", "rel_pos", "pos_from_end",
                                     "rel_to_last_switch", "n", "log_volume", "participation_ratio", "delta_log_volume", "layer"])
    return pd.concat(frames, ignore_index=True)


def steering_directions(cfg: RunConfig, mb: ModelBundle, variants, bank, ctx, limits, original) -> dict[str, object]:
    """``lang_dom`` at the dossier site (stage-B bank) and at the depth site (stage D's
    depth bank when cached in ``ctx``, else fitted here)."""
    out: dict[str, object] = {}
    dsite, depth = C.dossier_site(mb), C.depth_site(mb)
    if "lang_dom" in bank:
        out[dsite.key] = bank["lang_dom"]
    if depth.key != dsite.key:
        dbank = ctx.get("bank_depth") if isinstance(ctx, dict) else None
        if isinstance(dbank, C.DirectionBank) and dbank.site == depth and "lang_dom" in dbank:
            out[depth.key] = dbank["lang_dom"]
        else:
            std = [v for v in variants if not C.is_original(v)]
            out[depth.key] = language_axis(mb, std, depth, pool="segment_last", denoise_pcs=1, include_tail=False)
    return out


def steering_tables(mb: ModelBundle, groups: dict, directions: dict[str, object], eps_list: Sequence[float],
                    n_random: int, seed: int, n_boot: int, enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    fams = sorted({k[0] for k in groups})
    frames = []
    todo = [(fid, key) for fid in fams for key in directions]
    for fid, site_key in C.progress(todo, "G T3 steering", enabled=enabled):
        K0 = groups.get((fid, 0, "C1"), [])
        K8 = groups.get((fid, PREREG_K, PREREG_C), [])
        if not K0 or not K8:
            continue
        site = HookSite(int(site_key.split(".")[0][1:]), site_key.split(".")[1])
        df = steering_tracks_switching(mb, K0, K8, directions[site_key], site, list(eps_list), n_random=int(n_random), seed=int(seed))
        df.insert(0, "direction", getattr(directions[site_key], "name", "lang_dom"))
        df.insert(0, "inject_site", site_key)
        df.insert(0, "family", fid)
        df["cos_excess"] = df["cos_steer_vs_K"] - df["cos_rand_mean"]
        df["preregistered"] = np.isclose(df["eps"], PREREG_STEER_EPS)
        # the snapshot of maximal discrete displacement (per family, inject site): the pre-registered layer
        d = df.drop_duplicates("snapshot").set_index("snapshot")["disp_mahal_K"]
        d = d[np.isfinite(d)]
        best = str(d.idxmax()) if len(d) else ""
        df["max_disp_snapshot"] = best
        df["at_max_disp_layer"] = df["snapshot"] == best
        frames.append(df)
    cols = ["family", "inject_site", "direction", "snapshot", "layer", "eps", "cos_steer_vs_K", "cos_rand_mean", "cos_rand_sd",
            "norm_steer", "norm_K", "disp_mahal_steer", "disp_mahal_K", "cos_dir_vs_K", "n_K0", "n_K", "n_random",
            "cos_excess", "preregistered", "max_disp_snapshot", "at_max_disp_layer"]
    if not frames:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=["quantity", "inject_site", "snapshot", "layer", "eps",
                                                                 "preregistered", "min_effect_met", "mean", "ci_low",
                                                                 "ci_high", "se", "n_clusters", "n_rows"])
    st = pd.concat(frames, ignore_index=True)[cols]
    brows = []
    for (isk, snap, eps), g in st.groupby(["inject_site", "snapshot", "eps"], sort=True):
        for q in ("cos_steer_vs_K", "cos_excess"):
            r = C.boot_row(g, q, n_boot, seed, quantity=q, inject_site=isk, snapshot=snap, layer=int(g["layer"].iloc[0]),
                           eps=float(eps), preregistered=bool(np.isclose(eps, PREREG_STEER_EPS)))
            r["min_effect_met"] = bool(np.isfinite(r["mean"]) and r["mean"] >= PREREG_MIN_COS) if q == "cos_steer_vs_K" else None
            brows.append(r)
    for (isk, eps), g in st[st["at_max_disp_layer"]].groupby(["inject_site", "eps"], sort=True):
        r = C.boot_row(g, "cos_steer_vs_K", n_boot, seed, quantity="cos_at_max_disp_layer", inject_site=isk, snapshot="per_family",
                       layer=-1, eps=float(eps), preregistered=bool(np.isclose(eps, PREREG_STEER_EPS)))
        r["min_effect_met"] = bool(np.isfinite(r["mean"]) and r["mean"] >= PREREG_MIN_COS)
        brows.append(r)
    boot = pd.DataFrame(brows, columns=["quantity", "inject_site", "snapshot", "layer", "eps", "preregistered", "min_effect_met",
                                        "mean", "ci_low", "ci_high", "se", "n_clusters", "n_rows"])
    return st, boot


def lambda_by_layer_from_damage(profile: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Per-layer growth rate ``lambda_l = ln D_tail(l+1) - ln D_tail(l)`` of the depth
    injection with ``lang_dom`` (mean over doses and variants per family, K, C)."""
    if profile is None or profile.empty:
        return None
    p = profile[(profile["inject"] == "depth") & (profile["dir_family"] == "lang_dom")]
    if p.empty:
        return None
    rows = []
    for (tag, eps), g in p.groupby(["tag", "eps"], sort=False):
        g = g.sort_values("layer")
        D = np.clip(g["D_tail"].to_numpy(dtype=float), 1e-12, None)
        lam = np.diff(np.log(D))
        for l, val in zip(g["layer"].to_numpy()[:-1], lam):
            rows.append(dict(family_id=g["family"].iloc[0], K=int(g["K"].iloc[0]), C=str(g["C"].iloc[0]), layer=int(l), lam=float(val)))
    df = pd.DataFrame(rows)
    return df.groupby(["family_id", "K", "C", "layer"], as_index=False)["lam"].mean()


def kl_by_layer_from_lens(lens: pd.DataFrame) -> Optional[pd.DataFrame]:
    if lens is None or lens.empty:
        return None
    df = lens.rename(columns={"family": "family_id"})[["family_id", "K", "C", "layer", "kl_last"]].copy()
    return df.groupby(["family_id", "K", "C", "layer"], as_index=False)["kl_last"].mean().rename(columns={"kl_last": "kl"})


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Local geometry (G0-G2) and trajectories (T1-T3, T5); see the module docstring."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    sub = C.Substages("G", verbose=limits.progress)
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    site = C.dossier_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original, sae=kw.get("sae"))
    fams = C.select_families(cfg)
    snaps = all_block_inputs(mb)

    groups, mono_D, clouds = realization_clouds(fams, mb.tokenizer, limits.g_k_levels, limits.g_c_levels,
                                                limits.g_max_masks, cfg.seed)
    if ctx is not None:
        ctx["g_groups"], ctx["g_mono_D"] = groups, mono_D
    empty = pd.DataFrame()
    geom = sub.run("geometry", lambda: geometry_table(mb, groups, snaps, readouts=("mean", "last"), r_mean=limits.g_r_mean,
                                                      r_last=limits.g_r_last, deflate_k=tuple(limits.g_deflate_k)), empty)
    po, gate = sub.run("g0_gate", lambda: g0_tables(mb, groups, mono_D, snaps, limits.g_r_mean, PREREG_DEFLATE_K), (empty, empty))
    depth = sub.run("depth_trajectory", lambda: depth_trajectory(geom) if len(geom) else empty, empty)
    states = sub.run("pooled_states", lambda: pooled_states(mb, groups, snaps, ("mean", "last")), {})
    con, con_boot = sub.run("contraction", lambda: contraction_tables(geom, states, groups, limits.g_deflate_k, limits.g_n_subsample,
                                                                      cfg.seed, limits.n_boot), (empty, empty))
    del states
    psites = position_sites(mb, limits.g_position_sites)
    pos = sub.run("position_trajectory", lambda: position_tables(mb, groups, psites, enabled=limits.progress), empty)
    dirs = sub.run("steering_directions", lambda: steering_directions(cfg, mb, variants, bank, ctx, limits, original), {})
    eps_list = list(dict.fromkeys([float(e) for e in limits.g_steer_eps]))
    steer, steer_boot = sub.run("steering", lambda: steering_tables(mb, groups, dirs, eps_list, cfg.n_random_dirs, cfg.seed,
                                                                    limits.n_boot, enabled=limits.progress), (empty, empty))

    def _coupling():
        lam = lambda_by_layer_from_damage(C.load_stage_table(cfg, "D", "damage_profile"))
        kl = kl_by_layer_from_lens(C.load_stage_table(cfg, "D", "lens"))
        if (lam is None and kl is None) or depth.empty:
            return pd.DataFrame(columns=["family_id", "K", "C", "readout", "k", "partner", "measure", "n", "rho", "pvalue"])
        return couple_with_dynamics(depth, lam, kl)
    coupling = sub.run("coupling", _coupling, empty)

    tables = {"clouds": clouds, "geometry": geom, "pair_ordering": po, "g0_gate": gate, "depth_trajectory": depth,
              "contraction": con, "contraction_boot": con_boot, "position_traj": pos, "steering": steer,
              "steering_boot": steer_boot, "coupling": coupling}
    sub.write_tracebacks(run_dir)
    return C.finish_stage(run_dir, "G", cfg, mb, tables, limits=limits, t0=t0, site=site, snapshots=[s.key for s in snaps],
                          n_groups=len(groups), groups={group_label(k): len(v) for k, v in groups.items()},
                          n_mono_D=len(mono_D), k_levels=list(limits.g_k_levels), c_levels=list(limits.g_c_levels),
                          max_masks=limits.g_max_masks, deflate_k=list(limits.g_deflate_k),
                          position_sites=[s.key for s in psites], steering_sites={k: getattr(d, "name", "") for k, d in dirs.items()},
                          steer_eps=eps_list, g0_gate=(gate.iloc[0].to_dict() if len(gate) else None),
                          stage_d_available=C.stage_run_dir(cfg, "D") is not None, substages=sub.report(),
                          prereg=dict(contrast_7="g0_gate.pass_all / pass_middle (core_order_ok in >= 50% of snapshots)",
                                      contrast_8="contraction_boot rows with preregistered=True",
                                      contrast_9="steering_boot quantity=cos_at_max_disp_layer, eps=0.05"))
