"""Stage H - lambda dynamics: Biggio et al. Algorithm 1 in activation space
(DESIGN section 4.6; PREREGISTRATION section 3, contrasts 10-11).

Point ``x`` = the residual at the dossier site (block input of the last block) and the
last context position of a K=0 variant (one start per formulation, up to
``Limits.h_max_starts`` per family).  Target-class density = KDE over the *real*
last-position states of the K=8/C1 realization cloud of all selected families (stage G's
clouds, rebuilt here when stage G did not run) in the Mahalanobis-whitened coordinates of
the pooled K=0 + K=8 states (rank ``Limits.h_cov_rank``), ``k_nearest = Limits.h_knn``
(the paper's 50), median bandwidth.  Discriminants (``Limits.h_discriminants``):

* ``target_logit`` (b): ``g = -log p(target | h) + log p_0(target) + c`` with
  ``c = Limits.h_cross_nats`` - the boundary ``g < 0`` is crossed when the benefit
  ``B = log p - log p_0`` exceeds ``c`` nats;
* ``probe`` (a): ``g = -(w . h) + b`` with ``w`` the stage-B language probe
  (positive = German) and ``b`` the midpoint between the start's and the target cloud
  mean's projection, so ``g = 0`` is the half-way boundary towards the switched cloud;
* ``entropy`` (b'): ``g = -H(h) + H_0 + c`` (crossed when the entropy rises by ``c`` nats);
* ``sae_feature`` (c): ``g = -relu(pre_f(h)) + b`` for the most German-selective feature
  of the bank's SAE (or, for a synthetic placeholder SAE, the feature with the largest
  target-vs-start activation gap); ``b`` = midpoint between the start activation and the
  target cloud's median activation.  Skipped when no SAE is available.

Sweep ``lam = lam_rel * lambda_scale(range(g), n_eff, bw)`` for ``lam_rel`` in
``Limits.h_lambdas_rel`` (DESIGN: 0, 0.1, 0.3, 1, 3, 10, 30); the range of ``g`` is
measured over the start and up to 16 target points.  Paths: unit steps of
``Limits.h_step_rel * ||h0||`` inside the l2 ball ``Limits.h_dmax_rel * ||h0||``, at most
``Limits.h_n_steps`` steps, stop when F no longer decreases.

Tables
------
lambda_paths     one row per (start, discriminant, lam_rel): path geometry
                 (:func:`lsne.lambda_dynamics.trajectory_geometry`), data support of the
                 endpoint relative to the target cloud (:func:`manifold_stats`:
                 mahal_to_ref_mean, knn_mean_dist, local_log_volume, sae_fvu,
                 n_shared_features_with_ref), the next-token effect of inserting the
                 endpoint (:func:`endpoint_effects`: logp_target, entropy, p_freq, B) and
                 the KL of the endpoint prediction from the unperturbed one
lambda_steps     per-step curves g, density, F, dist_from_start of every path
lambda_corr      Spearman rho of every numeric path quantity with ``lam`` per start x
                 discriminant (:func:`geometry_lambda_correlation`); ``preregistered``
                 marks discriminant ``target_logit`` x {sae_fvu, steps_to_cross,
                 mahal_to_ref_mean} (contrast 10 i-iii; expected rho < 0 for i and iii)
lambda_boot      family cluster bootstrap of rho per (discriminant, quantity)
lambda_features  :func:`feature_lambda_correlation` per start (SAE latents vs lam);
                 ``candidate`` = |rho| >= 0.7 (contrast 10: frozen candidates)
lambda_features_summary  per (discriminant, feature): mean rho, n_starts, frac_candidate,
                 n_active_only_lambda0 / _pos
lambda_original  contrast 11: the same sweep from the State-2143 context (h_207 at the
                 dossier site, target 22905, discriminant target_logit); target cloud =
                 the original's K>0 variants plus the standard K=8/C1 cloud (flagged in
                 ``target_cloud``); ``B_lambda0``, ``B_lambda3`` (lam_rel = 3) and
                 ``same_sign`` on every row.  Empty when the anchor is unavailable.
Every row carries ``sae_source`` (``hub:...``, ``synthetic_random`` = placeholder,
``none``).
"""
from __future__ import annotations

import time
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..config import RunConfig, HookSite
from ..geometry import layer_covariance
from ..lambda_dynamics import (KDEMimicry, endpoint_effects, feature_lambda_correlation,
                               geometry_lambda_correlation, lambda_scale, lambda_sweep, make_discriminant,
                               manifold_stats, partial_forward, trajectory_geometry)
from ..model import ModelBundle, next_token_stats, run_with_cache
from . import common as C
from .stage_g_geometry import realization_clouds

PREREG_DISCRIMINANT = "target_logit"
# expected sign of rho (0 = open). Contrast 10(ii) uses the censored step count: uncrossed paths
# are set to n_steps+1 rather than dropped (the raw column 'steps_to_cross' is kept as secondary).
PREREG_QUANTITIES = {"sae_fvu": -1, "steps_to_cross_censored": 0, "mahal_to_ref_mean": -1}
PREREG_CANDIDATE_RHO = 0.7
PREREG_LAMBDA_REL_B = 3.0
N_RANGE_POINTS = 16


# --------------------------------------------------------------------------------------
# target cloud and start states
# --------------------------------------------------------------------------------------
def k8_cloud(cfg: RunConfig, mb: ModelBundle, limits: C.Limits, ctx: Optional[dict]) -> tuple[list, list]:
    """``(K8/C1 realization variants, K=0 variants)`` of the selected families: stage G's
    clouds from ``ctx`` when present, else rebuilt (K in {0, 8}, C1, ``g_max_masks``)."""
    groups = ctx.get("g_groups") if isinstance(ctx, dict) else None
    if not groups:
        fams = C.select_families(cfg)
        groups, _m, _c = realization_clouds(fams, mb.tokenizer, (0, 8), ("C1",), limits.g_max_masks, cfg.seed)
    k8 = [v for (fid, K, Cc), vs in groups.items() if int(K) == 8 and str(Cc) == "C1" for v in vs]
    k0 = [v for (fid, K, Cc), vs in groups.items() if int(K) == 0 for v in vs]
    return k8, k0


def last_states(mb: ModelBundle, variants: Sequence, site: HookSite) -> torch.Tensor:
    return torch.stack([C.last_state(mb, v, site) for v in variants], 0) if variants else torch.zeros(0, mb.d_model)


def starts_of(variants: Sequence, max_per_family: int) -> list:
    """K=0 standard variants, at most ``max_per_family`` per family id (formulation order)."""
    out: list = []
    count: dict[str, int] = {}
    for v in sorted(C.k_subset(variants, 0), key=lambda v: (str(v.family_id), str(v.formulation))):
        fid = str(v.family_id)
        if count.get(fid, 0) >= int(max_per_family):
            continue
        count[fid] = count.get(fid, 0) + 1
        out.append(v)
    return out


def select_sae_feature(sae, bank, H_target: torch.Tensor, H_start: torch.Tensor) -> Optional[int]:
    """Most German-selective feature of the bank (``features_D[0]``), else the feature with
    the largest mean activation gap target - start (synthetic placeholder SAEs)."""
    info = bank.meta.get("sae", {}) if bank is not None and isinstance(bank.meta, dict) else {}
    feats = info.get("features_D") or []
    if feats:
        return int(feats[0])
    if H_target.shape[0] == 0 or H_start.shape[0] == 0:
        return None
    with torch.no_grad():
        _, _, pre_t = sae.encode(H_target.float())
        _, _, pre_s = sae.encode(H_start.float())
    gap = torch.relu(pre_t).mean(0) - torch.relu(pre_s).mean(0)
    return int(torch.argmax(gap))


# --------------------------------------------------------------------------------------
# discriminants and the sweep
# --------------------------------------------------------------------------------------
def build_discriminants(mb: ModelBundle, kinds: Sequence[str], site: HookSite, ids: torch.Tensor, pos: int,
                        target_id: int, h0: torch.Tensor, bank, H_target: torch.Tensor, sae, feature: Optional[int],
                        cross_nats: float) -> dict[str, object]:
    out: dict[str, object] = {}
    with torch.no_grad():
        base = mb.model(input_ids=ids.to(mb.device), use_cache=False).logits[0, -1].float()
    logp = torch.log_softmax(base, -1)
    logp0 = float(logp[int(target_id)])
    H0 = float(-(logp.exp() * logp).sum())
    for kind in kinds:
        try:
            if kind == "target_logit":
                out[kind] = make_discriminant(mb, "target_logit", site, input_ids=ids, position=pos, target_id=int(target_id),
                                              b=logp0 + float(cross_nats))
            elif kind == "entropy":
                out[kind] = make_discriminant(mb, "entropy", site, input_ids=ids, position=pos, b=H0 + float(cross_nats))
            elif kind == "probe":
                if bank is None or "lang_probe" not in bank:
                    continue
                w = bank["lang_probe"].vec.float()
                mT = float(H_target.float().mean(0) @ w) if H_target.shape[0] else float(h0.float() @ w) + 1.0
                b = 0.5 * (float(h0.float() @ w) + mT)
                out[kind] = make_discriminant(mb, "probe", site, w=-w, b=b)
                out[kind].meta.update(direction="lang_probe", boundary="midpoint start/target-mean projection")
            elif kind == "sae_feature":
                if sae is None or feature is None:
                    continue
                W = sae.W_enc[int(feature)].detach().float().cpu()
                b_enc = float(sae.b_enc[int(feature)].detach().float().cpu())
                b_dec = sae.b_dec.detach().float().cpu()
                a0 = max(float((h0.float() - b_dec) @ W + b_enc), 0.0)
                aT = float(np.median(torch.relu((H_target.float() - b_dec) @ W + b_enc).numpy())) if H_target.shape[0] else a0 + 1.0
                b = 0.5 * (a0 + aT) if aT > a0 else a0 + 1.0
                out[kind] = make_discriminant(mb, "sae_feature", site, sae=sae, feature=int(feature), b=b)
            else:
                warnings.warn(f"unknown discriminant kind {kind!r} skipped")
        except Exception as e:  # noqa: BLE001 - one discriminant must not kill the start
            warnings.warn(f"discriminant {kind!r} skipped: {e!r}")
    return out


def g_range(disc, h0: torch.Tensor, H_target: torch.Tensor, n: int, seed: int) -> float:
    """Range of ``g`` over the start and up to ``n`` seeded target points (>= 1e-6)."""
    vals = [float(disc.fn(h0))]
    if H_target.shape[0]:
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(H_target.shape[0], size=min(int(n), H_target.shape[0]), replace=False)
        for i in idx:
            vals.append(float(disc.fn(H_target[int(i)])))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(max(np.ptp(vals) if vals.size else 0.0, 1e-6))


def sweep_start(mb: ModelBundle, v, site: HookSite, kinds: Sequence[str], bank, kde: KDEMimicry, H_target: torch.Tensor,
                U_r, lam_r, sae, feature: Optional[int], limits: C.Limits, seed: int, freq_ids, sae_source: str,
                context: str = "standard") -> tuple[list[dict], list[dict], dict[str, dict]]:
    """Lambda sweep from one start variant: (path rows, step rows, endpoints per discriminant)."""
    ids = C.ids_tensor(mb, v)
    T = int(ids.shape[1])
    pos = T - 1
    h0 = C.last_state(mb, v, site)
    norm0 = float(h0.norm())
    discs = build_discriminants(mb, kinds, site, ids, pos, int(v.target_id), h0, bank, H_target, sae, feature, limits.h_cross_nats)
    f = partial_forward(mb, ids, site, pos)
    with torch.no_grad():
        base_logits = mb.model(input_ids=ids.to(mb.device), use_cache=False).logits[0, -1].float().cpu()
    logp_base = torch.log_softmax(base_logits, -1)
    idc = dict(context=context, tag=v.tag, family=str(v.family_id), formulation=str(v.formulation), K=int(v.K), C=str(v.C),
               site=site.key, position=pos, n_tokens=T, target_id=int(v.target_id), norm_h0=norm0, sae_source=sae_source)
    rows, steps, ends = [], [], {}
    for kind, disc in discs.items():
        rng_g = g_range(disc, h0, H_target, N_RANGE_POINTS, seed)
        scale = lambda_scale(rng_g, kde.n_eff, kde.bandwidth)
        lambdas = [float(r) * scale for r in limits.h_lambdas_rel]
        sweep = lambda_sweep(h0, disc, kde, lambdas, step=limits.h_step_rel * norm0, d_max=limits.h_dmax_rel * norm0,
                             n_steps=int(limits.h_n_steps), eps_stop=1e-6, metric="l2")
        endpoints = {lam: tr.endpoint for lam, tr in sweep.items()}
        ends[kind] = endpoints
        eff = endpoint_effects(mb, ids, site, pos, endpoints, int(v.target_id), freq_ids=freq_ids).set_index("lam")
        P = torch.stack([tr.endpoint for tr in sweep.values()], 0)
        ms = manifold_stats(P, H_target, U_r=U_r, lam_r=lam_r, sae=sae, k=5) if H_target.shape[0] else None
        for i, (lam_rel, (lam, tr)) in enumerate(zip(limits.h_lambdas_rel, sweep.items())):
            geo = trajectory_geometry(tr)
            with torch.no_grad():
                lp = torch.log_softmax(f(tr.endpoint).cpu(), -1)
            kl_from_base = float((logp_base.exp() * (logp_base - lp)).sum())
            e = eff.loc[lam]
            row = dict(idc, discriminant=kind, lam_rel=float(lam_rel), lam=float(lam), lambda_unit=scale, g_range=rng_g,
                       kde_bandwidth=float(kde.bandwidth), kde_n_eff=int(kde.n_eff), n_target=int(H_target.shape[0]),
                       **{k: (np.nan if val is None else val) for k, val in geo.items() if k != "lam"},
                       crossed=bool(tr.steps_to_cross is not None),
                       # censored at n_steps+1: a path that never crosses (the lambda=0 'stands still'
                       # signature, DESIGN 4.6) ranks after every crossing path instead of being dropped
                       steps_to_cross_censored=float(tr.steps_to_cross if tr.steps_to_cross is not None else int(limits.h_n_steps) + 1),
                       logp_target=float(e["logp_target"]), p_target=float(e["p_target"]), entropy=float(e["entropy"]),
                       p_freq=float(e["p_freq"]), B=float(e["B"]), kl_from_base=kl_from_base,
                       logp_target_unperturbed=float(eff.attrs.get("logp_target_unperturbed", np.nan)))
            if ms is not None:
                m = ms.iloc[i]
                row.update(mahal_to_ref_mean=float(m["mahal_to_ref_mean"]), euclid_to_ref_mean=float(m["euclid_to_ref_mean"]),
                           knn_mean_dist=float(m["knn_mean_dist"]), knn_mean_mahal=float(m["knn_mean_mahal"]),
                           local_log_volume=float(m["local_log_volume"]), sae_fvu=float(m["sae_fvu"]),
                           n_shared_features_with_ref=(np.nan if pd.isna(m["n_shared_features_with_ref"]) else int(m["n_shared_features_with_ref"])),
                           n_active_features=(np.nan if pd.isna(m["n_active_features"]) else int(m["n_active_features"])))
            rows.append(row)
            for s in range(tr.points.shape[0]):
                steps.append(dict(context=context, tag=v.tag, family=str(v.family_id), formulation=str(v.formulation),
                                  discriminant=kind, lam_rel=float(lam_rel), step=s, g=float(tr.g[s]),
                                  density=float(tr.density[s]), F=float(tr.F[s]), dist_from_start=float(tr.dist_from_start[s])))
    return rows, steps, ends


PATH_COLS = ["context", "tag", "family", "formulation", "K", "C", "site", "position", "n_tokens", "target_id", "norm_h0",
             "sae_source", "discriminant", "lam_rel", "lam", "lambda_unit", "g_range", "kde_bandwidth", "kde_n_eff", "n_target",
             "path_length", "net_displacement", "tortuosity", "mean_turning_angle_deg", "n_steps", "steps_to_cross", "steps_to_cross_censored",
             "final_g", "final_density", "final_F", "dist_from_start", "converged", "reason", "initial_g", "g_drop", "crossed",
             "logp_target", "p_target", "entropy", "p_freq", "B", "kl_from_base", "logp_target_unperturbed",
             "mahal_to_ref_mean", "euclid_to_ref_mean", "knn_mean_dist", "knn_mean_mahal", "local_log_volume", "sae_fvu",
             "n_shared_features_with_ref", "n_active_features"]
STEP_COLS = ["context", "tag", "family", "formulation", "discriminant", "lam_rel", "step", "g", "density", "F", "dist_from_start"]
CORR_QUANTITIES = ["path_length", "net_displacement", "tortuosity", "mean_turning_angle_deg", "n_steps", "steps_to_cross",
                   "steps_to_cross_censored",
                   "final_g", "final_density", "dist_from_start", "crossed", "logp_target", "entropy", "p_freq", "B",
                   "kl_from_base", "mahal_to_ref_mean", "euclid_to_ref_mean", "knn_mean_dist", "knn_mean_mahal",
                   "local_log_volume", "sae_fvu", "n_shared_features_with_ref", "n_active_features"]


def correlation_tables(paths: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    ccols = ["tag", "family", "formulation", "discriminant", "quantity", "rho", "pvalue", "n", "preregistered", "expected_sign"]
    if paths.empty:
        return pd.DataFrame(columns=ccols), pd.DataFrame(columns=["discriminant", "quantity", "preregistered", "expected_sign",
                                                                  "sign_as_expected", "mean", "ci_low", "ci_high", "se",
                                                                  "n_clusters", "n_rows"])
    rows = []
    for (tag, fid, form, kind), g in paths.groupby(["tag", "family", "formulation", "discriminant"], sort=True):
        sub = g[["lam"] + [c for c in CORR_QUANTITIES if c in g.columns]].copy()
        sub["crossed"] = sub["crossed"].astype(float)
        cr = geometry_lambda_correlation(sub)
        for _, r in cr.iterrows():
            pre = bool(kind == PREREG_DISCRIMINANT and r["quantity"] in PREREG_QUANTITIES)
            rows.append(dict(tag=tag, family=fid, formulation=form, discriminant=kind, quantity=r["quantity"], rho=float(r["rho"]),
                             pvalue=float(r["pvalue"]), n=int(r["n"]), preregistered=pre,
                             expected_sign=(PREREG_QUANTITIES.get(r["quantity"], 0) if pre else 0)))
    corr = pd.DataFrame(rows, columns=ccols)
    brows = []
    for (kind, q), g in corr.groupby(["discriminant", "quantity"], sort=True):
        pre = bool(kind == PREREG_DISCRIMINANT and q in PREREG_QUANTITIES)
        exp = PREREG_QUANTITIES.get(q, 0) if pre else 0
        r = C.boot_row(g, "rho", n_boot, seed, discriminant=kind, quantity=q, preregistered=pre, expected_sign=exp)
        r["sign_as_expected"] = (bool(np.isfinite(r["mean"]) and np.sign(r["mean"]) == exp) if exp else None)
        brows.append(r)
    boot = pd.DataFrame(brows, columns=["discriminant", "quantity", "preregistered", "expected_sign", "sign_as_expected",
                                        "mean", "ci_low", "ci_high", "se", "n_clusters", "n_rows"])
    return corr, boot


def feature_tables(ends_by_start: list[tuple[dict, dict]], sae) -> tuple[pd.DataFrame, pd.DataFrame]:
    fcols = ["tag", "family", "formulation", "discriminant", "feature", "rho", "pvalue", "act_at_min_lambda", "act_at_max_lambda",
             "active_only_lambda0", "active_only_lambda_pos", "n_active", "max_act", "candidate"]
    scols = ["discriminant", "feature", "n_starts", "rho_mean", "rho_min", "rho_max", "frac_candidate", "n_active_only_lambda0",
             "n_active_only_lambda_pos", "candidate_all"]
    if sae is None or not ends_by_start:
        return pd.DataFrame(columns=fcols), pd.DataFrame(columns=scols)
    frames = []
    for idc, ends in ends_by_start:
        for kind, endpoints in ends.items():
            df = feature_lambda_correlation(endpoints, sae)
            if df.empty:
                continue
            df.insert(0, "discriminant", kind)
            df.insert(0, "formulation", idc["formulation"])
            df.insert(0, "family", idc["family"])
            df.insert(0, "tag", idc["tag"])
            df["candidate"] = df["rho"].abs() >= PREREG_CANDIDATE_RHO
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=fcols), pd.DataFrame(columns=scols)
    feats = pd.concat(frames, ignore_index=True)[fcols]
    rows = []
    for (kind, f), g in feats.groupby(["discriminant", "feature"], sort=True):
        rho = g["rho"].to_numpy(dtype=float)
        ok = np.isfinite(rho)
        rows.append(dict(discriminant=kind, feature=int(f), n_starts=int(len(g)),
                         rho_mean=float(np.nanmean(rho)) if ok.any() else np.nan, rho_min=float(np.nanmin(rho)) if ok.any() else np.nan,
                         rho_max=float(np.nanmax(rho)) if ok.any() else np.nan,
                         frac_candidate=float(g["candidate"].mean()), n_active_only_lambda0=int(g["active_only_lambda0"].sum()),
                         n_active_only_lambda_pos=int(g["active_only_lambda_pos"].sum()), candidate_all=bool(g["candidate"].all())))
    return feats, pd.DataFrame(rows, columns=scols)


def original_table(mb: ModelBundle, original, site: HookSite, kde_std: KDEMimicry, H_std: torch.Tensor, limits: C.Limits,
                   seed: int, freq_ids, sae, sae_source: str, cov_rank: int) -> pd.DataFrame:
    """Contrast 11: B(x)(lambda) for the State-2143 context (discriminant target_logit)."""
    cols = PATH_COLS + ["target_cloud", "n_target_orig", "B_lambda0", "B_lambda3", "same_sign"]
    if original is None:
        return pd.DataFrame(columns=cols)
    k0 = [v for v in original.variants if int(v.K) == 0]
    kpos = [v for v in original.variants if int(v.K) > 0]
    if not k0:
        return pd.DataFrame(columns=cols)
    H_orig = last_states(mb, kpos, site)
    H_target = torch.cat([H_orig, H_std], 0) if H_std.shape[0] else H_orig
    if H_target.shape[0] < 2:
        return pd.DataFrame(columns=cols)
    H_start = last_states(mb, k0, site)
    U_r, lam_r = layer_covariance(torch.cat([H_start, H_target], 0).numpy(), int(cov_rank))
    kde = KDEMimicry(H_target, bandwidth="median", metric="mahalanobis", U_r=U_r, lam_r=lam_r,
                     k_nearest=min(int(limits.h_knn), int(H_target.shape[0])), seed=seed)
    rows, _steps, _ends = sweep_start(mb, k0[0], site, ("target_logit",), None, kde, H_target, U_r, lam_r, sae, None, limits,
                                      seed, freq_ids, sae_source, context="original_2143")
    df = pd.DataFrame(rows, columns=PATH_COLS)
    df["target_cloud"] = "orig_K>0 + std_K8C1"
    df["n_target_orig"] = int(H_orig.shape[0])
    b0 = df.loc[np.isclose(df["lam_rel"], 0.0), "B"]
    b3 = df.loc[np.isclose(df["lam_rel"], PREREG_LAMBDA_REL_B), "B"]
    B0 = float(b0.iloc[0]) if len(b0) else np.nan
    B3 = float(b3.iloc[0]) if len(b3) else np.nan
    df["B_lambda0"], df["B_lambda3"] = B0, B3
    df["same_sign"] = bool(np.isfinite(B0) and np.isfinite(B3) and np.sign(B0) == np.sign(B3))
    return df[cols]


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Lambda sweeps from the K=0 starts (and the State-2143 context); see the module docstring."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    sub = C.Substages("H", verbose=limits.progress)
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    site = C.dossier_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original, sae=kw.get("sae"))
    sae, sae_source = C.get_sae(mb, bank, limits, kw)
    freq_ids = bank.freq_ids or None
    empty = pd.DataFrame()

    k8, k0_cloud = sub.run("clouds", lambda: k8_cloud(cfg, mb, limits, ctx), ([], []))
    H_target = sub.run("target_states", lambda: last_states(mb, k8, site), torch.zeros(0, mb.d_model))
    H_k0 = sub.run("start_states", lambda: last_states(mb, k0_cloud, site), torch.zeros(0, mb.d_model))
    starts = starts_of(variants, limits.h_max_starts)
    paths_rows: list[dict] = []
    step_rows: list[dict] = []
    ends_by_start: list[tuple[dict, dict]] = []
    kde = U_r = lam_r = feature = None
    if H_target.shape[0] >= 2:
        U_r, lam_r = layer_covariance(torch.cat([H_k0, H_target], 0).numpy(), int(limits.h_cov_rank))
        kde = KDEMimicry(H_target, bandwidth="median", metric="mahalanobis", U_r=U_r, lam_r=lam_r,
                         k_nearest=min(int(limits.h_knn), int(H_target.shape[0])), seed=cfg.seed)
        feature = select_sae_feature(sae, bank, H_target, H_k0) if sae is not None else None
        kinds = [k for k in limits.h_discriminants if k != "sae_feature" or sae is not None]
        for v in C.progress(starts, "H lambda sweeps", enabled=limits.progress):
            def _one(v=v):
                return sweep_start(mb, v, site, kinds, bank, kde, H_target, U_r, lam_r, sae, feature, limits, cfg.seed,
                                   freq_ids, sae_source)
            res = sub.run(f"sweep:{v.tag}", _one, None)
            if res is None:
                continue
            rows, steps, ends = res
            paths_rows.extend(rows)
            step_rows.extend(steps)
            ends_by_start.append((rows[0] if rows else dict(tag=v.tag, family=v.family_id, formulation=v.formulation), ends))
    else:
        sub.errors["target_cloud"] = f"fewer than two K=8/C1 target states ({int(H_target.shape[0])}); no sweep"
    paths = pd.DataFrame(paths_rows, columns=PATH_COLS)
    steps_df = pd.DataFrame(step_rows, columns=STEP_COLS)
    corr, boot = sub.run("correlations", lambda: correlation_tables(paths, limits.n_boot, cfg.seed), (empty, empty))
    feats, feats_summary = sub.run("features", lambda: feature_tables(ends_by_start, sae), (empty, empty))
    orig_df = sub.run("original", lambda: original_table(mb, original, site, kde, H_target, limits, cfg.seed, freq_ids, sae,
                                                         sae_source, limits.h_cov_rank), empty)

    tables = {"lambda_paths": paths, "lambda_steps": steps_df, "lambda_corr": corr, "lambda_boot": boot,
              "lambda_features": feats, "lambda_features_summary": feats_summary, "lambda_original": orig_df}
    sub.write_tracebacks(run_dir)
    return C.finish_stage(run_dir, "H", cfg, mb, tables, limits=limits, t0=t0, site=site, sae_source=sae_source,
                          sae_feature=feature, n_starts=len(starts), n_target=int(H_target.shape[0]), n_k0_cloud=int(H_k0.shape[0]),
                          discriminants=list(limits.h_discriminants), lambdas_rel=list(limits.h_lambdas_rel),
                          kde=(dict(bandwidth=kde.bandwidth, median_distance=kde.median_distance, n_eff=kde.n_eff, r=kde.r,
                                    metric=kde.metric) if kde is not None else None),
                          path=dict(n_steps=limits.h_n_steps, step_rel=limits.h_step_rel, dmax_rel=limits.h_dmax_rel,
                                    cross_nats=limits.h_cross_nats, ball_metric="l2"),
                          original=(original.info() if original is not None else None), substages=sub.report(),
                          prereg=dict(contrast_10="lambda_boot rows with preregistered=True (discriminant target_logit; "
                                                  "sae_fvu, steps_to_cross_censored (uncrossed paths at n_steps+1; raw steps_to_cross secondary), mahal_to_ref_mean); "
                                                  "lambda_features candidate=|rho|>=0.7",
                                      contrast_11="lambda_original: B_lambda0 vs B_lambda3, same_sign"))
