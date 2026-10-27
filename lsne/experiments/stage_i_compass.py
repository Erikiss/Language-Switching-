"""Stage I - domain compass, domain axes with counter-steering, head attribution and
steering at the worst head (DESIGN section 4.7; PREREGISTRATION contrasts 12-14).

Tables
------
compass            contrast 12: domain profile (:func:`lsne.atlas.domain_profile`) of the
                   State-2143 state ``h_207`` itself (``direction = "state"``) and of a
                   step of ``eps = 0.05`` along ``ent``, ``lang_dom``, ``freq`` and the
                   domain axes from that state, z against ``Limits.i_n_random_profile``
                   random directions of the same dose (32 pre-registered); one row per
                   (direction, domain) with ``rank`` (by z) and ``is_top``.  ``h_207`` is
                   the last context position of the verified anchor at the dossier site;
                   without the anchor the first K=0 variant is used (``state_source``).
                   SAE feature profiles (:func:`sae_feature_domain_profile`) are appended
                   for the bank's ``sae_*`` directions when an SAE is in use.
compass_summary    per direction: ``top_domain``, ``law_z``, ``law_rank``, ``law_is_top``
                   (the atlas expectation "law" for ``ent``; documentation, no test)
atlas_overlap      :func:`lsne.atlas.atlas_domain_overlap` of the client's stored atlas
                   extraction (``data/atlas``, ``kw['atlas_dir']``); empty without files
domain_axes        ``dom_law``, ``dom_culture_politics`` (culture_values + politics),
                   ``dom_protein_misfolding`` vs the ``everyday`` contrast at the dossier
                   site (:func:`lsne.atlas.domain_axis`; also written to
                   ``directions_domain.npz``) with their cosines to the bank directions
direction_domain   contrast 13 per variant (K=0 and K=8/C1) x direction x eps: B(x) for
                   ``dom_culture_politics_cs`` (counter-steer: the ``dom_law`` component
                   projected out of h, then the ``dom_culture_politics`` dose; the state is
                   written with ``Injector(mode="replace")``), ``dom_culture_politics``,
                   ``dom_law``, ``lang_dom`` (added doses) and ``rand`` (mean over the bank's
                   random directions); ``family_domain`` = pre-registered family group
direction_domain_T T(K=8, C1) = logp_0(K=8) - logp_0(K=0) per formulation with its family group
direction_domain_boot  family bootstrap of B per (family_domain, direction, eps) and of the
                   pre-registered contrast ``B(dom_culture_politics_cs) - B(lang_dom)``
                   (``quantity = "B_cs_minus_lang_dom"``); ``preregistered`` on ONE dose of the
                   culture/politics K=8/C1 rows: the dose frozen on a development run
                   (``kw['direction_domain_eps']``), else 0.05 (PREREGISTRATION fixes none;
                   manifest ``contrast_13_eps``); ``min_effect_met`` / ``verdict`` by the CI
                   rule (:func:`~lsne.experiments.common.min_effect_verdict`, 0.05 nats: met only
                   when the whole interval is >= 0.05), never by the point estimate
head_effects       mean-ablation effect of every head of the pre-registered layer grid on
                   log p(target) per variant (:func:`lsne.heads.head_ablation_effects`)
head_direct        direct logit effect of every head on the target at the last position
worst_head         :func:`lsne.heads.worst_head` ranking (weak = culture/politics families,
                   stable = everyday families; falls back to "all families vs none" when no
                   culture/politics family is selected, ``selection`` says which);
                   ``selected`` marks the worst head
head_steering      contrast 14 per variant x direction x eps: B when the direction is
                   applied through the worst head (:func:`lsne.heads.steer_at_head`, dose
                   relative to the head's own RMS) vs added to the residual at the block
                   output of the same layer with the same relative dose (``arm="residual"``,
                   ``dose_match="relative"``, the pre-registered wording) and with the same
                   *absolute* norm as the head arm's realised residual delta
                   (``arm="residual_abs"``, ``dose_match="absolute"``: the relative residual
                   arm writes ~30x the head arm's norm, so B_head - B_residual measures the
                   residual arm's damage); ``delta_norm`` = ||delta|| written into the residual
                   stream, ``audit_rel_err`` = Soll/Ist of the write (Injector / HeadInjector)
head_steering_boot family bootstrap of ``B_head - B_residual`` (relative dose) and of
                   ``B_head - B_residual_abs`` (``quantity="B_head_minus_residual_abs"``) per
                   (direction, eps); ``delta_norm_ratio`` = mean ||delta_residual|| /
                   ||delta_head||
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..atlas import (atlas_domain_overlap, counter_steer, domain_axis, domain_profile, domain_token_ids,
                     load_atlas_extraction, load_domains, sae_feature_domain_profile)
from ..config import RunConfig, HookSite
from ..directions import Direction, cosine
from ..heads import head_ablation_effects, head_direct_logit_effect, steer_at_head, worst_head
from ..model import Injector, ModelBundle, forward_logits, make_dose_delta, next_token_stats, run_with_cache
from . import common as C

COMPASS_EPS = 0.05
PREREG_MIN_B = 0.05
#: Dose of the pre-registered contrast-13 row when no dose was frozen on a development run.
PREREG_EPS_13 = 0.05
DOMAIN_AXES = {"dom_law": ("law", "everyday"), "dom_culture_politics": (["culture_values", "politics"], "everyday"),
               "dom_protein_misfolding": ("protein_misfolding", "everyday")}
DD_DIRECTIONS = ("dom_culture_politics_cs", "dom_culture_politics", "dom_law", "lang_dom", "rand")
DD_COLS = list(C.ID_COLS) + ["family_domain", "direction", "arm", "eps", "logp_0", "logp_delta", "B", "entropy_0", "entropy",
                             "p_freq", "delta_norm", "removed_coef", "cos_after_remove", "n_dirs", "audit_rel_err"]


# --------------------------------------------------------------------------------------
# compass
# --------------------------------------------------------------------------------------
def reference_state(mb: ModelBundle, variants, original, site: HookSite) -> tuple[torch.Tensor, str, int]:
    if original is not None:
        k0 = [v for v in original.variants if int(v.K) == 0]
        if k0:
            return C.last_state(mb, k0[0], site), "original_2143", int(k0[0].target_id)
    std = C.k_subset(variants, 0) or list(variants)
    return C.last_state(mb, std[0], site), str(std[0].tag), int(std[0].target_id)


def compass_tables(mb: ModelBundle, h: torch.Tensor, state_source: str, directions: dict[str, object], ids_E, ids_D,
                   n_random: int, seed: int, bank=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for lang, dids in (("E", ids_E), ("D", ids_D)):
        for name, d in [("state", None)] + list(directions.items()):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                df = domain_profile(mb, h, None if d is None else d, eps=COMPASS_EPS, domains_ids=dids, n_random=n_random, seed=seed)
            df = df.sort_values("z", ascending=False, kind="mergesort").reset_index(drop=True)
            for r, (_, pr) in enumerate(df.iterrows()):
                rows.append(dict(state_source=state_source, direction=name, dir_family=C.dir_family(name), lang=lang,
                                 mode=df.attrs.get("mode", ""), eps=(np.nan if d is None else COMPASS_EPS), domain=pr["domain"],
                                 mass=float(pr["mass"]), mass_before=float(pr["mass_before"]), mass_after=float(pr["mass_after"]),
                                 mass_random_mean=float(pr["mass_random_mean"]), mass_random_sd=float(pr["mass_random_sd"]),
                                 z=float(pr["z"]), n_ids=int(pr["n_ids"]), rank=r + 1, is_top=(r == 0)))
    if bank is not None and bank.sae is not None and bank.sae_names:
        feats = [int(bank[n].meta.get("feature", -1)) for n in bank.sae_names]
        names = {int(bank[n].meta.get("feature", -1)): n for n in bank.sae_names}
        for lang, dids in (("E", ids_E), ("D", ids_D)):
            df = sae_feature_domain_profile(mb, bank.sae, [f for f in feats if f >= 0], dids)
            for f, g in df.groupby("feature", sort=False):
                g = g.sort_values("z", ascending=False, kind="mergesort").reset_index(drop=True)
                for r, (_, pr) in enumerate(g.iterrows()):
                    rows.append(dict(state_source="sae_decoder", direction=names.get(int(f), f"sae{f}"), dir_family="sae", lang=lang,
                                     mode="unembedding", eps=np.nan, domain=pr["domain"], mass=float(pr["mean_logit"]),
                                     mass_before=np.nan, mass_after=np.nan, mass_random_mean=float(pr["mean_all"]),
                                     mass_random_sd=float(pr["sd_all"]), z=float(pr["z"]), n_ids=int(pr["n_ids"]), rank=r + 1,
                                     is_top=(r == 0)))
    cols = ["state_source", "direction", "dir_family", "lang", "mode", "eps", "domain", "mass", "mass_before", "mass_after",
            "mass_random_mean", "mass_random_sd", "z", "n_ids", "rank", "is_top"]
    comp = pd.DataFrame(rows, columns=cols)
    srows = []
    for (src, name, lang), g in comp.groupby(["state_source", "direction", "lang"], sort=False):
        law = g[g["domain"] == "law"]
        srows.append(dict(state_source=src, direction=name, lang=lang, top_domain=str(g.iloc[0]["domain"]), top_z=float(g.iloc[0]["z"]),
                          law_z=(float(law["z"].iloc[0]) if len(law) else np.nan), law_rank=(int(law["rank"].iloc[0]) if len(law) else -1),
                          law_is_top=bool(len(law) and bool(law["is_top"].iloc[0])), n_domains=int(len(g))))
    return comp, pd.DataFrame(srows, columns=["state_source", "direction", "lang", "top_domain", "top_z", "law_z", "law_rank",
                                              "law_is_top", "n_domains"])


# --------------------------------------------------------------------------------------
# domain axes and counter-steering
# --------------------------------------------------------------------------------------
def build_domain_axes(mb: ModelBundle, site: HookSite, domains: dict, limits: C.Limits, sub: C.Substages) -> dict[str, Direction]:
    out: dict[str, Direction] = {}
    for name, (dom, con) in DOMAIN_AXES.items():
        d = sub.run(f"axis:{name}", lambda dom=dom, con=con: domain_axis(mb, mb.tokenizer, dom, site, contrast=con, lang="E",
                                                                        n_templates=limits.i_n_templates, domains=domains,
                                                                        max_words=limits.i_max_words), None)
        if d is not None:
            d.name = name
            out[name] = d
    return out


def domain_axes_table(axes: dict[str, Direction], bank) -> pd.DataFrame:
    rows = []
    for name, d in axes.items():
        m = d.meta
        row = dict(name=name, site=d.site.key, domain=str(m.get("domain")), contrast=str(m.get("contrast")),
                   n_domain=int(m.get("n_domain", 0)), n_contrast=int(m.get("n_contrast", 0)),
                   n_words_domain=int(m.get("n_words_domain", 0)), n_words_contrast=int(m.get("n_words_contrast", 0)),
                   n_templates=int(m.get("n_templates", 0)), cos_denoised_plain=float(m.get("cos_denoised_plain", np.nan)),
                   diff_norm=float(m.get("diff_norm", np.nan)), n_mismatch=int(m.get("n_mismatch", 0)))
        for other in ("ent", "lang_dom", "freq", "lang_probe"):
            row[f"cos_{other}"] = cosine(d.vec, bank[other].vec) if bank is not None and other in bank else np.nan
        for other, d2 in axes.items():
            row[f"cos_{other}"] = cosine(d.vec, d2.vec)
        rows.append(row)
    return pd.DataFrame(rows)


def direction_domain_tables(mb: ModelBundle, variants, bank, axes: dict[str, Direction], site: HookSite, doses: Sequence[float],
                            enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    freq_t = bank.freq_tensor(mb.device)
    rand = [bank[n].vec for n in bank.rand_names]
    rows: list[dict] = []
    for v in C.progress(variants, "I direction x domain", enabled=enabled):
        ids = C.ids_tensor(mb, v)
        T = int(ids.shape[1])
        with torch.no_grad():
            cache = run_with_cache(mb, ids, [site])
            base = {k: float(x[0]) for k, x in next_token_stats(cache.logits[:, -1], int(v.target_id), freq_t).items()}
            h = cache.resid[site.key][0, T - 1].detach().float().cpu()
        idc = dict(C.variant_row(v), family_domain=C.family_domain(v.family_id))

        def _stats(delta=None, h_new=None):
            if delta is not None:
                inj = Injector(mb, site, delta.to(mb.dtype), positions=[T - 1], mode="add")
            else:
                inj = Injector(mb, site, h_new.to(mb.dtype), positions=[T - 1], mode="replace")
            with torch.no_grad(), inj:
                logits = forward_logits(mb, ids)
            st = {k: float(x[0]) for k, x in next_token_stats(logits[:, -1], int(v.target_id), freq_t).items()}
            return st, float(inj.audit.get("rel_err", np.nan))

        for eps in doses:
            eps = float(eps)
            for name in DD_DIRECTIONS:
                arm, removed, cos_after, n_dirs, dn = "add", np.nan, np.nan, 1, np.nan
                if name == "rand":
                    if not rand:
                        continue
                    vals = []
                    for r in rand:
                        delta = make_dose_delta(h, r, eps)
                        st, err = _stats(delta=delta)
                        vals.append(st)
                        dn = float(delta.norm())
                    st = {k: float(np.mean([x[k] for x in vals])) for k in vals[0]}
                    err, n_dirs = np.nan, len(rand)
                elif name == "dom_culture_politics_cs":
                    if "dom_law" not in axes or "dom_culture_politics" not in axes:
                        continue
                    h_new, info = counter_steer(h, axes["dom_law"], axes["dom_culture_politics"], None, eps, return_info=True)
                    st, err = _stats(h_new=h_new)
                    arm, removed, cos_after, dn = "counter_steer_replace", float(info["removed_coef"]), float(info["cos_after"]), float((h_new - h).norm())
                else:
                    d = axes.get(name) if name.startswith("dom_") else bank.get(name)
                    if d is None:
                        continue
                    delta = make_dose_delta(h, d.vec, eps)
                    st, err = _stats(delta=delta)
                    dn = float(delta.norm())
                rows.append(dict(idc, direction=name, arm=arm, eps=eps, logp_0=base["logp_target"], logp_delta=st["logp_target"],
                                 B=st["logp_target"] - base["logp_target"], entropy_0=base["entropy"], entropy=st["entropy"],
                                 p_freq=st.get("p_freq", np.nan), delta_norm=dn, removed_coef=removed, cos_after_remove=cos_after,
                                 n_dirs=n_dirs, audit_rel_err=err))
    dd = pd.DataFrame(rows, columns=DD_COLS)
    trows = []
    for (fid, form), g in dd.drop_duplicates("tag").groupby(["family", "formulation"], sort=True):
        k0, k8 = g[g["K"] == 0], g[(g["K"] == 8) & (g["C"] == "C1")]
        if len(k0) and len(k8):
            trows.append(dict(family=fid, formulation=form, family_domain=C.family_domain(fid), logp_0_K0=float(k0["logp_0"].iloc[0]),
                              logp_0_K8=float(k8["logp_0"].iloc[0]), T=float(k8["logp_0"].iloc[0] - k0["logp_0"].iloc[0])))
    return dd, pd.DataFrame(trows, columns=["family", "formulation", "family_domain", "logp_0_K0", "logp_0_K8", "T"])


def contrast13_eps(frozen, doses: Sequence[float]) -> tuple[Optional[float], str]:
    """``(eps, source)`` of the one pre-registered contrast-13 row: the dose frozen on a
    development run (``source="frozen_input"``), else :data:`PREREG_EPS_13` (``"default_0.05"``);
    ``(None, "... not in doses")`` when that dose is not swept."""
    eps, source = ((float(frozen), "frozen_input") if frozen is not None and frozen != ""
                   else (PREREG_EPS_13, f"default_{PREREG_EPS_13:g}"))
    if not any(np.isclose(eps, float(d)) for d in doses):
        return None, f"{source}: eps {eps:g} not in doses {list(doses)}"
    return eps, source


def direction_domain_boot(dd: pd.DataFrame, n_boot: int, seed: int, prereg_eps: Optional[float] = PREREG_EPS_13) -> pd.DataFrame:
    """Family bootstrap of B per (family_domain, direction, K, C, eps) and of the contrast-13
    difference ``B_cs_minus_lang_dom``; ``preregistered`` only on the culture/politics K=8/C1 row
    at ``prereg_eps`` (one dose); ``min_effect_met`` / ``verdict`` follow the CI rule."""
    cols = ["quantity", "family_domain", "direction", "K", "C", "eps", "preregistered", "min_effect_met", "mean", "ci_low", "ci_high",
            "se", "n_clusters", "n_rows", "verdict", "p_boot"]
    if dd.empty:
        return pd.DataFrame(columns=cols)
    std = dd[dd["kind"] == "standard"]
    rows = []
    for (grp, name, K, Cc, eps), g in std.groupby(["family_domain", "direction", "K", "C", "eps"], sort=True):
        rows.append(C.boot_row(g, "B", n_boot, seed, with_p=True, quantity="B", family_domain=grp, direction=name, K=int(K), C=str(Cc),
                               eps=float(eps), preregistered=False, min_effect_met=None, verdict=""))
    piv = std.pivot_table(index=["family", "formulation", "tag", "family_domain", "K", "C", "eps"], columns="direction", values="B")
    if "dom_culture_politics_cs" in piv.columns and "lang_dom" in piv.columns:
        piv = piv.reset_index()
        piv["diff"] = piv["dom_culture_politics_cs"] - piv["lang_dom"]
        for (grp, K, Cc, eps), g in piv.groupby(["family_domain", "K", "C", "eps"], sort=True):
            pre = bool(grp == "culture_politics" and int(K) == 8 and str(Cc) == "C1" and prereg_eps is not None
                       and np.isclose(float(eps), float(prereg_eps)))
            r = C.boot_row(g, "diff", n_boot, seed, with_p=True, quantity="B_cs_minus_lang_dom", family_domain=grp,
                           direction="dom_culture_politics_cs-lang_dom", K=int(K), C=str(Cc), eps=float(eps), preregistered=pre)
            r["min_effect_met"], r["verdict"] = C.min_effect_verdict(r["ci_low"], r["ci_high"], PREREG_MIN_B)
            rows.append(r)
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------------------
# heads
# --------------------------------------------------------------------------------------
def head_tables(mb: ModelBundle, variants, grid: Sequence[int], enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    eff = head_ablation_effects(mb, variants, list(grid), mode="mean")
    eff = eff.rename(columns={"family_id": "family"})
    eff["tag"] = [variants[int(i)].tag for i in eff["variant"]]
    eff["family_domain"] = [C.family_domain(f) for f in eff["family"]]
    eff = eff[["variant", "tag", "family", "formulation", "family_domain", "K", "C", "layer", "head", "logp_base", "logp_ablated", "delta"]]
    rows = []
    for v in C.progress(variants, "I head direct effects", enabled=enabled):
        ids = C.ids_tensor(mb, v)
        for layer in grid:
            de = head_direct_logit_effect(mb, ids, int(layer), int(v.target_id), position=-1)
            for h in range(mb.n_heads):
                rows.append(dict(C.variant_row(v), family_domain=C.family_domain(v.family_id), layer=int(layer), head=h,
                                 direct_effect=float(de[h])))
    direct = pd.DataFrame(rows, columns=list(C.ID_COLS) + ["family_domain", "layer", "head", "direct_effect"])
    return eff, direct


def split_selection_families(families: Sequence[str]) -> tuple[list[str], list[str]]:
    """Disjoint selection / evaluation families for the worst-head arm (pre-registration 14:
    the selection is frozen before the head is tested). Within every domain group the sorted
    family ids alternate: even positions select, odd positions evaluate."""
    by_group: dict[str, list[str]] = {}
    for f in sorted({str(x) for x in families}):
        by_group.setdefault(C.family_domain(f), []).append(f)
    sel, ev = [], []
    for fs in by_group.values():
        sel += fs[0::2]
        ev += fs[1::2]
    return sel, ev


def parse_frozen_head(spec) -> Optional[tuple[int, int]]:
    """``"L:H"`` (or a (layer, head) pair / dict) -> (layer, head); None if not given."""
    if spec is None or spec == "":
        return None
    if isinstance(spec, dict):
        return int(spec["layer"]), int(spec["head"])
    if isinstance(spec, (tuple, list)):
        return int(spec[0]), int(spec[1])
    a, _, b = str(spec).partition(":")
    return int(a), int(b)


def select_worst_head(eff: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    e = eff.rename(columns={"family": "family_id"})
    fams = sorted(set(e["family_id"].astype(str)))
    weak = [f for f in fams if C.family_domain(f) == "culture_politics"]
    stable = [f for f in fams if C.family_domain(f) == "everyday"]
    selection = "culture_politics_vs_everyday"
    if not weak:
        weak, stable, selection = fams, [], "fallback_all_families_vs_none"
    w = worst_head(e, weak, stable)
    table = w["table"].copy()
    table["selected"] = (table["layer"] == w["layer"]) & (table["head"] == w["head"])
    table["selection"] = selection
    table["families_weak"] = ",".join(weak)
    table["families_stable"] = ",".join(stable)
    info = dict(layer=w["layer"], head=w["head"], score=w["score"], loss=w["loss"], selection=selection, families_weak=weak,
                families_stable=stable)
    return table, info


def head_steering_tables(mb: ModelBundle, variants, head_info: dict, directions: dict[str, object], doses: Sequence[float],
                         n_boot: int, seed: int, enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = list(C.ID_COLS) + ["family_domain", "layer", "head", "direction", "eps", "arm", "logp_0", "logp_delta", "B", "rms_delta",
                              "cos_realised_direction", "preimage_rel_err", "delta_norm", "dose_match", "audit_rel_err"]
    bcols = ["quantity", "direction", "eps", "mean", "ci_low", "ci_high", "se", "n_clusters", "n_rows", "dose_match",
             "delta_norm_ratio", "p_boot"]
    if head_info.get("layer") is None or not directions:
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=bcols)
    layer, head = int(head_info["layer"]), int(head_info["head"])
    out_site = HookSite(layer, "block_output")
    rows = []
    for v in C.progress(variants, "I head steering", enabled=enabled):
        ids = C.ids_tensor(mb, v)
        T = int(ids.shape[1])
        with torch.no_grad():
            cache = run_with_cache(mb, ids, [out_site])
        logp0 = float(torch.log_softmax(cache.logits[0, -1], -1)[int(v.target_id)])
        h_out = cache.resid[out_site.key][0, T - 1].detach().float().cpu()
        idc = dict(C.variant_row(v), family_domain=C.family_domain(v.family_id), layer=layer, head=head)
        def _residual(delta: torch.Tensor) -> tuple[float, float]:
            inj = Injector(mb, out_site, delta.to(mb.dtype), positions=[T - 1], mode="add")
            with torch.no_grad(), inj:
                logits = forward_logits(mb, ids)
            return float(torch.log_softmax(logits[0, -1], -1)[int(v.target_id)]), float(inj.audit.get("rel_err", np.nan))

        for name, d in directions.items():
            vec = getattr(d, "vec", d)
            for eps in doses:
                eps = float(eps)
                head_norm = None
                try:
                    last, info = steer_at_head(mb, ids, layer, head, vec, eps, position=T - 1, return_info=True)
                    lp = float(torch.log_softmax(last, -1)[int(v.target_id)])
                    head_norm = float(info["realised"].float().norm())
                    rows.append(dict(idc, direction=name, eps=eps, arm="head", logp_0=logp0, logp_delta=lp, B=lp - logp0,
                                     rms_delta=float(info["rms_delta"]), cos_realised_direction=float(info["cos_realised_direction"]),
                                     preimage_rel_err=float(info["preimage_rel_err"]), delta_norm=head_norm, dose_match="relative",
                                     audit_rel_err=float(info.get("audit", {}).get("written_rel_err", np.nan))))
                except ValueError as e:
                    warnings.warn(f"head steering skipped for {v.tag}/{name}: {e}")
                delta = make_dose_delta(h_out, vec, eps)
                lp, err = _residual(delta)
                rows.append(dict(idc, direction=name, eps=eps, arm="residual", logp_0=logp0, logp_delta=lp, B=lp - logp0,
                                 rms_delta=float(delta.float().pow(2).mean().sqrt()), cos_realised_direction=1.0, preimage_rel_err=0.0,
                                 delta_norm=float(delta.float().norm()), dose_match="relative", audit_rel_err=err))
                if head_norm is not None:
                    # the same absolute ||delta|| as the head arm's realised residual delta, along the direction
                    u = torch.as_tensor(vec).detach().float().cpu().reshape(-1)
                    delta_abs = u / u.norm().clamp_min(1e-30) * head_norm
                    lp, err = _residual(delta_abs)
                    rows.append(dict(idc, direction=name, eps=eps, arm="residual_abs", logp_0=logp0, logp_delta=lp, B=lp - logp0,
                                     rms_delta=float(delta_abs.pow(2).mean().sqrt()), cos_realised_direction=1.0, preimage_rel_err=0.0,
                                     delta_norm=float(delta_abs.norm()), dose_match="absolute", audit_rel_err=err))
    hs = pd.DataFrame(rows, columns=cols)
    brows = []
    if len(hs):
        keys = ["family", "tag", "direction", "eps"]
        std = hs[hs["kind"] == "standard"]
        piv = std.pivot_table(index=keys, columns="arm", values="B").reset_index()
        nrm = std.pivot_table(index=keys, columns="arm", values="delta_norm").reset_index()
        for arm, q, match in (("residual", "B_head_minus_residual", "relative"), ("residual_abs", "B_head_minus_residual_abs", "absolute")):
            if "head" not in piv.columns or arm not in piv.columns:
                continue
            piv["diff"] = piv["head"] - piv[arm]
            ratio = nrm.set_index(keys)[arm] / nrm.set_index(keys)["head"] if {"head", arm} <= set(nrm.columns) else None
            piv["ratio"] = ratio.reindex(pd.MultiIndex.from_frame(piv[keys])).to_numpy() if ratio is not None else np.nan
            for (name, eps), g in piv.groupby(["direction", "eps"], sort=True):
                ratio = g["ratio"].to_numpy(dtype=float)
                brows.append(C.boot_row(g, "diff", n_boot, seed, with_p=True, quantity=q, direction=name, eps=float(eps),
                                        dose_match=match,
                                        delta_norm_ratio=float(np.nanmean(ratio)) if np.isfinite(ratio).any() else np.nan))
    return hs, pd.DataFrame(brows, columns=bcols)


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Compass, domain axes, direction x domain and the head arm; see the module docstring."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    sub = C.Substages("I", verbose=limits.progress)
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    site = C.dossier_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original, sae=kw.get("sae"))
    empty = pd.DataFrame()
    domains = load_domains()
    ids_E, ids_D = domain_token_ids(mb.tokenizer, domains, "E"), domain_token_ids(mb.tokenizer, domains, "D")

    axes = build_domain_axes(mb, site, domains, limits, sub)
    if ctx is not None:
        ctx["domain_axes"] = axes
    if axes:
        C.save_bank(C.DirectionBank(site=site, dirs=dict(axes)), run_dir, name="directions_domain")
    axes_tab = sub.run("domain_axes_table", lambda: domain_axes_table(axes, bank), empty)

    h, state_source, _tid = reference_state(mb, variants, original, site)
    comp_dirs = {n: bank[n] for n in ("ent", "lang_dom", "freq") if n in bank}
    comp_dirs.update(axes)
    comp, comp_summary = sub.run("compass", lambda: compass_tables(mb, h, state_source, comp_dirs, ids_E, ids_D,
                                                                   limits.i_n_random_profile, cfg.seed, bank), (empty, empty))
    atlas_dir = kw.get("atlas_dir") or str(C._REPO_ROOT / "data" / "atlas")
    clusters = load_atlas_extraction(atlas_dir)
    atlas = sub.run("atlas", lambda: atlas_domain_overlap(clusters, domains) if clusters else
                    pd.DataFrame(columns=["cluster_id", "domain", "n_overlap", "jaccard", "n_keywords", "n_domain_words", "overlap"]), empty)

    std = C.k_subset(variants, 0) + C.k_subset(variants, 8, "C1")
    dd, dd_T = sub.run("direction_domain", lambda: direction_domain_tables(mb, std, bank, axes, site, limits.i_doses, enabled=limits.progress),
                       (empty, empty))
    eps13, eps13_source = contrast13_eps(kw.get("direction_domain_eps"), limits.i_doses)
    dd_boot = sub.run("direction_domain_boot", lambda: direction_domain_boot(dd, limits.n_boot, cfg.seed, prereg_eps=eps13), empty)

    grid = C.prereg_layer_grid(mb.n_layers)
    eff, direct = sub.run("heads", lambda: head_tables(mb, std, grid, enabled=limits.progress), (empty, empty))
    # Worst head: frozen input if given (limits.i_frozen_head / kw worst_head), else selected on a
    # disjoint half of the families and tested on the other half (pre-registration 14).
    frozen = parse_frozen_head(kw.get("worst_head") or limits.i_frozen_head)
    sel_fams, eval_fams = split_selection_families([v.family_id for v in std])
    overlap = False
    if frozen is not None:
        wh_table, wh = empty, dict(layer=frozen[0], head=frozen[1], selection="frozen_input", score=np.nan, loss=np.nan)
        eval_variants = std
    else:
        if not eval_fams:                      # a single family (smoke runs): no disjoint split possible
            eval_fams, overlap = list(sel_fams), True
        eff_sel = eff[eff["family"].astype(str).isin(sel_fams)] if len(eff) else eff
        wh_table, wh = sub.run("worst_head", lambda: select_worst_head(eff_sel) if len(eff_sel) else (empty, {}), (empty, {}))
        eval_variants = [v for v in std if str(v.family_id) in set(eval_fams)]
    wh = dict(wh, selection_families=sel_fams if frozen is None else [], evaluation_families=sorted({str(v.family_id) for v in eval_variants}),
              selection_evaluation_overlap=overlap)
    if ctx is not None:
        ctx["worst_head"] = wh
    steer_dirs = {n: d for n, d in (("lang_dom", bank.get("lang_dom")), ("dom_culture_politics", axes.get("dom_culture_politics"))) if d is not None}
    hs, hs_boot = sub.run("head_steering", lambda: head_steering_tables(mb, eval_variants, wh, steer_dirs, limits.i_head_eps, limits.n_boot,
                                                                        cfg.seed, enabled=limits.progress), (empty, empty))

    tables = {"compass": comp, "compass_summary": comp_summary, "atlas_overlap": atlas, "domain_axes": axes_tab,
              "direction_domain": dd, "direction_domain_T": dd_T, "direction_domain_boot": dd_boot, "head_effects": eff,
              "head_direct": direct, "worst_head": wh_table, "head_steering": hs, "head_steering_boot": hs_boot}
    sub.write_tracebacks(run_dir)
    return C.finish_stage(run_dir, "I", cfg, mb, tables, limits=limits, t0=t0, site=site, state_source=state_source,
                          compass_eps=COMPASS_EPS, n_random_profile=limits.i_n_random_profile, domain_axes=list(axes),
                          domain_ids_dropped={k: int(n) for k, n in ids_E.meta.get("n_dropped", {}).items()},
                          atlas=dict(dir=str(atlas_dir), n_clusters=len(clusters)), layer_grid=[int(l) for l in grid],
                          worst_head=C.to_jsonable(wh), doses=list(limits.i_doses), head_eps=list(limits.i_head_eps),
                          n_variants=len(std), substages=sub.report(),
                          contrast_13_eps=dict(eps=eps13, source=eps13_source, min_effect=PREREG_MIN_B,
                                               rule="min_effect_met = ci_low >= 0.05 (CI rule, common.min_effect_verdict)"),
                          head_arm_dose=dict(relative="arm residual: eps x RMS of the residual (pre-registered wording)",
                                             absolute="arm residual_abs: ||delta|| = the head arm's realised ||W_O u||"),
                          prereg=dict(contrast_12="compass_summary: direction=ent, lang=E -> law_is_top / law_rank",
                                      contrast_13="direction_domain_boot quantity=B_cs_minus_lang_dom, family_domain=culture_politics, K=8/C1, "
                                                  "preregistered=True (one eps: contrast_13_eps)",
                                      contrast_14="head_steering_boot quantity=B_head_minus_residual (relative dose) and "
                                                  "B_head_minus_residual_abs (equal absolute ||delta||) (evaluation families only; "
                                                  "worst_head.selection_families disjoint unless selection_evaluation_overlap)"))
