"""Stage C - switching grid, dossier contrasts and counterfactual attribution.

Grid (DESIGN section 4.3): for every variant x direction (``ent``, ``lang_dom``, ``freq``,
every ``rand_XX``; each with sign +/-) x dose in ``cfg.doses`` the intervention benefit
``B(x) = log p_delta(target) - log p_0(target)`` at the State-2143 site (block input of
the last block, last context position), using the variant's OWN unperturbed baseline.

Tables
------
grid                 per variant x direction x sign x eps: logp_0, logp_delta, B, p_target,
                     entropy, p_freq (after the dose), audit rel_err
grid_summary         random directions averaged (``dir_family='rand'``: mean, sd, n over
                     directions); other families unchanged
contrasts            per formulation and mixed variant: T = logp_0(mixed) - logp_0(K=0) and
                     I = B(mixed) - B(K=0) per (dir_family, sign, eps)
contrasts_boot       family cluster bootstrap of T per (K, C) and of I per
                     (K, C, dir_family, sign, eps)
attribution          counterfactual position patching K=0 -> K=8/C1 and K=0 -> K=28/seg
                     (original family: K=0 -> largest K) on the block inputs of
                     ``config.layer_grid(n_layers)``, per token class with a random-position
                     null (:func:`lsne.patching.counterfactual_attribution`)
attribution_summary  mean recovery per (pair, site, class) over formulations
grad_x_act           gradient x activation of log p(target) per variant, site and position
grad_x_act_by_class  share of |grad x act| per token class
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from ..config import RunConfig, layer_grid, HookSite
from ..model import Injector, ModelBundle, forward_logits, make_dose_delta, next_token_stats, run_with_cache
from ..patching import counterfactual_attribution, grad_x_act
from ..stats import cluster_bootstrap
from . import common as C

GRID_FAMILIES = ("ent", "lang_dom", "freq", "rand")
GRID_COLS = list(C.ID_COLS) + ["direction", "dir_family", "sign", "eps", "logp_0", "logp_delta", "B", "p_target",
                               "entropy_0", "entropy", "p_freq_0", "p_freq", "delta_norm", "audit_rel_err"]


# --------------------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------------------
def grid_rows(mb: ModelBundle, variants, bank: C.DirectionBank, site: HookSite, doses, dir_names,
              enabled: bool = True) -> list[dict]:
    freq_t = bank.freq_tensor(mb.device)
    rows: list[dict] = []
    for v in C.progress(variants, "C grid", enabled=enabled):
        ids = C.ids_tensor(mb, v)
        T = int(ids.shape[1])
        with torch.no_grad():
            cache = run_with_cache(mb, ids, [site])
            base = next_token_stats(cache.logits[:, -1], int(v.target_id), freq_t)
            h_ref = cache.resid[site.key][0, T - 1].detach()
        b = {k: float(val[0]) for k, val in base.items()}
        for name in dir_names:
            vec = bank[name].vec.to(h_ref.device)
            for sign in (1, -1):
                for eps in doses:
                    eps = float(eps)
                    if eps == 0.0:
                        st, dn, err = b, 0.0, 0.0
                    else:
                        delta = make_dose_delta(h_ref, sign * vec, eps).to(h_ref.dtype)
                        inj = Injector(mb, site, delta, positions=[T - 1], mode="add")
                        with torch.no_grad(), inj:
                            logits = forward_logits(mb, ids)
                        st = {k: float(val[0]) for k, val in next_token_stats(logits[:, -1], int(v.target_id), freq_t).items()}
                        dn, err = float(delta.float().norm()), float(inj.audit.get("rel_err", float("nan")))
                    rows.append(dict(C.variant_row(v), direction=name, dir_family=C.dir_family(name), sign=sign, eps=eps,
                                     logp_0=b["logp_target"], logp_delta=st["logp_target"],
                                     B=st["logp_target"] - b["logp_target"], p_target=st["p_target"],
                                     entropy_0=b["entropy"], entropy=st["entropy"],
                                     p_freq_0=b.get("p_freq", float("nan")), p_freq=st.get("p_freq", float("nan")),
                                     delta_norm=dn, audit_rel_err=err))
    return rows


def summarise_grid(grid: pd.DataFrame) -> pd.DataFrame:
    """Average the random directions per (variant, sign, eps); keep the others."""
    if grid.empty:
        return pd.DataFrame(columns=list(C.ID_COLS) + ["dir_family", "sign", "eps", "B", "B_sd", "n_dirs",
                                                      "logp_0", "logp_delta", "entropy", "p_freq"])
    keys = list(C.ID_COLS) + ["dir_family", "sign", "eps"]
    agg = grid.groupby(keys, sort=False).agg(
        B=("B", "mean"), B_sd=("B", lambda s: float(np.std(s, ddof=0))), n_dirs=("direction", "nunique"),
        logp_0=("logp_0", "mean"), logp_delta=("logp_delta", "mean"), entropy=("entropy", "mean"),
        p_freq=("p_freq", "mean")).reset_index()
    return agg


def contrast_rows(summary: pd.DataFrame) -> pd.DataFrame:
    """T and I of every mixed variant relative to the K=0 variant of its formulation."""
    cols = ["family", "formulation", "tag", "K", "C", "kind", "dir_family", "sign", "eps", "T", "B_mixed", "B_mono", "I"]
    if summary.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (fam, form), g in summary.groupby(["family", "formulation"], sort=False):
        ref = g[g["K"] == 0]
        if ref.empty:
            continue
        ref_idx = ref.set_index(["dir_family", "sign", "eps"])
        for tag, gv in g[g["K"] != 0].groupby("tag", sort=False):
            for _, r in gv.iterrows():
                key = (r["dir_family"], r["sign"], r["eps"])
                if key not in ref_idx.index:
                    continue
                rr = ref_idx.loc[key]
                rows.append(dict(family=fam, formulation=form, tag=tag, K=int(r["K"]), C=r["C"], kind=r["kind"],
                                 dir_family=r["dir_family"], sign=int(r["sign"]), eps=float(r["eps"]),
                                 T=float(r["logp_0"] - rr["logp_0"]), B_mixed=float(r["B"]), B_mono=float(rr["B"]),
                                 I=float(r["B"] - rr["B"])))
    return pd.DataFrame(rows, columns=cols)


def bootstrap_contrasts(contrasts: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    cols = ["quantity", "K", "C", "dir_family", "sign", "eps", "mean", "ci_low", "ci_high", "se", "n_clusters", "n_rows"]
    if contrasts.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    std = contrasts[contrasts["kind"] == "standard"]
    if std.empty:
        return pd.DataFrame(columns=cols)
    # T does not depend on the direction: one row per (K, C) from the eps == 0 rows of one family
    t_src = std[(std["eps"] == 0.0) & (std["sign"] == 1)].drop_duplicates(["tag", "dir_family"])
    t_src = t_src.drop_duplicates(["tag"])
    for (K, Cc), g in t_src.groupby(["K", "C"], sort=True):
        r = cluster_bootstrap(g, "T", "family", n_boot=n_boot, seed=seed)
        rows.append(dict(quantity="T", K=int(K), C=Cc, dir_family="", sign=0, eps=0.0, mean=r["mean"],
                         ci_low=r["ci_low"], ci_high=r["ci_high"], se=r["se"], n_clusters=r["n_clusters"], n_rows=r["n_rows"]))
    for (K, Cc, fam, sign, eps), g in std[std["eps"] > 0].groupby(["K", "C", "dir_family", "sign", "eps"], sort=True):
        r = cluster_bootstrap(g, "I", "family", n_boot=n_boot, seed=seed)
        rows.append(dict(quantity="I", K=int(K), C=Cc, dir_family=fam, sign=int(sign), eps=float(eps), mean=r["mean"],
                         ci_low=r["ci_low"], ci_high=r["ci_high"], se=r["se"], n_clusters=r["n_clusters"], n_rows=r["n_rows"]))
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------------------
# counterfactual attribution and gradient x activation
# --------------------------------------------------------------------------------------
def attribution_pairs(variants) -> list[tuple[str, object, object]]:
    """``(pair_name, v_mono, v_mixed)`` per formulation: K=0 vs K=8/C1 and K=0 vs K=28/seg
    (original family: K=0 vs the largest K)."""
    refs = C.reference_of(variants)
    pairs = []
    groups: dict[tuple[str, str], list] = {}
    for v in variants:
        groups.setdefault(C.group_key(v), []).append(v)
    for key, vs in groups.items():
        ref = refs.get(key)
        if ref is None:
            continue
        if C.is_original(ref):
            mixed = [v for v in vs if int(v.K) > 0]
            if mixed:
                vmax = max(mixed, key=lambda v: int(v.K))
                pairs.append((f"K0_vs_K{vmax.K}orig", ref, vmax))
            continue
        for (K, Cc) in C.CORE_CONDITIONS[1:]:
            m = [v for v in vs if (int(v.K), str(v.C)) == (K, Cc)]
            if m:
                pairs.append((f"K0_vs_K{K}{Cc}", ref, m[0]))
    return pairs


def attribution_table(mb: ModelBundle, variants, sites, n_random: int, seed: int, enabled: bool = True) -> pd.DataFrame:
    frames = []
    for name, v_mono, v_mixed in C.progress(attribution_pairs(variants), "C attribution", enabled=enabled):
        try:
            df = counterfactual_attribution(mb, v_mono, v_mixed, sites, list(C.PATCH_CLASSES), n_random=n_random, seed=seed)
        except ValueError as e:   # tails that do not align (should have been excluded in stage A)
            df = pd.DataFrame([dict(site="", layer=-1, **{"class": ""}, n_positions=0, recovery=float("nan"),
                                    dst_positions=f"error: {e}")])
        df.insert(0, "pair", name)
        df.insert(1, "family", v_mixed.family_id)
        df.insert(2, "formulation", v_mixed.formulation)
        df.insert(3, "tag_mixed", v_mixed.tag)
        df.insert(4, "K_mixed", int(v_mixed.K))
        df.insert(5, "C_mixed", str(v_mixed.C))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["pair", "family", "formulation", "tag_mixed", "K_mixed", "C_mixed", "site", "layer",
                                     "class", "n_positions", "n_tail", "n_nontail", "logp_mono", "logp_mixed",
                                     "logp_patched", "recovery", "recovery_random_mean", "recovery_random_sd", "dst_positions"])
    return pd.concat(frames, ignore_index=True)


def summarise_attribution(att: pd.DataFrame) -> pd.DataFrame:
    cols = ["pair", "site", "layer", "class", "n", "recovery_mean", "recovery_sd", "recovery_random_mean",
            "n_positions_mean", "excess_over_random"]
    if att.empty or "class" not in att.columns:
        return pd.DataFrame(columns=cols)
    rows = []
    for (pair, site, layer, cls), g in att.groupby(["pair", "site", "layer", "class"], sort=True):
        rec = g["recovery"].to_numpy(dtype=float)
        rnd = g["recovery_random_mean"].to_numpy(dtype=float)
        rows.append(dict(pair=pair, site=site, layer=int(layer), **{"class": cls}, n=int(np.isfinite(rec).sum()),
                         recovery_mean=float(np.nanmean(rec)) if np.isfinite(rec).any() else float("nan"),
                         recovery_sd=float(np.nanstd(rec)) if np.isfinite(rec).any() else float("nan"),
                         recovery_random_mean=float(np.nanmean(rnd)) if np.isfinite(rnd).any() else float("nan"),
                         n_positions_mean=float(g["n_positions"].mean()),
                         excess_over_random=float(np.nanmean(rec - rnd)) if np.isfinite(rec - rnd).any() else float("nan")))
    return pd.DataFrame(rows, columns=cols)


def grad_x_act_tables(mb: ModelBundle, variants, sites, enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, by_cls = [], []
    for v in C.progress(variants, "C grad x act", enabled=enabled):
        a = v.annot
        ts = int(v.tail_start)
        for site in sites:
            g = grad_x_act(mb, v.ids, site, int(v.target_id)).numpy()
            tot = float(np.abs(g).sum())
            for t in range(v.n_tokens):
                rows.append(dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=int(v.K), C=str(v.C),
                                 site=site.key, layer=int(site.layer), position=t, klass=a["klass"][t], lang=a["lang"][t],
                                 is_tail=t >= ts, dist_to_last_switch=int(a["dist_to_last_switch"][t]), gxa=float(g[t])))
            classes: dict[str, list[float]] = {}
            for t in range(v.n_tokens):
                classes.setdefault(a["klass"][t], []).append(float(g[t]))
            classes["tail"] = [float(x) for x in g[ts:]]
            for cls, vals in classes.items():
                arr = np.asarray(vals)
                by_cls.append(dict(C.variant_row(v), site=site.key, layer=int(site.layer), **{"class": cls}, n=int(arr.size),
                                   gxa_sum=float(arr.sum()), abs_sum=float(np.abs(arr).sum()),
                                   abs_share=float(np.abs(arr).sum() / tot) if tot > 0 else float("nan"),
                                   gxa_mean=float(arr.mean()) if arr.size else float("nan")))
    gcols = ["tag", "family", "formulation", "K", "C", "site", "layer", "position", "klass", "lang", "is_tail",
             "dist_to_last_switch", "gxa"]
    ccols = list(C.ID_COLS) + ["site", "layer", "class", "n", "gxa_sum", "abs_sum", "abs_share", "gxa_mean"]
    return pd.DataFrame(rows, columns=gcols), pd.DataFrame(by_cls, columns=ccols)


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Grid B(x), contrasts T / I, counterfactual attribution and gradient x activation."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    allv = C.all_variants(variants, original)
    site = C.dossier_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original, sae=kw.get("sae"))

    fams = ("ent", "lang_dom", "freq") if limits.limited else GRID_FAMILIES
    dir_names = bank.names(fams)
    grid = pd.DataFrame(grid_rows(mb, allv, bank, site, cfg.doses, dir_names, enabled=limits.progress), columns=GRID_COLS)
    summary = summarise_grid(grid)
    contrasts = contrast_rows(summary)
    boot = bootstrap_contrasts(contrasts, n_boot=limits.n_boot, seed=cfg.seed)

    sites = [HookSite(int(l), "block_input") for l in layer_grid(mb.n_layers)]
    att = attribution_table(mb, allv, sites, n_random=limits.n_random_patch, seed=cfg.seed, enabled=limits.progress)
    gxa_variants = C.core_subset(allv) if limits.limited else allv
    gxa, gxa_cls = grad_x_act_tables(mb, gxa_variants, sites, enabled=limits.progress)

    tables = {
        "grid": grid, "grid_summary": summary, "contrasts": contrasts, "contrasts_boot": boot,
        "attribution": att, "attribution_summary": summarise_attribution(att),
        "grad_x_act": gxa, "grad_x_act_by_class": gxa_cls,
    }
    return C.finish_stage(run_dir, "C", cfg, mb, tables, limits=limits, t0=t0, site=site, directions=dir_names,
                          bank_source=bank.meta.get("source", ""),
                          doses=list(cfg.doses), layer_grid=[int(l) for l in layer_grid(mb.n_layers)],
                          n_variants=len(variants), n_pairs=len(attribution_pairs(allv)))
