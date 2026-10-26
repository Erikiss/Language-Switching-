"""Stage D - non-equilibrium diagnostics D1-D9 (DESIGN section 4.2).

Lattice diagnostics (every variant; directions ``lang_dom``, ``ent``, the random
directions and - if enabled - the SAE features; doses ``Limits.damage_doses`` +
``cfg.doses_steer``):

D1 damage          ``damage``          per (variant, injection, direction, eps): growth
                                       exponent lambda over depth on the tail positions,
                                       spreading fronts, final perturbation, B(x)
                   ``damage_profile``  per snapshot: mean D on the tail, D at the last
                                       token, fronts at thresholds 0.01 and 0.1
                   ``damage_phase``    phase-diagram table: means over variants per
                                       (K, C, injection, dir_family, eps)
   Injections: (a) ``dossier`` = block input of the last block at the last token (the
   State-2143 site; position front), (b) ``depth`` = block input of block ``n_layers//3``
   at the language-defined site :func:`lsne.stimuli.last_switched_position` (the last
   German token of the context, i.e. the token before the final D -> E switch; for K=0
   the position-matched last token of unit 9).  The last token of unit 9 is German only
   for C1 (C2/C3/seg realise the final switch inside unit 9), so a unit-defined site
   would confound the K x C phase diagram; the language, unit and token of the injected
   position are recorded (``inject_lang``, ``inject_unit``, ``inject_token``).
   Each injection uses the directions fitted **at its own site**: the ``dossier`` bank
   (stage B, block input of the last block) and a second bank built at the depth site
   (``directions_depth.npz``; ``lang_dom`` / ``ent`` are layer-specific, DESIGN section 3
   "pro Layer"); the isotropic random directions are shared.  ``dir_site`` names the
   direction's site in every row, the manifest records the cosine between the two
   versions of each shared direction.
D2 lens            ``lens``            logit-lens KL to the final prediction per snapshot
                                       (last position + tail mean), target rank / logp
                   ``lens_summary``    non-monotonicity indices per variant
                   ``update_norms``    relative update norm between snapshots
D9 CA              ``ca``              symbol-field complexity (full field and tail columns);
                                       sites below the absolute marker-mass floor of
                                       :func:`lsne.dynamics.ca_symbolize` are N
                   ``ca_layers``       symbol fractions and marker mass per snapshot

Generation diagnostics (core subset K=0, K=8/C1, K=28/seg; directions ``lang_dom`` and
``ent``; the dossier site):

D3 hysteresis      ``hysteresis``, ``hysteresis_curves``   null model = ``n_null`` independent
                                      restarts per level with their own seeds (disjoint from
                                      the ramp seed); ``null_area_sd`` / ``area_z`` summarise
                                      the null distribution of the area
D4 relaxation      ``relaxation``, ``relaxation_curves``   ``tau`` only when identifiable
                                      (``identifiable``, ``reason``), raw value in ``tau_raw``,
                                      ``tau_censored`` for bound-saturated fits, ``b``, ``b_se``,
                                      ``r2``, ``f_pvalue`` and the bootstrap CI of ``b``
D5 switch stats    ``switch_stats``   (on the hysteresis and relaxation samples)
D6 susceptibility  ``susceptibility`` (constant-dose runs at 0 + ``cfg.doses_steer``)
D7 FDT             ``fdt``            T_eff = Var_0(m) / chi for m = f_D and for the
                                      projection of the generated text's residuals onto
                                      ``lang_dom``
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import torch

from .. import stimuli
from ..config import RunConfig, HookSite
from ..directions import Direction, cosine
from ..dynamics import (DAMAGE_BATCH_SIZE, ca_complexity, ca_marker_mass, ca_mass_floor, damage_spreading_batch,
                        growth_exponent, logit_lens_trajectory, nonmonotonicity_index, resolve_markers, spreading_front,
                        symbolize_masses, update_norm_profile)
from ..generation import (block_fraction, fdt_ratio, hysteresis, language_trajectory, relaxation, schedule_const,
                          susceptibility, switch_interval_stats)
from ..model import ModelBundle, generate_steered, run_with_cache
from . import common as C

LATTICE_FAMILIES = ("lang_dom", "ent", "rand", "sae")
GEN_FAMILIES = ("lang_dom", "ent")
FRONT_THRESHOLDS = (0.01, 0.1)
#: Independent restart replicates of the D3 null model (``Limits.gen_n_null`` overrides).
HYST_N_NULL = 4
#: Marker-mass floor of the D9 symbolisation (``"auto"`` = CA_MASS_FLOOR_FACTOR x uniform).
CA_MASS_FLOOR = "auto"


# --------------------------------------------------------------------------------------
# D1 damage spreading
# --------------------------------------------------------------------------------------
def depth_injection_position(v) -> int:
    """Position of the ``depth`` injection: the last German token of the context
    (:func:`lsne.stimuli.last_switched_position`), which is comparable across C levels;
    for K=0 the last token of unit 9 (the C1 site)."""
    return int(stimuli.last_switched_position(v))


def position_info(v, pos: int) -> dict:
    """Language, unit, token and class of context position ``pos`` (tidy-row columns)."""
    a = v.annot
    pos = int(pos)
    return dict(inject_lang=str(a["lang"][pos]), inject_unit=int(a["unit"][pos]),
                inject_token=str(a["tokens"][pos]) if "tokens" in a else "",
                inject_klass=str(a["klass"][pos]) if "klass" in a else "")


def injections(mb: ModelBundle, v) -> list[tuple[str, HookSite, int]]:
    """``(name, site, position)`` of the two D1 injections of a variant (see module notes)."""
    return [("dossier", C.dossier_site(mb), int(v.n_tokens) - 1),
            ("depth", C.depth_site(mb), depth_injection_position(v))]


def bank_for_site(banks, site: HookSite) -> C.DirectionBank:
    """``banks`` is one :class:`~lsne.experiments.common.DirectionBank` (used at every
    site) or a mapping ``site.key -> DirectionBank``; returns the bank of ``site``."""
    if isinstance(banks, C.DirectionBank):
        return banks
    try:
        return banks[site.key]
    except KeyError as e:
        raise KeyError(f"no direction bank for injection site {site.key}; available: {sorted(banks)}") from e


def depth_bank(cfg: RunConfig, mb: ModelBundle, variants, site: HookSite, base: C.DirectionBank,
               ctx=None, limits: C.Limits | None = None, original=None) -> C.DirectionBank:
    """Direction bank fitted at the depth-injection ``site`` (``ent``, ``lang_dom``,
    ``lang_probe``, ``freq``; SAE features only when ``limits.use_sae`` and a published SAE
    exists for that site).  The random directions are isotropic and are shared with the
    ``base`` bank (same vectors, marked ``shared_from`` in their meta).  Cached in
    ``ctx['bank_depth']`` under :func:`~lsne.experiments.common.bank_signature`; if
    ``base`` already sits at ``site`` it is returned unchanged."""
    limits = limits or C.Limits()
    variants = list(variants)
    sig = C.bank_signature(variants, site, cfg, limits, original)
    if ctx is not None and isinstance(ctx.get("bank_depth"), C.DirectionBank) and ctx.get("bank_depth_signature") == sig:
        return ctx["bank_depth"]
    if base.site == site:
        bank = base
    else:
        bank = C.direction_bank(cfg, mb, variants, site, original=original, limits=limits, sae=None)
        for n in base.rand_names:
            bank.dirs[n] = Direction(n, base[n].vec.clone(), site, dict(base[n].meta, shared_from=base.site.key))
        bank.meta["source"] = "built"
        bank.meta["shared_from_base"] = dict(site=base.site.key, names=list(base.rand_names))
        bank.meta["cos_vs_base"] = {n: cosine(base[n].vec, bank[n].vec) for n in bank if n in base}
    if ctx is not None:
        ctx["bank_depth"], ctx["bank_depth_signature"] = bank, sig
    return bank


def damage_tables(mb: ModelBundle, variants, banks, dir_names, doses, enabled=True,
                  batch_size: int = DAMAGE_BATCH_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (damage, damage_profile).

    ``banks`` is a single bank or a mapping ``site.key -> DirectionBank`` (see
    :func:`bank_for_site`): every injection is made with the direction fitted at its own
    site, recorded as ``dir_site``.  A direction missing from a site's bank is skipped
    for that injection (one warning per name and site).

    All (direction, dose) pairs of one injection run through
    :func:`lsne.dynamics.damage_spreading_batch` in forwards of ``batch_size`` rows (the
    unperturbed reference is row 0 of every forward).  Nothing is kept across variants:
    the damage maps of an injection are reduced to table rows before the next one starts,
    and no reference cache is retained (stage D used to hold one ``all_block_inputs``
    cache with the full ``[1, T, V]`` logits per variant on the device until
    :func:`lens_tables`, ~36 MiB per pythia-160m variant, which ran the full 1153-variant
    run out of memory on a 40 GB A100)."""
    rows, prof = [], []
    missing: set = set()
    for v in C.progress(variants, "D1 damage", enabled=enabled):
        ids = C.ids_tensor(mb, v)
        T = int(v.n_tokens)
        tail = C.tail_positions(v)
        for inj_name, site, pos in injections(mb, v):
            bank = bank_for_site(banks, site)
            info = position_info(v, pos)
            positions = [p for p in tail if p >= pos] or [T - 1]
            todo = []
            for name in dir_names:
                if name not in bank:
                    if (name, site.key) not in missing:
                        warnings.warn(f"direction {name!r} has no version at {site.key}; skipped for the {inj_name} injection")
                        missing.add((name, site.key))
                    continue
                todo.extend((name, float(eps)) for eps in doses)
            if not todo:
                continue
            maps = damage_spreading_batch(mb, ids, site, pos, [bank[name].vec for name, _ in todo],
                                          [eps for _, eps in todo], target_id=int(v.target_id), batch_size=batch_size)
            for (name, eps), dm in zip(todo, maps):
                d = bank[name]
                ge = growth_exponent(dm, positions)
                fronts = {thr: spreading_front(dm, thr) for thr in FRONT_THRESHOLDS}
                D_tail = dm.D[:, positions].mean(axis=1)                      # [L+1]
                after = list(range(pos, T))
                rows.append(dict(C.variant_row(v), inject=inj_name, site=site.key, dir_site=d.site.key,
                                 inject_layer=int(dm.inject_layer), inject_pos=int(pos), **info,
                                 direction=name, dir_family=C.dir_family(name), eps=float(eps),
                                 lam=float(ge["lam"]), intercept=float(ge["intercept"]), r2=float(ge["r2"]),
                                 n_fit_layers=int(len(ge["layers"])),
                                 D_final_tail=float(D_tail[-1]), D_final_last=float(dm.D[-1, T - 1]),
                                 D_max=float(dm.D.max()), cos_final_last=float(dm.cos[-1, T - 1]),
                                 front01_final=int(fronts[0.01][-1]), front1_final=int(fronts[0.1][-1]),
                                 front01_max=int(fronts[0.01].max()), front1_max=int(fronts[0.1].max()),
                                 n_spread01_final=int((dm.D[-1, after] > 0.01).sum()),
                                 n_spread1_final=int((dm.D[-1, after] > 0.1).sum()),
                                 n_after=len(after), logp_target_ref=float(dm.logp_target_ref),
                                 logp_target_pert=float(dm.logp_target_pert),
                                 B=float(dm.logp_target_pert - dm.logp_target_ref), delta_norm=float(dm.delta_norm),
                                 audit_rel_err=float(dm.audit.get("rel_err", float("nan")))))
                for l, key in enumerate(dm.layers):
                    prof.append(dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=int(v.K), C=str(v.C),
                                     inject=inj_name, direction=name, dir_family=C.dir_family(name), eps=float(eps),
                                     layer=l, snapshot=key, D_tail=float(D_tail[l]), D_last=float(dm.D[l, T - 1]),
                                     D_after_mean=float(dm.D[l, after].mean()),
                                     front01=int(fronts[0.01][l]), front1=int(fronts[0.1][l])))
    dcols = list(C.ID_COLS) + ["inject", "site", "dir_site", "inject_layer", "inject_pos", "inject_lang", "inject_unit",
                               "inject_token", "inject_klass", "direction", "dir_family", "eps", "lam",
                               "intercept", "r2", "n_fit_layers", "D_final_tail", "D_final_last", "D_max", "cos_final_last",
                               "front01_final", "front1_final", "front01_max", "front1_max", "n_spread01_final",
                               "n_spread1_final", "n_after", "logp_target_ref", "logp_target_pert", "B", "delta_norm",
                               "audit_rel_err"]
    pcols = ["tag", "family", "formulation", "K", "C", "inject", "direction", "dir_family", "eps", "layer", "snapshot",
             "D_tail", "D_last", "D_after_mean", "front01", "front1"]
    return pd.DataFrame(rows, columns=dcols), pd.DataFrame(prof, columns=pcols)


def phase_table(damage: pd.DataFrame) -> pd.DataFrame:
    keys = ["K", "C", "kind", "inject", "dir_family", "eps"]
    cols = keys + ["n", "lam_mean", "lam_sd", "D_final_tail_mean", "front01_final_mean", "n_spread01_final_mean", "B_mean", "B_sd"]
    if damage.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for key, g in damage.groupby(keys, sort=True):
        lam = g["lam"].to_numpy(dtype=float)
        rows.append(dict(zip(keys, key), n=int(len(g)),
                         lam_mean=float(np.nanmean(lam)) if np.isfinite(lam).any() else float("nan"),
                         lam_sd=float(np.nanstd(lam)) if np.isfinite(lam).any() else float("nan"),
                         D_final_tail_mean=float(g["D_final_tail"].mean()),
                         front01_final_mean=float(g["front01_final"].mean()),
                         n_spread01_final_mean=float(g["n_spread01_final"].mean()),
                         B_mean=float(g["B"].mean()), B_sd=float(g["B"].std(ddof=0))))
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------------------
# D2 logit lens, D9 cellular automaton
# --------------------------------------------------------------------------------------
def lens_tables(mb: ModelBundle, variants, caches: dict | None, lexicon: dict, lang_axis, threshold: float,
                enabled=True, mass_floor=CA_MASS_FLOOR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """D2 lens tables and the D9 symbol field.  ``mass_floor`` is the absolute marker-mass
    floor of :func:`lsne.dynamics.ca_symbolize` (``"auto"`` or a probability mass).

    ``caches`` may map tag -> ``all_block_inputs`` reference cache; ``None`` or a missing tag
    recomputes the reference (one forward per variant, held only for that variant).
    :func:`run` passes ``None``: keeping a cache per variant does not fit the device."""
    lens_rows, summ_rows, upd_rows, ca_rows, cal_rows = [], [], [], [], []
    markers = resolve_markers(mb, lexicon, lang_axis)
    floor = ca_mass_floor(mass_floor, len(markers["E"]) + len(markers["D"]), mb.vocab)
    caches = caches or {}
    for v in C.progress(variants, "D2/D9 lens + CA", enabled=enabled):
        T = int(v.n_tokens)
        ref = caches.get(v.tag)
        if ref is None:
            ref = run_with_cache(mb, C.ids_tensor(mb, v), "all_block_inputs")
        tail = C.tail_positions(v)
        last = logit_lens_trajectory(mb, ref, target_id=int(v.target_id), positions=[T - 1])
        nxt_pos = [p for p in tail if p < T - 1] or [max(0, T - 2)]
        nxt = logit_lens_trajectory(mb, ref, target_id=[int(v.ids[p + 1]) for p in nxt_pos], positions=nxt_pos)
        upd = update_norm_profile(ref)                                            # [L, T]
        kl_last, kl_tail = last["kl"][:, 0], nxt["kl"].mean(axis=1)
        for l, key in enumerate(last["layers"]):
            lens_rows.append(dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=int(v.K), C=str(v.C),
                                  layer=l, snapshot=key, kl_last=float(kl_last[l]),
                                  target_rank_last=float(last["target_rank"][l, 0]),
                                  target_logp_last=float(last["target_logp"][l, 0]), kl_tail_mean=float(kl_tail[l]),
                                  next_logp_tail_mean=float(nxt["target_logp"][l].mean()),
                                  next_rank_tail_mean=float(nxt["target_rank"][l].mean())))
            if l < upd.shape[0]:
                upd_rows.append(dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=int(v.K), C=str(v.C),
                                     layer=l, update_tail_mean=float(upd[l, tail].mean()) if tail else float("nan"),
                                     update_last=float(upd[l, T - 1]), update_all_mean=float(upd[l].mean())))
        summ_rows.append(dict(C.variant_row(v), nmi_last=nonmonotonicity_index(kl_last), nmi_tail=nonmonotonicity_index(kl_tail),
                              kl_last_max=float(kl_last.max()), layer_of_max_kl_last=int(np.argmax(kl_last)),
                              kl_last_first=float(kl_last[0]), kl_tail_first=float(kl_tail[0]),
                              target_rank_final=float(last["target_rank"][-1, 0]),
                              first_layer_target_top10=int(np.argmax(last["target_rank"][:, 0] < 10))
                              if (last["target_rank"][:, 0] < 10).any() else -1,
                              update_tail_mean=float(upd[:, tail].mean()) if tail else float("nan")))
        # D9: symbol field from the marker masses with the absolute floor (sites below -> N)
        mE, mD = ca_marker_mass(mb, ref, markers)
        tot = mE + mD
        S = symbolize_masses(mE, mD, (threshold,), floor)
        full = ca_complexity(S)
        tl = ca_complexity(S[:, tail]) if len(tail) >= 2 else {k: float("nan") for k in full}
        ca_rows.append(dict(C.variant_row(v), **{k: float(val) for k, val in full.items()},
                            **{f"{k}_tail": float(val) for k, val in tl.items()},
                            frac_N=float((S == 0).mean()), frac_E=float((S == 1).mean()), frac_D=float((S == 2).mean()),
                            frac_D_tail=float((S[:, tail] == 2).mean()) if tail else float("nan"),
                            frac_D_final=float((S[-1] == 2).mean()), frac_D_final_tail=float((S[-1, tail] == 2).mean()) if tail else float("nan"),
                            mass_floor=float(floor), frac_below_floor=float((tot < floor).mean()),
                            marker_mass_median=float(np.median(tot))))
        for l in range(S.shape[0]):
            cal_rows.append(dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=int(v.K), C=str(v.C), layer=l,
                                 frac_N=float((S[l] == 0).mean()), frac_E=float((S[l] == 1).mean()),
                                 frac_D=float((S[l] == 2).mean()),
                                 frac_D_tail=float((S[l, tail] == 2).mean()) if tail else float("nan"),
                                 activity=float((S[l] != S[l - 1]).mean()) if l > 0 else float("nan"),
                                 marker_mass_median=float(np.median(tot[l])), frac_below_floor=float((tot[l] < floor).mean())))
        del ref
    return (pd.DataFrame(lens_rows), pd.DataFrame(summ_rows), pd.DataFrame(upd_rows), pd.DataFrame(ca_rows),
            pd.DataFrame(cal_rows))


# --------------------------------------------------------------------------------------
# D3-D7 generation diagnostics
# --------------------------------------------------------------------------------------
def _projection_order_parameter(mb: ModelBundle, prompt_ids: torch.Tensor, tokens: np.ndarray, site: HookSite,
                                axis: torch.Tensor) -> np.ndarray:
    """Per sample: mean projection onto ``axis`` of the (unsteered) residuals at ``site``
    over the generated positions of ``prompt + tokens``."""
    T0 = int(prompt_ids.shape[1])
    out = np.full(tokens.shape[0], np.nan)
    ax = axis.to(mb.device)
    for b in range(tokens.shape[0]):
        full = torch.cat([prompt_ids[0], torch.as_tensor(tokens[b], dtype=torch.long, device=mb.device)])[None, :]
        with torch.no_grad():
            h = run_with_cache(mb, full, [site]).resid[site.key][0, T0:].float()
        out[b] = float((h @ ax).mean())
    return out


def generation_tables(mb: ModelBundle, variants, bank: C.DirectionBank, dir_names, cfg: RunConfig, limits: C.Limits,
                      site: HookSite, enabled=True) -> dict[str, pd.DataFrame]:
    hy, hyc, rx, rxc, sw, su, fd = [], [], [], [], [], [], []
    eps_max = float(max(cfg.doses_steer)) if cfg.doses_steer else 1.0
    steer = [float(e) for e in cfg.doses_steer]
    lang_axis = bank["lang_dom"].vec if "lang_dom" in bank else None
    todo = [(v, n) for v in variants for n in dir_names]
    n_null = int(getattr(limits, "gen_n_null", HYST_N_NULL))
    for v, name in C.progress(todo, "D3-D7 generation", enabled=enabled):
        ids = C.ids_tensor(mb, v)
        vec = bank[name].vec
        idc = dict(C.variant_row(v), direction=name)
        # D3 (null: n_null independent restarts per level with their own seeds)
        h = hysteresis(mb, ids, site, vec, eps_max, n_steps=limits.gen_n_steps, tokens_per_step=limits.gen_tokens_per_step,
                       n_samples=cfg.gen_n_samples, seed=cfg.seed, temperature=cfg.gen_temperature, n_null=n_null)
        hy.append(dict(idc, eps_max=eps_max, n_steps=limits.gen_n_steps, tokens_per_step=limits.gen_tokens_per_step,
                       area=float(h["area"]), null_area=float(h["null_area"]), area_minus_null=float(h["area"] - h["null_area"]),
                       null_area_sd=float(h["null_area_sd"]), area_z=float(h["area_z"]), n_null=int(h["n_null"]),
                       n_null_pairs=int(len(h["null_areas"])),
                       up_mean=float(np.nanmean(h["up"])), down_mean=float(np.nanmean(h["down"])),
                       restart_mean=float(np.nanmean(h["restarts"])), fD_total=float(block_fraction(h["labels"])[0]),
                       n_switches_total=int(sum(len(s) for s in language_trajectory(mb.tokenizer, h["tokens"])["switches"]))))
        cur = h["curves"].copy()
        for k, val in idc.items():
            cur[k] = val
        hyc.append(cur[list(idc) + [c for c in h["curves"].columns if c not in idc]])
        s = switch_interval_stats(h["labels"])
        sw.append(dict(idc, source="hysteresis", eps=eps_max, **{k: s[k] for k in s if k != "intervals"}))
        # D4
        for eps in steer:
            r = relaxation(mb, ids, site, vec, eps, t_on=limits.gen_t_on, n_new=cfg.gen_n_new, n_samples=cfg.gen_n_samples,
                           seed=cfg.seed, temperature=cfg.gen_temperature, n_boot=limits.n_boot_tau)
            pre = r["fD_t"][: limits.gen_t_on]
            post = r["fD_t"][limits.gen_t_on:]
            fit = r["fit"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                rx.append(dict(idc, eps=eps, t_on=limits.gen_t_on, n_new=cfg.gen_n_new, tau=float(r["tau"]),
                               tau_lo=float(r["tau_ci"][0]), tau_hi=float(r["tau_ci"][1]),
                               tau_raw=float(fit["tau_raw"]), tau_censored=bool(fit["tau_censored"]),
                               identifiable=bool(fit["identifiable"]), reason=str(fit["reason"]),
                               a=float(fit["a"]), b=float(fit["b"]), b_se=float(fit["b_se"]), r2=float(fit["r2"]),
                               f_pvalue=float(fit["f_pvalue"]), n_fit=int(fit["n"]), n_eff=float(fit["n_eff"]),
                               b_lo=float(r["b_ci"][0]), b_hi=float(r["b_ci"][1]),
                               b_ci_excludes_zero=bool(r["b_ci_excludes_zero"]),
                               fD_pre=float(np.nanmean(pre)) if pre.size else float("nan"),
                               fD_post=float(np.nanmean(post)) if post.size else float("nan"),
                               fD_final=float(r["fD_t"][-1]), entropy_pre=float(r["entropy_t"][: limits.gen_t_on].mean()) if limits.gen_t_on > 0 else float("nan"),
                               entropy_post=float(r["entropy_t"][limits.gen_t_on:].mean()) if post.size else float("nan")))
            for t in range(len(r["t"])):
                rxc.append(dict(idc, eps=eps, t=int(t), fD_t=float(r["fD_t"][t]), entropy_t=float(r["entropy_t"][t]),
                                eps_t=float(r["eps_t"][t])))
            s = switch_interval_stats(r["labels"])
            sw.append(dict(idc, source="relaxation", eps=eps, **{k: s[k] for k in s if k != "intervals"}))
        # D6 / D7: constant-dose runs
        fD_by_eps: dict[float, np.ndarray] = {}
        proj_by_eps: dict[float, np.ndarray] = {}
        for i, eps in enumerate([0.0] + steer):
            g = generate_steered(mb, ids, site, vec, schedule_const(eps), n_new=cfg.gen_n_new, n_samples=cfg.gen_n_samples,
                                 temperature=cfg.gen_temperature, seed=cfg.seed + 2000 + i)
            traj = language_trajectory(mb.tokenizer, g["tokens"])
            fD_by_eps[eps] = np.array([block_fraction(traj["labels"][b])[0] for b in range(traj["labels"].shape[0])])
            if lang_axis is not None:
                proj_by_eps[eps] = _projection_order_parameter(mb, ids, g["tokens"].numpy(), site, lang_axis)
            s = switch_interval_stats(traj["labels"])
            sw.append(dict(idc, source="const", eps=eps, **{k: s[k] for k in s if k != "intervals"}))
        chi = susceptibility(fD_by_eps)
        chi_proj = susceptibility(proj_by_eps) if proj_by_eps else None
        for i, row in chi.iterrows():
            su.append(dict(idc, order_parameter="fD", **{k: float(row[k]) for k in ("eps", "n", "mean", "var", "sd", "chi")}))
        if chi_proj is not None:
            for i, row in chi_proj.iterrows():
                su.append(dict(idc, order_parameter="proj_lang_dom", **{k: float(row[k]) for k in ("eps", "n", "mean", "var", "sd", "chi")}))
        chi0 = float(chi["chi"].iloc[0]) if len(chi) else float("nan")
        chi1 = float(chi["chi"].iloc[1]) if len(chi) > 1 else float("nan")
        fdt_row = dict(idc, n_samples=cfg.gen_n_samples, var0_fD=float(np.var(fD_by_eps[0.0], ddof=1)) if len(fD_by_eps[0.0]) > 1 else float("nan"),
                       chi_fD_0=chi0, chi_fD_1=chi1, T_eff_fD_0=fdt_ratio(fD_by_eps[0.0], chi0), T_eff_fD_1=fdt_ratio(fD_by_eps[0.0], chi1),
                       eps_1=float(chi["eps"].iloc[1]) if len(chi) > 1 else float("nan"))
        if chi_proj is not None:
            cp0 = float(chi_proj["chi"].iloc[0]) if len(chi_proj) else float("nan")
            fdt_row.update(var0_proj=float(np.var(proj_by_eps[0.0], ddof=1)) if len(proj_by_eps[0.0]) > 1 else float("nan"),
                           chi_proj_0=cp0, T_eff_proj_0=fdt_ratio(proj_by_eps[0.0], cp0))
        fd.append(fdt_row)
    return dict(hysteresis=pd.DataFrame(hy), hysteresis_curves=(pd.concat(hyc, ignore_index=True) if hyc else pd.DataFrame()),
                relaxation=pd.DataFrame(rx), relaxation_curves=pd.DataFrame(rxc), switch_stats=pd.DataFrame(sw),
                susceptibility=pd.DataFrame(su), fdt=pd.DataFrame(fd))


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """D1-D9 on the lattice (all variants) and in generation (core subset)."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    allv = C.all_variants(variants, original)
    site = C.dossier_site(mb)
    dsite = C.depth_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original, sae=kw.get("sae"))
    dbank = depth_bank(cfg, mb, variants, dsite, bank, ctx=ctx, limits=limits, original=original)
    banks = {site.key: bank, dsite.key: dbank}
    if dbank is not bank:
        C.save_bank(dbank, run_dir, name="directions_depth")

    lattice_dirs = bank.names(LATTICE_FAMILIES)
    if limits.limited:
        rand = bank.rand_names[:8]
        lattice_dirs = [n for n in lattice_dirs if C.dir_family(n) != "rand"] + rand
    doses = tuple(float(e) for e in limits.damage_doses) + tuple(float(e) for e in cfg.doses_steer)
    damage_batch = int(getattr(limits, "damage_batch_size", DAMAGE_BATCH_SIZE))
    damage, profile = damage_tables(mb, allv, banks, lattice_dirs, doses, enabled=limits.progress,
                                    batch_size=damage_batch)
    lexicon = C.marker_lexicon(mb.tokenizer)
    lens, lens_summary, upd, ca, ca_layers = lens_tables(mb, allv, None, lexicon, bank.get("lang_dom"),
                                                         limits.ca_threshold, enabled=limits.progress,
                                                         mass_floor=CA_MASS_FLOOR)

    gen_dirs = bank.names(("lang_dom",) if limits.limited else GEN_FAMILIES)
    core = C.core_subset(allv)
    gen = generation_tables(mb, core, bank, gen_dirs, cfg, limits, site, enabled=limits.progress)

    tables = {"damage": damage, "damage_profile": profile, "damage_phase": phase_table(damage),
              "lens": lens, "lens_summary": lens_summary, "update_norms": upd, "ca": ca, "ca_layers": ca_layers}
    tables.update(gen)
    markers = resolve_markers(mb, lexicon, bank.get("lang_dom"))
    return C.finish_stage(run_dir, "D", cfg, mb, tables, limits=limits, t0=t0, site=site, depth_site=dsite,
                          depth_position="lsne.stimuli.last_switched_position",
                          bank_source=bank.meta.get("source", ""), depth_bank_source=dbank.meta.get("source", ""),
                          depth_directions=[n for n in lattice_dirs if n in dbank],
                          depth_directions_missing=[n for n in lattice_dirs if n not in dbank],
                          cos_dossier_vs_depth={n: cosine(bank[n].vec, dbank[n].vec) for n in lattice_dirs if n in dbank},
                          lattice_directions=lattice_dirs, generation_directions=gen_dirs, doses=list(doses),
                          marker_lexicon=lexicon,
                          ca=dict(threshold=float(limits.ca_threshold), mass_floor=CA_MASS_FLOOR,
                                  mass_floor_value=ca_mass_floor(CA_MASS_FLOOR, len(markers["E"]) + len(markers["D"]), mb.vocab)),
                          n_variants=len(variants), n_core=len(core),
                          gen=dict(n_new=cfg.gen_n_new, n_samples=cfg.gen_n_samples, n_steps=limits.gen_n_steps,
                                   tokens_per_step=limits.gen_tokens_per_step, t_on=limits.gen_t_on,
                                   n_null=int(getattr(limits, "gen_n_null", HYST_N_NULL)),
                                   doses_steer=list(cfg.doses_steer)))
