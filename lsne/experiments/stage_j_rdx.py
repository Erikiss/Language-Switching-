"""Stage J - RDX difference explanation, empirical Pile difficulty ranking, sharing
index and network activity (DESIGN section 4.8 a-c; PREREGISTRATION contrasts 15-16).

Tables
------
pile_difficulty   contrast 16: one row per sampled Pile training sequence
                  (:func:`lsne.pile_domains.sample_sequence_difficulty`, ``Limits.j_n_pile``
                  sequences by range request, cached under ``cache/pile_nll``): subfield
                  (keyword profile, :data:`lsne.pile_domains.PILE_MARKERS`), mean token NLL
                  under the model, plus ``subfield_lexicon`` = the lexicon-domain
                  classification of the same text; empty when ``LSNE_OFFLINE`` is set or
                  the fetch fails (manifest ``substages``)
pile_ranking      :func:`difficulty_ranking` (hardest first, bootstrap CI over sequences)
pile_rank_check   one row (:func:`misfolding_rank_check`): rank of ``protein_misfolding``,
                  ``in_hardest_third``, ``hardest`` and the resulting ``counterpole``
                  (``protein_misfolding`` when it is in the hardest third, else the
                  hardest subfield - the pre-registered replacement rule)
rdx               contrast 15: :func:`lsne.rdx.rdx_table` of the difference vectors
                  ``culture_politics_vs_misfolding`` (K=0 variants of F13-F16 vs F19-F20),
                  ``law_vs_culture_politics``, ``E_vs_D`` (all-English vs all-German
                  rendering of every formulation) and ``K0_vs_K8`` (K=0 vs K=8/C1) at the
                  pre-registered grid sites (block inputs; ``Limits.j_max_sites`` keeps the
                  last ones), readouts ``last`` and ``mean``, against the bases
                  ``domain_axes`` (dom_law / dom_culture_politics / dom_protein_misfolding
                  fitted at that site), ``sae`` (decoder rows of the language-selective
                  features at the SAE's site, or the most active latents of a synthetic
                  placeholder), ``heads`` (head contributions of the block at the site) and
                  ``pca`` (top ``Limits.j_n_pca`` components of the pooled contrast states);
                  z against ``Limits.j_n_random_bases`` random bases of the same size.
                  ``preregistered`` = culture_politics_vs_misfolding x domain_axes x last
rdx_summary       per (contrast, basis, readout): sites, share of sites with z >= 2,
                  ``min_effect_met`` (>= half of the sites; contrast 15).  The cross-domain
                  contrast is evaluated against the misfolding pole (``evaluated_pole``); rule
                  16 (``pile_rank_check``) may replace the counterpole by the hardest Pile
                  subfield (``counterpole_rule16``).  ``preregistered_form`` is True only when
                  both agree, False when rule 16 fired (contrast 15 then is NOT in its
                  pre-registered form: the misfolding result is exploratory), empty when the
                  ranking did not run; the manifest repeats this under ``rule16``
rdx_stability     :func:`lsne.rdx.stability` across families (E_vs_D, K0_vs_K8) and across
                  family pairs (cross-domain contrasts)
sharing           :func:`lsne.sharing.sharing_table` at the dossier site: Jaccard sharing
                  index of the active TopK latents between the conditions (family domain
                  groups of the K=0 variants, and K0 vs K8C1), network activity per
                  condition and the head-profile cosines over the layer grid;
                  ``sae_source`` marks the SAE (``synthetic_random`` = placeholder)
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..atlas import domain_axis, load_domains
from ..config import RunConfig, HookSite
from ..pile_domains import (classify_sequence, difficulty_ranking, lexicon_markers, misfolding_rank_check,
                            sample_sequence_difficulty)
from ..rdx import bases_from_heads, bases_from_pca, bases_from_sae, rdx_table, stability, variant_states_by_site
from ..sharing import sharing_table
from ..model import ModelBundle
from . import common as C
from .stage_g_geometry import realization_clouds
from .stage_i_compass import DOMAIN_AXES

PREREG_CONTRAST, PREREG_BASIS, PREREG_READOUT = "culture_politics_vs_misfolding", "domain_axes", "last"
#: The pole each cross-domain contrast is evaluated against (contrast 15: the misfolding families
#: F19-F20); rule 16 may name another counterpole (``pile_rank_check.counterpole``).
EVALUATED_POLES = {"culture_politics_vs_misfolding": "protein_misfolding"}
PREREG_Z = 2.0
RDX_COLS = ["contrast", "unit", "basis", "site", "layer", "readout", "k", "explained", "random_mean", "random_sd", "z", "p_random",
            "delta_norm", "n_a", "n_b", "top_component", "top_share", "sae_source", "preregistered", "z_ge_2"]


# --------------------------------------------------------------------------------------
# contrast 16: Pile difficulty
# --------------------------------------------------------------------------------------
def pile_tables(mb: ModelBundle, limits: C.Limits, seed: int, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols_check = ["subfield", "rank", "n_subfields", "present", "in_hardest_third", "hardest", "counterpole", "n_sequences"]
    if C.offline():
        raise RuntimeError("LSNE_OFFLINE is set: the Pile difficulty ranking needs range requests")
    cdir = Path(C.cache_dir("pile_nll"))
    df = sample_sequence_difficulty(mb, n_sequences=int(limits.j_n_pile), seed=int(seed), cache_dir=cdir,
                                    max_len=int(limits.j_pile_max_len), progress=False)
    df = df.copy()
    # lexicon-domain classification of the same sequences (cached npy files)
    from ..pile_fetch import fetch_sequence
    markers = lexicon_markers(load_domains())
    lex = []
    for i in df["seq_index"]:
        try:
            seq = fetch_sequence(int(i), cache_dir=cdir / "seqs")
            lex.append(classify_sequence(mb.tokenizer, np.asarray(seq)[: int(limits.j_pile_max_len)], markers=markers))
        except Exception:  # noqa: BLE001
            lex.append("")
    df["subfield_lexicon"] = lex
    rank = difficulty_ranking(df, n_boot=int(n_boot), seed=int(seed))
    chk = misfolding_rank_check(rank) if len(rank) else dict(subfield="protein_misfolding", rank=None, n_subfields=0, present=False,
                                                            in_hardest_third=False, hardest=None)
    chk = dict(chk)
    chk["counterpole"] = "protein_misfolding" if chk.get("in_hardest_third") else chk.get("hardest")
    chk["n_sequences"] = int(len(df))
    return df, rank, pd.DataFrame([chk], columns=cols_check)


# --------------------------------------------------------------------------------------
# contrast 15: RDX
# --------------------------------------------------------------------------------------
def rdx_sites(mb: ModelBundle, max_sites: Optional[int]) -> list[HookSite]:
    sites = C.grid_sites(mb)
    if max_sites is not None:
        sites = sites[-int(max_sites):]
    return sites


def contrast_sets(variants: Sequence, mono_D: dict) -> tuple[dict[str, tuple[list, list]], dict[str, dict[str, tuple[list, list]]]]:
    """``(pooled contrasts, per-unit contrasts)``; a unit is a family (within-family
    contrasts) or a ``culture family|misfold family`` pair (cross-domain contrasts)."""
    k0 = C.k_subset(variants, 0)
    k8 = C.k_subset(variants, 8, "C1")
    by_group: dict[str, list] = {}
    for v in k0:
        by_group.setdefault(C.family_domain(v.family_id), []).append(v)
    pooled: dict[str, tuple[list, list]] = {}
    units: dict[str, dict[str, tuple[list, list]]] = {}
    for name, (ga, gb) in (("culture_politics_vs_misfolding", ("culture_politics", "protein_misfolding")),
                           ("law_vs_culture_politics", ("law", "culture_politics"))):
        A, B = by_group.get(ga, []), by_group.get(gb, [])
        if A and B:
            pooled[name] = (A, B)
            per: dict[str, tuple[list, list]] = {}
            for fa in sorted({v.family_id for v in A}):
                for fb in sorted({v.family_id for v in B}):
                    per[f"{fa}|{fb}"] = ([v for v in A if v.family_id == fa], [v for v in B if v.family_id == fb])
            units[name] = per
    E = [v for v in k0 if (str(v.family_id), str(v.formulation)) in mono_D]
    D = [mono_D[(str(v.family_id), str(v.formulation))] for v in E]
    if E and D:
        pooled["E_vs_D"] = (E, D)
        units["E_vs_D"] = {fid: ([v for v in E if v.family_id == fid], [d for v, d in zip(E, D) if v.family_id == fid])
                           for fid in sorted({v.family_id for v in E})}
    fams8 = {v.family_id for v in k8}
    A = [v for v in k0 if v.family_id in fams8]
    if A and k8:
        pooled["K0_vs_K8"] = (A, k8)
        units["K0_vs_K8"] = {fid: ([v for v in A if v.family_id == fid], [v for v in k8 if v.family_id == fid]) for fid in sorted(fams8)}
    return pooled, units


def sae_basis_for(sae, X: np.ndarray, n: int) -> tuple[np.ndarray, list[str]]:
    """Decoder rows of the ``n`` latents with the largest mean activation on ``X``."""
    with torch.no_grad():
        _, _, pre = sae.encode(torch.as_tensor(X, dtype=torch.float32))
    act = torch.relu(pre).mean(0)
    idx = torch.argsort(act, descending=True)[: int(n)].tolist()
    return bases_from_sae(sae, idx)


def build_bases(mb: ModelBundle, sites: Sequence[HookSite], pooled: dict, sae, sae_source: str, bank, domains: dict,
                limits: C.Limits, ctx, sub: C.Substages) -> dict[str, dict]:
    """Per-site candidate bases ``{basis name: {site key: (M, names)}}``."""
    axes_by_site: dict[str, list] = {}
    cached = ctx.get("domain_axes") if isinstance(ctx, dict) else None
    for s in sites:
        dirs = []
        for name, (dom, con) in DOMAIN_AXES.items():
            if cached and name in cached and cached[name].site == s:
                dirs.append(cached[name])
                continue
            d = sub.run(f"axis:{name}@{s.key}", lambda dom=dom, con=con, s=s: domain_axis(
                mb, mb.tokenizer, dom, s, contrast=con, lang="E", n_templates=limits.i_n_templates, domains=domains,
                max_words=limits.i_max_words), None)
            if d is not None:
                d.name = name
                dirs.append(d)
        if dirs:
            axes_by_site[s.key] = dirs
    bases: dict[str, dict] = {"domain_axes": axes_by_site}
    # representative input for the head basis: the first variant of the first contrast
    first = next((pair[0][0] for pair in pooled.values() if pair[0]), None)
    if first is not None:
        bases["heads"] = {s.key: bases_from_heads(mb, first.ids, int(s.layer)) for s in sites}
    if sae is not None:
        feats: list[int] = []
        info = bank.meta.get("sae", {}) if bank is not None and isinstance(bank.meta, dict) else {}
        feats = [int(f) for f in (info.get("features_D") or []) + (info.get("features_E") or [])]
        sae_sites = sites if sae_source == "synthetic_random" else [s for s in sites if bank is not None and s == bank.site]
        if feats:
            bases["sae"] = {s.key: bases_from_sae(sae, feats) for s in sae_sites}
        else:
            bases["sae"] = {"_lazy": True, "_sites": [s.key for s in sae_sites]}
    return bases


def rdx_rows(mb: ModelBundle, contrasts: dict, bases: dict, sites: Sequence[HookSite], readouts: Sequence[str], sae, sae_source: str,
             n_random: int, n_pca: int, seed: int, unit: str = "pooled") -> pd.DataFrame:
    frames = []
    for cname, (A, B) in contrasts.items():
        for ro in readouts:
            SA = variant_states_by_site(mb, A, sites, ro)
            SB = variant_states_by_site(mb, B, sites, ro)
            for s in sites:
                key = s.key
                a, b = SA[key], SB[key]
                if a.shape[0] == 0 or b.shape[0] == 0:
                    continue
                X = np.concatenate([a, b], 0)
                site_bases: dict[str, object] = {}
                for bname, bval in bases.items():
                    if bname == "sae" and isinstance(bval, dict) and bval.get("_lazy"):
                        if key in bval["_sites"] and sae is not None:
                            site_bases["sae"] = sae_basis_for(sae, X, n_pca)
                        continue
                    if isinstance(bval, dict) and key in bval:
                        site_bases[bname] = bval[key]
                site_bases["pca"] = bases_from_pca(X, int(n_pca))
                df = rdx_table(None, {cname: ({key: a}, {key: b})}, site_bases, sites=[key], n_random=int(n_random), seed=int(seed))
                df["readout"] = ro
                frames.append(df)
    if not frames:
        return pd.DataFrame(columns=RDX_COLS)
    out = pd.concat(frames, ignore_index=True)
    out["unit"] = unit
    out["layer"] = [int(str(k).split(".")[0][1:]) for k in out["site"]]
    out["sae_source"] = np.where(out["basis"] == "sae", sae_source, "")
    out["preregistered"] = (out["contrast"] == PREREG_CONTRAST) & (out["basis"] == PREREG_BASIS) & (out["readout"] == PREREG_READOUT)
    out["z_ge_2"] = out["z"] >= PREREG_Z
    return out[RDX_COLS]


def rule16_status(counterpole: Optional[str], evaluated_pole: str = "protein_misfolding") -> dict:
    """Contrast 15 against rule 16: ``counterpole_rule16`` (the counterpole of
    ``pile_rank_check``, None when the ranking did not run), ``evaluated_pole`` (the pole the RDX
    contrast was computed with) and ``preregistered_form`` (True when they agree, False when rule
    16 replaced the counterpole, None when unknown)."""
    cp = None if counterpole is None or (isinstance(counterpole, float) and not np.isfinite(counterpole)) or str(counterpole) in ("", "None", "nan") \
        else str(counterpole)
    form = None if cp is None else bool(cp == evaluated_pole)
    note = ("rule 16 did not run (no Pile ranking): the form of contrast 15 is unknown" if cp is None else
            "contrast 15 evaluated in its pre-registered form" if form else
            f"rule 16 replaced the counterpole by {cp!r}, but contrast 15 was evaluated against {evaluated_pole!r}: "
            "NOT the pre-registered form (exploratory)")
    return dict(counterpole_rule16=cp, evaluated_pole=evaluated_pole, preregistered_form=form, note=note)


def rdx_summary(rdx: pd.DataFrame, counterpole: Optional[str] = None) -> pd.DataFrame:
    """Per (contrast, basis, readout) of the pooled rows: share of sites with z >= 2 and
    ``min_effect_met``; the cross-domain contrasts carry ``evaluated_pole``,
    ``counterpole_rule16`` and ``preregistered_form`` (:func:`rule16_status`)."""
    cols = ["contrast", "basis", "readout", "n_sites", "n_z_ge_2", "frac_z_ge_2", "z_mean", "z_min", "explained_mean", "preregistered",
            "min_effect_met", "evaluated_pole", "counterpole_rule16", "preregistered_form"]
    if rdx.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (c, b, ro), g in rdx[rdx["unit"] == "pooled"].groupby(["contrast", "basis", "readout"], sort=True):
        z = g["z"].to_numpy(dtype=float)
        ok = np.isfinite(z)
        n = int(ok.sum())
        n2 = int((z[ok] >= PREREG_Z).sum())
        pole = EVALUATED_POLES.get(str(c))
        st = rule16_status(counterpole, pole) if pole else dict(evaluated_pole="", counterpole_rule16=None, preregistered_form=None)
        rows.append(dict(contrast=c, basis=b, readout=ro, n_sites=n, n_z_ge_2=n2, frac_z_ge_2=(n2 / n) if n else np.nan,
                         z_mean=float(np.nanmean(z)) if ok.any() else np.nan, z_min=float(np.nanmin(z)) if ok.any() else np.nan,
                         explained_mean=float(np.nanmean(g["explained"])), preregistered=bool(g["preregistered"].any()),
                         min_effect_met=bool(n and n2 >= np.ceil(n / 2)), evaluated_pole=st["evaluated_pole"],
                         counterpole_rule16=st["counterpole_rule16"], preregistered_form=st["preregistered_form"]))
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Pile difficulty ranking, RDX and sharing; see the module docstring."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    sub = C.Substages("J", verbose=limits.progress)
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    site = C.dossier_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original, sae=kw.get("sae"))
    sae, sae_source = C.get_sae(mb, bank, limits, kw)
    empty = pd.DataFrame()
    domains = load_domains()

    pile, ranking, check = sub.run("pile_difficulty", lambda: pile_tables(mb, limits, cfg.seed, limits.n_boot), (empty, empty, empty))

    mono_D = ctx.get("g_mono_D") if isinstance(ctx, dict) else None
    if not mono_D:
        _g, mono_D, _c = realization_clouds(C.select_families(cfg), mb.tokenizer, (0,), ("C1",), 1, cfg.seed)
    pooled, units = contrast_sets(variants, mono_D)
    if limits.j_max_variants_per_group:
        n = int(limits.j_max_variants_per_group)
        pooled = {k: (a[:n], b[:n]) for k, (a, b) in pooled.items()}
    sites = rdx_sites(mb, getattr(limits, "j_max_sites", None))
    bases = sub.run("bases", lambda: build_bases(mb, sites, pooled, sae, sae_source, bank, domains, limits, ctx, sub), {})
    readouts = ("last", "mean")
    rdx = sub.run("rdx", lambda: rdx_rows(mb, pooled, bases, sites, readouts, sae, sae_source, limits.j_n_random_bases, limits.j_n_pca, cfg.seed),
                  pd.DataFrame(columns=RDX_COLS))

    def _stab():
        frames, by_unit = [], {}
        for cname, per in units.items():
            for u, (A, B) in per.items():
                df = rdx_rows(mb, {cname: (A, B)}, bases, sites, ("last",), sae, sae_source, limits.j_n_random_bases, limits.j_n_pca,
                              cfg.seed, unit=u)
                if len(df):
                    frames.append(df)
                    by_unit.setdefault(u, []).append(df)
        st = stability({u: pd.concat(v, ignore_index=True) for u, v in by_unit.items()}) if by_unit else pd.DataFrame()
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=RDX_COLS), st
    rdx_units, stab = sub.run("stability", _stab, (pd.DataFrame(columns=RDX_COLS), empty))
    rdx_all = pd.concat([rdx, rdx_units], ignore_index=True) if len(rdx_units) else rdx
    counterpole = check["counterpole"].iloc[0] if len(check) and "counterpole" in check.columns else None
    rule16 = dict(rule16_status(counterpole, EVALUATED_POLES[PREREG_CONTRAST]), contrast_15_evaluated=PREREG_CONTRAST in pooled)
    if rule16["preregistered_form"] is False and rule16["contrast_15_evaluated"]:
        warnings.warn(f"stage J: {rule16['note']}")
    summary = sub.run("rdx_summary", lambda: rdx_summary(rdx_all, counterpole), empty)

    def _sharing():
        if sae is None:
            raise RuntimeError("no SAE available (use --use-sae or a synthetic placeholder in smoke runs)")
        k0 = C.k_subset(variants, 0)
        groups: dict[str, list] = {}
        for v in k0:
            groups.setdefault(C.family_domain(v.family_id), []).append(v)
        groups["K0"] = k0
        groups["K8C1"] = C.k_subset(variants, 8, "C1")
        groups = {k: v for k, v in groups.items() if v}
        tab = sharing_table(mb, sae, groups, site=site, readout="last", layer_grid=list(C.prereg_layer_grid(mb.n_layers)))
        tab["sae_source"] = sae_source
        return tab
    sharing = sub.run("sharing", _sharing, empty)

    tables = {"pile_difficulty": pile, "pile_ranking": ranking, "pile_rank_check": check, "rdx": rdx_all, "rdx_summary": summary,
              "rdx_stability": stab, "sharing": sharing}
    sub.write_tracebacks(run_dir)
    return C.finish_stage(run_dir, "J", cfg, mb, tables, limits=limits, t0=t0, site=site, sae_source=sae_source,
                          rdx_sites=[s.key for s in sites], contrasts={k: (len(a), len(b)) for k, (a, b) in pooled.items()},
                          bases=list(bases), readouts=list(readouts), n_random_bases=limits.j_n_random_bases,
                          pile=dict(n=limits.j_n_pile, max_len=limits.j_pile_max_len, offline=C.offline(),
                                    check=(check.iloc[0].to_dict() if len(check) else None)),
                          substages=sub.report(), rule16=rule16,
                          prereg=dict(contrast_15="rdx_summary row contrast=culture_politics_vs_misfolding, basis=domain_axes, readout=last "
                                                  "(pre-registered form only when rdx_summary.preregistered_form is True: the rule-16 "
                                                  "counterpole equals the evaluated pole; see rule16)",
                                      contrast_16="pile_rank_check: rank / in_hardest_third / counterpole"))
