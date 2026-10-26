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
contrasts_boot       family cluster bootstrap of T per (K, C), of I per
                     (K, C, dir_family, sign, eps) and of the attribution excess
                     R_class - R_random per (K, C, layer, class) (``quantity="R_excess"``;
                     the row with ``class == "delim"`` is the pre-registered contrast 3
                     R_ws+punct+boundary - R_random, PREREGISTRATION.md section 3)
attribution          counterfactual position patching K=0 -> K=8/C1 and K=0 -> K=28/seg
                     (original family: K=0 -> largest K) on the block inputs of the
                     pre-registered layer grid :func:`attribution_layer_grid` ({2, 5, 8, 11}
                     for pythia-160m, DESIGN section 4.1), per token class of
                     :data:`lsne.experiments.common.PATCH_CLASSES` (incl. the union class
                     ``delim`` = ws + punct + boundary as one patch set) with a stratified
                     random-position null (:func:`lsne.patching.counterfactual_attribution`)
attribution_contrasts per pair, formulation, layer and class with a defined control:
                     R_excess = recovery - recovery_random_mean (the per-formulation rows
                     behind the ``R_excess`` bootstrap; ``preregistered`` marks ``delim``)
attribution_summary  mean recovery per (pair, site, class) over formulations
grad_x_act           gradient x activation of log p(target) per variant, site and position
grad_x_act_by_class  share of |grad x act| per token class
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch

from ..config import LAYER_GRID_160M, RunConfig, HookSite
from ..dynamics import last_logits
from ..model import Injector, ModelBundle, make_dose_delta, next_token_stats, run_with_cache
from ..patching import counterfactual_attribution, grad_x_act
from ..stats import cluster_bootstrap
from . import common as C

GRID_FAMILIES = ("ent", "lang_dom", "freq", "rand")
GRID_COLS = list(C.ID_COLS) + ["direction", "dir_family", "sign", "eps", "logp_0", "logp_delta", "B", "p_target",
                               "entropy_0", "entropy", "p_freq_0", "p_freq", "delta_norm", "audit_rel_err"]
#: Default rows (direction x sign x dose) per batched grid forward (``Limits.grid_batch_size``).
GRID_BATCH_SIZE = 64
#: Bump when the meaning of a checkpointed stage-C result changes (invalidates ``_resume``).
RESUME_VERSION = 1


# --------------------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------------------
class _GridDose:
    """Callable delta for :class:`~lsne.model.Injector`: row ``j`` of the chunk gets
    ``deltas[j]`` added at ``position`` (the sequential grid's delta, built from the
    variant's own baseline residual).  ``rel_err`` holds the per-row Soll/Ist record with
    the ``Injector`` semantics (intended = the delta, written = h_new - h in float32,
    ``||written - intended|| / ||intended||``) as a device tensor; ``audit`` marks the
    chunk for the Injector (no device sync)."""

    def __init__(self, deltas: torch.Tensor, position: int):
        self.deltas = deltas
        self.position = int(position)
        self.audit: dict = {}
        self.rel_err: torch.Tensor | None = None

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        if h.shape[0] != self.deltas.shape[0]:
            raise RuntimeError(f"chunk of {h.shape[0]} rows for {self.deltas.shape[0]} deltas")
        p = self.position
        h_new = h.clone()
        h_new[:, p, :] = h_new[:, p, :] + self.deltas.to(device=h.device, dtype=h.dtype)
        with torch.no_grad():
            intended = self.deltas.to(h.device).float()
            written = h_new[:, p, :].float() - h[:, p, :].float()
            self.rel_err = (written - intended).norm(dim=-1) / intended.norm(dim=-1).clamp_min(1e-30)
        self.audit = dict(n_rows=int(h.shape[0]), per_row=True, dtype=str(h.dtype))
        return h_new


def grid_rows(mb: ModelBundle, variants, bank: C.DirectionBank, site: HookSite, doses, dir_names,
              enabled: bool = True, batch_size: int = GRID_BATCH_SIZE) -> list[dict]:
    """Grid rows per variant x direction x sign x dose (row order: variant, direction,
    sign +1/-1, dose).  One ``run_with_cache`` per variant gives the baseline and the residual
    ``h_ref`` at ``(site, T-1)``; the dosed rows (``eps != 0``) of a variant then run in
    batched forwards of ``batch_size`` copies of the input, row ``j`` with its own delta
    ``make_dose_delta(h_ref, sign * dir, eps)`` added at ``T-1`` and the LM head on the last
    position only; the per-row statistics, delta norms and audit errors leave the device in
    one transfer per chunk.  Same rows as one forward per dose (float32 round-off of the
    batched kernels aside); ``eps == 0`` rows are the baseline."""
    freq_t = bank.freq_tensor(mb.device)
    batch_size = max(1, int(batch_size))
    rows: list[dict] = []
    for v in C.progress(variants, "C grid", enabled=enabled):
        ids = C.ids_tensor(mb, v)
        T = int(ids.shape[1])
        tid = int(v.target_id)
        with torch.no_grad():
            cache = run_with_cache(mb, ids, [site])
            base = next_token_stats(cache.logits[:, -1], tid, freq_t)
            h_ref = cache.resid[site.key][0, T - 1].detach()
        del cache
        b = {k: float(val[0]) for k, val in base.items()}
        specs = [(name, sign, float(eps)) for name in dir_names for sign in (1, -1) for eps in doses]
        todo = [i for i, (_, _, eps) in enumerate(specs) if eps != 0.0]
        out: dict[int, tuple[dict, float, float]] = {}
        for c0 in range(0, len(todo), batch_size):
            idx = todo[c0:c0 + batch_size]
            with torch.no_grad():
                deltas = torch.stack([make_dose_delta(h_ref, specs[i][1] * bank[specs[i][0]].vec.to(h_ref.device),
                                                      specs[i][2]).to(h_ref.dtype) for i in idx])          # [n, d]
                dose = _GridDose(deltas, T - 1)
                with Injector(mb, site, dose, mode="add"):
                    logits = last_logits(mb, ids.expand(len(idx), T).contiguous())                      # [n, V]
                st = next_token_stats(logits, tid, freq_t)
                keys = [k for k in ("logp_target", "p_target", "entropy", "p_freq") if k in st]
                vals = torch.stack([st[k].float() for k in keys] + [deltas.float().norm(dim=-1), dose.rel_err.float()])
                vals = vals.cpu().tolist()                                                              # one transfer
            del logits, st, deltas, dose
            for j, i in enumerate(idx):
                out[i] = ({k: vals[q][j] for q, k in enumerate(keys)}, vals[-2][j], vals[-1][j])
        for i, (name, sign, eps) in enumerate(specs):
            st, dn, err = out[i] if i in out else (b, 0.0, 0.0)
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


#: Columns of ``contrasts_boot``: T / I rows carry ``layer=-1`` and ``class=""``, the
#: attribution rows (``quantity="R_excess"``) carry ``dir_family=""``, ``sign=0``, ``eps=0``.
BOOT_COLS = ["quantity", "K", "C", "dir_family", "sign", "eps", "layer", "class", "mean", "ci_low", "ci_high", "se",
             "n_clusters", "n_rows"]


def bootstrap_contrasts(contrasts: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Family cluster bootstrap of T per (K, C) and of I per (K, C, dir_family, sign, eps)
    over the standard families (columns :data:`BOOT_COLS`)."""
    cols = BOOT_COLS
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
        rows.append(dict(quantity="T", K=int(K), C=Cc, dir_family="", sign=0, eps=0.0, layer=-1, **{"class": ""},
                         mean=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"], se=r["se"],
                         n_clusters=r["n_clusters"], n_rows=r["n_rows"]))
    for (K, Cc, fam, sign, eps), g in std[std["eps"] > 0].groupby(["K", "C", "dir_family", "sign", "eps"], sort=True):
        r = cluster_bootstrap(g, "I", "family", n_boot=n_boot, seed=seed)
        rows.append(dict(quantity="I", K=int(K), C=Cc, dir_family=fam, sign=int(sign), eps=float(eps), layer=-1,
                         **{"class": ""}, mean=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"], se=r["se"],
                         n_clusters=r["n_clusters"], n_rows=r["n_rows"]))
    return pd.DataFrame(rows, columns=cols)


ATT_CONTRAST_COLS = ["pair", "family", "formulation", "tag_mixed", "K", "C", "kind", "site", "layer", "class",
                     "preregistered", "n_positions", "recovery", "recovery_random_mean", "recovery_random_sd", "R_excess"]


def attribution_contrasts(att: pd.DataFrame) -> pd.DataFrame:
    """Per-formulation attribution excess ``R_excess = recovery - recovery_random_mean``
    for every (pair, layer, class) row of ``att`` whose stratified random control is
    defined (``control_defined``); ``preregistered`` marks the confirmatory class
    :data:`lsne.experiments.common.PREREG_PATCH_CLASS` (contrast 3,
    R_ws+punct+boundary - R_random).  Rows without a control (unmeasurable classes, or
    classes that exhaust their pools such as ``tail`` / ``all``) are left out: their
    excess is undefined, not zero."""
    if att.empty or "control_defined" not in att.columns:
        return pd.DataFrame(columns=ATT_CONTRAST_COLS)
    ok = att[att["control_defined"].astype(bool) & np.isfinite(att["recovery"].astype(float))
             & np.isfinite(att["recovery_random_mean"].astype(float))]
    if ok.empty:
        return pd.DataFrame(columns=ATT_CONTRAST_COLS)
    out = pd.DataFrame(dict(
        pair=ok["pair"].to_numpy(), family=ok["family"].to_numpy(), formulation=ok["formulation"].to_numpy(),
        tag_mixed=ok["tag_mixed"].to_numpy(), K=ok["K_mixed"].astype(int).to_numpy(), C=ok["C_mixed"].astype(str).to_numpy(),
        kind=(ok["kind"].to_numpy() if "kind" in ok.columns else np.full(len(ok), "standard")),
        site=ok["site"].to_numpy(), layer=ok["layer"].astype(int).to_numpy(), **{"class": ok["class"].to_numpy()},
        preregistered=(ok["class"] == C.PREREG_PATCH_CLASS).to_numpy(), n_positions=ok["n_positions"].astype(int).to_numpy(),
        recovery=ok["recovery"].astype(float).to_numpy(),
        recovery_random_mean=ok["recovery_random_mean"].astype(float).to_numpy(),
        recovery_random_sd=ok["recovery_random_sd"].astype(float).to_numpy()))
    out["R_excess"] = out["recovery"] - out["recovery_random_mean"]
    return out[ATT_CONTRAST_COLS].reset_index(drop=True)


def bootstrap_attribution(att_con: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Family cluster bootstrap of ``R_excess`` per (K, C, layer, class) over the standard
    families (``quantity="R_excess"``, columns :data:`BOOT_COLS`); the ``class == "delim"``
    rows are the pre-registered contrast 3 per layer of the attribution grid."""
    if att_con.empty:
        return pd.DataFrame(columns=BOOT_COLS)
    std = att_con[att_con["kind"] == "standard"]
    rows = []
    for (K, Cc, layer, cls), g in std.groupby(["K", "C", "layer", "class"], sort=True):
        r = cluster_bootstrap(g, "R_excess", "family", n_boot=n_boot, seed=seed)
        rows.append(dict(quantity="R_excess", K=int(K), C=str(Cc), dir_family="", sign=0, eps=0.0, layer=int(layer),
                         **{"class": str(cls)}, mean=r["mean"], ci_low=r["ci_low"], ci_high=r["ci_high"], se=r["se"],
                         n_clusters=r["n_clusters"], n_rows=r["n_rows"]))
    return pd.DataFrame(rows, columns=BOOT_COLS)


# --------------------------------------------------------------------------------------
# counterfactual attribution and gradient x activation
# --------------------------------------------------------------------------------------
def attribution_layer_grid(n_layers: int) -> tuple[int, ...]:
    """Pre-registered layer grid of the patching / grad x act analyses (DESIGN section 4.1).

    The last block of each quarter of the depth, ``ceil((i + 1) * n_layers / 4) - 1`` for
    ``i = 0..3``: ``(2, 5, 8, 11)`` for the 12 blocks of pythia-160m
    (= :data:`lsne.config.LAYER_GRID_160M`), ``(1, 2, 4, 5)`` for pythia-14m / 70m,
    ``(5, 11, 17, 23)`` for 24 blocks.  Block 0 is never part of the grid for
    ``n_layers >= 5``: its input is the token embedding, which is identical in both
    variants of a pair on the shared tail, so a patch there recovers nothing by
    construction.  (``lsne.config.layer_grid`` rounds ``i * (L - 1) / 3`` instead and
    yields ``(0, 4, 7, 11)`` for 12 blocks, which is not the pre-registered grid.)"""
    n_layers = int(n_layers)
    if n_layers <= 0:
        return ()
    grid = tuple(sorted({max(0, math.ceil((i + 1) * n_layers / 4) - 1) for i in range(4)}))
    if n_layers == 12:
        assert grid == tuple(LAYER_GRID_160M), (grid, LAYER_GRID_160M)
    return grid


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


def attribution_table(mb: ModelBundle, variants, sites, n_random: int, seed: int, enabled: bool = True,
                      pairs=None) -> pd.DataFrame:
    """Counterfactual attribution of every pair of :func:`attribution_pairs` (or of the
    given ``pairs``, e.g. one resume chunk of them)."""
    frames = []
    pairs = attribution_pairs(variants) if pairs is None else list(pairs)
    for name, v_mono, v_mixed in C.progress(pairs, "C attribution", enabled=enabled):
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
        df.insert(6, "kind", C.variant_kind(v_mixed))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["pair", "family", "formulation", "tag_mixed", "K_mixed", "C_mixed", "kind", "site",
                                     "layer", "class", "region", "n_positions", "n_tail", "n_nontail", "n_pool_tail",
                                     "n_pool_nontail", "n_null_subsets", "measurable", "control_defined", "logp_mono",
                                     "logp_mixed", "logp_patched", "recovery", "recovery_random_mean",
                                     "recovery_random_sd", "dst_positions"])
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
    """Grid B(x), contrasts T / I, counterfactual attribution and gradient x activation.

    Resumable (:class:`~lsne.experiments.common.Resume`, ``kw['resume']=False`` disables it):
    the grid, the attribution and gradient x activation are checkpointed per chunk of
    :data:`~lsne.experiments.common.RESUME_CHUNK` variants (pairs) under ``<run_dir>/_resume``,
    so a run that stops (session limit, disconnect, an exception in a later substage) picks
    up there when it is started again with the same inputs; the checkpoints are removed
    after the manifest is written."""
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
    lgrid = attribution_layer_grid(mb.n_layers)
    sites = [HookSite(int(l), "block_input") for l in lgrid]
    gxa_variants = C.core_subset(allv) if limits.limited else allv
    pairs = attribution_pairs(allv)
    sig = C.resume_signature(stage="C", version=RESUME_VERSION, tags=[v.tag for v in allv], directions=dir_names,
                             doses=[float(e) for e in cfg.doses], site=site.key, model=mb.spec.tag(),
                             dtype=mb.spec.dtype, bank=C.bank_signature(variants, site, cfg, limits, original),
                             limited=bool(limits.limited), seed=int(cfg.seed), n_random_patch=int(limits.n_random_patch),
                             layer_grid=[int(l) for l in lgrid], gxa_tags=[v.tag for v in gxa_variants],
                             pairs=[(n, a.tag, b.tag) for n, a, b in pairs], chunk=C.RESUME_CHUNK)
    res = C.Resume(run_dir, sig, enabled=bool(kw.get("resume", True)))
    grid_batch = int(getattr(limits, "grid_batch_size", GRID_BATCH_SIZE))

    grid = res.chunked("grid", allv, lambda vs: pd.DataFrame(
        grid_rows(mb, vs, bank, site, cfg.doses, dir_names, enabled=limits.progress, batch_size=grid_batch),
        columns=GRID_COLS))
    summary = summarise_grid(grid)
    contrasts = contrast_rows(summary)
    boot = bootstrap_contrasts(contrasts, n_boot=limits.n_boot, seed=cfg.seed)

    att = res.chunked("attribution", pairs, lambda ps: attribution_table(
        mb, allv, sites, n_random=limits.n_random_patch, seed=cfg.seed, enabled=limits.progress, pairs=ps))
    att_con = attribution_contrasts(att)
    # contrast 3 (R_delim - R_random per layer) next to T and I in contrasts_boot
    boot = pd.concat([boot, bootstrap_attribution(att_con, n_boot=limits.n_boot, seed=cfg.seed)], ignore_index=True)
    gxa, gxa_cls = res.chunked("grad_x_act", gxa_variants,
                               lambda vs: grad_x_act_tables(mb, vs, sites, enabled=limits.progress))

    tables = {
        "grid": grid, "grid_summary": summary, "contrasts": contrasts, "contrasts_boot": boot,
        "attribution": att, "attribution_contrasts": att_con, "attribution_summary": summarise_attribution(att),
        "grad_x_act": gxa, "grad_x_act_by_class": gxa_cls,
    }
    out = C.finish_stage(run_dir, "C", cfg, mb, tables, limits=limits, t0=t0, site=site, directions=dir_names,
                         bank_source=bank.meta.get("source", ""), unigram=bank.meta.get("unigram"),
                         doses=list(cfg.doses), layer_grid=[int(l) for l in lgrid],
                         patch_classes=list(C.PATCH_CLASSES), prereg_patch_class=C.PREREG_PATCH_CLASS,
                         n_variants=len(variants), n_pairs=len(pairs),
                         resumed_checkpoints=len(res.loaded), grid_batch_size=grid_batch)
    res.clear()
    return out
