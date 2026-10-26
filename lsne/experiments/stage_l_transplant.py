"""Stage L - cross-model activation transplant (DESIGN section 4.9; PREREGISTRATION
contrast 18): source adapter, alignment with the permuted-pairs null, residual
substitution (A), KV-cache planting (B) and both (C) against the controls.

Source (``Limits.l_source`` / ``--transplant-source``): ``rotated_self`` (default; the
target's own residuals seen through a seeded affine isometry ``3 H Q + 0.5``,
:class:`lsne.transplant.RotatedSourceModel` - the non-tautological stand-in: the alignment
has to undo ``Q``), ``self`` (plumbing smoke test only) or ``<model>@<revision>`` (an HF
causal LM such as ``EleutherAI/pythia-70m@step143000``; downloaded).  A BlockGPT-like
trace model is plugged in through the same :class:`SourceModel` interface once it exists
(docs/BLOCKGPT_FEASIBILITY.md).  Paired items = the English / German unit sentences of
the selected stimulus families (real translation pairs) plus their tails as neutral
sentences (``Limits.l_n_texts``).

Tables
------
transfer_test     contrast 18 per target grid layer (:func:`functional_transfer_test`,
                  source layer = the same index): ``cos_transferred_vs_native`` vs the
                  permuted-pairs null (``cos_null_vs_native``, ``z_cos_vs_null``),
                  ``r2_holdout`` vs ``r2_permuted`` / ``r2_holdout_null``, the German-fraction
                  shifts of steered continuations for the transferred and the native axis,
                  ``sign_agreement`` and ``readout_alive``; flags ``r2_gap_ok`` (>= 0.2),
                  ``cos_ok`` (>= 0.3), ``validated`` (all three incl. sign agreement)
transfer_summary  one row: the best layer (max cosine), its flags, ``validated``
layer_matching    :func:`layer_matching` R^2 table over source x target grid layers
alignments        one row per target layer of :func:`layer_alignments` (the maps that
                  plant the KV prefix at every layer): layer_src, r2_train/holdout/permuted
transplant        :func:`transplant_experiment` per prompt (K=0 contexts with their
                  targets) x mode (residual / kv / both) x control (none / random /
                  identity / permuted_alignment): entropy, logp_target, fD of a short
                  continuation, written_norm, kv_layers
transplant_summary per mode: fD and logp_target of the transplant arm minus each control
                  (mean over prompts), the sign of the fD shift vs the identity control and
                  whether it agrees with the native axis's shift (contrast 18, last clause)
"""
from __future__ import annotations

import time
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..config import ModelSpec, RunConfig, HookSite
from ..directions import cosine, diff_of_means
from ..model import ModelBundle
from . import common as C

try:
    from .. import transplant as TP
    HAVE_TRANSPLANT = True
except Exception as _e:  # noqa: BLE001 - the stage degrades to a manifest-only stub
    TP = None
    HAVE_TRANSPLANT = False
    _IMPORT_ERROR = repr(_e)

PREREG_R2_GAP, PREREG_MIN_COS = 0.2, 0.3


# --------------------------------------------------------------------------------------
# source
# --------------------------------------------------------------------------------------
def make_source(mb: ModelBundle, spec: str, seed: int):
    """``(source, kind)`` for ``Limits.l_source``: rotated_self | self | model@revision."""
    spec = str(spec or "rotated_self")
    base = TP.HFSourceModel.from_bundle(mb)
    if spec == "self":
        return base, "self"
    if spec == "rotated_self":
        return TP.RotatedSourceModel(base, seed=int(seed), scale=3.0, shift=0.5), "rotated_self"
    name, _, rev = spec.partition("@")
    src = TP.HFSourceModel(ModelSpec(name=name, revision=rev or "main", dtype="float32", device=str(mb.device)))
    return src, f"hf:{name}@{rev or 'main'}"


# --------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------
TT_COLS = ["layer_tgt", "layer_src", "site_tgt", "cos_transferred_vs_native", "cos_null_vs_native", "cos_null_std", "cos_null_max",
           "z_cos_vs_null", "n_null", "r2_train", "r2_holdout", "r2_permuted", "r2_holdout_null", "r2_holdout_null_max", "r2_gap",
           "n_pairs", "fD_baseline", "fD_transferred", "fD_native", "delta_fD_transferred", "delta_fD_native", "sign_agreement",
           "readout_alive", "n_ED_baseline", "n_ED_transferred", "n_ED_native", "eps", "r2_gap_ok", "cos_ok", "validated"]


def transfer_tables(mb: ModelBundle, src, texts_E, texts_D, texts_N, grid: Sequence[int], prompts: Sequence[str], cfg: RunConfig,
                    limits: C.Limits, enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, results = [], {}
    for l in C.progress(list(grid), "L functional transfer", enabled=enabled):
        site = HookSite(int(l), "block_input")
        l_src = int(min(l, src.n_layers))
        r = TP.functional_transfer_test(mb, src, texts_E, texts_D, l_src, site, texts_neutral=texts_N, n_gen=cfg.gen_n_new,
                                        n_samples=cfg.gen_n_samples, seed=cfg.seed, eps=limits.l_eps, prompts=list(prompts),
                                        temperature=cfg.gen_temperature, n_null=int(limits.l_n_null))
        results[int(l)] = r
        gap = float(r["r2_holdout"] - r["r2_permuted"])
        row = dict(layer_tgt=int(l), layer_src=l_src, site_tgt=site.key,
                   **{k: r[k] for k in ("cos_transferred_vs_native", "cos_null_vs_native", "cos_null_std", "cos_null_max", "z_cos_vs_null",
                                       "n_null", "r2_train", "r2_holdout", "r2_permuted", "r2_holdout_null", "r2_holdout_null_max",
                                       "n_pairs", "fD_baseline", "fD_transferred", "fD_native", "delta_fD_transferred",
                                       "delta_fD_native", "sign_agreement", "readout_alive", "eps")},
                   r2_gap=gap, n_ED_baseline=int(r["n_ED"]["baseline"]), n_ED_transferred=int(r["n_ED"]["transferred"]),
                   n_ED_native=int(r["n_ED"]["native"]))
        row["r2_gap_ok"] = bool(np.isfinite(gap) and gap >= PREREG_R2_GAP)
        row["cos_ok"] = bool(np.isfinite(r["cos_transferred_vs_native"]) and r["cos_transferred_vs_native"] >= PREREG_MIN_COS)
        row["validated"] = bool(row["r2_gap_ok"] and row["cos_ok"] and r["sign_agreement"])
        rows.append(row)
    tt = pd.DataFrame(rows, columns=TT_COLS)
    if tt.empty:
        return tt, pd.DataFrame(columns=["best_layer"] + TT_COLS[3:] + ["n_validated_layers", "n_layers"]), results
    # the best layer = the largest cosine among the validated layers (else among all)
    pool = tt[tt["validated"]] if tt["validated"].any() else tt
    best = pool.sort_values("cos_transferred_vs_native", ascending=False, kind="mergesort").iloc[0]
    summ = dict(best_layer=int(best["layer_tgt"]), **{k: best[k] for k in TT_COLS[3:]})
    summ["n_validated_layers"] = int(tt["validated"].sum())
    summ["n_layers"] = int(len(tt))
    return tt, pd.DataFrame([summ]), results


def alignment_tables(src, mb: ModelBundle, pairs, grid: Sequence[int]) -> tuple[pd.DataFrame, dict]:
    layers_src = sorted({int(min(l, src.n_layers)) for l in grid} | {src.n_layers})
    lm = TP.layer_matching(src, mb, pairs, layers_src=layers_src, layers_tgt=list(grid), readout="mean", method="procrustes")
    aligns = TP.layer_alignments(src, mb, pairs, layers_src=layers_src, readout="mean", method="procrustes")
    rows = [dict(layer_tgt=int(l), layer_src=int(a.layer_src), method=a.method, rank=a.rank, r2_train=a.r2_train, r2_holdout=a.r2_holdout,
                 r2_permuted=a.r2_permuted, gap=a.gap, n_pairs=a.n_pairs) for l, a in sorted(aligns.items())]
    return lm, dict(aligns=aligns, table=pd.DataFrame(rows, columns=["layer_tgt", "layer_src", "method", "rank", "r2_train", "r2_holdout",
                                                                      "r2_permuted", "gap", "n_pairs"]))


def transplant_tables(mb: ModelBundle, src, aligns: dict, v_src: torch.Tensor, native_axis: torch.Tensor, prompts, site: HookSite,
                      cfg: RunConfig, limits: C.Limits, delta_fD_native: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = TP.transplant_experiment(mb, src, aligns, v_src, list(prompts), site, eps=limits.l_eps, n_prefix=int(limits.l_n_prefix),
                                  gen_kw=dict(n_new=cfg.gen_n_new, n_samples=cfg.gen_n_samples, temperature=cfg.gen_temperature),
                                  native_axis=native_axis, seed=cfg.seed)
    df["source"] = getattr(src, "name", "")
    rows = []
    for mode, g in df.groupby("mode", sort=True):
        arm = g[g["control"] == "none"]
        row = dict(mode=mode, n_prompts=int(arm["prompt_idx"].nunique()), fD_transplant=float(arm["fD"].mean()),
                   logp_transplant=float(arm["logp_target"].mean()), entropy_transplant=float(arm["entropy"].mean()),
                   cos_native=float(arm["cos_native"].mean()), kv_layers=str(arm["kv_layers"].iloc[0]) if len(arm) else "")
        for con in ("random", "identity", "permuted_alignment"):
            c = g[g["control"] == con]
            row[f"fD_minus_{con}"] = float(arm["fD"].mean() - c["fD"].mean()) if len(c) else np.nan
            row[f"logp_minus_{con}"] = float(arm["logp_target"].mean() - c["logp_target"].mean()) if len(c) else np.nan
        d_id = row.get("fD_minus_identity", np.nan)
        row["sign_fD_vs_identity"] = int(np.sign(d_id)) if np.isfinite(d_id) and abs(d_id) > 1e-6 else 0
        row["sign_native"] = int(np.sign(delta_fD_native)) if np.isfinite(delta_fD_native) and abs(delta_fD_native) > 1e-6 else 0
        row["sign_agrees_with_native"] = bool(row["sign_fD_vs_identity"] != 0 and row["sign_fD_vs_identity"] == row["sign_native"])
        rows.append(row)
    return df, pd.DataFrame(rows)


def contrast18_verdict(tt_summary: pd.DataFrame, tp_summary: pd.DataFrame, source_kind: str) -> pd.DataFrame:
    """Pre-registration item 18 as one row: the functional transfer criteria at the best layer
    AND sign agreement with the native axis for residual substitution and KV planting.
    Sources that are the target itself (``self``) or an isometry of it (``rotated_self``) meet
    the R2 and cosine criteria by construction; they are plumbing checks and give no verdict."""
    if tt_summary is None or tt_summary.empty:
        return pd.DataFrame()
    t = tt_summary.iloc[0]
    signs = {}
    for mode in ("residual", "kv"):
        m = tp_summary[tp_summary["mode"] == mode] if tp_summary is not None and len(tp_summary) else pd.DataFrame()
        signs[mode] = bool(m["sign_agrees_with_native"].iloc[0]) if len(m) else False
    self_consistency = source_kind in ("self", "rotated_self")
    transfer_ok = bool(t["r2_gap_ok"] and t["cos_ok"] and t["sign_agreement"])
    validated = bool(transfer_ok and signs["residual"] and signs["kv"])
    return pd.DataFrame([dict(source_kind=source_kind, self_consistency=self_consistency, best_layer=int(t["best_layer"]),
                              r2_gap_ok=bool(t["r2_gap_ok"]), cos_ok=bool(t["cos_ok"]), sign_agreement_steering=bool(t["sign_agreement"]),
                              readout_alive=bool(t["readout_alive"]), sign_residual=signs["residual"], sign_kv=signs["kv"],
                              transfer_ok=transfer_ok, validated=(np.nan if self_consistency else validated),
                              counts_for_prereg=not self_consistency)])


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Stand-in validation of the transplant machinery; see the module docstring."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    if not HAVE_TRANSPLANT:
        warnings.warn(f"lsne.transplant is not importable ({_IMPORT_ERROR}); stage L writes a manifest only")
        return C.finish_stage(run_dir, "L", cfg, mb, {}, limits=limits, t0=t0, stub=True, reason=_IMPORT_ERROR)
    sub = C.Substages("L", verbose=limits.progress)
    variants, _checks = C.get_variants(cfg, mb, ctx)
    fams = C.select_families(cfg)
    empty = pd.DataFrame()
    source_spec = kw.get("transplant_source") or limits.l_source
    src, source_kind = make_source(mb, source_spec, cfg.seed)
    texts_E, texts_D, texts_N = C.family_text_pairs(fams, limits.l_n_texts)
    k0 = C.k_subset(variants, 0)[: int(limits.l_n_prompts)]
    prompts = [v.text for v in k0] or ["Yesterday we went to the market and"]
    grid = list(C.prereg_layer_grid(mb.n_layers))

    tt, tt_summary, results = sub.run("functional_transfer", lambda: transfer_tables(mb, src, texts_E, texts_D, texts_N, grid, prompts, cfg,
                                                                                    limits, enabled=limits.progress), (empty, empty, {}))
    # The transplant is run at a layer of the pre-registered grid: the best validated layer of the
    # functional transfer test (else the grid layer nearest the middle). v_src, the native axis and
    # the native sign reference are then taken at exactly this injection site.
    if len(tt_summary) and np.isfinite(tt_summary["best_layer"].iloc[0]):
        site_layer = int(tt_summary["best_layer"].iloc[0])
    else:
        site_layer = int(min(grid, key=lambda l: abs(l - mb.n_layers // 2))) if grid else mb.n_layers // 2
    site = HookSite(site_layer, "block_input")
    self_consistency = source_kind in ("self", "rotated_self")
    pairs = [(t, t) for t in texts_E + texts_D + texts_N]
    lm, al = sub.run("alignments", lambda: alignment_tables(src, mb, pairs, list(range(mb.n_layers + 1))), (empty, dict(aligns={}, table=empty)))

    def _transplant():
        if not al["aligns"]:
            raise RuntimeError("no alignments")
        key = site.layer if site.layer in results else None
        if key is not None:
            v_src, native, d_nat = results[key]["v_src"], results[key]["v_native"], float(results[key]["delta_fD_native"])
        else:
            S = torch.stack([TP._pool_rows(src.residuals(src.encode(t), [site.layer])[site.layer], "mean") for t in texts_E + texts_D])
            v_src = diff_of_means(S[len(texts_E):], S[: len(texts_E)])
            Tt = torch.stack([TP.target_residuals(mb, mb.encode(t), [site.layer])[site.layer].mean(0) for t in texts_E + texts_D])
            native, d_nat = diff_of_means(Tt[len(texts_E):], Tt[: len(texts_E)]), float("nan")
        return transplant_tables(mb, src, al["aligns"], v_src, native, [(v.text, int(v.target_id)) for v in k0] or prompts, site, cfg, limits, d_nat)
    tp, tp_summary = sub.run("transplant", _transplant, (empty, empty))

    verdict = contrast18_verdict(tt_summary, tp_summary, source_kind)
    tables = {"transfer_test": tt, "transfer_summary": tt_summary, "layer_matching": lm, "alignments": al["table"], "transplant": tp,
              "transplant_summary": tp_summary, "verdict": verdict}
    sub.write_tracebacks(run_dir)
    return C.finish_stage(run_dir, "L", cfg, mb, tables, limits=limits, t0=t0, site=site, source=dict(kind=source_kind, name=getattr(src, "name", ""),
                                                                                                     d_model=int(src.d_model), n_layers=int(src.n_layers)),
                          n_texts=dict(E=len(texts_E), D=len(texts_D), neutral=len(texts_N)), prompts=prompts, layer_grid=grid,
                          validated=(None if verdict.empty or self_consistency else bool(verdict["validated"].iloc[0])),
                          self_consistency=self_consistency,
                          transplant=dict(eps=limits.l_eps, n_prefix=limits.l_n_prefix, n_gen=cfg.gen_n_new, n_samples=cfg.gen_n_samples),
                          substages=sub.report(),
                          prereg=dict(contrast_18="verdict.validated = transfer_summary (r2_gap_ok, cos_ok, sign_agreement at the best layer) "
                                                  "AND transplant_summary.sign_agrees_with_native for modes residual and kv; "
                                                  "no verdict for self-consistency sources (self, rotated_self)"))
