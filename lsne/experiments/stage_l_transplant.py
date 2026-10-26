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

Trace source (``trace:<model_dir>``, :class:`lsne.chain.source.TraceLMSource`)
------------------------------------------------------------------------------
DESIGN 4.9 proper: a "sharing" direction of the trace LM, transplanted into Pythia, and its
effect on the language-switching task and on representational sharing.  Options: keywords
``trace_<name>`` (Python), ``Limits.l_trace_<name>`` when that field exists, or -- the
command-line path -- ``key=value`` items on the source spec,
``--transplant-source "trace:<model_dir>,shards=<itr>,k=5,standin=<stand-in run dir>"``
(:func:`parse_trace_spec`; ``lsne.cli run`` has no ``--trace-*`` flags).  ``shards`` (the ITR
shards; default the ``data`` entry of the model's ``manifest.json``), ``direction`` (a
``sharing_directions.pt`` of ``python -m lsne.chain.cli probe``, or found as
``<model_dir>/sharing_directions.pt``), ``layer`` (frozen source snapshot of the direction and
of every alignment, default the last, ``n_layers``), ``k`` (fan-out threshold, 5), ``n_probe``
(max matched pairs, 500 / smoke 50), ``n_pairs`` (alignment pairs, ``4 * l_n_texts``),
``length_ratio`` (1.25), ``limit`` (transactions read from the shards), ``n_substitute``
(positions of the residual substitution, default ``l_n_prefix``), ``site_layer``,
``n_prompts`` (K=0 variants of the transplant, default one per selected family, see
:func:`trace_prompt_variants`), ``split`` (``all`` / ``heldout`` / ``train``: transactions of
the LM's hash split only, default ``all``; the LM-split membership of the probe and
alignment items is recorded in the manifest), ``standin`` (the stage-L run directory of the
stand-in validation, PREREGISTRATION 18).  Helpers: :mod:`lsne.chain.pairs`.

1. *Probe* (``trace_probe``, ``trace_probe_match``): the length-matched fan-out probe
   (``fan_out_eth >= k`` vs ``== 1`` on net flows, required on the visible window too, 1:1
   within era on the visible ITR length; pairs with a repeated visible sequence dropped)
   under the trained model and its untrained twin; per source snapshot the probe criterion
   (:func:`lsne.chain.source.criterion_table`: split-half AUC trained minus untrained, paired
   bootstrap CI, permutation / random-direction nulls, ``counts``), ``frozen`` marks the
   frozen layer, ``feature_frozen`` whether the pre-specified probe was used (a relaxed
   fallback never supports a claim).  The direction ``v_src`` = difference of means at the
   frozen layer, or the saved file (``direction_source``); a file supports a claim only when
   it was written by the same probe (kind fanout, match length, mask none, readout mean,
   same k, length ratio and window; ``direction_spec_ok``).
2. *Alignment* (``layer_matching``, ``alignments``, ``trace_alignment_controls``): paired
   ``(shard row, its English text)`` items (label-balanced sample of transactions that fit
   the window); Procrustes maps fitted on pooled states residualised on the length
   covariates of :mod:`lsne.chain.source` (:func:`lsne.chain.pairs.residualised_alignment`:
   ``r2_*`` beyond length, intercept from the raw means, ``r2_holdout_raw`` for comparison);
   ``alignments`` = one map per target layer 0..L from the frozen source layer; the controls
   table (ridge) adds raw / residualised R² of the untrained twin, the length-only R² and the
   trained-minus-untrained gains with bootstrap CIs (:func:`lsne.chain.source.alignment_controls`).
3. *Transplant* (``transplant``, ``transplant_summary``): prompt pairs (carrier transaction,
   stimulus variant) for the K=0 variants (one per family, :func:`trace_prompt_variants`) and
   their K=8/C1 partners; the carrier is a matched probe negative (fan-out 1), the same for
   both variants of a formulation; residual substitution (last ``trace_n_substitute``
   positions), KV planting at every layer, both; controls random (one random direction per
   carrier, shared by the K=0 and K=8/C1 partner) / identity / permuted_alignment; site =
   the grid layer nearest the middle; ``cos_native`` against Pythia's own
   length-residualised direction between the English texts of the probe's positives and
   negatives.
4. *Readouts* (``trace_bt``, ``trace_effects``, ``trace_sharing``): ``logp_target`` and
   ``B = logp_target - logp_plain`` per variant, ``T = logp(K8C1) - logp(K0)`` per formulation
   (``trace_bt``); per mode x control the transplant-minus-control differences of B (per K and
   pooled), T, fD and entropy with family cluster-bootstrap CIs (``trace_effects``); the
   sharing index (Jaccard of active SAE latents K0 vs K8C1) and network activity
   (:mod:`lsne.sharing`) at the dossier site per arm and for the plain model
   (``trace_sharing``; ``sae_source`` marks a synthetic SAE in smoke runs).
5. ``trace_verdict`` (:func:`trace_verdict`): one row per mode.  ``sharing_effect_claimed`` =
   ``probe_ok`` (``feature_frozen`` AND the probe criterion ``counts`` at the frozen layer AND
   ``direction_spec_ok``) AND ``align_ok`` (residualised hold-out R² > 0 and gap >= 0.2 on
   every map the mode uses: the site layer for residual, every planted layer for kv, both
   for both) AND ``transplant_effect`` (B or T differs from both the random and the
   permuted-alignment control with CIs excluding 0 in the same direction, >= 3 families); it
   is **never** True without the probe criterion.  ``counts_for_prereg`` additionally needs
   a passed stand-in validation (``standin``: ``verdict.validated`` True for a language
   source).  ``claim`` states the reasons.

The functional transfer test (contrast 18, ``transfer_test`` / ``transfer_summary``) needs
paired language texts in both models and is skipped for a trace source (empty tables,
manifest ``functional_transfer.skipped``); ``verdict`` then holds one explanatory row.
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
from ..model import ModelBundle, run_with_cache
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
#: options of the trace source that can ride on ``--transplant-source trace:<dir>,<key>=<value>,...``
TRACE_SPEC_KEYS = ("shards", "direction", "layer", "k", "n_probe", "n_pairs", "site_layer", "length_ratio", "limit",
                   "n_substitute", "n_prompts", "standin", "split")


def parse_trace_spec(spec: str) -> tuple[str, dict]:
    """``"trace:<model_dir>[,key=value,...]"`` -> ``(model_dir, {key: value})``.  This is the
    command-line path to the trace options (``python -m lsne.cli run --stage L
    --transplant-source "trace:runs/trace_lm,shards=data/traces/itr,k=5,standin=runs/L/<tag>"``);
    keys: :data:`TRACE_SPEC_KEYS`, values are strings (paths must not contain commas)."""
    body = str(spec)[len("trace:"):] if str(spec).startswith("trace:") else str(spec)
    parts = body.split(",")
    model_dir, opts = parts[0].strip(), {}
    for p in parts[1:]:
        if not p.strip():
            continue
        key, sep, val = p.partition("=")
        key = key.strip().replace("-", "_")
        if not sep or key not in TRACE_SPEC_KEYS:
            raise ValueError(f"trace source option {p!r}: expected key=value with key in {TRACE_SPEC_KEYS}")
        opts[key] = val.strip()
    return model_dir, opts


def make_source(mb: ModelBundle, spec: str, seed: int):
    """``(source, kind)`` for ``Limits.l_source``: rotated_self | self | model@revision |
    ``trace:<model_dir>[,key=value,...]`` (a trace LM directory written by
    :func:`lsne.chain.train.train`, loaded as :class:`lsne.chain.source.TraceLMSource` on the
    target's device; kind ``"trace"``; options: :func:`parse_trace_spec`)."""
    spec = str(spec or "rotated_self")
    if spec.startswith("trace:"):
        from ..chain.source import TraceLMSource
        model_dir, _ = parse_trace_spec(spec)
        if not model_dir:
            raise ValueError("trace source needs a model directory: trace:<model_dir>")
        return TraceLMSource(model_dir, device=str(mb.device)), "trace"
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
    return df, summarise_transplant(df, delta_fD_native)


def summarise_transplant(df: pd.DataFrame, delta_fD_native: float) -> pd.DataFrame:
    """``transplant_summary``: per mode the transplant arm minus each control (fD, logp) and the
    sign of the fD shift vs the identity control against the native axis's shift."""
    if df is None or df.empty or "mode" not in df.columns:
        return pd.DataFrame()
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
    return pd.DataFrame(rows)


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
    if source_kind == "trace":
        kw = dict(kw)
        for key, val in parse_trace_spec(source_spec)[1].items():    # options on the spec string (the CLI path)
            if kw.get(f"trace_{key}") is None:
                kw[f"trace_{key}"] = val
        return run_trace(cfg, mb, run_dir, src, limits, kw, t0, sub, variants)
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


# ======================================================================================
# trace source (DESIGN 4.9 with lsne.chain): probe -> residualised alignments -> transplant
# ======================================================================================
TRACE_MIN_CLUSTERS = 3                 # families needed before a transplant effect can count
TRACE_MIN_PROBE_PAIRS = 4              # matched pairs needed by the probe criterion
TRACE_MODES = ("residual", "kv", "both")
TRACE_CONTROLS = ("random", "identity", "permuted_alignment")
FT_SKIP_REASON = ("trace source: the functional transfer test (contrast 18) needs paired language texts (E/D) in "
                  "both models; it validates the machinery with a language source and is skipped here")
LM_TRACE_COLS = ["layer_src", "layer_tgt", "r2_train", "r2_holdout", "r2_permuted", "gap", "r2_holdout_raw",
                 "r2_permuted_raw", "method", "rank", "n_pairs", "residualised_on"]
AL_COLS = ["layer_tgt", "layer_src", "method", "rank", "r2_train", "r2_holdout", "r2_permuted", "gap", "n_pairs"]
BT_COLS = ["prompt_idx", "tag", "family", "formulation", "K", "C", "carrier", "mode", "control", "logp_target",
           "logp_plain", "B", "fD", "entropy"]
T_COLS = ["family", "formulation", "mode", "control", "T", "T_plain", "dT"]
EFFECT_COLS = ["mode", "control", "quantity", "mean", "ci_low", "ci_high", "se", "n_clusters", "n_rows", "enough_clusters",
               "ci_excludes_0"]
SHARING_COLS = ["mode", "control", "measure", "condition", "value", "n_K0", "n_K8C1", "site", "sae_source"]


def _trace_opt(kw: dict, limits, name: str, default=None):
    """Option ``name`` of the trace path: keyword ``trace_<name>`` (the CLI), else
    ``Limits.l_trace_<name>`` when the field exists, else ``default``."""
    v = kw.get(f"trace_{name}")
    if v is None:
        v = getattr(limits, f"l_trace_{name}", None)
    return default if v is None else v


def _nearest_middle(grid: Sequence[int], n_layers: int) -> int:
    return int(min(grid, key=lambda l: abs(l - n_layers // 2))) if len(grid) else int(n_layers // 2)


def trace_probe(src, twin, index: pd.DataFrame, shards: str, frozen: int, limits: C.Limits, kw: dict, seed: int) -> dict:
    """Length-matched fan-out probe, the probe criterion per source snapshot and ``v_src``."""
    from ..chain import pairs as TPP
    from ..chain.labels import surface_baseline_auc
    from ..chain.source import direction_from_states

    n_probe = int(_trace_opt(kw, limits, "n_probe", 50 if limits.is_smoke else 500))
    k = int(_trace_opt(kw, limits, "k", TPP.DEFAULT_K))
    ratio = float(_trace_opt(kw, limits, "length_ratio", 1.25))
    pos_h, neg_h, mt, info = TPP.fanout_probe(index, n=n_probe, seed=seed, k=k, length_ratio=ratio, max_len=src.max_len)
    rows = TPP.fetch_rows(shards, pos_h + neg_h) if pos_h else {}
    pairs = [(rows[a], rows[b]) for a, b in zip(pos_h, neg_h) if a in rows and b in rows]
    # identical visible sequences are not independent pairs (split halves, pair bootstrap)
    keep = TPP.dedup_pairs([src.tokens(a) for a, _ in pairs], [src.tokens(b) for _, b in pairs])
    info["n_pairs_dropped_duplicates"] = len(pairs) - len(keep)
    pairs = [pairs[i] for i in keep]
    pos, neg = [a for a, _ in pairs], [b for _, b in pairs]
    n = len(pos)
    info["n_pairs_used"] = n
    vf = TPP.lm_val_frac(src.model_dir)
    lm_val = [TPP.is_lm_val(r["tx_hash"], vf) for r in pos + neg]
    info["lm_split"] = dict(val_frac=vf, n_items=2 * n, n_items_lm_val=int(sum(lm_val)))
    res = dict(info=info, match=mt, pos=pos, neg=neg, table=pd.DataFrame(), direction_meta={}, v_src=None,
               direction_source=None, file_info={}, direction_spec_ok=True)
    layers = list(range(src.n_layers + 1))
    if n >= TPP.MIN_PAIRS_DIRECTION:
        if twin is not None:
            S, S0 = TPP.probe_states(src, twin, pos + neg, layers)
            surface = surface_baseline_auc([r["tokens"] for r in pos], [r["tokens"] for r in neg], seed=seed,
                                           max_len=src.max_len)
            tab = TPP.probe_table(S, S0, n, layers, seed=seed, n_boot=int(limits.n_boot),
                                  n_perm=min(200, int(limits.n_boot)), surface=surface)
        else:   # no untrained twin: no probe criterion (never counts), the direction is still reported
            from ..chain.source import pooled_states
            S = pooled_states(src, pos + neg, layers, "mean")
            tab = pd.DataFrame([dict(layer=l, n_pairs=n, counts=False, note="no untrained twin") for l in layers])
        d = direction_from_states(S[frozen][:n], S[frozen][n:], frozen, src.n_layers, name="sharing_fanout", seed=seed,
                                  meta=dict(kind="fanout", k=info["k_used"], match=info["match"], source=src.name))
        res.update(table=tab, v_src=d.vec, direction_source=f"probe:fanout(k={info['k_used']}, match={info['match']})",
                   direction_meta={k_: v for k_, v in d.meta.items() if k_ != "vec_denoised"})
    path = TPP.find_direction_file(src.model_dir, _trace_opt(kw, limits, "direction"))
    if path:
        vec, finfo = TPP.load_direction(path, frozen, src.d_model, src.model_dir)
        if vec is not None:
            if res["v_src"] is not None:
                finfo["cos_with_probe_direction"] = cosine(vec, res["v_src"])
            # the file replaces v_src, but it supports a claim only if it is the frozen probe's direction
            ok, why = TPP.direction_spec_ok(finfo, k, ratio, src.max_len)
            finfo.update(spec_ok=ok, spec_mismatch=why)
            res.update(v_src=vec, direction_source=f"file:{path}", direction_spec_ok=ok)
        res["file_info"] = finfo
    tab = res["table"]
    if len(tab):
        tab.insert(1, "frozen", tab["layer"] == int(frozen))
        for key in ("k_requested", "k_used", "match", "matching", "feature_frozen"):
            tab[key] = info.get(key)
        for key in ("length_smd", "n_tokens_smd", "log_gas_smd"):
            tab[key] = info.get(key, np.nan)
        tab["direction_source"] = res["direction_source"]
        tab["direction_spec_ok"] = res["direction_spec_ok"]
    return res


def trace_alignment(src, twin, mb: ModelBundle, index: pd.DataFrame, shards: str, frozen: int, site_layer: int, probe: dict,
                    limits: C.Limits, kw: dict, seed: int) -> dict:
    """Residualised Procrustes alignments on paired (shard row, English text) items, the
    length / untrained controls (ridge) and Pythia's native direction of the probe texts."""
    from ..chain import pairs as TPP
    from ..chain.source import pooled_states

    n_pairs = int(_trace_opt(kw, limits, "n_pairs", 4 * int(limits.l_n_texts)))
    # transactions longer than the window are excluded: their text describes nodes the source never sees
    hashes = TPP.sample_pair_hashes(index, n_pairs, seed=seed, balance=True, max_len=src.max_len)
    rows = TPP.fetch_rows(shards, hashes)
    items = [rows[h] for h in hashes if h in rows and str(rows[h].get("text") or "").strip()]
    if len(items) < 5:
        raise ValueError(f"only {len(items)} alignment pairs with a text (need >= 5)")
    texts = [str(r["text"]) for r in items]
    pairs = [TP.PromptPair(source=r, text=t, name=r["tx_hash"]) for r, t in zip(items, texts)]
    Ls, Lt = list(range(src.n_layers + 1)), list(range(mb.n_layers + 1))
    S, T = TP.paired_states(src, mb, pairs, Ls, Lt, readout="mean")
    S0 = pooled_states(twin, items, Ls, "mean") if twin is not None else None
    cov = TPP.length_covariates(items, texts, tokenizer=mb.tokenizer, max_len=src.max_len)
    Z, names = TPP.covariate_matrix(cov, text=True)
    fits, lm_rows = {}, []
    for ls in Ls:
        for lt in Lt:
            a = TPP.residualised_alignment(S[ls], T[lt], Z, ls, lt, method="procrustes", holdout=0.2, seed=seed,
                                           covariate_names=names)
            fits[(ls, lt)] = a
            lm_rows.append(dict(layer_src=ls, layer_tgt=lt, r2_train=a.r2_train, r2_holdout=a.r2_holdout,
                                r2_permuted=a.r2_permuted, gap=a.gap, r2_holdout_raw=a.meta["r2_holdout_raw"],
                                r2_permuted_raw=a.meta["r2_permuted_raw"], method=a.method, rank=a.rank, n_pairs=a.n_pairs,
                                residualised_on=",".join(names)))
    aligns = {lt: fits[(int(frozen), lt)] for lt in Lt}
    al = pd.DataFrame([dict(layer_tgt=lt, layer_src=a.layer_src, method=a.method, rank=a.rank, r2_train=a.r2_train,
                            r2_holdout=a.r2_holdout, r2_permuted=a.r2_permuted, gap=a.gap, n_pairs=a.n_pairs)
                       for lt, a in sorted(aligns.items())], columns=AL_COLS)
    controls = pd.DataFrame()
    if TPP.alignment_controls is not None and S0 is not None:
        controls = TPP.alignment_controls(S, S0, T, Z, holdout=0.2, seed=seed, n_boot=int(limits.n_boot))
        controls.insert(0, "covariates", ",".join(names))
    # Pythia's own direction of the feature: English texts of the probe's positives vs negatives
    native, n_nat = None, min(len(probe.get("pos", [])), len(probe.get("neg", [])))
    if n_nat >= 2:
        ids_list = [mb.encode(str(r.get("text") or "")) for r in probe["pos"][:n_nat] + probe["neg"][:n_nat]]
        if all(int(i.shape[1]) > 0 for i in ids_list):
            Tn = torch.stack([TP.target_residuals(mb, i, [site_layer])[site_layer].mean(0) for i in ids_list])
            z = np.log(np.asarray([int(i.shape[1]) for i in ids_list], dtype=float))[:, None]
            R = torch.from_numpy(np.asarray(TPP.residualize(Tn, z))).float()
            try:
                native = diff_of_means(R[:n_nat], R[n_nat:], denoise_pcs=0)
            except ValueError:
                native = None
    vf = TPP.lm_val_frac(src.model_dir)
    lm_split = dict(val_frac=vf, n_items_lm_val=int(sum(TPP.is_lm_val(r["tx_hash"], vf) for r in items)))
    return dict(layer_matching=pd.DataFrame(lm_rows, columns=LM_TRACE_COLS), alignments=al, aligns=aligns,
                controls=controls, items=items, covariates=names, native=native, n_pairs=len(items), lm_split=lm_split)


def trace_prompt_variants(k0: Sequence, k8: Sequence, n_prompts=None, limits: Optional[C.Limits] = None) -> list:
    """The K=0 variants of the trace transplant, spread over the families: round-robin over
    the families in order (first formulation of every family, then the second, ...),
    formulations with a K=8/C1 partner first.  ``n_prompts`` (option ``trace_n_prompts``)
    sets the number; the default covers every selected family once and is at least
    ``Limits.l_n_prompts`` (a smoke run keeps ``l_n_prompts``).  The family cluster bootstrap
    of ``trace_effects`` needs >= ``TRACE_MIN_CLUSTERS`` families, so the first
    ``l_n_prompts`` variants of one family can never support an effect."""
    partners = {(v.family_id, v.formulation) for v in k8}
    fams: dict = {}
    for v in k0:
        fams.setdefault(v.family_id, []).append(v)
    for f, vs in fams.items():
        vs.sort(key=lambda v: (v.family_id, v.formulation) not in partners)      # stable: partnered first
    if n_prompts is None:
        base = int(limits.l_n_prompts) if limits is not None else 1
        n = base if (limits is not None and limits.is_smoke) else max(base, len(fams))
    else:
        n = int(n_prompts)
    out, depth = [], 0
    while len(out) < n and any(depth < len(vs) for vs in fams.values()):
        for vs in fams.values():
            if depth < len(vs) and len(out) < n:
                out.append(vs[depth])
        depth += 1
    return out


def trace_transplant(mb: ModelBundle, src, aligns: dict, v_src, native, variants, probe: dict, align: dict, site: HookSite,
                     meas_site: HookSite, cfg: RunConfig, limits: C.Limits, kw: dict) -> dict:
    """Prompt pairs (carrier transaction, stimulus variant) for K=0 and K=8/C1, the transplant
    experiment with the states at ``meas_site`` and the unmodified ("plain") baseline."""
    from ..chain import pairs as TPP

    if v_src is None:
        raise RuntimeError("no sharing direction (probe found < 2 pairs and no direction file)")
    k8 = C.k_subset(variants, 8, "C1")
    k0 = trace_prompt_variants(C.k_subset(variants, 0), k8, _trace_opt(kw, limits, "n_prompts"), limits)
    if not k0:
        raise RuntimeError("no K=0 stimulus variants")
    carriers = TPP.carrier_rows(probe.get("neg", []), align.get("items", []), len(k0), seed=cfg.seed)
    prompts, meta = [], []
    for gi, v0 in enumerate(k0):
        car = carriers[gi]
        partner = [v for v in k8 if (v.family_id, v.formulation) == (v0.family_id, v0.formulation)][:1]
        for v in [v0] + partner:
            prompts.append(TP.PromptPair(source=car, text=v.text, target=int(v.target_id), ids=[int(i) for i in v.ids],
                                         name=str(car.get("tx_hash"))))
            feat = car.get("features") or {}
            meta.append(dict(prompt_idx=len(meta), tag=v.tag, family=str(v.family_id), formulation=str(v.formulation),
                             K=int(v.K), C=str(v.C), carrier=str(car.get("tx_hash")), carrier_label=car.get("label"),
                             carrier_fan_out=feat.get("fan_out_eth"), carrier_n_tokens=car.get("n_tokens")))
    n_sub = int(_trace_opt(kw, limits, "n_substitute", int(limits.l_n_prefix)))
    df, states = TP.transplant_experiment(mb, src, aligns, v_src, prompts, site, eps=limits.l_eps, n_prefix=int(limits.l_n_prefix),
                                          n_substitute=n_sub, native_axis=native, seed=cfg.seed, state_sites=[meas_site],
                                          gen_kw=dict(n_new=cfg.gen_n_new, n_samples=cfg.gen_n_samples,
                                                      temperature=cfg.gen_temperature))
    mdf = pd.DataFrame(meta)
    df = df.join(mdf.set_index("prompt_idx"), on="prompt_idx")
    df["source"] = getattr(src, "name", "")
    plain_rows, plain_states = [], []
    for i, pp in enumerate(prompts):
        co = run_with_cache(mb, torch.as_tensor([list(pp.ids)], dtype=torch.long, device=mb.device), [meas_site])
        lp = float(torch.log_softmax(co.logits[0, -1].float(), -1)[int(pp.target)])
        plain_rows.append(dict(meta[i], logp_plain=lp))
        plain_states.append(co.resid[meas_site.key][0, -1].detach().float().cpu().numpy())
    return dict(df=df, states=states, plain=pd.DataFrame(plain_rows), plain_states=plain_states, meta=mdf,
                n_substitute=n_sub, n_prompts=len(prompts), n_families=int(mdf["family"].nunique()) if len(mdf) else 0)


def _diff_rows(a: pd.DataFrame, b: pd.DataFrame, col: str, keys: Sequence[str]) -> pd.DataFrame:
    m = a.merge(b[list(keys) + [col]], on=list(keys), suffixes=("", "_c"))
    return m.assign(diff=m[col] - m[f"{col}_c"])


def _effect_row(d: pd.DataFrame, n_boot: int, seed: int, **keys) -> dict:
    d = d[np.isfinite(d["diff"].to_numpy(float))] if len(d) else d
    if d.empty:
        r = dict(keys, mean=np.nan, ci_low=np.nan, ci_high=np.nan, se=np.nan, n_clusters=0, n_rows=0)
    else:
        r = C.boot_row(d, "diff", n_boot, seed, cluster_col="family", **keys)
    enough = bool(r["n_clusters"] >= TRACE_MIN_CLUSTERS)
    lo, hi = float(r["ci_low"]), float(r["ci_high"])
    r["enough_clusters"] = enough
    r["ci_excludes_0"] = bool(enough and np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
    return r


def trace_readouts(tp: dict, sae, sae_source: str, meas_site: HookSite, n_boot: int, seed: int) -> tuple:
    """``(trace_bt, trace_T, trace_effects, trace_sharing)`` from :func:`trace_transplant`."""
    from ..sharing import active_features, jaccard, network_activity

    df, plain = tp["df"], tp["plain"]
    bt = df.merge(plain[["prompt_idx", "logp_plain"]], on="prompt_idx", how="left")
    bt["B"] = bt["logp_target"] - bt["logp_plain"]
    bt = bt[BT_COLS].copy()
    # T per formulation and arm
    t_rows = []
    pl = plain.set_index("prompt_idx")
    for (mode, control), g in bt.groupby(["mode", "control"], sort=False):
        for (fam, form), h in g.groupby(["family", "formulation"], sort=True):
            a0, a8 = h[h["K"] == 0], h[(h["K"] == 8) & (h["C"] == "C1")]
            if a0.empty or a8.empty:
                continue
            Tv = float(a8["logp_target"].iloc[0] - a0["logp_target"].iloc[0])
            Tp = float(pl.loc[int(a8["prompt_idx"].iloc[0]), "logp_plain"] - pl.loc[int(a0["prompt_idx"].iloc[0]), "logp_plain"])
            t_rows.append(dict(family=fam, formulation=form, mode=mode, control=control, T=Tv, T_plain=Tp, dT=Tv - Tp))
    tt = pd.DataFrame(t_rows, columns=T_COLS)
    # transplant minus control, family cluster bootstrap
    eff = []
    keys_v = ["prompt_idx"]
    keys_t = ["family", "formulation"]
    for mode in [m for m in TRACE_MODES if m in set(bt["mode"])]:
        arm = bt[(bt["mode"] == mode) & (bt["control"] == "none")]
        arm_t = tt[(tt["mode"] == mode) & (tt["control"] == "none")]
        for control in TRACE_CONTROLS + ("plain",):
            if control == "plain":
                dv = arm.assign(diff=arm["B"])
                dt = arm_t.assign(diff=arm_t["dT"])
                per_q = dict(B_all=dv, B_K0=dv[dv["K"] == 0], B_K8C1=dv[dv["K"] == 8], T=dt)
            else:
                c = bt[(bt["mode"] == mode) & (bt["control"] == control)]
                ct = tt[(tt["mode"] == mode) & (tt["control"] == control)]
                if c.empty:
                    continue
                dv = _diff_rows(arm, c, "logp_target", keys_v)
                per_q = dict(B_all=dv, B_K0=dv[dv["K"] == 0], B_K8C1=dv[dv["K"] == 8],
                             T=_diff_rows(arm_t, ct, "T", keys_t), fD=_diff_rows(arm, c, "fD", keys_v),
                             entropy=_diff_rows(arm, c, "entropy", keys_v))
            for q, d in per_q.items():
                eff.append(_effect_row(d, n_boot, seed, mode=mode, control=control, quantity=q))
    effects = pd.DataFrame(eff, columns=EFFECT_COLS)
    # sharing index / network activity at the measurement site
    sh = []
    if sae is not None:
        arms = {}
        for i, r in enumerate(df.itertuples(index=False)):
            cond = "K0" if int(r.K) == 0 else ("K8C1" if int(r.K) == 8 and str(r.C) == "C1" else None)
            if cond is not None:
                arms.setdefault((r.mode, r.control), {"K0": [], "K8C1": []})[cond].append(tp["states"][i][meas_site.key])
        for i, r in enumerate(plain.itertuples(index=False)):
            cond = "K0" if int(r.K) == 0 else ("K8C1" if int(r.K) == 8 and str(r.C) == "C1" else None)
            if cond is not None:
                arms.setdefault(("plain", "plain"), {"K0": [], "K8C1": []})[cond].append(tp["plain_states"][i])
        for (mode, control), g in arms.items():
            nK0, nK8 = len(g["K0"]), len(g["K8C1"])
            base = dict(mode=mode, control=control, n_K0=nK0, n_K8C1=nK8, site=meas_site.key, sae_source=sae_source)
            X = {c: np.stack(v) for c, v in g.items() if v}
            if "K0" in X and "K8C1" in X:
                sh.append(dict(base, measure="sharing_jaccard", condition="K0|K8C1",
                               value=jaccard(active_features(sae, X["K0"]), active_features(sae, X["K8C1"]))))
            for cond, Xc in X.items():
                act = network_activity(sae, Xc)
                for q in ("frac_active", "mean_mass", "participation_ratio", "n_active_mean"):
                    sh.append(dict(base, measure=f"activity_{q}", condition=cond, value=float(act[q])))
    sharing = pd.DataFrame(sh, columns=SHARING_COLS)
    return bt, tt, effects, sharing


def _sharing_delta(sharing: pd.DataFrame, mode: str, control: str, measure: str, condition: str) -> float:
    if sharing is None or sharing.empty:
        return float("nan")
    sel = (sharing["measure"] == measure) & (sharing["condition"] == condition)
    a = sharing[sel & (sharing["mode"] == mode) & (sharing["control"] == "none")]["value"]
    b = sharing[sel & (sharing["mode"] == mode) & (sharing["control"] == control)]["value"]
    return float(a.iloc[0] - b.iloc[0]) if len(a) and len(b) else float("nan")


def _align_layer_ok(row: dict) -> bool:
    gap, r2 = float(row.get("gap", np.nan)), float(row.get("r2_holdout", np.nan))
    return bool(np.isfinite(gap) and gap >= PREREG_R2_GAP and np.isfinite(r2) and r2 > 0)


def read_standin(path) -> dict:
    """The stand-in validation of PREREGISTRATION 18 from a stage-L run with a language source
    (``path``: its run directory or ``verdict.csv``).  ``ok`` only when ``verdict.validated`` is
    True, ``counts_for_prereg`` is True and the source is not the target itself or an isometry
    of it (``self`` / ``rotated_self``: plumbing checks without a verdict)."""
    import os
    out = dict(path=None if path is None else str(path), ok=False, reason=None)
    if not path:
        out["reason"] = "no stand-in run given (option trace_standin / standin=<stage-L run dir>)"
        return out
    p = os.path.join(str(path), "verdict.csv") if os.path.isdir(str(path)) else str(path)
    try:
        v = pd.read_csv(p)
    except (OSError, ValueError, pd.errors.EmptyDataError) as e:
        out["reason"] = f"stand-in verdict unreadable: {p} ({type(e).__name__})"
        return out
    if v.empty:
        out["reason"] = f"stand-in verdict is empty: {p}"
        return out
    r = v.iloc[0].to_dict()

    def _true(x) -> bool:
        return str(x).strip().lower() in ("true", "1", "1.0")
    kind = str(r.get("source_kind", ""))
    out.update(path=p, source_kind=kind, validated=r.get("validated"), counts_for_prereg=r.get("counts_for_prereg"))
    if kind in ("self", "rotated_self", "trace") or _true(r.get("self_consistency", False)):
        out["reason"] = f"stand-in source {kind!r} is a self-consistency / non-language source (no verdict)"
    elif not _true(r.get("validated")):
        out["reason"] = f"stand-in not validated (verdict.validated = {r.get('validated')!r})"
    elif not _true(r.get("counts_for_prereg")):
        out["reason"] = "stand-in verdict does not count for the pre-registration"
    else:
        out["ok"] = True
    return out


def trace_verdict(probe: dict, align: dict, effects: pd.DataFrame, sharing: pd.DataFrame, frozen: int, site_layer: int,
                  sae_source: str, is_smoke: bool, modes: Sequence[str] = TRACE_MODES,
                  standin: Optional[dict] = None) -> pd.DataFrame:
    """One row per mode.  ``sharing_effect_claimed`` = ``probe_ok`` AND ``align_ok`` AND
    ``transplant_effect`` -- never True unless the probe criterion holds (``probe_ok``:
    the pre-specified length-matched probe (``feature_frozen``) with at least
    ``TRACE_MIN_PROBE_PAIRS`` pairs and ``counts`` at the frozen layer, i.e. the bootstrap CI
    of trained minus untrained split-half AUC above 0, and ``v_src`` from that probe -- a
    direction file counts only when it was written by the same probe specification,
    ``direction_spec_ok``).  ``align_ok``: residualised hold-out R² > 0 and gap to the
    permuted-pairs null >= 0.2 on every map the mode uses -- the site's target layer for
    ``residual``, every planted layer of the KV prefix for ``kv`` (all target layers), both
    for ``both`` (``align_ok_site``, ``n_planted_layers`` / ``n_planted_layers_ok``,
    ``planted_layers_failing``).  ``transplant_effect``: B (pooled over K=0 and K=8/C1) or T
    of the transplant arm differs from both the random-direction and the permuted-alignment
    control, both family-bootstrap CIs excluding 0 in the same direction with >=
    ``TRACE_MIN_CLUSTERS`` families.  ``counts_for_prereg`` additionally needs a passed
    stand-in validation (PREREGISTRATION 18: ``standin`` from :func:`read_standin`), a frozen
    feature and no smoke run.  Sharing index / network activity deltas are reported
    (descriptive, ``sae_source``)."""
    probe = probe or {}
    align = align or {}
    standin = standin if standin is not None else read_standin(None)
    info = probe.get("info", {}) or {}
    tab = probe.get("table")
    prow = {}
    if tab is not None and len(tab) and "layer" in tab:
        sel = tab[tab["layer"] == int(frozen)]
        prow = sel.iloc[0].to_dict() if len(sel) else {}
    n_probe = int(prow.get("n_pairs", 0) or 0)
    delta_lo = float(prow.get("delta_ci_lo", np.nan))
    delta_hi = float(prow.get("delta_ci_hi", np.nan))
    ci_excludes_0 = bool(np.isfinite(delta_lo) and delta_lo > 0)
    counts = bool(prow.get("counts", False))
    feature_frozen = bool(info.get("feature_frozen", False))
    dsrc = str(probe.get("direction_source") or "")
    spec_ok = bool(probe.get("direction_spec_ok", True)) and dsrc.startswith(("probe:", "file:"))
    probe_ok = bool(feature_frozen and n_probe >= TRACE_MIN_PROBE_PAIRS and counts and ci_excludes_0 and spec_ok)
    al = align.get("alignments")
    rows_by_layer = {}
    if al is not None and len(al):
        rows_by_layer = {int(r["layer_tgt"]): r for r in al.to_dict("records")}
    arow = rows_by_layer.get(int(site_layer), {})
    gap = float(arow.get("gap", np.nan))
    r2_hold = float(arow.get("r2_holdout", np.nan))
    # the permuted-pairs gap alone is a weak null (a map that predicts anything gets *worse*
    # than the mean on permuted targets, docs/TRACE_SOURCE.md section 7): the residualised
    # hold-out R2 must also be positive
    align_ok_site = _align_layer_ok(arow) if arow else False
    planted = sorted(rows_by_layer)                      # the KV prefix is planted through every map
    failing = [l for l in planted if not _align_layer_ok(rows_by_layer[l])]
    align_ok_planted = bool(planted) and not failing
    ctr = align.get("controls")
    crow = {}
    if ctr is not None and len(ctr):
        sel = ctr[(ctr["layer_src"] == int(frozen)) & (ctr["layer_tgt"] == int(site_layer))]
        crow = sel.iloc[0].to_dict() if len(sel) else {}
    gain_lo = float(crow.get("gain_resid_ci_lo", np.nan))
    rows = []
    for mode in modes:
        e = effects[effects["mode"] == mode] if effects is not None and len(effects) else pd.DataFrame(columns=EFFECT_COLS)

        def _get(control, q):
            s = e[(e["control"] == control) & (e["quantity"] == q)]
            return s.iloc[0].to_dict() if len(s) else {}

        q_ok = {}
        for q in ("B_all", "T"):
            r1, r2 = _get("random", q), _get("permuted_alignment", q)
            same = bool(r1 and r2 and np.sign(r1.get("mean", 0.0)) == np.sign(r2.get("mean", 0.0)))
            q_ok[q] = bool(r1.get("ci_excludes_0", False) and r2.get("ci_excludes_0", False) and same)
        transplant_effect = bool(any(q_ok.values()))
        uses_site = mode in ("residual", "both")
        uses_kv = mode in ("kv", "both")
        align_ok = bool((align_ok_site or not uses_site) and (align_ok_planted or not uses_kv))
        claimed = bool(probe_ok and align_ok and transplant_effect)
        r_rand, r_perm = _get("random", "B_all"), _get("permuted_alignment", "B_all")
        t_rand, t_perm = _get("random", "T"), _get("permuted_alignment", "T")
        n_clusters = int(max([int(x.get("n_clusters", 0) or 0) for x in (r_rand, r_perm, t_rand, t_perm)] or [0]))
        reasons = []
        if not feature_frozen:
            reasons.append(f"probe not the pre-specified length-matched fan-out probe (used k={info.get('k_used')}, "
                           f"match={info.get('match')})")
        if not spec_ok:
            reasons.append(f"direction {dsrc or 'missing'} is not the frozen probe's direction "
                           f"({'; '.join((probe.get('file_info') or {}).get('spec_mismatch') or []) or 'unknown spec'})")
        if n_probe < TRACE_MIN_PROBE_PAIRS:
            reasons.append(f"probe has {n_probe} matched pairs (< {TRACE_MIN_PROBE_PAIRS})")
        elif not (counts and ci_excludes_0):
            reasons.append(f"probe criterion not met (trained - untrained CI [{delta_lo:.3f}, {delta_hi:.3f}])")
        if uses_site and not align_ok_site:
            reasons.append(f"residualised alignment at target layer {site_layer}: r2_holdout {r2_hold:.3f} (needs > 0), "
                           f"gap {gap:.3f} (needs >= {PREREG_R2_GAP})")
        if uses_kv and not align_ok_planted:
            reasons.append(f"KV prefix planted through {len(planted)} maps, {len(failing)} fail r2_holdout > 0 / gap >= "
                           f"{PREREG_R2_GAP} (target layers {failing})")
        if e.empty:
            reasons.append("no transplant readouts (substage failed or skipped)")
        elif n_clusters < TRACE_MIN_CLUSTERS:
            reasons.append(f"only {n_clusters} families (< {TRACE_MIN_CLUSTERS}): no transplant effect can be shown")
        elif not transplant_effect:
            reasons.append("no transplant effect beyond the random and permuted-alignment controls")
        counts_prereg = bool(feature_frozen and spec_ok and not is_smoke and standin.get("ok"))
        prereg_note = ("" if counts_prereg else
                       "; does not count for PREREGISTRATION 18: " + "; ".join(
                           x for x in [None if standin.get("ok") else str(standin.get("reason")),
                                       None if feature_frozen else "feature not frozen",
                                       None if spec_ok else "direction not the frozen probe's",
                                       "smoke run" if is_smoke else None] if x))
        rows.append(dict(
            mode=mode, source_kind="trace", frozen_layer=int(frozen), site_layer=int(site_layer),
            direction_source=probe.get("direction_source"), direction_spec_ok=spec_ok, k_requested=info.get("k_requested"),
            k_used=info.get("k_used"), probe_match=info.get("match"), feature_frozen=feature_frozen, probe_n_pairs=n_probe,
            auc_trained=float(prow.get("auc_trained", np.nan)), auc_untrained=float(prow.get("auc_untrained", np.nan)),
            probe_delta=float(prow.get("delta", np.nan)), probe_delta_ci_lo=delta_lo, probe_delta_ci_hi=delta_hi,
            probe_counts=counts, probe_ok=probe_ok,
            align_r2_holdout=r2_hold, align_r2_permuted=float(arow.get("r2_permuted", np.nan)),
            align_gap=gap, align_ok_site=align_ok_site, n_planted_layers=len(planted) if uses_kv else 0,
            n_planted_layers_ok=(len(planted) - len(failing)) if uses_kv else 0,
            planted_layers_failing=",".join(str(l) for l in failing) if uses_kv else "",
            align_ok=align_ok, align_gain_resid=float(crow.get("gain_resid", np.nan)),
            align_gain_resid_ci_lo=gain_lo, align_beats_untrained=bool(np.isfinite(gain_lo) and gain_lo > 0),
            align_r2_length_only=float(crow.get("r2_length_only", np.nan)),
            B_vs_random=float(r_rand.get("mean", np.nan)), B_vs_permuted=float(r_perm.get("mean", np.nan)),
            T_vs_random=float(t_rand.get("mean", np.nan)), T_vs_permuted=float(t_perm.get("mean", np.nan)),
            n_clusters=n_clusters, effect_B=q_ok["B_all"], effect_T=q_ok["T"],
            transplant_effect=transplant_effect,
            d_sharing_jaccard_vs_random=_sharing_delta(sharing, mode, "random", "sharing_jaccard", "K0|K8C1"),
            d_sharing_jaccard_vs_permuted=_sharing_delta(sharing, mode, "permuted_alignment", "sharing_jaccard", "K0|K8C1"),
            d_activity_K8C1_vs_random=_sharing_delta(sharing, mode, "random", "activity_frac_active", "K8C1"),
            d_activity_K8C1_vs_permuted=_sharing_delta(sharing, mode, "permuted_alignment", "activity_frac_active", "K8C1"),
            sae_source=sae_source, sharing_descriptive=True,
            sharing_effect_claimed=claimed,
            standin_validated=bool(standin.get("ok")), standin_path=standin.get("path"),
            counts_for_prereg=counts_prereg,
            claim=(("sharing effect: probe criterion, residualised alignment and transplant effect beyond both controls"
                    if claimed else "no claim: " + "; ".join(reasons)) + prereg_note)))
    return pd.DataFrame(rows)


def _trace_sae(mb: ModelBundle, site: HookSite, limits: C.Limits, kw: dict) -> tuple:
    """SAE of the sharing readout: given, a published one at ``site`` (``limits.use_sae``), a
    synthetic placeholder (``limits.synthetic_sae_if_missing``, smoke) or none."""
    if kw.get("sae") is not None:
        return kw["sae"], "given"
    if limits.use_sae:
        try:
            hp = C._sae_hookpoint(mb, site)
            if hp is not None:
                from ..sae import TopKSAE
                return TopKSAE.load_from_hub(hp[0], hp[1], device=str(mb.device)), f"hub:{hp[0]}/{hp[1]}"
        except Exception as e:  # noqa: BLE001 - falls through to the placeholder / none
            warnings.warn(f"stage L: no published SAE at {site.key}: {e!r}")
    if limits.synthetic_sae_if_missing:
        return C.synthetic_sae(mb, n_latents=int(limits.synthetic_sae_latents)), "synthetic_random"
    return None, "none"


def run_trace(cfg: RunConfig, mb: ModelBundle, run_dir, src, limits: C.Limits, kw: dict, t0: float, sub, variants) -> dict:
    """Stage L with a trace source; see the module docstring (section "Trace source")."""
    from ..chain import pairs as TPP

    empty = pd.DataFrame()
    seed = int(cfg.seed)
    model_dir = src.model_dir
    shards, shards_how = TPP.resolve_shards(model_dir, _trace_opt(kw, limits, "shards"))
    grid = list(C.prereg_layer_grid(mb.n_layers))
    site_layer = int(_trace_opt(kw, limits, "site_layer", _nearest_middle(grid, mb.n_layers)))
    site = HookSite(site_layer, "block_input")
    meas_site = C.dossier_site(mb)
    frozen = int(_trace_opt(kw, limits, "layer", src.n_layers))
    if not 0 <= frozen <= src.n_layers:
        raise ValueError(f"trace_layer {frozen} outside 0..{src.n_layers}")
    twin_seed = TPP.init_seed_of(model_dir)
    try:
        twin = TPP.untrained_twin(src, seed=twin_seed)
    except Exception as e:  # noqa: BLE001 - recorded; without a twin the probe criterion never counts
        warnings.warn(f"stage L: untrained twin of {model_dir} failed: {e!r}")
        sub.errors["untrained_twin"] = repr(e)
        twin = None

    split = str(_trace_opt(kw, limits, "split", "all"))
    standin = read_standin(_trace_opt(kw, limits, "standin"))

    def _data():
        if shards is None:
            raise FileNotFoundError(shards_how)
        idx = TPP.shard_index(shards, limit=_trace_opt(kw, limits, "limit"), max_len=src.max_len)
        if split != "all" and len(idx):     # only the LM's held-out (or training) transactions
            idx = idx[TPP.split_filter(idx["tx_hash"].tolist(), split, TPP.lm_val_frac(model_dir))].reset_index(drop=True)
        if idx.empty:
            raise ValueError(f"no transactions in {shards} (split={split})")
        return idx
    index = sub.run("trace_data", _data, empty)
    probe = sub.run("trace_probe", lambda: trace_probe(src, twin, index, shards, frozen, limits, kw, seed), dict)
    align = sub.run("trace_alignment", lambda: trace_alignment(src, twin, mb, index, shards, frozen, site_layer, probe, limits, kw,
                                                               seed), dict)

    def _transplant():
        if not align.get("aligns"):
            raise RuntimeError("no alignments")
        return trace_transplant(mb, src, align["aligns"], probe.get("v_src"), align.get("native"), variants, probe, align, site,
                                meas_site, cfg, limits, kw)
    tp = sub.run("trace_transplant", _transplant, dict)
    sae, sae_source = _trace_sae(mb, meas_site, limits, kw)
    bt, tt, effects, sharing = (sub.run("trace_readouts", lambda: trace_readouts(tp, sae, sae_source, meas_site, int(limits.n_boot), seed),
                                        (empty, empty, empty, empty)) if tp else (empty, empty, empty, empty))
    verdict_t = trace_verdict(probe, align, effects, sharing, frozen, site_layer, sae_source, bool(limits.is_smoke),
                              standin=standin)
    tp_df = tp.get("df", empty) if tp else empty
    tables = {
        "transfer_test": pd.DataFrame(columns=TT_COLS),
        "transfer_summary": pd.DataFrame(columns=["best_layer"] + TT_COLS[3:] + ["n_validated_layers", "n_layers"]),
        "layer_matching": align.get("layer_matching", empty) if align else empty,
        "alignments": align.get("alignments", empty) if align else empty,
        "transplant": tp_df,
        "transplant_summary": summarise_transplant(tp_df, float("nan")),
        "verdict": pd.DataFrame([dict(source_kind="trace", self_consistency=False, validated=np.nan, counts_for_prereg=False,
                                      functional_transfer="skipped",
                                      note="contrast 18 (stand-in validation) needs a language source; "
                                           "the trace source is judged in trace_verdict")]),
        "trace_probe": probe.get("table", empty) if probe else empty,
        "trace_probe_match": probe.get("match", empty) if probe else empty,
        "trace_alignment_controls": align.get("controls", empty) if align else empty,
        "trace_bt": bt, "trace_T": tt, "trace_effects": effects, "trace_sharing": sharing, "trace_verdict": verdict_t,
    }
    tables = {k: (v if isinstance(v, pd.DataFrame) else empty) for k, v in tables.items()}
    sub.write_tracebacks(run_dir)
    info = (probe or {}).get("info", {})
    claimed = bool(len(verdict_t) and verdict_t["sharing_effect_claimed"].any())
    return C.finish_stage(
        run_dir, "L", cfg, mb, tables, limits=limits, t0=t0, site=site,
        source=dict(kind="trace", name=getattr(src, "name", ""), model_dir=str(model_dir), d_model=int(src.d_model),
                    n_layers=int(src.n_layers), vocab=int(len(src.vocab)), max_len=int(src.max_len)),
        functional_transfer=dict(skipped=True, reason=FT_SKIP_REASON),
        trace=dict(shards=shards, shards_from=shards_how, n_transactions=int(len(index)), frozen_layer=frozen,
                   site_layer=site_layer, measure_site=meas_site.key, untrained_twin_seed=int(twin_seed),
                   probe=info, direction_source=(probe or {}).get("direction_source"),
                   direction_file=(probe or {}).get("file_info"), direction_meta=(probe or {}).get("direction_meta"),
                   n_alignment_pairs=int((align or {}).get("n_pairs", 0)), covariates=(align or {}).get("covariates"),
                   split=split, lm_split=dict(probe=(info or {}).get("lm_split"), alignment=(align or {}).get("lm_split")),
                   standin=standin, n_families=int((tp or {}).get("n_families", 0)),
                   alignment_method="procrustes on length-residualised pooled states (intercept from raw means)",
                   native_axis=bool((align or {}).get("native") is not None),
                   carrier="matched probe negatives (fan-out 1); one carrier per formulation (K=0 and K=8/C1)",
                   n_prompts=int((tp or {}).get("n_prompts", 0)), n_substitute=(tp or {}).get("n_substitute"),
                   sae_source=sae_source, have_chain_controls=bool(TPP.HAVE_CHAIN_CONTROLS),
                   sharing_effect_claimed=claimed),
        validated=None, self_consistency=False,
        transplant=dict(eps=limits.l_eps, n_prefix=limits.l_n_prefix, n_gen=cfg.gen_n_new, n_samples=cfg.gen_n_samples),
        substages=sub.report(),
        prereg=dict(contrast_18="not applicable to a trace source (the stand-in validation needs a language source; "
                                "PREREGISTRATION 18: a trace source counts only after a passed stand-in validation, "
                                "trace_verdict.counts_for_prereg requires it via the standin option)",
                    trace="trace_verdict.sharing_effect_claimed = probe_ok (feature_frozen AND probe criterion: CI of "
                          "trained - untrained split-half AUC above 0 at the frozen layer AND v_src from that probe "
                          "specification) AND align_ok (residualised R2 > 0 and gap >= 0.2 at the site for residual, "
                          "on every planted layer for kv, both for both) AND transplant_effect (B or T vs random and "
                          f"permuted_alignment, CIs exclude 0, >= {TRACE_MIN_CLUSTERS} families); sharing / activity "
                          "deltas are descriptive"))
