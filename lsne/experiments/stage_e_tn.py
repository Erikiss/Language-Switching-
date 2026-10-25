"""Stage E - tensor-network linear response (DESIGN section 5, D8).

For the K=0 and K=8/C1 variants of the first ``Limits.n_families_tn`` families:

1. reference cache with attention patterns (``run_with_cache(..., need_attn=True)``);
2. per variant and direction (``lang_dom``, ``ent``): the exact first-order response
   (``linear_response_jvp``, forward-mode AD in float32) at *every* residual snapshot
   ``l0 .. L`` of the window, with the perturbation at the block input of block
   ``l0 = n_layers // 3`` and the injection position = last token of unit 9 (clamped into
   the window of the last ``cfg.tn_window`` positions);
3. the truncation basis ``P [d, r]`` (``cfg.tn_rank``) is the top-``r`` *uncentred* SVD
   basis of all those linear-response rows (:func:`response_basis`).  The factorised
   lattice truncates the residual stream to ``span(P)`` at every block, so the basis must
   contain the intermediate responses of the chain, not the residual *states*: a PCA of
   the states at the injection site captures ~99 % of the state variance of pythia-160m
   at r = 64 but only ~10 % of the response (the response lives in low-variance
   directions), which made the old ``factorization_error`` ~ 1.0 regardless of the
   factorisation.  ``tn_basis`` now reports the explained variance of the response rows;
   the per-row output-side capture ``||P P^T lin|| / ||lin||`` is ``linear_captured``;
4. per variant: the factorised lattice of the blocks ``l0 .. L-1``
   (:func:`lsne.tn_lattice.build_factorized_lattice`) with contraction paths from TNCO
   (fallback opt_einsum inside ``optimize_path``) and from opt_einsum plus the greedy / dp
   reference costs; when it fits into ``tn_full_rank_budget_mb`` (default 512 MB; always
   on pythia-14m, never on 160m: 8.5 GB) also the *untruncated* lattice (``r = d``,
   basis = identity) as the reference that separates the factorisation error from the
   truncation error (``factorization_error_full``, ``truncation_error``);
5. per direction and dose in ``Limits.tn_doses``: the measured non-linear response
   ``h'(+delta) - h`` **and** ``h'(-delta) - h`` at the output of the last block.  The
   measurement runs on a float64 copy of the model (:func:`float64_bundle`) with the
   attention softmax in float64 (:func:`float64_attention`): in float32 the forward pass
   has a round-off floor of ~1e-3 in the residual (pythia-14m), which at eps = 0.01 is ~10 %
   of the linear response and looks exactly like the hypothesised "large non-linearity at
   small eps" (and GPT-NeoX computes the softmax in float32 even for a float64 model, so
   the dtype of the copy alone is not enough).  The +/- pair gives the odd part
   (first + third order) and the even part (second order, ``= eps^2 delta^T H delta / 2``):
   ``nonlinearity_odd = ||odd - lin|| / ||lin||`` cancels the second order,
   ``nonlinearity_even = ||even|| / ||lin||`` is the second-order share.  The float32
   floor is still measured (``noise_floor_f32 = ||h_f32 - h_f64||`` per position) and
   every position with ``snr_f32 = ||lin|| / floor < 10`` is flagged (``above_floor_f32``):
   those are the positions where a float32 measurement would be noise.

Tables
------
tn_compare   per (variant, direction, eps, window position): norms of measured (+/-),
             linear, factorised (and full-rank factorised) responses, nonlinearity (plain,
             odd, even), factorization_error (vs. the full linear response, within
             span(P), and of the full-rank reference), truncation_error, cosines,
             linear_captured, noise_floor_f32, snr_f32, above_floor_f32
tn_summary   per (variant, direction, eps): norm-weighted means over the positions from
             the injection on, the last-position values, source_captured (input side,
             ``||P^T dir||^2``), linear_captured_wmean / linear_captured_frob (output side,
             ``||P P^T lin|| / ||lin||`` per position resp. over all positions from the
             injection on), n_below_floor_f32, measured_dtype
tn_paths     per (variant, basis, backend): n_tensors, path cost (log10 FLOPs), runtime,
             fallback (basis ``response_svd`` = truncated lattice, ``identity`` = full rank)
tn_basis     cumulative explained variance of the response rows per rank (and, for
             comparison, of the centred states at the injection site in the same basis)

Keyword options (``run(..., **kw)``): ``tn_measure_dtype`` (``"float64"`` default,
``"float32"``), ``tn_full_rank_reference`` (``None`` = automatic by memory budget, or
bool), ``tn_full_rank_budget_mb`` (default 512).
"""
from __future__ import annotations

import contextlib
import copy
import time
import warnings
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ..config import RunConfig, HookSite
from ..model import Injector, ModelBundle, make_dose_delta, run_with_cache
from ..tn_lattice import (LatticeTN, build_factorized_lattice, compare_responses, contract, cost_reference,
                          linear_response_jvp, optimize_path)
from . import common as C

TN_FAMILIES = ("lang_dom", "ent")
#: Memory budget (MB) of the ``K_l`` tensors below which the untruncated (r = d) reference
#: lattice is built automatically.
FULL_RANK_BUDGET_MB = 512.0
#: Positions whose linear response is below ``SNR_FLOOR_FACTOR`` x the float32 round-off
#: floor are flagged as not measurable in float32.
SNR_FLOOR_FACTOR = 10.0


def tn_subset(variants, n_families: int) -> list:
    """K=0 and K=8/C1 variants of the first ``n_families`` family ids (standard families)."""
    ids: list[str] = []
    for v in variants:
        if not C.is_original(v) and v.family_id not in ids:
            ids.append(v.family_id)
    keep = set(ids[: int(n_families)])
    return [v for v in variants if not C.is_original(v) and v.family_id in keep
            and (int(v.K), str(v.C)) in ((0, "C1"), (8, "C1"))]


def with_source(tn: LatticeTN, position_in_window: int, direction: torch.Tensor) -> LatticeTN:
    """Copy of ``tn`` whose source tensor carries ``onehot(position) (x) P^T direction``."""
    P = torch.as_tensor(tn.meta["basis"])
    inds, src = tn.tensors[0]
    new = torch.zeros(src.shape)
    new[int(position_in_window)] = P.T @ direction.float().cpu()
    tensors = [(inds, new.numpy().astype(src.dtype))] + list(tn.tensors[1:])
    return LatticeTN(tensors=tensors, output_inds=tn.output_inds, meta=dict(tn.meta))


def explained_variance(acts: torch.Tensor, P: torch.Tensor) -> np.ndarray:
    """Cumulative fraction of the (centred) variance of ``acts [N, d]`` captured by the
    first k columns of ``P [d, r]`` (PCA-style explained variance of the *states*)."""
    X = acts.float() - acts.float().mean(0, keepdim=True)
    tot = float((X ** 2).sum())
    proj = (X @ P) ** 2                                        # [N, r]
    per = proj.sum(0).numpy() / max(tot, 1e-30)
    return np.cumsum(per)


def response_basis(rows, r: int) -> tuple[torch.Tensor, np.ndarray]:
    """Orthonormal basis ``P [d, r]`` of the top-``r`` right singular vectors of the
    *uncentred* response rows ``rows [N, d]`` (float32) and the cumulative fraction of
    ``||rows||_F^2`` captured per rank.

    Rows are the linear responses of the chain at all snapshots and window positions
    (norm-weighted: large responses dominate, which minimises the total truncation error
    of the chain in the least-squares sense).  No centring: the mean response is itself a
    response and must stay in the span.  Column signs are fixed (largest-magnitude entry
    positive) and the basis is completed with a deterministic orthonormal complement when
    fewer than ``r`` singular directions exist."""
    X = torch.as_tensor(np.asarray(rows), dtype=torch.float64)
    X = X.reshape(-1, int(X.shape[-1]))
    N, d = X.shape
    r = int(max(1, min(r, d)))
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)
    tot = float((S ** 2).sum())
    ev = np.cumsum((S ** 2).numpy()) / max(tot, 1e-30)
    P = Vh[: min(r, N)].T.contiguous()                           # [d, r0]
    sign = torch.sign(P[P.abs().argmax(0), torch.arange(P.shape[1])])
    sign[sign == 0] = 1.0
    P = P * sign[None, :]
    if P.shape[1] < r:
        Q, _ = torch.linalg.qr(torch.cat([P, torch.eye(d, dtype=P.dtype)], 1))
        P = torch.cat([P, Q[:, P.shape[1]: r]], 1)
    ev = np.concatenate([ev, np.full(max(0, r - ev.size), ev[-1] if ev.size else 0.0)])[:r]
    return P.to(torch.float32), ev


# --------------------------------------------------------------------------------------
# float64 measurement of level (a)
# --------------------------------------------------------------------------------------
def float64_bundle(mb: ModelBundle) -> Optional[ModelBundle]:
    """Deep copy of ``mb.model`` in float64 wrapped in a new :class:`ModelBundle`
    (``spec.dtype = "float64"``); ``None`` with a warning when the copy or a float64 forward
    pass is not possible on this device (e.g. MPS)."""
    try:
        model = copy.deepcopy(mb.model).double().eval()
        for p in model.parameters():
            p.requires_grad_(False)
        out = ModelBundle(model=model, tokenizer=mb.tokenizer, spec=replace(mb.spec, dtype="float64"),
                          device=mb.device)
        with torch.no_grad():
            out.model(input_ids=torch.zeros(1, 2, dtype=torch.long, device=mb.device), use_cache=False)
        return out
    except Exception as e:  # noqa: BLE001 - unsupported dtype / device
        warnings.warn(f"float64 model copy not available ({e!r}); stage E measures in float32")
        return None


@contextlib.contextmanager
def float64_attention():
    """Within the context GPT-NeoX's eager attention computes the softmax in float64 when
    the query is float64 (transformers hard-codes ``dtype=torch.float32``); any other dtype
    is untouched (the original function is called), so float32 models are bitwise
    unaffected.  Yields ``True`` when the patch could be installed."""
    try:
        from transformers.models.gpt_neox import modeling_gpt_neox as mg
        orig = mg.eager_attention_forward
    except (ImportError, AttributeError):   # pragma: no cover - other transformers versions
        yield False
        return

    def eager_float64(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        if query.dtype != torch.float64:
            return orig(module, query, key, value, attention_mask, scaling, dropout=dropout, **kwargs)
        w = torch.matmul(query, key.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.functional.softmax(w, dim=-1, dtype=torch.float64)
        w = nn.functional.dropout(w, p=dropout, training=module.training)
        return torch.matmul(w, value).transpose(1, 2).contiguous(), w

    mg.eager_attention_forward = eager_float64
    try:
        yield True
    finally:
        mg.eager_attention_forward = orig


def measured_response(bundle: ModelBundle, ref_out: torch.Tensor, ids: torch.Tensor, site_from: HookSite,
                      position: int, delta: torch.Tensor, out_site: HookSite, window: slice) -> np.ndarray:
    """``h'(delta) - h`` at ``out_site`` on the window (float64 numpy ``[T_w, d]``);
    ``ref_out`` is the unperturbed residual ``[T, d]`` of the same bundle."""
    with torch.no_grad(), Injector(bundle, site_from, delta, positions=[position], mode="add"):
        pert = run_with_cache(bundle, ids, [out_site])
    return (pert.resid[out_site.key][0, window].double() - ref_out[window].double()).cpu().numpy()


def full_rank_bytes(mb: ModelBundle, n_blocks: int, T_w: int, itemsize: int = 4) -> int:
    """Memory of the ``K_l`` tensors of the untruncated lattice: ``n_blocks (2+H) T_w d^2``."""
    return int(n_blocks * (2 + mb.n_heads) * T_w * mb.d_model ** 2 * itemsize)


def _path_rows(tn: LatticeTN, v, basis_name: str, limits: C.Limits, seed: int) -> tuple[dict, list[dict]]:
    """Path optimisation of ``tn`` with both backends + reference costs; returns
    ``(infos, rows)`` for ``tn_paths``."""
    infos, rows = {}, []
    n_blocks = len(tn.meta["layers"])
    for backend in ("tnco", "opt_einsum"):
        info = optimize_path(tn, backend=backend, n_steps=limits.tn_n_steps, n_runs=limits.tn_n_runs, seed=seed)
        infos[backend] = info
        rows.append(dict(C.variant_row(v), basis=basis_name, backend=backend, backend_used=info.get("backend"),
                         n_tensors=tn.n_tensors, T_w=tn.meta["T_w"], r=int(tn.meta["r"]), n_blocks=n_blocks,
                         cost_log10=float(info.get("cost_log10", float("nan"))),
                         cost_log10_opt_einsum=float(info.get("cost_log10_opt_einsum", float("nan"))),
                         runtime_s=float(info.get("runtime_s", float("nan"))),
                         largest_intermediate_log10=float(info.get("largest_intermediate_log10", float("nan"))),
                         fallback_reason=str(info.get("fallback_reason", "")), nbytes=tn.nbytes()))
    ref = cost_reference(tn)
    for name in ("greedy", "dp"):
        rows.append(dict(C.variant_row(v), basis=basis_name, backend=f"reference_{name}", backend_used="opt_einsum",
                         n_tensors=tn.n_tensors, T_w=tn.meta["T_w"], r=int(tn.meta["r"]), n_blocks=n_blocks,
                         cost_log10=float(ref.get(f"{name}_cost_log10", float("nan"))),
                         cost_log10_opt_einsum=float(ref.get(f"{name}_cost_log10", float("nan"))),
                         runtime_s=float(ref.get(f"{name}_runtime_s", float("nan"))),
                         largest_intermediate_log10=float("nan"), fallback_reason="", nbytes=tn.nbytes()))
    return infos, rows


def _wmean(values: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(values) & np.isfinite(w)
    s = float(w[ok].sum()) if ok.any() else 0.0
    return float((values[ok] * w[ok]).sum() / s) if s > 0 else float("nan")


def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Linear response: measured (float64, +/- dose) vs. exact first-order vs. factorised
    tensor network in a response-fitted basis (with an untruncated reference when it fits)."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    site = C.dossier_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original, sae=kw.get("sae"))
    dir_names = bank.names(TN_FAMILIES) or list(bank.keys())[:1]

    L, d = mb.n_layers, mb.d_model
    l0 = C.depth_layer(mb)
    site_from = HookSite(l0, "block_input")
    out_site = HookSite(L - 1, "block_output")
    layers = range(l0, L)
    snapshots = list(range(l0, L + 1))
    subset = tn_subset(variants, limits.n_families_tn)
    r = int(min(cfg.tn_rank, d))

    # ---- measurement dtype (finding: float32 round-off floor) -------------------------
    measure_dtype = str(kw.get("tn_measure_dtype", "float64"))
    mb64 = float64_bundle(mb) if measure_dtype == "float64" else None
    measured_dtype = "float64" if mb64 is not None else "float32"

    # ---- reference caches and exact linear responses at every snapshot ---------------
    caches = {v.tag: run_with_cache(mb, C.ids_tensor(mb, v), "all_block_inputs", need_attn=True) for v in subset}
    geom: dict[str, tuple[slice, int, int, int]] = {}
    lin_all: dict[tuple[str, str], np.ndarray] = {}        # (tag, direction) -> [n_snap, T_w, d]
    for v in subset:
        T = int(v.n_tokens)
        T_w = int(min(cfg.tn_window, T))
        w0 = T - T_w
        position = int(min(max(C.unit_last_position(v), w0), T - 1))
        geom[v.tag] = (slice(w0, T), position, w0, T_w)
        ids = C.ids_tensor(mb, v)
        for name in dir_names:
            lin_all[(v.tag, name)] = linear_response_jvp(mb, ids, site_from, position, bank[name].vec.to(mb.device),
                                                         snapshots, slice(w0, T))

    # ---- truncation basis fitted on the responses ------------------------------------
    if lin_all:
        rows = np.concatenate([a.reshape(-1, d) for a in lin_all.values()], 0)
        P, ev = response_basis(rows, r)
        acts = torch.cat([caches[v.tag].resid[site_from.key][0].float().cpu() for v in subset], 0)
        basis_df = pd.DataFrame(dict(rank=np.arange(1, r + 1), explained_variance_cum=ev,
                                     explained_variance_states_cum=explained_variance(acts, P),
                                     n_rows=int(rows.shape[0]), source="linear_response_svd"))
    else:
        P, basis_df = torch.eye(d), pd.DataFrame(columns=["rank", "explained_variance_cum", "explained_variance_states_cum",
                                                          "n_rows", "source"])
    Pn = P.numpy().astype(np.float64)

    # ---- untruncated reference lattice? ----------------------------------------------
    T_w_max = max((g[3] for g in geom.values()), default=0)
    fr_bytes = full_rank_bytes(mb, len(list(layers)), T_w_max)
    budget = float(kw.get("tn_full_rank_budget_mb", FULL_RANK_BUDGET_MB)) * 2 ** 20
    want_full = kw.get("tn_full_rank_reference")
    if r >= d:
        full_mode = "same_lattice"          # the main lattice is already untruncated
    elif (bool(want_full) if want_full is not None else fr_bytes <= budget):
        full_mode = "computed"
    else:
        full_mode = "skipped"

    cmp_rows, summ_rows, path_rows = [], [], []
    with float64_attention() as softmax64:
        if mb64 is None:
            softmax64 = False
        for v in C.progress(subset, "E tensor network", enabled=limits.progress):
            cache = caches[v.tag]
            window, position, w0, T_w = geom[v.tag]
            w_positions = list(range(w0, w0 + T_w))
            ids = C.ids_tensor(mb, v)
            tn = build_factorized_lattice(mb, cache, site_from, window, P, layers, position=position, direction=P[:, 0])
            infos, rows_p = _path_rows(tn, v, "response_svd", limits, cfg.seed)
            path_rows.extend(rows_p)
            path = infos["tnco"]["path"]
            tn_full, path_full = None, None
            if full_mode == "computed":
                tn_full = build_factorized_lattice(mb, cache, site_from, window, torch.eye(d), layers,
                                                   position=position, direction=torch.eye(d)[:, 0])
                infos_full, rows_f = _path_rows(tn_full, v, "identity", limits, cfg.seed)
                path_rows.extend(rows_f)
                path_full = infos_full["tnco"]["path"]
            # unperturbed reference of the measuring bundle and the float32 round-off floor
            ref32 = cache.resid[out_site.key][0].detach()
            if mb64 is not None:
                ref64 = run_with_cache(mb64, ids, [out_site]).resid[out_site.key][0].detach()
                floor = (ref32[window].double() - ref64[window].double()).norm(dim=1).cpu().numpy()
                bundle, ref_out = mb64, ref64
            else:
                floor = np.full(T_w, np.nan)
                bundle, ref_out = mb, ref32
            h_ref = cache.resid[site_from.key][0, position].detach()
            for name in dir_names:
                unit_vec = bank[name].vec.to(h_ref.device)
                captured = float((P.T @ unit_vec.float().cpu()).norm() ** 2)          # input side
                lin_unit = lin_all[(v.tag, name)][-1].astype(np.float64)                # [T_w, d] at snapshot L
                lin_in = (lin_unit @ Pn) @ Pn.T                                          # P P^T lin
                y = contract(with_source(tn, position - w0, unit_vec), path)              # [T_w, r]
                fact_unit = np.asarray(y, dtype=np.float64) @ Pn.T
                if full_mode == "computed":
                    yf = contract(with_source(tn_full, position - w0, unit_vec), path_full)   # basis = identity
                    fact_full_unit = np.asarray(yf, dtype=np.float64)
                elif full_mode == "same_lattice":
                    fact_full_unit = fact_unit
                else:
                    fact_full_unit = None
                ln_unit = np.linalg.norm(lin_unit, axis=1)
                lin_captured = np.linalg.norm(lin_in, axis=1) / np.maximum(ln_unit, 1e-30)
                err_inspan = np.linalg.norm(lin_in - fact_unit, axis=1) / np.maximum(np.linalg.norm(lin_in, axis=1), 1e-30)
                for eps in limits.tn_doses:
                    delta = make_dose_delta(h_ref, unit_vec, float(eps)).to(h_ref.dtype)
                    scale = float(delta.float().norm())
                    m_plus = measured_response(bundle, ref_out, ids, site_from, position, delta, out_site, window)
                    m_minus = measured_response(bundle, ref_out, ids, site_from, position, -delta, out_site, window)
                    lin, fact = lin_unit * scale, fact_unit * scale
                    df = compare_responses(m_plus, lin, fact, positions=w_positions)
                    odd, even = (m_plus - m_minus) / 2.0, (m_plus + m_minus) / 2.0
                    ln = np.maximum(np.linalg.norm(lin, axis=1), 1e-30)
                    df["measured_norm_minus"] = np.linalg.norm(m_minus, axis=1)
                    df["odd_norm"] = np.linalg.norm(odd, axis=1)
                    df["even_norm"] = np.linalg.norm(even, axis=1)
                    df["nonlinearity_odd"] = np.linalg.norm(odd - lin, axis=1) / ln
                    df["nonlinearity_even"] = np.linalg.norm(even, axis=1) / ln
                    df["linear_captured"] = lin_captured
                    df["factorization_error_inspan"] = err_inspan
                    if fact_full_unit is not None:
                        ff = fact_full_unit * scale
                        df["factorized_full_norm"] = np.linalg.norm(ff, axis=1)
                        df["factorization_error_full"] = np.linalg.norm(lin - ff, axis=1) / ln
                        df["truncation_error"] = np.linalg.norm(fact - ff, axis=1) / ln
                    else:
                        df["factorized_full_norm"] = np.nan
                        df["factorization_error_full"] = np.nan
                        df["truncation_error"] = np.nan
                    df["noise_floor_f32"] = floor
                    df["snr_f32"] = ln / np.maximum(floor, 1e-30)
                    df["above_floor_f32"] = df["snr_f32"] >= SNR_FLOOR_FACTOR
                    for k, val in dict(C.variant_row(v), direction=name, eps=float(eps), inject_pos=position,
                                       delta_norm=scale, source_captured=captured, measured_dtype=measured_dtype,
                                       softmax_float64=bool(softmax64)).items():
                        df[k] = val
                    df["rel_pos"] = df["position"] - position
                    cmp_rows.append(df)
                    after = df[df["position"] >= position]
                    sel = np.arange(T_w) >= position - w0
                    w = after["linear_norm"].to_numpy(dtype=float)
                    col = lambda c: after[c].to_numpy(dtype=float)  # noqa: E731
                    # global (Frobenius) output-side capture over the positions from the injection on
                    captured_out = float(np.linalg.norm(lin_in[sel]) / max(np.linalg.norm(lin_unit[sel]), 1e-30))
                    summ_rows.append(dict(
                        C.variant_row(v), direction=name, eps=float(eps), inject_pos=position, T_w=T_w,
                        r=int(P.shape[1]), source_captured=captured, linear_captured_frob=captured_out,
                        n_positions=int(len(after)),
                        measured_dtype=measured_dtype, softmax_float64=bool(softmax64),
                        full_rank_reference=full_mode,
                        nonlinearity_wmean=_wmean(col("nonlinearity"), w),
                        nonlinearity_odd_wmean=_wmean(col("nonlinearity_odd"), w),
                        nonlinearity_even_wmean=_wmean(col("nonlinearity_even"), w),
                        factorization_error_wmean=_wmean(col("factorization_error"), w),
                        factorization_error_inspan_wmean=_wmean(col("factorization_error_inspan"), w),
                        factorization_error_full_wmean=_wmean(col("factorization_error_full"), w),
                        truncation_error_wmean=_wmean(col("truncation_error"), w),
                        linear_captured_wmean=_wmean(col("linear_captured"), w),
                        noise_floor_f32_wmean=_wmean(col("noise_floor_f32"), w),
                        snr_f32_min=float(np.nanmin(col("snr_f32"))) if np.isfinite(col("snr_f32")).any() else float("nan"),
                        n_below_floor_f32=int((~after["above_floor_f32"].to_numpy(dtype=bool)).sum())
                        if np.isfinite(floor).any() else -1,
                        nonlinearity_last=float(df["nonlinearity"].iloc[-1]),
                        nonlinearity_odd_last=float(df["nonlinearity_odd"].iloc[-1]),
                        nonlinearity_even_last=float(df["nonlinearity_even"].iloc[-1]),
                        factorization_error_last=float(df["factorization_error"].iloc[-1]),
                        factorization_error_full_last=float(df["factorization_error_full"].iloc[-1]),
                        linear_captured_last=float(df["linear_captured"].iloc[-1]),
                        cos_measured_linear_last=float(df["cos_measured_linear"].iloc[-1]),
                        cos_linear_factorized_last=float(df["cos_linear_factorized"].iloc[-1]),
                        measured_norm_last=float(df["measured_norm"].iloc[-1]),
                        linear_norm_last=float(df["linear_norm"].iloc[-1]),
                        factorized_norm_last=float(df["factorized_norm"].iloc[-1]),
                        snr_f32_last=float(df["snr_f32"].iloc[-1]),
                        path_backend=infos["tnco"].get("backend", ""),
                        path_cost_log10=float(infos["tnco"].get("cost_log10_opt_einsum", float("nan")))))
    softmax_used = bool(mb64 is not None and softmax64)
    del caches, lin_all, mb64
    tables = {
        "tn_compare": pd.concat(cmp_rows, ignore_index=True) if cmp_rows else pd.DataFrame(),
        "tn_summary": pd.DataFrame(summ_rows),
        "tn_paths": pd.DataFrame(path_rows),
        "tn_basis": basis_df,
    }
    return C.finish_stage(run_dir, "E", cfg, mb, tables, limits=limits, t0=t0, site_from=site_from, layers=list(layers),
                          bank_source=bank.meta.get("source", ""),
                          rank=int(P.shape[1]), window=int(cfg.tn_window), directions=dir_names, n_subset=len(subset),
                          subset=[v.tag for v in subset], basis="linear_response_svd",
                          basis_n_rows=int(basis_df["n_rows"].iloc[0]) if len(basis_df) else 0,
                          measured_dtype=measured_dtype, softmax_float64=softmax_used,
                          full_rank_reference=full_mode, full_rank_bytes=fr_bytes,
                          full_rank_budget_mb=float(kw.get("tn_full_rank_budget_mb", FULL_RANK_BUDGET_MB)))
