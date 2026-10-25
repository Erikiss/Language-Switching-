"""Stage E - tensor-network linear response (DESIGN section 5, D8).

For the K=0 and K=8/C1 variants of the first ``Limits.n_families_tn`` families:

1. reference cache with attention patterns (``run_with_cache(..., need_attn=True)``);
2. PCA basis ``P [d, r]`` (``cfg.tn_rank``) of the block-input residuals at the injection
   site (block input of block ``n_layers // 3``) pooled over the subset;
3. per variant: window = the last ``cfg.tn_window`` positions, injection position = the
   last token of unit 9 (clamped into the window); the factorised lattice of the blocks
   ``n_layers//3 .. n_layers-1`` (:func:`lsne.tn_lattice.build_factorized_lattice`);
   contraction paths from TNCO (fallback opt_einsum inside ``optimize_path``) and from
   opt_einsum, plus the greedy / dp reference costs;
4. per direction (``lang_dom``, ``ent``) and dose in ``Limits.tn_doses``: the measured
   non-linear response (residual differences at the output of the last block), the exact
   first-order response (``linear_response_jvp``) and the tensor-network response, all
   for the same dose vector; :func:`lsne.tn_lattice.compare_responses` per position.

Tables
------
tn_compare   per (variant, direction, eps, window position): norms, nonlinearity,
             factorisation error, cosines
tn_summary   per (variant, direction, eps): norm-weighted means of the per-position errors
tn_paths     per (variant, backend): n_tensors, path cost (log10 FLOPs), runtime, fallback
tn_basis     explained variance of the PCA basis per rank
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
import torch

from ..config import RunConfig, HookSite
from ..model import Injector, ModelBundle, make_dose_delta, run_with_cache
from ..tn_lattice import (LatticeTN, build_factorized_lattice, compare_responses, contract, cost_reference,
                          linear_response_jvp, optimize_path, pca_basis)
from . import common as C

TN_FAMILIES = ("lang_dom", "ent")


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
    """Cumulative fraction of the (centred) variance captured by the first k basis vectors."""
    X = acts.float() - acts.float().mean(0, keepdim=True)
    tot = float((X ** 2).sum())
    proj = (X @ P) ** 2                                        # [N, r]
    per = proj.sum(0).numpy() / max(tot, 1e-30)
    return np.cumsum(per)


def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Linear response: measured vs. exact first-order vs. factorised tensor network."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    site = C.dossier_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original, sae=kw.get("sae"))
    dir_names = bank.names(TN_FAMILIES) or list(bank.keys())[:1]

    L = mb.n_layers
    l0 = C.depth_layer(mb)
    site_from = HookSite(l0, "block_input")
    out_site = HookSite(L - 1, "block_output")
    layers = range(l0, L)
    subset = tn_subset(variants, limits.n_families_tn)

    caches = {v.tag: run_with_cache(mb, C.ids_tensor(mb, v), "all_block_inputs", need_attn=True) for v in subset}
    if subset:
        acts = torch.cat([caches[v.tag].resid[site_from.key][0].float().cpu() for v in subset], 0)
        r = int(min(cfg.tn_rank, mb.d_model))
        P = pca_basis(acts, r)
        ev = explained_variance(acts, P)
        basis_df = pd.DataFrame(dict(rank=np.arange(1, r + 1), explained_variance_cum=ev))
    else:
        P, basis_df = torch.eye(mb.d_model), pd.DataFrame(columns=["rank", "explained_variance_cum"])

    cmp_rows, summ_rows, path_rows = [], [], []
    for v in C.progress(subset, "E tensor network", enabled=limits.progress):
        cache = caches[v.tag]
        T = int(v.n_tokens)
        T_w = int(min(cfg.tn_window, T))
        w0 = T - T_w
        window = slice(w0, T)
        position = int(min(max(C.unit_last_position(v), w0), T - 1))
        w_positions = list(range(w0, T))
        tn = build_factorized_lattice(mb, cache, site_from, window, P, layers, position=position, direction=P[:, 0])
        infos = {}
        for backend in ("tnco", "opt_einsum"):
            info = optimize_path(tn, backend=backend, n_steps=limits.tn_n_steps, n_runs=limits.tn_n_runs, seed=cfg.seed)
            infos[backend] = info
            path_rows.append(dict(C.variant_row(v), backend=backend, backend_used=info.get("backend"), n_tensors=tn.n_tensors,
                                  T_w=T_w, r=int(P.shape[1]), n_blocks=len(list(layers)),
                                  cost_log10=float(info.get("cost_log10", float("nan"))),
                                  cost_log10_opt_einsum=float(info.get("cost_log10_opt_einsum", float("nan"))),
                                  runtime_s=float(info.get("runtime_s", float("nan"))),
                                  largest_intermediate_log10=float(info.get("largest_intermediate_log10", float("nan"))),
                                  fallback_reason=str(info.get("fallback_reason", "")), nbytes=tn.nbytes()))
        ref = cost_reference(tn)
        for name in ("greedy", "dp"):
            path_rows.append(dict(C.variant_row(v), backend=f"reference_{name}", backend_used="opt_einsum", n_tensors=tn.n_tensors,
                                  T_w=T_w, r=int(P.shape[1]), n_blocks=len(list(layers)),
                                  cost_log10=float(ref.get(f"{name}_cost_log10", float("nan"))),
                                  cost_log10_opt_einsum=float(ref.get(f"{name}_cost_log10", float("nan"))),
                                  runtime_s=float(ref.get(f"{name}_runtime_s", float("nan"))),
                                  largest_intermediate_log10=float("nan"), fallback_reason="",
                                  nbytes=tn.nbytes()))
        path = infos["tnco"]["path"]
        h_ref = cache.resid[site_from.key][0, position].detach()
        ids = C.ids_tensor(mb, v)
        for name in dir_names:
            unit_vec = bank[name].vec.to(h_ref.device)
            # fraction of the perturbation that lies inside the PCA basis (truncation loss)
            captured = float((P.T @ unit_vec.float().cpu()).norm() ** 2)
            lin_unit = linear_response_jvp(mb, ids, site_from, position, unit_vec, [L], window)[0]        # [T_w, d]
            y = contract(with_source(tn, position - w0, unit_vec), path)                                   # [T_w, r]
            fact_unit = (np.asarray(y, dtype=np.float64) @ tn.meta["basis"].T.astype(np.float64)).astype(np.float32)
            for eps in limits.tn_doses:
                delta = make_dose_delta(h_ref, unit_vec, float(eps)).to(h_ref.dtype)
                scale = float(delta.float().norm())
                with torch.no_grad(), Injector(mb, site_from, delta, positions=[position], mode="add"):
                    pert = run_with_cache(mb, ids, [out_site])
                measured = (pert.resid[out_site.key][0, window].float() - cache.resid[out_site.key][0, window].float()).cpu().numpy()
                df = compare_responses(measured, lin_unit * scale, fact_unit * scale, positions=w_positions)
                for k, val in dict(C.variant_row(v), direction=name, eps=float(eps), inject_pos=position,
                                   delta_norm=scale, source_captured=captured).items():
                    df[k] = val
                df["rel_pos"] = df["position"] - position
                cmp_rows.append(df)
                after = df[df["position"] >= position]
                w = after["linear_norm"].to_numpy(dtype=float)
                wsum = float(w.sum()) if w.size else 0.0
                summ_rows.append(dict(C.variant_row(v), direction=name, eps=float(eps), inject_pos=position, T_w=T_w,
                                      r=int(P.shape[1]), source_captured=captured, n_positions=int(len(after)),
                                      nonlinearity_wmean=float((after["nonlinearity"] * w).sum() / wsum) if wsum > 0 else float("nan"),
                                      factorization_error_wmean=float((after["factorization_error"] * w).sum() / wsum) if wsum > 0 else float("nan"),
                                      nonlinearity_last=float(df["nonlinearity"].iloc[-1]),
                                      factorization_error_last=float(df["factorization_error"].iloc[-1]),
                                      cos_measured_linear_last=float(df["cos_measured_linear"].iloc[-1]),
                                      cos_linear_factorized_last=float(df["cos_linear_factorized"].iloc[-1]),
                                      measured_norm_last=float(df["measured_norm"].iloc[-1]),
                                      linear_norm_last=float(df["linear_norm"].iloc[-1]),
                                      factorized_norm_last=float(df["factorized_norm"].iloc[-1]),
                                      path_backend=infos["tnco"].get("backend", ""),
                                      path_cost_log10=float(infos["tnco"].get("cost_log10_opt_einsum", float("nan")))))
    del caches
    tables = {
        "tn_compare": pd.concat(cmp_rows, ignore_index=True) if cmp_rows else pd.DataFrame(),
        "tn_summary": pd.DataFrame(summ_rows),
        "tn_paths": pd.DataFrame(path_rows),
        "tn_basis": basis_df,
    }
    return C.finish_stage(run_dir, "E", cfg, mb, tables, limits=limits, t0=t0, site_from=site_from, layers=list(layers),
                          rank=int(P.shape[1]), window=int(cfg.tn_window), directions=dir_names, n_subset=len(subset),
                          subset=[v.tag for v in subset])
