"""Trajectories of the local geometry (DESIGN.md section 4.5): depth (T1), position (T2),
dose/steering (T3) and the coupling with the non-equilibrium diagnostics (T5).

Builds on :mod:`lsne.geometry`. Residual snapshots are indexed by *layer* ``l`` in the
lattice convention of DESIGN section 4.2: ``"L{n}.block_input" -> n`` and
``"L{n}.block_output" -> n + 1`` (so a model with ``L`` blocks has ``L + 1`` snapshots
``0 .. L``; the other hook sites map to ``n`` as well).
"""
from __future__ import annotations

import re
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from .config import HookSite
from .geometry import (align_tail_states, layer_covariance, log_volume, mahalanobis,
                       participation_ratio, per_position_log_volume, variant_ids)
from .model import Injector, ModelBundle, all_block_inputs, make_dose_delta, run_with_cache

__all__ = ["layer_index", "depth_trajectory", "position_trajectory",
           "steering_tracks_switching", "couple_with_dynamics", "breathing_index"]

_SITE_RE = re.compile(r"^L(\d+)\.(block_input|block_output|mlp_output|attn_output)$")
_COND_COLS = ("family_id", "K", "C", "readout", "k")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def layer_index(site_key: str) -> int:
    """Numeric lattice layer of a site key (see module docstring)."""
    m = _SITE_RE.match(str(site_key))
    if not m:
        raise ValueError(f"not a site key: {site_key!r}")
    n, kind = int(m.group(1)), m.group(2)
    return n + 1 if kind == "block_output" else n


def breathing_index(delta: Sequence[float]) -> int:
    """Number of sign changes of a series of increments, ignoring zeros and NaNs."""
    d = np.asarray(list(delta), dtype=np.float64)
    s = np.sign(d[np.isfinite(d) & (d != 0)])
    return int(np.sum(s[1:] != s[:-1])) if s.size > 1 else 0


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _direction_vec(direction, d: int) -> torch.Tensor:
    """Accept a ``Direction`` (``.vec``), a tensor or an array; return a unit float32 CPU vector."""
    v = getattr(direction, "vec", direction)
    if isinstance(v, np.ndarray):
        v = torch.from_numpy(v)
    v = torch.as_tensor(v).detach().float().cpu().reshape(-1)
    if v.numel() != d:
        raise ValueError(f"direction has {v.numel()} components, model has d={d}")
    n = float(v.norm())
    if n <= 0 or not np.isfinite(n):
        raise ValueError("direction must be a non-zero finite vector")
    return v / n


def _annot(v) -> dict:
    a = v.get("annot") if isinstance(v, Mapping) else getattr(v, "annot", None)
    return dict(a or {})


def _opt_int(x) -> Optional[int]:
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(xf):
        return None
    return int(xf)


# --------------------------------------------------------------------------------------
# T1: depth
# --------------------------------------------------------------------------------------
def depth_trajectory(geom_table: pd.DataFrame) -> pd.DataFrame:
    """Depth trajectory of the local geometry per condition.

    ``geom_table`` is the output of :func:`lsne.geometry.geometry_table` (columns ``site``,
    ``readout``, ``k``, ``family_id``, ``K``, ``C``, ``log_volume`` and optionally
    ``participation_ratio``, ``disp_euclid``, ``disp_mahal``, ``n``). Rows are grouped by
    ``(family_id, K, C, readout, k)`` (whichever of these exist) and ordered by the
    numeric layer of ``site`` (:func:`layer_index`; a ``layer`` column is used if present).

    Long format, one row per layer, with the per-layer curves (``log_volume``,
    ``participation_ratio``, ``disp_euclid``, ``disp_mahal``), their increments
    ``delta_log_volume`` (``log_volume(l) - log_volume(l-1)``, NaN at the first layer),
    ``delta_participation_ratio``, ``delta_disp_mahal`` and the per-condition summaries
    repeated on every row: ``breathing_index`` (sign changes of ``delta_log_volume``,
    zeros ignored), ``breathing_amplitude`` (mean ``|delta_log_volume|``),
    ``layer_of_max_contraction`` (layer at which ``delta_log_volume`` is most negative; NaN
    when no increment is negative: nothing contracts), ``layer_of_max_expansion`` (most
    positive increment; NaN when none is positive), ``net_change`` (last minus first log-volume),
    ``total_variation`` and ``n_layers``.
    """
    df = geom_table.copy()
    if "layer" not in df.columns:
        df["layer"] = df["site"].map(layer_index)
    cond_cols = [c for c in _COND_COLS if c in df.columns]
    curve_cols = [c for c in ("log_volume", "participation_ratio", "disp_euclid", "disp_mahal")
                  if c in df.columns]
    keep = cond_cols + ["site", "layer", "n"] if "n" in df.columns else cond_cols + ["site", "layer"]
    keep = [c for c in keep if c in df.columns]
    frames = []
    grouped = df.groupby(cond_cols, dropna=False, sort=True) if cond_cols else [((), df)]
    for _, g in grouped:
        g = g.sort_values("layer").reset_index(drop=True)
        g = g.drop_duplicates("layer")
        out = g[keep + curve_cols].copy()
        lv = out["log_volume"].to_numpy(dtype=np.float64)
        delta = np.full_like(lv, np.nan)
        if lv.size > 1:
            delta[1:] = np.diff(lv)
        out["delta_log_volume"] = delta
        for c, name in (("participation_ratio", "delta_participation_ratio"),
                        ("disp_mahal", "delta_disp_mahal")):
            if c in out.columns:
                vals = out[c].to_numpy(dtype=np.float64)
                dd = np.full_like(vals, np.nan)
                if vals.size > 1:
                    dd[1:] = np.diff(vals)
                out[name] = dd
        finite = np.isfinite(delta)
        layers = out["layer"].to_numpy()
        out["breathing_index"] = breathing_index(delta)
        out["breathing_amplitude"] = float(np.mean(np.abs(delta[finite]))) if finite.any() else np.nan
        out["layer_of_max_contraction"] = (layers[np.nanargmin(delta)] if finite.any() and np.nanmin(delta) < 0
                                           else np.nan)
        out["layer_of_max_expansion"] = (layers[np.nanargmax(delta)] if finite.any() and np.nanmax(delta) > 0
                                         else np.nan)
        lv_f = lv[np.isfinite(lv)]
        out["net_change"] = float(lv_f[-1] - lv_f[0]) if lv_f.size > 1 else np.nan
        out["total_variation"] = float(np.sum(np.abs(delta[finite]))) if finite.any() else np.nan
        out["n_layers"] = int(len(out))
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=keep + curve_cols + ["delta_log_volume", "breathing_index",
                                                         "breathing_amplitude",
                                                         "layer_of_max_contraction"])
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# T2: position
# --------------------------------------------------------------------------------------
def position_trajectory(mb: ModelBundle, group_variants: Sequence, site: HookSite,
                        align: str = "tail_from_end", n_prefix_bins: int = 8) -> pd.DataFrame:
    """Per-position log-volume of the realization cloud of one condition at ``site``.

    Three regions (column ``region``):

    * ``"tail"``: the shared tail (positions ``>= annot['tail_start']``), aligned from the
      END over the common tail length (``align="tail_from_end"``) or from ``tail_start``
      (``align="tail_from_start"``); ``index`` is the aligned position, ``pos_from_end``
      the offset from the end (``-1`` = last position).
    * ``"prefix_bin"``: the prefix (positions ``< tail_start``) in ``n_prefix_bins``
      relative bins; per variant the states of a bin are mean-pooled first, then the
      cloud across variants is measured; ``rel_pos`` is the bin centre in ``[0, 1]``.
    * ``"from_switch"``: positions aligned on the last language switch
      (``annot['last_switch_pos']``, present for every variant), ``index`` = offset
      ``>= 0`` from the switch, over the common length.

    ``rel_to_last_switch`` is the mean (over variants) of the absolute position minus the
    variant's ``last_switch_pos`` (NaN if unavailable). ``delta_log_volume`` is the
    increment along the region (the paper's per-position increments).
    Columns: site, region, index, rel_pos, pos_from_end, rel_to_last_switch, n,
    log_volume, participation_ratio, delta_log_volume.
    """
    if align not in ("tail_from_end", "tail_from_start"):
        raise ValueError(f"align must be 'tail_from_end' or 'tail_from_start', got {align!r}")
    H, tails, switches = [], [], []
    for v in group_variants:
        ids = variant_ids(v)
        T = int(ids.numel())
        if T == 0:
            continue
        co = run_with_cache(mb, ids[None, :], [site])
        H.append(co.resid[site.key][0].detach().float().cpu().numpy().astype(np.float64))
        a = _annot(v)
        ts = _opt_int(a.get("tail_start", T))
        tails.append(T if ts is None else max(0, min(ts, T)))
        switches.append(_opt_int(a.get("last_switch_pos")))
    cols = ["site", "region", "index", "rel_pos", "pos_from_end", "rel_to_last_switch", "n",
            "log_volume", "participation_ratio", "delta_log_volume"]
    n = len(H)
    if n == 0:
        return pd.DataFrame(columns=cols)
    Ts = [h.shape[0] for h in H]
    have_switch = all(s is not None for s in switches)
    rows = []

    def _region(name, clouds, indices, rel_pos, pos_from_end, rel_switch, ns):
        lvs = [log_volume(G) if G is not None else np.nan for G in clouds]
        prs = [participation_ratio(G) if G is not None else np.nan for G in clouds]
        for i in range(len(clouds)):
            rows.append(dict(site=site.key, region=name, index=indices[i], rel_pos=rel_pos[i],
                             pos_from_end=pos_from_end[i], rel_to_last_switch=rel_switch[i],
                             n=ns[i], log_volume=lvs[i], participation_ratio=prs[i],
                             delta_log_volume=(lvs[i] - lvs[i - 1]) if i > 0 else np.nan))

    # --- tail ---------------------------------------------------------------------------
    m = min(T - ts for T, ts in zip(Ts, tails))
    if m > 0:
        if align == "tail_from_end":
            R = align_tail_states([(h, ts) for h, ts in zip(H, tails)])
            abs_pos = [[T - m + j for j in range(m)] for T in Ts]
        else:
            R = np.stack([h[ts:ts + m] for h, ts in zip(H, tails)], 0)
            abs_pos = [[ts + j for j in range(m)] for ts in tails]
        lv = per_position_log_volume(R)
        prs = [participation_ratio(R[:, j, :]) for j in range(m)]
        for j in range(m):
            rs = (float(np.mean([abs_pos[i][j] - switches[i] for i in range(n)]))
                  if have_switch else np.nan)
            pfe = [abs_pos[i][j] - Ts[i] for i in range(n)]
            rows.append(dict(site=site.key, region="tail", index=j, rel_pos=np.nan,
                             pos_from_end=float(np.mean(pfe)), rel_to_last_switch=rs, n=n,
                             log_volume=float(lv[j]), participation_ratio=prs[j],
                             delta_log_volume=float(lv[j] - lv[j - 1]) if j > 0 else np.nan))
    # --- prefix bins --------------------------------------------------------------------
    nb = int(n_prefix_bins)
    if nb > 0:
        clouds, rel_sw, ns = [], [], []
        for b in range(nb):
            vecs, offs = [], []
            for i in range(n):
                L = tails[i]
                if L <= 0:
                    continue
                pos = np.arange(L)
                bins = np.minimum((pos * nb) // L, nb - 1)
                sel = pos[bins == b]
                if sel.size == 0:
                    continue
                vecs.append(H[i][sel].mean(0))
                if have_switch:
                    offs.append(float(sel.mean()) - switches[i])
            clouds.append(np.stack(vecs, 0) if len(vecs) >= 2 else None)
            ns.append(len(vecs))
            rel_sw.append(float(np.mean(offs)) if (have_switch and offs) else np.nan)
        _region("prefix_bin", clouds, list(range(nb)), [(b + 0.5) / nb for b in range(nb)],
                [np.nan] * nb, rel_sw, ns)
    # --- aligned on the last switch -----------------------------------------------------
    if have_switch:
        M = min(T - s for T, s in zip(Ts, switches))
        if M > 0 and all(0 <= s < T for s, T in zip(switches, Ts)):
            R = np.stack([h[s:s + M] for h, s in zip(H, switches)], 0)
            lv = per_position_log_volume(R)
            clouds = [R[:, j, :] for j in range(M)]
            _region("from_switch", clouds, list(range(M)), [np.nan] * M,
                    [float(np.mean([switches[i] + j - Ts[i] for i in range(n)])) for j in range(M)],
                    [float(j) for j in range(M)], [n] * M)
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------------------
# T3: dose / steering
# --------------------------------------------------------------------------------------
def steering_tracks_switching(mb: ModelBundle, variants_K0: Sequence, variants_K: Sequence,
                              direction, site: HookSite, eps_list: Sequence[float],
                              n_random: int = 32, seed: int = 0) -> pd.DataFrame:
    """Does continuous steering reproduce the geometry of discrete switching (T3)?

    At every residual snapshot (:func:`lsne.model.all_block_inputs`) at or after the
    injection site, the discrete switching displacement is
    ``v_K = mean_last(K) - mean_last(K0)`` over last-position states. For every ``eps`` in
    ``eps_list`` each K0 variant is re-run with :class:`Injector` adding
    ``make_dose_delta(h_ref_last, direction, eps)`` at the last position of ``site``
    (``h_ref_last`` = the variant's own unsteered residual there); the steering
    displacement is ``v_steer = mean(h_steered - h)`` over the K0 variants. Reported per
    ``(snapshot, eps)``: ``cos_steer_vs_K``, ``norm_steer``, ``norm_K``, the Mahalanobis
    displacement of the mean steered state from the K0 cloud (``disp_mahal_steer``) and of
    the K cloud (``disp_mahal_K``), both with the covariance of the pooled K0 and K states
    at that snapshot (``r = min(32, N-1, d)``), plus the same cosine for ``n_random``
    isotropic random unit directions (``cos_rand_mean``, ``cos_rand_sd``) and the static
    cosine between the direction itself and ``v_K`` (``cos_dir_vs_K``). Cosines of a
    zero displacement (e.g. ``eps = 0``) are NaN.

    ``direction`` may be a :class:`lsne.directions.Direction` (``.vec``) or a vector.
    Columns: snapshot, layer, eps, cos_steer_vs_K, cos_rand_mean, cos_rand_sd, norm_steer,
    norm_K, disp_mahal_steer, disp_mahal_K, cos_dir_vs_K, n_K0, n_K, n_random.
    """
    d = mb.d_model
    dvec = _direction_vec(direction, d)
    snaps = all_block_inputs(mb)
    snap_keys = [s.key for s in snaps]
    rec_sites = list(snaps) + ([site] if site.key not in snap_keys else [])
    ids0 = [variant_ids(v) for v in variants_K0]
    idsK = [variant_ids(v) for v in variants_K]
    if not ids0 or not idsK:
        raise ValueError("need at least one K0 and one K variant")

    def _last_states(ids_list, sites):
        out = {s.key: [] for s in sites}
        for ids in ids_list:
            co = run_with_cache(mb, ids[None, :], sites)
            for s in sites:
                out[s.key].append(co.resid[s.key][0, -1].detach().float().cpu().numpy().astype(np.float64))
        return {k: np.stack(v, 0) for k, v in out.items()}

    H0 = _last_states(ids0, rec_sites)
    HK = _last_states(idsK, snaps)
    h_ref = [torch.from_numpy(H0[site.key][j]).float() for j in range(len(ids0))]

    inject_idx = site.layer if site.site == "block_input" else site.layer + 1
    active = [(i, key) for i, key in enumerate(snap_keys) if i >= inject_idx]
    stats = {}
    for i, key in active:
        X = np.concatenate([H0[key], HK[key]], 0)
        U_r, lam_r = layer_covariance(X, 32)
        m0, mK = H0[key].mean(0), HK[key].mean(0)
        stats[key] = dict(U=U_r, lam=lam_r, m0=m0, vK=mK - m0,
                          dispK=mahalanobis(mK, m0, U_r, lam_r))

    gen = torch.Generator().manual_seed(int(seed))
    R = torch.randn(int(n_random), d, generator=gen)
    R = R / R.norm(dim=1, keepdim=True).clamp_min(1e-30)
    dirs = [("steer", dvec)] + [(f"rand{i}", R[i]) for i in range(int(n_random))]

    def _steered_means(vec: torch.Tensor, eps: float) -> dict:
        """Per active snapshot: (v_steer, mean steered state)."""
        diffs = {key: [] for _, key in active}
        for j, ids in enumerate(ids0):
            delta = make_dose_delta(h_ref[j], vec, float(eps))
            T = int(ids.numel())
            with Injector(mb, site, delta, positions=[T - 1]):
                co = run_with_cache(mb, ids[None, :], snaps)
            for _, key in active:
                hs = co.resid[key][0, -1].detach().float().cpu().numpy().astype(np.float64)
                diffs[key].append(hs - H0[key][j])
        out = {}
        for _, key in active:
            D = np.stack(diffs[key], 0)
            v = D.mean(0)
            out[key] = (v, H0[key].mean(0) + v)
        return out

    rows = []
    for eps in eps_list:
        res = {name: _steered_means(vec, eps) for name, vec in dirs}
        for i, key in active:
            st = stats[key]
            v_steer, mean_steered = res["steer"][key]
            rand_cos = np.array([_cosine(res[f"rand{q}"][key][0], st["vK"]) for q in range(int(n_random))],
                                dtype=np.float64)
            rc = rand_cos[np.isfinite(rand_cos)]
            rows.append(dict(
                snapshot=key, layer=i, eps=float(eps),
                cos_steer_vs_K=_cosine(v_steer, st["vK"]),
                cos_rand_mean=float(rc.mean()) if rc.size else np.nan,
                cos_rand_sd=float(rc.std(ddof=1)) if rc.size > 1 else np.nan,
                norm_steer=float(np.linalg.norm(v_steer)),
                norm_K=float(np.linalg.norm(st["vK"])),
                disp_mahal_steer=mahalanobis(mean_steered, st["m0"], st["U"], st["lam"]),
                disp_mahal_K=st["dispK"],
                cos_dir_vs_K=_cosine(dvec.numpy().astype(np.float64), st["vK"]),
                n_K0=len(ids0), n_K=len(idsK), n_random=int(n_random)))
    cols = ["snapshot", "layer", "eps", "cos_steer_vs_K", "cos_rand_mean", "cos_rand_sd",
            "norm_steer", "norm_K", "disp_mahal_steer", "disp_mahal_K", "cos_dir_vs_K",
            "n_K0", "n_K", "n_random"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------------------
# T5: coupling with the dynamics diagnostics
# --------------------------------------------------------------------------------------
_LAMBDA_COLS = ("lambda", "lam", "growth_lambda", "lambda_l", "slope")
_KL_COLS = ("kl", "KL", "lens_kl", "kl_l", "nonmonotonicity")


def _value_col(df: pd.DataFrame, candidates: Sequence[str], exclude: Sequence[str]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c]):
            return c
    raise ValueError(f"no value column found among {list(df.columns)}")


def _norm_keys(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c == "layer":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        elif c == "K":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        else:
            df[c] = df[c].astype(str)
    return df


def couple_with_dynamics(depth_traj: pd.DataFrame, damage_lambda_by_layer: Optional[pd.DataFrame],
                         lens_kl_by_layer: Optional[pd.DataFrame],
                         measure: str = "delta_log_volume") -> pd.DataFrame:
    """Spearman correlation across layers between the log-volume increments of the depth
    trajectory and the damage growth rate ``lambda_l`` (D1) / the logit-lens KL ``KL_l``
    (D2), per condition (T5).

    ``damage_lambda_by_layer`` needs a ``layer`` column and a value column (``lambda``,
    ``lam``, ... or the first numeric non-key column); ``lens_kl_by_layer`` likewise
    (``kl``, ``KL``, ...). Each partner table is joined on ``(family_id, K, C, layer)`` when
    those columns exist on both sides, else on ``layer`` only (duplicates per key are
    averaged first). Either table may be ``None``.

    Columns: the condition columns present in ``depth_traj`` (``family_id, K, C, readout,
    k``), ``partner`` (``"lambda"`` or ``"kl"``), ``measure``, ``n`` (layers), ``rho``,
    ``pvalue``. ``rho`` is NaN with fewer than three finite layer pairs.
    """
    if measure not in depth_traj.columns:
        raise ValueError(f"{measure!r} not in depth_traj")
    if "layer" not in depth_traj.columns:
        depth_traj = depth_traj.copy()
        depth_traj["layer"] = depth_traj["site"].map(layer_index)
    cond_cols = [c for c in _COND_COLS if c in depth_traj.columns]
    partners = []
    if damage_lambda_by_layer is not None:
        partners.append(("lambda", damage_lambda_by_layer, _LAMBDA_COLS))
    if lens_kl_by_layer is not None:
        partners.append(("kl", lens_kl_by_layer, _KL_COLS))
    out_cols = cond_cols + ["partner", "measure", "n", "rho", "pvalue"]
    rows = []
    for name, pdf, cands in partners:
        if "layer" not in pdf.columns:
            raise ValueError(f"{name} table needs a 'layer' column")
        join = ["layer"] + [c for c in ("family_id", "K", "C")
                            if c in pdf.columns and c in depth_traj.columns]
        vcol = _value_col(pdf, cands, exclude=join + ["site", "readout", "k"])
        p = _norm_keys(pdf[join + [vcol]], join)
        p = p.groupby(join, dropna=False, as_index=False)[vcol].mean().rename(columns={vcol: "_partner"})
        left = _norm_keys(depth_traj, join)
        merged = left.merge(p, on=join, how="inner")
        groups = merged.groupby(cond_cols, dropna=False, sort=True) if cond_cols else [((), merged)]
        for cond, g in groups:
            cond = cond if isinstance(cond, tuple) else (cond,)
            x = g[measure].to_numpy(dtype=np.float64)
            y = g["_partner"].to_numpy(dtype=np.float64)
            ok = np.isfinite(x) & np.isfinite(y)
            nn = int(ok.sum())
            rho, pv = np.nan, np.nan
            if nn >= 3:
                r = spearmanr(x[ok], y[ok])
                rho, pv = float(r[0]), float(r[1])
            row = dict(zip(cond_cols, cond))
            row.update(partner=name, measure=measure, n=nn, rho=rho, pvalue=pv)
            rows.append(row)
    return pd.DataFrame(rows, columns=out_cols)
