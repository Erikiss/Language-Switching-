"""Trace source in Stage L: shards, probe, residualised alignments, carriers (DESIGN §4.9).

This module connects the trace LM (:class:`lsne.chain.source.TraceLMSource`) to the
transplant machinery of :mod:`lsne.transplant` for
:mod:`lsne.experiments.stage_l_transplant` (``--transplant-source trace:<model_dir>``):

* **ITR shards** (:func:`resolve_shards`, :func:`shard_index`, :func:`fetch_rows`): the
  shards written by ``python -m lsne.chain.cli serialize``; explicit path, else the
  ``data`` entry of the model's ``manifest.json`` (the shards it was trained on).
* **Sharing direction** (:func:`fanout_probe`, :func:`probe_states`, :func:`load_direction`):
  the *length-matched* fan-out probe (:func:`lsne.chain.labels.probe_sets` with
  ``match="length"``: 1:1 within era on the visible ITR length, caliper ``length_ratio``),
  pooled states of the trained model and of its untrained twin
  (:func:`lsne.chain.source.untrained_twin`), the probe criterion of
  :func:`lsne.chain.source.criterion_table` (split-half AUC trained minus untrained with a
  paired bootstrap CI), and the difference-of-means direction at the frozen source layer;
  a ``sharing_directions.pt`` written by ``lsne.chain.cli probe`` replaces the in-stage
  direction when given or found in the model directory.  When the length-matched probe
  finds fewer than two pairs the matching is relaxed (era only, then ``k = 2``); the
  result is flagged (``feature_frozen = False``) and can never support a claim.
* **Residualised alignments** (:func:`residualised_alignment`): source and target pooled
  states of paired ``(shard row, its English text)`` items are residualised on the
  length covariates of :func:`lsne.chain.source.length_covariates` /
  :func:`~lsne.chain.source.covariate_matrix` (log ITR tokens, log1p call nodes, log1p
  value calls, log text tokens; coefficients from the training pairs only,
  :func:`lsne.chain.source.residualize`), the map ``W`` is fitted on the residuals and the
  intercept on the raw training means (``b = mean(Y) - mean(X) W``), so ``H_S W + b`` can
  be applied to raw per-token source states while ``r2_*`` measure only what the source
  predicts *beyond* length.
* **Carriers** (:func:`carrier_rows`): the source item paired with each stimulus variant
  of the transplant (a trace and a German/English stimulus have no natural pairing): the
  probe's matched negatives (fan-out 1), so the dose along the sharing direction moves a
  single-receiver transaction towards "sharing"; the same carrier for the K=0 and K=8/C1
  variant of one formulation.

The residualisation, untrained twin and probe criterion come from
:mod:`lsne.chain.source` (round-2 controls); minimal local fallbacks keep this module
working when an older ``lsne.chain.source`` lacks them (``HAVE_CHAIN_CONTROLS``).
"""
from __future__ import annotations

import copy
import glob
import json
import os
from types import SimpleNamespace
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from ..transplant import Alignment, fit_alignment
from . import labels as LB

DEFAULT_K = 5
MIN_PAIRS_DIRECTION = 2
DIRECTION_FILE = "sharing_directions.pt"

try:  # round-2 controls of lsne.chain.source
    from .source import (alignment_controls, alignment_split, covariate_matrix, criterion_table, init_seed_of,
                         length_covariates, residualize, untrained_twin)
    HAVE_CHAIN_CONTROLS = True
except ImportError:  # pragma: no cover - older lsne.chain.source: local fallbacks below
    HAVE_CHAIN_CONTROLS = False
    alignment_controls = criterion_table = None

    def alignment_split(n: int, holdout: float = 0.2, seed: int = 0) -> tuple:
        perm = np.random.default_rng(seed).permutation(int(n))
        n_hold = int(round(float(holdout) * n))
        if holdout > 0:
            n_hold = max(n_hold, 1)
        n_hold = min(n_hold, int(n) - 2)
        return perm[:n_hold], perm[n_hold:]

    def residualize(X, Z, train_idx=None) -> np.ndarray:
        X = torch.as_tensor(X).double().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, np.float64)
        Z1 = np.column_stack([np.ones(len(Z)), np.asarray(Z, np.float64)])
        tr = np.arange(len(X)) if train_idx is None else np.asarray(train_idx)
        beta, *_ = np.linalg.lstsq(Z1[tr], X[tr], rcond=None)
        return X - Z1 @ beta

    def length_covariates(items, texts=None, tokenizer=None, max_len: int = 512) -> pd.DataFrame:
        from .vocab import item_tokens
        rows = []
        for it in items:
            toks = item_tokens(it, None)
            sf = LB.surface_features(toks)
            rows.append(dict(n_tokens=min(len(toks), int(max_len)), n_tokens_raw=len(toks), n_nodes=sf["n_call"],
                             n_value_calls=sf["n_value_calls_nonzero"]))
        df = pd.DataFrame(rows, columns=["n_tokens", "n_tokens_raw", "n_nodes", "n_value_calls"])
        if texts is not None:
            df["n_text_tokens"] = [len(tokenizer(t)["input_ids"]) for t in texts]
        return df

    def covariate_matrix(cov: pd.DataFrame, text: bool = False) -> tuple:
        cols = [np.log(np.maximum(cov["n_tokens"].to_numpy(float), 1.0)), np.log1p(cov["n_nodes"].to_numpy(float)),
                np.log1p(cov["n_value_calls"].to_numpy(float))]
        names = ["log_n_tokens", "log1p_n_nodes", "log1p_n_value_calls"]
        if text:
            cols.append(np.log(np.maximum(cov["n_text_tokens"].to_numpy(float), 1.0)))
            names.append("log_n_text_tokens")
        return np.column_stack(cols), names

    def init_seed_of(model_dir: str, default: int = 0) -> int:
        try:
            with open(os.path.join(str(model_dir), "manifest.json"), encoding="utf-8") as fh:
                return int(json.load(fh).get("hparams", {}).get("seed", default))
        except (OSError, ValueError, TypeError, AttributeError):
            return int(default)

    def untrained_twin(src, seed: int = 0):
        from ..model import ModelBundle
        model = src.mb.model
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            twin = type(model)(copy.deepcopy(model.config))
        twin = twin.to(dtype=next(model.parameters()).dtype).eval().to(src.device)
        for p in twin.parameters():
            p.requires_grad_(False)
        out = copy.copy(src)
        out.mb = ModelBundle(model=twin, tokenizer=src.mb.tokenizer, spec=src.mb.spec, device=src.mb.device)
        out.name = f"{src.name}:untrained(seed={int(seed)})"
        return out


# ======================================================================================
# model directory, shards, rows
# ======================================================================================
def model_dir_of(spec: str) -> str:
    """``"trace:<dir>"`` (or a bare directory) -> the directory."""
    s = str(spec)
    return s[len("trace:"):] if s.startswith("trace:") else s


def read_model_manifest(model_dir: str) -> dict:
    try:
        with open(os.path.join(str(model_dir), "manifest.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _existing(path: Optional[str], model_dir: str) -> Optional[str]:
    if not path:
        return None
    cands = [path]
    if not os.path.isabs(path):
        d = os.path.abspath(str(model_dir))
        for _ in range(4):           # relative to the model directory and its parents
            cands.append(os.path.join(d, path))
            d = os.path.dirname(d)
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def resolve_shards(model_dir: str, explicit: Optional[str] = None) -> tuple[Optional[str], str]:
    """``(path, how)``: the ITR shards for stage L.  ``explicit`` (``--trace-shards`` /
    ``Limits.l_trace_shards``) wins; else the ``data`` entry of the model's ``manifest.json``
    (the shards the model was trained on; relative paths are tried against the working
    directory and the model directory's parents).  ``(None, reason)`` when nothing exists."""
    if explicit:
        p = _existing(str(explicit), model_dir)
        return (p, "explicit") if p else (None, f"explicit shards path not found: {explicit}")
    data = read_model_manifest(model_dir).get("data")
    p = _existing(data, model_dir) if data else None
    if p:
        return p, "model_manifest"
    return None, ("no shards: pass --trace-shards / Limits.l_trace_shards (the model manifest names "
                  f"{data!r})" if data else "no shards: pass --trace-shards / Limits.l_trace_shards")


def shard_index(path: str, limit: Optional[int] = None, max_len: int = 512) -> pd.DataFrame:
    """One row per transaction of the shards: every :func:`lsne.chain.labels.sharing_features`
    key, ``tx_hash``, ``label``, ``n_tokens`` (raw ITR length) and ``n_tokens_visible``
    (``min(n_tokens, max_len)``, what the model sees); tokens and texts stay on disk."""
    from .cli import iter_rows
    rows = []
    for r in iter_rows(path, limit=limit):
        f = dict(r.get("features") or {})
        f["tx_hash"] = r["tx_hash"]
        f["label"] = r.get("label")
        f["n_tokens"] = int(r.get("n_tokens", len(r.get("tokens", []))))
        f["n_tokens_visible"] = min(f["n_tokens"], int(max_len))
        f.setdefault("era", r.get("era"))
        f.setdefault("block_number", r.get("block_number"))
        rows.append(f)
    return pd.DataFrame(rows)


def fetch_rows(path: str, hashes: Sequence[str]) -> dict:
    """``{tx_hash: shard row}`` for ``hashes`` (one pass over the shards)."""
    from .cli import iter_rows
    want = set(hashes)
    return {r["tx_hash"]: r for r in iter_rows(path, hashes=want)} if want else {}


def sample_pair_hashes(index: pd.DataFrame, n: int, seed: int = 0, balance: bool = True) -> list:
    """Hashes of ``n`` alignment pairs: equal numbers per sharing label (``balance``; labels
    with fewer transactions contribute all of theirs) or a plain random sample (seeded)."""
    rng = np.random.default_rng(int(seed))
    if index.empty:
        return []
    if not balance or "label" not in index.columns:
        idx = rng.permutation(len(index))[: int(n)]
        return [str(index["tx_hash"].iloc[i]) for i in idx]
    by = {lab: g["tx_hash"].tolist() for lab, g in index.groupby(index["label"].astype(str), sort=True)}
    per = max(1, int(n) // max(1, len(by)))
    out = []
    for lab in sorted(by):
        hs = by[lab]
        out += [hs[i] for i in rng.permutation(len(hs))[:per]]
    if len(out) < int(n):                     # fill up from the remaining transactions
        rest = [h for h in index["tx_hash"].tolist() if h not in set(out)]
        out += [rest[i] for i in rng.permutation(len(rest))[: int(n) - len(out)]]
    return out[: int(n)]


# ======================================================================================
# the sharing probe
# ======================================================================================
def _probe(index: pd.DataFrame, k: int, n: int, seed: int, match: str, length_ratio: float, max_len: int):
    recs = [SimpleNamespace(tx_hash=h) for h in index["tx_hash"]]
    if match == "length":
        try:
            return LB.probe_sets(recs, index, "fanout", int(n), seed=int(seed), k=int(k), match="length",
                                 length_ratio=float(length_ratio), length_col="n_tokens", length_cap=int(max_len))
        except TypeError:   # an lsne.chain.labels without length matching
            return [], [], pd.DataFrame()
    return LB.probe_sets(recs, index, "fanout", int(n), seed=int(seed), k=int(k), gas_bins=1)


def fanout_probe(index: pd.DataFrame, n: int = 500, seed: int = 0, k: int = DEFAULT_K, length_ratio: float = 1.25,
                 max_len: int = 512, min_pairs: int = MIN_PAIRS_DIRECTION) -> tuple[list, list, pd.DataFrame, dict]:
    """Length-matched fan-out probe sets on a :func:`shard_index` frame.

    Positives ``fan_out_eth >= k``, negatives ``fan_out_eth == 1``, 1:1 within era on the
    visible ITR length (:func:`lsne.chain.labels.probe_sets`, ``match="length"``,
    ``length_cap=max_len``).  With fewer than ``min_pairs`` pairs the attempts continue
    with era-only matching, then with ``k = 2`` (length, then era only); the first attempt
    with enough pairs is used.  Returns ``(pos_hashes, neg_hashes, match_table, info)``,
    ``info``: ``k_requested, k_used, match, matching, n_pairs, attempts, feature_frozen``
    (True only for the pre-specified length-matched ``k`` probe)."""
    attempts = [(int(k), "length"), (int(k), "era_only")]
    if int(k) > 2:
        attempts += [(2, "length"), (2, "era_only")]
    tried = []
    pos, neg, mt, used = [], [], pd.DataFrame(), None
    for kk, match in attempts:
        p, q, m = _probe(index, kk, n, seed, match, length_ratio, max_len)
        tried.append(dict(k=kk, match=match, n_pairs=len(p)))
        if len(p) >= int(min_pairs):
            pos, neg, mt, used = p, q, m, (kk, match)
            break
    info = dict(k_requested=int(k), k_used=(used[0] if used else None), match=(used[1] if used else None),
                matching=(str(mt.attrs.get("matching", "")) if used is not None else None), n_pairs=len(pos),
                attempts=tried, length_ratio=float(length_ratio), max_len=int(max_len),
                feature_frozen=bool(used is not None and used == (int(k), "length")))
    for key in ("length_smd", "n_tokens_smd", "gas_smd", "log_gas_smd", "pool_pos", "pool_neg", "n_pos_unmatched"):
        if used is not None and key in mt.attrs:
            v = mt.attrs[key]
            info[key] = (None if v is None else (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v))
    return [r.tx_hash for r in pos], [r.tx_hash for r in neg], mt, info


def probe_states(src, twin, rows: Sequence, layers: Sequence[int], readout: str = "mean") -> tuple[dict, dict]:
    """Pooled states ``{layer: [N, d]}`` of ``rows`` under the trained source and its twin."""
    from .source import pooled_states
    layers = [int(l) for l in layers]
    return pooled_states(src, list(rows), layers, readout), pooled_states(twin, list(rows), layers, readout)


def _local_criterion(S: dict, S0: dict, n_pos: int, layers: Sequence[int], seed: int = 0, n_boot: int = 1000,
                     **_kw) -> pd.DataFrame:
    """Fallback probe criterion (split-half AUC trained - untrained, paired bootstrap over pairs)."""
    from .source import holdout_auc
    rows = []
    for l in layers:
        Sp, Sq, Up, Uq = S[l][:n_pos], S[l][n_pos:], S0[l][:n_pos], S0[l][n_pos:]
        n = min(len(Sp), len(Sq))
        r = dict(layer=int(l), n_pairs=int(n), auc_trained=np.nan, auc_untrained=np.nan, delta=np.nan,
                 delta_ci_lo=np.nan, delta_ci_hi=np.nan, counts=False)
        if n >= 4:
            a1, a0 = holdout_auc(Sp, Sq, seed=seed, denoise_pcs=0), holdout_auc(Up, Uq, seed=seed, denoise_pcs=0)
            rng = np.random.default_rng(seed + 1)
            d = []
            for _ in range(int(n_boot)):
                ix = torch.from_numpy(rng.integers(0, n, n)).long()
                d.append(holdout_auc(Sp[ix], Sq[ix], seed=seed, denoise_pcs=0)
                         - holdout_auc(Up[ix], Uq[ix], seed=seed, denoise_pcs=0))
            lo, hi = np.nanquantile(d, [0.025, 0.975]) if d else (np.nan, np.nan)
            r.update(auc_trained=a1, auc_untrained=a0, delta=a1 - a0, delta_ci_lo=float(lo), delta_ci_hi=float(hi),
                     counts=bool(np.isfinite(lo) and lo > 0))
        rows.append(r)
    return pd.DataFrame(rows)


def probe_table(S: dict, S0: dict, n_pos: int, layers: Sequence[int], seed: int = 0, n_boot: int = 1000,
                n_perm: int = 200, surface: Optional[dict] = None) -> pd.DataFrame:
    """The probe criterion per residual snapshot (:func:`lsne.chain.source.criterion_table`:
    split-half AUC of the trained model minus its untrained twin, paired bootstrap CI over
    matched pairs, label-permutation and random-direction nulls; ``counts`` = CI above 0
    and the trained AUC above the permutation null's 95 % quantile)."""
    layers = [int(l) for l in layers]
    if criterion_table is not None:
        return criterion_table(S, S0, int(n_pos), layers=layers, surface=surface, seed=int(seed), n_boot=int(n_boot),
                               n_perm=int(n_perm), n_random=int(n_perm))
    return _local_criterion(S, S0, int(n_pos), layers, seed=seed, n_boot=n_boot)


# ======================================================================================
# a saved direction file (python -m lsne.chain.cli probe)
# ======================================================================================
def find_direction_file(model_dir: str, explicit: Optional[str] = None) -> Optional[str]:
    """``explicit`` (a ``sharing_directions.pt`` or the probe output directory holding one),
    else ``<model_dir>/sharing_directions.pt`` when present, else None."""
    for c in ([explicit] if explicit else []) + [os.path.join(str(model_dir), DIRECTION_FILE)]:
        if not c:
            continue
        p = os.path.join(c, DIRECTION_FILE) if os.path.isdir(c) else c
        if os.path.isfile(p):
            return p
    return None


def load_direction(path: str, layer: int, d_model: int, model_dir: Optional[str] = None) -> tuple:
    """``(vec [d] or None, info)`` of the direction at residual snapshot ``layer`` in a file
    written by ``lsne.chain.cli probe``.  Refused (``None`` + reason) when the width differs,
    the layer is missing, or the file names a different model directory."""
    info = dict(path=str(path))
    try:
        saved = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001 - reported, the in-stage direction is used instead
        info["refused"] = f"unreadable: {e!r}"
        return None, info
    info.update(kind=saved.get("kind"), readout=saved.get("readout"), n_pairs=saved.get("n_pairs"),
                model=saved.get("model"))
    if saved.get("d_model") is not None and int(saved["d_model"]) != int(d_model):
        info["refused"] = f"d_model {saved['d_model']} != {d_model}"
        return None, info
    if model_dir is not None and saved.get("model"):
        same = os.path.realpath(str(saved["model"])) == os.path.realpath(str(model_dir))
        if not same and os.path.basename(os.path.normpath(str(saved["model"]))) != os.path.basename(
                os.path.normpath(str(model_dir))):
            info["refused"] = f"file was written for model {saved['model']!r}"
            return None, info
    layers = saved.get("layers", {})
    ent = layers.get(int(layer), layers.get(str(int(layer))))
    if ent is None or ent.get("vec") is None:
        info["refused"] = f"no direction for layer {layer} (has {sorted(layers)})"
        return None, info
    vec = torch.as_tensor(ent["vec"]).detach().float().cpu().reshape(-1)
    if vec.numel() != int(d_model) or not torch.isfinite(vec).all() or float(vec.norm()) == 0.0:
        info["refused"] = "vector has the wrong size or is degenerate"
        return None, info
    return vec / vec.norm(), info


# ======================================================================================
# residualised alignment
# ======================================================================================
def residualised_alignment(X, Y, Z, layer_src: int, layer_tgt: int, method: str = "procrustes", holdout: float = 0.2,
                           seed: int = 0, covariate_names: Optional[Sequence[str]] = None, **fit_kw) -> Alignment:
    """Alignment fitted on length-residualised pooled states (see the module docstring).

    ``X [N, d_S]`` / ``Y [N, d_T]`` raw pooled states, ``Z [N, k]`` covariates.  Both sides are
    residualised on ``[1, Z]`` with coefficients from the training split of
    :func:`lsne.transplant.fit_alignment` (same seed), ``W`` is fitted on the residuals
    (``r2_train/holdout/permuted`` are therefore *beyond length*) and
    ``b = mean(Y_train) - mean(X_train) W`` (raw means).  ``meta`` adds ``residualised_on``
    and ``r2_holdout_raw`` (the same method on the raw states)."""
    Xn = torch.as_tensor(X).double().numpy() if isinstance(X, torch.Tensor) else np.asarray(X, np.float64)
    Yn = torch.as_tensor(Y).double().numpy() if isinstance(Y, torch.Tensor) else np.asarray(Y, np.float64)
    hold, train = alignment_split(len(Xn), holdout, seed)
    Xr, Yr = residualize(Xn, Z, train), residualize(Yn, Z, train)
    a = fit_alignment(Xr, Yr, method=method, holdout=holdout, seed=seed, layer_src=layer_src, layer_tgt=layer_tgt, **fit_kw)
    raw = fit_alignment(Xn, Yn, method=method, holdout=holdout, seed=seed, layer_src=layer_src, layer_tgt=layer_tgt,
                        **fit_kw)
    W = a.W.double().numpy()
    b = Yn[train].mean(0) - Xn[train].mean(0) @ W
    meta = dict(a.meta)
    meta.update(residualised_on=list(covariate_names or []), r2_holdout_raw=float(raw.r2_holdout),
                r2_permuted_raw=float(raw.r2_permuted), intercept="raw_training_means")
    return Alignment(W=a.W.clone(), b=torch.from_numpy(b).float(), method=f"{a.method}+len_resid", rank=a.rank,
                     layer_src=int(layer_src), layer_tgt=int(layer_tgt), r2_train=a.r2_train, r2_holdout=a.r2_holdout,
                     r2_permuted=a.r2_permuted, n_pairs=a.n_pairs, meta=meta)


# ======================================================================================
# carriers
# ======================================================================================
def carrier_rows(preferred: Sequence[dict], fallback: Sequence[dict], n: int, seed: int = 0) -> list:
    """``n`` source items for the transplant prompts: the ``preferred`` rows (the probe's
    matched negatives) in a seeded order, cycled; ``fallback`` rows when there are none."""
    pool = list(preferred) or list(fallback)
    if not pool:
        return []
    order = np.random.default_rng(int(seed)).permutation(len(pool))
    return [pool[order[i % len(pool)]] for i in range(int(n))]


__all__ = ["HAVE_CHAIN_CONTROLS", "DEFAULT_K", "DIRECTION_FILE", "model_dir_of", "read_model_manifest", "resolve_shards",
           "shard_index", "fetch_rows", "sample_pair_hashes", "fanout_probe", "probe_states", "probe_table",
           "find_direction_file", "load_direction", "residualised_alignment", "carrier_rows", "alignment_controls",
           "alignment_split", "covariate_matrix", "length_covariates", "residualize", "untrained_twin", "init_seed_of"]
