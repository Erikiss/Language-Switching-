"""The trained trace LM as a :class:`lsne.transplant.SourceModel` (docs/TRACE_SOURCE_SPEC.md §7).

``TraceLMSource(model_dir)`` loads the model directory written by
:func:`lsne.chain.train.train` through :func:`lsne.model.load_model` (``ModelSpec(name=model_dir,
revision=None)``; with transformers 5.20 a local directory loads with any revision, a
direct loader is kept only as fallback), so the residual hooks are the ones the target
side uses.  Items are :class:`~lsne.chain.itr.TxRecord` objects (serialised with the
model's vocabulary), raw or mapped ITR token lists, ITR shard rows (dicts / objects with
``tokens``) or token ids; they are truncated to ``max_len`` (512) tokens.

How the direction enters Stage L (DESIGN §4.9): :func:`paired_items` gives
``(record, render_text(record))`` pairs, i.e. the same transaction as ITR for the source
and as English text for Pythia; :func:`lsne.transplant.layer_matching` /
:func:`~lsne.transplant.layer_alignments` fit ``H_T ~ H_S W + b`` on them, and the
:func:`sharing_direction` of matched probe sets (:func:`lsne.chain.labels.probe_sets`) is
transferred with :func:`~lsne.transplant.transfer_direction` or used as ``v_src`` in
:func:`~lsne.transplant.transplant_experiment`.

**Controls (round 2, docs/TRACE_SOURCE.md "Kontrollierte Sonden").**  The first run showed
that an untrained model separates the probe sets as well as the trained one, and that two
length features explain most of the alignment R².  Hence:

* :func:`untrained_twin` -- the same architecture, vocabulary, tokenizer and
  serialisation with the weights re-initialised exactly like
  :func:`lsne.chain.train.make_model` (``seed`` = the training seed gives the trained
  model's own initial weights);
* :func:`probe_criterion` / :func:`criterion_table` -- the probe criterion: split-half AUC
  of the trained model minus the same for the untrained twin on the *same* split, with a
  paired bootstrap CI over matched pairs (both halves resampled separately, the direction
  refitted on every replicate), a label-permutation null, a random-direction null and the
  surface baseline; a direction **counts** only if the CI of the difference lies above 0
  (and the trained AUC above the permutation null's 95 % quantile);
* :func:`length_covariates` / :func:`alignment_controls` -- the alignment against length:
  source and target pooled states are residualised on length covariates (log ITR tokens,
  log1p call nodes, log1p value-carrying calls; optionally log text tokens) with
  coefficients fitted on the training pairs only, and ``r2_holdout`` is reported raw,
  residualised, for a length-only regression and for the untrained source, with the gain
  trained minus untrained and its bootstrap CI over the hold-out pairs.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from ..config import HookSite, ModelSpec
from ..directions import Direction, cosine, diff_of_means
from ..model import DTYPES, ModelBundle, load_model, resolve_device
from ..transplant import fit_alignment, target_residuals
from .itr import TxRecord
from .labels import mask_tokens, surface_features
from .text import render_text
from .vocab import TraceVocab, item_tokens

READOUTS = ("mean", "last", "call")
LENGTH_COVARIATES = ("log_n_tokens", "log1p_n_nodes", "log1p_n_value_calls")
TEXT_COVARIATE = "log_n_text_tokens"


def _local_bundle(spec: ModelSpec) -> ModelBundle:
    """Fallback loader for a local model directory (no revision argument at all)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = resolve_device(spec.device)
    tok = AutoTokenizer.from_pretrained(spec.name)
    model = AutoModelForCausalLM.from_pretrained(spec.name, dtype=DTYPES[spec.dtype],
                                                 attn_implementation=spec.attn_implementation)
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    return ModelBundle(model=model, tokenizer=tok, spec=spec, device=dev)


def load_trace_bundle(model_dir: str, device: str = "auto", dtype: str = "float32") -> ModelBundle:
    """:func:`lsne.model.load_model` on a local trace-LM directory (``revision=None``)."""
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"trace LM directory not found: {model_dir}")
    spec = ModelSpec(name=str(model_dir), revision=None, dtype=dtype, device=device)
    try:
        return load_model(spec)
    except (OSError, ValueError, TypeError):  # a transformers version that rejects the revision for local dirs
        return _local_bundle(spec)


def _snapshot_site(n_layers: int, layer: int) -> HookSite:
    """Residual snapshot ``layer`` (0..L, the :mod:`lsne.transplant` convention) -> HookSite."""
    if 0 <= int(layer) < n_layers:
        return HookSite(int(layer), "block_input")
    if int(layer) == n_layers:
        return HookSite(n_layers - 1, "block_output")
    raise ValueError(f"layer {layer} outside 0..{n_layers}")


class TraceLMSource:
    """:class:`lsne.transplant.SourceModel` adapter for a trained trace LM directory."""

    def __init__(self, model_dir: str, device: str = "auto", max_len: int = 512, dtype: str = "float32",
                 serialize_kw: Optional[dict] = None):
        self.model_dir = str(model_dir)
        self.mb = load_trace_bundle(self.model_dir, device=device, dtype=dtype)
        self.spec = self.mb.spec
        vpath = os.path.join(self.model_dir, "vocab.json")
        self.vocab = TraceVocab.load(vpath) if os.path.exists(vpath) else TraceVocab.from_tokenizer(self.mb.tokenizer)
        if len(self.vocab) != self.mb.vocab:
            raise ValueError(f"vocab.json has {len(self.vocab)} tokens, the model {self.mb.vocab}")
        self.max_len = int(max_len)
        self.serialize_kw = dict(serialize_kw or self.vocab.meta.get("params", {}).get("serialize_kw", {}) or {})
        self.name = f"trace-lm:{os.path.basename(os.path.normpath(self.model_dir))}"
        self.d_model = int(self.mb.d_model)
        self.n_layers = int(self.mb.n_layers)

    @property
    def tokenizer(self):
        return self.mb.tokenizer

    @property
    def device(self) -> torch.device:
        return self.mb.device

    def tokens(self, item) -> list:
        """Vocabulary-mapped ITR tokens of ``item`` (truncated to ``max_len``)."""
        return item_tokens(item, self.vocab, self.serialize_kw)[: self.max_len]

    def encode(self, item) -> torch.Tensor:
        """``LongTensor[1, T]`` (``T <= max_len``) for a TxRecord, token list, shard row or ids."""
        if isinstance(item, torch.Tensor) or (isinstance(item, (list, tuple)) and item and
                                              isinstance(item[0], (int, np.integer))):
            ids = torch.as_tensor(item, dtype=torch.long).reshape(1, -1)[:, : self.max_len]
            return ids.to(self.device)
        toks = self.tokens(item)
        if not toks:
            raise ValueError("empty ITR token list")
        ids = self.vocab.ids(toks)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def residuals(self, ids: torch.Tensor, layers: Optional[Sequence[int]] = None) -> dict:
        """``{layer: Tensor[T, d]}`` (float32, CPU): block input of ``layer`` for ``layer < L``,
        output of the last block for ``layer == L``."""
        return target_residuals(self.mb, ids, layers)

    def __repr__(self) -> str:
        return f"TraceLMSource({self.name}, d={self.d_model}, L={self.n_layers}, vocab={len(self.vocab)})"


def init_seed_of(model_dir: str, default: int = 0) -> int:
    """The initialisation / training seed recorded in ``model_dir/manifest.json``
    (``hparams.seed``, written by :func:`lsne.chain.train.train`), else ``default``."""
    p = os.path.join(str(model_dir), "manifest.json")
    try:
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        return int(m.get("hparams", {}).get("seed", default))
    except (OSError, ValueError, TypeError, AttributeError):
        return int(default)


def lm_val_frac(model_dir: str, default: Optional[float] = None) -> Optional[float]:
    """The hash split of the LM's training corpus (``corpus.val_frac`` of ``manifest.json``,
    written by ``python -m lsne.chain.cli train``): a transaction belongs to the LM's held-out
    part when the last 8 hex digits of its hash / 2**32 are below it (:func:`is_lm_val`)."""
    p = os.path.join(str(model_dir), "manifest.json")
    try:
        with open(p, encoding="utf-8") as fh:
            m = json.load(fh)
        v = (m.get("corpus") or {}).get("val_frac")
        return default if v is None else float(v)
    except (OSError, ValueError, TypeError, AttributeError):
        return default


def is_lm_val(tx_hash: str, val_frac: Optional[float]) -> bool:
    """Membership of a transaction in the LM's held-out split (same rule as the training
    corpus, ``lsne.chain.cli._is_val``)."""
    return bool(val_frac) and float(val_frac) > 0 and (int(str(tx_hash)[-8:], 16) / 2 ** 32) < float(val_frac)


SPLITS = ("all", "heldout", "train")


def split_filter(hashes: Sequence[str], split: str, val_frac: Optional[float]) -> list:
    """Booleans: which transactions belong to ``split`` (``"all"``, ``"heldout"`` = the LM's
    validation split, ``"train"``); ``heldout`` / ``train`` need ``val_frac``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if split == "all":
        return [True] * len(hashes)
    if not val_frac:
        raise ValueError(f"split={split!r} needs the LM's val_frac (model manifest corpus.val_frac)")
    v = [is_lm_val(h, val_frac) for h in hashes]
    return v if split == "heldout" else [not x for x in v]


def model_fingerprint(model_dir: str) -> Optional[str]:
    """Short content fingerprint of a model directory: sha256 over ``config.json``,
    ``vocab.json`` and, per weights file, its size plus its first and last MiB.  Survives
    copying / moving the directory (Drive mounts), changes with every retraining."""
    import hashlib

    d = str(model_dir)
    h = hashlib.sha256()
    found = False
    for name in ("config.json", "vocab.json"):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                h.update(name.encode() + b"\0" + fh.read())
            found = True
    for name in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if name.endswith((".safetensors", ".bin")) and os.path.isfile(os.path.join(d, name)):
            p = os.path.join(d, name)
            size = os.path.getsize(p)
            with open(p, "rb") as fh:
                head = fh.read(1 << 20)
                fh.seek(max(0, size - (1 << 20)))
                tail = fh.read(1 << 20)
            h.update(f"{name}\0{size}\0".encode() + head + tail)
            found = True
    return h.hexdigest()[:24] if found else None


def dedup_pairs(pos_items: Sequence, neg_items: Sequence) -> list:
    """Indices of the matched pairs to keep so that no token sequence occurs twice in the
    probe set: pairs are scanned in order and dropped when the positive's or the negative's
    (vocabulary-mapped, truncated, masked) sequence was already seen in either class.  The
    split halves and the pair bootstrap of :func:`probe_criterion` treat pairs as independent
    units; identical sequences in both halves would make the split-half AUC partly in-sample
    and the CIs too narrow."""
    seen, keep = set(), []
    for i, (p, q) in enumerate(zip(pos_items, neg_items)):
        kp, kq = tuple(p), tuple(q)
        if kp in seen or kq in seen or kp == kq:
            continue
        seen.update((kp, kq))
        keep.append(i)
    return keep


def untrained_twin(src: TraceLMSource, seed: int = 0) -> TraceLMSource:
    """Control source: identical architecture (the trained model's config), vocabulary,
    tokenizer, serialisation and ``max_len``, weights freshly initialised exactly like
    :func:`lsne.chain.train.make_model` (``torch.manual_seed(seed)`` inside a forked RNG,
    then the model constructor).  With ``seed`` = the training seed (:func:`init_seed_of`)
    these are the trained model's own initial weights.  The model is not written to disk."""
    model = src.mb.model
    cfg = copy.deepcopy(model.config)
    ref = next(model.parameters())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        twin = type(model)(cfg)
    twin = twin.to(dtype=ref.dtype).eval().to(src.device)
    for p in twin.parameters():
        p.requires_grad_(False)
    out = copy.copy(src)
    out.mb = ModelBundle(model=twin, tokenizer=src.mb.tokenizer, spec=src.mb.spec, device=src.mb.device)
    out.spec = out.mb.spec
    out.name = f"{src.name}:untrained(seed={int(seed)})"
    return out


def masked_items(src, items: Sequence, mask="dao") -> list:
    """Vocabulary-mapped token lists of ``items`` (truncated to ``src.max_len``) with the
    identity tokens of ``mask`` replaced by ``[A_OOV]`` (:func:`lsne.chain.labels.mask_tokens`);
    the result encodes with ``src.encode``.  Applied identically to positives and negatives."""
    return [mask_tokens(src.tokens(it), mask) for it in items]


# ======================================================================================
# pooled states and the sharing direction
# ======================================================================================
def _pool(H: torch.Tensor, toks: Optional[list], readout: str) -> torch.Tensor:
    H = torch.as_tensor(H).detach().float().cpu()
    if readout == "mean":
        return H.mean(0)
    if readout == "last":
        return H[-1].clone()
    if readout == "call":
        pos = [i for i, t in enumerate(toks or []) if t == "[CALL]" and i < H.shape[0]]
        return H[pos].mean(0) if pos else H.mean(0)
    raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")


def pooled_states(src, items: Sequence, layers: Sequence[int], readout: str = "mean") -> dict:
    """``{layer: Tensor[N, d]}`` pooled residuals of ``items``: ``"mean"`` over positions,
    ``"last"`` position or ``"call"`` = mean over the ``[CALL]`` positions (falls back to the
    mean when there is none; needs a :class:`TraceLMSource`)."""
    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    layers = [int(l) for l in layers]
    out = {l: [] for l in layers}
    for it in items:
        ids = src.encode(it)
        toks = src.tokens(it) if readout == "call" else None
        hs = src.residuals(ids, layers)
        for l in layers:
            out[l].append(_pool(hs[l], toks, readout))
    return {l: torch.stack(v) if v else torch.zeros(0, src.d_model) for l, v in out.items()}


def holdout_auc(S_pos: torch.Tensor, S_neg: torch.Tensor, seed: int = 0, denoise_pcs: int = 0) -> float:
    """Split-half generalisation of the difference-of-means direction: the (matched) pairs are
    split in two halves, the direction is fitted on one half and the AUC of the projections
    (positive > negative) is measured on the other; mean over both directions of the split.
    NaN with fewer than 2 pairs per half."""
    n = min(len(S_pos), len(S_neg))
    if n < 4:
        return float("nan")
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    aucs = []
    for a, b in (halves, halves[::-1]):
        try:
            v = diff_of_means(S_pos[a], S_neg[a], denoise_pcs=denoise_pcs)
        except ValueError:
            continue
        p = (S_pos[b].float() @ v).numpy()
        q = (S_neg[b].float() @ v).numpy()
        gt = (p[:, None] > q[None, :]).mean() + 0.5 * (p[:, None] == q[None, :]).mean()
        aucs.append(float(gt))
    return float(np.mean(aucs)) if aucs else float("nan")


# ======================================================================================
# probe criterion: trained minus untrained, with bootstrap CI and nulls
# ======================================================================================
def split_halves(n: int, seed: int = 0) -> tuple:
    """The two halves of :func:`holdout_auc` (and :func:`lsne.chain.labels.surface_baseline_auc`)."""
    perm = np.random.default_rng(int(seed)).permutation(int(n))
    return perm[: int(n) // 2], perm[int(n) // 2:]


def _auc_cols(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Column-wise AUC ``P(p > q) + 0.5 P(p == q)`` (Mann-Whitney with average ranks):
    ``P [m, R]``, ``Q [k, R]`` -> ``[R]``."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if P.ndim == 1:
        P, Q = P[:, None], Q[:, None]
    m, k = P.shape[0], Q.shape[0]
    ranks = rankdata(np.concatenate([P, Q], 0), axis=0)
    return (ranks[:m].sum(0) - m * (m + 1) / 2.0) / float(m * k)


def _np64(S) -> np.ndarray:
    return torch.as_tensor(S).detach().double().cpu().numpy() if isinstance(S, torch.Tensor) else \
        np.asarray(S, dtype=np.float64)


def _fold_auc(Xp: np.ndarray, Xq: np.ndarray, fit: np.ndarray, ev: np.ndarray) -> float:
    """Difference-of-means direction fitted on the pairs ``fit``, AUC of the projections of the
    pairs ``ev`` (0.5 when the mean difference vanishes)."""
    v = Xp[fit].mean(0) - Xq[fit].mean(0)
    if not np.any(v):
        return 0.5
    return float(_auc_cols(Xp[ev] @ v, Xq[ev] @ v)[0])


def _split_half(Xp: np.ndarray, Xq: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    return 0.5 * (_fold_auc(Xp, Xq, A, B) + _fold_auc(Xp, Xq, B, A))


def split_half_auc(S_pos, S_neg, seed: int = 0) -> float:
    """:func:`holdout_auc` (``denoise_pcs=0``) in float64 numpy; NaN with fewer than 4 pairs."""
    Xp, Xq = _np64(S_pos), _np64(S_neg)
    n = min(len(Xp), len(Xq))
    if n < 4:
        return float("nan")
    A, B = split_halves(n, seed)
    return _split_half(Xp[:n], Xq[:n], A, B)


def _quantile_ci(x: np.ndarray, level: float) -> tuple:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return float("nan"), float("nan")
    a = (1.0 - float(level)) / 2.0
    return float(np.quantile(x, a)), float(np.quantile(x, 1.0 - a))


def random_directions(d: int, n: int = 200, seed: int = 1234) -> np.ndarray:
    """``[n, d]`` random unit directions (seeded, float64)."""
    R = torch.randn(int(n), int(d), generator=torch.Generator().manual_seed(int(seed))).double()
    return (R / R.norm(dim=1, keepdim=True)).numpy()


def random_direction_auc(S_pos, S_neg, seed: int = 0, n_random: int = 200, random_seed: int = 1234) -> np.ndarray:
    """Random-direction null: AUC of fixed random unit directions (not fitted) on the two
    halves of the split, averaged over the halves and folded to ``max(a, 1 - a)`` (generous
    to the null).  Returns ``[n_random]``."""
    Xp, Xq = _np64(S_pos), _np64(S_neg)
    n = min(len(Xp), len(Xq))
    if n < 4 or int(n_random) <= 0:
        return np.full(0, np.nan)
    R = random_directions(Xp.shape[1], n_random, random_seed)
    A, B = split_halves(n, seed)
    a = 0.5 * (_auc_cols(Xp[A] @ R.T, Xq[A] @ R.T) + _auc_cols(Xp[B] @ R.T, Xq[B] @ R.T))
    return np.maximum(a, 1.0 - a)


def probe_criterion(S_pos, S_neg, S0_pos, S0_neg, seed: int = 0, n_boot: int = 1000, n_perm: int = 200,
                    n_random: int = 200, ci: float = 0.95, n_tests: int = 1, boot_seed: int = 20261026,
                    perm_seed: int = 7, random_seed: int = 1234) -> dict:
    """The round-2 probe criterion for one residual snapshot.

    ``S_pos``/``S_neg`` are the pooled states of the matched positives / negatives (row
    ``i`` of both = matched pair ``i``) under the trained model, ``S0_pos``/``S0_neg`` the
    same records under the untrained twin.  With the split of :func:`holdout_auc`
    (``seed``):

    * ``auc_trained`` / ``auc_untrained``: split-half AUC of the difference-of-means
      direction (fit on one half, AUC on the other, mean of both directions);
      ``delta = auc_trained - auc_untrained``;
    * bootstrap over matched pairs (``n_boot``): each half is resampled with replacement
      separately (no pair can land in both the fitting and the evaluation half), the
      direction is refitted, the *same* resampled pairs are used for both models ->
      percentile CIs ``delta_ci_lo/hi`` (level ``ci``; the decision CI, conservative: it
      contains the direction-estimation uncertainty), ``auc_trained_ci_*``,
      ``auc_untrained_ci_*``; ``delta_ci_eval_lo/hi``: the same draws with the directions of
      the original halves fixed (evaluation sampling only, for comparison); with
      ``n_tests > 1`` also the Bonferroni level ``1 - (1 - ci) / n_tests``
      (``delta_ci_lo_adj/hi_adj``, ``counts_adj``);
    * label-permutation null (``n_perm``): positive and negative swapped inside random
      pairs, same swaps for both models -> ``perm_auc_p95``/``perm_p`` (trained AUC) and
      ``perm_delta_p95``/``perm_delta_p`` (difference); p-values ``(1 + #null >= obs) /
      (1 + n_perm)``;
    * random-direction null (:func:`random_direction_auc`, trained and untrained states);
    * ``counts = delta_ci_lo > 0 and auc_trained > perm_auc_p95``.
    """
    Xp, Xq, Yp, Yq = (_np64(x) for x in (S_pos, S_neg, S0_pos, S0_neg))
    n = min(len(Xp), len(Xq), len(Yp), len(Yq))
    out = dict(n_pairs=int(n), ci_level=float(ci), n_boot=int(n_boot), n_perm=int(n_perm), n_random=int(n_random))
    keys = ("auc_trained", "auc_untrained", "delta", "delta_ci_lo", "delta_ci_hi", "delta_boot_sd",
            "delta_boot_mean", "delta_ci_eval_lo", "delta_ci_eval_hi",
            "auc_trained_ci_lo", "auc_trained_ci_hi", "auc_untrained_ci_lo", "auc_untrained_ci_hi",
            "perm_auc_mean", "perm_auc_p95", "perm_p", "perm_delta_p95", "perm_delta_p",
            "random_dir_auc_mean", "random_dir_auc_p95", "random_dir_auc_max", "random_dir_auc_p95_untrained")
    out.update({k: float("nan") for k in keys})
    out["counts"] = False
    if int(n_tests) > 1:
        out.update(ci_level_adj=1.0 - (1.0 - float(ci)) / int(n_tests), delta_ci_lo_adj=float("nan"),
                   delta_ci_hi_adj=float("nan"), counts_adj=False)
    if n < 4:
        return out
    Xp, Xq, Yp, Yq = Xp[:n], Xq[:n], Yp[:n], Yq[:n]
    A, B = split_halves(n, seed)
    a1, a0 = _split_half(Xp, Xq, A, B), _split_half(Yp, Yq, A, B)
    out.update(auc_trained=a1, auc_untrained=a0, delta=a1 - a0)
    # bootstrap over matched pairs, halves resampled separately, same draws for both models;
    # (i) refit: the direction is refitted on the resampled fitting half (decision CI, includes
    # the direction-estimation uncertainty); (ii) eval-only: directions of the original halves
    # fixed, only the evaluation pairs resampled (test-sampling uncertainty alone, narrower)
    rng = np.random.default_rng(int(boot_seed))
    b1, b0, e1, e0 = (np.empty(int(n_boot)) for _ in range(4))

    def scores(P, Q, fit, ev):
        v = P[fit].mean(0) - Q[fit].mean(0)
        return P[ev] @ v, Q[ev] @ v

    sc1 = (scores(Xp, Xq, A, B), scores(Xp, Xq, B, A))
    sc0 = (scores(Yp, Yq, A, B), scores(Yp, Yq, B, A))
    for i in range(int(n_boot)):
        ia = rng.integers(0, len(A), len(A))
        ib = rng.integers(0, len(B), len(B))
        b1[i] = _split_half(Xp, Xq, A[ia], B[ib])
        b0[i] = _split_half(Yp, Yq, A[ia], B[ib])
        e1[i] = 0.5 * (_auc_cols(sc1[0][0][ib], sc1[0][1][ib])[0] + _auc_cols(sc1[1][0][ia], sc1[1][1][ia])[0])
        e0[i] = 0.5 * (_auc_cols(sc0[0][0][ib], sc0[0][1][ib])[0] + _auc_cols(sc0[1][0][ia], sc0[1][1][ia])[0])
    if int(n_boot) > 1:
        d = b1 - b0
        out["delta_ci_lo"], out["delta_ci_hi"] = _quantile_ci(d, ci)
        out["delta_boot_sd"] = float(np.std(d, ddof=1))
        out["delta_boot_mean"] = float(np.mean(d))
        out["delta_ci_eval_lo"], out["delta_ci_eval_hi"] = _quantile_ci(e1 - e0, ci)
        out["auc_trained_ci_lo"], out["auc_trained_ci_hi"] = _quantile_ci(b1, ci)
        out["auc_untrained_ci_lo"], out["auc_untrained_ci_hi"] = _quantile_ci(b0, ci)
        if int(n_tests) > 1:
            out["delta_ci_lo_adj"], out["delta_ci_hi_adj"] = _quantile_ci(d, out["ci_level_adj"])
    # label-permutation null (swap positive / negative inside random pairs)
    rng = np.random.default_rng(int(perm_seed))
    p1, pd_ = np.empty(int(n_perm)), np.empty(int(n_perm))
    for i in range(int(n_perm)):
        sw = (rng.random(n) < 0.5)[:, None]
        q1 = _split_half(np.where(sw, Xq, Xp), np.where(sw, Xp, Xq), A, B)
        q0 = _split_half(np.where(sw, Yq, Yp), np.where(sw, Yp, Yq), A, B)
        p1[i], pd_[i] = q1, q1 - q0
    if int(n_perm) > 0:
        out.update(perm_auc_mean=float(p1.mean()), perm_auc_p95=float(np.quantile(p1, 0.95)),
                   perm_p=float((1 + (p1 >= a1).sum()) / (1 + len(p1))),
                   perm_delta_p95=float(np.quantile(pd_, 0.95)),
                   perm_delta_p=float((1 + (pd_ >= a1 - a0).sum()) / (1 + len(pd_))))
    # random-direction null
    if int(n_random) > 0:
        r1 = random_direction_auc(Xp, Xq, seed=seed, n_random=n_random, random_seed=random_seed)
        r0 = random_direction_auc(Yp, Yq, seed=seed, n_random=n_random, random_seed=random_seed)
        out.update(random_dir_auc_mean=float(r1.mean()), random_dir_auc_p95=float(np.quantile(r1, 0.95)),
                   random_dir_auc_max=float(r1.max()), random_dir_auc_p95_untrained=float(np.quantile(r0, 0.95)))
    perm_ok = (not np.isfinite(out["perm_auc_p95"])) or a1 > out["perm_auc_p95"]
    out["counts"] = bool(np.isfinite(out["delta_ci_lo"]) and out["delta_ci_lo"] > 0 and perm_ok)
    if int(n_tests) > 1:
        out["counts_adj"] = bool(np.isfinite(out["delta_ci_lo_adj"]) and out["delta_ci_lo_adj"] > 0 and perm_ok)
    return out


def criterion_table(S: dict, S0: dict, n_pos: int, layers: Optional[Sequence[int]] = None,
                    surface: Optional[dict] = None, **kw) -> pd.DataFrame:
    """:func:`probe_criterion` for every residual snapshot: ``S`` / ``S0`` are
    ``{layer: [2n, d]}`` pooled states (first ``n_pos`` rows positives, then the matched
    negatives in the same order) of the trained model and the untrained twin.  ``n_tests``
    defaults to the number of layers (Bonferroni columns).  ``surface`` (the result of
    :func:`lsne.chain.labels.surface_baseline_auc` with the same seed) adds
    ``surface_auc`` (combined) and ``beats_surface`` (``auc_trained`` > it)."""
    layers = sorted(S) if layers is None else [int(l) for l in layers]
    kw.setdefault("n_tests", len(layers))
    n = int(n_pos)
    rows = []
    for l in layers:
        r = probe_criterion(S[l][:n], S[l][n:], S0[l][:n], S0[l][n:], **kw)
        r = dict(layer=int(l), **r)
        if surface is not None:
            r["surface_auc"] = float(surface.get("combined", float("nan")))
            r["beats_surface"] = bool(np.isfinite(r["surface_auc"]) and r["auc_trained"] > r["surface_auc"])
        rows.append(r)
    return pd.DataFrame(rows)


def direction_from_states(S_pos: torch.Tensor, S_neg: torch.Tensor, layer: int, n_layers: int,
                          denoise_pcs: int = 0, name: str = "sharing", meta: Optional[dict] = None,
                          seed: int = 0) -> Direction:
    """Difference-of-means :class:`~lsne.directions.Direction` (``mean(pos) - mean(neg)``)
    at residual snapshot ``layer`` with the denoised variant (1 within-class PC removed) and
    separation statistics in ``meta``: ``vec_denoised``, ``cos_plain_denoised``,
    ``proj_pos_mean`` / ``proj_neg_mean``, ``separation_d`` (in-sample Cohen's d of the
    projections, optimistic) and ``auc_holdout`` (split-half, :func:`holdout_auc`)."""
    S_pos, S_neg = torch.as_tensor(S_pos).float(), torch.as_tensor(S_neg).float()
    vec = diff_of_means(S_pos, S_neg, denoise_pcs=int(denoise_pcs))
    try:
        vec_dn = diff_of_means(S_pos, S_neg, denoise_pcs=1)
        cos_dn = cosine(vec, vec_dn)
    except ValueError:
        vec_dn, cos_dn = None, float("nan")
    p, q = S_pos @ vec, S_neg @ vec
    pooled_sd = float(torch.cat([p - p.mean(), q - q.mean()]).std()) if len(p) + len(q) > 2 else float("nan")
    sep = float((p.mean() - q.mean()) / pooled_sd) if pooled_sd and np.isfinite(pooled_sd) and pooled_sd > 0 \
        else float("nan")
    m = dict(layer=int(layer), n_pos=int(len(S_pos)), n_neg=int(len(S_neg)), denoise_pcs=int(denoise_pcs),
             vec_denoised=vec_dn, cos_plain_denoised=float(cos_dn), proj_pos_mean=float(p.mean()),
             proj_neg_mean=float(q.mean()), separation_d=sep,
             auc_holdout=holdout_auc(S_pos, S_neg, seed=seed, denoise_pcs=int(denoise_pcs)),
             auc_holdout_denoised=holdout_auc(S_pos, S_neg, seed=seed, denoise_pcs=1),
             mean_diff_norm=float((S_pos.mean(0) - S_neg.mean(0)).norm()))
    m.update(meta or {})
    return Direction(name=name, vec=vec, site=_snapshot_site(int(n_layers), int(layer)), meta=m)


def sharing_direction(src, pos: Sequence, neg: Sequence, layer: int, readout: str = "mean",
                      denoise_pcs: int = 0, name: str = "sharing", seed: int = 0) -> Direction:
    """Sharing direction in the source model: :func:`lsne.directions.diff_of_means` of the pooled
    residuals (snapshot ``layer``) of the positive vs. the negative records (``denoise_pcs=0``
    by default; the denoised variant is reported in ``meta["vec_denoised"]``, see
    :func:`direction_from_states`).  ``site`` is the source model's HookSite of that snapshot."""
    S = pooled_states(src, list(pos) + list(neg), [int(layer)], readout)[int(layer)]
    n = len(pos)
    return direction_from_states(S[:n], S[n:], int(layer), src.n_layers, denoise_pcs=denoise_pcs, name=name,
                                 seed=seed, meta=dict(readout=readout, source=getattr(src, "name", "")))


# ======================================================================================
# pairs for the alignment
# ======================================================================================
def paired_items(records: Sequence, max_nodes: int = 12, address_style: str = "short") -> list:
    """``[(item_for_source, english_text_for_target)]`` — the input of
    :func:`lsne.transplant.layer_matching` / :func:`~lsne.transplant.fit_alignment`.  Records
    are rendered with :func:`lsne.chain.text.render_text`; shard rows / pair-file rows that
    carry no record use their stored ``text``."""
    out = []
    for r in records:
        if isinstance(r, TxRecord):
            out.append((r, render_text(r, max_nodes=max_nodes, address_style=address_style)))
            continue
        rec = r.get("record") if isinstance(r, dict) else getattr(r, "record", None)
        if rec is not None:
            rec = rec if isinstance(rec, TxRecord) else TxRecord.from_dict(rec)
            out.append((rec, render_text(rec, max_nodes=max_nodes, address_style=address_style)))
            continue
        text = r.get("text") if isinstance(r, dict) else getattr(r, "text", None)
        if not text:
            raise ValueError("item has neither a record nor a stored text")
        out.append((r, str(text)))
    return out


def load_pairs(path: str, with_hashes: bool = False):
    """Read a pair file written by ``python -m lsne.chain.cli pair`` -> ``[(tokens, text)]``
    (``with_hashes``: ``(pairs, tx_hashes)``)."""
    out, hashes = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append((list(d["tokens"]), str(d["text"])))
                hashes.append(d.get("tx_hash"))
    return (out, hashes) if with_hashes else out


# ======================================================================================
# alignment controls: length covariates, residualisation, trained vs. untrained
# ======================================================================================
def length_covariates(items: Sequence, texts: Optional[Sequence[str]] = None, tokenizer=None,
                      max_len: int = 512) -> pd.DataFrame:
    """Length features of the source items (raw ITR tokens of a record / token list / shard
    row): ``n_tokens`` (visible length ``min(len, max_len)``), ``n_tokens_raw``, ``n_nodes``
    (``[CALL]`` entries = call-tree nodes) and ``n_value_calls`` (calls with a non-zero
    ``V:`` value as the tokens show them -- reverted and ``call_code`` / ``delegate_call``
    value calls included, unlike the label feature
    :func:`lsne.chain.labels.sharing_features` ``n_value_calls``), counted on the full
    transaction; with ``texts`` and the target ``tokenizer`` also ``n_text_tokens``."""
    rows = []
    for it in items:
        toks = item_tokens(it, None)
        sf = surface_features(toks)
        rows.append(dict(n_tokens=min(len(toks), int(max_len)), n_tokens_raw=len(toks), n_nodes=sf["n_call"],
                         n_value_calls=sf["n_value_calls_nonzero"]))
    df = pd.DataFrame(rows, columns=["n_tokens", "n_tokens_raw", "n_nodes", "n_value_calls"])
    if texts is not None:
        if tokenizer is None:
            raise ValueError("n_text_tokens needs the target tokenizer")
        df["n_text_tokens"] = [len(tokenizer(t)["input_ids"]) for t in texts]
    return df


def covariate_matrix(cov: pd.DataFrame, text: bool = False) -> tuple:
    """``(Z [N, k], names)``: ``log n_tokens``, ``log1p n_nodes``, ``log1p n_value_calls``
    (:data:`LENGTH_COVARIATES`) and with ``text=True`` ``log n_text_tokens``."""
    cols = [np.log(np.maximum(cov["n_tokens"].to_numpy(float), 1.0)), np.log1p(cov["n_nodes"].to_numpy(float)),
            np.log1p(cov["n_value_calls"].to_numpy(float))]
    names = list(LENGTH_COVARIATES)
    if text:
        cols.append(np.log(np.maximum(cov["n_text_tokens"].to_numpy(float), 1.0)))
        names.append(TEXT_COVARIATE)
    return np.column_stack(cols), names


def alignment_split(n: int, holdout: float = 0.2, seed: int = 0) -> tuple:
    """``(idx_holdout, idx_train)`` exactly as :func:`lsne.transplant.fit_alignment` splits."""
    perm = np.random.default_rng(seed).permutation(int(n))
    n_hold = int(round(float(holdout) * n))
    if holdout > 0:
        n_hold = max(n_hold, 1)
    n_hold = min(n_hold, int(n) - 2)
    return perm[:n_hold], perm[n_hold:]


def residualize(X, Z, train_idx=None) -> np.ndarray:
    """``X - [1, Z] beta`` with ``beta`` the least-squares fit of ``X`` on ``[1, Z]`` over the
    rows ``train_idx`` (default all rows), applied to every row (no hold-out leakage)."""
    X = _np64(X)
    Z = np.asarray(Z, dtype=np.float64)
    Z1 = np.column_stack([np.ones(len(Z)), Z])
    tr = np.arange(len(X)) if train_idx is None else np.asarray(train_idx)
    beta, *_ = np.linalg.lstsq(Z1[tr], X[tr], rcond=None)
    return X - Z1 @ beta


#: smallest hold-out for the alignment bootstrap CIs (below: NaN, the CI cannot be estimated)
MIN_HOLDOUT_BOOT = 8
#: share of draws that must be usable (target variance not ~0) for a CI
MIN_VALID_BOOT = 0.9


def _boot_r2_gain(Y: np.ndarray, P1: np.ndarray, P0: np.ndarray, n_boot: int, rng) -> tuple:
    """Bootstrap over hold-out rows of ``R2(P1) - R2(P0)`` and of ``R2(P1)``: ``(gain, r2_1, r2_0)``
    arrays.  Draws whose resampled target variance is ~0 (a few distinct rows) give no R²
    and are dropped (NaN); with fewer than :data:`MIN_HOLDOUT_BOOT` hold-out rows or fewer than
    :data:`MIN_VALID_BOOT` usable draws every value is NaN (no CI rather than an absurd one)."""
    m = len(Y)
    if m < MIN_HOLDOUT_BOOT or n_boot <= 1:
        return np.full(0, np.nan), np.full(0, np.nan), np.full(0, np.nan)
    C = np.zeros((int(n_boot), m))
    draws = rng.integers(0, m, (int(n_boot), m))
    np.add.at(C, (np.repeat(np.arange(int(n_boot)), m), draws.ravel()), 1.0)
    rowsq = (Y ** 2).sum(1)
    mean_b = (C @ Y) / m
    ss_tot = C @ rowsq - m * (mean_b ** 2).sum(1)
    e1 = ((Y - P1) ** 2).sum(1)
    e0 = ((Y - P0) ** 2).sum(1)
    full = float(((Y - Y.mean(0)) ** 2).sum())
    ok = np.isfinite(ss_tot) & (ss_tot > 1e-6 * max(full, 1e-300))
    with np.errstate(divide="ignore", invalid="ignore"):
        r1 = np.where(ok, 1.0 - (C @ e1) / np.where(ok, ss_tot, 1.0), np.nan)
        r0 = np.where(ok, 1.0 - (C @ e0) / np.where(ok, ss_tot, 1.0), np.nan)
    if ok.mean() < MIN_VALID_BOOT:
        nan = np.full(int(n_boot), np.nan)
        return nan, nan, nan
    return r1 - r0, r1, r0


def _predict(a, X: np.ndarray) -> np.ndarray:
    return X @ a.W.double().numpy() + a.b.double().numpy()


def alignment_controls(S: dict, S0: dict, T: dict, Z, layers_src: Optional[Sequence[int]] = None,
                       layers_tgt: Optional[Sequence[int]] = None, alpha: float = 1.0, holdout: float = 0.2,
                       seed: int = 0, n_boot: int = 1000, ci: float = 0.95, z_alpha: float = 1e-3,
                       boot_seed: int = 20261026) -> pd.DataFrame:
    """Alignment ``H_T ~ H_S W + b`` (ridge, :func:`lsne.transplant.fit_alignment`, the same
    seeded split everywhere) with length controls, one row per (source layer, target layer).

    ``S`` / ``S0``: ``{layer: [N, d_S]}`` pooled source states of the trained model / the
    untrained twin, ``T``: ``{layer: [N, d_T]}`` pooled target states of the paired texts,
    ``Z [N, k]`` length covariates (:func:`covariate_matrix`).  Columns:

    * ``r2_raw`` / ``r2_raw_untrained``: hold-out R² of the plain alignment; ``r2_permuted``
      the permuted-target null of the trained one; ``gain_raw`` = trained - untrained;
    * ``r2_resid`` / ``r2_resid_untrained``: both source and target residualised on ``[1,
      Z]`` (:func:`residualize`, coefficients from the training pairs only), then the same
      fit -- the share of the *length-free* target variance the source predicts;
      ``gain_resid`` = trained - untrained;
    * ``r2_length_only``: ``Z`` alone as the source (ridge ``z_alpha`` ~ OLS);
    * ``*_ci_lo`` / ``*_ci_hi``: percentile CIs (level ``ci``) of the gains from a paired
      bootstrap over the hold-out pairs (the fitted maps fixed).
    """
    Z = np.asarray(Z, dtype=np.float64)
    N = len(Z)
    layers_src = sorted(S) if layers_src is None else [int(l) for l in layers_src]
    layers_tgt = sorted(T) if layers_tgt is None else [int(l) for l in layers_tgt]
    hold, train = alignment_split(N, holdout, seed)
    fit_kw = dict(method="ridge", alpha=float(alpha), holdout=float(holdout), seed=int(seed))
    Tr = {lt: residualize(T[lt], Z, train) for lt in layers_tgt}
    Sr = {ls: residualize(S[ls], Z, train) for ls in layers_src}
    S0r = {ls: residualize(S0[ls], Z, train) for ls in layers_src}
    len_only = {}
    for lt in layers_tgt:
        a = fit_alignment(Z, _np64(T[lt]), method="ridge", alpha=float(z_alpha), holdout=float(holdout), seed=int(seed))
        len_only[lt] = a.r2_holdout
    rows = []
    for ls in layers_src:
        Xs, X0 = _np64(S[ls]), _np64(S0[ls])
        for lt in layers_tgt:
            Y = _np64(T[lt])
            a1 = fit_alignment(Xs, Y, layer_src=ls, layer_tgt=lt, **fit_kw)
            a0 = fit_alignment(X0, Y, layer_src=ls, layer_tgt=lt, **fit_kw)
            r1 = fit_alignment(Sr[ls], Tr[lt], layer_src=ls, layer_tgt=lt, **fit_kw)
            r0 = fit_alignment(S0r[ls], Tr[lt], layer_src=ls, layer_tgt=lt, **fit_kw)
            row = dict(layer_src=int(ls), layer_tgt=int(lt), alpha=float(alpha), n_pairs=int(N), n_holdout=len(hold),
                       r2_raw=a1.r2_holdout, r2_raw_untrained=a0.r2_holdout, gain_raw=a1.r2_holdout - a0.r2_holdout,
                       r2_permuted=a1.r2_permuted, r2_train_raw=a1.r2_train,
                       r2_resid=r1.r2_holdout, r2_resid_untrained=r0.r2_holdout,
                       gain_resid=r1.r2_holdout - r0.r2_holdout, r2_resid_permuted=r1.r2_permuted,
                       r2_length_only=len_only[lt])
            for tag, (fa, fb, Ysrc, Xa, Xb) in dict(raw=(a1, a0, Y, Xs, X0),
                                                     resid=(r1, r0, Tr[lt], Sr[ls], S0r[ls])).items():
                rng = np.random.default_rng(int(boot_seed))       # same draws for every cell
                g, _, _ = _boot_r2_gain(Ysrc[hold], _predict(fa, Xa[hold]), _predict(fb, Xb[hold]), n_boot, rng)
                row[f"gain_{tag}_ci_lo"], row[f"gain_{tag}_ci_hi"] = _quantile_ci(g, ci)
            rows.append(row)
    return pd.DataFrame(rows)
