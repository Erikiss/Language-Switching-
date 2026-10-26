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
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import numpy as np
import torch

from ..config import HookSite, ModelSpec
from ..directions import Direction, cosine, diff_of_means
from ..model import DTYPES, ModelBundle, load_model, resolve_device
from ..transplant import target_residuals
from .itr import TxRecord
from .text import render_text
from .vocab import TraceVocab, item_tokens

READOUTS = ("mean", "last", "call")


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


def load_pairs(path: str) -> list:
    """Read a pair file written by ``python -m lsne.chain.cli pair`` -> ``[(tokens, text)]``."""
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                out.append((list(d["tokens"]), str(d["text"])))
    return out
