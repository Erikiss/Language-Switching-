"""Minimal TopK sparse autoencoder in the EleutherAI ``sparsify`` format.

Hub layout (verified for ``EleutherAI/sae-pythia-70m-32k``): one folder per hookpoint
(``layers.N``, ``layers.N.mlp``, ``layers.N.attention``) holding ``cfg.json`` with keys
``k, num_latents, d_in, normalize_decoder, ...`` and ``sae.safetensors`` with tensors
``encoder.weight [F,d]``, ``encoder.bias [F]``, ``W_dec [F,d]``, ``b_dec [d]``.

Semantics (as in ``sparsify.Sae``):
    pre   = (x - b_dec) @ W_enc.T + b_enc
    acts  = relu(pre);  keep the k largest per sample (TopK)
    x_hat = sum_j acts_j * W_dec[idx_j] + b_dec
"""
from __future__ import annotations

import json
import os
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import SAE_REPOS

_EPS = 1e-30


class TopKSAE(nn.Module):
    """TopK SAE with parameters ``W_enc [F,d]``, ``b_enc [F]``, ``W_dec [F,d]``, ``b_dec [d]``
    and the integer sparsity ``k``. Parameters are frozen (inference only)."""

    def __init__(self, d_in: int, num_latents: int, k: int, cfg: Optional[dict] = None):
        super().__init__()
        self.d_in = int(d_in)
        self.num_latents = int(num_latents)
        self.k = int(k)
        self.cfg = dict(cfg or {})
        self.W_enc = nn.Parameter(torch.zeros(self.num_latents, self.d_in), requires_grad=False)
        self.b_enc = nn.Parameter(torch.zeros(self.num_latents), requires_grad=False)
        self.W_dec = nn.Parameter(torch.zeros(self.num_latents, self.d_in), requires_grad=False)
        self.b_dec = nn.Parameter(torch.zeros(self.d_in), requires_grad=False)

    # ---- construction ------------------------------------------------------------------
    @classmethod
    def from_tensors(cls, W_enc: torch.Tensor, b_enc: torch.Tensor, W_dec: torch.Tensor,
                     b_dec: torch.Tensor, k: int, cfg: Optional[dict] = None) -> "TopKSAE":
        """Build an SAE from explicit tensors (used by tests and for synthetic SAEs)."""
        W_enc = torch.as_tensor(W_enc).detach().float()
        b_enc = torch.as_tensor(b_enc).detach().float()
        W_dec = torch.as_tensor(W_dec).detach().float()
        b_dec = torch.as_tensor(b_dec).detach().float()
        F, d = W_enc.shape
        if W_dec.shape != (F, d) or b_enc.shape != (F,) or b_dec.shape != (d,):
            raise ValueError(f"inconsistent shapes: W_enc {tuple(W_enc.shape)}, b_enc {tuple(b_enc.shape)}, "
                             f"W_dec {tuple(W_dec.shape)}, b_dec {tuple(b_dec.shape)}")
        if not (1 <= int(k) <= F):
            raise ValueError(f"k={k} must be in [1, {F}]")
        sae = cls(d, F, int(k), cfg)
        with torch.no_grad():
            sae.W_enc.copy_(W_enc)
            sae.b_enc.copy_(b_enc)
            sae.W_dec.copy_(W_dec)
            sae.b_dec.copy_(b_dec)
        return sae.eval()

    @classmethod
    def load_from_hub(cls, repo: str, hookpoint: str, device="cpu",
                      cache_dir: Optional[str] = None) -> "TopKSAE":
        """Download ``<hookpoint>/cfg.json`` and ``<hookpoint>/sae.safetensors`` from the
        Hugging Face repo ``repo`` (EleutherAI sparsify layout) and build the SAE on ``device``."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cfg_path = hf_hub_download(repo_id=repo, filename=f"{hookpoint}/cfg.json", cache_dir=cache_dir)
        st_path = hf_hub_download(repo_id=repo, filename=f"{hookpoint}/sae.safetensors", cache_dir=cache_dir)
        with open(cfg_path) as f:
            cfg = json.load(f)
        tensors = load_file(st_path)
        return cls.from_state(tensors, cfg, device=device, repo=repo, hookpoint=hookpoint)

    @classmethod
    def from_state(cls, tensors: dict, cfg: dict, device="cpu", **extra) -> "TopKSAE":
        """Build from a sparsify state dict (keys ``encoder.weight``, ``encoder.bias``,
        ``W_dec``, ``b_dec``) and its cfg dict."""
        W_enc = tensors["encoder.weight"]
        b_enc = tensors.get("encoder.bias", torch.zeros(W_enc.shape[0]))
        W_dec = tensors["W_dec"]
        b_dec = tensors.get("b_dec", torch.zeros(W_enc.shape[1]))
        k = int(cfg["k"])
        F, d = W_enc.shape
        if "num_latents" in cfg and int(cfg["num_latents"]) != F:
            raise ValueError(f"cfg num_latents={cfg['num_latents']} but encoder has {F} rows")
        if "d_in" in cfg and int(cfg["d_in"]) != d:
            raise ValueError(f"cfg d_in={cfg['d_in']} but encoder has {d} columns")
        meta = dict(cfg)
        meta.update(extra)
        sae = cls.from_tensors(W_enc, b_enc, W_dec, b_dec, k, cfg=meta)
        return sae.to(torch.device(device)).eval()

    # ---- forward -----------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x [..., d] -> (top_acts [..., k], top_idx [..., k], pre [..., F])``.

        ``pre = (x - b_dec) @ W_enc.T + b_enc`` (returned pre-ReLU); the TopK selection is
        made on ``relu(pre)`` and ``top_acts`` are the selected post-ReLU values."""
        x = x.to(self.W_enc.device, self.W_enc.dtype)
        pre = (x - self.b_dec) @ self.W_enc.T + self.b_enc
        acts = torch.relu(pre)
        top_acts, top_idx = torch.topk(acts, self.k, dim=-1)
        return top_acts, top_idx, pre

    def decode(self, top_acts: torch.Tensor, top_idx: torch.Tensor) -> torch.Tensor:
        """``x_hat = sum_j top_acts_j * W_dec[top_idx_j] + b_dec``."""
        rows = self.W_dec[top_idx]                                     # [..., k, d]
        x_hat = (top_acts.to(rows.dtype)[..., None] * rows).sum(-2) + self.b_dec
        return x_hat

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Reconstruct ``x`` and return ``(x_hat, fvu)`` with
        ``fvu = ||x - x_hat||^2 / ||x - mean(x)||^2`` summed over all samples."""
        top_acts, top_idx, _ = self.encode(x)
        x_hat = self.decode(top_acts, top_idx)
        return x_hat, fvu(x.to(x_hat.device, x_hat.dtype), x_hat)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Dense sparse-code ``[..., F]`` (zeros outside the top-k). Memory ``N x F``."""
        top_acts, top_idx, pre = self.encode(x)
        dense = torch.zeros_like(pre)
        dense.scatter_(-1, top_idx, top_acts)
        return dense

    def extra_repr(self) -> str:
        return f"d_in={self.d_in}, num_latents={self.num_latents}, k={self.k}"


# --------------------------------------------------------------------------------------
# metrics / feature analysis
# --------------------------------------------------------------------------------------
def fvu(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    """Fraction of variance unexplained, totals over all samples:
    ``sum ||x - x_hat||^2 / sum ||x - mean(x)||^2`` (mean over samples)."""
    x = x.detach().float().reshape(-1, x.shape[-1])
    x_hat = x_hat.detach().float().reshape(-1, x_hat.shape[-1])
    resid = (x - x_hat).pow(2).sum()
    total = (x - x.mean(0, keepdim=True)).pow(2).sum()
    return float(resid / total.clamp_min(_EPS))


def _chunks(acts: torch.Tensor, chunk: int):
    for i in range(0, acts.shape[0], chunk):
        yield acts[i:i + chunk]


def sae_fvu(sae: TopKSAE, acts: torch.Tensor, chunk: int = 1024) -> float:
    """FVU of ``sae`` on ``acts [N,d]`` (evaluated in chunks; the mean is taken over all N)."""
    X = torch.as_tensor(acts).detach().float().reshape(-1, acts.shape[-1])
    mu = X.mean(0, keepdim=True)
    total = (X - mu).pow(2).sum()
    resid = 0.0
    with torch.no_grad():
        for xb in _chunks(X, chunk):
            top_acts, top_idx, _ = sae.encode(xb)
            x_hat = sae.decode(top_acts, top_idx).float().cpu()
            resid += float((xb - x_hat).pow(2).sum())
    return float(resid / float(total.clamp_min(_EPS)))


def feature_table(sae: TopKSAE, acts: torch.Tensor, labels: Sequence[str],
                  chunk: int = 1024) -> pd.DataFrame:
    """Per-feature activation statistics by language label.

    Columns: ``feature``, ``rate_E``/``rate_D`` (fraction of samples of that label in which
    the feature is active, i.e. selected by TopK with a positive value), ``mean_E``/``mean_D``
    (mean activation over all samples of that label, zeros included) and
    ``selectivity = log((rate_D + 1e-6) / (rate_E + 1e-6))`` (> 0: German-selective).
    Samples with other labels are ignored. ``df.attrs`` holds ``n_E`` and ``n_D``.
    """
    X = torch.as_tensor(acts).detach().float().reshape(-1, acts.shape[-1])
    lab = np.asarray(list(labels))
    if lab.shape[0] != X.shape[0]:
        raise ValueError(f"{X.shape[0]} activations but {lab.shape[0]} labels")
    F = sae.num_latents
    stats = {}
    for lang in ("E", "D"):
        mask = torch.as_tensor(lab == lang)
        Xl = X[mask]
        cnt = torch.zeros(F, dtype=torch.float64)
        tot = torch.zeros(F, dtype=torch.float64)
        with torch.no_grad():
            for xb in _chunks(Xl, chunk):
                if xb.shape[0] == 0:
                    continue
                top_acts, top_idx, _ = sae.encode(xb)
                top_acts = top_acts.float().cpu().double()
                top_idx = top_idx.cpu().reshape(-1)
                active = (top_acts > 0).double().reshape(-1)
                cnt.index_add_(0, top_idx, active)
                tot.index_add_(0, top_idx, top_acts.reshape(-1))
        n = int(Xl.shape[0])
        stats[lang] = (cnt / max(n, 1), tot / max(n, 1), n)
    rate_E, mean_E, n_E = stats["E"]
    rate_D, mean_D, n_D = stats["D"]
    sel = torch.log((rate_D + 1e-6) / (rate_E + 1e-6))
    df = pd.DataFrame(dict(feature=np.arange(F, dtype=np.int64), rate_E=rate_E.numpy(), rate_D=rate_D.numpy(),
                           mean_E=mean_E.numpy(), mean_D=mean_D.numpy(), selectivity=sel.numpy()))
    df.attrs["n_E"] = n_E
    df.attrs["n_D"] = n_D
    return df


def language_selective_features(table: pd.DataFrame, n: int = 10) -> dict:
    """Top-``n`` German-selective (highest selectivity) and English-selective (lowest)
    feature indices among features that are active at least once. Returns
    ``dict(D=[...], E=[...])`` of python ints (the lists are disjoint)."""
    t = table[(table["rate_E"] > 0) | (table["rate_D"] > 0)]
    t = t.sort_values(["selectivity", "feature"], ascending=[False, True], kind="mergesort")
    D = [int(i) for i in t["feature"].head(int(n)).tolist()]
    tE = t[t["selectivity"] < 0].sort_values(["selectivity", "feature"], ascending=[True, True], kind="mergesort")
    E = [int(i) for i in tE["feature"].head(int(n)).tolist() if int(i) not in D]
    if not E:   # no negative selectivity: fall back to the lowest-selectivity active features
        tail = t.sort_values(["selectivity", "feature"], ascending=[True, True], kind="mergesort")
        E = [int(i) for i in tail["feature"].head(int(n)).tolist() if int(i) not in D]
    return dict(D=D, E=E)


def feature_direction(sae: TopKSAE, idx: int) -> torch.Tensor:
    """Unit decoder row ``W_dec[idx]`` as a float32 CPU vector (steering direction)."""
    v = sae.W_dec[int(idx)].detach().float().cpu()
    return v / v.norm().clamp_min(_EPS)


def sae_repo_for(model_name: str) -> Optional[tuple[str, str]]:
    """``(repo, hookpoint_fmt)`` for a Pythia model name from ``config.SAE_REPOS``; also
    accepts a bare name like ``"pythia-160m"``. ``None`` if no SAE is registered."""
    if model_name in SAE_REPOS:
        return SAE_REPOS[model_name]
    short = model_name.split("/")[-1]
    for name, val in SAE_REPOS.items():
        if name.split("/")[-1] == short:
            return val
    return None


__all__ = ["TopKSAE", "fvu", "sae_fvu", "feature_table", "language_selective_features",
           "feature_direction", "sae_repo_for"]
