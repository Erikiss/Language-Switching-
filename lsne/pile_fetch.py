"""Fetch individual training sequences of the Pythia (standard, preshuffled) Pile by
HTTP range request, without downloading the 600 GB dataset.

Layout facts (verified 2026-10-25): the dataset is 21 shards `document-00000-of-00020.bin`
... `document-00020-of-00020.bin`; shards 0..19 are exactly 30_000_000_000 bytes, the
last one 78_336_000 bytes; total = 600_078_336_000 bytes = 300_039_168_000 uint16
tokens = 146_432_000 sequences x 2049 tokens = 143_000 batches x 1024 sequences.
The `.bin` files are raw little-endian uint16 token streams (MMapIndexedDataset data
file, no header), so sequence i starts at byte 2*2049*i of the concatenated stream.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .config import STATE2143

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{fname}"


def _hf_headers() -> dict:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    h = {"User-Agent": "lsne/0.1"}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def byte_range_for_sequence(seq_index: int, seq_len: int = STATE2143["seq_len"],
                            shard_bytes: int = STATE2143["shard_bytes"]) -> list[tuple[int, int, int]]:
    """Return a list of (shard, start, end_inclusive) byte ranges covering the sequence
    (two entries if the sequence straddles a shard boundary)."""
    start = seq_index * seq_len * 2
    end = start + seq_len * 2 - 1
    s0, l0 = divmod(start, shard_bytes)
    s1, l1 = divmod(end, shard_bytes)
    if s0 == s1:
        return [(s0, l0, l1)]
    return [(s0, l0, shard_bytes - 1), (s1, 0, l1)]


def fetch_sequence(seq_index: int, cache_dir: Optional[str | Path] = None,
                   repo: str = STATE2143["dataset_repo"], timeout: float = 120.0) -> np.ndarray:
    """Fetch one 2049-token training sequence as np.uint16 (little-endian decoded)."""
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"seq_{seq_index}.npy"
        if cp.exists():
            return np.load(cp)
    buf = bytearray()
    for shard, a, b in byte_range_for_sequence(seq_index):
        url = HF_RESOLVE.format(repo=repo, fname=STATE2143["shard_fmt"].format(i=shard))
        headers = _hf_headers()
        headers["Range"] = f"bytes={a}-{b}"
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        if r.status_code not in (200, 206):
            raise RuntimeError(f"range request failed: HTTP {r.status_code} for {url}")
        if r.status_code == 200 and len(r.content) != (b - a + 1):
            raise RuntimeError("server ignored the Range header; refusing to download the full shard")
        buf.extend(r.content)
    toks = np.frombuffer(bytes(buf), dtype="<u2").astype(np.uint16)
    if toks.shape[0] != STATE2143["seq_len"]:
        raise RuntimeError(f"expected {STATE2143['seq_len']} tokens, got {toks.shape[0]}")
    if cache_dir:
        np.save(cache_dir / f"seq_{seq_index}.npy", toks)
    return toks


def sha256_tokens(tokens: np.ndarray) -> str:
    """SHA256 over little-endian uint16 bytes (dossier convention)."""
    return hashlib.sha256(np.asarray(tokens, dtype="<u2").tobytes()).hexdigest()


def sample_unigram_counts(n_sequences: int = 2000, seed: int = 0, vocab: int = 50304,
                          cache_dir: Optional[str | Path] = None,
                          n_total_sequences: int = 146_432_000) -> np.ndarray:
    """Unigram token counts from `n_sequences` uniformly sampled training sequences
    (range requests; ~4 KB each). Cached as npz keyed by (n, seed)."""
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cp = cache_dir / f"unigram_n{n_sequences}_s{seed}.npz"
        if cp.exists():
            return np.load(cp)["counts"]
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total_sequences, size=n_sequences, replace=False)
    counts = np.zeros(vocab, dtype=np.int64)
    for i in idx:
        toks = fetch_sequence(int(i), cache_dir=None)
        counts += np.bincount(toks.astype(np.int64), minlength=vocab)[:vocab]
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache_dir / f"unigram_n{n_sequences}_s{seed}.npz", counts=counts, idx=idx)
    return counts


def frequent_token_ids(counts: np.ndarray, top_n: int = 100) -> np.ndarray:
    return np.argsort(-counts)[:top_n]
