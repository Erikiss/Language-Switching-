"""Fetch individual training sequences of the Pythia (standard, preshuffled) Pile by
HTTP range request, without downloading the 600 GB dataset.

Layout facts (verified 2026-10-25): the dataset is 21 shards `document-00000-of-00020.bin`
... `document-00020-of-00020.bin`; shards 0..19 are exactly 30_000_000_000 bytes, the
last one 78_336_000 bytes; total = 600_078_336_000 bytes = 300_039_168_000 uint16
tokens = 146_432_000 sequences x 2049 tokens = 143_000 batches x 1024 sequences.
The `.bin` files are raw little-endian uint16 token streams (MMapIndexedDataset data
file, no header), so sequence i starts at byte 2*2049*i of the concatenated stream.

Safety of the range requests (:func:`fetch_range`): the request is streamed and the body
is read only after the response proves that the server honoured the ``Range`` header
(HTTP 206, ``Content-Range`` parsed and matched when present; or HTTP 200 with a
``Content-Length`` equal to the requested length). Any other answer, in particular a 200
without a matching length (= the whole 30 GB shard), closes the connection and raises
before a single payload byte is downloaded.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import requests

from .config import STATE2143

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{fname}"

_CHUNK = 1 << 16
_CONTENT_RANGE_RE = re.compile(r"\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*", re.IGNORECASE)


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


def parse_content_range(value: Optional[str]) -> Optional[tuple[int, int, Optional[int]]]:
    """``"bytes a-b/total"`` -> ``(a, b, total)`` (``total`` is ``None`` for ``*``);
    ``None`` when the header is missing or malformed."""
    if value is None:
        return None
    m = _CONTENT_RANGE_RE.fullmatch(value)
    if not m:
        return None
    total = None if m.group(3) == "*" else int(m.group(3))
    return int(m.group(1)), int(m.group(2)), total


def _int_header(headers, name: str) -> Optional[int]:
    v = headers.get(name) if headers is not None else None
    if v is None:
        return None
    try:
        return int(str(v).strip())
    except ValueError:
        return None


def fetch_range(url: str, start: int, end_inclusive: int, timeout: float = 120.0) -> bytes:
    """GET ``bytes=start-end_inclusive`` of ``url`` as a *streamed* request and return
    exactly those bytes.

    The payload is read only after the response headers show that the server honoured the
    ``Range`` header:

    * HTTP 206: ``Content-Range`` (when present) must be ``bytes start-end/...`` for the
      requested offsets and ``Content-Length`` (when present) must equal the requested
      length;
    * HTTP 200: accepted only with a ``Content-Length`` equal to the requested length
      (some mirrors answer a satisfiable range this way);
    * anything else, including a 200 with another or no ``Content-Length`` (the whole
      shard) and redirect/4xx/5xx errors: the connection is closed and ``RuntimeError``
      is raised without downloading the body.

    The body is then read in chunks; a short body, or a server that keeps sending beyond
    the requested length (detected after at most one extra 64 KB chunk), raises
    ``RuntimeError``. The response is always closed.
    """
    start, end_inclusive = int(start), int(end_inclusive)
    n = end_inclusive - start + 1
    if n <= 0:
        raise ValueError(f"empty byte range {start}-{end_inclusive}")
    headers = _hf_headers()
    headers["Range"] = f"bytes={start}-{end_inclusive}"
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
    try:
        status = int(r.status_code)
        if status == 206:
            cr = r.headers.get("Content-Range")
            if cr is not None:
                parsed = parse_content_range(cr)
                if parsed is None or parsed[0] != start or parsed[1] != end_inclusive:
                    raise RuntimeError(f"HTTP 206 with unexpected Content-Range {cr!r} for "
                                       f"bytes={start}-{end_inclusive} of {url}")
            cl = _int_header(r.headers, "Content-Length")
            if cl is not None and cl != n:
                raise RuntimeError(f"HTTP 206 with Content-Length {cl} for a {n}-byte range of {url}")
        elif status == 200:
            cl = _int_header(r.headers, "Content-Length")
            if cl != n:
                raise RuntimeError(f"server ignored the Range header (HTTP 200, Content-Length "
                                   f"{r.headers.get('Content-Length')!r} for a {n}-byte range) for {url}; "
                                   f"refusing to download the full shard")
        else:
            raise RuntimeError(f"range request failed: HTTP {status} for {url}")
        buf = bytearray()
        for chunk in r.iter_content(chunk_size=min(n, _CHUNK)):   # ends after Content-Length bytes
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > n:
                raise RuntimeError(f"server sent more than the {n} requested bytes for {url}")
        if len(buf) != n:
            raise RuntimeError(f"short read: {len(buf)} of {n} bytes for bytes={start}-{end_inclusive} of {url}")
        return bytes(buf)
    finally:
        r.close()


def fetch_sequence(seq_index: int, cache_dir: Optional[str | Path] = None,
                   repo: str = STATE2143["dataset_repo"], timeout: float = 120.0) -> np.ndarray:
    """Fetch one 2049-token training sequence as np.uint16 (little-endian decoded).

    One streamed range request per shard the sequence touches (two when it straddles a
    shard boundary), each validated by :func:`fetch_range` before its body is read."""
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"seq_{seq_index}.npy"
        if cp.exists():
            return np.load(cp)
    buf = bytearray()
    for shard, a, b in byte_range_for_sequence(seq_index):
        url = HF_RESOLVE.format(repo=repo, fname=STATE2143["shard_fmt"].format(i=shard))
        buf.extend(fetch_range(url, a, b, timeout=timeout))
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
