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

Transient failures (HTTP 429 / 5xx, a short body, connection errors and timeouts) are
retried :data:`FETCH_ATTEMPTS` times in total with exponential backoff (``Retry-After``
honoured on 429/503); a refused range, a wrong ``Content-Range`` or a 404 is raised at once.
:func:`sample_unigram_counts` fetches its sequences concurrently (one ``requests.Session``
per worker thread), keeps a resumable progress file and writes its cache atomically.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from requests.exceptions import ChunkedEncodingError as _ChunkedEncodingError
from requests.exceptions import ConnectionError as _ConnectionError
from requests.exceptions import Timeout as _Timeout

from .config import STATE2143

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/{fname}"

_CHUNK = 1 << 16
_CONTENT_RANGE_RE = re.compile(r"\s*bytes\s+(\d+)-(\d+)/(\d+|\*)\s*", re.IGNORECASE)
#: Attempts per range request (the first one included) for transient failures.
FETCH_ATTEMPTS = 5
#: Backoff before attempt k + 1: ``BACKOFF_BASE_S * 2**k`` s (1, 2, 4, 8 s), at least the
#: server's ``Retry-After``, at most ``BACKOFF_MAX_S``.
BACKOFF_BASE_S = 1.0
BACKOFF_MAX_S = 60.0
#: Worker threads of :func:`sample_unigram_counts`.
UNIGRAM_WORKERS = 8
#: Sequences between two writes of the unigram progress file.
UNIGRAM_CHECKPOINT_EVERY = 200
_sleep = time.sleep          # module attribute so that tests can skip the backoff


class TransientFetchError(RuntimeError):
    """HTTP 429 / 5xx or a short body: worth another attempt (``retry_after`` in s or None)."""

    def __init__(self, msg: str, retry_after: Optional[float] = None):
        super().__init__(msg)
        self.retry_after = retry_after


def _retry_after(headers) -> Optional[float]:
    v = headers.get("Retry-After") if headers is not None else None
    try:
        return max(0.0, float(str(v).strip())) if v is not None else None
    except ValueError:          # HTTP-date form: fall back to the exponential backoff
        return None


def _savez_atomic(path: Path, **arrays) -> None:
    """``np.savez`` to a temporary file next to ``path``, then ``os.replace`` (a reader never
    sees a half-written file, also with several notebooks on one Drive)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


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


def fetch_range(url: str, start: int, end_inclusive: int, timeout: float = 120.0,
                session: Optional[requests.Session] = None, attempts: Optional[int] = None) -> bytes:
    """GET ``bytes=start-end_inclusive`` of ``url`` as a *streamed* request and return
    exactly those bytes (through ``session`` when given, else ``requests.get``).

    Transient failures - HTTP 429 / 5xx, a short body, ``requests`` connection errors,
    timeouts and broken chunked bodies (also while the body is read, which urllib3's
    ``Retry`` cannot cover for a streamed response) - are retried up to ``attempts`` (default :data:`FETCH_ATTEMPTS`)
    attempts in total with exponential backoff; every other failure below is raised at once.
    Each attempt is one :func:`_fetch_range_once`, which reads the body only after the
    headers prove that the Range header was honoured.
    """
    n_att = FETCH_ATTEMPTS if attempts is None else max(1, int(attempts))
    for k in range(n_att):
        try:
            return _fetch_range_once(url, start, end_inclusive, timeout, session)
        except (TransientFetchError, _ConnectionError, _Timeout, _ChunkedEncodingError) as e:
            if k == n_att - 1:
                raise
            wait = BACKOFF_BASE_S * (2 ** k)
            ra = getattr(e, "retry_after", None)
            if ra is not None:
                wait = max(wait, float(ra))
            _sleep(min(wait, BACKOFF_MAX_S))
    raise AssertionError("unreachable")  # pragma: no cover


def _fetch_range_once(url: str, start: int, end_inclusive: int, timeout: float = 120.0,
                      session: Optional[requests.Session] = None) -> bytes:
    """One attempt of :func:`fetch_range`: GET ``bytes=start-end_inclusive`` of ``url`` as a
    *streamed* request and return exactly those bytes.

    The payload is read only after the response headers show that the server honoured the
    ``Range`` header:

    * HTTP 206: ``Content-Range`` (when present) must be ``bytes start-end/...`` for the
      requested offsets and ``Content-Length`` (when present) must equal the requested
      length;
    * HTTP 200: accepted only with a ``Content-Length`` equal to the requested length
      (some mirrors answer a satisfiable range this way);
    * anything else, including a 200 with another or no ``Content-Length`` (the whole
      shard) and redirect/4xx/5xx errors: the connection is closed and ``RuntimeError``
      is raised without downloading the body (:class:`TransientFetchError` for 429 / 5xx).

    The body is then read in chunks; a short body (:class:`TransientFetchError`), or a
    server that keeps sending beyond the requested length (detected after at most one extra
    64 KB chunk), raises ``RuntimeError``. The response is always closed.
    """
    start, end_inclusive = int(start), int(end_inclusive)
    n = end_inclusive - start + 1
    if n <= 0:
        raise ValueError(f"empty byte range {start}-{end_inclusive}")
    headers = _hf_headers()
    headers["Range"] = f"bytes={start}-{end_inclusive}"
    get = session.get if session is not None else requests.get
    r = get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=True)
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
        elif status == 429 or 500 <= status < 600:
            raise TransientFetchError(f"range request failed: HTTP {status} for {url}", _retry_after(r.headers))
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
            raise TransientFetchError(f"short read: {len(buf)} of {n} bytes for bytes={start}-{end_inclusive} of {url}")
        return bytes(buf)
    finally:
        r.close()


def fetch_sequence(seq_index: int, cache_dir: Optional[str | Path] = None,
                   repo: str = STATE2143["dataset_repo"], timeout: float = 120.0,
                   session: Optional[requests.Session] = None) -> np.ndarray:
    """Fetch one 2049-token training sequence as np.uint16 (little-endian decoded).

    One streamed range request per shard the sequence touches (two when it straddles a
    shard boundary), each validated (and retried on transient errors) by
    :func:`fetch_range` before its body is read.  The ``cache_dir`` file is written
    atomically."""
    cache_dir = Path(cache_dir) if cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cp = cache_dir / f"seq_{seq_index}.npy"
        if cp.exists():
            return np.load(cp)
    buf = bytearray()
    for shard, a, b in byte_range_for_sequence(seq_index):
        url = HF_RESOLVE.format(repo=repo, fname=STATE2143["shard_fmt"].format(i=shard))
        buf.extend(fetch_range(url, a, b, timeout=timeout, session=session))
    toks = np.frombuffer(bytes(buf), dtype="<u2").astype(np.uint16)
    if toks.shape[0] != STATE2143["seq_len"]:
        raise RuntimeError(f"expected {STATE2143['seq_len']} tokens, got {toks.shape[0]}")
    if cache_dir:
        _save_npy_atomic(cache_dir / f"seq_{seq_index}.npy", toks)
    return toks


def sha256_tokens(tokens: np.ndarray) -> str:
    """SHA256 over little-endian uint16 bytes (dossier convention)."""
    return hashlib.sha256(np.asarray(tokens, dtype="<u2").tobytes()).hexdigest()


def sample_unigram_counts(n_sequences: int = 2000, seed: int = 0, vocab: int = 50304,
                          cache_dir: Optional[str | Path] = None,
                          n_total_sequences: int = 146_432_000, workers: int = UNIGRAM_WORKERS,
                          checkpoint_every: int = UNIGRAM_CHECKPOINT_EVERY) -> np.ndarray:
    """Unigram token counts from `n_sequences` uniformly sampled training sequences
    (range requests; ~4 KB each). Cached as npz keyed by (n, seed).

    The sequences are fetched by ``workers`` threads, one ``requests.Session`` each, and
    every range request is retried on transient errors (:func:`fetch_range`); the int64
    counts are summed in the main thread, so they do not depend on the completion order
    and equal those of a sequential run.  With ``cache_dir`` the counts so far are kept in
    ``unigram_n<n>_s<seed>.partial.npz`` (``counts``, ``done_idx``, ``idx``; rewritten every
    ``checkpoint_every`` sequences and when a fetch finally fails), so a restart fetches only
    the missing sequences; the final npz is written atomically and the progress file removed."""
    cache_dir = Path(cache_dir) if cache_dir else None
    cp = prog = None
    if cache_dir:
        cp = cache_dir / f"unigram_n{n_sequences}_s{seed}.npz"
        if cp.exists():
            return np.load(cp)["counts"]
        prog = cache_dir / f"unigram_n{n_sequences}_s{seed}.partial.npz"
    rng = np.random.default_rng(seed)
    idx = rng.choice(n_total_sequences, size=n_sequences, replace=False)
    counts = np.zeros(vocab, dtype=np.int64)
    done: set[int] = set()
    if prog is not None and prog.exists():
        try:
            with np.load(prog) as z:
                if np.array_equal(z["idx"], idx) and z["counts"].shape == (vocab,):
                    counts = np.asarray(z["counts"], dtype=np.int64).copy()
                    done = {int(i) for i in z["done_idx"]}
        except Exception:  # noqa: BLE001 - unreadable progress file: start over
            counts, done = np.zeros(vocab, dtype=np.int64), set()
    todo = [int(i) for i in idx if int(i) not in done]

    def save_progress():
        if prog is not None:
            _savez_atomic(prog, counts=counts, done_idx=np.asarray(sorted(done), dtype=np.int64), idx=idx)

    local = threading.local()
    sessions: list = []
    lock = threading.Lock()

    def one(i: int):
        s = getattr(local, "session", None)
        if s is None:
            s = local.session = requests.Session()
            with lock:
                sessions.append(s)
        return i, fetch_sequence(i, cache_dir=None, session=s)

    ex = ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="unigram")
    n_new = 0
    try:
        for i, toks in ex.map(one, todo):            # in submission order; raises the first failure
            counts += np.bincount(toks.astype(np.int64), minlength=vocab)[:vocab]
            done.add(i)
            n_new += 1
            if n_new % max(1, int(checkpoint_every)) == 0:
                save_progress()
    except BaseException:
        ex.shutdown(wait=True, cancel_futures=True)  # running fetches end, queued ones are dropped
        if n_new:
            save_progress()
        raise
    finally:
        ex.shutdown(wait=True)
        for s in sessions:
            try:
                s.close()
            except Exception:  # noqa: BLE001
                pass
    if cp is not None:
        _savez_atomic(cp, counts=counts, idx=idx)
        if prog is not None and prog.exists():
            prog.unlink()
    return counts


def frequent_token_ids(counts: np.ndarray, top_n: int = 100) -> np.ndarray:
    return np.argsort(-counts)[:top_n]
