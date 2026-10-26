"""ethPandaOps Xatu Parquet chunks: URLs, resumable download, normalised reading.

Data source: ethPandaOps Xatu (https://ethpandaops.io/posts/xatu-execution-layer/),
licence CC BY 4.0, attribution "ethPandaOps Xatu".  One Parquet file per table and
per 1 000 blocks::

    https://data.ethpandaops.io/xatu/mainnet/databases/default/canonical_execution_<table>/1000/<start>.parquet

Normalisation done by :func:`read_chunk` (see docs/TRACE_SOURCE_SPEC.md §0 and §1):

* every binary string is decoded to ``str`` and lower-cased (``native_transfers`` stores
  hashes and addresses in upper-case hex); empty strings become ``None``;
* ``fixed_size_binary[32]`` values (``value``, ``from_value``/``to_value`` of the balance
  table) are uint256 **little-endian** and become Python ``int``;
* ``fixed_size_binary[66]`` hashes are ASCII ``"0x…"``; block-level rows (rewards, the
  fork's irregular state change) carry ``"0x"`` followed by 64 *zero bytes*; they are
  normalised to ``"0x" + "0" * 64`` and recognised by :func:`is_block_level`;
* the trace column ``action_value`` is a DECIMAL string and becomes ``int``;
* ``trace_address`` ``None``/``""`` (root) becomes ``()``, ``"1_4_0"`` becomes ``(1, 4, 0)``;
* storage values, slots, topics, inputs stay lower-case hex strings.

The local cache uses the flat layout ``<cache_dir>/canonical_execution_<table>_<start>.parquet``.
"""
from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Optional, Sequence, Union

import pandas as pd

XATU_BASE = "https://data.ethpandaops.io/xatu/mainnet/databases/default"
TABLES = ("traces", "transaction", "native_transfers", "erc20_transfers", "logs", "storage_diffs",
          "storage_reads", "block")
PREFIX = "canonical_execution_"
CHUNK = 1000
ZERO_HASH = "0x" + "0" * 64
PARQUET_MAGIC = b"PAR1"

_TABLE_RE = re.compile(r"^[a-z0-9_]+$")


# ======================================================================================
# names and URLs
# ======================================================================================
def _table_name(table: str) -> str:
    """``"traces"`` or ``"canonical_execution_traces"`` -> ``"traces"`` (validated)."""
    t = str(table).strip()
    if t.startswith(PREFIX):
        t = t[len(PREFIX):]
    if not _TABLE_RE.match(t):
        raise ValueError(f"invalid Xatu table name {table!r}")
    return t


def _check_start(start: int) -> int:
    s = int(start)
    if s < 0 or s % CHUNK:
        raise ValueError(f"chunk start must be a non-negative multiple of {CHUNK}, got {start}")
    return s


def chunk_url(table: str, start: int, base_url: str = XATU_BASE) -> str:
    """URL of one 1 000-block chunk; ``table`` without the ``canonical_execution_`` prefix."""
    return f"{base_url.rstrip('/')}/{PREFIX}{_table_name(table)}/{CHUNK}/{_check_start(start)}.parquet"


def chunk_path(cache_dir: str, table: str, start: int) -> str:
    """Local cache path ``<cache_dir>/canonical_execution_<table>_<start>.parquet``."""
    return os.path.join(str(cache_dir), f"{PREFIX}{_table_name(table)}_{_check_start(start)}.parquet")


def chunk_starts(block_from: int, block_to: int) -> list[int]:
    """Multiples of 1 000 whose chunks cover the half-open block range ``[block_from, block_to)``."""
    lo, hi = int(block_from), int(block_to)
    if lo < 0:
        raise ValueError("block_from must be >= 0")
    if hi <= lo:
        return []
    return list(range(lo // CHUNK * CHUNK, hi, CHUNK))


def _as_table_list(tables) -> list[str]:
    if isinstance(tables, str):
        tables = [t for t in tables.split(",") if t.strip()]
    out = [_table_name(t) for t in tables]
    if not out:
        raise ValueError("no tables given")
    return out


# ======================================================================================
# download
# ======================================================================================
def _valid_parquet_file(path: str) -> bool:
    try:
        size = os.path.getsize(path)
        if size < 12:
            return False
        with open(path, "rb") as fh:
            head = fh.read(4)
            fh.seek(-4, os.SEEK_END)
            tail = fh.read(4)
        return head == PARQUET_MAGIC and tail == PARQUET_MAGIC
    except OSError:
        return False


class _PermanentError(Exception):
    pass


def _fetch_one(url: str, path: str, retries: int, timeout: float, backoff: float,
               session=None, sleep: Callable[[float], None] = time.sleep) -> tuple[int, Optional[str]]:
    """Download ``url`` to ``path`` atomically (tmp file + rename); returns (bytes, error)."""
    import requests

    getter = session.get if session is not None else requests.get
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    last_err: Optional[str] = None
    for attempt in range(int(retries) + 1):
        try:
            with getter(url, stream=True, timeout=timeout) as resp:
                if resp.status_code in (400, 401, 403, 404, 410):
                    raise _PermanentError(f"HTTP {resp.status_code}")
                if resp.status_code != 200:
                    raise IOError(f"HTTP {resp.status_code}")
                n = 0
                with open(tmp, "wb") as fh:
                    for block in resp.iter_content(chunk_size=1 << 20):
                        if block:
                            fh.write(block)
                            n += len(block)
                expected = resp.headers.get("Content-Length")
                if expected is not None and expected.isdigit() and int(expected) != n:
                    raise IOError(f"truncated download ({n} of {expected} bytes)")
            if not _valid_parquet_file(tmp):
                raise IOError("downloaded file is not a complete Parquet file")
            os.replace(tmp, path)
            return n, None
        except _PermanentError as exc:
            last_err = str(exc)
            break
        except Exception as exc:  # network errors, truncation, invalid file: retry
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < int(retries):
                sleep(backoff * (2 ** attempt))
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
    return 0, last_err


def download_chunks(tables, block_from: int, block_to: int, cache_dir: str = "cache/xatu",
                    every: int = 1, max_workers: int = 4, retries: int = 4,
                    base_url: str = XATU_BASE, timeout: float = 120.0, backoff: float = 1.0,
                    session=None, log_fn: Optional[Callable[[str], None]] = None) -> pd.DataFrame:
    """Download the chunks of ``tables`` covering ``[block_from, block_to)`` into ``cache_dir``.

    * resumable: existing non-empty files (complete Parquet magic at both ends) are skipped;
    * atomic: each file is written to a temporary name and renamed when complete;
    * exponential back-off: ``backoff * 2**attempt`` seconds between ``retries`` retries;
      HTTP 4xx (e.g. 404 for a chunk that does not exist) is not retried;
    * ``every=k`` takes every k-th chunk start (sampling across a long range).

    Returns ``DataFrame(table, start, path, bytes, ok, error, cached)``.
    """
    tabs = _as_table_list(tables)
    every = int(every)
    if every < 1:
        raise ValueError("every must be >= 1")
    starts = chunk_starts(block_from, block_to)[::every]
    os.makedirs(cache_dir, exist_ok=True)
    jobs = [(t, s) for t in tabs for s in starts]

    def work(job):
        t, s = job
        path = chunk_path(cache_dir, t, s)
        if os.path.exists(path) and os.path.getsize(path) > 0 and _valid_parquet_file(path):
            return dict(table=t, start=s, path=path, bytes=os.path.getsize(path), ok=True, error=None,
                        cached=True)
        n, err = _fetch_one(chunk_url(t, s, base_url), path, retries, timeout, backoff, session=session)
        if log_fn is not None:
            log_fn(f"xatu {t} {s}: " + ("ok" if err is None else err))
        return dict(table=t, start=s, path=path, bytes=n, ok=err is None, error=err, cached=False)

    if not jobs:
        rows = []
    elif int(max_workers) <= 1 or len(jobs) == 1:
        rows = [work(j) for j in jobs]
    else:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            rows = list(ex.map(work, jobs))
    cols = ["table", "start", "path", "bytes", "ok", "error", "cached"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df = df.sort_values(["table", "start"], kind="stable").reset_index(drop=True)
    return df


# ======================================================================================
# normalised reading
# ======================================================================================
def _to_str(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray, memoryview)):
        v = bytes(v).decode("utf-8", errors="replace")
    s = str(v)
    return s.lower() if s != "" else None


def normalize_hash(v) -> Optional[str]:
    """ASCII ``"0x…"`` hash (any case, bytes or str) -> lower-case str; block-level -> ZERO_HASH."""
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray, memoryview)):
        v = bytes(v).decode("latin-1")
    s = str(v).lower()
    if is_block_level(s):
        return ZERO_HASH
    return s


def normalize_address(v) -> Optional[str]:
    """Hex address (bytes or str, any case) -> lower-case str (``None`` for missing / empty)."""
    return _to_str(v)


def le_uint(v) -> Optional[int]:
    """uint256 little-endian ``fixed_binary[32]`` -> int (ints / decimal or hex strings pass)."""
    if v is None:
        return None
    if isinstance(v, int):
        return int(v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return int.from_bytes(bytes(v), "little")
    return parse_int(v)


def parse_int(v) -> Optional[int]:
    """Decimal (or ``0x`` hex) string / bytes -> int; ints pass; empty -> None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, (bytes, bytearray, memoryview)):
        v = bytes(v).decode("ascii", errors="strict")
    s = str(v).strip().lower()
    if s == "":
        return None
    if s.startswith("0x"):
        return int(s[2:] or "0", 16)
    return int(s)


def parse_trace_address(v) -> tuple:
    """``None`` / ``""`` -> ``()`` (root); ``"1_4_0"`` -> ``(1, 4, 0)``; tuples/lists pass."""
    if v is None:
        return ()
    if isinstance(v, (tuple, list)):
        return tuple(int(x) for x in v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        v = bytes(v).decode("ascii")
    s = str(v).strip()
    if s == "":
        return ()
    return tuple(int(x) for x in s.split("_"))


def is_block_level(tx_hash) -> bool:
    """True for block-level rows: ``"0x"`` followed by zero bytes (or ASCII zeros), or no hash."""
    if tx_hash is None:
        return True
    if isinstance(tx_hash, (bytes, bytearray, memoryview)):
        tx_hash = bytes(tx_hash).decode("latin-1")
    s = str(tx_hash)
    body = s[2:] if s[:2].lower() == "0x" else s
    return all(c in "0\x00" for c in body)


def _converter(name: str, typ) -> Optional[Callable]:
    import pyarrow as pa

    if pa.types.is_fixed_size_binary(typ):
        if typ.byte_width == 32:
            return le_uint
        return normalize_hash
    if pa.types.is_binary(typ) or pa.types.is_large_binary(typ) or pa.types.is_string(typ) \
            or pa.types.is_large_string(typ):
        if name == "action_value":
            return parse_int
        if name == "trace_address":
            return parse_trace_address
        if name in ("transaction_hash", "block_hash"):
            return normalize_hash
        return _to_str
    return None


def read_parquet_rows(path: str, columns: Optional[Sequence[str]] = None) -> list[dict]:
    """Read one Xatu Parquet file into normalised Python rows (see module docstring)."""
    import pyarrow.parquet as pq

    tbl = pq.read_table(path, columns=list(columns) if columns is not None else None)
    names, cols = [], []
    for fld in tbl.schema:
        vals = tbl.column(fld.name).to_pylist()
        conv = _converter(fld.name, fld.type)
        names.append(fld.name)
        cols.append([conv(v) for v in vals] if conv is not None else vals)
    if not names:
        return []
    return [dict(zip(names, row)) for row in zip(*cols)]


def read_chunk(table: str, start: int, cache_dir: str = "cache/xatu",
               columns: Optional[Sequence[str]] = None) -> list[dict]:
    """Rows of one cached chunk with normalised fields (see module docstring)."""
    path = chunk_path(cache_dir, table, start)
    if not os.path.exists(path):
        raise FileNotFoundError(f"chunk not in cache: {path} (run download_chunks first)")
    return read_parquet_rows(path, columns=columns)


def available_tables(start: int, cache_dir: str = "cache/xatu",
                     tables: Iterable[str] = TABLES) -> list[str]:
    """Tables whose chunk ``start`` is present in ``cache_dir``."""
    return [_table_name(t) for t in tables if os.path.exists(chunk_path(cache_dir, t, start))]


def load_chunk_rows(start: int, cache_dir: str = "cache/xatu",
                    tables: Union[str, Iterable[str]] = TABLES) -> dict[str, list[dict]]:
    """``{table: rows}`` for every table of chunk ``start`` present in the cache (missing skipped);
    the input format of :func:`lsne.chain.itr.build_records`."""
    tabs = _as_table_list(tables)
    return {t: read_chunk(t, start, cache_dir) for t in available_tables(start, cache_dir, tabs)}


def cached_starts(cache_dir: str = "cache/xatu", table: str = "traces") -> list[int]:
    """Sorted chunk starts of ``table`` present in ``cache_dir``."""
    pat = re.compile(rf"^{PREFIX}{_table_name(table)}_(\d+)\.parquet$")
    if not os.path.isdir(cache_dir):
        return []
    out = []
    for fn in os.listdir(cache_dir):
        m = pat.match(fn)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)
