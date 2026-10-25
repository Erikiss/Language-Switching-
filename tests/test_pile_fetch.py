"""Tests for lsne.pile_fetch (round 2, G4 items 1-2).

Offline part: byte ranges of shard-straddling and non-straddling sequences, the streamed
range request against a fake HTTP server over a deterministic "virtual Pile" (the body is
read only after the server proved it honoured the Range header; a 200 for the whole shard
is refused before any payload byte), the sequence / unigram caches, unigram sampling with
n=2. Network part (marked ``network``): the State 2143 anchor by a real range request with
the dossier hash check, and two real sequences for the unigram counts.
"""
import hashlib
import re
import types

import numpy as np
import pytest

import lsne.pile_fetch as PF
from lsne.config import STATE2143
from lsne.pile_fetch import (byte_range_for_sequence, fetch_range, fetch_sequence, frequent_token_ids,
                             parse_content_range, sample_unigram_counts, sha256_tokens)

SEQ_LEN = STATE2143["seq_len"]            # 2049 tokens
SEQ_BYTES = 2 * SEQ_LEN                   # 4098 bytes
SHARD = STATE2143["shard_bytes"]          # 30_000_000_000
N_SEQ = 146_432_000
ANCHOR = STATE2143["sequence_index"]      # 5_025_183 (shard 0, DESIGN section 1)
STRADDLE = SHARD // SEQ_BYTES             # 7_320_644: the sequence crossing the shard 0 | 1 boundary
VOCAB = 50304


# --------------------------------------------------------------------------------------
# virtual Pile + fake HTTP server
# --------------------------------------------------------------------------------------
def virtual_tokens(first: int, n: int) -> np.ndarray:
    """Deterministic virtual token stream: token i = (i * 7919 + 13) % VOCAB, little-endian uint16."""
    i = np.arange(first, first + n, dtype=np.int64)
    return ((i * 7919 + 13) % VOCAB).astype("<u2")


def virtual_bytes(shard: int, a: int, b: int) -> bytes:
    """Bytes a..b (inclusive) of virtual shard ``shard`` (shards are 30 GB slices of the stream)."""
    g0, g1 = shard * SHARD + a, shard * SHARD + b
    t0, t1 = g0 // 2, g1 // 2
    raw = virtual_tokens(t0, t1 - t0 + 1).tobytes()
    off = g0 - 2 * t0
    return raw[off: off + (b - a + 1)]


class _HugeBody:
    """Stands in for a 30 GB body: any read is a test failure."""

    def __len__(self):
        return SHARD

    def __getitem__(self, item):
        raise AssertionError("the full-shard body must never be read")


class FakeResponse:
    def __init__(self, status, headers, body):
        self.status_code = status
        self.headers = headers
        self._body = body
        self.served = 0                 # payload bytes handed out through iter_content
        self.closed = False
        self.content_touched = False

    def iter_content(self, chunk_size=1):
        for i in range(0, len(self._body), chunk_size):
            chunk = self._body[i:i + chunk_size]
            self.served += len(chunk)
            yield chunk

    @property
    def content(self):
        self.content_touched = True
        return self._body

    def close(self):
        self.closed = True


class FakePile:
    """Replacement for ``requests.get``: serves byte ranges of the virtual shards and records
    every request. ``mode`` selects the server behaviour."""

    def __init__(self, mode="206"):
        self.mode = mode
        self.calls = []
        self.responses = []

    def __call__(self, url, headers=None, timeout=None, allow_redirects=True, stream=False, **kw):
        m = re.search(r"document-(\d{5})-of-00020\.bin$", url)
        assert m, url
        shard = int(m.group(1))
        rng = re.fullmatch(r"bytes=(\d+)-(\d+)", headers["Range"])
        a, b = int(rng.group(1)), int(rng.group(2))
        n = b - a + 1
        self.calls.append(dict(shard=shard, a=a, b=b, stream=stream, url=url, timeout=timeout))
        body = virtual_bytes(shard, a, b)
        ok = {"Content-Range": f"bytes {a}-{b}/{SHARD}", "Content-Length": str(n)}
        mode = self.mode
        if mode == "206":
            r = FakeResponse(206, ok, body)
        elif mode == "206-no-content-range":
            r = FakeResponse(206, {"Content-Length": str(n)}, body)
        elif mode == "206-bare":
            r = FakeResponse(206, {}, body)
        elif mode == "200-exact":
            r = FakeResponse(200, {"Content-Length": str(n)}, body)
        elif mode == "200-full":
            r = FakeResponse(200, {"Content-Length": str(SHARD)}, _HugeBody())
        elif mode == "200-no-length":
            r = FakeResponse(200, {}, _HugeBody())
        elif mode == "206-wrong-content-range":
            r = FakeResponse(206, {"Content-Range": f"bytes {a + 2}-{b + 2}/{SHARD}", "Content-Length": str(n)}, body)
        elif mode == "206-wrong-length":
            r = FakeResponse(206, {"Content-Range": ok["Content-Range"], "Content-Length": str(n + 1)}, body)
        elif mode == "short":
            r = FakeResponse(206, ok, body[:-1])
        elif mode == "long":
            r = FakeResponse(206, ok, body + b"\x00" * 5)
        elif mode == "404":
            r = FakeResponse(404, {"Content-Length": "15"}, b"Entry not found")
        else:  # pragma: no cover
            raise ValueError(mode)
        self.responses.append(r)
        return r


@pytest.fixture
def fake(monkeypatch):
    def make(mode="206"):
        srv = FakePile(mode)
        monkeypatch.setattr(PF, "requests", types.SimpleNamespace(get=srv))
        return srv
    return make


def _url(shard: int) -> str:
    return PF.HF_RESOLVE.format(repo=STATE2143["dataset_repo"], fname=STATE2143["shard_fmt"].format(i=shard))


# --------------------------------------------------------------------------------------
# byte ranges
# --------------------------------------------------------------------------------------
def test_byte_range_for_sequence_non_straddling():
    assert byte_range_for_sequence(0) == [(0, 0, SEQ_BYTES - 1)]
    # the anchor: 4098 bytes from byte offset 5 025 183 * 4098 of shard 0 (DESIGN section 1)
    assert byte_range_for_sequence(ANCHOR) == [(0, ANCHOR * SEQ_BYTES, ANCHOR * SEQ_BYTES + SEQ_BYTES - 1)]
    assert ANCHOR * SEQ_BYTES + SEQ_BYTES - 1 < SHARD
    # the sequence just before the straddling one lies entirely in shard 0
    (s, a, b), = byte_range_for_sequence(STRADDLE - 1)
    assert s == 0 and b < SHARD and b - a + 1 == SEQ_BYTES
    # the last sequence ends exactly on the last byte of the short last shard (78 336 000 bytes)
    (s, a, b), = byte_range_for_sequence(N_SEQ - 1)
    assert s == 20 and b == 78_336_000 - 1 and b - a + 1 == SEQ_BYTES


@pytest.mark.parametrize("boundary", [1, 2, 7, 20])
def test_byte_range_for_sequence_straddles_shard_boundary(boundary):
    idx = boundary * SHARD // SEQ_BYTES
    start = idx * SEQ_BYTES
    assert start < boundary * SHARD < start + SEQ_BYTES           # 30e9 is not a multiple of 4098
    ranges = byte_range_for_sequence(idx)
    assert len(ranges) == 2
    (s0, a0, b0), (s1, a1, b1) = ranges
    assert (s0, s1) == (boundary - 1, boundary)
    assert b0 == SHARD - 1 and a1 == 0
    assert a0 == start - (boundary - 1) * SHARD
    assert (b0 - a0 + 1) + (b1 - a1 + 1) == SEQ_BYTES
    if boundary == 1:
        assert idx == STRADDLE == 7_320_644
        assert ranges == [(0, 29_999_999_112, 29_999_999_999), (1, 0, 3209)]


def test_parse_content_range():
    assert parse_content_range("bytes 0-9/30000000000") == (0, 9, 30_000_000_000)
    assert parse_content_range("bytes 5-7/*") == (5, 7, None)
    assert parse_content_range(" BYTES 1-2/3 ") == (1, 2, 3)
    for bad in (None, "", "bytes 0-9", "bytes */10", "0-9/10", "bytes a-b/c"):
        assert parse_content_range(bad) is None


# --------------------------------------------------------------------------------------
# fetch_sequence against the fake server
# --------------------------------------------------------------------------------------
def test_fetch_sequence_straddling_concatenates_two_streamed_requests(fake):
    srv = fake("206")
    toks = fetch_sequence(STRADDLE)
    assert toks.dtype == np.uint16 and toks.shape == (SEQ_LEN,)
    assert np.array_equal(toks, virtual_tokens(STRADDLE * SEQ_LEN, SEQ_LEN).astype(np.uint16))
    assert [(c["shard"], c["a"], c["b"]) for c in srv.calls] == [(0, 29_999_999_112, SHARD - 1), (1, 0, 3209)]
    assert [c["url"] for c in srv.calls] == [_url(0), _url(1)]
    assert STATE2143["dataset_repo"] in srv.calls[0]["url"]
    assert all(c["stream"] is True for c in srv.calls)
    for r in srv.responses:
        assert r.closed and not r.content_touched
    assert [r.served for r in srv.responses] == [888, 3210]
    assert sum(r.served for r in srv.responses) == SEQ_BYTES


def test_fetch_sequence_non_straddling_single_request_and_cache(fake, tmp_path):
    srv = fake("206")
    toks = fetch_sequence(ANCHOR, cache_dir=tmp_path)
    assert len(srv.calls) == 1
    c = srv.calls[0]
    assert (c["shard"], c["a"], c["b"]) == byte_range_for_sequence(ANCHOR)[0] and c["stream"] is True
    assert c["url"] == _url(0)
    assert srv.responses[0].served == SEQ_BYTES and srv.responses[0].closed
    assert toks.dtype == np.uint16 and np.array_equal(toks, virtual_tokens(ANCHOR * SEQ_LEN, SEQ_LEN).astype(np.uint16))
    assert (tmp_path / f"seq_{ANCHOR}.npy").exists()
    again = fetch_sequence(ANCHOR, cache_dir=tmp_path)
    assert len(srv.calls) == 1 and np.array_equal(again, toks)           # served from the cache
    assert sha256_tokens(toks) == hashlib.sha256(toks.astype("<u2").tobytes()).hexdigest()
    # a different repo name goes into the URL
    fetch_sequence(ANCHOR + 1, repo="other/repo")
    assert srv.calls[-1]["url"] == PF.HF_RESOLVE.format(repo="other/repo", fname="document-00000-of-00020.bin")


@pytest.mark.parametrize("mode", ["206", "206-no-content-range", "206-bare", "200-exact"])
def test_fetch_range_accepts_honoured_range_responses(fake, mode):
    srv = fake(mode)
    a, b = 1000, 1000 + SEQ_BYTES - 1
    data = fetch_range(_url(3), a, b)
    assert data == virtual_bytes(3, a, b) and len(data) == SEQ_BYTES
    assert srv.calls[0]["stream"] is True and srv.responses[0].closed and not srv.responses[0].content_touched


@pytest.mark.parametrize("mode, needle", [
    ("200-full", "refusing to download the full shard"),
    ("200-no-length", "refusing to download the full shard"),
    ("206-wrong-content-range", "Content-Range"),
    ("206-wrong-length", "Content-Length"),
    ("404", "HTTP 404"),
])
def test_fetch_range_rejects_before_reading_the_body(fake, mode, needle):
    srv = fake(mode)
    with pytest.raises(RuntimeError, match=needle):
        fetch_range(_url(0), 0, SEQ_BYTES - 1)
    r = srv.responses[0]
    assert r.served == 0 and not r.content_touched and r.closed
    # the same through fetch_sequence (no cache file is written)
    with pytest.raises(RuntimeError, match=needle):
        fetch_sequence(ANCHOR)
    assert all(x.served == 0 and x.closed for x in srv.responses)


@pytest.mark.parametrize("mode, needle", [("short", "short read"), ("long", "more than")])
def test_fetch_range_rejects_short_and_long_bodies(fake, mode, needle):
    srv = fake(mode)
    with pytest.raises(RuntimeError, match=needle):
        fetch_range(_url(0), 0, 99)
    assert srv.responses[0].closed
    with pytest.raises(RuntimeError, match=needle):
        fetch_sequence(ANCHOR)


def test_fetch_range_empty_range_is_an_error(fake):
    fake("206")
    with pytest.raises(ValueError):
        fetch_range(_url(0), 10, 9)


# --------------------------------------------------------------------------------------
# unigram sampling, n = 2
# --------------------------------------------------------------------------------------
def test_sample_unigram_counts_two_sequences(fake, tmp_path):
    srv = fake("206")
    counts = sample_unigram_counts(2, seed=0, vocab=VOCAB, cache_dir=tmp_path)
    assert counts.shape == (VOCAB,) and counts.dtype == np.int64 and int(counts.sum()) == 2 * SEQ_LEN
    idx = np.random.default_rng(0).choice(N_SEQ, size=2, replace=False)
    expect = np.zeros(VOCAB, dtype=np.int64)
    for i in idx:
        expect += np.bincount(virtual_tokens(int(i) * SEQ_LEN, SEQ_LEN).astype(np.int64), minlength=VOCAB)
    assert np.array_equal(counts, expect)
    # one range request per sequence (two if it straddles a shard boundary)
    n_calls = len(srv.calls)
    assert n_calls in (2, 3) and all(c["stream"] is True for c in srv.calls)
    assert sum(r.served for r in srv.responses) == 2 * SEQ_BYTES
    assert sorted({c["shard"] for c in srv.calls}) == sorted({int(i * SEQ_BYTES // SHARD) for i in idx} |
                                                              {int((i * SEQ_BYTES + SEQ_BYTES - 1) // SHARD) for i in idx})
    z = np.load(tmp_path / "unigram_n2_s0.npz")
    assert np.array_equal(z["idx"], idx) and np.array_equal(z["counts"], counts)
    again = sample_unigram_counts(2, seed=0, vocab=VOCAB, cache_dir=tmp_path)
    assert len(srv.calls) == n_calls and np.array_equal(again, counts)     # cached: no new request
    # another seed draws other sequences; no cache -> fresh requests
    other = sample_unigram_counts(2, seed=1, vocab=VOCAB)
    assert len(srv.calls) > n_calls and int(other.sum()) == 2 * SEQ_LEN and not np.array_equal(other, counts)
    top = frequent_token_ids(counts, top_n=5)
    assert top.shape == (5,) and np.array_equal(np.sort(counts[top])[::-1], np.sort(counts)[::-1][:5])


# --------------------------------------------------------------------------------------
# network: the real thing
# --------------------------------------------------------------------------------------
def _skip_if_unreachable(e: BaseException) -> None:
    import requests
    if isinstance(e, requests.RequestException) or re.search(r"HTTP [45]\d\d", str(e)):
        pytest.skip(f"Pile range request unavailable: {e}")       # offline, proxy, rate limit
    raise e


@pytest.mark.network
def test_anchor_fetch_by_range_request_matches_dossier_hashes(tmp_path):
    """Fetches sequence 5 025 183 (4098 bytes of shard 0) from the Hub into a fresh cache
    dir and checks both dossier SHA256 hashes and the target id (DESIGN section 1)."""
    from lsne.anchor import verify_state2143
    try:
        toks = fetch_sequence(ANCHOR, cache_dir=tmp_path)
    except Exception as e:  # noqa: BLE001 - offline / proxy
        _skip_if_unreachable(e)
    assert toks.dtype == np.uint16 and toks.shape == (SEQ_LEN,)
    ctx = STATE2143["context_len"]
    assert sha256_tokens(toks) == STATE2143["sha256_sequence"]
    assert sha256_tokens(toks[:ctx]) == STATE2143["sha256_context"]
    assert sha256_tokens(toks[:ctx]).startswith(STATE2143["sha256_context_prefix_dossier"])
    assert int(toks[ctx]) == STATE2143["target_id"]
    v = verify_state2143(toks)
    assert v["sequence_ok"] and v["context_ok"] and v["context_prefix_ok"] and v["target_ok"]
    assert (tmp_path / f"seq_{ANCHOR}.npy").exists()
    assert np.array_equal(np.load(tmp_path / f"seq_{ANCHOR}.npy"), toks)


@pytest.mark.network
def test_sample_unigram_counts_two_real_sequences(tmp_path):
    """Two uniformly drawn training sequences by range request (~4 KB each)."""
    try:
        counts = sample_unigram_counts(2, seed=0, vocab=VOCAB, cache_dir=tmp_path)
    except Exception as e:  # noqa: BLE001 - offline / proxy
        _skip_if_unreachable(e)
    assert counts.shape == (VOCAB,) and int(counts.sum()) == 2 * SEQ_LEN and (counts >= 0).all()
    assert (tmp_path / "unigram_n2_s0.npz").exists()
    z = np.load(tmp_path / "unigram_n2_s0.npz")
    assert np.array_equal(z["idx"], np.random.default_rng(0).choice(N_SEQ, size=2, replace=False))
    assert frequent_token_ids(counts, top_n=3).shape == (3,)
