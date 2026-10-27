"""Guards of the trace pipeline (lsne.chain.cli): serialisation that survives failed chunks and
dead workers, a vocabulary that refuses an empty corpus, a trainer that refuses a vocabulary
which does not belong to its corpus.

The first A100 run of notebook 12 lost 558 of 2 135 chunks when one serialisation worker died
(``concurrent.futures.process.BrokenProcessPool``), never wrote ``serialize_params.json``, then
built a vocabulary from 0 transactions (exit 0) and trained on it with an address OOV rate of
0.967.  Fixture: tests/fixtures/xatu (ethPandaOps Xatu, CC BY 4.0; four traces chunks
1 428 000, 1 718 000, 1 920 000, 1 978 000).  Faults are injected by replacing the per-chunk
worker; the pool forks its workers after the replacement, so they run the faulty worker too,
and a crash is a real ``os._exit`` of a worker process.
"""
import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from lsne.chain import cli as chain_cli
from lsne.chain import vocab as V

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "xatu")
STARTS = [1428000, 1718000, 1920000, 1978000]
FAULT_ENV = "LSNE_TEST_SERIALIZE_FAULTS"
_REAL_WORKER = chain_cli._serialize_chunk


def _first_attempt(marker_dir: str, start: int) -> bool:
    """True exactly once per chunk across processes (atomic O_EXCL create)."""
    try:
        os.close(os.open(os.path.join(marker_dir, f"attempt-{start}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def _faulty_worker(job: tuple) -> dict:
    """The per-chunk worker with the faults of ``$LSNE_TEST_SERIALIZE_FAULTS`` (JSON: ``dir``
    for markers, ``faults`` {start: kind}, ``always`` [starts that fail on every attempt])."""
    cfg = json.loads(os.environ[FAULT_ENV])
    start, out_dir = int(job[0]), job[2]
    kind = cfg["faults"].get(str(start))
    if kind is None or not (start in cfg.get("always", []) or _first_attempt(cfg["dir"], start)):
        return _REAL_WORKER(job)
    if "params_seen" in cfg:                 # serialize_params.json must exist before any chunk is processed
        with open(os.path.join(cfg["dir"], f"params-{start}"), "w") as fh:
            fh.write(str(os.path.exists(os.path.join(out_dir, "serialize_params.json"))))
    if kind == "error":                      # the worker's own exception handling (e.g. a corrupt parquet file)
        return dict(start=start, path=chain_cli.shard_path(out_dir, start), n_tx=0, n_tokens=0, skipped=False,
                    error="ValueError: injected")
    if kind == "raise":                      # an exception escaping the worker
        raise RuntimeError("injected")
    if kind == "crash":                      # the worker process dies (OOM kill)
        os._exit(17)
    if kind == "crash_after_write":          # dies after its shard is on disk, before the result arrives
        _REAL_WORKER(job)
        os._exit(17)
    if kind == "crash_after_others":         # dies once every other chunk is written (deterministic round 0)
        others = [chain_cli.shard_path(out_dir, s) for s in STARTS if s != start]
        t0 = time.time()
        while not all(os.path.exists(p) for p in others) and time.time() - t0 < 30:
            time.sleep(0.05)
        time.sleep(0.5)
        os._exit(17)
    raise AssertionError(kind)


@pytest.fixture
def faults(monkeypatch, tmp_path):
    """``set(faults, always=(), params_seen=False)`` installs the faulty worker."""
    marker = tmp_path / "markers"
    marker.mkdir()

    def set_(faults: dict, always=(), params_seen=False):
        cfg = dict(dir=str(marker), faults={str(k): v for k, v in faults.items()}, always=[int(s) for s in always])
        if params_seen:
            cfg["params_seen"] = True
        monkeypatch.setenv(FAULT_ENV, json.dumps(cfg))
        monkeypatch.setattr(chain_cli, "_serialize_chunk", _faulty_worker)
        return marker
    return set_


def _serialize(out, workers=1, **kw):
    lines = []
    df = chain_cli.serialize_chunks(FIX, STARTS[0], STARTS[-1] + 1000, str(out), workers=workers, log_fn=lines.append,
                                    **kw)
    return df.set_index("start"), lines


# --------------------------------------------------------------------------------------
# serialize
# --------------------------------------------------------------------------------------
def test_retry_schedule():
    assert chain_cli.retry_schedule(12) == [12, 6, 3, 1]
    assert chain_cli.retry_schedule(4) == [4, 2, 1]
    assert chain_cli.retry_schedule(2) == [2, 1]
    assert chain_cli.retry_schedule(1) == [1, 1] and chain_cli.retry_schedule(0) == [1, 1]


def test_serialize_logs_every_chunk_as_it_completes(tmp_path):
    # a worker killed in an earlier session left a partial file behind (on Drive it would stay forever)
    (tmp_path / "itr").mkdir()
    stale = tmp_path / "itr" / "itr-001718000.jsonl.gz.tmp.4242.gz"
    stale.write_bytes(b"\x1f\x8b partial")
    df, lines = _serialize(tmp_path / "itr", workers=3)
    assert not stale.exists()
    assert df["error"].isna().all() and (df["attempts"] == 1).all() and df["errors"].isna().all()
    assert len(chain_cli.list_shards(str(tmp_path / "itr"))) == 4
    done = [l for l in lines if l.startswith("serialize ") and " tx, " in l]
    assert len(done) == 4 and sorted(int(l.split()[1].rstrip(":")) for l in done) == STARTS
    assert sorted(l.rsplit("[", 1)[1] for l in done) == ["1/4]", "2/4]", "3/4]", "4/4]"]
    # resume: a second call skips every shard
    df2, lines2 = _serialize(tmp_path / "itr", workers=3)
    assert df2["skipped"].all() and all("skipped (exists)" in l for l in lines2)


def test_serialize_retries_failed_chunks_in_process(tmp_path, faults):
    faults({1718000: "error", 1920000: "raise"})
    df, lines = _serialize(tmp_path / "itr", workers=1)
    assert df["error"].isna().all() and len(chain_cli.list_shards(str(tmp_path / "itr"))) == 4
    assert df.loc[1718000, "attempts"] == 2 and df.loc[1718000, "errors"] == "ValueError: injected"
    assert df.loc[1920000, "attempts"] == 2 and df.loc[1920000, "errors"] == "RuntimeError: injected"
    assert (df.loc[[1428000, 1978000], "attempts"] == 1).all()
    assert any(l.startswith("serialize 1718000: FAILED ValueError: injected") for l in lines)
    assert any(l.startswith("serialize: retry round 1 with 1 worker(s) for 2 chunk(s): 1718000, 1920000") for l in lines)


def test_serialize_recovers_from_a_worker_crash(tmp_path, faults):
    """BrokenProcessPool: the chunks finished before the crash are kept (never rewritten), the
    chunk of the dead worker is retried with fewer workers."""
    faults({1978000: "crash_after_others"})
    out = tmp_path / "itr"
    df, lines = _serialize(out, workers=2)
    assert df["error"].isna().all() and len(chain_cli.list_shards(str(out))) == 4
    assert (df.loc[[1428000, 1718000, 1920000], "attempts"] == 1).all()
    assert df.loc[1978000, "attempts"] == 2 and "BrokenProcessPool" in df.loc[1978000, "errors"]
    assert any(l.startswith("serialize 1978000: FAILED BrokenProcessPool") for l in lines)
    assert any(l.startswith("serialize: retry round 1 with 1 worker(s) for 1 chunk(s): 1978000") for l in lines)
    rows = list(chain_cli.iter_rows(str(out)))
    assert len(rows) == 19 and sum(r["block_number"] >= 1978000 for r in rows) == 1
    assert not [f for f in os.listdir(out) if ".tmp." in f]                   # no partial files left behind


def test_serialize_counts_a_shard_written_before_the_worker_died(tmp_path, faults):
    faults({1920000: "crash_after_write"})
    out = tmp_path / "itr"
    df, lines = _serialize(out, workers=2, overwrite=True)
    assert df["error"].isna().all() and len(chain_cli.list_shards(str(out))) == 4
    assert df.loc[1920000, "attempts"] == 2 and bool(df.loc[1920000, "skipped"])     # not rewritten
    assert sum(1 for r in chain_cli.iter_rows(chain_cli.shard_path(str(out), 1920000))) == 10


def test_serialize_cli_lists_missing_chunks_and_exits_nonzero(tmp_path, faults, capsys, monkeypatch):
    """A chunk that kills its worker on every attempt: params written first, the other chunks
    written (the last round isolates every chunk in its own worker), exit code 2 and the missing
    chunk listed; the next call resumes with that chunk alone."""
    marker = faults({1718000: "crash"}, always=[1718000], params_seen=True)
    out = str(tmp_path / "itr")
    rc = chain_cli.main(["serialize", "--cache", FIX, "--from", "1428000", "--to", "1979000", "--out", out,
                         "--workers", "2"])
    text = capsys.readouterr().out
    assert rc == 2
    assert (marker / "params-1718000").read_text() == "True"            # serialize_params.json existed before
    params = json.load(open(os.path.join(out, "serialize_params.json")))
    assert params["block_from"] == 1428000 and params["serialize_kw"]["max_args"] == 4
    starts = sorted(int(chain_cli.SHARD_RE.match(os.path.basename(p)).group(1)) for p in chain_cli.list_shards(out))
    assert starts == [1428000, 1920000, 1978000]
    assert "ERROR 1 of 4 chunks missing after 2 rounds (workers [2, 1]): 1718000" in text
    assert "MISSING 1718000: BrokenProcessPool" in text and "retry round 1 with 1 worker(s)" in text
    # the next call resumes: only the missing chunk is serialised
    monkeypatch.delenv(FAULT_ENV)
    monkeypatch.setattr(chain_cli, "_serialize_chunk", _REAL_WORKER)
    assert chain_cli.main(["serialize", "--cache", FIX, "--from", "1428000", "--to", "1979000", "--out", out,
                           "--workers", "2"]) == 0
    text = capsys.readouterr().out
    assert text.count("skipped (exists)") == 3 and "serialize 1718000: 5 tx" in text
    assert len(chain_cli.list_shards(out)) == 4


def _shard_tokens(out: str) -> dict:
    """``{start: [tokens of every row]}`` of the shards in ``out``."""
    return {int(chain_cli.SHARD_RE.match(os.path.basename(p)).group(1)): [r["tokens"] for r in chain_cli.iter_rows(p)]
            for p in chain_cli.list_shards(out)}


def test_overwrite_never_keeps_an_old_option_shard(tmp_path, faults, capsys, monkeypatch):
    """``--overwrite`` into shards of other options while a worker dies: the chunk whose rewrite
    failed in round 0 must not survive with the old tokens (the retry rounds and a plain resume
    skip existing shards).  Either every shard equals a clean serialisation with the new options,
    or the chunk is reported missing (exit 2) and the next resume without ``--overwrite`` fills it."""
    base = ["serialize", "--cache", FIX, "--from", "1428000", "--to", "1979000"]
    clean = str(tmp_path / "clean")
    assert chain_cli.main(base + ["--out", clean, "--max-args", "4"]) == 0
    ref = _shard_tokens(clean)
    assert sorted(ref) == STARTS
    for case, always in (("once", ()), ("always", (1428000,))):
        out = str(tmp_path / case)
        monkeypatch.setattr(chain_cli, "_serialize_chunk", _REAL_WORKER)
        assert chain_cli.main(base + ["--out", out, "--max-args", "2"]) == 0
        old = _shard_tokens(out)
        assert old[1428000] != ref[1428000]                                   # the options change the tokens
        faults({1428000: "crash"}, always=always)
        capsys.readouterr()
        rc = chain_cli.main(base + ["--out", out, "--max-args", "4", "--overwrite", "--workers", "2"])
        text = capsys.readouterr().out
        assert "--overwrite removed 4 existing shard(s)" in text
        assert json.load(open(os.path.join(out, "serialize_params.json")))["serialize_kw"]["max_args"] == 4
        now = _shard_tokens(out)
        assert all(now[s] == ref[s] for s in now)                             # never an old-option shard
        if not always:                                                        # the retry round rewrote the chunk
            assert rc == 0 and now == ref and "retry round 1" in text
            continue
        assert rc == 2 and "MISSING 1428000" in text and 1428000 not in now   # missing, not stale
        # the resume the CLI and notebook advise (no --overwrite) fills the chunk with the new options
        monkeypatch.delenv(FAULT_ENV)
        monkeypatch.setattr(chain_cli, "_serialize_chunk", _REAL_WORKER)
        assert chain_cli.main(base + ["--out", out, "--max-args", "4", "--workers", "2"]) == 0
        assert _shard_tokens(out) == ref
        assert capsys.readouterr().out.count("skipped (exists)") == 3


def test_serialize_refuses_to_mix_options(tmp_path, capsys):
    out = str(tmp_path / "itr")
    args = ["serialize", "--cache", FIX, "--from", "1428000", "--to", "1429000", "--out", out]
    assert chain_cli.main(args) == 0
    assert chain_cli.main(args + ["--max-args", "2"]) == 2
    assert "written with other options" in capsys.readouterr().out
    assert json.load(open(os.path.join(out, "serialize_params.json")))["serialize_kw"]["max_args"] == 4
    assert chain_cli.main(args + ["--max-args", "2", "--overwrite"]) == 0
    assert json.load(open(os.path.join(out, "serialize_params.json")))["serialize_kw"]["max_args"] == 2


# --------------------------------------------------------------------------------------
# vocab / train
# --------------------------------------------------------------------------------------
def test_vocab_exits_nonzero_without_transactions(tmp_path, capsys):
    empty, voc = tmp_path / "itr_train", str(tmp_path / "vocab")
    empty.mkdir()
    assert chain_cli.main(["vocab", "--in", str(empty), "--out", voc]) == 1           # no shard (the first A100 run)
    assert chain_cli.main(["vocab", "--in", str(tmp_path / "missing"), "--out", voc]) == 1
    import gzip
    with gzip.open(empty / "itr-001428000.jsonl.gz", "wt") as fh:                    # a shard without a transaction
        fh.write("\n")
    assert chain_cli.main(["vocab", "--in", str(empty), "--out", voc]) == 1
    text = capsys.readouterr().out
    assert "no ITR shards" in text and "hold 0 transactions; no vocabulary written" in text
    assert not os.path.exists(voc)


def test_train_aborts_when_the_vocabulary_does_not_fit_the_corpus(tmp_path, capsys):
    a, b, voc, lm = (str(tmp_path / x) for x in ("itr_a", "itr_b", "vocab_a", "lm"))
    assert chain_cli.main(["serialize", "--cache", FIX, "--from", "1428000", "--to", "1429000", "--out", a]) == 0
    assert chain_cli.main(["serialize", "--cache", FIX, "--from", "1920000", "--to", "1921000", "--out", b]) == 0
    assert chain_cli.main(["vocab", "--in", a, "--out", voc]) == 0
    train = ["train", "--in", b, "--vocab", voc, "--size", "tiny", "--out", lm, "--steps", "2", "--block-size", "32",
             "--batch-size", "2", "--eval-every", "1", "--save-every", "0", "--val-frac", "0", "--device", "cpu"]
    assert chain_cli.main(train) == 1
    text = capsys.readouterr().out
    assert "train: ERROR address OOV rate 0.647 of the 10 training transactions exceeds 0.5" in text
    assert not os.path.exists(os.path.join(lm, "manifest.json"))
    # the check can be switched off; the manifest records the OOV rate of the training split
    assert chain_cli.main(train + ["--max-oov-address", "1"]) == 0
    man = json.load(open(os.path.join(lm, "manifest.json")))
    assert man["corpus"]["oov_train"]["rate_A"] == pytest.approx(0.647, abs=1e-3)
    assert man["corpus"]["oov_train"]["n_lists"] == 10 and man["corpus"]["oov_val"]["n_lists"] == 0


def test_oov_guard_and_split_statistics(tmp_path):
    assert chain_cli.oov_guard(dict(n_tx_train=5, oov_train=dict(rate_A=0.51))) is not None
    assert chain_cli.oov_guard(dict(n_tx_train=5, oov_train=dict(rate_A=0.5))) is None
    assert chain_cli.oov_guard(dict(oov=dict(rate_A=0.97))) is not None                 # manifests before oov_train
    assert chain_cli.oov_guard(dict(oov_train=dict(rate_A=float("nan")))) is None       # no address tokens
    assert chain_cli.oov_guard(dict(oov_train=dict(rate_A=None))) is None               # NaN in a JSON manifest
    assert chain_cli.oov_guard(dict(oov_train=dict(rate_A=0.97)), None) is None
    assert chain_cli.oov_guard(dict(oov_train=dict(rate_A=0.97)), 1.0) is None
    out = str(tmp_path / "itr")
    assert chain_cli.main(["serialize", "--cache", FIX, "--from", "1718000", "--to", "1921000", "--out", out]) == 0
    rows = [r for r in chain_cli.iter_rows(out) if not chain_cli._is_val(r["tx_hash"], 0.2)]
    v = V.build_vocab((r["tokens"] for r in rows), n_addresses=50)          # notebook 12's smoke vocabulary
    _, _, st = chain_cli.corpus_blocks(out, v, 32, 512, 0.2, log_fn=lambda m: None)
    assert st["oov_train"]["n_lists"] == st["n_tx_train"] and st["oov_val"]["n_lists"] == st["n_tx_val"]
    assert st["oov_train"]["rate_A"] == 0.0 and chain_cli.oov_guard(st) is None
    assert st["oov"]["n_A"] == st["oov_train"]["n_A"] + st["oov_val"]["n_A"]


def test_probe_criterion_reports_beats_surface():
    """``counts_beyond_surface`` = counts AND beats_surface (the frozen feature's criterion)."""
    from lsne.chain import source as S

    rng = np.random.default_rng(0)
    n, d = 40, 8
    Xp, Xq = rng.normal(size=(n, d)) + 1.5, rng.normal(size=(n, d))
    Yp, Yq = rng.normal(size=(n, d)), rng.normal(size=(n, d))
    S1 = {1: np.vstack([Xp, Xq])}
    S0 = {1: np.vstack([Yp, Yq])}
    hi = S.criterion_table(S1, S0, n, surface=dict(combined=0.999), seed=0, n_boot=200, n_perm=50, n_random=20)
    lo = S.criterion_table(S1, S0, n, surface=dict(combined=0.6), seed=0, n_boot=200, n_perm=50, n_random=20)
    assert bool(hi.loc[0, "counts"]) and not bool(hi.loc[0, "beats_surface"]) and not bool(hi.loc[0, "counts_beyond_surface"])
    assert bool(lo.loc[0, "counts"]) and bool(lo.loc[0, "beats_surface"]) and bool(lo.loc[0, "counts_beyond_surface"])
    assert "counts_beyond_surface" not in S.criterion_table(S1, S0, n, seed=0, n_boot=20, n_perm=5, n_random=5).columns
