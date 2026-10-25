import json
import logging
import os

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite, ModelSpec, RunConfig
from lsne import io as IO


def test_rundir_layout(tmp_path):
    rd = IO.RunDir(tmp_path, "D", "pythia-14m", "step98000")
    assert rd.path == tmp_path / "D" / "pythia-14m@step98000"
    assert rd.path.is_dir()
    assert rd.file("grid.csv") == rd.path / "grid.csv"
    assert os.fspath(rd) == str(rd.path) and (rd / "x") == rd.path / "x"
    # ModelSpec-like argument: tag from the model name, revision from the spec
    rd2 = IO.RunDir(tmp_path, "C", ModelSpec(name="EleutherAI/pythia-160m", revision="step99000"))
    assert rd2.path == tmp_path / "C" / "pythia-160m@step99000"
    rd3 = IO.RunDir(tmp_path, "C", ModelSpec(name="EleutherAI/pythia-160m", revision="step99000"), "main")
    assert rd3.path.name == "pythia-160m@main"
    rd4 = IO.RunDir(tmp_path, "E", "org/model", None, mkdir=False)
    assert rd4.path.name == "org-model@main" and not rd4.path.exists()


def test_manifest_roundtrip(tmp_path):
    rd = IO.RunDir(tmp_path, "B", "pythia-14m", "step98000")
    spec = ModelSpec(name="EleutherAI/pythia-14m", revision="step98000", device="cpu")
    site = HookSite(layer=3, site="block_input")
    m = IO.write_manifest(rd, model=spec, hook=site, cfg=RunConfig(model=spec), doses=np.array([0.0, 0.01]),
                          n=np.int64(5), x=np.float32(1.5), flag=np.bool_(True), t=torch.tensor([1, 2]),
                          dtype=torch.float32, hashes={"ctx": "84c8"}, levels=(2, 4, 8), s={"a"})
    assert rd.manifest_path.exists()
    loaded = json.loads(rd.manifest_path.read_text(encoding="utf-8"))
    assert loaded == m
    for key in ("timestamp", "git_sha", "lsne_version", "torch_version", "transformers_version", "hostname"):
        assert key in loaded and isinstance(loaded[key], str) and loaded[key]
    assert loaded["timestamp"].endswith("+00:00")
    assert loaded["lsne_version"] == "0.1.0"
    assert loaded["git_sha"] == "unknown" or len(loaded["git_sha"]) == 40
    assert loaded["model"] == {"name": "EleutherAI/pythia-14m", "revision": "step98000", "dtype": "float32",
                               "device": "cpu", "attn_implementation": "eager"}
    assert loaded["hook"] == {"layer": 3, "site": "block_input"}
    assert loaded["cfg"]["model"]["name"] == "EleutherAI/pythia-14m" and loaded["cfg"]["k_levels"] == [0, 2, 4, 8]
    assert loaded["doses"] == [0.0, 0.01] and loaded["n"] == 5 and loaded["x"] == 1.5 and loaded["flag"] is True
    assert loaded["t"] == [1, 2] and loaded["dtype"] == "torch.float32"
    assert loaded["hashes"] == {"ctx": "84c8"} and loaded["levels"] == [2, 4, 8] and loaded["s"] == ["a"]
    assert IO.read_manifest(rd) == loaded
    # a plain path works as run_dir too
    m2 = IO.write_manifest(tmp_path / "plain", stage="X")
    assert (tmp_path / "plain" / "manifest.json").exists() and m2["stage"] == "X"


def test_git_sha_unknown_on_failure(tmp_path):
    assert IO.git_sha(tmp_path / "does-not-exist") == "unknown"
    sha = IO.git_sha()
    assert sha == "unknown" or (len(sha) == 40 and all(c in "0123456789abcdef" for c in sha))


def test_save_load_df_roundtrip(tmp_path):
    rd = IO.RunDir(tmp_path, "D", "pythia-14m", "step98000")
    df = pd.DataFrame(dict(family=["F01", "F02", "F03"], K=[0, 2, 4], logp=[-1.5, -2.25, np.nan],
                           label=["a", "b", "c"]))
    p = IO.save_df(df, rd, "grid")
    assert p == rd.file("grid.csv") and p.exists()
    back = IO.load_df(rd, "grid")
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    # name with extension is normalised; parquet (if any engine is present) must agree with the csv
    IO.save_df(df, rd, "grid2.csv")
    assert rd.file("grid2.csv").exists()
    if rd.file("grid2.parquet").exists():
        pd.testing.assert_frame_equal(pd.read_parquet(rd.file("grid2.parquet")), df, check_dtype=False)
    with pytest.raises(FileNotFoundError):
        IO.load_df(rd, "missing")


def test_save_df_logs_missing_parquet_engine_once(tmp_path, monkeypatch, caplog):
    """Without a parquet engine the CSV is the record and the skip is logged exactly once."""
    monkeypatch.setattr(IO, "_parquet_skip_logged", False)

    def no_engine(self, *args, **kwargs):
        raise ImportError("Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_engine)
    df = pd.DataFrame(dict(a=[1, 2], b=["x", "y"]))
    with caplog.at_level(logging.WARNING, logger="lsne.io"):
        p1 = IO.save_df(df, tmp_path, "one")
        p2 = IO.save_df(df, tmp_path, "two")
    assert p1.exists() and p2.exists()
    assert not (tmp_path / "one.parquet").exists() and not (tmp_path / "two.parquet").exists()
    msgs = [r.getMessage() for r in caplog.records if r.name == "lsne.io"]
    assert len(msgs) == 1 and "pyarrow" in msgs[0] and "one.csv" in msgs[0]
    pd.testing.assert_frame_equal(IO.load_df(tmp_path, "one"), df, check_dtype=False)


def test_drive_root_and_colab(monkeypatch, tmp_path):
    monkeypatch.setattr(IO, "DRIVE_MOUNT", str(tmp_path / "no-such-mount"))
    monkeypatch.delenv("LSNE_RUNS", raising=False)
    assert IO.drive_root() == "./runs"
    monkeypatch.setenv("LSNE_RUNS", str(tmp_path / "runs"))
    assert IO.drive_root() == str(tmp_path / "runs")
    mount = tmp_path / "MyDrive"
    mount.mkdir()
    monkeypatch.setattr(IO, "DRIVE_MOUNT", str(mount))
    monkeypatch.setattr(IO, "DRIVE_ROOT", str(mount / "LSNE"))
    assert IO.drive_root() == str(mount / "LSNE")
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)
    assert isinstance(IO.is_colab(), bool)
    monkeypatch.setenv("COLAB_RELEASE_TAG", "release-x")
    assert IO.is_colab() is True


def test_to_jsonable_nested():
    obj = dict(a=np.arange(2), b=(np.float64(0.5), {1, 2}), c=HookSite(0, "mlp_output"), d=None, e=b"x")
    out = IO.to_jsonable(obj)
    assert out == {"a": [0, 1], "b": [0.5, [1, 2]], "c": {"layer": 0, "site": "mlp_output"}, "d": None, "e": "x"}
    json.dumps(out)
