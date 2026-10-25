"""Run directories, manifests and DataFrame persistence.

Layout: ``<root>/<stage>/<model_tag>@<revision>/`` holds one stage's outputs for one
model checkpoint (``manifest.json`` + ``<name>.csv`` [+ ``<name>.parquet``]).  On Colab the
root defaults to the mounted Drive (``/content/drive/MyDrive/LSNE``), otherwise to the
``LSNE_RUNS`` environment variable or ``./runs``.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import importlib.util
import json
import logging
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, os.PathLike]
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DRIVE_MOUNT = "/content/drive/MyDrive"
DRIVE_ROOT = DRIVE_MOUNT + "/LSNE"
MANIFEST_NAME = "manifest.json"

logger = logging.getLogger(__name__)
_parquet_skip_logged = False   # the missing-engine notice is emitted once per process


# ======================================================================================
# environment
# ======================================================================================
def is_colab() -> bool:
    """True when running inside Google Colab."""
    if "google.colab" in sys.modules or os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError, AttributeError):
        return False


def drive_root() -> str:
    """Default root for run outputs: ``/content/drive/MyDrive/LSNE`` if the Colab Drive
    mount exists, else ``$LSNE_RUNS`` or ``./runs``."""
    if Path(DRIVE_MOUNT).exists():
        return DRIVE_ROOT
    return os.environ.get("LSNE_RUNS", "./runs")


def git_sha(cwd: Optional[PathLike] = None) -> str:
    """``git rev-parse HEAD`` of the package checkout ('unknown' on any failure)."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(cwd or PACKAGE_ROOT),
                           capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown"
    if r.returncode != 0:
        return "unknown"
    sha = r.stdout.strip()
    return sha if sha else "unknown"


def lsne_version() -> str:
    try:
        from . import __version__
        return str(__version__)
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _module_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "not installed"


# ======================================================================================
# run directory
# ======================================================================================
class RunDir:
    """``<root>/<stage>/<model_tag>@<revision>/`` (created on construction).

    ``model_tag`` may be a string (e.g. ``"pythia-160m"``) or a ``ModelSpec``-like object
    with ``name``/``revision`` attributes, in which case the tag is the last path
    component of its name and ``revision`` defaults to the spec's revision.
    """

    def __init__(self, root: PathLike, stage: str, model_tag, revision: Optional[str] = None,
                 mkdir: bool = True):
        if not isinstance(model_tag, (str, bytes)) and hasattr(model_tag, "name"):
            if revision is None:
                revision = getattr(model_tag, "revision", None)
            model_tag = str(model_tag.name).split("/")[-1]
        if revision is None:
            revision = "main"
        self.root = Path(root)
        self.stage = str(stage)
        self.model_tag = str(model_tag).replace("/", "-")
        self.revision = str(revision).replace("/", "-")
        self.path = self.root / self.stage / f"{self.model_tag}@{self.revision}"
        if mkdir:
            self.path.mkdir(parents=True, exist_ok=True)

    def file(self, name: str) -> Path:
        """Path of a file inside the run directory."""
        return self.path / name

    @property
    def manifest_path(self) -> Path:
        return self.file(MANIFEST_NAME)

    def __fspath__(self) -> str:
        return str(self.path)

    def __truediv__(self, other) -> Path:
        return self.path / other

    def __str__(self) -> str:
        return str(self.path)

    def __repr__(self) -> str:
        return f"RunDir({str(self.path)!r})"


def _as_dir(run_dir) -> Path:
    if isinstance(run_dir, RunDir):
        return run_dir.path
    return Path(run_dir)


# ======================================================================================
# JSON serialisation of arbitrary metadata
# ======================================================================================
def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, numpy/torch scalars and arrays, paths, sets,
    tuples, enums and datetimes into JSON-serialisable Python objects."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, enum.Enum):
        return to_jsonable(obj.value)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    # torch objects without importing torch at module level
    mod = type(obj).__module__ or ""
    if mod.startswith("torch"):
        if hasattr(obj, "tolist"):
            try:
                return to_jsonable(obj.detach().cpu().tolist())
            except Exception:
                pass
        return str(obj)
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return to_jsonable(obj.item())
        except Exception:
            pass
    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return to_jsonable(obj.tolist())
        except Exception:
            pass
    return str(obj)


def environment_info() -> dict:
    """Timestamp, git sha, package and library versions, host."""
    return dict(
        timestamp=_dt.datetime.now(_dt.timezone.utc).isoformat(),
        git_sha=git_sha(),
        lsne_version=lsne_version(),
        torch_version=_module_version("torch"),
        transformers_version=_module_version("transformers"),
        numpy_version=_module_version("numpy"),
        python_version=platform.python_version(),
        hostname=socket.gethostname(),
        platform=platform.platform(),
    )


def write_manifest(run_dir, **meta) -> dict:
    """Write ``manifest.json`` into ``run_dir`` (a :class:`RunDir` or a path).

    The manifest holds the environment (timestamp in UTC, git sha, lsne/torch/
    transformers versions, hostname, ...) plus every keyword in ``meta`` (dataclasses such
    as ``ModelSpec``/``HookSite``/``RunConfig`` are serialised via ``dataclasses.asdict``,
    numpy/torch values converted).  ``meta`` keys override environment keys of the same
    name.  Returns the manifest dict."""
    d = _as_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    manifest = environment_info()
    manifest["run_dir"] = str(d)
    manifest.update(to_jsonable(dict(meta)))
    path = d / MANIFEST_NAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def read_manifest(run_dir) -> dict:
    with open(_as_dir(run_dir) / MANIFEST_NAME, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================================================
# DataFrames
# ======================================================================================
def _stem(name: str) -> str:
    for ext in (".csv", ".parquet"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def save_df(df: pd.DataFrame, run_dir, name: str) -> Path:
    """Save ``df`` as ``<name>.csv`` (index dropped) in ``run_dir``, plus ``<name>.parquet``
    when a parquet engine (pyarrow/fastparquet) is available.  Without an engine the
    parquet copy is skipped and a warning is logged once per process; the CSV is the
    record either way.  Returns the CSV path."""
    global _parquet_skip_logged
    d = _as_dir(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    stem = _stem(name)
    csv_path = d / f"{stem}.csv"
    df.to_csv(csv_path, index=False)
    pq_path = d / f"{stem}.parquet"
    try:
        df.to_parquet(pq_path, index=False)
    except Exception as e:
        # no parquet engine installed (or unsupported dtypes): the CSV is the record
        if isinstance(e, ImportError) and not _parquet_skip_logged:
            _parquet_skip_logged = True
            logger.warning("parquet copy of %s skipped: no parquet engine (install pyarrow); "
                           "the CSV is the record -- %s", csv_path, e)
        if pq_path.exists():
            try:
                pq_path.unlink()
            except OSError:
                pass
    return csv_path


def load_df(run_dir, name: str) -> pd.DataFrame:
    """Load ``<name>`` from ``run_dir``: parquet if present and readable, else CSV."""
    d = _as_dir(run_dir)
    stem = _stem(name)
    pq_path = d / f"{stem}.parquet"
    if pq_path.exists():
        try:
            return pd.read_parquet(pq_path)
        except Exception:
            pass
    csv_path = d / f"{stem}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no {stem}.csv or {stem}.parquet in {d}")
    return pd.read_csv(csv_path)
