"""Test-time training (TTT) as a *structural injection* (DESIGN §4.8d, SPEC ``lsne/ttt.py``).

Continued training of a frozen checkpoint on a small contrast corpus (the Ethereum /
DAO-fork 2016 texts the client provides under ``data/ttt/dao_fork/``; nothing in the
repository contains that corpus) and measurement of

1. **memorisation speed** per snippet: the snippet's own mean NLL after every update
   step, ``steps_to_threshold`` = first step at which it falls below a pre-registered
   threshold (:func:`memorisation_speed`, :func:`ttt_memorisation_table`, optionally over
   several checkpoints because the dossier suggests phase dependence);
2. **before/after** effects on every other quantity, measured on the same stimulus grid
   (:func:`before_after` with any ``measure_fn(mb) -> DataFrame``);
3. **three control arms** with the same training budget: ``neutral`` (same-size corpus
   from a neutral domain), ``nofork`` (same era/domain without the split narrative) and
   ``permuted`` (token-permuted DAO corpus, :func:`permute_corpus`).

Training conventions (dossier replay conventions, DESIGN §1/§6): AdamW, gradient-norm
clipping at 1.0, FP32 master weights (a model that is not float32 is refused), seeded and
deterministic batching. Training never mutates the passed :class:`~lsne.model.ModelBundle`
unless ``inplace=True``: :func:`memorisation_speed` works on a state-dict snapshot that is
restored in a ``finally`` block, :func:`test_time_train` on a deep copy of the model.

Corpus format (:func:`load_corpus`): a directory or a file; ``*.txt`` (one document per
file, or documents separated by a line consisting of ``---``), ``*.jsonl`` (field
``text``). When a directory is scanned, note files (``README*``, any extension) and
hidden files (``.gitkeep`` etc.) are skipped, so the git-ignored drop folders of the
repository (which hold only such a note) still count as empty. Tokenised without special
tokens and chunked into pieces of at most ``seq_len`` tokens. A missing or empty path
falls back to the built-in public-domain placeholder
``data/ttt/placeholder_public_domain.txt`` (prose written for this repository; no
historical documents) and marks the result as a placeholder
(``corpus.meta['is_placeholder']`` and the module flag :data:`LAST_LOAD_WAS_PLACEHOLDER`).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import ModelSpec
from .model import ModelBundle, load_model

__all__ = [
    "TTT_DATA_DIR", "DAO_CORPUS_DIR", "PLACEHOLDER_PATH", "CONTROL_ARMS",
    "CORPUS_SUFFIXES", "NOTE_FILE_PREFIXES", "LAST_LOAD_WAS_PLACEHOLDER", "LAST_LOAD_META",
    "Corpus", "is_note_file", "read_documents", "tokenise_documents", "load_corpus",
    "placeholder_corpus",
    "permute_corpus", "corpus_for_control", "pad_id_of", "collate",
    "WeightSnapshot", "select_train_params", "TTTResult", "snippet_nll", "corpus_nll",
    "memorisation_speed", "test_time_train", "before_after", "ttt_memorisation_table",
]

_REPO_ROOT = Path(__file__).resolve().parent.parent
#: ``data/ttt`` of the repository (loader root, see ``data/ttt/README.md``).
TTT_DATA_DIR = _REPO_ROOT / "data" / "ttt"
#: Where the client drops the DAO-fork corpus (``*.txt`` / ``*.jsonl``); not in the repo.
DAO_CORPUS_DIR = TTT_DATA_DIR / "dao_fork"
#: Built-in placeholder used when no corpus is present.
PLACEHOLDER_PATH = TTT_DATA_DIR / "placeholder_public_domain.txt"
#: Control arms of DESIGN §4.8d (3).
CONTROL_ARMS = ("neutral", "nofork", "permuted")
#: Names accepted for "the corpus itself" in ``control=``.
_INJECTION_NAMES = (None, "", "none", "injection", "corpus")
#: File suffixes read as corpus documents when a directory is scanned.
CORPUS_SUFFIXES = (".txt", ".jsonl")
#: Files whose (lower-cased) name starts with one of these are notes, not documents;
#: they are skipped when a directory is scanned (``README``, ``README.txt``, ...).
NOTE_FILE_PREFIXES = ("readme",)

DOC_SEPARATOR = "---"
IGNORE_INDEX = -100
_OPTIMIZERS = ("adamw", "adam", "sgd")

#: True when the last :func:`load_corpus` call fell back to the placeholder corpus.
LAST_LOAD_WAS_PLACEHOLDER: bool = False
#: Meta dict of the last :func:`load_corpus` call (same content as ``corpus.meta``).
LAST_LOAD_META: dict = {}


# --------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------
class Corpus(list):
    """``list[Tensor[T]]`` of token snippets that also carries ``meta``.

    ``meta`` holds ``source`` (file or directory actually read), ``requested`` (the path
    that was asked for), ``is_placeholder``, ``n_docs``, ``n_snippets``, ``n_tokens``,
    ``seq_len`` and ``control`` (None or the control-arm name). Slicing returns a plain
    list; use ``Corpus(sub, corpus.meta)`` to keep the meta.
    """

    def __init__(self, snippets: Sequence[torch.Tensor] = (), meta: Optional[dict] = None):
        super().__init__(snippets)
        self.meta: dict = dict(meta or {})

    @property
    def is_placeholder(self) -> bool:
        return bool(self.meta.get("is_placeholder", False))

    @property
    def n_tokens(self) -> int:
        return int(sum(int(s.numel()) for s in self))


def _resolve_path(path) -> Optional[Path]:
    """As given, or relative to the repository root if it does not exist from the cwd."""
    if path is None:
        return None
    p = Path(path)
    if p.exists():
        return p
    alt = _REPO_ROOT / p
    return alt if alt.exists() else p


def _read_txt(path: Path) -> list[str]:
    docs, cur = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip() == DOC_SEPARATOR:
            docs.append("\n".join(cur))
            cur = []
        else:
            cur.append(line)
    docs.append("\n".join(cur))
    return [d.strip() for d in docs if d.strip()]


def _read_jsonl(path: Path, text_field: str = "text") -> list[str]:
    docs = []
    with path.open(encoding="utf-8") as fh:
        for ln, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{ln}: invalid JSON line ({e})") from e
            text = obj.get(text_field) if isinstance(obj, dict) else None
            if isinstance(text, str) and text.strip():
                docs.append(text.strip())
    return docs


def is_note_file(path) -> bool:
    """True for files a directory scan must not read as corpus documents: hidden files
    (name starting with ``.``, e.g. ``.gitkeep``) and notes whose name starts with one
    of :data:`NOTE_FILE_PREFIXES` (case-insensitive, any extension: ``README``,
    ``README.txt``, ``readme.md``). The drop folders ``data/ttt/{dao_fork,neutral,nofork}``
    ship with such a note only, so they must still yield no documents."""
    name = Path(path).name
    low = name.lower()
    return name.startswith(".") or any(low.startswith(pfx) for pfx in NOTE_FILE_PREFIXES)


def _corpus_files(path: Path) -> list[Path]:
    """Files to read: the file itself (whatever its name, an explicit request), or the
    ``*.txt`` / ``*.jsonl`` entries of a directory minus note / hidden files."""
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(p for p in path.iterdir()
                      if p.is_file() and p.suffix.lower() in CORPUS_SUFFIXES
                      and not is_note_file(p))
    return []


def read_documents(path) -> list[str]:
    """Raw documents (strings) of a file or a directory (non-recursive, sorted file
    order): ``*.txt`` with ``---`` separators, ``*.jsonl`` with a ``text`` field.
    In a directory, note files (``README*``) and hidden files are skipped
    (:func:`is_note_file`); an explicitly given file is always read.
    Missing path -> ``[]``."""
    p = _resolve_path(path)
    docs: list[str] = []
    if p is None:
        return docs
    for f in _corpus_files(p):
        if f.suffix.lower() == ".jsonl":
            docs.extend(_read_jsonl(f))
        else:
            docs.extend(_read_txt(f))
    return docs


def tokenise_documents(docs: Sequence[str], tokenizer, seq_len: int = 512,
                       min_len: int = 2) -> list[torch.Tensor]:
    """Tokenise without special tokens and chunk each document into pieces of at most
    ``seq_len`` tokens (CPU LongTensors); pieces shorter than ``min_len`` are dropped
    (a causal-LM loss needs at least two tokens)."""
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    out: list[torch.Tensor] = []
    for d in docs:
        ids = tokenizer(d, add_special_tokens=False)["input_ids"]
        for s in range(0, len(ids), seq_len):
            chunk = ids[s:s + seq_len]
            if len(chunk) >= min_len:
                out.append(torch.tensor(chunk, dtype=torch.long))
    return out


def load_corpus(path, tokenizer, max_docs: Optional[int] = None, seq_len: int = 512) -> Corpus:
    """Tokenised snippets of ``path`` (directory or file; see module docstring).

    Falls back to the built-in placeholder when ``path`` is None, missing, or yields no
    documents; the result then has ``meta['is_placeholder'] = True`` and the module flag
    :data:`LAST_LOAD_WAS_PLACEHOLDER` is set. ``max_docs`` truncates the document list
    before tokenisation.
    """
    global LAST_LOAD_WAS_PLACEHOLDER, LAST_LOAD_META
    p = _resolve_path(path)
    docs = read_documents(p) if p is not None else []
    is_placeholder = False
    source = str(p) if p is not None else None
    if not docs:
        docs = read_documents(PLACEHOLDER_PATH)
        if not docs:
            raise FileNotFoundError(f"placeholder corpus missing: {PLACEHOLDER_PATH}")
        is_placeholder = True
        source = str(PLACEHOLDER_PATH)
    n_available = len(docs)
    if max_docs is not None:
        docs = docs[:int(max_docs)]
    snippets = tokenise_documents(docs, tokenizer, seq_len=seq_len)
    meta = dict(source=source, requested=None if path is None else str(path),
                is_placeholder=is_placeholder, n_docs=len(docs), n_docs_available=n_available,
                n_snippets=len(snippets), n_tokens=int(sum(len(s) for s in snippets)),
                seq_len=int(seq_len), control=None)
    LAST_LOAD_WAS_PLACEHOLDER = is_placeholder
    LAST_LOAD_META = dict(meta)
    return Corpus(snippets, meta)


def placeholder_corpus(tokenizer, seq_len: int = 512, n_chunks: Optional[int] = None) -> Corpus:
    """The built-in placeholder corpus (optionally the first ``n_chunks`` snippets)."""
    c = load_corpus(None, tokenizer, seq_len=seq_len)
    if n_chunks is not None:
        c = Corpus(list(c)[:n_chunks], dict(c.meta, n_snippets=min(n_chunks, len(c))))
    return c


def permute_corpus(snippets: Sequence[torch.Tensor], seed: int = 0) -> Corpus:
    """Token-permuted control: every snippet keeps its multiset of tokens (and its
    length) in a seeded random order. Deterministic in ``seed`` and snippet index."""
    out = []
    for i, s in enumerate(snippets):
        s = torch.as_tensor(s, dtype=torch.long)
        g = torch.Generator().manual_seed(int(seed) * 100_003 + i)
        perm = torch.randperm(int(s.numel()), generator=g).to(s.device)
        out.append(s[perm])
    meta = dict(getattr(snippets, "meta", {}) or {})
    meta.update(control="permuted", permute_seed=int(seed), n_snippets=len(out))
    return Corpus(out, meta)


def corpus_for_control(corpus, control: Optional[str], controls: Optional[dict] = None,
                       seed: int = 0):
    """Corpus variant of a control arm: ``None`` -> ``corpus``; ``'permuted'`` ->
    :func:`permute_corpus` (or ``controls['permuted']`` if given); ``'neutral'`` /
    ``'nofork'`` -> ``controls[control]`` (a clear error if missing)."""
    if control in _INJECTION_NAMES:
        return corpus
    if control not in CONTROL_ARMS:
        raise ValueError(f"unknown control arm {control!r}; expected None or one of {CONTROL_ARMS}")
    if controls and control in controls and controls[control] is not None:
        return controls[control]
    if control == "permuted":
        return permute_corpus(corpus, seed=seed)
    raise ValueError(
        f"control={control!r} needs a matched corpus: pass controls={{{control!r}: <list of "
        f"token snippets>}} (e.g. load_corpus('data/ttt/{control}', tok)); "
        f"see data/ttt/README.md. Nothing in the repository contains that corpus.")


# --------------------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------------------
def pad_id_of(tokenizer) -> int:
    """``pad_token_id`` if the tokenizer has one, else ``eos_token_id``, else 0."""
    for attr in ("pad_token_id", "eos_token_id"):
        v = getattr(tokenizer, attr, None)
        if v is not None:
            return int(v)
    return 0


def collate(snippets: Sequence[torch.Tensor], pad_id: int, device=None) -> dict[str, torch.Tensor]:
    """Right-pad to the longest snippet: ``input_ids``, ``attention_mask`` and ``labels``
    (``-100`` on pads, so padded positions carry no loss)."""
    snippets = [torch.as_tensor(s, dtype=torch.long).reshape(-1) for s in snippets]
    if not snippets:
        raise ValueError("empty batch")
    B, T = len(snippets), max(int(s.numel()) for s in snippets)
    ids = torch.full((B, T), int(pad_id), dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.long)
    labels = torch.full((B, T), IGNORE_INDEX, dtype=torch.long)
    for i, s in enumerate(snippets):
        n = int(s.numel())
        ids[i, :n] = s
        mask[i, :n] = 1
        labels[i, :n] = s
    batch = dict(input_ids=ids, attention_mask=mask, labels=labels)
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    return batch


def _snippet_batch(snippet, mb: ModelBundle) -> dict[str, torch.Tensor]:
    """Tensor[T] / Tensor[1,T] / Tensor[B,T] / list of tensors -> padded batch."""
    if isinstance(snippet, (list, tuple)):
        parts = list(snippet)
    else:
        t = torch.as_tensor(snippet, dtype=torch.long)
        parts = [t] if t.dim() == 1 else [row for row in t.reshape(-1, t.shape[-1])]
    if any(int(p.numel()) < 2 for p in parts):
        raise ValueError("every snippet needs at least two tokens for a causal-LM loss")
    return collate(parts, pad_id_of(mb.tokenizer), mb.device)


class _EpochSampler:
    """Seeded shuffled batch sampler that cycles through epochs."""

    def __init__(self, n: int, batch: int, seed: int):
        self.n = int(n)
        self.batch = max(1, min(int(batch), self.n))
        self.gen = torch.Generator().manual_seed(int(seed))
        self.order: list[int] = []
        self.pos = 0
        self.epoch = 0

    def next(self) -> list[int]:
        if self.pos >= len(self.order):
            self.order = torch.randperm(self.n, generator=self.gen).tolist()
            self.pos = 0
            self.epoch += 1
        idx = self.order[self.pos:self.pos + self.batch]
        self.pos += len(idx)
        return idx


# --------------------------------------------------------------------------------------
# weights, parameter selection, optimiser
# --------------------------------------------------------------------------------------
def _as_model(x) -> nn.Module:
    return x.model if isinstance(x, ModelBundle) else x


class WeightSnapshot:
    """CPU copies of a model's full ``state_dict`` (parameters and buffers).

    ``restore()`` writes them back in place (bitwise) and switches ``requires_grad`` off.
    Context manager: ``with WeightSnapshot(mb): ...`` restores on exit, also after an
    exception.
    """

    def __init__(self, model_or_bundle):
        self.model = _as_model(model_or_bundle)
        self.state = {k: v.detach().to("cpu", copy=True) for k, v in self.model.state_dict().items()}

    def restore(self) -> None:
        with torch.no_grad():
            self.model.load_state_dict(self.state, strict=True)
        for p in self.model.parameters():
            p.requires_grad_(False)
            p.grad = None

    def max_abs_diff(self) -> float:
        """Largest absolute deviation of the live weights from the snapshot."""
        with torch.no_grad():
            cur = self.model.state_dict()
            diffs = [float((cur[k].detach().cpu().float() - v.float()).abs().max())
                     for k, v in self.state.items() if v.numel() > 0 and v.is_floating_point()]
        return max(diffs) if diffs else 0.0

    def __enter__(self) -> "WeightSnapshot":
        return self

    def __exit__(self, *exc) -> bool:
        self.restore()
        return False


def select_train_params(model_or_bundle, train_params: Union[str, Callable, Sequence[str]] = "all"
                        ) -> list[tuple[str, nn.Parameter]]:
    """Named parameters selected by ``train_params``:

    ``'all'``; ``'mlp'`` (names containing ``.mlp.``); ``'attn'`` (``.attention.``);
    ``'last_n:<k>'`` (the last ``k`` transformer blocks; not the final LayerNorm or the
    unembedding); a callable ``name -> bool``; or an explicit list of parameter names.
    """
    model = _as_model(model_or_bundle)
    named = list(model.named_parameters())
    if train_params == "all":
        sel = named
    elif train_params == "mlp":
        sel = [(n, p) for n, p in named if ".mlp." in n]
    elif train_params == "attn":
        sel = [(n, p) for n, p in named if ".attention." in n]
    elif isinstance(train_params, str) and train_params.startswith("last_n:"):
        k = int(train_params.split(":", 1)[1])
        L = int(model.config.num_hidden_layers)
        tags = {f".layers.{l}." for l in range(max(L - k, 0), L)}
        sel = [(n, p) for n, p in named if any(t in n for t in tags)]
    elif callable(train_params):
        sel = [(n, p) for n, p in named if train_params(n)]
    elif isinstance(train_params, (list, tuple, set, frozenset)):
        names = set(train_params)
        sel = [(n, p) for n, p in named if n in names]
    else:
        raise ValueError(f"unknown train_params {train_params!r}")
    if not sel:
        raise ValueError(f"train_params={train_params!r} selects no parameters")
    return sel


def _make_optimizer(params, name: str, lr: float, weight_decay: float):
    name = str(name).lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}; expected one of {_OPTIMIZERS}")


def _check_fp32(model: nn.Module) -> None:
    dt = next(model.parameters()).dtype
    if dt != torch.float32:
        raise ValueError(f"TTT trains FP32 master weights (dossier convention); the model is "
                         f"{dt}. Load it with ModelSpec(dtype='float32').")


def _freeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
        p.grad = None


def _train_params_tag(train_params) -> str:
    if isinstance(train_params, str):
        return train_params
    if callable(train_params):
        return f"callable:{getattr(train_params, '__name__', 'fn')}"
    return f"names:{len(list(train_params))}"


# --------------------------------------------------------------------------------------
# losses / steps
# --------------------------------------------------------------------------------------
def snippet_nll(model_or_bundle, batch: dict[str, torch.Tensor]) -> float:
    """Mean token NLL (HF causal-LM loss, pads ignored) of a collated batch; eval mode,
    no grad; the model's training flag is restored afterwards."""
    model = _as_model(model_or_bundle)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                        labels=batch["labels"], use_cache=False)
        return float(out.loss.detach().float())
    finally:
        model.train(was_training)


def corpus_nll(mb: ModelBundle, snippets: Sequence[torch.Tensor]) -> pd.DataFrame:
    """``DataFrame(snippet, n_tokens, nll)`` with the mean NLL of every snippet under
    ``mb`` (a ready-made ``measure_fn`` for :func:`before_after`)."""
    rows = []
    for i, s in enumerate(snippets):
        b = _snippet_batch(s, mb)
        rows.append(dict(snippet=i, n_tokens=int(b["attention_mask"].sum()), nll=snippet_nll(mb, b)))
    return pd.DataFrame(rows, columns=["snippet", "n_tokens", "nll"])


def _train_step(model: nn.Module, opt, params: list[nn.Parameter], batch: dict,
                grad_clip: float) -> tuple[float, float]:
    """One AdamW/SGD step on a collated batch; returns (loss, pre-clip grad norm)."""
    model.train()
    opt.zero_grad(set_to_none=True)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                labels=batch["labels"], use_cache=False)
    loss = out.loss
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite training loss {float(loss)}")
    loss.backward()
    if grad_clip is not None and grad_clip > 0:
        gnorm = torch.nn.utils.clip_grad_norm_(params, float(grad_clip))
    else:
        gnorm = torch.norm(torch.stack([p.grad.detach().float().norm() for p in params if p.grad is not None]))
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(loss.detach().float()), float(gnorm)


# --------------------------------------------------------------------------------------
# memorisation speed
# --------------------------------------------------------------------------------------
@dataclass
class TTTResult:
    """NLL curve of one snippet under continued training.

    ``nll_curve[k]`` is the snippet's mean NLL after ``k`` update steps (``[0]`` before
    training), ``steps_to_threshold`` the first ``k`` with ``nll_curve[k] <= threshold``
    (None if never; 0 if already below), ``final_nll = nll_curve[-1]``. ``meta`` carries
    lr, steps, threshold, n_tokens, train_params, optimizer, weight_decay, grad_clip,
    seed, n_trainable, loss_curve (training loss per step), grad_norm_curve, auc_drop.
    """
    nll_curve: np.ndarray
    steps_to_threshold: Optional[int]
    final_nll: float
    meta: dict = field(default_factory=dict)

    @property
    def nll_start(self) -> float:
        return float(self.nll_curve[0])

    @property
    def auc_drop(self) -> float:
        """Sum over steps of (nll_start - nll_k): area between the curve and its start."""
        return float(np.sum(self.nll_curve[0] - self.nll_curve[1:]))

    def to_row(self) -> dict:
        return dict(n_tokens=self.meta.get("n_tokens"), nll_start=self.nll_start,
                    steps_to_threshold=self.steps_to_threshold, final_nll=self.final_nll,
                    auc_drop=self.auc_drop)


def memorisation_speed(mb: ModelBundle, snippet, lr: float = 1e-4, steps: int = 50,
                       threshold: float = 0.5, optimizer: str = "adamw", weight_decay: float = 0.0,
                       seed: int = 0, train_params="all", inplace: bool = False,
                       grad_clip: float = 1.0) -> TTTResult:
    """Continued training on ONE snippet (``Tensor[T]``; also ``[B,T]`` or a list for a
    small batch) and the snippet's own mean NLL before training and after every step
    (eval mode, no grad).

    AdamW (or ``'adam'`` / ``'sgd'``), gradient-norm clipping at ``grad_clip``, FP32.
    Only the parameters selected by ``train_params`` (see :func:`select_train_params`)
    get ``requires_grad``. Unless ``inplace=True`` the model's state dict is snapshotted
    before and restored afterwards (in a ``finally`` block, so a failure restores too);
    ``requires_grad`` is switched off again in every case.
    """
    model = mb.model
    _check_fp32(model)
    batch = _snippet_batch(snippet, mb)
    n_tokens = int(batch["attention_mask"].sum())
    snap = None if inplace else WeightSnapshot(model)
    was_training = model.training
    torch.manual_seed(int(seed))
    nll: list[float] = [snippet_nll(model, batch)]
    losses: list[float] = []
    gnorms: list[float] = []
    n_trainable = 0
    try:
        named = select_train_params(model, train_params)
        params = [p for _, p in named]
        n_trainable = int(sum(p.numel() for p in params))
        for p in params:
            p.requires_grad_(True)
        opt = _make_optimizer(params, optimizer, lr, weight_decay)
        for _ in range(int(steps)):
            l, g = _train_step(model, opt, params, batch, grad_clip)
            losses.append(l)
            gnorms.append(g)
            nll.append(snippet_nll(model, batch))
    finally:
        _freeze(model)
        if snap is not None:
            snap.restore()
        model.train(was_training)
    curve = np.asarray(nll, dtype=np.float64)
    hit = np.nonzero(curve <= float(threshold))[0]
    stt = int(hit[0]) if len(hit) else None
    meta = dict(lr=float(lr), steps=int(steps), threshold=float(threshold), n_tokens=n_tokens,
                train_params=_train_params_tag(train_params), optimizer=str(optimizer).lower(),
                weight_decay=float(weight_decay), grad_clip=grad_clip, seed=int(seed),
                n_trainable=n_trainable, loss_curve=np.asarray(losses, dtype=np.float64),
                grad_norm_curve=np.asarray(gnorms, dtype=np.float64), inplace=bool(inplace),
                restored=snap is not None, model=mb.spec.tag(), nll_start=float(curve[0]),
                auc_drop=float(np.sum(curve[0] - curve[1:])))
    return TTTResult(nll_curve=curve, steps_to_threshold=stt, final_nll=float(curve[-1]), meta=meta)


# --------------------------------------------------------------------------------------
# test-time training of a bundle
# --------------------------------------------------------------------------------------
def test_time_train(mb: ModelBundle, corpus: Sequence[torch.Tensor], lr: float = 1e-4,
                    steps: int = 200, batch: int = 4, seed: int = 0, control: Optional[str] = None,
                    inplace: bool = False, controls: Optional[dict] = None, train_params="all",
                    optimizer: str = "adamw", weight_decay: float = 0.0, grad_clip: float = 1.0,
                    log_every: int = 0) -> ModelBundle:
    """Continued training of ``mb`` on ``corpus`` (or the control-arm variant selected by
    ``control`` in ``{None, 'neutral', 'nofork', 'permuted'}``; ``neutral`` / ``nofork``
    need ``controls={'neutral': [...], 'nofork': [...]}``) for ``steps`` seeded shuffled
    batches of ``batch`` snippets (right-padded, pads masked and label-ignored).

    Returns a NEW :class:`~lsne.model.ModelBundle` with a deep-copied, trained model
    (``inplace=True`` trains and returns ``mb`` itself). The trained bundle carries the
    training statistics as the attribute ``ttt_stats`` (``loss`` and ``grad_norm`` per
    step, batch indices, control, corpus meta, hyper-parameters). All parameters are
    frozen again and the model is in eval mode afterwards.
    """
    arm = corpus_for_control(corpus, control, controls, seed=seed)
    arm = list(arm) if not isinstance(arm, Corpus) else arm
    if len(arm) == 0:
        raise ValueError("empty corpus")
    _check_fp32(mb.model)
    if inplace:
        bundle = mb
    else:
        bundle = ModelBundle(model=copy.deepcopy(mb.model), tokenizer=mb.tokenizer,
                             spec=replace(mb.spec), device=mb.device)
    model = bundle.model
    pad_id = pad_id_of(mb.tokenizer)
    sampler = _EpochSampler(len(arm), batch, seed)
    torch.manual_seed(int(seed))
    losses: list[float] = []
    gnorms: list[float] = []
    idx_log: list[list[int]] = []
    n_trainable = 0
    try:
        named = select_train_params(model, train_params)
        params = [p for _, p in named]
        n_trainable = int(sum(p.numel() for p in params))
        for p in params:
            p.requires_grad_(True)
        opt = _make_optimizer(params, optimizer, lr, weight_decay)
        for k in range(int(steps)):
            idx = sampler.next()
            b = collate([arm[i] for i in idx], pad_id, mb.device)
            l, g = _train_step(model, opt, params, b, grad_clip)
            losses.append(l)
            gnorms.append(g)
            idx_log.append(idx)
            if log_every and (k + 1) % log_every == 0:
                print(f"[ttt] step {k + 1}/{steps} loss {l:.4f} |g| {g:.3f}")
    finally:
        _freeze(model)
        model.eval()
    bundle.ttt_stats = dict(
        loss=np.asarray(losses, dtype=np.float64), grad_norm=np.asarray(gnorms, dtype=np.float64),
        batch_indices=idx_log, control=control, n_snippets=len(arm),
        n_tokens=int(sum(int(torch.as_tensor(s).numel()) for s in arm)), steps=int(steps),
        lr=float(lr), batch=int(batch), seed=int(seed), epochs=sampler.epoch,
        train_params=_train_params_tag(train_params), optimizer=str(optimizer).lower(),
        weight_decay=float(weight_decay), grad_clip=grad_clip, n_trainable=n_trainable,
        inplace=bool(inplace), corpus_meta=dict(getattr(arm, "meta", {}) or {}),
        model=mb.spec.tag())
    return bundle


# --------------------------------------------------------------------------------------
# before / after and the memorisation table
# --------------------------------------------------------------------------------------
def _as_frame(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if isinstance(x, pd.Series):
        return x.to_frame().T.reset_index(drop=True)
    if isinstance(x, dict):
        scalar = all(np.isscalar(v) or v is None for v in x.values())
        return pd.DataFrame([x]) if scalar else pd.DataFrame(x)
    if np.isscalar(x):
        return pd.DataFrame({"value": [x]})
    return pd.DataFrame(x)


def before_after(mb: ModelBundle, mb_trained: ModelBundle,
                 measure_fn: Callable[[ModelBundle], pd.DataFrame], label: str = "") -> pd.DataFrame:
    """``measure_fn(mb)`` and ``measure_fn(mb_trained)`` concatenated with the columns
    ``phase`` in ``{'before', 'after'}`` and ``label`` in front. ``measure_fn`` may also
    return a dict or a Series (one row)."""
    frames = []
    for phase, b in (("before", mb), ("after", mb_trained)):
        df = _as_frame(measure_fn(b))
        df.insert(0, "label", label)
        df.insert(0, "phase", phase)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


_TABLE_COLUMNS = ["checkpoint", "arm", "snippet", "n_tokens", "nll_start",
                  "steps_to_threshold", "final_nll", "auc_drop"]


def ttt_memorisation_table(mb: ModelBundle, corpus: Sequence[torch.Tensor],
                           controls: Optional[dict] = None, checkpoints: Optional[list] = None,
                           arm_name: str = "injection", max_snippets: Optional[int] = None,
                           include_permuted: bool = False, progress: bool = False,
                           **kw) -> pd.DataFrame:
    """:func:`memorisation_speed` for every snippet of every arm (``arm_name`` = the
    corpus itself, plus each entry of ``controls``, e.g. ``neutral`` / ``nofork`` /
    ``permuted``; ``include_permuted=True`` adds a permuted arm if none is given) at every
    checkpoint.

    ``checkpoints=None`` uses only ``mb``; otherwise each revision string is loaded via
    :func:`lsne.model.load_model` with ``ModelSpec(name=mb.spec.name, revision=ckpt,
    dtype='float32', device=str(mb.device))`` (``mb`` is reused for its own revision).
    Remaining ``**kw`` (lr, steps, threshold, optimizer, weight_decay, seed,
    train_params, grad_clip) go to :func:`memorisation_speed`.

    Returns a tidy ``DataFrame(checkpoint, arm, snippet, n_tokens, nll_start,
    steps_to_threshold, final_nll, auc_drop)``; ``steps_to_threshold`` is nullable Int64.
    """
    arms: dict[str, list] = {arm_name: list(corpus)}
    for name, snips in (controls or {}).items():
        if snips is None:
            continue
        if name == arm_name:
            raise ValueError(f"control arm name {name!r} collides with arm_name")
        arms[name] = list(snips)
    if include_permuted and "permuted" not in arms:
        arms["permuted"] = list(permute_corpus(corpus, seed=int(kw.get("seed", 0))))
    if max_snippets is not None:
        arms = {k: v[:int(max_snippets)] for k, v in arms.items()}
    ckpts = [None] if checkpoints is None else list(checkpoints)
    rows = []
    for ckpt in ckpts:
        if ckpt is None or ckpt == mb.spec.revision:
            b, name = mb, mb.spec.revision
        else:
            b = load_model(ModelSpec(name=mb.spec.name, revision=str(ckpt), dtype="float32",
                                     device=str(mb.device)))
            name = str(ckpt)
        for arm, snips in arms.items():
            for i, s in enumerate(snips):
                r = memorisation_speed(b, s, **kw)
                rows.append(dict(checkpoint=name, arm=arm, snippet=i, **r.to_row()))
                if progress:
                    print(f"[ttt] {name} {arm} snippet {i}: steps_to_threshold={r.steps_to_threshold} "
                          f"final_nll={r.final_nll:.3f}")
        if b is not mb:
            del b
    df = pd.DataFrame(rows, columns=_TABLE_COLUMNS)
    # from the row dicts, not the frame: pandas coerces None -> NaN in a float column
    df["steps_to_threshold"] = pd.array([r["steps_to_threshold"] for r in rows], dtype="Int64")
    return df
