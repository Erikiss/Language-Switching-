"""Vocabulary and Hugging Face tokenizer for the ITR token stream (docs/TRACE_SOURCE_SPEC.md §5).

The vocabulary has two parts:

* **closed classes** (fixed, identical for every corpus, so ids are stable): the special
  tokens of the spec (``[PAD]`` = 0, ``[BOS]`` = 1, ``[EOS]`` = 2, structural tokens,
  ``[A_OOV]``/``[F_OOV]``/``[T_OOV]``, ``[UNK]``), depth tokens ``[D0]``..``[D32]``,
  call-type and error tokens, the numeric tokens (``N0``, ``N10e-1``..``N90e-1`` for 1..9,
  every ``N{m}e{k}`` for m 10..99 and k 0..77) and everything else
  :func:`lsne.chain.itr.fixed_tokens` enumerates (slot buckets ``S:*``, ``A:none``,
  ``V:``/``G:``-prefixed numerics);
* **open classes** from the corpus: the ``n_addresses`` most frequent addresses
  (``A:0x…``, the DAO contracts always included), ``n_selectors`` function selectors
  (``F:0x…``) and ``n_topics`` event topics (``T:0x…``), ordered by frequency (ties
  lexicographically, so the build is deterministic).

Everything outside the vocabulary maps to ``[A_OOV]`` / ``[F_OOV]`` / ``[T_OOV]`` (exactly as
:func:`lsne.chain.itr.serialize` does with a vocab), anything else to ``[UNK]``.

The tokenizer is a ``tokenizers.WordLevel`` model with ``WhitespaceSplit`` (ITR tokens
contain no whitespace, so ``" ".join(tokens)`` round-trips exactly), wrapped as a
``PreTrainedTokenizerFast``; ``make_tokenizer(vocab, out_dir)`` saves it together with
``vocab.json`` so that ``AutoTokenizer.from_pretrained(out_dir)`` works.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from . import itr
from .labels import DAO_ADDRESSES

PAD, BOS, EOS, UNK = "[PAD]", "[BOS]", "[EOS]", "[UNK]"
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
OOV = {"A": "[A_OOV]", "F": "[F_OOV]", "T": "[T_OOV]"}
OPEN_PREFIXES = ("A:", "F:", "T:")
MAX_DEPTH_TOKEN = 32
VOCAB_FILE = "vocab.json"

_HEAD = [PAD, BOS, EOS, "[TX]", "[/TX]", "[START]", "[END]", "[CALL]", "[STATE]", "[LOG]", "[INs]", "[OUTs]",
         "[W]", "[R]", "[WORD]", "[A_OOV]", "[F_OOV]", "[T_OOV]", UNK]


def _dedupe(seq: Iterable[str]) -> list:
    seen, out = set(), []
    for t in seq:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


#: The spec's SPECIAL list: special/structural tokens, depth, call-type, error and (bare) numeric tokens.
SPECIAL = _dedupe(_HEAD + [f"[D{d}]" for d in range(MAX_DEPTH_TOKEN + 1)]
                  + [f"[CT:{c}]" for c in itr.CALL_TYPES] + [f"[ERR:{e}]" for e in itr.ERROR_KINDS]
                  + itr.numeric_tokens())


def closed_tokens(max_depth_token: int = MAX_DEPTH_TOKEN) -> list:
    """Every closed-class token, in id order: :data:`SPECIAL` followed by the remaining
    tokens of :func:`lsne.chain.itr.fixed_tokens` (slot buckets, ``*:none``, ``V:``/``G:`` numerics)."""
    return _dedupe(SPECIAL + itr.fixed_tokens(max_depth_token))


def _is_open(tok: str) -> bool:
    return tok[:2] in OPEN_PREFIXES and not tok.endswith(":none")


# ======================================================================================
# vocabulary
# ======================================================================================
@dataclass
class TraceVocab:
    """Token list (index = id) plus the open-class members; ``meta`` holds build statistics."""
    tokens: list
    top_addresses: list
    top_selectors: list
    top_topics: list
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.tokens = list(self.tokens)
        self.top_addresses = list(self.top_addresses)
        self.top_selectors = list(self.top_selectors)
        self.top_topics = list(self.top_topics)
        self._index = {t: i for i, t in enumerate(self.tokens)}
        if len(self._index) != len(self.tokens):
            raise ValueError("TraceVocab.tokens contains duplicates")
        for t in (PAD, BOS, EOS, UNK, *OOV.values()):
            if t not in self._index:
                raise ValueError(f"TraceVocab is missing the special token {t}")

    # ---- lookup ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.tokens)

    def __contains__(self, tok: str) -> bool:
        return tok in self._index

    @property
    def index(self) -> dict:
        """``{token: id}``."""
        return self._index

    @property
    def unk_id(self) -> int:
        return self._index[UNK]

    @property
    def pad_id(self) -> int:
        return self._index[PAD]

    @property
    def bos_id(self) -> int:
        return self._index[BOS]

    @property
    def eos_id(self) -> int:
        return self._index[EOS]

    def map_token(self, tok: str) -> str:
        """In-vocabulary token, else ``[A_OOV]``/``[F_OOV]``/``[T_OOV]`` for open classes, else ``[UNK]``."""
        if tok in self._index:
            return tok
        if tok[:2] in OPEN_PREFIXES:
            return OOV[tok[0]]
        return UNK

    def map_tokens(self, tokens: Sequence[str]) -> list:
        """Map a raw ITR token list (``serialize(rec)`` without vocab) to vocabulary tokens;
        equal to ``serialize(rec, vocab)``."""
        idx = self._index
        return [t if t in idx else self.map_token(t) for t in tokens]

    def ids(self, tokens: Sequence[str]) -> list:
        """Token ids of a (raw or mapped) token list (OOV / UNK mapping included)."""
        idx = self._index
        return [idx[t] if t in idx else idx[self.map_token(t)] for t in tokens]

    def decode(self, ids: Sequence[int]) -> list:
        return [self.tokens[int(i)] for i in ids]

    # ---- persistence -----------------------------------------------------------------
    def to_dict(self) -> dict:
        return dict(tokens=self.tokens, top_addresses=self.top_addresses, top_selectors=self.top_selectors,
                    top_topics=self.top_topics, meta=self.meta)

    def save(self, path: str) -> str:
        """Write ``vocab.json`` (``path`` is a directory or a ``.json`` file); returns the file path."""
        p = os.path.join(path, VOCAB_FILE) if not str(path).endswith(".json") else str(path)
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)
        os.replace(tmp, p)
        return p

    @classmethod
    def load(cls, path: str) -> "TraceVocab":
        """Read ``vocab.json`` from a directory or a file path."""
        p = os.path.join(path, VOCAB_FILE) if os.path.isdir(path) else str(path)
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(tokens=d["tokens"], top_addresses=d.get("top_addresses", []),
                   top_selectors=d.get("top_selectors", []), top_topics=d.get("top_topics", []),
                   meta=d.get("meta", {}))

    @classmethod
    def from_tokenizer(cls, tokenizer) -> "TraceVocab":
        """Rebuild the vocabulary from a saved tokenizer (when ``vocab.json`` is missing)."""
        v = tokenizer.get_vocab()
        tokens = [t for t, _ in sorted(v.items(), key=lambda kv: kv[1])]
        if [v[t] for t in tokens] != list(range(len(tokens))):
            raise ValueError("tokenizer ids are not contiguous 0..V-1")
        return cls(tokens=tokens, top_addresses=[t[2:] for t in tokens if t.startswith("A:") and _is_open(t)],
                   top_selectors=[t[2:] for t in tokens if t.startswith("F:") and _is_open(t)],
                   top_topics=[t[2:] for t in tokens if t.startswith("T:") and _is_open(t)],
                   meta=dict(rebuilt_from_tokenizer=True))


def item_tokens(item, vocab: Optional[TraceVocab] = None, serialize_kw: Optional[dict] = None) -> list:
    """ITR tokens of one item: a :class:`~lsne.chain.itr.TxRecord` (serialised), a token list,
    a dict with ``"tokens"`` (or ``"record"``) or an object with a ``tokens`` attribute.
    With ``vocab`` the tokens are mapped into it (OOV tokens), else raw."""
    if isinstance(item, itr.TxRecord):
        return itr.serialize(item, vocab, **(serialize_kw or {}))
    toks = None
    if isinstance(item, dict):
        if item.get("tokens") is not None:
            toks = item["tokens"]
        elif item.get("record") is not None:
            return itr.serialize(itr.TxRecord.from_dict(item["record"]), vocab, **(serialize_kw or {}))
    elif isinstance(item, str):
        toks = item.split()
    elif getattr(item, "tokens", None) is not None:
        toks = item.tokens
    elif getattr(item, "record", None) is not None:
        return item_tokens(item.record, vocab, serialize_kw)
    elif isinstance(item, (list, tuple)):
        toks = item
    if toks is None:
        raise TypeError(f"cannot get ITR tokens from {type(item).__name__}")
    toks = [str(t) for t in toks]
    return vocab.map_tokens(toks) if vocab is not None else toks


def build_vocab(records_iter: Iterable, n_addresses: int = 20000, n_selectors: int = 5000, n_topics: int = 2000,
                always_include: Iterable[str] = tuple(DAO_ADDRESSES.values()), min_count: int = 1,
                max_depth_token: int = MAX_DEPTH_TOKEN, serialize_kw: Optional[dict] = None) -> TraceVocab:
    """Count the open-class tokens (``A:``/``F:``/``T:``, without ``*:none``) of the raw ITR
    serialisation of every item (records, token lists, shard dicts; see :func:`item_tokens`)
    and keep the most frequent ones (at least ``min_count`` occurrences).  ``always_include``
    addresses (default: the DAO contracts) come first and count towards ``n_addresses``.

    ``meta``: n_items, n_tokens, distinct open-class values seen per class, the share of
    open-class *occurrences* covered by the vocabulary per class (``coverage_A`` …), sizes."""
    counts = {"A": Counter(), "F": Counter(), "T": Counter()}
    n_items = n_tokens = 0
    for item in records_iter:
        toks = item_tokens(item, None, serialize_kw)
        n_items += 1
        n_tokens += len(toks)
        for t in toks:
            if _is_open(t):
                counts[t[0]][t[2:]] += 1
    always = _dedupe(str(a).lower() for a in (always_include or ()))

    def top(c: Counter, n: int, first: Sequence[str] = ()) -> list:
        keep = list(first)[: max(int(n), len(first))]
        have = set(keep)
        ranked = sorted(((v, k) for k, v in c.items() if v >= int(min_count) and k not in have),
                        key=lambda x: (-x[0], x[1]))
        keep += [k for _, k in ranked[: max(0, int(n) - len(keep))]]
        return keep

    top_a = top(counts["A"], n_addresses, always)
    top_f = top(counts["F"], n_selectors)
    top_t = top(counts["T"], n_topics)
    closed = closed_tokens(max_depth_token)
    tokens = _dedupe(closed + [f"A:{a}" for a in top_a] + [f"F:{f}" for f in top_f] + [f"T:{t}" for t in top_t])

    def coverage(c: Counter, keep: list) -> float:
        tot = sum(c.values())
        if tot == 0:
            return float("nan")
        ks = set(keep)
        return sum(v for k, v in c.items() if k in ks) / tot

    meta = dict(n_items=n_items, n_tokens=n_tokens, n_closed=len(closed), size=len(tokens),
                n_addresses=len(top_a), n_selectors=len(top_f), n_topics=len(top_t),
                distinct_addresses=len(counts["A"]), distinct_selectors=len(counts["F"]),
                distinct_topics=len(counts["T"]), coverage_A=coverage(counts["A"], top_a),
                coverage_F=coverage(counts["F"], top_f), coverage_T=coverage(counts["T"], top_t),
                params=dict(n_addresses=int(n_addresses), n_selectors=int(n_selectors), n_topics=int(n_topics),
                            min_count=int(min_count), max_depth_token=int(max_depth_token),
                            always_include=always, serialize_kw=dict(serialize_kw or {})))
    return TraceVocab(tokens=tokens, top_addresses=top_a, top_selectors=top_f, top_topics=top_t, meta=meta)


# ======================================================================================
# tokenizer
# ======================================================================================
def make_tokenizer(vocab: TraceVocab, out_dir: Optional[str] = None, model_max_length: int = 512):
    """``PreTrainedTokenizerFast`` over ``tokenizers.WordLevel`` + ``WhitespaceSplit`` with
    unk ``[UNK]`` and pad/bos/eos ``[PAD]``/``[BOS]``/``[EOS]``; ids equal ``vocab.index``.
    With ``out_dir`` it is saved there (``save_pretrained``) together with ``vocab.json``,
    so ``AutoTokenizer.from_pretrained(out_dir)`` loads it."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tk = Tokenizer(models.WordLevel(vocab=dict(vocab.index), unk_token=UNK))
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token=UNK, pad_token=PAD, bos_token=BOS,
                                  eos_token=EOS, model_max_length=int(model_max_length))
    got = tok.get_vocab()
    if len(got) != len(vocab) or any(got.get(t) != i for t, i in ((PAD, vocab.pad_id), (BOS, vocab.bos_id),
                                                                 (EOS, vocab.eos_id), (UNK, vocab.unk_id))):
        raise RuntimeError("tokenizer ids diverge from the vocabulary")
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        tok.save_pretrained(out_dir)
        vocab.save(out_dir)
    return tok


def load_tokenizer(path: str):
    """``AutoTokenizer.from_pretrained(path)``."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


def encode_records(records: Iterable, vocab: TraceVocab, tokenizer=None, max_len: Optional[int] = 512,
                   serialize_kw: Optional[dict] = None) -> list:
    """Token ids per item (records, token lists, shard dicts), truncated to ``max_len``
    (``None``: no truncation).  Ids come from ``tokenizer``'s vocabulary when given (it must
    agree with ``vocab``) — identical to ``tokenizer(" ".join(tokens))`` — else from ``vocab``."""
    index = vocab.index
    if tokenizer is not None:
        tv = tokenizer.get_vocab()
        if len(tv) != len(index) or any(tv.get(t) != i for t, i in ((UNK, vocab.unk_id), (EOS, vocab.eos_id))):
            raise ValueError("tokenizer and vocab disagree (different vocabulary files?)")
        index = tv
    unk = index[UNK]
    out = []
    for item in records:
        toks = item_tokens(item, vocab, serialize_kw)
        if max_len is not None:
            toks = toks[: int(max_len)]
        out.append([index.get(t, unk) for t in toks])
    return out


def oov_rates(token_lists: Iterable[Sequence[str]]) -> dict:
    """OOV statistics of vocabulary-mapped token lists: ``rate_A``/``rate_F``/``rate_T`` =
    ``[X_OOV]`` / (``[X_OOV]`` + in-vocabulary ``X:`` tokens, ``X:none`` excluded),
    ``rate_open`` over all three classes, ``rate_unk`` = ``[UNK]`` / all tokens, plus the counts."""
    n = dict(A=0, F=0, T=0)
    oov = dict(A=0, F=0, T=0)
    n_tok = n_unk = n_lists = 0
    inv = {v: k for k, v in OOV.items()}
    for toks in token_lists:
        n_lists += 1
        for t in toks:
            n_tok += 1
            if t in inv:
                oov[inv[t]] += 1
            elif t == UNK:
                n_unk += 1
            elif _is_open(t):
                n[t[0]] += 1
    out = dict(n_lists=n_lists, n_tokens=n_tok, n_unk=n_unk,
               rate_unk=(n_unk / n_tok) if n_tok else float("nan"))
    for c in "AFT":
        tot = n[c] + oov[c]
        out[f"n_{c}"] = tot
        out[f"n_{c}_oov"] = oov[c]
        out[f"rate_{c}"] = oov[c] / tot if tot else float("nan")
    tot = sum(n.values()) + sum(oov.values())
    out["rate_open"] = sum(oov.values()) / tot if tot else float("nan")
    return out
