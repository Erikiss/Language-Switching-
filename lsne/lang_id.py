"""Deterministic, offline word-level language identification (English 'E' / German 'D'
/ neither 'N') for stimuli and generated text.

Evidence per word, in order of precedence:

1. digits, punctuation-only strings -> 'N';
2. function-word / frequent-word lexicons (`data/lexicon/function_words.json`): a word
   found in exactly one lexicon is decided by it;
3. character cues: umlauts and sharp s, ``sch``, ``tz``, ``pf``, German suffixes
   (-ung, -keit, -heit, -lich, -isch, -chen, -schaft, -ieren/-iert, ...), German verb
   endings (-ibt, -eht, -ingt, -icht, ...) versus English cues (``th``, ``wh``, ``sh``,
   ``ght``, ``oo``, ``ea``, ``ay``, ``ow``, bare ``c``, -ing, -tion, -ly, -ness, -ment,
   -ous, -ity, -ful, -ed, -'s, ...); a net score of magnitude >= `DECISIVE` decides;
4. everything else (words in both lexicons, weak cues, capitalised nouns) is resolved
   from the decided labels of the two neighbouring words on each side (weights 1.0 and
   0.5, halved across a sentence boundary) added to the word's own cue score.  A
   capitalised mid-sentence word without decisive evidence is a German noun in a German
   context and a proper name ('N') in an English context or without context; a
   sentence-initial capitalised unknown word without any cue is a name ('N').

`label_tokens(tokenizer, ids)` labels every token with the label of the word it
belongs to (a word is a maximal run of tokens starting at a word-start token; whitespace
and punctuation tokens are 'N').  `label_text(text)` labels whitespace-delimited words.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

LEXICON_PATH = Path(__file__).resolve().parent.parent / "data" / "lexicon" / "function_words.json"

LABELS = ("E", "D", "N")
DECISIVE = 2.0            # |cue score| needed for a cue-only decision
CONTEXT_WEIGHTS = (1.0, 0.5)   # nearest and second-nearest word on each side
SENTENCE_DISCOUNT = 0.5   # context weight factor across a sentence boundary
_SENTENCE_PUNCT = ".!?:;"
_CLOSING = "\"'’”)]»"
_OPENING = "\"'‘“(„«»‚["

# ---- character cues ------------------------------------------------------------------
# (pattern, weight); positive = German, negative = English.  Applied to the lowercase
# word core (letters, inner apostrophes and hyphens only).
_GERMAN_SUBSTR = (("ä", 3.0), ("ö", 3.0), ("ü", 3.0), ("ß", 3.0), ("sch", 2.0), ("tz", 2.0),
                  ("pf", 2.0), ("tsch", 1.0), ("eu", 1.0), ("ei", 1.0), ("ie", 0.25), ("ch", 0.5),
                  ("z", 1.0))
_GERMAN_PREFIX = (("ver", 1.5), ("vor", 1.0), ("zer", 1.5), ("ent", 0.5), ("ge", 0.5), ("be", 0.5),
                  ("er", 0.5), ("um", 0.5), ("zu", 1.0))
_GERMAN_SUFFIX = (("ung", 2.0), ("ungen", 2.0), ("keit", 3.0), ("heit", 3.0), ("lich", 2.0),
                  ("lichen", 2.0), ("isch", 2.0), ("ische", 2.0), ("ischen", 2.0), ("chen", 1.5),
                  ("lein", 2.0), ("schaft", 3.0), ("ieren", 2.0), ("iert", 2.0), ("ierte", 2.0),
                  ("igen", 1.5), ("igt", 1.5), ("nis", 1.0), ("tum", 1.0), ("bar", 0.5), ("los", 0.5),
                  ("ibt", 2.0), ("eht", 2.0), ("ingt", 2.0), ("icht", 1.5), ("ehrt", 2.0),
                  ("ndet", 2.0), ("iest", 1.5), ("mmt", 2.0), ("llt", 2.0), ("nnt", 2.0),
                  ("ckt", 2.0), ("tet", 1.5), ("eint", 1.5), ("ert", 0.5), ("en", 0.75),
                  ("ern", 0.5), ("em", 0.5), ("ig", 1.0), ("ik", 0.5))
_ENGLISH_SUBSTR = (("th", 1.5), ("wh", 2.0), ("sh", 1.5), ("ght", 2.0), ("oo", 1.0), ("ea", 1.0),
                   ("ay", 1.0), ("ow", 1.0), ("aw", 1.0), ("ew", 0.5), ("ey", 0.5), ("oy", 1.0),
                   ("ou", 0.5), ("y", 0.5), ("'s", 1.5), ("'t", 2.0), ("'re", 2.0),
                   ("'ll", 2.0), ("'ve", 2.0), ("'d", 1.0), ("'m", 1.0))
_ENGLISH_SUFFIX = (("ing", 2.0), ("tion", 2.0), ("sion", 1.5), ("ly", 2.0), ("ness", 2.0),
                   ("ment", 1.5), ("ous", 2.0), ("ity", 2.0), ("ful", 2.0), ("less", 2.0),
                   ("ed", 1.5), ("ies", 1.5), ("ers", 0.5), ("ally", 2.0), ("ship", 2.0),
                   ("able", 1.5), ("ible", 1.5), ("ive", 1.0), ("ves", 1.0), ("ism", 0.5),
                   ("s", 1.0), ("al", 0.75), ("ic", 0.5), ("age", 1.0), ("ure", 1.0),
                   ("ant", 0.5), ("ent", 0.5))
_WORD_CORE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)


# ======================================================================================
# lexicon
# ======================================================================================
@lru_cache(maxsize=4)
def _load_lexicon_cached(path: str) -> tuple[frozenset, frozenset]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return frozenset(w.lower() for w in d["D"]), frozenset(w.lower() for w in d["E"])


def load_lexicon(path: str | Path = LEXICON_PATH) -> dict[str, set[str]]:
    """Return ``{"D": set, "E": set}`` of lowercase lexicon words."""
    D, E = _load_lexicon_cached(str(path))
    return {"D": set(D), "E": set(E)}


# ======================================================================================
# single-word evidence
# ======================================================================================
def word_core(word: str) -> str:
    """Alphabetic core of a word (inner apostrophes/hyphens kept), lowercase; '' if none."""
    m = _WORD_CORE.search(word)
    return m.group().lower() if m else ""


def _has_bare_c(w: str) -> bool:
    """A 'c' not followed by h/k (English cue: records, discussion, compares)."""
    return any(ch == "c" and (i + 1 >= len(w) or w[i + 1] not in "hk") for i, ch in enumerate(w))


def cue_score(core: str) -> float:
    """Net character-cue score of a lowercase word core (> 0 German, < 0 English)."""
    if not core:
        return 0.0
    s = 0.0
    for pat, wgt in _GERMAN_SUBSTR:
        if pat in core:
            s += wgt
    for pat, wgt in _GERMAN_PREFIX:
        if core.startswith(pat) and len(core) > len(pat) + 2:
            s += wgt
    for pat, wgt in _GERMAN_SUFFIX:
        if core.endswith(pat) and len(core) > len(pat) + 1:
            s += wgt
    for pat, wgt in _ENGLISH_SUBSTR:
        if pat == "sh":
            if "sh" in core.replace("sch", ""):
                s -= wgt
        elif pat in core:
            s -= wgt
    for pat, wgt in _ENGLISH_SUFFIX:
        if core.endswith(pat) and len(core) > len(pat) + 1:
            if pat == "s" and core.endswith(("ss", "us", "is", "as", "es")):
                continue
            s -= wgt
    if _has_bare_c(core):
        s -= 1.5
    return s


def _is_capitalised(word: str, core: str) -> bool:
    first = word.lstrip(_OPENING)[:1]
    return first.isalpha() and first.isupper() and not core.isupper()


def word_evidence(word: str, sentence_initial: bool = True,
                  lexicon: Optional[dict[str, set[str]]] = None) -> tuple[Optional[str], float, bool]:
    """(decided label or None, cue score, capitalised-mid-sentence flag) for one word.

    The label is decided ('E', 'D' or 'N') by digits/punctuation, by a lexicon hit in
    exactly one language, by a cue score of magnitude >= `DECISIVE`, or (as 'N') for a
    sentence-initial capitalised unknown word without any cue (a proper name)."""
    core = word_core(word)
    if not core:
        return "N", 0.0, False
    if any(ch.isdigit() for ch in word):
        return "N", 0.0, False
    if lexicon is None:
        D, E = _load_lexicon_cached(str(LEXICON_PATH))
    else:
        D, E = lexicon["D"], lexicon["E"]
    in_D, in_E = core in D, core in E
    if not (in_D or in_E):
        stem = re.split(r"['’]", core)[0]        # possessive/clitic: "kafka's" -> "kafka"
        if stem != core and (stem in D or stem in E):
            in_D, in_E = (stem in D and not core.endswith(("'s", "’s"))), stem in E
    if in_D and not in_E:
        return "D", 0.0, False
    if in_E and not in_D:
        return "E", 0.0, False
    capitalised = _is_capitalised(word, core)
    score = cue_score(core)
    if in_D and in_E:
        return None, score * 0.5, False          # ambiguous lexicon word: context decides
    if score >= DECISIVE:
        return "D", score, False
    if score <= -DECISIVE:
        return "E", score, False
    if capitalised and sentence_initial and score == 0.0:
        return "N", 0.0, False                   # cue-less sentence-initial name
    return None, score, capitalised and not sentence_initial


def _resolve(score: float, capitalised: bool, ctx: float) -> str:
    """Final label from the word's cue score and the weighted context vote `ctx`
    (> 0 German neighbours, < 0 English neighbours)."""
    if capitalised:
        if ctx > 0:
            return "D"
        if ctx < 0:
            return "D" if score >= 1.5 else "N"
        return "D" if score >= 1.0 else "N"
    total = score + ctx
    if total > 0:
        return "D"
    if total < 0:
        return "E"
    return "N"


def classify_word(word: str, left: Sequence[str] = (), right: Sequence[str] = (),
                  sentence_initial: bool = True,
                  lexicon: Optional[dict[str, set[str]]] = None) -> str:
    """Label a single word ('E', 'D' or 'N').  `left`/`right` are the labels of the
    neighbouring words (nearest first, up to two per side) used for ambiguous words."""
    label, score, capitalised = word_evidence(word, sentence_initial, lexicon)
    if label is not None:
        return label
    ctx = 0.0
    for side in (left, right):
        for lab, w in zip(side, CONTEXT_WEIGHTS):
            ctx += w if lab == "D" else (-w if lab == "E" else 0.0)
    return _resolve(score, capitalised, ctx)


# ======================================================================================
# sequences of words
# ======================================================================================
def _ends_sentence(word: str) -> bool:
    w = word.rstrip(_CLOSING)
    return bool(w) and w[-1] in _SENTENCE_PUNCT


def label_words(words: Sequence[str], lexicon: Optional[dict[str, set[str]]] = None) -> list[str]:
    """Label a sequence of words (strings in text order).

    Pass 0 decides words by lexicon / strong cues; pass 1 resolves the others from the
    decided labels of their two nearest word neighbours per side (weights 1.0 / 0.5,
    halved across a sentence boundary) plus their own cue score; pass 2 repeats this
    with the pass-1 labels for words that had no context in pass 1."""
    n = len(words)
    cores = [word_core(w) for w in words]
    real = [i for i in range(n) if cores[i]]                 # words with letters
    # sentence end flags per real word (trailing punctuation-only tokens count)
    ends = [False] * n
    for p, i in enumerate(real):
        nxt = real[p + 1] if p + 1 < len(real) else n
        ends[i] = any(_ends_sentence(words[j]) for j in range(i, nxt))
    labels: list[Optional[str]] = ["N"] * n
    scores = [0.0] * n
    caps = [False] * n
    for p, i in enumerate(real):
        sent_init = p == 0 or ends[real[p - 1]]
        labels[i], scores[i], caps[i] = word_evidence(words[i], sent_init, lexicon)
    pos_of = {i: p for p, i in enumerate(real)}

    def context(i: int, labs: Sequence[Optional[str]]) -> float:
        ctx = 0.0
        p = pos_of[i]
        for direction in (-1, 1):
            crossed = False
            for step, w in enumerate(CONTEXT_WEIGHTS, start=1):
                q = p + direction * step
                if not 0 <= q < len(real):
                    break
                j = real[q]
                crossed = crossed or ends[j if direction < 0 else real[q - 1]]
                wt = w * (SENTENCE_DISCOUNT if crossed else 1.0)
                lab = labs[j]
                if lab == "D":
                    ctx += wt
                elif lab == "E":
                    ctx -= wt
        return ctx

    undecided = [i for i in real if labels[i] is None]
    pass1 = list(labels)
    no_ctx: list[int] = []
    for i in undecided:
        c = context(i, labels)
        if c == 0.0:
            no_ctx.append(i)
        pass1[i] = _resolve(scores[i], caps[i], c)
    final = list(pass1)
    for i in no_ctx:
        final[i] = _resolve(scores[i], caps[i], context(i, pass1))
    return [lab if lab is not None else "N" for lab in final]


def split_words(text: str) -> list[str]:
    """Whitespace-delimited words of `text` (punctuation attached)."""
    return re.findall(r"\S+", text)


def label_text(text: str, lexicon: Optional[dict[str, set[str]]] = None) -> list[str]:
    """Per-word labels ('E'/'D'/'N') for the whitespace-delimited words of `text`
    (aligned with `split_words(text)`)."""
    return label_words(split_words(text), lexicon)


# ======================================================================================
# token level
# ======================================================================================
def token_words(tokenizer, ids: Sequence[int]) -> list[tuple[int, int, str]]:
    """Group tokens into words: ``(start, end, text)`` with half-open token spans.

    A new word starts at the first token, at a token whose decoded string starts with
    whitespace (word_start), after a whitespace-only or punctuation-only token, and at
    every whitespace-only or punctuation-only token (which form groups of their own).
    Each group is decoded jointly, so multi-byte characters split across tokens are
    recovered."""
    strs = [tokenizer.decode([int(i)]) for i in ids]
    groups: list[list[int]] = []
    prev_break = True
    for t, s in enumerate(strs):
        is_ws = s != "" and s.strip() == ""
        is_punct = (not is_ws) and not any(ch.isalnum() for ch in s) and "�" not in s
        if t == 0 or prev_break or is_ws or is_punct or s[:1].isspace():
            groups.append([t])
        else:
            groups[-1].append(t)
        prev_break = is_ws or is_punct
    out = []
    for g in groups:
        text = tokenizer.decode([int(ids[t]) for t in g]).strip()
        out.append((g[0], g[-1] + 1, text))
    return out


def label_tokens(tokenizer, ids: Sequence[int], lexicon: Optional[dict[str, set[str]]] = None) -> list[str]:
    """Per-token labels: each token gets the label of the word it belongs to;
    whitespace-only and punctuation-only tokens are 'N'."""
    ids = [int(i) for i in ids]
    if not ids:
        return []
    words = token_words(tokenizer, ids)
    wlabels = label_words([w[2] for w in words], lexicon)
    out = ["N"] * len(ids)
    for (s, e, _), lab in zip(words, wlabels):
        for t in range(s, e):
            out[t] = lab
    return out


# ======================================================================================
# sequence statistics
# ======================================================================================
def language_fraction(labels: Sequence[str], lang: str = "D") -> float:
    """Fraction of E/D-labelled items (ignoring 'N') carrying `lang`; 0.0 if none."""
    ed = [x for x in labels if x in ("E", "D")]
    if not ed:
        return 0.0
    return sum(1 for x in ed if x == lang) / len(ed)


def switch_positions(labels: Sequence[str]) -> list[int]:
    """Indices at which the (E/D) language differs from the previous E/D label
    ('N' items are transparent)."""
    out: list[int] = []
    prev: Optional[str] = None
    for i, x in enumerate(labels):
        if x not in ("E", "D"):
            continue
        if prev is not None and x != prev:
            out.append(i)
        prev = x
    return out


def switch_count(labels: Sequence[str]) -> int:
    """Number of E<->D switches in a label sequence."""
    return len(switch_positions(labels))


def inter_switch_intervals(labels: Sequence[str]) -> list[int]:
    """Distances (in items) between consecutive switch positions."""
    pos = switch_positions(labels)
    return [b - a for a, b in zip(pos, pos[1:])]


def language_runs(labels: Sequence[str]) -> list[tuple[str, int]]:
    """Run-length encoding of the E/D labels ('N' transparent): [(lang, length), ...]."""
    runs: list[tuple[str, int]] = []
    for x in labels:
        if x not in ("E", "D"):
            continue
        if runs and runs[-1][0] == x:
            runs[-1] = (x, runs[-1][1] + 1)
        else:
            runs.append((x, 1))
    return runs
