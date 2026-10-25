"""Deterministic, offline word-level language identification (English 'E' / German 'D'
/ neither 'N') for stimuli and generated text.

Evidence per word, in order of precedence:

1. digits, punctuation-only strings -> 'N'; English clitics (``'s``, ``n't``, ``'ll``,
   ...) -> 'E'; proper names of the stimulus schema (``data/lexicon/function_words.json``
   key ``"N"``: Lena, Tomas, Kraft, ..., also with a genitive ``-s``) -> 'N';
2. the lexicons (keys ``"D"`` / ``"E"`` of the same file: function words, auxiliaries
   and the common verbs / nouns / adjectives of everyday narratives with their inflected
   forms): a lowercase or sentence-initial word found in exactly one lexicon is decided
   by it; a *capitalised mid-sentence* word found only in the English lexicon is not
   (German nouns are capitalised: ``im letzten Moment``) and goes to step 4;
3. character cues: umlauts and ``ß`` are decisive for German.  Everything else is weak
   evidence - ``sch``, ``tz``, ``pf``, ``ei``/``eu``, German suffixes (-ung, -keit, -heit,
   -lich, -isch, -chen, -schaft, -ieren/-iert, ...) and verb endings (-eht, -ibt, -mmt,
   -llt, ...) versus English cues (``th``, ``wh``, ``sh``, ``gh``, ``oo``, ``ea``, ``ay``,
   ``ow``, ``tch``, ``dge``, bare ``c``, -ing, -tion, -ly, -ness, -ed, -ous, -ful, ...).
   Only a lowercase or sentence-initial word whose net cue score reaches ``DECISIVE`` is
   decided by cues alone;
4. everything else (words in both lexicons, unknown words with weak cues, capitalised
   mid-sentence words) is resolved from the labels of its neighbouring words *within the
   same sentence* (sentence boundaries at ``. ! ?``): the nearest word on each side with
   weight 1.0, the second with weight 0.5 (undecided words count 0, names and digits are
   transparent), plus half of the word's own cue score.  A net vote of magnitude >=
   ``MARGIN`` decides; a conflict between the two sides (a language boundary) needs a
   clear majority and a one-sided context contradicted by the word's own cues is left
   open, i.e. the word is 'N' rather than guessed.  A capitalised unknown word is a
   German noun in a German context and a proper name ('N') in an English context.  Two
   positional rules: ``am`` is English only after ``I``; English ``an`` precedes a vowel.
   Words whose decided context was weak get a second pass in which the resolved labels
   of their neighbours count with half weight.

`label_tokens(tokenizer, ids)` labels every token with the label of the word it
belongs to (a word is a maximal run of tokens starting at a word-start token; whitespace
and punctuation tokens are 'N').  `label_text(text)` labels whitespace-delimited words.
`evaluate_on_families(families_dir, tokenizer)` reports the accuracy against the segment
languages of the stimulus bank per (family, formulation, language).

The sequence statistics (`switch_positions`, `switch_count`, `language_runs`, ...) treat
'N' as transparent, so `switch_count` is a lower bound on the designed number of
switches: a one-word segment that is a name is invisible (see `switch_count`).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

LEXICON_PATH = Path(__file__).resolve().parent.parent / "data" / "lexicon" / "function_words.json"

LABELS = ("E", "D", "N")
DECISIVE = 2.0            # |cue score| needed for a cue-only decision
UMLAUT_SCORE = 3.0        # cue score of an umlaut or sharp s (decisive for every word)
CONTEXT_WEIGHTS = (1.0, 0.5)   # nearest and second-nearest word on each side (undecided words count 0)
MARGIN = 0.75             # |vote| needed to resolve an undecided word
CONFLICT_MARGIN = 1.5     # |left + right| needed when the two sides vote for different languages
CUE_WEIGHT = 0.5          # weight of the word's own cue score in the context resolution
CAP_CUE_WEIGHT = 0.25     # ... for capitalised mid-sentence words (German nouns carry English-looking suffixes)
CAP_PRIOR = 0.5           # German prior of a capitalised mid-sentence word (English capitalises only names)
RESOLVED_WEIGHT = 0.5     # weight of a context-resolved (not decided) neighbour in the second pass
_SENTENCE_PUNCT = ".!?"
_VOWELS = "aeiouäöü"
_CLOSING = "\"'’”)]»"
_OPENING = "\"'‘“(„«»‚[_"

# ---- character cues ------------------------------------------------------------------
# (pattern, weight); positive = German, negative = English.  Applied to the lowercase
# word core (letters, inner apostrophes and hyphens only).
_GERMAN_SUBSTR = (("ä", 3.0), ("ö", 3.0), ("ü", 3.0), ("ß", 3.0),
                  ("sch", 1.0), ("tsch", 0.5), ("tz", 1.0), ("pf", 1.0), ("ei", 0.5), ("eu", 0.5),
                  ("z", 0.25))
# (prefix, weight, minimum word length)
_GERMAN_PREFIX = (("ver", 1.0, 6), ("zer", 1.0, 6), ("zu", 0.5, 5), ("ge", 0.25, 5), ("vor", 0.5, 6),
                  ("nach", 1.0, 7), ("durch", 1.0, 8), ("unter", 0.5, 8), ("auf", 0.5, 6), ("aus", 0.5, 6))
# (suffix, weight, minimum word length)
_GERMAN_SUFFIX = (("ungen", 2.0, 7), ("ung", 1.5, 6), ("keit", 2.0, 6), ("heit", 2.0, 6),
                  ("lich", 2.0, 6), ("liche", 2.0, 7), ("lichen", 2.0, 8), ("licher", 2.0, 8),
                  ("liches", 2.0, 8), ("isch", 2.0, 6), ("ische", 2.0, 7), ("ischen", 2.0, 8),
                  ("ischer", 2.0, 8), ("isches", 2.0, 8), ("chen", 1.0, 6), ("lein", 1.0, 6),
                  ("schaft", 2.0, 8), ("ieren", 2.0, 7), ("iert", 2.0, 6), ("ierte", 2.0, 7),
                  ("ierten", 2.0, 8), ("ionen", 1.5, 7), ("igen", 0.75, 6), ("igt", 1.0, 5),
                  ("ig", 0.5, 4), ("ik", 0.5, 4), ("nis", 0.5, 5), ("tum", 0.5, 5), ("los", 0.5, 5),
                  ("bar", 0.25, 5), ("en", 0.5, 4), ("ern", 0.25, 5), ("eln", 0.5, 5), ("sten", 0.5, 6),
                  ("eht", 1.0, 4), ("ibt", 1.0, 4), ("ingt", 1.0, 5), ("icht", 1.0, 5), ("ehrt", 1.0, 5),
                  ("ndet", 1.0, 5), ("mmt", 1.0, 4), ("llt", 1.0, 4), ("nnt", 1.0, 4), ("ckt", 1.0, 4),
                  ("tet", 0.75, 5), ("tzt", 1.0, 4), ("ngt", 1.0, 4), ("rft", 1.0, 4), ("elt", 0.5, 5),
                  ("ert", 0.25, 5))
_ENGLISH_SUBSTR = (("th", 1.5), ("wh", 2.0), ("sh", 1.0), ("gh", 1.0), ("ght", 1.0), ("oo", 1.0),
                   ("ea", 1.0), ("ee", 0.5), ("ay", 1.0), ("ow", 1.0), ("aw", 1.0), ("ew", 1.0),
                   ("ey", 0.5), ("oy", 1.0), ("oi", 0.5), ("oa", 1.0), ("ou", 0.5), ("ai", 0.5),
                   ("tch", 1.0), ("dge", 1.0), ("wr", 1.0), ("zz", 1.0), ("y", 0.5))
_ENGLISH_SUFFIX = (("ing", 1.0, 5), ("tion", 1.5, 6), ("sion", 1.0, 6), ("ly", 1.5, 4), ("ness", 1.5, 6),
                   ("ment", 1.0, 6), ("ous", 1.5, 5), ("ity", 1.5, 5), ("ful", 1.5, 5), ("less", 1.5, 6),
                   ("ed", 1.5, 4), ("ies", 1.0, 5), ("ally", 1.5, 6), ("ship", 1.5, 6), ("able", 1.0, 6),
                   ("ible", 1.0, 6), ("ive", 0.5, 5), ("ves", 0.5, 5), ("ism", 0.5, 5), ("s", 0.5, 4),
                   ("al", 0.5, 5), ("ic", 0.5, 5), ("ure", 0.5, 5), ("ate", 0.5, 5), ("ual", 1.0, 5),
                   ("ize", 1.0, 5), ("ward", 1.0, 6), ("wards", 1.0, 7), ("ish", 0.5, 5), ("ty", 0.5, 5),
                   ("ry", 0.5, 5))
_BARE_C_WEIGHT = 1.0
_WORD_CORE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*", re.UNICODE)
_CLITIC = re.compile(r"['’](s|t|re|ll|ve|d|m)$")


# ======================================================================================
# lexicon
# ======================================================================================
@lru_cache(maxsize=4)
def _load_lexicon_cached(path: str) -> tuple[frozenset, frozenset, frozenset]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return (frozenset(w.lower() for w in d["D"]), frozenset(w.lower() for w in d["E"]),
            frozenset(w.lower() for w in d.get("N", ())))


def load_lexicon(path: str | Path = LEXICON_PATH) -> dict[str, set[str]]:
    """Return ``{"D": set, "E": set, "N": set}`` of lowercase lexicon words (``"N"`` are
    the proper names that are always labelled 'N')."""
    D, E, N = _load_lexicon_cached(str(path))
    return {"D": set(D), "E": set(E), "N": set(N)}


def _lexicon_sets(lexicon: Optional[dict]) -> tuple[frozenset, frozenset, frozenset]:
    if lexicon is None:
        return _load_lexicon_cached(str(LEXICON_PATH))
    N = lexicon.get("N")
    if N is None:
        N = _load_lexicon_cached(str(LEXICON_PATH))[2]
    return frozenset(lexicon["D"]), frozenset(lexicon["E"]), frozenset(N)


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
    n = len(core)
    for pat, wgt in _GERMAN_SUBSTR:
        if pat in core:
            s += wgt
    for pat, wgt, min_len in _GERMAN_PREFIX:
        if n >= min_len and core.startswith(pat):
            s += wgt
    for pat, wgt, min_len in _GERMAN_SUFFIX:
        if n >= min_len and core.endswith(pat):
            s += wgt
    no_sch = core.replace("sch", "")
    for pat, wgt in _ENGLISH_SUBSTR:
        if pat == "sh":
            if "sh" in no_sch:
                s -= wgt
        elif pat in core:
            s -= wgt
    for pat, wgt, min_len in _ENGLISH_SUFFIX:
        if n >= min_len and core.endswith(pat):
            if pat == "s" and core.endswith(("ss", "us", "is", "as")):
                continue
            s -= wgt
    if _has_bare_c(core):
        s -= _BARE_C_WEIGHT
    return s


def _is_capitalised(word: str) -> bool:
    m = _WORD_CORE.search(word)
    if not m:
        return False
    raw = m.group()
    return raw[0].isupper() and not (len(raw) > 1 and raw.isupper())


@dataclass(frozen=True)
class WordEvidence:
    """Evidence about one word: a decided label (or None), its cue score (> 0 German),
    whether it is capitalised mid-sentence, and its lexicon membership."""
    label: Optional[str]
    score: float
    capitalised: bool          # capitalised, not sentence-initial
    in_D: bool = False
    in_E: bool = False
    cap_any: bool = False      # capitalised (sentence-initial or not)


def _evidence(word: str, sentence_initial: bool, lex: tuple[frozenset, frozenset, frozenset]) -> WordEvidence:
    core = word_core(word)
    if not core or any(ch.isdigit() for ch in word):
        return WordEvidence("N", 0.0, False)
    D, E, N = lex
    if _CLITIC.search(core):
        return WordEvidence("E", 0.0, False)                 # English clitic: Kafka's, don't, we'll
    if core in N or (core.endswith("s") and core[:-1] in N):
        return WordEvidence("N", 0.0, False)                 # proper name (also genitive: Pauls)
    cap_any = _is_capitalised(word)
    cap_mid = cap_any and not sentence_initial
    in_D, in_E = core in D, core in E
    if not (in_D or in_E) and ("-" in core or "'" in core or "’" in core):
        last = re.split(r"['’\-]", core)[-1]                 # Kafka-Deutung -> deutung
        if last in N:
            return WordEvidence("N", 0.0, False)
        in_D, in_E = last in D, last in E
    score = cue_score(core)
    if in_D and in_E:
        return WordEvidence(None, score, cap_mid, True, True, cap_any)
    if in_D:
        return WordEvidence("D", score, False, True, False, cap_any)
    if in_E:
        if cap_mid:                                          # a capitalised noun in German text?
            return WordEvidence(None, score, True, False, True, cap_any)
        return WordEvidence("E", score, False, False, True, cap_any)
    if score >= UMLAUT_SCORE or (score >= DECISIVE and not cap_mid):
        return WordEvidence("D", score, False, False, False, cap_any)
    if not cap_mid:
        if score <= -DECISIVE:
            return WordEvidence("E", score, False, False, False, cap_any)
        if cap_any and score == 0.0:
            return WordEvidence("N", 0.0, False, False, False, cap_any)   # cue-less sentence-initial name
    return WordEvidence(None, score, cap_mid, False, False, cap_any)


def word_evidence(word: str, sentence_initial: bool = True,
                  lexicon: Optional[dict[str, set[str]]] = None) -> tuple[Optional[str], float, bool]:
    """(decided label or None, cue score, capitalised-mid-sentence flag) for one word."""
    ev = _evidence(word, sentence_initial, _lexicon_sets(lexicon))
    return ev.label, ev.score, ev.capitalised


def _resolve(ev: WordEvidence, left: float, right: float, lean: float = 0.0) -> str:
    """Final label of an undecided word from the context votes of its left and right
    neighbours (> 0 German, < 0 English) and its own cue score (`lean`: an extra cue
    from the word's syntactic position, see `label_words`).

    A capitalised mid-sentence word is a German noun unless the context is English, in
    which case it is the English word if the lexicon or decisive cues say so and a name
    ('N') otherwise.  For other words a conflict between the two sides (a language
    boundary) needs a clear majority, and a one-sided context contradicted by the word's
    own cues leaves the word open ('N')."""
    ambiguous = ev.in_D and ev.in_E
    score = lean if ambiguous else ev.score + lean        # a word of both lexicons: its spelling is no cue
    if ev.capitalised and not ambiguous:                  # unknown / English-only capitalised noun
        g = left + right + CAP_CUE_WEIGHT * ev.score + CAP_PRIOR
        if g >= MARGIN:
            return "D"
        if g <= -MARGIN:
            return "E" if (ev.in_E or ev.score <= -DECISIVE) else "N"
        return "N"
    ctx = left + right
    if left != 0.0 and right != 0.0 and (left > 0) != (right > 0):
        if abs(ctx) < CONFLICT_MARGIN:
            return "N"
    elif (left == 0.0) != (right == 0.0) and score != 0.0 and (score > 0) != (ctx > 0):
        return "N"
    g = ctx + (CAP_CUE_WEIGHT if ev.capitalised else CUE_WEIGHT) * score
    if g >= MARGIN:
        return "D"
    if g <= -MARGIN:
        if ev.cap_any and not (ev.in_E or ev.score <= -1.0):
            return "N"                      # capitalised unknown word in English context: a name
        return "E"
    return "N"


def classify_word(word: str, left: Sequence[str] = (), right: Sequence[str] = (),
                  sentence_initial: bool = True,
                  lexicon: Optional[dict[str, set[str]]] = None) -> str:
    """Label a single word ('E', 'D' or 'N').  `left`/`right` are the labels of the
    neighbouring words (nearest first, up to two per side) used for undecided words."""
    ev = _evidence(word, sentence_initial, _lexicon_sets(lexicon))
    if ev.label is not None:
        return ev.label
    votes = []
    for side in (left, right):
        v = 0.0
        for lab, w in zip(side, CONTEXT_WEIGHTS):
            v += w if lab == "D" else (-w if lab == "E" else 0.0)
        votes.append(v)
    return _resolve(ev, votes[0], votes[1])


# ======================================================================================
# sequences of words
# ======================================================================================
def _ends_sentence(word: str) -> bool:
    w = word.rstrip(_CLOSING)
    return bool(w) and w[-1] in _SENTENCE_PUNCT


def label_words(words: Sequence[str], lexicon: Optional[dict[str, set[str]]] = None) -> list[str]:
    """Label a sequence of words (strings in text order).

    Pass 0 decides words by names / lexicon / decisive cues and two positional rules:
    ``am`` is English only after ``I``; a capitalised mid-sentence word of the English
    lexicon is English unless the preceding word is German (``im letzten Moment``).
    Pass 1 resolves the other words from the labels of their two nearest neighbours per
    side within the same sentence (weights 1.0 / 0.5; undecided words count 0, names and
    digits are transparent) plus half their own cue score (`_resolve`); for ``an`` the
    following word's initial adds a cue (English ``an`` precedes a vowel).  Pass 2 repeats
    this for words whose decided context was weak (|vote| < `MARGIN`), now counting the
    pass-1 labels of their neighbours with half weight."""
    lex = _lexicon_sets(lexicon)
    n = len(words)
    cores = [word_core(w) for w in words]
    real = [i for i in range(n) if cores[i]]                 # words with letters
    m = len(real)
    # sentence end flags per real word (trailing punctuation-only tokens count)
    ends = [False] * n
    for p, i in enumerate(real):
        nxt = real[p + 1] if p + 1 < m else n
        ends[i] = any(_ends_sentence(words[j]) for j in range(i, nxt))
    sent = [0] * m                                           # sentence id per real position
    for p in range(1, m):
        sent[p] = sent[p - 1] + (1 if ends[real[p - 1]] else 0)
    evs: list[WordEvidence] = []
    for p, i in enumerate(real):
        sent_init = p == 0 or ends[real[p - 1]]
        evs.append(_evidence(words[i], sent_init, lex))
    labels: list[Optional[str]] = [ev.label for ev in evs]
    leans = [0.0] * m
    for p, ev in enumerate(evs):
        if ev.label is not None:
            continue
        core = cores[real[p]]
        prev_core = cores[real[p - 1]] if p > 0 and sent[p - 1] == sent[p] else ""
        prev_label = labels[p - 1] if p > 0 and sent[p - 1] == sent[p] else None
        if core == "am" and ev.in_D and ev.in_E:
            labels[p] = "E" if prev_core == "i" else "D"
        elif core == "an" and ev.in_D and ev.in_E and p + 1 < m and sent[p + 1] == sent[p]:
            leans[p] = -1.0 if cores[real[p + 1]][:1] in _VOWELS else 1.0
        elif ev.capitalised and ev.in_E and not ev.in_D and prev_label != "D":
            labels[p] = "E"
    decided = [lab is not None for lab in labels]

    def votes(p: int, labs: Sequence[Optional[str]], weights: Sequence[float]) -> tuple[float, float]:
        out = []
        for direction in (-1, 1):
            v = 0.0
            step = 0
            q = p + direction
            while 0 <= q < m and sent[q] == sent[p] and step < len(CONTEXT_WEIGHTS):
                lab = labs[q]
                if lab == "N" and decided[q]:
                    q += direction                           # names / digits are transparent
                    continue
                w = CONTEXT_WEIGHTS[step] * weights[q]
                if lab == "D":
                    v += w
                elif lab == "E":
                    v -= w
                step += 1
                q += direction
            out.append(v)
        return out[0], out[1]

    undecided = [p for p in range(m) if not decided[p]]
    full = [1.0] * m
    pass1 = list(labels)
    weak: list[int] = []
    for p in undecided:
        left, right = votes(p, labels, full)
        if abs(left + right) < MARGIN:
            weak.append(p)
        pass1[p] = _resolve(evs[p], left, right, leans[p])
    final = list(pass1)
    if weak:
        half = [1.0 if decided[q] else RESOLVED_WEIGHT for q in range(m)]
        for p in weak:
            left, right = votes(p, pass1, half)
            final[p] = _resolve(evs[p], left, right, leans[p])
    out = ["N"] * n
    for p, i in enumerate(real):
        out[i] = final[p] if final[p] is not None else "N"
    return out


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
    ('N' items are transparent).

    Because 'N' items are transparent, a language run that consists of 'N' items only
    is invisible: a one-word segment that is a proper name (``Lena``, ``Kraft``) or a
    number does not interrupt the surrounding language, so the switch into it and the
    switch out of it are both missed (one switch when the run opens or closes the
    text).  The count is therefore a *lower bound* on the designed K of a stimulus,
    see :func:`switch_count`."""
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
    """Number of E<->D switches in a label sequence (``len(switch_positions)``).

    This is a **lower bound** on the number of switches in the text: 'N' labels are
    transparent, so a language run made of 'N' words only - a one-word segment that is a
    proper name of the stimulus schema or a number - is not seen and its two switches
    (one at the start or end of the text) are not counted.  On the stimulus bank the
    label-based count equals K for every C1 and C2 variant; it is K - 1 for K = 8 at C3
    (``subj_1`` is a lone name in English before the first German run) and K - 1 ... K - 9
    for the segment-alternating K = 16 / 28 conditions, where every name subject is a
    one-segment run (measured in ``tests/test_lang_id.py::test_switch_count_tracks_K``).
    For an exact count of the designed switches use the mask (``stimuli.count_switches``)
    or the ``boundary`` tokens of ``stimuli.annotate``."""
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


# ======================================================================================
# diagnostics on the stimulus bank
# ======================================================================================
def _family_masks(family) -> list[tuple[int, str, list[str]]]:
    """The (K, C, mask) conditions of `stimuli.build_variants` without a tokenizer."""
    from .config import C_LEVELS, K_LEVELS, K_LEVELS_EXTREME
    from .stimuli import segment_masks
    out = [(0, "C1", segment_masks(family.n_units, 0, "C1"))]
    for K in K_LEVELS:
        if K == 0:
            continue
        for C in C_LEVELS:
            out.append((K, C, segment_masks(family.n_units, K, C)))
    for K in K_LEVELS_EXTREME:
        out.append((K, "seg", segment_masks(family.n_units, K, "seg")))
    return out


def evaluate_on_families(families_dir=None, tokenizer=None, lexicon: Optional[dict[str, set[str]]] = None,
                         verbose: bool = False):
    """Accuracy of the classifier against the segment languages of every family file in
    `families_dir` (default: the stimulus bank; files failing the schema are skipped).

    Every (K, C) variant of every family is labelled - at the token level with
    `label_tokens` when a `tokenizer` is given (words = `token_words` groups, truth from
    `stimuli.annotate`), otherwise at the word level with `label_text` (truth from the
    segment owning the word's first character).  Returns a pandas DataFrame with one row
    per (family, formulation, lang): ``n_words`` (words of that language), ``accuracy``
    (correct / labelled, 'N' labels excluded), ``misclassified_words`` (list of
    ``"word->label"`` strings, unique, most frequent first), ``n_errors``,
    ``n_unlabelled`` (words labelled 'N').  With `verbose=True` the misclassified words
    are printed per group."""
    from collections import Counter
    import pandas as pd
    from .stimuli import FAMILIES_DIR, load_families, render, _segment_spans, _char_owner
    if families_dir is None:
        families_dir = FAMILIES_DIR
    fams = load_families(families_dir, strict=False)
    rows = []
    for fam in fams:
        stats = {lang: dict(n=0, ok=0, unl=0, err=Counter()) for lang in ("E", "D")}
        if tokenizer is not None:
            from .stimuli import build_variants
            for v in build_variants(fam, tokenizer):
                labs = label_tokens(tokenizer, v.ids, lexicon)
                for s, e, text in token_words(tokenizer, v.ids):
                    if not word_core(text):
                        continue
                    truth = v.annot["lang"][s]
                    _tally(stats[truth], text, labs[s], truth)
        else:
            for K, C, mask in _family_masks(fam):
                text, segments = render(fam, mask)
                owner = _char_owner(len(text), _segment_spans(text, segments))
                labs = label_text(text, lexicon)
                for m, lab in zip(re.finditer(r"\S+", text), labs):
                    if not word_core(m.group()):
                        continue
                    truth = segments[owner[m.start()]].lang
                    _tally(stats[truth], m.group(), lab, truth)
        for lang in ("E", "D"):
            st = stats[lang]
            labelled = st["n"] - st["unl"]
            errs = [f"{w}->{lab}" for (w, lab), _ in st["err"].most_common()]
            rows.append(dict(family=fam.family_id, formulation=fam.formulation, lang=lang,
                             n_words=st["n"], accuracy=(st["ok"] / labelled if labelled else 1.0),
                             misclassified_words=errs, n_errors=sum(st["err"].values()),
                             n_unlabelled=st["unl"]))
            if verbose and errs:
                print(f"{fam.tag} {lang}: acc={rows[-1]['accuracy']:.3f} errors={errs}")
    return pd.DataFrame(rows, columns=["family", "formulation", "lang", "n_words", "accuracy",
                                       "misclassified_words", "n_errors", "n_unlabelled"])


def _tally(st: dict, word: str, lab: str, truth: str) -> None:
    st["n"] += 1
    if lab == "N":
        st["unl"] += 1
    elif lab == truth:
        st["ok"] += 1
    else:
        st["err"][(word, lab)] += 1
