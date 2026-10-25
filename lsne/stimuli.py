"""Stimulus bank: families of meaning units, language masks (K x C), rendering,
tokenisation, per-token annotation, consistency checks and the original State 2143
case study (`data/stimuli/original_2143.json`).

Conventions (docs/SPEC.md, docs/DESIGN.md section 2, data/stimuli/SCHEMA.md):

* A *family* has 10 meaning units; every unit has three segments ``subj | pred | adj``
  in English (E) and German (D).  A unit renders as ``subj + " " + pred + " " + adj + "."``,
  units are joined by single spaces and the fixed English tail follows after one more
  space: ``units_text + " " + tail_E``.  The target (a string starting with a space) is
  appended directly to the context and must tokenize to exactly one token.
* Masks are per *segment* (length ``3 * n_units``), derived from the dossier's unit-level
  masks (``config.MASKS_UNIT_LEVEL``) and the structure level C (C1/C2/C3), or from a
  segment-level alternation for the exploratory K = 16 / 28 conditions (``C="seg"``).
* The original State 2143 context is treated as a family of ``kind="original"``: six
  switchable units (one text segment each, already carrying their whitespace and
  punctuation) plus the fixed English tail; unit texts are concatenated *directly*.

Unit-level invariants versus segment-level realisation
------------------------------------------------------
The dossier invariants "unit 1 and unit 10 always E, unit 9 always D in mixed
conditions" hold for the **unit-level masks** (``config.MASKS_UNIT_LEVEL``).  The C
table of DESIGN section 2.1 is normative for how a switch at the boundary k -> k+1 is
*realised*: C2 moves ``adj_k`` and C3 moves ``pred_k adj_k`` into the language of unit
k+1, i.e. **inside** unit k.  Consequently, at segment level,

* unit 9 is entirely German only for C1; for C2 it is ``D D E`` and for C3 ``D E E``
  (the final D -> E switch is realised inside unit 9);
* unit 1 is entirely English except for K = 8 (switch at 1 -> 2): C2 gives ``E E D``,
  C3 gives ``E D D``;
* unit 10 is entirely English in every C1/C2/C3 condition and for K = 16; for K = 28
  (28 switches over 29 segment boundaries) unit 10 is ``D E E``.

:func:`unit_composition` reports this per unit and the tests assert the full table.
Measurement sites must therefore be defined by *language*, not by unit index:
:func:`last_switched_position` gives the last German token of the context (the token
right before the final D -> E switch), which is comparable across C levels; the last
token of unit 9 is German only for C1.
"""
from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

from .config import C_LEVELS, K_LEVELS, K_LEVELS_EXTREME, MASKS_UNIT_LEVEL

SEGMENT_NAMES: tuple[str, ...] = ("subj", "pred", "adj")
LANGS = ("E", "D")
TOKEN_CLASSES = ("ws", "punct", "boundary", "word_start", "word_cont")
MAX_TOKENS = 2000

_REPO_ROOT = Path(__file__).resolve().parent.parent
FAMILIES_DIR = _REPO_ROOT / "data" / "stimuli" / "families"
ORIGINAL_JSON = _REPO_ROOT / "data" / "stimuli" / "original_2143.json"

_FINAL_PUNCT = ".!?;:,"
_SENTENCE_END = (".", "!", "?", ":", ";", '"', "'", ")")


# ======================================================================================
# data classes
# ======================================================================================
class Segment(NamedTuple):
    """One rendered segment: its text, language, 1-based unit index and segment name.

    The tail is reported as a segment with ``name="tail"``, ``lang="E"`` and
    ``unit = n_units + 1``."""
    text: str
    lang: str
    unit: int
    name: str


@dataclass
class Unit:
    """One meaning unit with its English (E) and German (D) segments."""
    id: int
    E: dict[str, str]
    D: dict[str, str]

    def seg(self, lang: str, name: str) -> str:
        if lang not in LANGS:
            raise ValueError(f"language must be 'E' or 'D', got {lang!r}")
        return (self.E if lang == "E" else self.D)[name]


@dataclass
class Family:
    """A stimulus family (one formulation).

    ``kind="standard"``: ten units with segments ``subj/pred/adj``, rendered with added
    spaces and periods.  ``kind="original"``: the State 2143 case study, units carry one
    ``text`` segment each (with their own whitespace/punctuation) and are concatenated
    directly; ``target_id`` is fixed (22905) and ``masks`` holds the unit-level masks."""
    family_id: str
    formulation: str
    units: list[Unit]
    tail_E: str
    target: str
    title: str = ""
    topic: str = ""
    notes: str = ""
    target_note: str = ""
    author: str = ""
    kind: str = "standard"
    segment_names: tuple[str, ...] = SEGMENT_NAMES
    target_id: Optional[int] = None
    masks: Optional[dict[int, str]] = None
    source: Optional[str] = None

    @property
    def n_units(self) -> int:
        return len(self.units)

    @property
    def n_segments(self) -> int:
        return len(self.units) * len(self.segment_names)

    @property
    def tag(self) -> str:
        return f"{self.family_id}{self.formulation}"


@dataclass
class Variant:
    """One rendered + tokenised stimulus.

    ``mask`` is per segment ('E'/'D'); ``ids`` are the context token ids (no special
    tokens); ``target_id`` is the single token id of ``family.target`` appended to the
    context.  ``annot`` holds per-token lists (``lang``, ``unit``, ``seg``, ``klass``,
    ``dist_to_last_switch``, ``tokens``) and scalars (``last_switch_pos``, ``tail_start``,
    ``n_words``, ``n_tokens``, ``F100``, ``n_switches``, ``n_tokens_E/D``, ``n_words_E/D``,
    ``boundary_positions``)."""
    family_id: str
    formulation: str
    K: int
    C: str
    mask: list[str]
    text: str
    ids: list[int]
    target_id: int
    annot: dict
    target: str = ""
    segments: list[Segment] = field(default_factory=list)

    @property
    def tag(self) -> str:
        return f"{self.family_id}{self.formulation}:K{self.K}:{self.C}"

    @property
    def n_tokens(self) -> int:
        return len(self.ids)

    @property
    def tail_start(self) -> int:
        return int(self.annot["tail_start"])

    @property
    def tail_ids(self) -> list[int]:
        return list(self.ids[self.tail_start:])

    def to_json(self) -> dict:
        d = asdict(self)
        d["segments"] = [tuple(s) for s in self.segments]
        return d


# ======================================================================================
# loading / validation of family JSON files
# ======================================================================================
def _check_segment(problems: list[str], where: str, s) -> None:
    if not isinstance(s, str) or not s.strip():
        problems.append(f"{where}: empty or non-string segment")
        return
    if s != s.strip():
        problems.append(f"{where}: leading/trailing whitespace in {s!r}")
    if s.rstrip() and s.rstrip()[-1] in _FINAL_PUNCT:
        problems.append(f"{where}: final punctuation in {s!r} (the renderer adds the period)")


def validate_family_dict(d: dict, n_units: int = 10) -> list[str]:
    """Return a list of schema problems (empty = valid) for a family JSON dict.

    Checks SCHEMA.md hard rules 1, 2, 5 (structure), 6 (target starts with a space;
    the single-token property needs a tokenizer and is checked in `build_variants`)
    and 7 (target word absent from units and tail)."""
    p: list[str] = []
    if not isinstance(d, dict):
        return ["not a JSON object"]
    for key in ("family_id", "formulation", "units", "tail_E", "target"):
        if key not in d:
            p.append(f"missing key {key!r}")
    if p:
        return p
    units = d["units"]
    if not isinstance(units, list) or len(units) != n_units:
        p.append(f"expected exactly {n_units} units, got {len(units) if isinstance(units, list) else type(units)}")
        return p
    for i, u in enumerate(units, start=1):
        if not isinstance(u, dict):
            p.append(f"unit {i}: not an object")
            continue
        if u.get("id") != i:
            p.append(f"unit {i}: id must be {i}, got {u.get('id')!r}")
        for lang in LANGS:
            segs = u.get(lang)
            if not isinstance(segs, dict):
                p.append(f"unit {i}: missing {lang!r} segments")
                continue
            for name in SEGMENT_NAMES:
                if name not in segs:
                    p.append(f"unit {i}.{lang}: missing segment {name!r}")
                else:
                    _check_segment(p, f"unit {i}.{lang}.{name}", segs[name])
    tail = d["tail_E"]
    if not isinstance(tail, str) or not tail.strip():
        p.append("tail_E empty")
    elif tail != tail.strip():
        p.append("tail_E has leading/trailing whitespace")
    tgt = d["target"]
    if not isinstance(tgt, str) or not tgt.startswith(" ") or not tgt.strip():
        p.append(f"target must be a string starting with a space, got {tgt!r}")
    else:
        word = tgt.strip()
        ctx_words = []
        for u in units:
            if isinstance(u, dict):
                for lang in LANGS:
                    segs = u.get(lang) or {}
                    ctx_words.extend(str(v) for v in segs.values() if isinstance(v, str))
        ctx_words.append(str(tail))
        if _word_in_text(word, " ".join(ctx_words)):
            p.append(f"target leak: {word!r} appears as a word in the units or the tail")
    return p


def family_from_dict(d: dict, source: Optional[str] = None) -> Family:
    """Build a `Family` from a validated JSON dict (no validation here)."""
    units = [Unit(id=int(u["id"]), E=dict(u["E"]), D=dict(u["D"])) for u in d["units"]]
    return Family(
        family_id=str(d["family_id"]), formulation=str(d["formulation"]), units=units,
        tail_E=str(d["tail_E"]), target=str(d["target"]), title=str(d.get("title", "")),
        topic=str(d.get("topic", "")), notes=str(d.get("notes", "")),
        target_note=str(d.get("target_note", "")), author=str(d.get("author", "")),
        source=source)


def load_family(path: str | Path) -> Family:
    """Load and validate one family JSON file (raises ValueError with the problem list)."""
    path = Path(path)
    d = json.loads(path.read_text(encoding="utf-8"))
    problems = validate_family_dict(d)
    if problems:
        raise ValueError(f"{path.name}: " + "; ".join(problems))
    return family_from_dict(d, source=str(path))


def load_families(dir: str | Path = FAMILIES_DIR, strict: bool = False,
                  errors: Optional[dict[str, list[str]]] = None) -> list[Family]:
    """Load all ``*.json`` families in `dir` (sorted by file name).

    Files that fail the schema are skipped with a warning listing every problem (or raise
    with `strict=True`).  If `errors` is a dict it is filled with ``{file: [problems]}``."""
    dir = Path(dir)
    fams: list[Family] = []
    failed: dict[str, list[str]] = {}
    for path in sorted(dir.glob("*.json")):
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001 - report any parse problem
            failed[path.name] = [f"JSON parse error: {e}"]
            continue
        problems = validate_family_dict(d)
        if problems:
            failed[path.name] = problems
            continue
        fams.append(family_from_dict(d, source=str(path)))
    if errors is not None:
        errors.update(failed)
    if failed:
        listing = "\n".join(f"  {name}: " + "; ".join(p) for name, p in failed.items())
        msg = f"{len(failed)} family file(s) in {dir} failed the schema and were skipped:\n{listing}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return fams


# ======================================================================================
# masks
# ======================================================================================
def count_switches(mask: Sequence[str]) -> int:
    """Number of language changes between consecutive entries of a mask."""
    m = list(mask)
    return sum(1 for a, b in zip(m, m[1:]) if a != b)


def unit_level_masks(K: int) -> list[str]:
    """Dossier unit-level mask for K in {0, 2, 4, 8} as a list of 'E'/'D' per unit.

    These masks carry the dossier invariants (unit 1 and unit 10 E, unit 9 D for K > 0,
    last switch before unit 10).  At segment level the C2/C3 realisation moves segments of
    a unit into the neighbouring language, see :func:`segment_masks_from_unit_mask` and
    :func:`unit_composition`."""
    if K not in MASKS_UNIT_LEVEL:
        raise ValueError(f"no unit-level mask for K={K}; available: {sorted(MASKS_UNIT_LEVEL)}")
    m = list(MASKS_UNIT_LEVEL[K])
    assert count_switches(m) == K, (K, m)
    return m


def unit_composition(mask: Sequence[str], n_seg: int = len(SEGMENT_NAMES)) -> list[str]:
    """Per-unit language composition of a segment-level mask: one string of ``n_seg``
    characters per unit (``"DDE"`` = subj and pred German, adj English).

    This is the quantity the unit-level invariants have to be checked against: a unit is
    "entirely German" only if its string is ``"D" * n_seg``."""
    m = list(mask)
    n_seg = int(n_seg)
    if n_seg < 1 or len(m) % n_seg != 0:
        raise ValueError(f"mask length {len(m)} is not a multiple of {n_seg} segments per unit")
    return ["".join(m[i:i + n_seg]) for i in range(0, len(m), n_seg)]


def _alternation_mask(n_segments: int, K: int) -> list[str]:
    """Segment-level mask starting and ending in E with exactly K switches placed as
    evenly as possible over the segment boundaries.

    The first segment is always E.  The last three segments (the whole last unit) are
    kept E whenever that is feasible (K <= n_segments - 4); otherwise only the last
    segment is forced to E (for K = 28 with 30 segments the last unit is ``D E E``:
    with 29 boundaries and the first and last three segments fixed to E at most 26
    switches are possible)."""
    if n_segments < 2:
        raise ValueError("need at least two segments")
    if K % 2 != 0:
        raise ValueError(f"segment-level alternation must start and end in E -> K must be even, got {K}")
    n_tail_E = 3 if K <= n_segments - 4 else 1
    n_free = n_segments - n_tail_E          # segments 0 .. n_free-1 may vary (segment 0 is E)
    n_boundaries = n_free - 1               # boundaries inside the free region
    if K > n_boundaries:
        raise ValueError(f"K={K} switches do not fit into {n_boundaries} free boundaries")
    switch_at = set()
    for j in range(K):
        switch_at.add(int((j + 0.5) * n_boundaries / K))   # strictly increasing for K <= n_boundaries
    assert len(switch_at) == K
    mask = ["E"]
    for b in range(n_segments - 1):
        prev = mask[-1]
        if b < n_boundaries and b in switch_at:
            mask.append("D" if prev == "E" else "E")
        else:
            mask.append(prev)
    assert mask[-1] == "E" and count_switches(mask) == K, (K, mask)
    return mask


def segment_masks_from_unit_mask(unit_mask: Sequence[str], C: str = "C1") -> list[str]:
    """Per-segment mask (three entries per unit) from an arbitrary unit-level mask
    (string or sequence of 'E'/'D') and a structure level (DESIGN section 2.1):

    C1: every segment of unit k carries the unit-level language m_k.
    C2: for a switch at the boundary k -> k+1, ``adj_k`` takes m_{k+1}
        (switch before the adjunct; the sentence frame stays open).
    C3: for a switch at k -> k+1, ``pred_k`` and ``adj_k`` take m_{k+1}
        (switch between subject and verb, across subject-verb agreement).
    The number of switches is preserved.

    Note that for C2/C3 the switch is realised *inside* unit k, so the unit-level
    invariants (unit 1 / 10 all E, unit 9 all D) do not carry over to the segment
    level: with the dossier masks unit 9 ends in English (``D D E`` / ``D E E``) and for
    K = 8 unit 1 ends in German (``E E D`` / ``E D D``).  Use :func:`unit_composition`
    to inspect this and :func:`last_switched_position` for a language-defined site."""
    m = list(unit_mask)
    bad = [x for x in m if x not in LANGS]
    if bad:
        raise ValueError(f"unit mask entries must be 'E'/'D', got {bad[:3]}")
    if C not in C_LEVELS:
        raise ValueError(f"unknown C level {C!r}; expected one of {C_LEVELS}")
    n_seg = len(SEGMENT_NAMES)
    out: list[str] = []
    for k in range(len(m)):
        segs = [m[k]] * n_seg
        if k + 1 < len(m) and m[k] != m[k + 1]:
            if C == "C2":
                segs[2] = m[k + 1]
            elif C == "C3":
                segs[1] = m[k + 1]
                segs[2] = m[k + 1]
        out.extend(segs)
    assert count_switches(out) == count_switches(m)
    return out


def segment_masks(n_units: int = 10, K: int = 0, C: str = "C1") -> list[str]:
    """Per-segment mask (length ``3 * n_units``) for a (K, C) condition.

    C1/C2/C3 apply `segment_masks_from_unit_mask` to the dossier unit-level mask of K
    (`unit_level_masks`); "seg" is the segment-level alternation with K in {16, 28}
    (see `_alternation_mask`).  The number of switches always equals K.

    Segment-level composition of the fixed units (see the module notes): unit 10 is
    all English except for K = 28 (``D E E``); unit 9 is all German for C1 only
    (C2 ``D D E``, C3 ``D E E``, K = 16 ``D D E``, K = 28 ``E D E``); unit 1 is all
    English except for K = 8 at C2 (``E E D``) / C3 (``E D D``) and the "seg"
    conditions (K = 16 ``E D D``, K = 28 ``E D E``)."""
    if C == "seg":
        if K not in K_LEVELS_EXTREME:
            raise ValueError(f"C='seg' is defined for K in {K_LEVELS_EXTREME}, got {K}")
        return _alternation_mask(n_units * len(SEGMENT_NAMES), K)
    if C not in C_LEVELS:
        raise ValueError(f"unknown C level {C!r}; expected one of {C_LEVELS + ('seg',)}")
    if K == 0:
        m = ["E"] * n_units
    else:
        m = unit_level_masks(K)
        if len(m) != n_units:
            raise ValueError(f"unit-level masks are defined for {len(m)} units, got n_units={n_units}")
    out = segment_masks_from_unit_mask(m, C)
    assert count_switches(out) == K, (K, C, out)
    return out


def mask_K(mask: Sequence[str]) -> int:
    """K of a mask = number of switches."""
    return count_switches(mask)


# ======================================================================================
# rendering
# ======================================================================================
def _normalise_mask(family: Family, mask) -> list[str]:
    m = list(mask)
    if family.kind == "original":
        expected = family.n_units
    else:
        expected = family.n_segments
    if len(m) != expected:
        raise ValueError(f"mask length {len(m)} != {expected} for family {family.tag} ({family.kind})")
    bad = [x for x in m if x not in LANGS]
    if bad:
        raise ValueError(f"mask entries must be 'E'/'D', got {bad[:3]}")
    return m


def render(family: Family, mask) -> tuple[str, list[Segment]]:
    """Render a family under a per-segment mask.

    Returns ``(text, segments)`` where `segments` lists ``Segment(text, lang, unit, name)``
    in text order (tail included as the last segment, name "tail", lang "E").
    Standard families: ``subj + " " + pred + " " + adj + "."`` per unit, units joined by
    single spaces, then ``" " + tail_E``.  Original family: unit texts are concatenated
    directly and the tail follows directly (nothing is added)."""
    m = _normalise_mask(family, mask)
    if family.kind == "original":
        return _render_original(family, m)
    segments: list[Segment] = []
    unit_texts: list[str] = []
    i = 0
    for u in family.units:
        parts = []
        for name in family.segment_names:
            lang = m[i]
            s = u.seg(lang, name)
            segments.append(Segment(s, lang, u.id, name))
            parts.append(s)
            i += 1
        unit_texts.append(" ".join(parts) + ".")
    text = " ".join(unit_texts) + " " + family.tail_E
    segments.append(Segment(family.tail_E, "E", family.n_units + 1, "tail"))
    return text, segments


def render_text(family: Family, mask) -> str:
    """Rendered context text only (see `render`)."""
    return render(family, mask)[0]


def _render_original(fam: Family, mask: list[str]) -> tuple[str, list[Segment]]:
    segments: list[Segment] = []
    parts: list[str] = []
    for u, lang in zip(fam.units, mask):
        s = u.seg(lang, "text")
        segments.append(Segment(s, lang, u.id, "text"))
        parts.append(s)
    parts.append(fam.tail_E)
    segments.append(Segment(fam.tail_E, "E", fam.n_units + 1, "tail"))
    return "".join(parts), segments


def render_original(fam: Family, mask6: str) -> str:
    """Text of the original State 2143 family under a unit-level mask (e.g. "EEDDDE"):
    unit texts concatenated directly, tail appended directly, nothing added."""
    if fam.kind != "original":
        raise ValueError("render_original expects the family built by original_2143_family")
    return _render_original(fam, _normalise_mask(fam, mask6))[0]


# ======================================================================================
# annotation
# ======================================================================================
def _word_in_text(word: str, text: str) -> bool:
    """True if `word` occurs as a whole word (case-insensitive) in `text`."""
    return re.search(r"(?<!\w)" + re.escape(word) + r"(?!\w)", text, flags=re.IGNORECASE) is not None


def _segment_spans(text: str, segments: Sequence[Segment]) -> list[tuple[int, int]]:
    """Character spans of the segments, located sequentially in `text`.  Only joiner
    characters (whitespace / punctuation) may lie between consecutive segments."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for seg in segments:
        j = text.find(seg.text, cursor)
        if j < 0:
            raise ValueError(f"segment {seg.text!r} not found in text after position {cursor}")
        gap = text[cursor:j]
        if any(ch.isalnum() for ch in gap):
            raise ValueError(f"unexpected text {gap!r} before segment {seg.text!r}")
        spans.append((j, j + len(seg.text)))
        cursor = j + len(seg.text)
    return spans


def _char_owner(n_chars: int, spans: Sequence[tuple[int, int]]) -> list[int]:
    """Segment index owning each character; joiner characters belong to the preceding
    segment (characters before the first segment to segment 0)."""
    owner = [-1] * n_chars
    for idx, (s, e) in enumerate(spans):
        for c in range(s, min(e, n_chars)):
            owner[c] = idx
    cur = 0
    for c in range(n_chars):
        if owner[c] < 0:
            owner[c] = cur
        else:
            cur = owner[c]
    return owner


def _token_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids and character offsets of `text` (no special tokens)."""
    if getattr(tokenizer, "is_fast", False):
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
    # slow-tokenizer fallback: accumulate decoded token lengths (exact for pure-ASCII text)
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    offsets, pos = [], 0
    for i in ids:
        s = tokenizer.decode([i])
        offsets.append((pos, pos + len(s)))
        pos += len(s)
    if pos != len(text):
        raise ValueError("cannot align tokens to characters with this tokenizer")
    return ids, offsets


def token_class(tok_str: str, is_first: bool, is_boundary: bool, after_ws: bool = False) -> str:
    """Token class: 'ws' (whitespace only), 'punct' (no alphanumerics), 'boundary'
    (first word token of a segment whose language differs from the previous one),
    'word_start' (starts with whitespace, is the first token, or directly follows a
    whitespace-only token such as the lone " " in " erscheint" -> [" ", "ers", "che", "int"]),
    else 'word_cont'."""
    if tok_str != "" and tok_str.strip() == "":
        return "ws"
    if not any(ch.isalnum() for ch in tok_str):
        return "punct"
    if is_boundary:
        return "boundary"
    if is_first or after_ws or tok_str[:1].isspace():
        return "word_start"
    return "word_cont"


def annotate(tokenizer, text: str, segments_with_lang: Sequence) -> dict:
    """Tokenise `text` and annotate every token with the segment it belongs to.

    `segments_with_lang` is the list of ``(segment_text, lang, unit_idx, segment_name)``
    returned by `render` (the tail must be the last entry, name "tail").  A token belongs
    to the segment that contains its first non-space character (whitespace-only tokens
    to the segment owning their first character; joiner characters belong to the
    preceding segment).

    Returns a dict with per-token lists ``tokens`` (decoded), ``lang``, ``unit``, ``seg``
    (segment index, tail = last), ``klass``, ``dist_to_last_switch`` (tokens since the
    most recent boundary token at or before this one; -1 before the first switch) and
    scalars ``last_switch_pos`` (index of the last boundary token, -1 if none),
    ``tail_start``, ``n_tokens``, ``n_words`` (whitespace-delimited words), ``F100``
    (= 100 * n_switches / (n_words - 1)), ``n_switches``, ``n_tokens_E``, ``n_tokens_D``,
    ``n_words_E``, ``n_words_D`` and ``boundary_positions``; ``ids`` holds the token ids
    (removed again by `make_variant`, which stores them on the Variant)."""
    segments = [Segment(*s) for s in segments_with_lang]
    if not segments:
        raise ValueError("need at least one segment")
    spans = _segment_spans(text, segments)
    owner = _char_owner(len(text), spans)
    ids, offsets = _token_offsets(tokenizer, text)
    n = len(ids)
    tok_strs = [text[s:e] for s, e in offsets]
    tail_idx = len(segments) - 1 if segments[-1].name == "tail" else None

    seg_of_tok: list[int] = []
    for (s, e) in offsets:
        c = s
        while c < e and text[c].isspace():
            c += 1
        if c >= e:                       # whitespace-only token
            c = min(s, len(text) - 1)
        seg_of_tok.append(owner[c] if 0 <= c < len(text) else len(segments) - 1)

    # boundary tokens: first alphanumeric-bearing token of a segment whose language
    # differs from the previous segment's language
    boundary_seg = set()
    for i in range(1, len(segments)):
        if segments[i].lang != segments[i - 1].lang:
            boundary_seg.add(i)
    seen_word_in_seg: set[int] = set()
    klass: list[str] = []
    boundary_positions: list[int] = []
    for t in range(n):
        sidx = seg_of_tok[t]
        has_alnum = any(ch.isalnum() for ch in tok_strs[t])
        is_boundary = has_alnum and sidx in boundary_seg and sidx not in seen_word_in_seg
        if has_alnum:
            seen_word_in_seg.add(sidx)
        after_ws = t > 0 and klass[t - 1] == "ws"
        k = token_class(tok_strs[t], t == 0, is_boundary, after_ws)
        if k == "boundary":
            boundary_positions.append(t)
        klass.append(k)

    lang = [segments[s].lang for s in seg_of_tok]
    unit = [int(segments[s].unit) for s in seg_of_tok]
    dist: list[int] = []
    last = -1
    for t in range(n):
        if klass[t] == "boundary":
            last = t
        dist.append(t - last if last >= 0 else -1)
    tail_start = n
    if tail_idx is not None:
        for t in range(n):
            if seg_of_tok[t] == tail_idx:
                tail_start = t
                break

    # words: whitespace-delimited runs, language from the segment owning the first char
    words = [(m.start(), m.group()) for m in re.finditer(r"\S+", text)]
    n_words = len(words)
    n_words_E = sum(1 for (c, _) in words if segments[owner[c]].lang == "E")
    n_words_D = n_words - n_words_E
    n_switches = len(boundary_positions)
    F100 = 100.0 * n_switches / (n_words - 1) if n_words > 1 else 0.0
    return dict(
        tokens=tok_strs, lang=lang, unit=unit, seg=seg_of_tok, klass=klass,
        dist_to_last_switch=dist, boundary_positions=boundary_positions,
        last_switch_pos=(boundary_positions[-1] if boundary_positions else -1),
        tail_start=tail_start, n_tokens=n, n_words=n_words, F100=F100,
        n_switches=n_switches,
        n_tokens_E=sum(1 for x in lang if x == "E"), n_tokens_D=sum(1 for x in lang if x == "D"),
        n_words_E=n_words_E, n_words_D=n_words_D, ids=ids,
    )


# ======================================================================================
# variants
# ======================================================================================
def target_token_id(tokenizer, text: str, target: str) -> int:
    """Id of `target` appended to `text`; raises if it is not exactly one extra token
    or if appending it changes the tokenisation of the context."""
    ids = list(tokenizer(text, add_special_tokens=False)["input_ids"])
    full = list(tokenizer(text + target, add_special_tokens=False)["input_ids"])
    if len(full) != len(ids) + 1 or full[:-1] != ids:
        extra = full[len(ids):] if full[:len(ids)] == ids else full[-3:]
        raise ValueError(
            f"target {target!r} must add exactly one token to the context; got {len(full) - len(ids)} "
            f"extra token(s) {[tokenizer.decode([i]) for i in extra]} (context prefix unchanged: {full[:len(ids)] == ids})")
    return int(full[-1])


def make_variant(family: Family, mask, K: int, C: str, tokenizer) -> Variant:
    """Render, tokenise and annotate one (family, mask) condition."""
    m = _normalise_mask(family, mask)
    text, segments = render(family, m)
    annot = annotate(tokenizer, text, segments)
    ids = annot.pop("ids")
    tgt = target_token_id(tokenizer, text, family.target)
    if family.target_id is not None and tgt != family.target_id:
        raise ValueError(f"{family.tag}: target id {tgt} != fixed target id {family.target_id}")
    n_words = annot["n_words"]
    annot["F100"] = 100.0 * K / (n_words - 1) if n_words > 1 else 0.0
    return Variant(family_id=family.family_id, formulation=family.formulation, K=int(K), C=C,
                   mask=m, text=text, ids=ids, target_id=tgt, annot=annot, target=family.target,
                   segments=segments)


def build_variants(family: Family, tokenizer, K_levels=K_LEVELS, C_levels=C_LEVELS,
                   include_seg: bool = True) -> list[Variant]:
    """All variants of a standard family: K=0 once (C="C1"), then every K>0 in
    `K_levels` crossed with `C_levels`, then K in {16, 28} with C="seg"
    (``1 + 3*3 + 2 = 12`` for the defaults)."""
    if family.kind == "original":
        raise ValueError("use build_original_variants for the State 2143 case study")
    out: list[Variant] = []
    if 0 in K_levels:
        out.append(make_variant(family, segment_masks(family.n_units, 0, "C1"), 0, "C1", tokenizer))
    for K in K_levels:
        if K == 0:
            continue
        for C in C_levels:
            out.append(make_variant(family, segment_masks(family.n_units, K, C), K, C, tokenizer))
    if include_seg:
        for K in K_LEVELS_EXTREME:
            out.append(make_variant(family, segment_masks(family.n_units, K, "seg"), K, "seg", tokenizer))
    return out


def last_position_in_language(v: Variant, lang: str = "D", before_tail: bool = True) -> Optional[int]:
    """Index of the last context token whose segment language is ``lang`` (``None`` if
    there is none).  With ``before_tail`` (default) only positions before the tail
    (``t < tail_start``) are considered; the tail is English in every variant anyway."""
    if lang not in LANGS:
        raise ValueError(f"lang must be 'E' or 'D', got {lang!r}")
    langs = list(v.annot.get("lang", []))
    stop = int(v.tail_start) if before_tail else len(langs)
    for t in range(min(stop, len(langs)) - 1, -1, -1):
        if langs[t] == lang:
            return t
    return None


def last_switched_position(v: Variant) -> int:
    """Language-defined measurement site: the last German token of the context.

    For every mixed variant (K > 0) this is the token immediately before the final
    D -> E switch, whatever the structure level: the end of unit 9 for C1, the end of
    ``pred_9`` for C2, the end of ``subj_9`` for C3, and the last German segment of the
    K = 16 / 28 conditions (for the original State 2143 family: the last German unit).
    The site is therefore always a German token and comparable across C, unlike the
    last token of unit 9, which is English for C2/C3 and the "seg" conditions.

    For a monolingual variant (K = 0, no German token) the position-matched fallback is
    the last token of the last switchable unit (unit 9 of a standard family, i.e. the
    C1 site; the last unit before the tail of the original family); it is always
    ``< tail_start``."""
    pos = last_position_in_language(v, "D", before_tail=True)
    if pos is not None:
        return int(pos)
    tail_start = int(v.tail_start)
    units = [int(u) for u in v.annot.get("unit", [])][:tail_start]
    if not units:
        return max(0, tail_start - 1)
    # standard family: unit 10 is English under every dossier mask, the last switched unit
    # is unit 9 (= n_units - 1); original family: every unit before the tail is switchable
    # and its K=5 mask ends in German, so the last unit before the tail is the match
    target_unit = max(units) if _is_original_variant(v) else max(units) - 1
    cands = [t for t, u in enumerate(units) if u == target_unit]
    return int(cands[-1]) if cands else max(0, tail_start - 1)


def _is_original_variant(v: Variant) -> bool:
    """True for variants of the State 2143 family (``C == "orig"`` / ``text`` segments)."""
    return str(getattr(v, "C", "")) == "orig" or any(s.name == "text" for s in getattr(v, "segments", []) or [])


def reference_variant(variants: Sequence[Variant]) -> Variant:
    """The K=0 (monolingual English) variant of a list of variants."""
    for v in variants:
        if v.K == 0:
            return v
    raise ValueError("no K=0 variant present")


def check_variant(v: Variant, ref_variant: Variant) -> list[str]:
    """Consistency problems of a variant relative to the K=0 reference of the same
    family/formulation: target id differs, tail token ids differ (BPE boundaries),
    target word leaks into the context, context too long (> MAX_TOKENS)."""
    problems: list[str] = []
    if (v.family_id, v.formulation) != (ref_variant.family_id, ref_variant.formulation):
        problems.append(f"reference belongs to {ref_variant.family_id}{ref_variant.formulation}, "
                        f"variant to {v.family_id}{v.formulation}")
    if v.target_id != ref_variant.target_id:
        problems.append(f"target id {v.target_id} differs from reference {ref_variant.target_id}")
    if v.tail_ids != ref_variant.tail_ids:
        problems.append(f"tail token ids differ from reference ({len(v.tail_ids)} vs {len(ref_variant.tail_ids)} tokens)")
    word = v.target.strip()
    if word and _word_in_text(word, v.text):
        problems.append(f"target leak: {word!r} appears as a word in the context")
    if v.n_tokens > MAX_TOKENS:
        problems.append(f"n_tokens {v.n_tokens} > {MAX_TOKENS}")
    return problems


def check_all(variants: Sequence[Variant]) -> dict[str, list[str]]:
    """`check_variant` for every variant against the K=0 reference; keyed by variant tag."""
    ref = reference_variant(variants)
    return {v.tag: check_variant(v, ref) for v in variants}


def variant_summary(variants: Sequence[Variant]):
    """One row per variant (pandas DataFrame) with the scalar annotation fields."""
    import pandas as pd
    rows = []
    for v in variants:
        a = v.annot
        rows.append(dict(family_id=v.family_id, formulation=v.formulation, K=v.K, C=v.C,
                         mask="".join(v.mask), n_tokens=a["n_tokens"], n_words=a["n_words"],
                         F100=a["F100"], n_switches=a["n_switches"], tail_start=a["tail_start"],
                         last_switch_pos=a["last_switch_pos"], n_tokens_E=a["n_tokens_E"],
                         n_tokens_D=a["n_tokens_D"], n_words_E=a["n_words_E"],
                         n_words_D=a["n_words_D"], target_id=v.target_id))
    return pd.DataFrame(rows)


# ======================================================================================
# original State 2143 case study
# ======================================================================================
def load_original_spec(path: str | Path = ORIGINAL_JSON) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def original_2143_family(anchor, tokenizer, path: str | Path = ORIGINAL_JSON) -> Family:
    """Family of ``kind="original"``: six switchable units (English text decoded from
    `anchor.context_ids` with the token bounds of original_2143.json, German text from
    ``units_D``) plus the fixed English tail (unit 7).  Target "bey" (22905)."""
    spec = load_original_spec(path)
    bounds = list(spec["unit_token_bounds"])
    ctx = list(anchor.context_ids)
    if bounds[-1] != len(ctx):
        raise ValueError(f"token bounds end at {bounds[-1]} but the context has {len(ctx)} tokens")
    tail_unit = int(spec["tail_unit"])
    n_switch = tail_unit - 1
    units: list[Unit] = []
    for k in range(n_switch):
        e_text = tokenizer.decode(ctx[bounds[k]:bounds[k + 1]])
        d_text = spec["units_D"][str(k + 1)]
        units.append(Unit(id=k + 1, E={"text": e_text}, D={"text": d_text}))
    tail = tokenizer.decode(ctx[bounds[n_switch]:bounds[n_switch + 1]])
    masks = {int(k): v for k, v in spec.get("masks_6_units", {}).items()}
    return Family(family_id=str(spec.get("family_id", "ORIG2143")), formulation="orig", units=units,
                  tail_E=tail, target=str(spec.get("target_text", "bey")),
                  title=str(spec.get("title", "")), topic="State 2143 original context",
                  notes=str(spec.get("note", "")), kind="original", segment_names=("text",),
                  target_id=int(spec.get("target_id", 22905)), masks=masks, source=str(path))


def build_original_variants(anchor, tokenizer, path: str | Path = ORIGINAL_JSON) -> list[Variant]:
    """Variants of the original family for every mask in ``masks_6_units`` (C="orig",
    K = number of switches inside the six units, as in the JSON keys)."""
    fam = original_2143_family(anchor, tokenizer, path)
    out: list[Variant] = []
    for K, mask in sorted((fam.masks or {}).items()):
        m = list(mask)
        if count_switches(m) != K:
            raise ValueError(f"mask {mask!r} has {count_switches(m)} switches, key says {K}")
        out.append(make_variant(fam, m, K, "orig", tokenizer))
    return out
