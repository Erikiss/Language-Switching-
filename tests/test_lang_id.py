"""Tests for lsne.lang_id: lexicon, word/token labelling, switch statistics."""
import json
from pathlib import Path

import pytest

from lsne import lang_id as L
from lsne import stimuli as S

REPO = Path(__file__).resolve().parents[1]
FAM_DIR = REPO / "data" / "stimuli" / "families"
CACHE_DIR = REPO / "cache" / "state2143"


@pytest.fixture(scope="module")
def families():
    return S.load_families(FAM_DIR, strict=True)


@pytest.fixture(scope="module")
def variants(families, tok):
    return {f.tag: S.build_variants(f, tok) for f in families}


# ======================================================================================
# lexicon
# ======================================================================================
def test_lexicon_file():
    d = json.loads(L.LEXICON_PATH.read_text(encoding="utf-8"))
    for lang in ("D", "E"):
        words = d[lang]
        assert len(words) >= 250
        assert all(w == w.strip().lower() and w for w in words)
        assert len(set(words)) == len(words)
    lex = L.load_lexicon()
    assert "und" in lex["D"] and "the" in lex["E"] and "nicht" in lex["D"] and "which" in lex["E"]
    assert "in" in lex["D"] and "in" in lex["E"]        # ambiguous words live in both lists


# ======================================================================================
# single words / cues
# ======================================================================================
def test_classify_single_words():
    assert L.classify_word("the") == "E" and L.classify_word("und") == "D"
    assert L.classify_word("Geschichte") == "D"          # 'sch' + 'ch' + 'ich'
    assert L.classify_word("enthält") == "D"             # umlaut
    assert L.classify_word("zurück") == "D"
    assert L.classify_word("Darstellungen") == "D"       # -ungen
    assert L.classify_word("beschreibt") == "D"          # sch + -ibt
    assert L.classify_word("characterization") == "E"    # -tion, 'th'
    assert L.classify_word("months") == "E"              # 'th'
    assert L.classify_word("quickly") == "E"
    assert L.classify_word("Kafka's") == "E"             # possessive clitic + context-free
    assert L.classify_word("1234") == "N" and L.classify_word("...") == "N" and L.classify_word("3)Kraft") == "N"
    assert L.classify_word("Kraft") == "N"               # cue-less sentence-initial name
    assert L.classify_word("Kraft", sentence_initial=False) == "N"
    assert L.classify_word("in") == "N"                  # ambiguous without context
    assert L.classify_word("in", left=["E"], right=["E"]) == "E"
    assert L.classify_word("in", left=["D", "D"]) == "D"
    assert L.classify_word("in", left=["E"], right=["D"]) == "N"     # balanced context, no cue


def test_cue_score_sign():
    assert L.cue_score("straße") > 0 and L.cue_score("möglichkeit") > 0
    assert L.cue_score("thought") < 0 and L.cue_score("discussion") < 0
    assert L.cue_score("") == 0.0


def test_context_resolution():
    assert L.label_text("in the house") == ["E", "E", "E"]
    assert L.label_text("in dem Haus") == ["D", "D", "D"]
    assert L.label_text("Der Autor beschreibt ein Tier.") == ["D"] * 5          # capitalised nouns in German context
    labs = L.label_text("The report was written by Kraft in the morning.")
    assert labs[5] == "N" and all(x == "E" for i, x in enumerate(labs) if i != 5)   # name in English context
    assert L.label_text("Kraft") == ["N"]
    assert L.label_text("") == []


# ======================================================================================
# accuracy on the stimulus bank
# ======================================================================================
def _segment_accuracy(families):
    n = ok = 0
    errors = []
    for fam in families:
        strings = [(u.seg(lang, name), lang) for u in fam.units for lang in ("E", "D") for name in S.SEGMENT_NAMES]
        strings.append((fam.tail_E, "E"))
        for text, lang in strings:
            for w, lab in zip(L.split_words(text), L.label_text(text)):
                if lab == "N":
                    continue
                n += 1
                if lab == lang:
                    ok += 1
                else:
                    errors.append((fam.tag, text, w, lab))
    return ok / n, n, errors


def test_segment_accuracy(families):
    acc, n, errors = _segment_accuracy(families)
    assert n > 250
    assert acc >= 0.95, errors


def test_label_tokens_matches_annotation(variants, tok):
    n = ok = 0
    errors = []
    for tag, vs in variants.items():
        for v in vs:
            labs = L.label_tokens(tok, v.ids)
            assert len(labs) == len(v.ids) and set(labs) <= {"E", "D", "N"}
            for t, (lab, truth, k) in enumerate(zip(labs, v.annot["lang"], v.annot["klass"])):
                if k in ("ws", "punct"):
                    assert lab == "N"
                    continue
                if lab == "N":
                    continue
                n += 1
                if lab == truth:
                    ok += 1
                else:
                    errors.append((v.tag, t, v.annot["tokens"][t], truth, lab))
            # tokens of one word share the label
            for s, e, _ in L.token_words(tok, v.ids):
                assert len(set(labs[s:e])) == 1
            if v.K == 0:
                assert L.language_fraction(labs, "D") == 0.0 and L.switch_count(labs) == 0
            else:
                assert 0.2 < L.language_fraction(labs, "D") < 0.8
    assert n > 1000
    assert ok / n >= 0.95, errors[:20]


def test_switch_count_tracks_K(variants, tok):
    # label-based switch counts should follow the designed K for the sentence-level masks
    for tag, vs in variants.items():
        for v in vs:
            if v.C != "C1":
                continue
            labs = L.label_tokens(tok, v.ids)
            assert abs(L.switch_count(labs) - v.K) <= max(1, v.K // 2)


def test_german_units_of_original_json():
    spec = json.loads((REPO / "data" / "stimuli" / "original_2143.json").read_text(encoding="utf-8"))
    for k, text in spec["units_D"].items():
        labs = L.label_text(text)
        assert L.language_fraction(labs, "D") >= 0.95, (k, list(zip(L.split_words(text), labs)))


@pytest.mark.network
def test_anchor_text_is_english(tok):
    from lsne.anchor import load_anchor
    try:
        a = load_anchor(CACHE_DIR, tokenizer=tok)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"anchor unavailable: {e}")
    labs = L.label_tokens(tok, a.context_ids)
    assert len(labs) == 208
    assert L.language_fraction(labs, "E") >= 0.95
    assert L.switch_count(labs) == 0
    wl = L.label_text(a.text)
    assert L.language_fraction(wl, "E") >= 0.95


# ======================================================================================
# token grouping and statistics
# ======================================================================================
def test_label_tokens_mixed_sentence(tok):
    text = "He reads the report. Er liest den Bericht."
    ids = tok(text, add_special_tokens=False)["input_ids"]
    labs = L.label_tokens(tok, ids)
    words = L.token_words(tok, ids)
    assert [w[2] for w in words] == ["He", "reads", "the", "report", ".", "Er", "liest", "den", "Bericht", "."]
    per_word = [labs[s] for s, e, _ in words]
    assert per_word == ["E", "E", "E", "E", "N", "D", "D", "D", "D", "N"]
    assert L.switch_positions(labs) == [words[5][0]]
    assert L.label_tokens(tok, []) == []


def test_token_words_umlaut_and_whitespace(tok):
    ids = tok("Die Geschichte enthält einen Traum.\n\nEr erscheint.", add_special_tokens=False)["input_ids"]
    words = [w[2] for w in L.token_words(tok, ids)]
    assert "enthält" in words and "erscheint" in words and "" in words       # the "\n" tokens form empty groups
    labs = L.label_tokens(tok, ids)
    assert L.language_fraction(labs, "D") == 1.0


def test_switch_statistics():
    labs = ["E", "E", "N", "D", "D", "E", "N", "E", "D"]
    assert L.switch_positions(labs) == [3, 5, 8]
    assert L.switch_count(labs) == 3
    assert L.inter_switch_intervals(labs) == [2, 3]
    assert L.language_fraction(labs, "D") == pytest.approx(3 / 7)
    assert L.language_fraction(labs, "E") == pytest.approx(4 / 7)
    assert L.language_fraction(["N", "N"]) == 0.0 and L.language_fraction([]) == 0.0
    assert L.switch_positions(["N", "E", "E"]) == [] and L.inter_switch_intervals(["E"]) == []
    assert L.language_runs(labs) == [("E", 2), ("D", 2), ("E", 2), ("D", 1)]
