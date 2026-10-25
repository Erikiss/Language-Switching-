"""Tests for lsne.lang_id: lexicon, word/token labelling, switch statistics, accuracy on
the stimulus bank (every family file present; files failing the schema are skipped)."""
import json
import warnings
from pathlib import Path

import pytest

from lsne import lang_id as L
from lsne import stimuli as S

REPO = Path(__file__).resolve().parents[1]
FAM_DIR = REPO / "data" / "stimuli" / "families"
CACHE_DIR = REPO / "cache" / "state2143"


@pytest.fixture(scope="module")
def families():
    """All family files that pass the schema (others are skipped with a warning)."""
    errors = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fams = S.load_families(FAM_DIR, strict=False, errors=errors)
    if errors:
        warnings.warn(f"skipped family files failing the schema: {sorted(errors)}")
    assert fams, "no valid family file found"
    return fams


@pytest.fixture(scope="module")
def variants(families, tok):
    out = {}
    for f in families:
        try:
            out[f.tag] = S.build_variants(f, tok)
        except ValueError as e:                              # e.g. target not a single token
            warnings.warn(f"skipped {f.tag}: {e}")
    assert out
    return out


# ======================================================================================
# lexicon
# ======================================================================================
def test_lexicon_file():
    d = json.loads(L.LEXICON_PATH.read_text(encoding="utf-8"))
    for lang, min_size in (("D", 600), ("E", 800)):
        words = d[lang]
        assert len(words) >= min_size
        assert all(w == w.strip().lower() and w for w in words)
        assert len(set(words)) == len(words)
    lex = L.load_lexicon()
    assert "und" in lex["D"] and "the" in lex["E"] and "nicht" in lex["D"] and "which" in lex["E"]
    assert "in" in lex["D"] and "in" in lex["E"]        # ambiguous words live in both lists
    assert set(d["ambiguous"]) == lex["D"] & lex["E"]
    assert "her" not in lex["D"] and "the" not in lex["D"] and "die" not in lex["E"]
    # inflected function words and common narrative verbs / nouns
    for w in ("einem", "ihrem", "diesen", "wartet", "warten", "wartete", "bringt", "liest",
              "schreibt", "bericht", "kinder", "kindern", "hunden", "ersten", "letzten"):
        assert w in lex["D"], w
    for w in ("kitchen", "frozen", "open", "between", "neighbour", "neighbor", "museum", "queue",
              "reads", "carried", "buys", "children", "loaves", "shelves"):
        assert w in lex["E"], w
    # proper names of the schema are always 'N'
    assert {"lena", "tomas", "mara", "jonas", "ida", "paul", "clara", "felix", "nora", "anton",
            "mia", "oskar", "elias", "hanna", "rosa", "kraft"} <= lex["N"]


# ======================================================================================
# single words / cues
# ======================================================================================
def test_classify_single_words():
    assert L.classify_word("the") == "E" and L.classify_word("und") == "D"
    assert L.classify_word("Geschichte") == "D"          # lexicon ('sch' + 'ch' + 'ich')
    assert L.classify_word("enthält") == "D"             # umlaut
    assert L.classify_word("zurück") == "D"
    assert L.classify_word("Darstellungen") == "D"       # -ungen
    assert L.classify_word("beschreibt") == "D"          # sch + -ibt
    assert L.classify_word("characterization") == "E"    # -tion, 'th'
    assert L.classify_word("months") == "E"              # 'th'
    assert L.classify_word("quickly") == "E"
    assert L.classify_word("Kafka's") == "E"             # possessive clitic + context-free
    assert L.classify_word("1234") == "N" and L.classify_word("...") == "N" and L.classify_word("3)Kraft") == "N"
    assert L.classify_word("Kraft") == "N"               # name
    assert L.classify_word("Kraft", sentence_initial=False) == "N"
    assert L.classify_word("Lena") == "N" and L.classify_word("Pauls", sentence_initial=False) == "N"
    assert L.classify_word("in") == "N"                  # ambiguous without context
    assert L.classify_word("in", left=["E"], right=["E"]) == "E"
    assert L.classify_word("in", left=["D", "D"]) == "D"
    assert L.classify_word("in", left=["E"], right=["D"]) == "N"     # balanced context, no cue


def test_cue_score_sign():
    assert L.cue_score("straße") > 0 and L.cue_score("möglichkeit") > 0
    assert L.cue_score("thought") < 0 and L.cue_score("discussion") < 0
    assert L.cue_score("") == 0.0
    # cues alone are not decisive for ordinary words: the context decides
    assert abs(L.cue_score("kitchen")) < L.DECISIVE and abs(L.cue_score("frozen")) < L.DECISIVE
    assert abs(L.cue_score("station")) < L.DECISIVE and abs(L.cue_score("moment")) < L.DECISIVE


def test_context_resolution():
    assert L.label_text("in the house") == ["E", "E", "E"]
    assert L.label_text("in dem Haus") == ["D", "D", "D"]
    assert L.label_text("Der Autor beschreibt ein Tier.") == ["D"] * 5          # capitalised nouns in German context
    labs = L.label_text("The report was written by Kraft in the morning.")
    assert labs[5] == "N" and all(x == "E" for i, x in enumerate(labs) if i != 5)   # name in English context
    assert L.label_text("Kraft") == ["N"]
    assert L.label_text("") == []
    # capitalised German nouns that are English lexicon words, and vice versa
    assert L.label_text("Felix packt ihren Arm im letzten Moment.") == ["N", "D", "D", "D", "D", "D", "D"]
    assert L.label_text("Die Ingenieurin erklärt das Problem mit ruhiger Stimme.")[1:] == ["D"] * 7
    assert L.label_text("The exhibition begins on Saturday morning.") == ["E"] * 6
    assert L.label_text("The sun warms the glass roof in April.") == ["E"] * 8
    assert L.label_text("Die Station bleibt ruhig bis zum Morgen.") == ["D"] * 7
    # listed names are always 'N'; unknown capitalised words are a name in English context
    # and a noun in German context
    assert L.label_text("The programme contains a sonata by Schubert.")[-1] == "N"
    assert L.label_text("Das Programm enthält eine Sonate von Schubert.")[-1] == "N"
    assert L.label_text("The programme contains a sonata by Brahms.")[-1] == "N"
    assert L.label_text("Das Programm enthält eine Sonate von Brahms.")[-1] == "D"
    assert L.label_text("Der Bauer fährt den Traktor in die Scheune.") == ["D"] * 8
    # positional rules: 'am' after 'I' is English, English 'an' precedes a vowel
    assert L.label_text("I am tired after the long walk.") == ["E"] * 7
    assert L.label_text("Seine Frau wartet am Kai.")[-2] == "D"
    assert L.label_text("An old man presses the call button.")[0] == "E"
    assert L.label_text("Das Boot liegt an der Mole.")[-3] == "D"


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
                assert L.language_fraction(labs, "D") == 0.0 and L.switch_count(labs) == 0, v.tag
            else:
                assert 0.2 < L.language_fraction(labs, "D") < 0.8, v.tag
    assert n > 1000
    assert ok / n >= 0.95, errors[:20]


def test_switch_count_tracks_K(variants, tok):
    # label-based switch counts should follow the designed K for the sentence-level masks
    for tag, vs in variants.items():
        for v in vs:
            if v.C != "C1":
                continue
            labs = L.label_tokens(tok, v.ids)
            assert abs(L.switch_count(labs) - v.K) <= max(1, v.K // 2), v.tag


def test_evaluate_on_families_word_level(families):
    df = L.evaluate_on_families(FAM_DIR, tokenizer=None)
    assert list(df.columns[:6]) == ["family", "formulation", "lang", "n_words", "accuracy", "misclassified_words"]
    assert len(df) == 2 * len(families)
    assert set(df["lang"]) == {"E", "D"}
    assert (df["n_words"] > 50).all()
    bad = df[df["accuracy"] < 0.97]
    assert bad.empty, bad[["family", "formulation", "lang", "accuracy", "misclassified_words"]].to_string()
    assert all(isinstance(x, list) for x in df["misclassified_words"])


def test_evaluate_on_families_token_level(families, tok):
    df = L.evaluate_on_families(FAM_DIR, tokenizer=tok)
    assert len(df) == 2 * len(families)
    bad = df[df["accuracy"] < 0.97]
    assert bad.empty, bad[["family", "formulation", "lang", "accuracy", "misclassified_words"]].to_string()
    assert df["accuracy"].mean() >= 0.99


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
# generated text (hand-written sentences)
# ======================================================================================
def test_generated_text_switches():
    # a sentence switching E -> D -> E mid-sentence yields exactly two switches
    text = "He opens the door und geht hinaus into the garden."
    labs = L.label_text(text)
    assert labs == ["E", "E", "E", "E", "D", "D", "D", "E", "E", "E"]
    assert L.switch_count(labs) == 2 and L.inter_switch_intervals(labs) == [3]
    text2 = "The old man walked slowly towards the station, aber der Zug war schon weg, so he sat down on a bench."
    labs2 = L.label_text(text2)
    assert L.switch_count(labs2) == 2 and 0.2 < L.language_fraction(labs2, "D") < 0.5
    # monolingual paragraphs on an unseen topic: no foreign labels, no switches
    en = ("Nora feeds the chickens behind the barn every morning. Her brother repairs the old tractor "
          "in the yard. A grey cat sleeps on the warm engine. The farmer sells eggs and honey at the "
          "village market on Saturday. Anton counts the coins at the kitchen table after dinner.")
    de = ("Nora füttert die Hühner hinter der Scheune jeden Morgen. Ihr Bruder repariert den alten "
          "Traktor im Hof. Eine graue Katze schläft auf dem warmen Motor. Der Bauer verkauft Eier und "
          "Honig auf dem Dorfmarkt am Samstag. Anton zählt die Münzen am Küchentisch nach dem Abendessen.")
    labs_en, labs_de = L.label_text(en), L.label_text(de)
    assert L.language_fraction(labs_en, "D") == 0.0 and L.switch_count(labs_en) == 0
    assert L.language_fraction(labs_de, "D") == 1.0 and L.switch_count(labs_de) == 0
    assert labs_en.count("N") <= 2 and labs_de.count("N") <= 2       # only the names
    # digits, punctuation and nonsense do not create switches
    labs3 = L.label_text("the the 12345 ### ,,, --- the")
    assert L.switch_count(labs3) == 0 and L.language_fraction(labs3, "E") == 1.0


def test_label_tokens_generated_text(tok):
    text = "He opens the door und geht hinaus into the garden."
    ids = tok(text, add_special_tokens=False)["input_ids"]
    labs = L.label_tokens(tok, ids)
    assert L.switch_count(labs) == 2
    per_word = [labs[s] for s, e, _ in L.token_words(tok, ids)]
    assert per_word == ["E", "E", "E", "E", "D", "D", "D", "E", "E", "E", "N"]


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
