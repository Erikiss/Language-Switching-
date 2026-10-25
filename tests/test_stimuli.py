"""Tests for lsne.stimuli: masks, rendering, variants, annotation, consistency checks and
the original State 2143 case study."""
import copy
import json
from pathlib import Path

import pytest

from lsne import stimuli as S
from lsne.config import C_LEVELS, K_LEVELS, K_LEVELS_EXTREME, MASKS_UNIT_LEVEL

REPO = Path(__file__).resolve().parents[1]
FAM_DIR = REPO / "data" / "stimuli" / "families"
CACHE_DIR = REPO / "cache" / "state2143"


# ======================================================================================
# fixtures
# ======================================================================================
@pytest.fixture(scope="module")
def families():
    fams = S.load_families(FAM_DIR, strict=True)
    tags = {f.tag for f in fams}
    assert {"DEV01A", "DEV01B"} <= tags
    return fams


@pytest.fixture(scope="module")
def dev01(families):
    return {f.tag: f for f in families if f.family_id == "DEV01"}


@pytest.fixture(scope="module")
def variants(families, tok):
    return {f.tag: S.build_variants(f, tok) for f in families}


@pytest.fixture(scope="module")
def anchor(tok):
    from lsne.anchor import load_anchor
    try:
        return load_anchor(CACHE_DIR, tokenizer=tok)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"State 2143 anchor unavailable (cache or network): {e}")


# ======================================================================================
# masks
# ======================================================================================
def test_unit_level_masks():
    for K, s in MASKS_UNIT_LEVEL.items():
        m = S.unit_level_masks(K)
        assert "".join(m) == s and len(m) == 10
        assert S.count_switches(m) == K
        assert m[0] == "E" and m[-1] == "E"
        if K > 0:
            assert m[8] == "D"          # unit 9 always German in mixed conditions
    with pytest.raises(ValueError):
        S.unit_level_masks(3)


@pytest.mark.parametrize("K", [0, 2, 4, 8])
@pytest.mark.parametrize("C", list(C_LEVELS))
def test_segment_masks_switch_counts_KC(K, C):
    m = S.segment_masks(10, K, C)
    assert len(m) == 30
    assert set(m) <= {"E", "D"}
    assert S.count_switches(m) == K
    assert m[0] == "E"
    assert m[-3:] == ["E", "E", "E"]            # unit 10 entirely English


@pytest.mark.parametrize("K", list(K_LEVELS_EXTREME))
def test_segment_masks_seg(K):
    m = S.segment_masks(10, K, "seg")
    assert len(m) == 30 and S.count_switches(m) == K
    assert m[0] == "E" and m[-1] == "E"
    if K <= 26:
        assert m[-3:] == ["E", "E", "E"]
    # switches as evenly spaced as possible: consecutive gaps differ by at most one
    sw = [i for i in range(29) if m[i] != m[i + 1]]
    gaps = [b - a for a, b in zip(sw, sw[1:])]
    assert max(gaps) - min(gaps) <= 1
    if K == 28:
        assert sw == list(range(28))            # every boundary but the last one


def test_segment_masks_c2_c3_structure():
    for K in (2, 4, 8):
        unit = S.unit_level_masks(K)
        c1, c2, c3 = (S.segment_masks(10, K, C) for C in C_LEVELS)
        for k in range(10):
            segs1, segs2, segs3 = (c[3 * k:3 * k + 3] for c in (c1, c2, c3))
            assert segs1 == [unit[k]] * 3
            if k < 9 and unit[k] != unit[k + 1]:
                assert segs2 == [unit[k], unit[k], unit[k + 1]]
                assert segs3 == [unit[k], unit[k + 1], unit[k + 1]]
            else:
                assert segs2 == segs1 == segs3
    assert S.segment_masks(10, 0, "C3") == ["E"] * 30


def test_segment_masks_from_unit_mask_arbitrary():
    assert S.segment_masks_from_unit_mask("DEDE", "C1") == list("DDDEEEDDDEEE")
    assert S.segment_masks_from_unit_mask("DEDE", "C2") == list("DDEEEDDDEEEE")
    assert S.segment_masks_from_unit_mask("DEDE", "C3") == list("DEEEDDDEEEEE")
    assert S.segment_masks_from_unit_mask(["E", "E"], "C3") == ["E"] * 6
    for K in (2, 4, 8):
        for C in C_LEVELS:
            assert S.segment_masks_from_unit_mask(S.unit_level_masks(K), C) == S.segment_masks(10, K, C)
    with pytest.raises(ValueError):
        S.segment_masks_from_unit_mask("EEX", "C1")
    with pytest.raises(ValueError):
        S.segment_masks_from_unit_mask("EED", "seg")


#: Segment-level composition of units 1, 9 and 10 for every (K, C) condition of the
#: bank (DESIGN section 2.1 realises a switch at k -> k+1 inside unit k).  This is the
#: normative table the unit-level invariants have to be read against.
UNIT_COMPOSITION_1_9_10 = {
    (0, "C1"): ("EEE", "EEE", "EEE"),
    (2, "C1"): ("EEE", "DDD", "EEE"), (2, "C2"): ("EEE", "DDE", "EEE"), (2, "C3"): ("EEE", "DEE", "EEE"),
    (4, "C1"): ("EEE", "DDD", "EEE"), (4, "C2"): ("EEE", "DDE", "EEE"), (4, "C3"): ("EEE", "DEE", "EEE"),
    (8, "C1"): ("EEE", "DDD", "EEE"), (8, "C2"): ("EED", "DDE", "EEE"), (8, "C3"): ("EDD", "DEE", "EEE"),
    (16, "seg"): ("EDD", "DDE", "EEE"), (28, "seg"): ("EDE", "EDE", "DEE"),
}


def test_unit_composition_every_condition():
    """The per-unit composition of every (K, C) condition is exactly the documented table:
    unit 9 is entirely German only for C1, unit 1 is entirely English except for K=8 at
    C2/C3 and the seg conditions, unit 10 is entirely English except for K=28."""
    conds = [(0, "C1")] + [(K, C) for K in (2, 4, 8) for C in C_LEVELS] + [(K, "seg") for K in K_LEVELS_EXTREME]
    assert set(conds) == set(UNIT_COMPOSITION_1_9_10)
    for K, C in conds:
        m = S.segment_masks(10, K, C)
        comp = S.unit_composition(m)
        assert len(comp) == 10 and all(len(c) == 3 for c in comp)
        assert (comp[0], comp[8], comp[9]) == UNIT_COMPOSITION_1_9_10[(K, C)], (K, C, comp)
        if C in C_LEVELS:
            unit = S.unit_level_masks(K) if K else ["E"] * 10
            for k in range(10):
                switch_after = k < 9 and unit[k] != unit[k + 1]
                if C == "C1" or not switch_after:
                    assert comp[k] == unit[k] * 3, (K, C, k, comp[k])
                else:          # the switch is realised inside unit k
                    assert comp[k] != unit[k] * 3 and comp[k][0] == unit[k] and comp[k][-1] == unit[k + 1]
    # the unit-level invariants hold for the unit masks ...
    for K in (2, 4, 8):
        u = S.unit_level_masks(K)
        assert u[0] == "E" and u[8] == "D" and u[9] == "E"
    # ... but "unit 9 entirely German" holds at segment level for C1 only
    assert {C for K in (2, 4, 8) for C in C_LEVELS if S.unit_composition(S.segment_masks(10, K, C))[8] == "DDD"} == {"C1"}
    with pytest.raises(ValueError):
        S.unit_composition(["E"] * 7)


def test_last_switched_position_is_language_defined(variants):
    """The depth-injection site is the last German token for every mixed variant (all C,
    both seg conditions) and the last token of unit 9 for the monolingual variant."""
    for tag, vs in variants.items():
        for v in vs:
            a = v.annot
            pos = S.last_switched_position(v)
            assert 0 <= pos < v.tail_start
            if v.K == 0:
                assert S.last_position_in_language(v, "D") is None
                assert a["lang"][pos] == "E" and a["unit"][pos] == 9
                assert a["unit"][pos + 1] == 10
            else:
                assert a["lang"][pos] == "D", (v.tag, pos)
                assert all(x == "E" for x in a["lang"][pos + 1:]), v.tag        # last German token
                assert pos == S.last_position_in_language(v, "D")
                # token right before the final D -> E boundary token
                assert a["klass"][pos + 1] == "boundary" and a["boundary_positions"][-1] == pos + 1
                expected_unit = {"C1": 9, "C2": 9, "C3": 9, "seg": 9 if v.K == 16 else 10}[v.C]
                assert a["unit"][pos] == expected_unit, (v.tag, a["unit"][pos])
            assert S.last_position_in_language(v, "E", before_tail=False) == v.n_tokens - 1
            assert S.last_position_in_language(v, "E") == v.tail_start - 1       # unit 10 ends in E everywhere
    # C1 vs C2/C3 of the same K differ in where the last German token sits within unit 9
    vs = variants["DEV01A"]
    by = {(v.K, v.C): v for v in vs}
    for K in (2, 4, 8):
        segs = {C: by[(K, C)].annot["seg"][S.last_switched_position(by[(K, C)])] % 3 for C in C_LEVELS}
        assert segs == {"C1": 2, "C2": 1, "C3": 0}          # adj_9 / pred_9 / subj_9
    with pytest.raises(ValueError):
        S.last_position_in_language(vs[0], "X")


def test_segment_masks_errors():
    with pytest.raises(ValueError):
        S.segment_masks(10, 15, "seg")           # odd
    with pytest.raises(ValueError):
        S.segment_masks(10, 2, "C9")
    with pytest.raises(ValueError):
        S.segment_masks(10, 3, "C1")
    with pytest.raises(ValueError):
        S.segment_masks(7, 2, "C1")              # unit-level masks are for 10 units
    with pytest.raises(ValueError):
        S._alternation_mask(30, 30)


# ======================================================================================
# loading / rendering
# ======================================================================================
def test_families_follow_schema(families):
    for f in families:
        assert f.n_units == 10 and f.n_segments == 30
        assert f.target.startswith(" ")
        for u in f.units:
            for lang in ("E", "D"):
                for name in S.SEGMENT_NAMES:
                    s = u.seg(lang, name)
                    assert s == s.strip() and s and s[-1] not in ".!?,;:"
            assert 4 <= len(" ".join(u.E.values()).split()) <= 11


def test_load_families_skips_bad_files(tmp_path):
    good = json.loads((FAM_DIR / "DEV01A.json").read_text(encoding="utf-8"))
    bad = copy.deepcopy(good)
    bad["units"] = bad["units"][:9]
    bad2 = copy.deepcopy(good)
    bad2["target"] = "bark"
    bad2["units"][0]["E"]["adj"] = "in the morning."
    (tmp_path / "GOOD.json").write_text(json.dumps(good), encoding="utf-8")
    (tmp_path / "BAD1.json").write_text(json.dumps(bad), encoding="utf-8")
    (tmp_path / "BAD2.json").write_text(json.dumps(bad2), encoding="utf-8")
    (tmp_path / "BAD3.json").write_text("{not json", encoding="utf-8")
    errors = {}
    with pytest.warns(UserWarning, match="failed the schema"):
        fams = S.load_families(tmp_path, errors=errors)
    assert [f.tag for f in fams] == ["DEV01A"]
    assert set(errors) == {"BAD1.json", "BAD2.json", "BAD3.json"}
    assert any("10 units" in p for p in errors["BAD1.json"])
    assert any("space" in p for p in errors["BAD2.json"])
    assert any("punctuation" in p for p in errors["BAD2.json"])
    with pytest.raises(ValueError):
        S.load_families(tmp_path, strict=True)


def test_validate_family_dict_leak():
    d = json.loads((FAM_DIR / "DEV01A.json").read_text(encoding="utf-8"))
    assert S.validate_family_dict(d) == []
    d["target"] = " dogs"                        # occurs in unit 6
    assert any("leak" in p for p in S.validate_family_dict(d))


def test_render_rules(dev01):
    fam = dev01["DEV01A"]
    text, segs = S.render(fam, ["E"] * 30)
    expected_units = [" ".join(u.E[n] for n in S.SEGMENT_NAMES) + "." for u in fam.units]
    assert text == " ".join(expected_units) + " " + fam.tail_E
    assert text == S.render_text(fam, ["E"] * 30)
    assert len(segs) == 31 and segs[-1] == S.Segment(fam.tail_E, "E", 11, "tail")
    assert [s.name for s in segs[:3]] == list(S.SEGMENT_NAMES)
    assert all(s.unit == i // 3 + 1 for i, s in enumerate(segs[:30]))
    mask = S.segment_masks(10, 2, "C2")
    text2, segs2 = S.render(fam, mask)
    assert "Ein Soldat erscheint im Traum." in text2
    assert "Die Diskussion kehrt zur Geschichte zurück after a pause." in text2   # C2: adj_9 English
    assert text2.endswith(" " + fam.tail_E)
    assert [s.lang for s in segs2[:30]] == mask
    with pytest.raises(ValueError):
        S.render(fam, ["E"] * 10)


# ======================================================================================
# variants and annotation
# ======================================================================================
def test_build_variants_count_and_order(variants, families, tok):
    fam_by_tag = {f.tag: f for f in families}
    for tag, vs in variants.items():
        assert len(vs) == 12
        conds = [(v.K, v.C) for v in vs]
        assert conds[0] == (0, "C1")
        assert conds[1:10] == [(K, C) for K in (2, 4, 8) for C in C_LEVELS]
        assert conds[10:] == [(16, "seg"), (28, "seg")]
        for v in vs:
            assert S.count_switches(v.mask) == v.K and len(v.mask) == 30
            assert v.text == S.render_text(fam_by_tag[tag], v.mask)
            assert v.ids == tok(v.text, add_special_tokens=False)["input_ids"]
            assert v.tag == f"{tag}:K{v.K}:{v.C}"
    # subsets of the grid
    fam = fam_by_tag["DEV01A"]
    sub = S.build_variants(fam, tok, K_levels=(0, 2), C_levels=("C1",), include_seg=False)
    assert [(v.K, v.C) for v in sub] == [(0, "C1"), (2, "C1")]


def test_check_variants_pass(variants):
    for tag, vs in variants.items():
        problems = S.check_all(vs)
        assert all(not p for p in problems.values()), problems
        ref = S.reference_variant(vs)
        assert ref.K == 0
        assert len({v.target_id for v in vs}) == 1


def test_check_variant_detects_problems(variants, tok):
    vs = variants["DEV01A"]
    ref = vs[0]
    v = copy.deepcopy(vs[1])
    v.target_id += 1
    assert any("target id" in p for p in S.check_variant(v, ref))
    v = copy.deepcopy(vs[1])
    v.ids[-1] = ref.ids[-1] + 1
    assert any("tail" in p for p in S.check_variant(v, ref))
    v = copy.deepcopy(vs[1])
    v.text = v.text + " Bark."
    assert any("leak" in p for p in S.check_variant(v, ref))
    v = copy.deepcopy(vs[1])
    v.ids = v.ids + [0] * 2000
    v.annot["tail_start"] = len(v.ids) - len(ref.tail_ids)
    assert any("2000" in p for p in S.check_variant(v, ref))
    assert S.check_variant(vs[1], ref) == []


def test_annotation_classes(variants, tok):
    for tag, vs in variants.items():
        for v in vs:
            a = v.annot
            n = len(v.ids)
            assert a["n_tokens"] == n == len(a["lang"]) == len(a["unit"]) == len(a["klass"]) \
                == len(a["dist_to_last_switch"]) == len(a["tokens"]) == len(a["seg"])
            assert set(a["klass"]) <= set(S.TOKEN_CLASSES)
            assert {"word_start", "word_cont", "punct"} <= set(a["klass"])
            assert a["klass"].count("boundary") == v.K == a["n_switches"]
            assert a["boundary_positions"] == [i for i, k in enumerate(a["klass"]) if k == "boundary"]
            assert a["last_switch_pos"] == (a["boundary_positions"][-1] if v.K else -1)
            assert a["klass"][0] == "word_start"
            # tail: starts where tail_E starts, is English, has identical ids across variants
            ts = a["tail_start"]
            assert tok.decode(v.ids[ts:]) == " " + v.segments[-1].text
            assert all(x == "E" for x in a["lang"][ts:]) and all(u == 11 for u in a["unit"][ts:])
            assert set(a["unit"][:ts]) <= set(range(1, 11))
            assert a["lang"] == [v.mask[s] if s < 30 else "E" for s in a["seg"]]
            assert a["n_tokens_E"] + a["n_tokens_D"] == n and a["n_tokens_E"] == a["lang"].count("E")
            assert a["n_words_E"] + a["n_words_D"] == a["n_words"] == len(v.text.split())
            # distance to the most recent switch
            last = -1
            for t in range(n):
                if a["klass"][t] == "boundary":
                    last = t
                assert a["dist_to_last_switch"][t] == (t - last if last >= 0 else -1)
            if v.K == 0:
                assert set(a["lang"]) == {"E"} and a["n_tokens_D"] == 0 and a["last_switch_pos"] == -1
            else:
                assert a["n_tokens_D"] > 0 and a["last_switch_pos"] < ts
            # the tokens assigned to a segment spell exactly that segment (plus its period)
            for s_idx, seg in enumerate(v.segments):
                joined = "".join(a["tokens"][t] for t in range(n) if a["seg"][t] == s_idx).strip()
                assert joined.startswith(seg.text), (v.tag, seg, joined)
                assert joined[len(seg.text):] in ("", ".")


def test_annotation_segment_languages(variants):
    v = [x for x in variants["DEV01A"] if (x.K, x.C) == (2, "C1")][0]
    a = v.annot
    for t in range(a["n_tokens"]):
        u = a["unit"][t]
        if u in (5, 6, 7, 8, 9):
            assert a["lang"][t] == "D"
        else:
            assert a["lang"][t] == "E"
    boundaries = a["boundary_positions"]
    assert len(boundaries) == 2
    assert a["tokens"][boundaries[0]].strip() == "Ein" and a["tokens"][boundaries[1]].strip() == "The"
    assert a["unit"][boundaries[0]] == 5 and a["unit"][boundaries[1]] == 10


def test_F100_dossier_masks(variants):
    for tag, vs in variants.items():
        for v in vs:
            n_words = v.annot["n_words"]
            assert v.annot["F100"] == pytest.approx(100.0 * v.K / (n_words - 1))
        v2 = [x for x in vs if (x.K, x.C) == (2, "C1")][0]
        assert v2.annot["F100"] == pytest.approx(100.0 * 2 / (v2.annot["n_words"] - 1))
        assert vs[0].annot["F100"] == 0.0
        c1 = [x for x in vs if x.C == "C1" or x.C == "seg"]
        f = [x.annot["F100"] for x in sorted(c1, key=lambda x: x.K)]
        assert f == sorted(f)


def test_target_single_token(tok, dev01):
    fam = dev01["DEV01A"]
    text = S.render_text(fam, ["E"] * 30)
    assert S.target_token_id(tok, text, " bark") == tok(" bark", add_special_tokens=False)["input_ids"][0]
    with pytest.raises(ValueError):
        S.target_token_id(tok, text, " Fewkoombey")
    assert S.target_token_id(tok, text, fam.target) == 21939


def test_annotate_direct(tok):
    text = "Kraft reads the report. Er liest den Bericht. The end"
    segs = [("Kraft", "E", 1, "subj"), ("reads the report", "E", 1, "pred"),
            ("Er", "D", 2, "subj"), ("liest den Bericht", "D", 2, "pred"), ("The end", "E", 3, "tail")]
    a = S.annotate(tok, text, segs)
    toks = a["tokens"]
    assert a["n_switches"] == 2 and len(a["boundary_positions"]) == 2
    assert toks[a["boundary_positions"][0]] == " Er" and toks[a["boundary_positions"][1]] == " The"
    assert all(a["klass"][b] == "boundary" for b in a["boundary_positions"])
    assert a["tail_start"] == a["boundary_positions"][1]
    assert a["lang"][toks.index(".")] == "E"          # the period belongs to the preceding segment
    assert a["n_words"] == 10 and a["n_words_D"] == 4 and a["n_words_E"] == 6
    assert a["klass"][0] == "word_start" and a["klass"][1] == "word_cont"      # "K" + "raft"
    with pytest.raises(ValueError):
        S.annotate(tok, text, [("nonexistent", "E", 1, "subj")])


def test_variant_summary(variants):
    df = S.variant_summary(variants["DEV01B"])
    assert len(df) == 12 and {"K", "C", "F100", "n_tokens", "n_words", "tail_start"} <= set(df.columns)
    assert df["target_id"].nunique() == 1


# ======================================================================================
# original State 2143 case study
# ======================================================================================
@pytest.mark.network
def test_original_family_reproduces_anchor(anchor, tok):
    fam = S.original_2143_family(anchor, tok)
    assert fam.kind == "original" and fam.n_units == 6 and fam.target_id == 22905
    spec = S.load_original_spec()
    bounds = spec["unit_token_bounds"]
    assert render_original_all_E(fam) == anchor.text
    assert fam.units[1].D["text"].startswith("\n\n")
    # tail = unit 7 = tokens 186..207, starts with "For", preceded by ".—" (25277)
    assert fam.tail_E.startswith("For") and bounds[-2] == 186
    assert anchor.context_ids[185] == 25277 and tok.decode([25277]) == ".—"
    assert tok.decode(anchor.context_ids[186:208]) == fam.tail_E

    vs = S.build_original_variants(anchor, tok)
    assert [v.K for v in vs] == [0, 2, 4, 5] and all(v.C == "orig" for v in vs)
    ref = S.reference_variant(vs)
    assert ref.ids == list(anchor.context_ids)
    assert ref.annot["tail_start"] == 186 and ref.tail_ids == list(anchor.context_ids[186:208])
    for v in vs:
        assert v.target_id == 22905
        assert v.tail_ids == list(anchor.context_ids[186:208])
        assert v.ids[v.tail_start - 1] == 25277
        assert v.annot["tokens"][v.tail_start] == "For"
        assert S.check_variant(v, ref) == []
        assert set(v.annot["unit"]) <= set(range(1, 8)) and all(u == 7 for u in v.annot["unit"][v.tail_start:])
        expected_switches = v.K + (1 if v.mask[-1] == "D" else 0)     # D->E at the tail boundary
        assert v.annot["n_switches"] == expected_switches
        assert v.annot["F100"] == pytest.approx(100.0 * v.K / (v.annot["n_words"] - 1))
        # language-defined site: last German token, or the last token of unit 6 for K=0
        pos = S.last_switched_position(v)
        assert pos < v.tail_start
        if v.K == 0:
            assert pos == v.tail_start - 1 and v.annot["unit"][pos] == 6
        else:
            assert v.annot["lang"][pos] == "D" and all(x == "E" for x in v.annot["lang"][pos + 1:])
            last_D_unit = max(i + 1 for i, m in enumerate(v.mask) if m == "D")
            assert v.annot["unit"][pos] == last_D_unit
    assert vs[3].annot["klass"][vs[3].tail_start] == "boundary"       # K=5 ends in D
    assert S.last_switched_position(vs[3]) == vs[3].tail_start - 1


def render_original_all_E(fam):
    return S.render_original(fam, "EEEEEE")


@pytest.mark.network
def test_original_render_concatenates_directly(anchor, tok):
    fam = S.original_2143_family(anchor, tok)
    mask = "EDEDED"
    text = S.render_original(fam, mask)
    parts = [u.seg(l, "text") for u, l in zip(fam.units, mask)] + [fam.tail_E]
    assert text == "".join(parts)
    text2, segs = S.render(fam, list(mask))
    assert text2 == text and len(segs) == 7 and segs[-1].name == "tail" and segs[-1].unit == 7
    with pytest.raises(ValueError):
        S.render_original(fam, "EEE")
    with pytest.raises(ValueError):
        S.build_variants(fam, tok)
