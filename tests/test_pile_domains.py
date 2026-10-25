"""Tests for lsne.pile_domains: keyword-profile classification (hand-written texts),
sequence NLL (pythia-14m, cross-checked against lsne.model.forward_logits), the
difficulty table / ranking on synthetic sequences, the new lexicon domains, and one
slow+network test that fetches 3 Pile sequences by range request."""
import re

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.atlas import load_domains
from lsne.model import forward_logits
from lsne.pile_domains import (N_PILE_SEQUENCES, PILE_MARKERS, SUBFIELD_CHILDREN, classify_sequence, classify_text,
                               difficulty_ranking, hardest_subfield, lexicon_markers, misfolding_rank_check,
                               sample_indices, sample_sequence_difficulty, sequence_difficulty_table, sequence_nll)

TEXTS = {
    "law": "The district court granted the defendant's motion to dismiss. On appeal, the plaintiff argued that "
           "the statute, pursuant to 28 U.S.C. 1331, conferred jurisdiction. In Smith v. Jones the court held "
           "otherwise, and the judgment was affirmed.",
    "protein": "The kinase domain binds ATP, and the catalytic residue Asp166 is required for substrate "
               "phosphorylation. The crystal structure of the enzyme shows an alpha-helix packed against the "
               "peptide-binding groove; the mutant protein lost ligand binding.",
    "protein_misfolding": "Prion protein misfolding converts the native alpha-helical conformation into beta-sheet "
                          "rich amyloid fibrils. These aggregates seed further aggregation; oligomers of amyloid-beta "
                          "and tau form plaques and tangles in Alzheimer disease, and chaperones counteract the "
                          "misfolded protein.",
    "code": "def load(path):\n    import json\n    with open(path) as f:\n        data = json.load(f)\n"
            "    if data == None:\n        return []\n    return [x for x in data if x != 0]\n\n"
            "class Foo:\n    def __init__(self):\n        self.x = 1\n",
    "math": "Theorem 2. Let $G$ be a finite group and $H$ a subgroup. Then $|H|$ divides $|G|$. Proof. By Lemma 1 "
            "the cosets partition $G$. \\begin{equation} |G| = [G:H]\\,|H| \\end{equation} which proves the "
            "proposition for every integer $n$.",
    "pubmed": "Abstract: Patients (n=120) were randomized to placebo or treatment. Methods: a cohort study. Results: "
              "expression was significantly higher (p < 0.01) in the clinical group; in vitro assays confirmed "
              "this (Smith et al., doi:10.1000/xyz).",
    "other": "I went to the kitchen and made a coffee, then took the bus to the shop to buy bread for breakfast. "
             "The weather was fine and the garden looked nice.",
}


# --------------------------------------------------------------------------------------
def test_markers_well_formed():
    for name in ("pubmed", "protein", "protein_misfolding", "code", "math", "law", "news", "web", "books", "wikipedia"):
        assert name in PILE_MARKERS and len(PILE_MARKERS[name]) >= 5
    for name, ms in PILE_MARKERS.items():
        assert all(isinstance(m, str) and m for m in ms)
        for m in ms:
            if m.startswith("re:"):
                re.compile(m[3:])
            else:
                assert m == m.lower(), (name, m)                      # substrings are matched on lowercased text
    assert SUBFIELD_CHILDREN["protein"] == ("protein_misfolding",)
    assert N_PILE_SEQUENCES == 146_432_000


def test_classify_text_on_hand_written_texts():
    for expected, text in TEXTS.items():
        res = classify_text(text)
        assert res["subfield"] == expected, (expected, res["subfield"], res["hits"])
        assert set(res["scores"]) == set(PILE_MARKERS) and res["n_words"] > 0
        assert all(s >= 0 for s in res["scores"].values())
    # hierarchy: misfolding wins over protein although protein markers hit as well
    res = classify_text(TEXTS["protein_misfolding"])
    assert res["hits"]["protein"] >= 3 and res["subfield"] == "protein_misfolding"
    # a PubMed-style abstract about amyloid is refined to the most specific child
    res = classify_text(TEXTS["pubmed"] + " Amyloid fibrils and prion aggregates were observed in the amyloid plaques.")
    assert res["subfield"] == "protein_misfolding" and res["hits"]["pubmed"] >= 3
    # a high min_hits threshold gives 'other'
    assert classify_text(TEXTS["law"], min_hits=100)["subfield"] == "other"
    # custom markers (and the regex convention)
    custom = {"foo": ["foo", r"re:\bbar\d+\b"], "baz": ["baz"]}
    r = classify_text("foo foo bar1 bar22 nothing", markers=custom, min_hits=1)
    assert r["subfield"] == "foo" and r["hits"] == {"foo": 4, "baz": 0}
    assert np.isclose(r["scores"]["foo"], 4 / 2)
    assert classify_text("foo " * 100, markers=custom, min_hits=1, cap=10)["scores"]["foo"] == 5.0   # capped
    with pytest.raises(ValueError):
        classify_text("x", markers={"empty": []})


def test_classify_sequence_round_trip_through_tokenizer(tok):
    for expected, text in TEXTS.items():
        ids = np.asarray(tok(text, add_special_tokens=False)["input_ids"], dtype=np.uint16)
        assert classify_sequence(tok, ids) == expected
    ids = torch.tensor(tok(TEXTS["code"], add_special_tokens=False)["input_ids"])
    res = classify_sequence(tok, ids, return_scores=True)
    assert isinstance(res, dict) and res["subfield"] == "code" and res["hits"]["code"] >= 3


def test_lexicon_domains_and_markers(tok):
    dom = load_domains()["domains"]
    for name in ("biomed_protein", "protein_misfolding"):
        assert name in dom
        for lang in ("E", "D"):
            words = dom[name][lang]
            assert len(words) >= 40 and len(set(words)) == len(words)
            assert all(isinstance(w, str) and w.strip() == w and w for w in words)
    assert {"prion", "amyloid", "fibril", "chaperone", "misfolding", "tau", "synuclein", "scrapie", "tangle"} <= set(dom["protein_misfolding"]["E"])
    assert all(w[0].isupper() for w in dom["protein_misfolding"]["D"])          # German nouns capitalised
    mk = lexicon_markers(load_domains())
    assert set(mk) == set(dom) and all(m.startswith("re:") for ms in mk.values() for m in ms)
    # classification into the lexicon domains (both languages, inflected forms)
    text_d = "Prionen, Amyloid und Fibrillen: die Fehlfaltung des Proteins erzeugt Plaques, Oligomere und Aggregate."
    r = classify_text(text_d, markers=mk, min_hits=2)
    assert r["subfield"] == "protein_misfolding" and r["hits"]["protein_misfolding"] >= 6
    assert classify_text(TEXTS["law"], markers=mk, min_hits=2)["subfield"] == "law"
    assert classify_text(TEXTS["protein_misfolding"], markers=mk, min_hits=2)["subfield"] == "protein_misfolding"


# --------------------------------------------------------------------------------------
def test_sequence_nll_matches_forward_logits(mb, sample_ids):
    ids = sample_ids[0].numpy()
    res = sequence_nll(mb, ids, max_len=512)
    T = ids.shape[0]
    assert res["n_tokens"] == T - 1 and res["nll_per_token"].shape == (T - 1,)
    assert np.isfinite(res["nll_per_token"]).all() and (res["nll_per_token"] >= 0).all()
    logits = forward_logits(mb, sample_ids)[0]
    logp = torch.log_softmax(logits[:-1], -1)
    manual = -logp[torch.arange(T - 1), sample_ids[0, 1:]].numpy()
    assert np.allclose(res["nll_per_token"], manual, atol=1e-5)
    assert np.isclose(res["mean_nll"], manual.mean())
    n_tail = int(np.ceil(0.25 * (T - 1)))
    assert np.isclose(res["mean_nll_tail"], manual[-n_tail:].mean())
    # truncation (causal model: the first 7 predictions agree up to float32 kernel round-off)
    short = sequence_nll(mb, ids, max_len=8)
    assert short["n_tokens"] == 7 and np.allclose(short["nll_per_token"], manual[:7], atol=1e-3)
    assert np.allclose(sequence_nll(mb, torch.tensor(ids), max_len=8)["nll_per_token"], short["nll_per_token"])
    with pytest.raises(ValueError):
        sequence_nll(mb, ids[:1])


def test_difficulty_table_and_ranking_on_synthetic_sequences(mb, tok):
    texts = [TEXTS["law"], TEXTS["law"] + " The tribunal and the attorney for the plaintiff appealed again.",
             TEXTS["protein_misfolding"], TEXTS["protein_misfolding"] + " Amyloid plaques and prion fibrils.",
             TEXTS["code"], TEXTS["other"]]
    seqs = [np.asarray(tok(t, add_special_tokens=False)["input_ids"], dtype=np.uint16) for t in texts]
    df = sequence_difficulty_table(mb, seqs, max_len=64)
    assert list(df.columns) == ["seq_index", "subfield", "n_tokens", "mean_nll", "mean_nll_tail", "score", "n_hits"]
    assert len(df) == 6 and df.seq_index.tolist() == list(range(6))
    assert df.subfield.tolist() == ["law", "law", "protein_misfolding", "protein_misfolding", "code", "other"]
    assert (df.n_tokens <= 63).all() and np.isfinite(df[["mean_nll", "mean_nll_tail"]].values).all()
    assert (df[df.subfield != "other"].n_hits >= 3).all() and (df[df.subfield == "other"].n_hits == 0).all()
    assert np.isclose(df.mean_nll[0], sequence_nll(mb, seqs[0], max_len=64)["mean_nll"])
    rk = difficulty_ranking(df, n_boot=200, seed=0)
    assert list(rk.columns) == ["subfield", "n_seqs", "mean_nll", "sd", "se", "ci_low", "ci_high", "mean_nll_tail", "rank"]
    assert len(rk) == 4 and rk["rank"].tolist() == [1, 2, 3, 4]
    assert rk.mean_nll.is_monotonic_decreasing                                   # hardest first
    assert ((rk.ci_low <= rk.mean_nll + 1e-12) & (rk.mean_nll <= rk.ci_high + 1e-12)).all()
    for name, g in df.groupby("subfield"):
        row = rk[rk.subfield == name].iloc[0]
        assert row.n_seqs == len(g) and np.isclose(row.mean_nll, g.mean_nll.mean())
    assert hardest_subfield(rk, exclude=()) == rk.iloc[0].subfield
    assert hardest_subfield(rk) == rk[rk.subfield != "other"].iloc[0].subfield
    chk = misfolding_rank_check(rk)
    assert chk["present"] and chk["n_subfields"] == 3 and 1 <= chk["rank"] <= 3
    assert chk["in_hardest_third"] == (chk["rank"] <= 1)
    assert chk["hardest"] == hardest_subfield(rk)
    absent = misfolding_rank_check(rk, subfield="math")
    assert not absent["present"] and absent["rank"] is None and not absent["in_hardest_third"]
    # a sequence index list is passed through
    assert sequence_difficulty_table(mb, seqs[:2], [10, 20], max_len=32).seq_index.tolist() == [10, 20]
    with pytest.raises(ValueError):
        sequence_difficulty_table(mb, seqs[:2], [1], max_len=32)


def test_difficulty_ranking_on_synthetic_nll_table():
    rng = np.random.default_rng(0)
    rows = []
    for name, mu in (("code", 1.0), ("math", 4.0), ("protein_misfolding", 3.0), ("other", 2.0), ("law", 2.5)):
        for i in range(30):
            rows.append(dict(seq_index=i, subfield=name, n_tokens=511, mean_nll=mu + 0.1 * rng.standard_normal(),
                             mean_nll_tail=mu - 0.2))
    df = pd.DataFrame(rows)
    rk = difficulty_ranking(df, n_boot=500, seed=1)
    assert rk.subfield.tolist() == ["math", "protein_misfolding", "law", "other", "code"]
    assert rk["rank"].tolist() == [1, 2, 3, 4, 5] and (rk.n_seqs == 30).all()
    assert (rk.ci_high - rk.ci_low > 0).all() and (rk.ci_high - rk.ci_low < 0.2).all()
    assert np.allclose(rk.mean_nll_tail, rk.mean_nll - 0.2, atol=0.05)
    assert np.allclose(rk.se, rk.sd / np.sqrt(30))
    assert hardest_subfield(rk) == "math" and hardest_subfield(rk, exclude=("math", "other")) == "protein_misfolding"
    chk = misfolding_rank_check(rk)
    assert chk == dict(subfield="protein_misfolding", rank=2, n_subfields=4, present=True, in_hardest_third=True, hardest="math")
    assert difficulty_ranking(df, n_boot=0).ci_low.equals(difficulty_ranking(df, n_boot=0).mean_nll)
    assert difficulty_ranking(df, min_seqs=31).empty
    with pytest.raises(ValueError):
        difficulty_ranking(df.drop(columns="mean_nll"))
    with pytest.raises(ValueError):
        hardest_subfield(rk, exclude=tuple(rk.subfield))


def test_sample_indices_deterministic_and_distinct():
    idx = sample_indices(50, seed=3)
    assert idx.shape == (50,) and len(set(idx.tolist())) == 50 and (idx >= 0).all() and (idx < N_PILE_SEQUENCES).all()
    assert np.array_equal(idx, sample_indices(50, seed=3)) and not np.array_equal(idx, sample_indices(50, seed=4))


@pytest.mark.slow
@pytest.mark.network
def test_sample_sequence_difficulty_three_pile_sequences(mb, tmp_path):
    """Fetches 3 training sequences by HTTP range request (~4 KB each), classifies them and
    measures their NLL under pythia-14m; checks the npy / csv caches."""
    import requests
    try:
        df = sample_sequence_difficulty(mb, n_sequences=3, seed=0, cache_dir=tmp_path, max_len=128)
    except (requests.RequestException, RuntimeError) as e:          # pragma: no cover - offline
        pytest.skip(f"Pile range request unavailable: {e}")
    assert list(df.columns) == ["seq_index", "subfield", "n_tokens", "mean_nll", "mean_nll_tail", "score", "n_hits"]
    assert len(df) == 3 and df.seq_index.tolist() == sample_indices(3, 0).tolist()
    assert (df.n_tokens == 127).all() and np.isfinite(df[["mean_nll", "mean_nll_tail"]].values).all()
    assert df.subfield.isin(list(PILE_MARKERS) + ["other"]).all()
    assert df.attrs["cached"] is False and df.attrs["n_sequences"] == 3
    assert sorted(p.name for p in (tmp_path / "seqs").glob("seq_*.npy")) == sorted(f"seq_{i}.npy" for i in df.seq_index)
    csvs = list(tmp_path.glob("pile_nll_n3_s0_*_L128.csv"))
    assert len(csvs) == 1
    df2 = sample_sequence_difficulty(mb, n_sequences=3, seed=0, cache_dir=tmp_path, max_len=128)
    assert df2.attrs["cached"] is True
    pd.testing.assert_frame_equal(df2, df, check_dtype=False)
    rk = difficulty_ranking(df, n_boot=20)
    assert len(rk) == df.subfield.nunique() and rk["rank"].iloc[0] == 1
