"""Tests for lsne.atlas (pythia-14m, CPU) and the domain lexicon data/lexicon/domains.json."""
import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite
from lsne.directions import Direction, random_directions
from lsne.model import logit_lens, make_dose_delta, run_with_cache
from lsne.sae import TopKSAE
from lsne.atlas import (DEFAULT_DOMAINS_PATH, atlas_domain_overlap, counter_steer, domain_axis, domain_names,
                        domain_profile, domain_token_ids, keyword_readout, load_atlas_extraction, load_domains,
                        sae_feature_domain_profile)

#: The nine compass domains of DESIGN section 4.7 (disjoint English word lists, >= 30
#: single-token words each) ...
CORE_DOMAINS = ["law", "politics", "economy", "culture_values", "family", "religion", "wellbeing", "science", "everyday"]
#: ... plus the two RDX contrast poles of DESIGN section 4.8b (biomedical vocabulary
#: overlaps `science` / `wellbeing` by design; rarer nomenclature -> fewer single-token words).
POLE_DOMAINS = ["biomed_protein", "protein_misfolding"]
#: Every domain of data/lexicon/domains.json, in file order (profiles have one row per domain).
DOMAINS = CORE_DOMAINS + POLE_DOMAINS


@pytest.fixture(scope="module")
def domains():
    return load_domains()


@pytest.fixture(scope="module")
def ids_E(tok, domains):
    return domain_token_ids(tok, domains, "E")


@pytest.fixture(scope="module")
def h_last(mb, sample_ids):
    site = HookSite(mb.n_layers - 1, "block_output")
    return run_with_cache(mb, sample_ids, [site]).resid[site.key][0, -1].clone()


# --------------------------------------------------------------------------------------
def test_domains_file_schema(domains):
    assert DEFAULT_DOMAINS_PATH.exists()
    assert domain_names(domains) == DOMAINS
    for name in DOMAINS:
        for lang in ("E", "D"):
            words = domains["domains"][name][lang]
            assert len(words) >= 40, (name, lang, len(words))
            assert len(set(words)) == len(words)
            assert all(w.strip() == w and " " not in w for w in words)
            if lang == "E":
                assert all(w == w.lower() for w in words)
    for lang in ("E", "D"):
        ts = domains["templates"][lang]
        assert len(ts) >= 6
        assert all(t.count("{w}") == 1 and t.endswith("{w}") for t in ts)
    # English word lists are disjoint across the compass domains (the section-4.8b poles
    # share biomedical vocabulary with `science` / `wellbeing` by construction and are only
    # required to be disjoint from each other)
    seen = {}
    for name in CORE_DOMAINS:
        for w in domains["domains"][name]["E"]:
            assert w not in seen, (w, name, seen.get(w))
            seen[w] = name
    pole_words = [set(domains["domains"][n]["E"]) for n in POLE_DOMAINS]
    assert not (pole_words[0] & pole_words[1]), pole_words[0] & pole_words[1]
    with pytest.raises(FileNotFoundError):
        load_domains("/nonexistent/domains.json")


def test_domain_token_ids_keeps_single_token_words(tok, ids_E):
    toy = {"x": {"E": ["court", "antidisestablishmentarianism", "kitchen", "court"],
                 "D": ["Gericht", "Minister"]}}
    out = domain_token_ids(tok, toy, "E")
    assert list(out) == ["x"] and out["x"] == [tok(" court", add_special_tokens=False)["input_ids"][0],
                                               tok(" kitchen", add_special_tokens=False)["input_ids"][0]]
    assert out.meta["dropped"]["x"] == ["antidisestablishmentarianism"]
    assert out.meta["kept"]["x"] == ["court", "kitchen", "court"] and out.meta["n_dropped"]["x"] == 1
    outD = domain_token_ids(tok, toy, "D")
    assert outD["x"] == [tok(" Minister", add_special_tokens=False)["input_ids"][0]]
    assert outD.meta["dropped"]["x"] == ["Gericht"]
    # the inner mapping and the full file dict are both accepted
    assert domain_token_ids(tok, {"domains": toy, "templates": {}}, "E")["x"] == out["x"]
    with pytest.raises(ValueError):
        domain_token_ids(tok, toy, "F")
    # shipped lexicon: every English compass domain keeps plenty of single-token words; the
    # protein poles carry rarer nomenclature (prion, amyloid, ... are multi-token) but still
    # enough ids for a mass readout
    for name in DOMAINS:
        assert len(ids_E[name]) >= (30 if name in CORE_DOMAINS else 20), (name, len(ids_E[name]))
        for i in ids_E[name]:
            assert tok.decode([i]).startswith(" ")
    assert ids_E.meta["lang"] == "E"


def test_keyword_readout(mb, h_last):
    r = keyword_readout(mb, h_last, None, top_k=7)
    assert r["mode"] == "lens" and len(r["promoted"]) == 7 and len(r["suppressed"]) == 7
    assert all(isinstance(t, str) and isinstance(s, float) for t, s in r["promoted"])
    scores = [s for _, s in r["promoted"]]
    assert scores == sorted(scores, reverse=True)
    lens = logit_lens(mb, h_last)
    assert torch.allclose(r["scores"], lens) and np.isclose(scores[0], float(lens[:len(mb.tokenizer)].max()))
    d = random_directions(mb.d_model, 1, 2)[0]
    r2 = keyword_readout(mb, h_last, d, eps=0.3, top_k=10)
    assert r2["mode"] == "delta" and len(r2["promoted"]) == 10 and len(r2["suppressed_ids"]) == 10
    expected = logit_lens(mb, h_last + make_dose_delta(h_last, d, 0.3)) - lens
    assert torch.allclose(r2["scores"], expected, atol=1e-5)
    assert r2["promoted"][0][1] >= r2["promoted"][-1][1] >= r2["suppressed"][-1][1] >= r2["suppressed"][0][1]
    r3 = keyword_readout(mb, h_last, Direction("x", d, HookSite(0)), eps=0.3, top_k=10)
    assert torch.allclose(r3["scores"], r2["scores"])
    with pytest.raises(ValueError):
        keyword_readout(mb, h_last[:-1])


def test_domain_profile_rows_and_finite_z(mb, h_last, ids_E):
    d = random_directions(mb.d_model, 1, 5)[0]
    df = domain_profile(mb, h_last, d, eps=0.2, domains_ids=ids_E, n_random=16, seed=0)
    assert list(df.domain) == DOMAINS and len(df) == len(DOMAINS)
    for col in ("mass", "mass_random_mean", "mass_random_sd", "z"):
        assert np.isfinite(df[col]).all(), col
    assert np.allclose(df.mass, df.mass_after - df.mass_before)
    assert (df.n_ids == [len(ids_E[n]) for n in DOMAINS]).all()
    assert (df.mass_before > 0).all() and (df.mass_before <= 1).all()
    assert df.attrs["mode"] == "step" and df.attrs["n_random"] == 16
    # deterministic in the seed; the direction itself is not in its own null -> |z| finite and reproducible
    df2 = domain_profile(mb, h_last, d, eps=0.2, domains_ids=ids_E, n_random=16, seed=0)
    assert np.allclose(df.z, df2.z)
    # state profile (no direction): mass == mass_before, random token-set null
    df0 = domain_profile(mb, h_last, None, domains_ids=ids_E, n_random=8, seed=1)
    assert df0.attrs["mode"] == "state" and np.allclose(df0.mass, df0.mass_before)
    assert np.isfinite(df0.z).all() and (df0.mass_random_sd >= 0).all()
    # default lexicon is loaded when domains_ids is None
    df_def = domain_profile(mb, h_last, d, eps=0.2, n_random=4)
    assert list(df_def.domain) == DOMAINS


def test_sae_feature_domain_profile(mb, ids_E):
    d, F = mb.d_model, 12
    g = torch.Generator().manual_seed(0)
    W_enc = torch.randn(F, d, generator=g)
    W_dec = torch.randn(F, d, generator=g)
    W = mb.lm_head.weight.detach().float()
    gain = mb.final_ln.weight.detach().float()
    W_dec[1] = gain * W[torch.tensor(ids_E["law"])].mean(0)      # planted law feature
    sae = TopKSAE.from_tensors(W_enc, torch.zeros(F), W_dec, torch.zeros(d), 3)
    df = sae_feature_domain_profile(mb, sae, 0, ids_E)
    assert list(df.columns) == ["feature", "domain", "mean_logit", "mean_all", "sd_all", "z", "z_token", "n_ids"]
    assert list(df.domain) == DOMAINS and (df.feature == 0).all()
    assert np.isfinite(df[["mean_logit", "z", "z_token"]].values).all()
    assert np.allclose(df.z, df.z_token * np.sqrt(df.n_ids))
    assert np.allclose(df.z_token, (df.mean_logit - df.mean_all) / df.sd_all)
    df1 = sae_feature_domain_profile(mb, sae, [1, 0], ids_E)
    assert len(df1) == 2 * len(DOMAINS) and list(df1.feature.unique()) == [1, 0]
    law = df1[(df1.feature == 1)].set_index("domain").z
    assert law.idxmax() == "law"


def test_domain_axis_unit_vector_site_and_counts(mb, tok):
    site = HookSite(mb.n_layers // 2, "block_input")
    ax = domain_axis(mb, tok, "law", site, contrast="everyday", lang="E", n_templates=2, max_words=6)
    assert isinstance(ax, Direction) and ax.name == "dom_law" and ax.site == site
    assert ax.vec.shape == (mb.d_model,) and torch.isclose(ax.vec.norm(), torch.tensor(1.0), atol=1e-6)
    m = ax.meta
    assert m["n_domain"] == 12 and m["n_contrast"] == 12 and m["n_templates"] == 2
    assert m["n_words_domain"] == 6 and m["n_words_contrast"] == 6 and m["lang"] == "E"
    assert m["dropped_domain"] == [] and m["dropped_contrast"] == [] and m["n_mismatch"] == 0
    assert -1 <= m["cos_denoised_plain"] <= 1 and m["diff_norm"] > 0
    # several domains / German side / single-token filter
    axD = domain_axis(mb, None, ["culture_values", "politics"], site, contrast="everyday", lang="D",
                      n_templates=2, max_words=5, single_token_only=True)
    assert axD.name == "dom_culture_values+politics" and axD.meta["domain"] == ["culture_values", "politics"]
    assert axD.meta["n_domain"] > 0 and axD.meta["n_contrast"] > 0
    assert len(axD.meta["dropped_domain"]) > 0                     # German nouns are mostly multi-token
    assert axD.meta["n_domain"] == 2 * axD.meta["n_words_domain"]
    with pytest.raises(KeyError):
        domain_axis(mb, tok, "astrology", site)
    with pytest.raises(ValueError):
        domain_axis(mb, tok, "law", site, contrast="law")


def test_counter_steer_removes_component(mb, h_last):
    d = mb.d_model
    site = HookSite(0)
    remove = Direction("dom_law", random_directions(d, 1, 7)[0], site)
    then = Direction("dom_culture", random_directions(d, 1, 8)[0], site)
    out, info = counter_steer(h_last, remove, then, None, 0.1, return_info=True)
    assert out.shape == h_last.shape and out.dtype == h_last.dtype
    projected = out - make_dose_delta(h_last, then.vec, 0.1)
    assert abs(float(projected @ remove.vec) / float(projected.norm())) < 1e-5     # cosine with `remove` ~ 0
    assert abs(info["cos_after"]) < 1e-5 and np.isclose(info["removed_coef"], float(h_last @ remove.vec))
    assert np.isclose(info["added_norm"], float(make_dose_delta(h_last, then.vec, 0.1).norm()))
    # fixed-dose removal: an RMS dose eps_remove along unit(remove), then the step
    out2 = counter_steer(h_last, remove.vec, then, 0.2, 0.0)
    removed = h_last - out2
    assert np.isclose(float(removed.pow(2).mean().sqrt() / h_last.pow(2).mean().sqrt()), 0.2, rtol=1e-4)
    assert abs(float(removed @ remove.vec) / float(removed.norm()) - 1.0) < 1e-5
    # eps_then adds exactly the dose vector on top of the counter-steered state
    out3 = counter_steer(h_last, remove.vec, then.vec, 0.2, 0.3)
    assert torch.allclose(out3 - out2, make_dose_delta(h_last, then.vec, 0.3), atol=1e-6)
    # mean-centred projection
    mean = torch.full((d,), 0.5)
    out4 = counter_steer(h_last, remove, then, None, 0.0, mean=mean)
    assert abs(float((out4 - mean) @ remove.vec)) < 1e-5
    with pytest.raises(ValueError):
        counter_steer(h_last, remove.vec[:-1], then, None, 0.1)


def test_load_atlas_extraction_parses_example(tmp_path):
    txt = ("# susceptibility atlas extraction (example)\n"
           "Cluster 1: court, judge, Gericht; verdict\n"
           "- lawyer\n"
           "- weather\n"
           "\n"
           "2: kitchen, bus, Garten\n"
           "\n"
           "some keywords, no id, church\n"
           "- faith\n"
           "- \"prayer\"\n")
    p = tmp_path / "atlas.txt"
    p.write_text(txt, encoding="utf-8")
    cl = load_atlas_extraction(str(p))
    assert [c["cluster_id"] for c in cl] == ["1", "2", "block2"]
    assert cl[0]["keywords"] == ["court", "judge", "Gericht", "verdict", "lawyer", "weather"]
    assert cl[1]["keywords"] == ["kitchen", "bus", "Garten"]
    assert cl[2]["keywords"] == ["some keywords", "no id", "church", "faith", "prayer"]
    assert all(set(c) >= {"cluster_id", "keywords", "raw"} for c in cl)
    assert "Cluster 1" in cl[0]["raw"] and cl[1]["raw"] == "2: kitchen, bus, Garten"
    assert load_atlas_extraction(str(tmp_path / "missing.txt")) == []
    # a directory of files is parsed too
    (tmp_path / "b.txt").write_text("7: faith, prayer\n", encoding="utf-8")
    cl_dir = load_atlas_extraction(str(tmp_path))
    assert len(cl_dir) == 4 and {c["source"] for c in cl_dir} == {"atlas.txt", "b.txt"}


def test_atlas_domain_overlap(domains):
    clusters = [dict(cluster_id="1", keywords=["Court", "judge", "gericht", "weather", "xyzzy"]),
                dict(cluster_id="2", keywords=[])]
    df = atlas_domain_overlap(clusters, domains)
    assert len(df) == 2 * len(DOMAINS)
    assert list(df.columns)[:4] == ["cluster_id", "domain", "n_overlap", "jaccard"]
    law = df[(df.cluster_id == "1") & (df.domain == "law")].iloc[0]
    n_law = len(set(w.lower() for w in domains["domains"]["law"]["E"] + domains["domains"]["law"]["D"]))
    assert law.n_overlap == 3 and law.overlap == ["court", "gericht", "judge"]
    assert np.isclose(law.jaccard, 3 / (5 + n_law - 3))
    every = df[(df.cluster_id == "1") & (df.domain == "everyday")].iloc[0]
    assert every.n_overlap == 1 and every.overlap == ["weather"]
    assert (df[df.cluster_id == "2"].n_overlap == 0).all() and (df[df.cluster_id == "2"].jaccard == 0).all()
    assert df[(df.cluster_id == "1") & (df.domain == "science")].iloc[0].n_overlap == 0
    # English only
    df_e = atlas_domain_overlap(clusters, domains, langs=("E",))
    assert df_e[(df_e.cluster_id == "1") & (df_e.domain == "law")].iloc[0].n_overlap == 2
