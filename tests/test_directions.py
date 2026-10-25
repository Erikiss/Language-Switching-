"""Tests for lsne.directions (CPU, pythia-14m fixture, synthetic E/D stimuli)."""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite
from lsne.directions import (Direction, collect_activations, cosine_table, diff_of_means,
                             entropy_direction, frequency_axis, language_axis, load_axis_npz,
                             probe_axis, random_directions, unembed_projection, unit)
from lsne.model import make_dose_delta

ENGLISH = [
    "The soldier reads the report in the morning.",
    "My sister writes a letter to her friend.",
    "The children play in the garden after school.",
    "The old man drinks his coffee by the window.",
    "Our neighbour repairs the fence every summer.",
    "The teacher explains the lesson to the class.",
]
GERMAN = [
    "Der Soldat liest den Bericht am Morgen.",
    "Meine Schwester schreibt einen Brief an ihre Freundin.",
    "Die Kinder spielen nach der Schule im Garten.",
    "Der alte Mann trinkt seinen Kaffee am Fenster.",
    "Unser Nachbar repariert jeden Sommer den Zaun.",
    "Die Lehrerin erklärt der Klasse die Lektion.",
]
TAIL = "The report is about the"


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def make_variants(tok):
    """Minimal variant objects: unit 1 English, unit 2 German, then a shared English tail.
    Built without lsne.stimuli (SimpleNamespace with ids, target_id, annot, ...)."""
    out = []
    target_id = _ids(tok, " soldier")[0]
    for i, (e, d) in enumerate(zip(ENGLISH, GERMAN)):
        seg_ids = [_ids(tok, e), _ids(tok, " " + d), _ids(tok, " " + TAIL)]
        langs = ["E", "D", "E"]
        ids, lang, unit_, klass = [], [], [], []
        for u, (s, lg) in enumerate(zip(seg_ids, langs)):
            ids += s
            lang += [lg] * len(s)
            unit_ += [u + 1] * len(s)
            klass += ["word_start"] * len(s)
        tail_start = len(seg_ids[0]) + len(seg_ids[1])
        out.append(types.SimpleNamespace(
            ids=ids, target_id=target_id,
            annot=dict(lang=lang, unit=unit_, klass=klass, tail_start=tail_start),
            family_id=f"T{i:02d}", formulation="A", K=2, C="C1"))
    return out


@pytest.fixture(scope="module")
def variants(tok):
    return make_variants(tok)


@pytest.fixture(scope="module")
def site(mb):
    return HookSite(mb.n_layers // 2, "block_input")


@pytest.fixture(scope="module")
def collected(mb, variants, site):
    return collect_activations(mb, variants, site, which="segment_last")


# --------------------------------------------------------------------------------------
def test_direction_dataclass_normalises_and_scales(mb, site):
    v = torch.arange(1, mb.d_model + 1, dtype=torch.float32) * 3.0
    d = Direction("x", v, site, {"a": 1})
    assert d.vec.dtype == torch.float32 and d.vec.device.type == "cpu"
    assert torch.isclose(d.vec.norm(), torch.tensor(1.0), atol=1e-6)
    h_ref = torch.randn(mb.d_model, generator=torch.Generator().manual_seed(0)) * 5
    delta = d.scaled_for(h_ref, 0.01)
    assert torch.allclose(delta, make_dose_delta(h_ref, d.vec, 0.01))
    assert torch.isclose(delta.pow(2).mean().sqrt(), 0.01 * h_ref.pow(2).mean().sqrt(), rtol=1e-4)
    with pytest.raises(ValueError):
        Direction("zero", torch.zeros(mb.d_model), site)


def test_entropy_direction_unit_and_orthogonal(mb, sample_ids):
    site = HookSite(mb.n_layers - 1, "block_input")
    pos = sample_ids.shape[1] - 1
    d = entropy_direction(mb, sample_ids, site, pos)
    assert d.name == "ent" and d.site == site and d.vec.shape == (mb.d_model,)
    assert torch.isclose(d.vec.norm(), torch.tensor(1.0), atol=1e-6)
    assert d.meta["entropy"] > 0 and d.meta["grad_norm"] > 0 and np.isnan(d.meta["cos_to_axis"])
    # ARI-style orthogonalisation against a random axis (plus a bit of the gradient so the
    # overlap is not negligible)
    axis = torch.randn(mb.d_model, generator=torch.Generator().manual_seed(7)) + 2.0 * d.vec
    d2 = entropy_direction(mb, sample_ids, site, pos, remove_axis=axis)
    assert torch.isclose(d2.vec.norm(), torch.tensor(1.0), atol=1e-6)
    assert abs(float(d2.vec @ unit(axis))) < 1e-5
    assert d2.meta["removed_axis"] and abs(d2.meta["cos_to_axis"]) > 0.05
    assert d2.meta["removed_fraction"] > 0


def test_load_axis_npz(tmp_path, mb):
    assert load_axis_npz(str(tmp_path / "missing.npz")) is None
    a = np.random.default_rng(0).normal(size=mb.d_model).astype(np.float32)
    p = tmp_path / "axes.npz"
    np.savez(p, other=np.zeros(3), axis=a)
    t = load_axis_npz(str(p))
    assert isinstance(t, torch.Tensor) and t.dtype == torch.float32
    assert np.allclose(t.numpy(), a)
    p2 = tmp_path / "axes2.npz"
    np.savez(p2, foo=a[None, :])            # unknown key, 2-D -> first array, first row
    assert np.allclose(load_axis_npz(str(p2)).numpy(), a)


def test_collect_activations_segments(mb, variants, site, collected):
    acts, labels, meta = collected
    n_var = len(variants)
    assert acts.shape == (3 * n_var, mb.d_model) and acts.dtype == torch.float32
    assert labels == ["E", "D", "E"] * n_var
    assert isinstance(meta, pd.DataFrame)
    for c in ("family", "K", "C", "unit", "position"):
        assert c in meta.columns
    assert list(meta["unit"]) == [1, 2, 3] * n_var
    # segment-final positions: the last token of each segment
    v0 = variants[0]
    pos0 = meta.loc[meta["variant"] == 0, "position"].tolist()
    assert pos0[-1] == len(v0.ids) - 1 and pos0[1] == v0.annot["tail_start"] - 1
    assert bool(meta["is_tail"].iloc[2]) and not bool(meta["is_tail"].iloc[0])
    # all_tokens: one row per token, tail tokens labelled E
    acts_t, labels_t, meta_t = collect_activations(mb, variants[:2], site, which="all_tokens")
    n_tok = sum(len(v.ids) for v in variants[:2])
    assert acts_t.shape == (n_tok, mb.d_model) and len(labels_t) == n_tok
    assert labels_t[:len(variants[0].ids)] == variants[0].annot["lang"]
    with pytest.raises(ValueError):
        collect_activations(mb, variants[:1], site, which="bogus")


def test_diff_of_means_unit_and_denoise(collected):
    acts, labels, _ = collected
    lab = np.asarray(labels)
    D, E = acts[torch.as_tensor(lab == "D")], acts[torch.as_tensor(lab == "E")]
    v0 = diff_of_means(D, E, denoise_pcs=0)
    v1 = diff_of_means(D, E, denoise_pcs=1)
    for v in (v0, v1):
        assert v.shape == (acts.shape[1],) and torch.isclose(v.norm(), torch.tensor(1.0), atol=1e-6)
    plain = D.mean(0) - E.mean(0)
    assert torch.allclose(v0, plain / plain.norm(), atol=1e-6)
    assert float(v0 @ v1) < 1 - 1e-4                  # denoising changes the vector
    # the denoised vector is orthogonal to the top PC of the pooled centred data
    X = torch.cat([D, E]); X = X - X.mean(0)
    pc1 = torch.linalg.svd(X.double(), full_matrices=False).Vh[0].float()
    assert abs(float(v1 @ pc1)) < 1e-4
    # synthetic: class difference is the dominant variance direction -> denoising removes most of it
    g = torch.Generator().manual_seed(1)
    Xd = torch.randn(40, 16, generator=g); Xe = torch.randn(40, 16, generator=g)
    Xd[:, 0] += 6.0
    vs0, vs1 = diff_of_means(Xd, Xe, 0), diff_of_means(Xd, Xe, 1)
    assert abs(float(vs0[0])) > 0.9 and abs(float(vs1[0])) < 0.3


def test_language_axis(mb, variants, site):
    d = language_axis(mb, variants, site)
    assert d.name == "lang_dom" and d.site == site
    assert torch.isclose(d.vec.norm(), torch.tensor(1.0), atol=1e-6)
    assert d.meta["n_D"] == len(variants) and d.meta["n_E"] == 2 * len(variants)
    assert -1 <= d.meta["cos_denoised_plain"] <= 1
    assert torch.isclose(d.meta["vec_plain"].norm(), torch.tensor(1.0), atol=1e-6)
    assert abs(float(d.vec @ d.meta["vec_plain"]) - d.meta["cos_denoised_plain"]) < 1e-5


def test_probe_axis_separable_synthetic():
    g = torch.Generator().manual_seed(0)
    d = 24
    A = torch.randn(60, d, generator=g); B = torch.randn(60, d, generator=g)
    A[:, 3] += 4.0; B[:, 3] -= 4.0                     # separable along feature 3
    A[:, 5] *= 10.0; B[:, 5] *= 10.0                   # a large-variance nuisance feature
    acts = torch.cat([A, B]); labels = ["D"] * 60 + ["E"] * 60
    w, acc = probe_axis(acts, labels, C=1.0)
    assert w.shape == (d,) and torch.isclose(w.norm(), torch.tensor(1.0), atol=1e-6)
    assert acc > 0.8
    # D is +; the separating feature dominates (its weight is divided by its large sd when
    # mapped back from standardised space, so it is not ~1 but clearly the largest component)
    assert float(w[3]) > 0.5 and int(w.abs().argmax()) == 3
    # ignores foreign labels and is deterministic
    w2, acc2 = probe_axis(torch.cat([acts, torch.zeros(3, d)]), labels + ["N"] * 3, C=1.0)
    assert torch.allclose(w, w2, atol=1e-5) and acc2 == acc
    with pytest.raises(ValueError):
        probe_axis(acts, ["E"] * 120)


def test_probe_axis_on_model_activations(mb, variants, site):
    acts, labels, _ = collect_activations(mb, variants, site, which="all_tokens")
    w, acc = probe_axis(acts, labels, C=1.0)
    assert acc > 0.8 and torch.isclose(w.norm(), torch.tensor(1.0), atol=1e-6)


def test_random_directions():
    R = random_directions(32, 5, seed=3)
    assert R.shape == (5, 32) and R.dtype == torch.float32
    assert torch.allclose(R.norm(dim=-1), torch.ones(5), atol=1e-6)
    assert torch.equal(R, random_directions(32, 5, seed=3))
    assert not torch.allclose(R, random_directions(32, 5, seed=4))


def test_frequency_axis_fake_counts(mb):
    V = mb.lm_head.weight.shape[0]
    counts = np.zeros(V, dtype=np.int64)
    top = [13, 15, 253, 2, 1000, 4567]
    for r, i in enumerate(top):
        counts[i] = 1000 - r
    counts[8000:8010] = 1
    d = frequency_axis(mb, counts, top_n=len(top))
    assert d.name == "freq" and torch.isclose(d.vec.norm(), torch.tensor(1.0), atol=1e-6)
    assert d.meta["top_ids"] == top and d.meta["top_n"] == len(top)
    W = mb.lm_head.weight.detach().float()
    expect = W[torch.tensor(top)].mean(0) - W.mean(0)
    assert torch.allclose(d.vec, expect / expect.norm(), atol=1e-5)
    assert 0 < d.meta["top_mass"] <= 1
    # counts shorter than the unembedding (tokenizer length) are handled
    d2 = frequency_axis(mb, counts[:len(mb.tokenizer)], top_n=3)
    assert d2.meta["top_ids"] == top[:3] and d2.meta["n_vocab"] == len(mb.tokenizer)


def test_cosine_table(mb, site):
    R = random_directions(mb.d_model, 3, seed=0)
    dirs = [Direction("a", R[0], site), Direction("b", R[1], site), Direction("c", R[2], site),
            Direction("a", -R[0], site)]
    tab = cosine_table(dirs)
    assert isinstance(tab, pd.DataFrame) and tab.shape == (4, 4)
    assert list(tab.index) == list(tab.columns) and len(set(tab.index)) == 4
    assert np.allclose(tab.values, tab.values.T)
    assert np.allclose(np.diag(tab.values), 1.0)
    assert np.isclose(tab.iloc[0, 3], -1.0)
    assert np.all(np.abs(tab.values) <= 1 + 1e-6)


def test_unembed_projection(mb, site):
    d = Direction("ent", random_directions(mb.d_model, 1, seed=5)[0], site)
    out = unembed_projection(mb, d, top_k=25)
    assert set(out) >= {"promoted", "suppressed"}
    assert len(out["promoted"]) == 25 and len(out["suppressed"]) == 25
    for tok_str, score in out["promoted"] + out["suppressed"]:
        assert isinstance(tok_str, str) and isinstance(score, float)
    ps = [s for _, s in out["promoted"]]; ss = [s for _, s in out["suppressed"]]
    assert ps == sorted(ps, reverse=True) and ss == sorted(ss)
    assert ps[0] > ss[0]
    # explicit formula, raw tensor input, h_rms scaling
    out2 = unembed_projection(mb, d.vec, top_k=5, h_rms=2.0)
    v = d.vec / d.vec.pow(2).mean().sqrt() * 2.0
    scores = mb.lm_head.weight.detach().float() @ (mb.final_ln.weight.detach().float() * v)
    assert torch.allclose(out2["scores"], scores, atol=1e-4)
    assert np.isclose(out2["promoted"][0][1], 2.0 * out["promoted"][0][1], rtol=1e-4)
