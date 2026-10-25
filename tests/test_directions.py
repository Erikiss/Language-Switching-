"""Tests for lsne.directions (CPU, pythia-14m fixture, synthetic E/D stimuli)."""
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite
from lsne.directions import (DENOISE_GATE_COS, Direction, collect_activations, cosine_table,
                             denoise_report, diff_of_means, entropy_direction, frequency_axis,
                             language_axis, load_axis_npz, probe_axis, random_directions,
                             unembed_projection, unit, within_class_pcs)
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


def test_collect_activations_include_tail_false_drops_the_unpaired_tail(mb, variants, site, collected):
    """Round 2, G4 item 4: with include_tail=False only the paired unit segments remain."""
    acts, labels, meta = collected                                  # include_tail=True (default)
    acts_p, labels_p, meta_p = collect_activations(mb, variants, site, which="segment_last", include_tail=False)
    n_var = len(variants)
    assert acts_p.shape == (2 * n_var, mb.d_model) and labels_p == ["E", "D"] * n_var
    assert list(meta_p["unit"]) == [1, 2] * n_var and not meta_p["is_tail"].any()
    assert (meta_p["position"] < meta_p["n_tokens"]).all()
    for v, (_, r) in zip(variants, meta_p[meta_p["unit"] == 2].iterrows()):
        assert int(r["position"]) == v.annot["tail_start"] - 1       # last token before the tail
    # the kept rows are exactly the non-tail rows of the full collection (same activations)
    keep = ~meta["is_tail"].to_numpy()
    assert torch.allclose(acts_p, acts[torch.as_tensor(keep)])
    assert meta_p[["variant", "unit", "position", "lang"]].reset_index(drop=True).equals(
        meta.loc[keep, ["variant", "unit", "position", "lang"]].reset_index(drop=True))
    # all_tokens: positions >= tail_start are dropped
    acts_t, labels_t, meta_t = collect_activations(mb, variants[:2], site, which="all_tokens", include_tail=False)
    n_pre = sum(v.annot["tail_start"] for v in variants[:2])
    assert acts_t.shape == (n_pre, mb.d_model) and len(labels_t) == n_pre and not meta_t["is_tail"].any()
    assert labels_t[:variants[0].annot["tail_start"]] == variants[0].annot["lang"][:variants[0].annot["tail_start"]]
    assert int(meta_t["position"].max()) < max(v.annot["tail_start"] for v in variants[:2])


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
    # the denoised vector is orthogonal to the top PC of the WITHIN-class centred data
    # (D - mean_D stacked on E - mean_E), and the PC is a unit vector of the right size
    V, frac = within_class_pcs(D, E, 1)
    assert V.shape == (acts.shape[1], 1) and frac.shape == (1,) and 0 < float(frac[0]) <= 1
    assert torch.isclose(V[:, 0].norm(), torch.tensor(1.0), atol=1e-5)
    pc1w = V[:, 0]
    assert abs(float(v1 @ pc1w)) < 1e-4
    # exact identity: removing one unit component c = pc1w.v0 from v0 leaves cos = sqrt(1 - c^2)
    c = float(v0 @ pc1w)
    assert abs(float(v0 @ v1) - float(np.sqrt(max(1.0 - c * c, 0.0)))) < 1e-4
    # the within-class PCs are built from the per-class centred data: the same SVD by hand
    Xw = torch.cat([D - D.mean(0), E - E.mean(0)])
    pc1_hand = torch.linalg.svd(Xw.double(), full_matrices=False).Vh[0].float()
    assert abs(abs(float(pc1w @ pc1_hand)) - 1.0) < 1e-5
    # denoise_pcs larger than the rank bound is clipped, never an error
    v_big = diff_of_means(D[:2], E[:2], denoise_pcs=10 ** 6)
    assert torch.isclose(v_big.norm(), torch.tensor(1.0), atol=1e-6)
    assert within_class_pcs(D[:2], E[:2], 10 ** 6)[0].shape[1] == 2        # (2-1) + (2-1)
    with pytest.raises(ValueError):
        diff_of_means(D[:0], E, denoise_pcs=0)


def test_diff_of_means_denoise_keeps_signal_removes_shared_nuisance():
    """DESIGN section 3: denoising projects out the top SHARED component.

    (a) A separable class difference that is the dominant variance direction of the
        POOLED data must SURVIVE denoising (the pooled PC1 would be the class difference
        itself; the within-class PC1 is just noise and nearly orthogonal to it).
    (b) A planted large-variance direction present in BOTH classes (shared nuisance) must
        be removed, and the class signal must come out cleaner than in the plain difference.
    """
    d, n = 16, 40
    # ---- (a) signal survives ----------------------------------------------------------
    g = torch.Generator().manual_seed(1)
    Xd = torch.randn(n, d, generator=g); Xe = torch.randn(n, d, generator=g)
    Xd[:, 0] += 6.0
    vs0, vs1 = diff_of_means(Xd, Xe, 0), diff_of_means(Xd, Xe, 1)
    assert abs(float(vs0[0])) > 0.9 and abs(float(vs1[0])) > 0.9
    mu = Xd.mean(0) - Xe.mean(0)
    assert float(mu @ vs1) / float(mu @ vs0) > 0.9        # separation kept along the axis
    # the pooled PC1 IS the class difference here; the old construction removed the signal
    X = torch.cat([Xd, Xe]); X = X - X.mean(0)
    pc1_pooled = torch.linalg.svd(X.double(), full_matrices=False).Vh[0].float()
    assert abs(float(pc1_pooled[0])) > 0.9
    # ---- (b) planted shared nuisance ---------------------------------------------------
    for seed in (0, 1, 2):
        g = torch.Generator().manual_seed(seed)
        Xd = torch.randn(n, d, generator=g); Xe = torch.randn(n, d, generator=g)
        Xd[:, 0] += 2.0                                              # class signal on axis 0
        u = torch.randn(d, generator=g); u[0] = 0.0; u = unit(u)    # nuisance axis, orthogonal to e0
        Xd = Xd + (10.0 * torch.randn(n, 1, generator=g)) * u       # same large-variance axis
        Xe = Xe + (10.0 * torch.randn(n, 1, generator=g)) * u       # in BOTH classes
        v0, v1 = diff_of_means(Xd, Xe, 0), diff_of_means(Xd, Xe, 1)
        V, frac = within_class_pcs(Xd, Xe, 1)
        assert abs(float(V[:, 0] @ u)) > 0.99 and float(frac[0]) > 0.5   # PC1 = the nuisance
        assert abs(float(v1 @ u)) < 0.05                                 # nuisance removed
        assert abs(float(v1 @ u)) < abs(float(v0 @ u))
        assert abs(float(v1[0])) > 0.85 and abs(float(v1[0])) >= abs(float(v0[0])) - 1e-6
        rep = denoise_report(Xd, Xe, 1, warn=False)
        assert rep["dprime_denoised"] > rep["dprime_plain"]              # cleaner separation
        assert rep["denoise_pcs"] == 1 and rep["cos_pcs_plain"][0] == pytest.approx(abs(float(V[:, 0] @ v0)), abs=1e-5)


def test_denoise_report_gate():
    """The gate flags a denoised axis that lost most of the measured separation."""
    d, n = 16, 40
    # class shift ALONG the shared high-variance axis: denoising must remove it -> gate fails
    g = torch.Generator().manual_seed(1)
    Xd = torch.randn(n, d, generator=g); Xe = torch.randn(n, d, generator=g)
    Xd[:, 5] *= 10.0; Xe[:, 5] *= 10.0; Xd[:, 5] += 5.0
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rep = denoise_report(Xd, Xe, 1)
    assert not rep["gate_ok"] and rep["cos_denoised_plain"] < DENOISE_GATE_COS
    assert rep["separation_kept"] == pytest.approx(rep["cos_denoised_plain"], abs=1e-5)
    assert rep["gate_cos"] == DENOISE_GATE_COS
    assert len(w) == 1 and issubclass(w[0].category, UserWarning) and "cos(denoised, plain)" in str(w[0].message)
    assert np.isfinite(rep["dprime_plain"]) and np.isfinite(rep["dprime_denoised"])
    # warn=False silences it, the numbers are identical
    with warnings.catch_warnings(record=True) as w2:
        warnings.simplefilter("always")
        rep2 = denoise_report(Xd, Xe, 1, warn=False)
    assert len(w2) == 0 and rep2["cos_denoised_plain"] == pytest.approx(rep["cos_denoised_plain"])
    # class shift orthogonal to the shared axis: gate passes, no warning
    g = torch.Generator().manual_seed(2)
    Xd = torch.randn(n, d, generator=g); Xe = torch.randn(n, d, generator=g)
    Xd[:, 5] *= 10.0; Xe[:, 5] *= 10.0; Xd[:, 0] += 5.0
    with warnings.catch_warnings(record=True) as w3:
        warnings.simplefilter("always")
        rep3 = denoise_report(Xd, Xe, 1)
    assert rep3["gate_ok"] and rep3["cos_denoised_plain"] > 0.9 and len(w3) == 0
    assert torch.isclose(rep3["vec_denoised"].norm(), torch.tensor(1.0), atol=1e-6)
    assert torch.isclose(rep3["vec_plain"].norm(), torch.tensor(1.0), atol=1e-6)
    assert 0 < rep3["within_var_removed"] < 1
    # denoise_pcs=0 and the degenerate one-sample-per-class case: plain axis, gate ok, no PCs
    for args in ((Xd, Xe, 0), (Xd[:1], Xe[:1], 1)):
        with warnings.catch_warnings(record=True) as w4:
            warnings.simplefilter("always")
            rep4 = denoise_report(*args)
        assert rep4["denoise_pcs"] == 0 and rep4["gate_ok"] and len(w4) == 0
        assert torch.allclose(rep4["vec_denoised"], rep4["vec_plain"], atol=1e-6)
        assert rep4["cos_pcs_plain"] == [] and rep4["within_var_removed"] == 0.0


def test_language_axis(mb, variants, site, collected):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        d = language_axis(mb, variants, site)
    assert d.name == "lang_dom" and d.site == site
    assert torch.isclose(d.vec.norm(), torch.tensor(1.0), atol=1e-6)
    # DESIGN section 3 / round 2 G4 item 4: only the paired unit segments form the classes;
    # the shared English tail (one more 'E' segment per variant) is excluded by default
    assert d.meta["include_tail"] is False
    assert d.meta["n_D"] == len(variants) and d.meta["n_E"] == len(variants)
    assert -1 <= d.meta["cos_denoised_plain"] <= 1
    assert torch.isclose(d.meta["vec_plain"].norm(), torch.tensor(1.0), atol=1e-6)
    assert abs(float(d.vec @ d.meta["vec_plain"]) - d.meta["cos_denoised_plain"]) < 1e-5
    # the axis is the within-class denoised difference of the tail-free collected activations
    acts, labels, _ = collect_activations(mb, variants, site, which="segment_last", include_tail=False)
    lab = np.asarray(labels)
    D, E = acts[torch.as_tensor(lab == "D")], acts[torch.as_tensor(lab == "E")]
    assert torch.allclose(d.vec, diff_of_means(D, E, denoise_pcs=1), atol=1e-5)
    assert abs(float(d.vec @ within_class_pcs(D, E, 1)[0][:, 0])) < 1e-4
    # include_tail=True reproduces the old construction on the full collection (tail in E)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dt = language_axis(mb, variants, site, include_tail=True)
    acts_f, labels_f, _ = collected
    labf = np.asarray(labels_f)
    Df, Ef = acts_f[torch.as_tensor(labf == "D")], acts_f[torch.as_tensor(labf == "E")]
    assert dt.meta["include_tail"] is True and dt.meta["n_E"] == 2 * len(variants) and dt.meta["n_D"] == len(variants)
    assert torch.allclose(dt.vec, diff_of_means(Df, Ef, denoise_pcs=1), atol=1e-5)
    assert not torch.allclose(dt.meta["vec_plain"], d.meta["vec_plain"], atol=1e-3)   # the tail did move the E mean
    # gate / provenance in meta, consistent with the warning that was (not) emitted
    m = d.meta
    for k in ("separation_kept", "cos_pcs_plain", "within_var_removed", "dprime_plain",
              "dprime_denoised", "gate_ok", "gate_cos", "denoise_pcs", "denoise_pcs_requested"):
        assert k in m, k
    assert m["denoise_pcs"] == 1 and m["denoise_pcs_requested"] == 1 and m["gate_cos"] == DENOISE_GATE_COS
    assert m["separation_kept"] == pytest.approx(m["cos_denoised_plain"], abs=1e-5)
    assert len(m["cos_pcs_plain"]) == 1 and 0 <= m["cos_pcs_plain"][0] <= 1 + 1e-6
    assert 0 < m["within_var_removed"] <= 1
    assert np.isfinite(m["dprime_plain"]) and np.isfinite(m["dprime_denoised"])
    assert m["gate_ok"] == (m["cos_denoised_plain"] >= DENOISE_GATE_COS)
    gate_warnings = [x for x in w if issubclass(x.category, UserWarning) and "cos(denoised, plain)" in str(x.message)]
    assert len(gate_warnings) == (0 if m["gate_ok"] else 1)
    # denoise_pcs=0: the plain axis, gate trivially ok, no warning
    with warnings.catch_warnings(record=True) as w0:
        warnings.simplefilter("always")
        d0 = language_axis(mb, variants, site, denoise_pcs=0)
    assert torch.allclose(d0.vec, d0.meta["vec_plain"], atol=1e-6) and torch.allclose(d0.vec, d.meta["vec_plain"], atol=1e-6)
    assert d0.meta["gate_ok"] and d0.meta["denoise_pcs"] == 0 and d0.meta["cos_denoised_plain"] == pytest.approx(1.0, abs=1e-5)
    assert not [x for x in w0 if "cos(denoised, plain)" in str(x.message)]


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
