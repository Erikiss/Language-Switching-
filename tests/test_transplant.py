"""Tests for lsne.transplant (DESIGN section 4.9): source adapters, alignment with
permutation null, residual substitution, KV-cache planting (prefix and cache API), the
transplant experiment table (which layers are really planted) and the functional transfer
test.  The stand-in validation is non-trivial: a rotated/scaled/shifted copy of the target
and a cross-layer alignment, each judged against the refit-on-permuted-pairs null; the
identical-model case is kept only as a plumbing smoke test, and sign_agreement is checked
to be a measurement (False for an ineffective dose or a dead readout)."""
import contextlib
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite, ModelSpec
from lsne.directions import cosine, diff_of_means, random_directions
from lsne.model import rms, run_with_cache
from lsne import transplant as tp

# Everyday sentences, no proper names (paired: the same sentence in both models).
TEXTS_E = [
    "The children played in the garden all afternoon.",
    "She poured a cup of tea and sat by the window.",
    "The train arrived late because of the heavy snow.",
    "We walked along the river and watched the boats.",
    "He forgot his umbrella and got wet in the rain.",
    "The bakery on the corner sells fresh bread every morning.",
    "My neighbour waters her flowers before breakfast.",
    "The old dog sleeps under the kitchen table.",
    "They painted the fence white last weekend.",
    "A cold wind blew across the empty street.",
    "The library closes early on Fridays.",
    "I need to buy milk, eggs and butter today.",
    "The soup is too hot to eat right now.",
    "Our teacher wrote the answer on the board.",
    "The bus was crowded, so we stood near the door.",
    "A small boat drifted slowly toward the shore.",
    "The lamp in the hallway flickers at night.",
    "She keeps her keys in the left pocket of her coat.",
    "The farmer sold potatoes and onions at the market.",
    "We planted tomatoes behind the house in May.",
    "The clock on the wall is ten minutes fast.",
    "He reads the newspaper while the coffee brews.",
    "The road to the village is closed after the storm.",
    "The cat jumped onto the roof of the shed.",
]
TEXTS_D = [
    "Die Kinder spielten den ganzen Nachmittag im Garten.",
    "Sie goss eine Tasse Tee ein und setzte sich ans Fenster.",
    "Der Zug kam wegen des starken Schnees zu spät an.",
    "Wir gingen am Fluss entlang und sahen den Booten zu.",
    "Er vergaß seinen Schirm und wurde im Regen nass.",
    "Die Bäckerei an der Ecke verkauft jeden Morgen frisches Brot.",
    "Meine Nachbarin gießt ihre Blumen vor dem Frühstück.",
    "Der alte Hund schläft unter dem Küchentisch.",
    "Sie strichen den Zaun letztes Wochenende weiß.",
    "Ein kalter Wind wehte über die leere Straße.",
    "Die Bibliothek schließt freitags früh.",
    "Ich muss heute Milch, Eier und Butter kaufen.",
    "Die Suppe ist im Moment zu heiß zum Essen.",
    "Unsere Lehrerin schrieb die Antwort an die Tafel.",
    "Der Bus war voll, also standen wir neben der Tür.",
    "Ein kleines Boot trieb langsam zum Ufer.",
    "Die Lampe im Flur flackert nachts.",
    "Sie bewahrt ihre Schlüssel in der linken Manteltasche auf.",
    "Der Bauer verkaufte Kartoffeln und Zwiebeln auf dem Markt.",
    "Wir pflanzten im Mai Tomaten hinter dem Haus.",
    "Die Uhr an der Wand geht zehn Minuten vor.",
    "Er liest die Zeitung, während der Kaffee kocht.",
    "Die Straße zum Dorf ist nach dem Sturm gesperrt.",
    "Die Katze sprang auf das Dach des Schuppens.",
]
TEXTS_N = [   # extra neutral sentences for the alignment fit (both languages, short)
    "Water boils at one hundred degrees.", "The meeting starts at nine.", "Apples and pears grow on trees.",
    "The road was closed for repairs.", "Birds sing early in the morning.", "The soup needs more salt.",
    "Der Himmel ist heute grau.", "Das Fenster steht offen.", "Die Katze sitzt auf dem Dach.",
    "Wir essen um zwölf Uhr.", "Das Auto ist rot.", "Der Laden ist geschlossen.",
    "Two plus two is four.", "The door is locked from the inside.", "Rice is cooked with water.",
    "The bridge crosses a narrow river.", "Glass breaks when it falls.", "The bell rings at noon.",
    "Der Brief liegt auf dem Tisch.", "Es regnet seit dem Morgen.", "Die Blumen sind gelb und rot.",
    "Das Brot ist noch warm.", "Der Weg führt durch den Wald.", "Die Tür ist aus Holz.",
]


@pytest.fixture(scope="module")
def src14(mb):
    return tp.HFSourceModel.from_bundle(mb)


@pytest.fixture(scope="module")
def stub():
    return tp.TraceSourceStub(d_model=48, n_layers=3, vocab=512, seed=1)


def _pooled(mb, texts, layer):
    return torch.stack([tp.target_residuals(mb, mb.encode(t), [layer])[layer].mean(0) for t in texts])


def _entropy(logits):
    lp = torch.log_softmax(logits.float(), -1)
    return float(-(lp.exp() * lp).sum())


# ---------------------------------------------------------------------------------------
# source adapters
# ---------------------------------------------------------------------------------------
def test_hf_source_model_implements_protocol(mb, src14, sample_ids):
    assert isinstance(src14, tp.SourceModel)
    L, d = mb.n_layers, mb.d_model
    assert (src14.n_layers, src14.d_model) == (L, d)
    ids = src14.encode(" the reflections are simply")
    assert ids.shape[0] == 1 and ids.dtype == torch.long
    T = sample_ids.shape[1]
    res = src14.residuals(sample_ids, [0, L // 2, L])
    assert set(res) == {0, L // 2, L} and all(h.shape == (T, d) for h in res.values())
    co = run_with_cache(mb, sample_ids, "all_block_inputs")
    assert torch.allclose(res[L // 2], co.resid[f"L{L // 2}.block_input"][0], atol=1e-5)
    assert torch.allclose(res[L], co.resid[f"L{L - 1}.block_output"][0], atol=1e-5)
    assert len(src14.residuals(sample_ids)) == L + 1
    with pytest.raises(ValueError):
        src14.residuals(sample_ids, [L + 1])


def test_trace_stub_is_a_deterministic_source(stub):
    assert isinstance(stub, tp.SourceModel)
    ids = stub.encode("transfer value from a to b")
    assert ids.shape == (1, 6) and ids.dtype == torch.long
    res = stub.residuals(ids)
    assert set(res) == set(range(stub.n_layers + 1))
    assert all(h.shape == (6, stub.d_model) for h in res.values())
    again = tp.TraceSourceStub(d_model=48, n_layers=3, vocab=512, seed=1).residuals(ids)
    assert all(torch.equal(res[l], again[l]) for l in res)
    other = tp.TraceSourceStub(d_model=48, n_layers=3, vocab=512, seed=2).residuals(ids)
    assert not torch.allclose(res[3], other[3])
    # causal context dependence: a different first token changes later states (not layer 0 elsewhere)
    ids2 = ids.clone()
    ids2[0, 0] = (ids2[0, 0] + 1) % stub.vocab
    res2 = stub.residuals(ids2)
    assert torch.equal(res[0][1:], res2[0][1:]) and not torch.allclose(res[3][-1], res2[3][-1])
    with pytest.raises(ValueError):
        stub.residuals(ids, [stub.n_layers + 1])


# ---------------------------------------------------------------------------------------
# alignment
# ---------------------------------------------------------------------------------------
def test_fit_alignment_ridge_recovers_a_linear_map():
    g = torch.Generator().manual_seed(0)
    X = torch.randn(80, 20, generator=g)
    W0 = torch.randn(20, 12, generator=g) / 20 ** 0.5
    b0 = torch.randn(12, generator=g)
    Y = X @ W0 + b0 + 0.02 * torch.randn(80, 12, generator=g)
    a = tp.fit_alignment(X, Y, method="ridge", alpha=1e-3, holdout=0.25, seed=3, layer_src=2, layer_tgt=5)
    assert a.W.shape == (20, 12) and a.b.shape == (12,) and a.n_pairs == 80
    assert (a.layer_src, a.layer_tgt, a.method) == (2, 5, "ridge")
    assert a.r2_train > 0.99 and a.r2_holdout > 0.99
    assert a.r2_permuted < 0.2 and a.r2_holdout > a.r2_permuted
    assert torch.allclose(a.W, W0, atol=0.02) and torch.allclose(a.b, b0, atol=0.05)
    assert a.meta["n_holdout"] == 20 and a.meta["n_train"] == 60
    # transfer helpers
    H = tp.transfer_states(a, X[:5])
    assert H.shape == (5, 12) and torch.allclose(H, Y[:5], atol=0.1)
    v = tp.transfer_direction(a, torch.ones(20))
    assert v.shape == (12,) and abs(float(v.norm()) - 1.0) < 1e-5
    assert cosine(v, W0.T @ torch.ones(20)) > 0.99
    with pytest.raises(ValueError):
        tp.fit_alignment(X, Y[:-1])
    with pytest.raises(ValueError):
        tp.transfer_states(a, torch.zeros(3, 7))


def test_fit_alignment_on_paired_model_states(mb):
    """Same model, layer 1 -> layer L//2: paired states of 40 short sentences."""
    texts = TEXTS_E[:20] + TEXTS_D[:20]
    L = mb.n_layers
    S, T = _pooled(mb, texts, 1), _pooled(mb, texts, L // 2)
    for method in ("ridge", "procrustes"):
        a = tp.fit_alignment(S, T, method=method, alpha=0.1, seed=0, layer_src=1, layer_tgt=L // 2)
        assert a.n_pairs == 40 and a.W.shape == (mb.d_model, mb.d_model)
        assert np.isfinite([a.r2_train, a.r2_holdout, a.r2_permuted]).all()
        assert a.r2_holdout > a.r2_permuted and a.gap > 0.3
    # the permutation null on the hold-out is seeded and reproducible
    a1 = tp.fit_alignment(S, T, seed=4)
    a2 = tp.fit_alignment(S, T, seed=4)
    assert a1.r2_permuted == a2.r2_permuted and a1.meta["idx_holdout"] == a2.meta["idx_holdout"]


def test_procrustes_is_orthogonal_in_the_pca_bases():
    # anisotropic source: 18 strong directions carry the signal (PCA-truncated Procrustes
    # can only map what the top-r principal directions of each side contain)
    g = torch.Generator().manual_seed(1)
    X = torch.randn(50, 30, generator=g) * torch.cat([torch.full((18,), 3.0), torch.full((12,), 0.1)])
    Q, _ = torch.linalg.qr(torch.randn(18, 18, generator=g))
    Y = 2.0 * X[:, :18] @ Q + 0.05 * torch.randn(50, 18, generator=g)
    a = tp.fit_alignment(X, Y, method="procrustes", holdout=0.2, seed=0)
    n_train = a.meta["n_train"]
    assert a.rank == min(30, 18, n_train - 1) == 18
    P_S, P_T, R, s = a.meta["P_S"], a.meta["P_T"], a.meta["R"], a.meta["scale"]
    assert P_S.shape == (30, 18) and P_T.shape == (18, 18) and R.shape == (18, 18)
    eye = torch.eye(18)
    assert torch.allclose(P_S.T @ P_S, eye, atol=1e-4) and torch.allclose(P_T.T @ P_T, eye, atol=1e-4)
    assert torch.allclose(R.T @ R, eye, atol=1e-4)
    assert torch.allclose(a.W, s * P_S @ R @ P_T.T, atol=1e-5)
    # W expressed in the PCA bases is s times an orthogonal matrix
    M = P_S.T @ a.W @ P_T / s
    assert torch.allclose(M.T @ M, eye, atol=1e-4)
    assert s > 1.0 and a.r2_holdout > 0.95 and a.r2_holdout > a.r2_permuted
    a8 = tp.fit_alignment(X, Y, method="procrustes", rank=8, seed=0)
    assert a8.rank == 8 and a8.meta["R"].shape == (8, 8)
    sv = torch.linalg.svdvals(a8.W.double())
    assert sv[7] > 1e-3 * sv[0] and sv[8] < 1e-5 * sv[0]          # numerically rank 8


def test_permute_alignment_control():
    g = torch.Generator().manual_seed(2)
    X = torch.randn(40, 10, generator=g)
    Y = torch.randn(40, 6, generator=g)
    a = tp.fit_alignment(X, Y, alpha=0.5)
    p = tp.permute_alignment(a, seed=0)
    assert p.W.shape == a.W.shape and not torch.equal(p.W, a.W)
    assert torch.allclose(p.W.norm(), a.W.norm()) and p.method.endswith("+permuted")
    assert torch.allclose(p.W.sum(1), a.W.sum(1))        # row sums survive a column permutation


# ---------------------------------------------------------------------------------------
# injection A: residual substitution
# ---------------------------------------------------------------------------------------
def test_substitute_residual_identity_control(mb, sample_ids):
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    co = run_with_cache(mb, sample_ids, [site])
    T = sample_ids.shape[1]
    pos = [2, 3, T - 1]
    own = co.resid[site.key][0, pos]
    lo, info = tp.substitute_residual(mb, sample_ids, site, pos, own, return_info=True)
    assert lo.shape == (mb.vocab,)
    assert torch.allclose(lo, co.logits[0, -1], atol=1e-4)
    assert info["audit"]["written_norm"] < 1e-4 and info["n_calls"] == 1 and info["positions"] == pos
    assert torch.allclose(info["logits"], co.logits[0], atol=1e-4)
    g = torch.Generator().manual_seed(0)
    lo2, info2 = tp.substitute_residual(mb, sample_ids, site, pos, own + 0.5 * torch.randn(3, mb.d_model, generator=g),
                                        return_info=True)
    assert not torch.allclose(lo2, co.logits[0, -1], atol=1e-3) and info2["audit"]["written_norm"] > 0
    # negative positions count from the end; wrong shapes are rejected
    assert torch.allclose(tp.substitute_residual(mb, sample_ids, site, [-1], own[-1:]), co.logits[0, -1], atol=1e-4)
    with pytest.raises(ValueError):
        tp.substitute_residual(mb, sample_ids, site, [0, 1], own)


# ---------------------------------------------------------------------------------------
# injection B: KV-cache planting
# ---------------------------------------------------------------------------------------
def test_plant_kv_prefix_n0_equals_plain_forward(mb, sample_ids):
    plain = run_with_cache(mb, sample_ids, []).logits[0]
    res = tp.plant_kv_prefix(mb, sample_ids, {})
    assert res["n_prefix"] == 0 and res["logits"].shape == plain.shape
    assert torch.equal(res["logits"], plain) and torch.equal(res["logits_last"], plain[-1])
    assert torch.equal(tp.plant_kv_cache(mb, sample_ids, {})["logits"], plain)


def test_plant_kv_prefix_identity_reproduces_real_prefix(mb, sample_ids):
    """Planting the target's own block inputs of a real prefix at every layer makes the
    placeholder prefix indistinguishable from the real one for the following tokens."""
    L = mb.n_layers
    prefix = mb.encode("The committee met on Tuesday and")
    n = prefix.shape[1]
    full = torch.cat([prefix, sample_ids], 1)
    own = tp.target_residuals(mb, full)
    ref = run_with_cache(mb, full, []).logits[0, n:]
    res = tp.plant_kv_prefix(mb, sample_ids, {l: own[l][:n] for l in range(L + 1)}, placeholder_id=0)
    assert res["n_prefix"] == n and res["layers"] == list(range(L + 1))
    assert torch.allclose(res["logits"], ref, atol=1e-4)
    assert all(res["audits"][l]["n_calls"] == 1 for l in range(L + 1))
    # a prefix of placeholders without planting is a different context
    bare = run_with_cache(mb, torch.cat([torch.zeros(1, n, dtype=torch.long), sample_ids], 1), []).logits[0, n:]
    assert not torch.allclose(bare, ref, atol=1e-2)
    # planting only the middle layer leaves the other layers' prefix K/V wrong
    part = tp.plant_kv_prefix(mb, sample_ids, {L // 2: own[L // 2][:n]})
    assert part["layers"] == [L // 2] and not torch.allclose(part["logits"], ref, atol=1e-2)


def _realistic_prefix(mb, n, seed, scale=0.1):
    """Prefix states at every layer with the model's own scale: the target's residuals of
    another sentence plus ``scale`` relative-RMS noise (random states of unit scale would
    be far out of distribution at the early layers, whose residual RMS is ~0.05)."""
    own = tp.target_residuals(mb, mb.encode("Die Katze sprang auf das Dach des Schuppens und"))
    g = torch.Generator().manual_seed(seed)
    return {l: own[l][:n] + scale * rms(own[l][:n])[:, None] * torch.randn(n, mb.d_model, generator=g)
            for l in range(mb.n_layers + 1)}


def test_plant_kv_prefix_replaces_every_listed_layer(mb, sample_ids):
    """Each listed layer's prefix block input is overwritten independently: dropping any
    single layer below L changes the real tokens' logits; layer L (last block output)
    is never attended to and changes nothing."""
    L = mb.n_layers
    n = 4
    H = _realistic_prefix(mb, n, seed=0)
    full = tp.plant_kv_prefix(mb, sample_ids, H)
    plain = tp.plant_kv_prefix(mb, sample_ids, {})["logits"]
    assert full["logits"].shape == plain.shape and not torch.allclose(full["logits"], plain, atol=1e-2)
    for l in range(L + 1):
        assert full["audits"][l]["n_calls"] == 1 and full["audits"][l]["written_norm"] > 0
        keep = [k for k in range(L + 1) if k != l]
        drop = tp.plant_kv_prefix(mb, sample_ids, H, layers=keep)
        assert drop["layers"] == keep
        if l < L:
            assert not torch.allclose(drop["logits"], full["logits"], atol=1e-3), f"layer {l} had no effect"
        else:
            assert torch.allclose(drop["logits"], full["logits"], atol=1e-5)
    with pytest.raises(ValueError):
        tp.plant_kv_prefix(mb, sample_ids, {0: H[0], 1: H[1][:2]})
    with pytest.raises(ValueError):
        tp.plant_kv_prefix(mb, sample_ids, {L + 1: H[0]})


def test_plant_kv_cache_agrees_with_prefix(mb, sample_ids):
    L = mb.n_layers
    n = 6
    H = {l: h for l, h in _realistic_prefix(mb, n, seed=1).items() if l < L}
    a = tp.plant_kv_prefix(mb, sample_ids, H, placeholder_id=3)
    b = tp.plant_kv_cache(mb, sample_ids, H, placeholder_id=3, strict=True)
    assert b["fallback"] is False and b["n_prefix"] == n and b["cache_len"] == n + sample_ids.shape[1]
    assert torch.allclose(a["logits"], b["logits"], atol=1e-3)
    assert torch.allclose(a["logits_last"], b["logits_last"], atol=1e-3)
    # Planting a subset of layers: the un-planted prefix layers are now computed in a
    # length-n forward (cache) vs. inside the length-(n+T) forward (prefix).  That differs
    # by the cache API's own fp32 noise, which on CPU is ~2e-3 in the logits even for a
    # plain, uninjected two-pass forward (final LayerNorm amplifies 1e-5 hidden diffs).
    pre = mb.encode("The committee met on Tuesday and")
    with torch.no_grad():
        one = mb.model(input_ids=torch.cat([pre, sample_ids], 1), use_cache=False).logits[0, pre.shape[1]:]
        o1 = mb.model(input_ids=pre, use_cache=True)
        two = mb.model(input_ids=sample_ids, past_key_values=o1.past_key_values, use_cache=True).logits[0]
    noise = float((one - two).abs().max())
    sub_c = tp.plant_kv_cache(mb, sample_ids, H, layers=[0, 2])
    sub_p = tp.plant_kv_prefix(mb, sample_ids, H, layers=[0, 2])
    assert sub_c["layers"] == [0, 2]
    assert torch.allclose(sub_c["logits"], sub_p["logits"], atol=max(1e-3, 5 * noise))
    assert float((sub_c["logits"] - sub_p["logits"]).abs().max()) < 0.05


# ---------------------------------------------------------------------------------------
# layer matching / experiment table (stand-in source)
# ---------------------------------------------------------------------------------------
def test_layer_matching_returns_table(stub, mb):
    pairs = [(t, t) for t in TEXTS_E[:8] + TEXTS_D[:8]]
    df = tp.layer_matching(stub, mb, pairs, layers_src=[0, stub.n_layers], layers_tgt=[1, mb.n_layers // 2],
                           holdout=0.25, alpha=0.1)
    assert isinstance(df, pd.DataFrame) and len(df) == 4
    assert {"layer_src", "layer_tgt", "r2_holdout", "r2_permuted", "gap"} <= set(df.columns)
    assert set(df.layer_src) == {0, stub.n_layers} and set(df.layer_tgt) == {1, mb.n_layers // 2}
    assert np.allclose(df.gap, df.r2_holdout - df.r2_permuted) and (df.n_pairs == 16).all()
    assert np.isfinite(df[["r2_train", "r2_holdout", "r2_permuted"]].to_numpy()).all()


def test_layer_alignments_one_map_per_target_layer(stub, mb):
    """layer_alignments gives the {layer_tgt: Alignment} dict injection B needs (all L+1
    target layers by default) and picks, per target layer, the source layer with the
    largest gap of the layer_matching table."""
    L = mb.n_layers
    pairs = [(t, t) for t in TEXTS_E[:8] + TEXTS_D[:8]]
    kw = dict(holdout=0.25, alpha=0.1)
    df = tp.layer_matching(stub, mb, pairs, layers_src=[0, stub.n_layers], layers_tgt=[1, L // 2], **kw)
    al = tp.layer_alignments(stub, mb, pairs, layers_src=[0, stub.n_layers], layers_tgt=[1, L // 2], **kw)
    assert set(al) == {1, L // 2}
    for lt, a in al.items():
        assert isinstance(a, tp.Alignment) and a.layer_tgt == lt and a.W.shape == (stub.d_model, mb.d_model)
        best = df[df.layer_tgt == lt].sort_values("gap", ascending=False).iloc[0]
        assert a.layer_src == best.layer_src and a.gap == pytest.approx(best.gap)
    full = tp.layer_alignments(stub, mb, pairs, layers_src=[stub.n_layers], alpha=0.1)
    assert sorted(full) == list(range(L + 1)) and all(a.layer_src == stub.n_layers for a in full.values())


def _spy_replacements(monkeypatch, entered):
    """Record the (site key, positions, rows) specs every replacements() context enters."""
    orig = tp.replacements

    @contextlib.contextmanager
    def spy(mb_, specs):
        entered.append([(s.key, list(p), int(H.shape[0])) for s, p, H in specs])
        with orig(mb_, specs) as reps:
            yield reps
    monkeypatch.setattr(tp, "replacements", spy)


def test_transplant_experiment_smoke(stub, mb, monkeypatch):
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    pairs = [(t, t) for t in TEXTS_E[:8] + TEXTS_D[:8]]
    S, T = tp.paired_states(stub, mb, pairs, [stub.n_layers], [L // 2])
    align = tp.fit_alignment(S[stub.n_layers], T[L // 2], alpha=0.1, layer_src=stub.n_layers, layer_tgt=L // 2)
    v_src = random_directions(stub.d_model, 1, 0)[0]
    native = random_directions(mb.d_model, 1, 1)[0]
    prompt = "The weather was"
    entered = []
    _spy_replacements(monkeypatch, entered)
    # a single Alignment can only plant its own layer: the selected-layers variant, announced loudly
    with pytest.warns(UserWarning, match="single Alignment plants the KV prefix only at target layer"):
        df = tp.transplant_experiment(mb, stub, align, v_src, [(prompt, " cold")], site, eps=1.0, n_prefix=2,
                                      gen_kw=dict(n_new=4, n_samples=1), native_axis=native, seed=0)
    assert len(df) == 3 * 4
    assert set(df["mode"]) == {"residual", "kv", "both"}
    assert set(df["control"]) == {"none", "random", "identity", "permuted_alignment"}
    cols = {"entropy", "logp_target", "fD", "n_prefix", "n_substituted", "cos_native", "r2_holdout", "r2_permuted",
            "written_norm", "kv_layers", "n_kv_layers"}
    assert cols <= set(df.columns)
    assert np.isfinite(df[["entropy", "logp_target", "fD", "written_norm"]].to_numpy()).all()
    assert np.allclose(df.cos_native, cosine(tp.transfer_direction(align, v_src), native))
    assert (df.loc[df["mode"] == "residual", "n_prefix"] == 0).all()
    assert (df.loc[df["mode"] != "residual", "n_prefix"] == 2).all()
    assert (df.loc[df["mode"] == "residual", "kv_layers"] == "").all() and (df.loc[df["mode"] == "residual", "n_kv_layers"] == 0).all()
    assert (df.loc[df["mode"] != "residual", "kv_layers"] == str(L // 2)).all()
    assert (df.loc[df["mode"] != "residual", "n_kv_layers"] == 1).all()
    T_src = stub.encode(prompt).shape[1]
    assert (df.loc[df["mode"] != "kv", "n_substituted"] == min(T_src, mb.encode(prompt).shape[1])).all()
    # identity control with residual substitution = the plain forward
    plain = run_with_cache(mb, mb.encode(prompt), []).logits[0, -1]
    row = df[(df["mode"] == "residual") & (df.control == "identity")].iloc[0]
    assert row.entropy == pytest.approx(_entropy(plain), abs=1e-4) and row.written_norm == 0.0
    # the transplant arm actually rewrites the residual
    row2 = df[(df["mode"] == "residual") & (df.control == "none")].iloc[0]
    assert row2.written_norm > 0 and row2.entropy != pytest.approx(_entropy(plain), abs=1e-6)
    # what was really planted: one spec per replacements() call (forward + generation) per row
    kv_specs = [e for e in entered if any(p == [0, 1] for _, p, _ in e)]
    assert len(kv_specs) == 2 * 4 * 2                                  # (kv, both) x 4 arms x (forward, generate)
    assert all([(k, p) for k, p, _ in e if p == [0, 1]] == [(site.key, [0, 1])] for e in kv_specs)


def test_transplant_experiment_plants_every_layer_with_layer_alignments(stub, mb, monkeypatch):
    """DESIGN 4.9 injection B: with a {layer_tgt: Alignment} dict over 0..L the KV prefix
    is planted at every residual layer (verified on the hooks that are actually entered),
    the kv_layers column records it and no single-layer warning is raised."""
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    pairs = [(t, t) for t in TEXTS_E[:8] + TEXTS_D[:8]]
    aligns = tp.layer_alignments(stub, mb, pairs, layers_src=[stub.n_layers], alpha=0.1)
    v_src = random_directions(stub.d_model, 1, 0)[0]
    prompt = "The weather was"
    n = 2
    entered = []
    _spy_replacements(monkeypatch, entered)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        df = tp.transplant_experiment(mb, stub, aligns, v_src, [(prompt, " cold")], site, eps=1.0, n_prefix=n,
                                      gen_kw=dict(n_new=4, n_samples=1), controls=("identity",), seed=0)
    all_layers = ",".join(str(l) for l in range(L + 1))
    assert (df.loc[df["mode"] != "residual", "kv_layers"] == all_layers).all()
    assert (df.loc[df["mode"] != "residual", "n_kv_layers"] == L + 1).all()
    assert (df.loc[df["mode"] == "residual", "kv_layers"] == "").all()
    expect_sites = [HookSite(l, "block_input").key for l in range(L)] + [HookSite(L - 1, "block_output").key]
    T = mb.encode(prompt).shape[1]
    k = min(stub.encode(prompt).shape[1], T)
    # every replacements() call of a kv/both row entered one prefix spec per layer 0..L ...
    kv_calls = [e for e in entered if any(p == list(range(n)) for _, p, _ in e)]
    both_calls = [e for e in kv_calls if any(p != list(range(n)) for _, p, _ in e)]
    assert len(kv_calls) == 2 * 2 * 2 and len(both_calls) == 2 * 2      # (kv, both) x 2 arms x (forward, generate)
    for e in kv_calls:
        prefix = [(key, p, r) for key, p, r in e if p == list(range(n))]
        assert [key for key, _, _ in prefix] == expect_sites and all(r == n for _, _, r in prefix)
    # ... and the "both" rows additionally substitute the last k real positions at site
    for e in both_calls:
        subst = [(key, p, r) for key, p, r in e if p != list(range(n))]
        assert subst == [(site.key, [n + T - k + j for j in range(k)], k)]
    # the identity arm plants the target's own states: logits finite, prefix of n rows at every layer
    assert np.isfinite(df[["entropy", "fD", "written_norm"]].to_numpy()).all()
    with pytest.raises(ValueError):
        tp.transplant_experiment(mb, stub, {L + 1: aligns[L]}, v_src, [prompt], site, eps=1.0, modes=("kv",))


# ---------------------------------------------------------------------------------------
# functional transfer test (the machinery is validated with the language axis)
# ---------------------------------------------------------------------------------------
def test_functional_transfer_identical_model_is_a_plumbing_smoke_test(mb, src14):
    """pythia-14m as source AND target at the same layer.  This is a *plumbing* check only:
    source and target states are the same tensors, so v_src == v_native, the Procrustes
    map is a symmetric projector onto the training PCs (blind to a W/W^T mix-up) and
    cos > 0.9 / r2 > 0.95 / sign agreement hold by construction.  The informative
    self-consistency tests are the rotated-source and cross-layer tests below."""
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    r = tp.functional_transfer_test(mb, src14, TEXTS_E, TEXTS_D, L // 2, site, texts_neutral=TEXTS_N,
                                    n_gen=16, n_samples=4, seed=0)
    assert r["n_pairs"] == len(TEXTS_E) + len(TEXTS_D) + len(TEXTS_N)
    assert r["v_transferred"].shape == (mb.d_model,) and r["v_src"].shape == (mb.d_model,)
    assert isinstance(r["alignment"], tp.Alignment) and isinstance(r["alignment_null"], tp.Alignment)
    assert r["alignment_null"].method.endswith("+null")
    assert cosine(r["v_src"], r["v_native"]) == pytest.approx(1.0, abs=1e-5)     # the tautology, made explicit
    W = r["alignment"].W.double()
    assert float((W - W.T).abs().max()) < 1e-5 * float(W.abs().max())           # symmetric projector
    for k in ("cos_transferred_vs_native", "cos_null_vs_native", "cos_null_std", "cos_null_max", "z_cos_vs_null",
              "delta_fD_transferred", "delta_fD_native", "r2_holdout", "r2_holdout_null", "r2_holdout_null_max",
              "r2_permuted", "fD_baseline"):
        assert np.isfinite(r[k])
    assert r["n_null"] == 10 and r["cos_null_max"] >= r["cos_null_vs_native"]
    assert r["readout_alive"] is True and set(r["n_ED"]) == {"baseline", "transferred", "native"}
    assert all(n > 0 for n in r["n_ED"].values())


def test_functional_transfer_undoes_a_known_rotation(mb, src14):
    """Non-trivial self-consistency: the source is pythia-14m seen through a known affine
    isometry ``3 H Q + 0.5``.  The alignment must undo Q (so W is not a projector any
    more and the W-vs-W^T convention is exercised), the transferred language axis must
    coincide with the native one far better than the permuted-pairs null does, and it
    must steer the German fraction in the same direction as the native axis."""
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    rot = tp.RotatedSourceModel(src14, seed=0, scale=3.0, shift=0.5)
    assert isinstance(rot, tp.SourceModel) and (rot.d_model, rot.n_layers) == (mb.d_model, L)
    ids = rot.encode(TEXTS_E[0])
    base = src14.residuals(ids, [L // 2])[L // 2]
    assert torch.allclose(rot.residuals(ids, [L // 2])[L // 2], 3.0 * base @ rot.Q + 0.5, atol=1e-5)
    assert torch.allclose(rot.Q.T @ rot.Q, torch.eye(mb.d_model), atol=1e-4)
    with pytest.raises(ValueError):
        tp.RotatedSourceModel(src14, Q=torch.ones(mb.d_model, mb.d_model))
    for fit_kw in (None, dict(method="ridge", alpha=0.01)):
        r = tp.functional_transfer_test(mb, rot, TEXTS_E, TEXTS_D, L // 2, site, texts_neutral=TEXTS_N,
                                        n_gen=16, n_samples=4, seed=0, fit_kw=fit_kw)
        a = r["alignment"]
        # the source axis is the rotated native axis (Q v_src = v_native); W is not symmetric
        assert cosine(rot.Q @ r["v_src"], r["v_native"]) > 0.999
        W = a.W.double()
        assert float((W - W.T).abs().max()) > 0.1 * float(W.abs().max())
        # transfer undoes the rotation; the transposed convention would not (sensitivity check)
        assert r["cos_transferred_vs_native"] > 0.9
        assert cosine(a.W @ r["v_src"], r["v_native"]) < 0.5
        # ... and beats the refit-on-permuted-pairs null (mean over n_null derangements and
        # its best draw: a single draw scatters with std ~0.3 since half the permuted pairs
        # stay within one language) by a wide margin
        assert r["n_null"] == 10 and 0 < r["cos_null_std"] < 0.6
        assert abs(r["cos_null_vs_native"]) < 0.5
        assert r["cos_transferred_vs_native"] - r["cos_null_vs_native"] > 0.5
        assert r["cos_transferred_vs_native"] > r["cos_null_max"] + 0.2
        assert np.isfinite(r["z_cos_vs_null"]) and r["z_cos_vs_null"] > 1.0   # bounded cosine: z is descriptive only
        assert r["r2_holdout"] > 0.95 and r["r2_holdout"] - r["r2_holdout_null"] > 0.5
        assert r["r2_holdout"] > r["r2_holdout_null_max"] + 0.5
        assert r["r2_holdout"] > r["r2_permuted"] and r["r2_permuted"] < 0.5
        # functional: both axes push towards German, the flag is a live measurement
        assert r["readout_alive"] is True and r["sign_agreement"] is True
        assert r["delta_fD_native"] > 0 and r["delta_fD_transferred"] > 0


def test_functional_transfer_across_layers(mb, src14):
    """Source layer L//2 + 1 -> target layer L//2 of the same model: the map is a genuine
    (non-projector) layer-to-layer alignment, and the language axis of the deeper layer
    still transfers functionally and beats the null."""
    L = mb.n_layers
    if L // 2 + 1 > L:
        pytest.skip("model too shallow for a cross-layer alignment")
    site = HookSite(L // 2, "block_input")
    r = tp.functional_transfer_test(mb, src14, TEXTS_E, TEXTS_D, L // 2 + 1, site, texts_neutral=TEXTS_N,
                                    n_gen=16, n_samples=4, seed=0)
    assert (r["alignment"].layer_src, r["alignment"].layer_tgt) == (L // 2 + 1, L // 2)
    W = r["alignment"].W.double()
    assert float((W - W.T).abs().max()) > 0.1 * float(W.abs().max())
    assert cosine(r["v_src"], r["v_native"]) < 0.999                     # not the same axis any more
    assert r["cos_transferred_vs_native"] > 0.9
    assert r["cos_transferred_vs_native"] - r["cos_null_vs_native"] > 0.5
    assert r["cos_transferred_vs_native"] > r["cos_null_max"] + 0.2
    assert r["r2_holdout"] > 0.8 and r["r2_holdout"] - r["r2_holdout_null"] > 0.5
    assert r["r2_holdout"] > r["r2_holdout_null_max"] + 0.5
    assert r["r2_holdout"] > r["r2_permuted"]
    assert r["readout_alive"] is True and r["sign_agreement"] is True
    assert r["delta_fD_native"] > 0 and r["delta_fD_transferred"] > 0


def test_sign_agreement_is_a_measurement(mb, src14, monkeypatch):
    """sign_agreement must not be vacuously True: an ineffective dose (eps=0, both deltas
    0), a dead readout (no E/D-labelled tokens) or deltas below min_delta give False."""
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    kw = dict(texts_neutral=TEXTS_N[:12], n_gen=12, n_samples=2, seed=1)
    # eps = 0: steering does nothing, both deltas are exactly 0 -> no agreement to report
    r0 = tp.functional_transfer_test(mb, src14, TEXTS_E[:12], TEXTS_D[:12], L // 2, site, eps=0.0, **kw)
    assert r0["delta_fD_transferred"] == 0.0 and r0["delta_fD_native"] == 0.0
    assert r0["readout_alive"] is True and all(n > 0 for n in r0["n_ED"].values())
    assert r0["sign_agreement"] is False
    # deltas below min_delta do not count either (the deltas themselves are still reported)
    r1 = tp.functional_transfer_test(mb, src14, TEXTS_E[:12], TEXTS_D[:12], L // 2, site, eps=1.0, min_delta=1.5, **kw)
    assert r1["min_delta"] == 1.5 and r1["sign_agreement"] is False
    assert abs(r1["delta_fD_native"]) <= 1.0 and np.isfinite(r1["delta_fD_transferred"])
    # dead readout: no E/D label at all -> flagged, warned, never "agreeing"
    monkeypatch.setattr(tp, "block_fraction", lambda labels, lang="D": (0.0, 0))
    with pytest.warns(UserWarning, match="readout is dead"):
        r2 = tp.functional_transfer_test(mb, src14, TEXTS_E[:12], TEXTS_D[:12], L // 2, site, eps=1.0, **kw)
    assert r2["readout_alive"] is False and all(n == 0 for n in r2["n_ED"].values())
    assert r2["delta_fD_transferred"] == 0.0 and r2["delta_fD_native"] == 0.0
    assert r2["sign_agreement"] is False


def test_functional_transfer_with_given_native_axis(mb, src14):
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    S_E, S_D = _pooled(mb, TEXTS_E[:12], L // 2), _pooled(mb, TEXTS_D[:12], L // 2)
    native = diff_of_means(S_D, S_E)
    r = tp.functional_transfer_test(mb, src14, TEXTS_E[:12], TEXTS_D[:12], L // 2, site, native_axis=-native,
                                    texts_neutral=TEXTS_N[:12], n_gen=12, n_samples=2, seed=1, fit_kw=dict(alpha=0.01))
    # a sign-flipped native axis: cosine negative, signs disagree (the native arm pushes to English)
    assert r["cos_transferred_vs_native"] < -0.9
    assert torch.allclose(r["v_native"], -native)
    assert r["delta_fD_transferred"] > 0 and r["delta_fD_native"] <= 0 and r["sign_agreement"] is False


@pytest.mark.slow
@pytest.mark.network
def test_cross_size_pythia70m_to_14m(mb):
    """pythia-70m (d=512) -> pythia-14m (d=128): only alignment above the permutation null
    and finite outputs are asserted (a real cross-model transfer is not guaranteed here)."""
    src = tp.HFSourceModel(ModelSpec(name="EleutherAI/pythia-70m", revision="step98000", dtype="float32", device="cpu"))
    assert isinstance(src, tp.SourceModel) and src.d_model != mb.d_model
    L_s, L_t = src.n_layers, mb.n_layers
    texts = TEXTS_E[:20] + TEXTS_D[:20]
    S, T = tp.paired_states(src, mb, [(t, t) for t in texts], [L_s // 2], [L_t // 2])
    a = tp.fit_alignment(S[L_s // 2], T[L_t // 2], method="procrustes", layer_src=L_s // 2, layer_tgt=L_t // 2)
    assert a.W.shape == (src.d_model, mb.d_model) and a.n_pairs == 40
    assert a.r2_holdout > a.r2_permuted
    r = tp.functional_transfer_test(mb, src, TEXTS_E, TEXTS_D, L_s // 2, HookSite(L_t // 2, "block_input"),
                                    texts_neutral=TEXTS_N, n_gen=16, n_samples=4, seed=0)
    for k in ("cos_transferred_vs_native", "delta_fD_transferred", "delta_fD_native", "r2_holdout", "r2_permuted"):
        assert np.isfinite(r[k])
    assert r["r2_holdout"] > r["r2_permuted"]
    assert torch.isfinite(tp.transfer_states(a, S[L_s // 2][:3])).all()


# ---------------------------------------------------------------------------------------
# prompt pairs: the source encodes one item, the target another (trace source, DESIGN 4.9)
# ---------------------------------------------------------------------------------------
class _SpySource:
    """Wraps a SourceModel and records every item it is asked to encode."""

    def __init__(self, base):
        self.base, self.seen = base, []
        self.name, self.d_model, self.n_layers = "spy:" + base.name, base.d_model, base.n_layers

    def encode(self, item):
        self.seen.append(item)
        return self.base.encode(item)

    def residuals(self, ids, layers=None):
        return self.base.residuals(ids, layers)


def test_as_prompt_pair_normalisation():
    pp = tp.as_prompt_pair("The weather was")
    assert (pp.source, pp.text, pp.target, pp.paired) == ("The weather was", "The weather was", None, False)
    pp = tp.as_prompt_pair(("The weather was", " cold"))                  # the old (text, target) form
    assert (pp.source, pp.text, pp.target, pp.paired) == ("The weather was", "The weather was", " cold", False)
    ids = [5, 6, 7, 8]
    pp = tp.as_prompt_pair((ids, "The weather was", 123))                 # (source_item, text, target)
    assert pp.source == ids and pp.text == "The weather was" and pp.target == 123 and pp.paired
    row = {"tx_hash": "0xabc", "tokens": ["[TX]", "[/TX]"], "text": "Account 0x1 sends 1 ETH."}
    pp = tp.as_prompt_pair(row)                                            # a shard row is its own source item
    assert pp.source is row and pp.text == row["text"] and pp.name == "0xabc" and pp.paired
    assert tp.as_prompt_pair({"source": ids, "text": "x", "target_id": 3}).target == 3
    given = tp.PromptPair(source=ids, text="hi", target=1, ids=[1, 2])
    assert tp.as_prompt_pair(given) is given and tp.as_prompt_pair(given, target=9).target == 9
    assert tp.source_name(row) == "0xabc" and tp.source_name(ids) == "ids[4]" and tp.source_name("a b") == "a b"
    for bad in ({"tokens": ["[TX]"]}, (ids,), (ids, 5), 3.5):
        with pytest.raises((ValueError, TypeError)):
            tp.as_prompt_pair(bad)


def test_plain_string_prompts_unchanged(stub, mb):
    """Plain strings and (text, target) tuples keep the old behaviour: the same text in both
    models, identical numbers to the explicit same-text PromptPair."""
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    pairs = [(t, t) for t in TEXTS_E[:8] + TEXTS_D[:8]]
    aligns = tp.layer_alignments(stub, mb, pairs, layers_src=[stub.n_layers], alpha=0.1)
    v_src = random_directions(stub.d_model, 1, 0)[0]
    prompt, kw = "The weather was", dict(eps=1.0, n_prefix=2, gen_kw=dict(n_new=3, n_samples=1), seed=0)
    spy = _SpySource(stub)
    old = tp.transplant_experiment(mb, spy, aligns, v_src, [(prompt, " cold")], site, **kw)
    assert spy.seen == [prompt] and not old["paired"].any() and (old["source_item"] == prompt).all()
    T = mb.encode(prompt).shape[1]
    assert (old["n_src_tokens"] == stub.encode(prompt).shape[1]).all() and (old["n_tgt_tokens"] == T).all()
    new = tp.transplant_experiment(mb, stub, aligns, v_src, [tp.PromptPair(prompt, prompt, " cold")], site, **kw)
    num = ["entropy", "logp_target", "fD", "written_norm", "n_prefix", "n_substituted"]
    assert np.allclose(old[num].to_numpy(float), new[num].to_numpy(float), equal_nan=True)
    plain = tp.transplant_experiment(mb, stub, aligns, v_src, [prompt], site, targets=[" cold"], **kw)
    assert np.allclose(old[num].to_numpy(float), plain[num].to_numpy(float), equal_nan=True)
    # the helpers still take ids, and a text / pair encodes its target side
    ids = mb.encode(prompt)
    H = {L // 2: torch.zeros(2, mb.d_model)}
    for x in (prompt, tp.PromptPair([1, 2, 3], prompt)):
        assert torch.equal(tp.plant_kv_prefix(mb, x, H)["logits"], tp.plant_kv_prefix(mb, ids, H)["logits"])
        assert torch.equal(tp.substitute_residual(mb, x, site, [-1], torch.zeros(1, mb.d_model)),
                           tp.substitute_residual(mb, ids, site, [-1], torch.zeros(1, mb.d_model)))


def test_prompt_pairs_encode_differently_on_both_sides(stub, mb):
    """A pair's source side goes to the source model, its text to the target: different
    items, different lengths; the one-prompt helpers equal the manual route."""
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    pairs = [(t, t) for t in TEXTS_E[:8] + TEXTS_D[:8]]
    aligns = tp.layer_alignments(stub, mb, pairs, layers_src=[stub.n_layers], alpha=0.1)
    v_src = random_directions(stub.d_model, 1, 0)[0]
    src_ids = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]                # a "trace": 11 source tokens
    text = "The weather was"
    pair = tp.PromptPair(source=src_ids, text=text, target=" cold", name="tx-1")
    spy = _SpySource(stub)
    meas = HookSite(L - 1, "block_input")
    df, states = tp.transplant_experiment(mb, spy, aligns, v_src, [pair], site, eps=1.0, n_prefix=2, n_substitute=1,
                                          gen_kw=dict(n_new=3, n_samples=1), seed=0, state_sites=[meas])
    assert spy.seen == [src_ids]                                           # the source never saw the text
    T = mb.encode(text).shape[1]
    assert df["paired"].all() and (df["source_item"] == "tx-1").all() and (df["prompt"] == text).all()
    assert (df["n_src_tokens"] == len(src_ids)).all() and (df["n_tgt_tokens"] == T).all() and len(src_ids) != T
    assert (df.loc[df["mode"] != "kv", "n_substituted"] == 1).all()
    # states at the measurement site: one dict per row, last real position
    assert len(states) == len(df) and all(s[meas.key].shape == (mb.d_model,) for s in states)
    i_id = int(df.index[(df["mode"] == "residual") & (df["control"] == "identity")][0])
    own = run_with_cache(mb, mb.encode(text), [meas]).resid[meas.key][0, -1]
    assert np.allclose(states[i_id][meas.key], own.numpy(), atol=1e-4)
    # the pair's text side is what the target encodes; the source side is the trace
    H_S = stub.residuals(stub.encode(src_ids), [stub.n_layers])[stub.n_layers]
    H_T = tp.transplanted_states(stub, aligns, pair, v_src, 1.0)
    a = aligns[L // 2]
    manual = tp.transfer_states(a, tp._dose_rows(H_S, v_src, 1.0))
    assert H_T[L // 2].shape == (len(src_ids), mb.d_model) and torch.allclose(H_T[L // 2], manual, atol=1e-5)
    lo = tp.transplant_residual(mb, stub, aligns, pair, site, v_src, 1.0, n_substitute=2)
    ref = tp.substitute_residual(mb, mb.encode(text), site, [T - 2, T - 1], manual[-2:])
    assert torch.allclose(lo, ref, atol=1e-5)
    kv = tp.transplant_kv(mb, stub, aligns, pair, v_src, 1.0, n_prefix=3)
    ref_kv = tp.plant_kv_prefix(mb, mb.encode(text), {l: tp.transfer_states(al, tp._dose_rows(H_S, v_src, 1.0))[-3:]
                                                      for l, al in aligns.items()})
    assert kv["n_prefix"] == 3 and torch.allclose(kv["logits"], ref_kv["logits"], atol=1e-5)
    # a residual row of the experiment = the one-prompt helper (same dose, same positions)
    row = df[(df["mode"] == "residual") & (df["control"] == "none")].iloc[0]
    lo1 = tp.transplant_residual(mb, stub, aligns, pair, site, v_src, 1.0, n_substitute=1)
    assert row["entropy"] == pytest.approx(_entropy(lo1), abs=1e-4)
    # paired_states reads pairs (source item, target text) and PromptPairs alike
    S1, T1 = tp.paired_states(stub, mb, [(src_ids, text)], [0], [1])
    S2, T2 = tp.paired_states(stub, mb, [pair], [0], [1])
    assert torch.allclose(S1[0], S2[0]) and torch.allclose(T1[1], T2[1])
    assert torch.allclose(T2[1][0], tp.target_residuals(mb, mb.encode(text), [1])[1].mean(0))


def test_random_control_shared_by_prompts_of_one_carrier(stub, mb, monkeypatch):
    """The random control draws one direction per source item: two prompts on the same
    carrier (the K=0 variant and its K=8/C1 partner in stage L) get the same random
    direction, as the transplant arm doses both with the same v_src; different carriers get
    different directions; plain prompts keep one direction per prompt index."""
    L = mb.n_layers
    site = HookSite(L // 2, "block_input")
    pairs = [(t, t) for t in TEXTS_E[:8] + TEXTS_D[:8]]
    aligns = tp.layer_alignments(stub, mb, pairs, layers_src=[stub.n_layers], alpha=0.1)
    v_src = random_directions(stub.d_model, 1, 0)[0]
    drawn = []
    real = tp.random_directions

    def spy(d, n, seed=0):
        drawn.append(int(seed))
        return real(d, n, seed)
    monkeypatch.setattr(tp, "random_directions", spy)
    car_a, car_b = [11, 12, 13, 14], [21, 22, 23, 24]
    prompts = [tp.PromptPair(source=car_a, text="The weather was", target=" cold", name="0xa"),
               tp.PromptPair(source=car_a, text="Das Wetter war", target=" kalt", name="0xa"),
               tp.PromptPair(source=car_b, text="The weather was", target=" cold", name="0xb")]
    tp.transplant_experiment(mb, stub, aligns, v_src, prompts, site, eps=1.0, n_prefix=2, modes=("residual",),
                             controls=("random",), gen_kw=dict(n_new=2, n_samples=1), seed=0)
    assert len(drawn) == 3 and drawn[0] == drawn[1] != drawn[2]
    assert tp._random_key(prompts[0], 0) == tp._random_key(prompts[1], 1) != tp._random_key(prompts[2], 2)
    plain = tp.as_prompt_pair("The weather was")
    assert tp._random_key(plain, 0) == 0 and tp._random_key(plain, 3) == 3
