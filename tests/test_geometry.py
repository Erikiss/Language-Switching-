"""Tests for lsne.geometry (Meaning-Contracts measures, DESIGN section 4.4)."""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from lsne import geometry as G
from lsne.config import HookSite, MASKS_UNIT_LEVEL
from lsne.model import run_with_cache


# --------------------------------------------------------------------------------------
# helpers: tiny hand-made stimulus bank (2 families x 2 formulations x E/D)
# --------------------------------------------------------------------------------------
TAIL = " The dog barks"
BANK = {
    "F01": {("A", "E"): "The cat sleeps on the mat.", ("A", "D"): "Die Katze schläft auf der Matte.",
            ("B", "E"): "A cat is sleeping on a mat.", ("B", "D"): "Eine Katze schläft auf einer Matte."},
    "F02": {("A", "E"): "The farmer reads the report in the morning.",
            ("A", "D"): "Der Bauer liest den Bericht am Morgen.",
            ("B", "E"): "In the morning the farmer reads a report.",
            ("B", "D"): "Am Morgen liest der Bauer einen Bericht."},
}


def make_variant(tok, text, family_id, formulation, K, C, lang):
    ids = tok(text + TAIL, add_special_tokens=False)["input_ids"]
    T = len(ids)
    tail_start = T - 3
    annot = dict(lang=[lang] * tail_start + ["E"] * (T - tail_start),
                 unit=[1] * tail_start + [2] * (T - tail_start),
                 klass=["word_start"] * T, tail_start=tail_start,
                 last_switch_pos=max(tail_start - 2, 0))
    return types.SimpleNamespace(ids=ids, target_id=ids[-1], annot=annot, family_id=family_id,
                                 formulation=formulation, K=K, C=C, lang=lang)


@pytest.fixture(scope="module")
def bank(tok):
    out = {}
    for fam, d in BANK.items():
        for (form, lang), text in d.items():
            out[(fam, form, lang)] = make_variant(tok, text, fam, form, 0 if lang == "E" else 2, "C1", lang)
    return out


@pytest.fixture(scope="module")
def groups(bank):
    g = {}
    for fam in BANK:
        g[(fam, 0, "C1")] = [bank[(fam, "A", "E")], bank[(fam, "B", "E")]]
        g[(fam, 2, "C1")] = [bank[(fam, "A", "D")], bank[(fam, "B", "D")]]
    return g


# --------------------------------------------------------------------------------------
# synthetic clouds
# --------------------------------------------------------------------------------------
def test_log_volume_shrinks_with_cloud():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 10))
    lv, lv_half = G.log_volume(X), G.log_volume(0.5 * X)
    assert np.isfinite(lv) and np.isfinite(lv_half)
    assert lv_half < lv
    # torch input and numpy input agree
    assert G.log_volume(torch.tensor(X, dtype=torch.float32)) == pytest.approx(lv, abs=1e-3)
    assert np.isnan(G.log_volume(X[:1]))


def test_participation_ratio_isotropic():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 10))
    pr = G.participation_ratio(X)
    assert 8.0 < pr <= 10.0
    # a 1-d cloud has PR 1
    line = np.outer(rng.standard_normal(50), rng.standard_normal(10))
    assert G.participation_ratio(line) == pytest.approx(1.0, abs=1e-9)


def test_mahalanobis_properties():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 10)) @ np.diag(np.linspace(1, 5, 10))
    U, lam = G.layer_covariance(X, 128)
    assert U.shape == (10, 10) and lam.shape == (10,)
    assert np.all(np.diff(lam) <= 1e-12)                       # descending
    assert G.layer_covariance(X, 4)[0].shape == (10, 4)
    assert G.layer_covariance(X[:6], 128)[1].shape == (5,)      # r = N-1
    x, y = rng.standard_normal(10), rng.standard_normal(10)
    assert G.mahalanobis(x, x, U, lam) == 0.0
    assert G.mahalanobis(x, y, U, lam) == pytest.approx(G.mahalanobis(y, x, U, lam))
    assert G.mahalanobis(x, y, U, lam) > 0
    # identity basis, unit eigenvalues -> Euclidean distance
    assert G.mahalanobis(x, y, np.eye(10), np.ones(10)) == pytest.approx(np.linalg.norm(x - y))
    disp = G.displacement(x, y, U, lam)
    assert disp["euclid"] == pytest.approx(np.linalg.norm(x - y))
    assert disp["mahal"] == pytest.approx(G.mahalanobis(x, y, U, lam))
    assert np.isnan(G.displacement(x, y)["mahal"])


def test_deflate_removes_top_direction():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 8))
    X[:, 0] *= 10.0                                            # dominant direction
    Xd, basis = G.deflate(X, 1)
    assert basis.shape == (1, 8) and Xd.shape == X.shape
    assert np.abs(Xd @ basis[0]).max() < 1e-9                 # residual projection ~ 0
    assert abs(abs(basis[0, 0]) - 1.0) < 0.1                   # top direction is axis 0
    assert np.allclose(Xd.mean(0), 0.0, atol=1e-12)
    # k = 0 is just centring; a supplied basis is reused
    X0, b0 = G.deflate(X, 0)
    assert b0.shape == (0, 8) and np.allclose(X0, X - X.mean(0))
    Xd2, b2 = G.deflate(X, 1, basis=basis)
    assert np.allclose(Xd2, Xd) and np.allclose(np.abs(b2), np.abs(basis))


def test_pool_states():
    H = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    assert np.allclose(G.pool_states(H, "mean"), H.mean(1))
    assert np.allclose(G.pool_states(H, "last"), H[:, -1])
    assert np.allclose(G.pool_states(H, "mean", positions=[1, 2]), H[:, 1:3].mean(1))
    assert np.allclose(G.pool_states(torch.tensor(H)[0], "last"), H[:1, -1])
    with pytest.raises(ValueError):
        G.pool_states(H, "median")


def test_per_position_and_alignment():
    rng = np.random.default_rng(4)
    R = rng.standard_normal((8, 5, 6))
    lv = G.per_position_log_volume(R)
    assert lv.shape == (5,) and np.all(np.isfinite(lv))
    assert lv[2] == pytest.approx(G.log_volume(R[:, 2, :]))
    A = G.align_tail_states([(rng.standard_normal((12, 4)), 9), (rng.standard_normal((15, 4)), 10)])
    assert A.shape == (2, 3, 4)
    h = rng.standard_normal((7, 4))
    assert np.allclose(G.align_tail_states([(h, 2), (h, 4)])[0], h[-3:])


# --------------------------------------------------------------------------------------
# masks
# --------------------------------------------------------------------------------------
def test_enumerate_unit_masks():
    masks = G.enumerate_unit_masks(2)
    assert len(masks) == 8 and len(set(masks)) == 8
    for m in masks:
        assert len(m) == 10 and set(m) <= {"E", "D"}
        assert m[0] == "E" and m[9] == "E" and m[8] == "D"
        assert G.count_switches(m) == 2
    assert MASKS_UNIT_LEVEL[2] in masks
    for K in (4, 8):
        ms = G.enumerate_unit_masks(K)
        assert MASKS_UNIT_LEVEL[K] in ms
        assert all(G.count_switches(m) == K and m[0] == "E" and m[-1] == "E" and m[-2] == "D" for m in ms)
    assert G.enumerate_unit_masks(0) == ["E" * 10]
    assert G.enumerate_unit_masks(0, n_units=7) == ["E" * 7]
    assert G.enumerate_unit_masks(3) == []                     # odd K impossible
    assert G.enumerate_unit_masks(2, n_units=4) == ["EEDE", "EDDE"]


def test_unit_mask_to_segment_mask():
    um = "EEEEDDDDDE"
    c1 = G.unit_mask_to_segment_mask(um, "C1")
    c2 = G.unit_mask_to_segment_mask(um, "C2")
    c3 = G.unit_mask_to_segment_mask(um, "C3")
    for sm in (c1, c2, c3):
        assert len(sm) == 30 and G.count_switches(sm) == 2
    assert "".join(c1) == "EEEEEEEEEEEE" + "DDDDDDDDDDDDDDD" + "EEE"
    # unit 4 (index 3) is the E unit before the E->D boundary; unit 9 (index 8) before D->E
    adj4, pred4, adj9, pred9 = 3 * 3 + 2, 3 * 3 + 1, 8 * 3 + 2, 8 * 3 + 1
    assert c2[adj4] == "D" and c2[pred4] == "E" and c2[adj9] == "E" and c2[pred9] == "D"
    assert c3[adj4] == "D" and c3[pred4] == "D" and c3[8 * 3] == "D"
    assert c3[adj9] == "E" and c3[pred9] == "E" and c3[8 * 3] == "D"
    assert c3[3 * 3] == "E"                                     # subject keeps the unit language
    # unchanged units are identical across levels
    assert c1[:9] == c2[:9] == c3[:9] and c1[27:] == c2[27:] == c3[27:]
    assert G.unit_mask_to_segment_mask("E" * 10, "C3") == ["E"] * 30
    with pytest.raises(ValueError):
        G.unit_mask_to_segment_mask("EEX", "C1")
    with pytest.raises(ValueError):
        G.unit_mask_to_segment_mask(um, "C4")


def test_segment_masks_consistent_with_stimuli_module():
    """If lsne.stimuli provides segment_masks_from_unit_mask, both implementations agree."""
    try:
        from lsne.stimuli import segment_masks_from_unit_mask as fn
    except Exception:  # module not written yet / no such function
        pytest.skip("lsne.stimuli.segment_masks_from_unit_mask not available")
    for K in (2, 4, 8):
        for um in G.enumerate_unit_masks(K):
            for C in ("C1", "C2", "C3"):
                assert list(fn(um, C)) == G.unit_mask_to_segment_mask(um, C), (um, C)


def test_realization_groups_with_dummy_builder():
    fam_a = types.SimpleNamespace(family_id="F01", formulation="A", units=list(range(10)))
    fam_b = types.SimpleNamespace(family_id="F01", formulation="B", units=list(range(10)))
    built = []

    def build(fam, seg_mask):
        assert len(seg_mask) == 30
        v = types.SimpleNamespace(family_id=fam.family_id, formulation=fam.formulation,
                                  mask=list(seg_mask), K=G.count_switches(seg_mask))
        built.append(v)
        return v

    groups = G.realization_groups([fam_a, fam_b], build, K_levels=(0, 2), C_levels=("C1", "C2", "C3"))
    assert set(groups) == {("F01", 0, "C1"), ("F01", 2, "C1"), ("F01", 2, "C2"), ("F01", 2, "C3")}
    assert len(groups[("F01", 0, "C1")]) == 2                  # one per formulation
    assert len(groups[("F01", 2, "C1")]) == 16                 # 8 masks x 2 formulations
    assert all(v.K == 2 for v in groups[("F01", 2, "C3")])
    assert {v.formulation for v in groups[("F01", 2, "C2")]} == {"A", "B"}
    assert len(built) == 2 + 3 * 16


# --------------------------------------------------------------------------------------
# group geometry / pair ordering on synthetic data
# --------------------------------------------------------------------------------------
def test_group_geometry_two_groups():
    rng = np.random.default_rng(5)
    a = rng.standard_normal((16, 12))
    b = 0.5 * rng.standard_normal((16, 12)) + 3.0
    df = G.group_geometry({"a": a, "b": b}, r=128, deflate_k=(0, 2))
    assert set(df["group"]) == {"a", "b"} and len(df) == 4
    num = ["log_volume", "participation_ratio", "disp_euclid", "disp_mahal"]
    assert np.isfinite(df[num].to_numpy()).all()
    ref = df[(df.group == "a")]
    assert (ref.disp_euclid == 0).all() and (ref.disp_mahal == 0).all()
    row_b0 = df[(df.group == "b") & (df.k == 0)].iloc[0]
    assert row_b0.disp_euclid == pytest.approx(np.linalg.norm(b.mean(0) - a.mean(0)))
    assert row_b0.disp_mahal > 0
    assert (df.r == 12).all() and (df.n == 16).all()
    # the smaller cloud has the smaller log-volume
    lv = df[df.k == 0].set_index("group")["log_volume"]
    assert lv["b"] < lv["a"]
    # explicit reference group
    df2 = G.group_geometry({"a": a, "b": b}, ref_group="b", deflate_k=(0,))
    assert (df2[df2.group == "b"].disp_mahal == 0).all() and (df2[df2.group == "a"].disp_mahal > 0).all()


def test_pair_ordering_synthetic_core_order():
    """Translations closest, free paraphrases next, other families farthest -> gate passes."""
    rng = np.random.default_rng(6)
    d, rows, X = 16, [], []
    fams = [f"F{i}" for i in range(8)]
    for fam in fams:
        centre = 10.0 * rng.standard_normal(d)
        for form in ("A", "B"):
            fc = centre + 1.0 * rng.standard_normal(d)
            for lang in ("E", "D"):
                X.append(fc + 0.05 * rng.standard_normal(d))
                rows.append(dict(family=fam, formulation=form, lang=lang))
    X = np.stack(X)                                             # N = 32 >> r = 6
    df = G.pair_ordering(X, pd.DataFrame(rows), r=6, k=0)
    assert list(df["category"][:4]) == list(G.PAIR_CATEGORIES)
    n_f = len(fams)
    assert df["n_pairs"].tolist() == [2 * n_f, 2 * n_f, 2 * n_f, 16 * n_f * (n_f - 1) // 2]
    assert bool(df["core_order_ok"].iloc[0]) is True
    m = df.set_index("category")["mean_mahal"]
    assert m["exact_translation"] < m["same_meaning_same_lang"] < m["other_family"]
    assert df["mean_cosine"].between(-1, 1).all()
    assert df.attrs["r"] == 6
    # with N <= d + 1 and r = N-1 the whitened points form a regular simplex: all pairwise
    # distances are equal and the gate is uninformative -> warned
    sub = slice(0, 12)                                          # 3 families, N = 12 <= d = 16
    with pytest.warns(UserWarning, match="regular simplex"):
        deg = G.pair_ordering(X[sub], pd.DataFrame(rows[sub]), r=128, k=0)
    assert deg.attrs["r"] == 11
    assert np.allclose(deg["sd_mahal"], 0.0, atol=1e-6)
    assert np.allclose(deg["mean_mahal"], deg["mean_mahal"].iloc[0], atol=1e-6)


# --------------------------------------------------------------------------------------
# pythia-14m
# --------------------------------------------------------------------------------------
def test_geometry_table_pythia(mb, groups):
    sites = [HookSite(1, "block_input"), HookSite(mb.n_layers - 1, "block_output")]
    gt = G.geometry_table(mb, groups, sites, readouts=("mean", "last"), deflate_k=(0, 2))
    cols = ["site", "readout", "k", "family_id", "K", "C", "group", "n", "log_volume",
            "participation_ratio", "disp_euclid", "disp_mahal"]
    assert cols == list(gt.columns[:len(cols)])
    assert set(gt.site) == {s.key for s in sites} and set(gt.readout) == {"mean", "last"}
    assert set(gt.k) == {0, 2} and set(gt.K) == {0, 2} and set(gt.family_id) == {"F01", "F02"}
    assert len(gt) == 2 * 2 * 2 * 4
    num = ["log_volume", "participation_ratio", "disp_euclid", "disp_mahal"]
    assert np.isfinite(gt[num].to_numpy()).all()
    assert (gt.n == 2).all()
    # K=0 groups are their own references; K=2 groups refer to the K=0 group of their family
    k0 = gt[gt.K == 0]
    assert (k0.disp_euclid == 0).all() and (k0.disp_mahal == 0).all()
    k2 = gt[gt.K == 2]
    assert (k2.disp_euclid > 0).all() and (k2.disp_mahal > 0).all()
    assert (k2.ref_group == k2.family_id + "|K0|C1").all()
    # effective rank: min(r, N-1, d) with N = 8 pooled states
    assert (gt.r == min(128, 8 - 1, mb.d_model)).all()
    # the mean readout of one variant equals the mean over positions of its residual
    v = groups[("F01", 0, "C1")][0]
    co = run_with_cache(mb, torch.tensor([v.ids]), [sites[0]])
    pooled = G.pool_states(co.resid[sites[0].key], "mean")
    assert pooled.shape == (1, mb.d_model)


def test_pair_ordering_pythia(mb, bank):
    site = HookSite(mb.n_layers // 2, "block_input")
    X, rows = [], []
    for (fam, form, lang), v in bank.items():
        co = run_with_cache(mb, torch.tensor([v.ids]), [site])
        X.append(G.pool_states(co.resid[site.key], "mean")[0])
        rows.append(dict(family=fam, formulation=form, lang=lang))
    # N = 8 pooled states -> r = 7 = N-1: the gate is degenerate on this tiny bank (warned),
    # but every quantity must still be well-defined and finite.
    with pytest.warns(UserWarning, match="regular simplex"):
        df = G.pair_ordering(np.stack(X), pd.DataFrame(rows), r=128, k=2)
    assert list(df["category"]) == list(G.PAIR_CATEGORIES)
    assert df["n_pairs"].tolist() == [4, 4, 4, 16]
    assert np.isfinite(df[["mean_mahal", "mean_cosine"]].to_numpy()).all()
    assert df["mean_cosine"].between(-1, 1).all()
    assert df["core_order_ok"].iloc[0] in (True, False)
    assert df["core_order_ok"].iloc[1:].isna().all()
