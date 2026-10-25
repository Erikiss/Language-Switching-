"""Tests for lsne.patching (counterfactual position patching) on pythia-14m."""
import math
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite
from lsne.model import Injector, forward_logits, run_with_cache
from lsne.patching import (ATTRIBUTION_COLUMNS, SWITCH_CONTEXT_WINDOW, align_from_end,
                           align_shared_suffix, align_tail, counterfactual_attribution,
                           draw_null_pairs, grad_x_act, null_subset_count, patch_run,
                           position_classes, recovery_fraction)

UNIT1_E = "The soldier reads the report in the morning."
UNIT2_E = " The captain writes the letter at night."
UNIT2_D = " Der Kapitän schreibt den Brief in der Nacht."
UNIT2_SUBJ_D = " Der Kapitän"                   # C3-style: German subject ...
UNIT2_PRED_E = " writes the letter at night."   # ... English predicate + adjunct
TAIL = " The report is about the"
TAIL_WS = "\n\nThe report is about the"         # tail with two real whitespace tokens
TARGET = " soldier"


def _ids(tok, text):
    return tok(text, add_special_tokens=False)["input_ids"]


def _klass(tok, ids):
    out = []
    for i in ids:
        s = tok.decode([i])
        if s.strip() == "":
            out.append("ws")
        elif all((not c.isalnum()) and (not c.isspace()) for c in s):
            out.append("punct")
        elif s.startswith(" "):
            out.append("word_start")
        else:
            out.append("word_cont")
    return out


def make_variant(tok, segments, name, tail=TAIL):
    """Duck-typed variant from (text, lang) segments followed by the English tail
    (no lsne.stimuli). Boundary = first alphanumeric-bearing token of a segment whose
    language differs (as in lsne.stimuli.annotate)."""
    ids, lang, unit, klass, dist = [], [], [], [], []
    last_switch = -1
    prev_lang = None
    for u, (text, lg) in enumerate(segments + [(tail, "E")]):
        seg = _ids(tok, text)
        seg_k = _klass(tok, seg)
        if prev_lang is not None and lg != prev_lang:
            j = next(j for j, i in enumerate(seg) if any(c.isalnum() for c in tok.decode([i])))
            seg_k[j] = "boundary"
            last_switch = len(ids) + j
        prev_lang = lg
        ids += seg
        lang += [lg] * len(seg)
        unit += [u + 1] * len(seg)
        klass += seg_k
        dist += [(len(ids) - len(seg) + j - last_switch) if last_switch >= 0 else -1 for j in range(len(seg))]
    tail_start = len(ids) - len(_ids(tok, tail))
    target_id = _ids(tok, tail + TARGET)[-1]
    return types.SimpleNamespace(
        ids=ids, target_id=target_id,
        annot=dict(lang=lang, unit=unit, klass=klass, dist_to_last_switch=dist,
                   tail_start=tail_start, last_switch_pos=last_switch),
        family_id="T01", formulation="A", K=0 if last_switch < 0 else 2, C="C1", name=name)


@pytest.fixture(scope="module")
def pair(tok):
    mono = make_variant(tok, [(UNIT1_E, "E"), (UNIT2_E, "E")], "mono")
    mixed = make_variant(tok, [(UNIT1_E, "E"), (UNIT2_D, "D")], "mixed")
    return mono, mixed


@pytest.fixture(scope="module")
def pair_c3(tok):
    """C3-style pair: the switch boundary (' writes') lies inside the shared suffix."""
    mono = make_variant(tok, [(UNIT1_E, "E"), (UNIT2_E, "E")], "mono")
    mixed = make_variant(tok, [(UNIT1_E, "E"), (UNIT2_SUBJ_D, "D"), (UNIT2_PRED_E, "E")], "mixed_c3")
    return mono, mixed


@pytest.fixture(scope="module")
def pair_ws(tok):
    """Pair whose tail carries two real whitespace tokens ('\\n', '\\n')."""
    mono = make_variant(tok, [(UNIT1_E, "E"), (UNIT2_E, "E")], "mono_ws", tail=TAIL_WS)
    mixed = make_variant(tok, [(UNIT1_E, "E"), (UNIT2_D, "D")], "mixed_ws", tail=TAIL_WS)
    return mono, mixed


def test_position_classes(pair):
    mono, mixed = pair
    pc = position_classes(mixed)
    n = len(mixed.ids)
    for k in ("ws", "punct", "boundary", "word_start", "word_cont", "tail", "all", "switch_context",
              "post_punct", "delim"):
        assert k in pc
    assert pc["all"] == list(range(n))
    assert pc["tail"] == list(range(mixed.annot["tail_start"], n))
    assert pc["boundary"] == [t for t, k in enumerate(mixed.annot["klass"]) if k == "boundary"]
    assert len(pc["boundary"]) == 2            # E->D and D->E (tail)
    b = pc["boundary"][0]
    assert all(b + j in pc["switch_context"] for j in range(SWITCH_CONTEXT_WINDOW + 1))
    assert pc["punct"] and all(mixed.annot["klass"][t] == "punct" for t in pc["punct"])
    # all klass-derived classes partition the positions
    covered = sorted(pc["ws"] + pc["punct"] + pc["boundary"] + pc["word_start"] + pc["word_cont"])
    assert covered == list(range(n))
    # delim = ws + punct + boundary (the pre-registered union), sorted and duplicate free
    assert pc["delim"] == sorted(set(pc["ws"]) | set(pc["punct"]) | set(pc["boundary"]))
    # post_punct = the (space-carrying) token right after every punctuation token
    klass = mixed.annot["klass"]
    assert pc["post_punct"] == [t for t in range(1, n) if klass[t - 1] == "punct" and klass[t] not in ("punct", "ws")]
    assert len(pc["post_punct"]) == 2          # ' Der' after the first '.', ' The' after the second
    assert all(t - 1 in pc["punct"] for t in pc["post_punct"])
    # monolingual variant: no boundaries, and no lang-change fallback either
    assert position_classes(mono)["boundary"] == []
    # fallback: klass without boundary labels -> boundaries from language changes
    v = types.SimpleNamespace(ids=mixed.ids, target_id=mixed.target_id,
                              annot=dict(lang=mixed.annot["lang"], klass=["word_start"] * n,
                                         tail_start=mixed.annot["tail_start"]))
    assert position_classes(v)["boundary"] == pc["boundary"]


def test_align_tail(pair):
    mono, mixed = pair
    pairs = align_tail(mono, mixed)
    n_tail = len(mono.ids) - mono.annot["tail_start"]
    assert len(pairs) == n_tail
    assert pairs[-1] == (len(mono.ids) - 1, len(mixed.ids) - 1)
    assert all(mono.ids[s] == mixed.ids[d] for s, d in pairs)
    assert [d for _, d in pairs] == list(range(mixed.annot["tail_start"], len(mixed.ids)))
    # mismatching tails are rejected
    bad = types.SimpleNamespace(ids=mixed.ids[:-1] + [mixed.ids[-1] + 1], target_id=mixed.target_id,
                                annot=dict(mixed.annot))
    with pytest.raises(ValueError):
        align_tail(mono, bad)
    # end-aligned non-tail positions keep only matching ids
    ext = align_from_end(mono, mixed, list(range(mixed.annot["tail_start"])))
    assert all(mono.ids[s] == mixed.ids[d] for s, d in ext)


def test_align_shared_suffix(pair, pair_c3):
    mono, mixed = pair
    suf = align_shared_suffix(mono, mixed)
    tail = align_tail(mono, mixed)
    # contains the tail, is contiguous, end-aligned and id-matched
    assert suf[-len(tail):] == tail
    assert all(mono.ids[s] == mixed.ids[d] for s, d in suf)
    assert all((suf[i + 1][0] - suf[i][0], suf[i + 1][1] - suf[i][1]) == (1, 1) for i in range(len(suf) - 1))
    # both unit-2 variants end with '.', so the suffix extends exactly one token into the prefix
    assert len(suf) == len(tail) + 1
    s0, d0 = suf[0]
    assert mono.ids[s0 - 1] != mixed.ids[d0 - 1]       # maximal: the next pair mismatches
    # C3-style pair: the English predicate of unit 2 incl. the boundary token lies in the suffix
    mono3, mixed3 = pair_c3
    suf3 = align_shared_suffix(mono3, mixed3)
    b_de, b_ed = position_classes(mixed3)["boundary"]     # ' Der' (E->D), ' writes' (D->E)
    assert b_de not in {d for _, d in suf3}
    assert b_ed in {d for _, d in suf3} and b_ed < mixed3.annot["tail_start"]
    # identical variants: the whole sequence is shared
    assert align_shared_suffix(mono, mono) == [(t, t) for t in range(len(mono.ids))]


def test_null_subset_count_and_draw():
    assert null_subset_count(11, 11, 8, 0) == 1           # class == tail pool -> degenerate
    assert null_subset_count(11, 9, 8, 0) == math.comb(11, 9)
    assert null_subset_count(11, 1, 8, 3) == 11 * math.comb(8, 3)
    assert null_subset_count(11, 0, 8, 0) == 1
    assert null_subset_count(3, 4, 8, 0) == 0            # more positions than the pool holds
    tail_pool = [(s, s + 2) for s in range(20, 31)]
    prefix_pool = [(s, s + 2) for s in range(10, 18)]
    rng = np.random.default_rng(0)
    for _ in range(20):
        rp = draw_null_pairs(rng, tail_pool, prefix_pool, 3, 2)
        assert len(rp) == 5 and len({d for _, d in rp}) == 5
        assert sum(p in tail_pool for p in rp) == 3 and sum(p in prefix_pool for p in rp) == 2
        assert [d for _, d in rp] == sorted(d for _, d in rp)
    assert draw_null_pairs(rng, tail_pool, prefix_pool, 0, 0) == []
    assert draw_null_pairs(rng, tail_pool, [], 2, 0) and all(p in tail_pool for p in draw_null_pairs(rng, tail_pool, [], 2, 0))


def test_patch_run_last_block_output_recovers_source(mb, pair):
    mono, mixed = pair
    L = mb.n_layers
    site = HookSite(L - 1, "block_output")
    pairs = align_tail(mono, mixed)
    r = patch_run(mb, mono.ids, mixed.ids, site, pairs, mixed.target_id)
    assert set(r) >= {"logp_patched", "logp_src", "logp_dst"}
    # the last position's block output fixes the logits -> patched == source
    assert abs(r["logp_patched"] - r["logp_src"]) < 1e-4
    assert abs(r["logp_src"] - r["logp_dst"]) > 1e-3     # the text effect exists
    # no pairs -> destination run
    r0 = patch_run(mb, mono.ids, mixed.ids, site, [], mixed.target_id)
    assert abs(r0["logp_patched"] - r0["logp_dst"]) < 1e-6


def test_recovery_fraction():
    assert recovery_fraction(-1.0, -2.0, 0.0) == pytest.approx(0.5)
    assert np.isnan(recovery_fraction(-1.0, -2.0, -2.0))


def test_counterfactual_attribution(mb, pair):
    mono, mixed = pair
    L = mb.n_layers
    sites = [HookSite(0, "block_input"), HookSite(L // 2, "block_input"), HookSite(L - 1, "block_output")]
    classes = ["tail", "all", "boundary", "switch_context", "ws", "punct", "post_punct", "delim", "word_start"]
    with pytest.warns(UserWarning, match="ws"):          # the empty class is flagged, not silent
        df = counterfactual_attribution(mb, mono, mixed, sites, classes, n_random=3, seed=0)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(ATTRIBUTION_COLUMNS)
    assert len(df) == len(sites) * len(classes)
    assert (df["region"] == "suffix").all()
    assert (df["logp_mono"] - df["logp_mixed"]).abs().gt(1e-3).all()
    last = df[df["site"] == sites[2].key].set_index("class")
    mid = df[df["site"] == sites[1].key].set_index("class")
    first = df[df["site"] == sites[0].key].set_index("class")
    suffix = align_shared_suffix(mono, mixed)
    n_tail = len(align_tail(mono, mixed))
    # pools: the shared suffix split at tail_start
    assert (df["n_pool_tail"] == n_tail).all() and (df["n_pool_nontail"] == len(suffix) - n_tail).all()
    assert (df["n_tail"] + df["n_nontail"] == df["n_positions"]).all()
    # full tail / full suffix at the last block output: logits of the last position are the mono ones
    assert last.loc["tail", "recovery"] == pytest.approx(1.0, abs=1e-3)
    assert last.loc["all", "recovery"] == pytest.approx(1.0, abs=1e-3)
    assert last.loc["tail", "n_positions"] == n_tail and last.loc["tail", "n_nontail"] == 0
    assert last.loc["all", "n_positions"] == len(suffix) and last.loc["all", "n_nontail"] == len(suffix) - n_tail
    # block-0 input of the tail is the token embedding -> identical in both variants
    assert abs(first.loc["tail", "recovery"]) < 1e-3
    # whitespace tokens do not occur in this text: nothing patchable, flagged as unmeasurable
    assert first.loc["ws", "n_positions"] == 0 and np.isnan(first.loc["ws", "recovery"])
    assert not first.loc["ws", "measurable"] and not first.loc["ws", "control_defined"]
    assert np.isnan(first.loc["ws", "recovery_random_mean"])
    assert (df["measurable"] == (df["n_positions"] > 0)).all()
    # classes that exhaust their pools have no random control: it would be the measurement itself
    for cls in ("tail", "all"):
        assert not last.loc[cls, "control_defined"] and last.loc[cls, "n_null_subsets"] == 1
        assert np.isnan(last.loc[cls, "recovery_random_mean"]) and np.isnan(last.loc[cls, "recovery_random_sd"])
    # ... and the control exists exactly where it is defined
    ok = df["control_defined"]
    assert (ok == ((df["n_positions"] > 0) & (df["n_null_subsets"] > 1))).all()
    assert ok.any()
    assert np.isfinite(df.loc[ok, "recovery_random_mean"]).all()
    assert (df.loc[ok, "recovery_random_sd"] >= 0).all()
    assert df.loc[~ok, "recovery_random_mean"].isna().all()
    # the first boundary (' Der') is not in the shared suffix; the tail boundary is
    b = position_classes(mixed)["boundary"]
    assert last.loc["boundary", "n_positions"] <= len(b)
    assert last.loc["switch_context", "n_positions"] >= last.loc["boundary", "n_positions"]
    # the pre-registered union class is measurable here through punct (+ boundary on the tail)
    assert last.loc["delim", "n_positions"] >= last.loc["punct", "n_positions"] > 0
    assert last.loc["post_punct", "n_positions"] > 0 and last.loc["post_punct", "control_defined"]
    # mid-layer block input: the stratified control of a saturated class (word_start covers most
    # of both pools) is finite and close to, but not identical with, the measurement
    assert mid.loc["word_start", "control_defined"] and np.isfinite(mid.loc["word_start", "recovery_random_mean"])
    assert mid.loc["word_start", "recovery_random_mean"] != mid.loc["word_start", "recovery"]
    # a row depends only on (seed, site, class): a subset call reproduces it exactly
    df2 = counterfactual_attribution(mb, mono, mixed, sites[1:2], ["word_start"], n_random=3, seed=0)
    pd.testing.assert_frame_equal(df[(df["site"] == sites[1].key) & (df["class"] == "word_start")].reset_index(drop=True),
                                  df2.reset_index(drop=True))
    df3 = counterfactual_attribution(mb, mono, mixed, sites[1:2], ["word_start"], n_random=3, seed=1)
    assert df3.loc[0, "recovery_random_mean"] != df2.loc[0, "recovery_random_mean"]
    assert df3.loc[0, "recovery"] == df2.loc[0, "recovery"]          # the measurement itself is seed free
    with pytest.raises(ValueError):
        counterfactual_attribution(mb, mono, mixed, sites[:1], ["bogus"], n_random=1)
    with pytest.raises(ValueError):
        counterfactual_attribution(mb, mono, mixed, sites[:1], ["tail"], n_random=1, region="bogus")


def test_counterfactual_attribution_prefix_boundary_is_controlled_within_its_stratum(mb, pair_c3):
    """The boundary token lies in the prefix part of the shared suffix: it must be patched
    (n_nontail=1) and its null must be drawn from the prefix pool, never from the tail."""
    mono, mixed = pair_c3
    site = HookSite(mb.n_layers // 2, "block_input")
    ts = mixed.annot["tail_start"]
    suffix = align_shared_suffix(mono, mixed)
    prefix_pool = [p for p in suffix if p[1] < ts]
    tail_pool = [p for p in suffix if p[1] >= ts]
    assert len(prefix_pool) >= 4
    for region in ("suffix", "tail"):
        df = counterfactual_attribution(mb, mono, mixed, [site], ["boundary", "switch_context", "tail"],
                                        n_random=4, seed=0, region=region).set_index("class")
        assert df.loc["boundary", "n_positions"] == 1 and df.loc["boundary", "n_nontail"] == 1
        assert df.loc["boundary", "n_tail"] == 0
        assert df.loc["switch_context", "n_nontail"] == SWITCH_CONTEXT_WINDOW + 1
        assert df.loc["boundary", "control_defined"] and np.isfinite(df.loc["boundary", "recovery_random_mean"])
        assert df.loc["boundary", "n_null_subsets"] == df.loc["boundary", "n_pool_nontail"] >= 4
        # the null of a prefix-only class never touches the tail
        assert df.loc["boundary", "n_pool_tail"] == len(tail_pool)
        assert int(df.loc["boundary", "dst_positions"]) == position_classes(mixed)["boundary"][-1]   # ' writes'
        assert not df.loc["tail", "control_defined"]
    # the stratified draw itself: n_nontail positions from the prefix pool only
    rng = np.random.default_rng(0)
    for _ in range(10):
        rp = draw_null_pairs(rng, tail_pool, prefix_pool, 0, 1)
        assert len(rp) == 1 and rp[0] in prefix_pool and rp[0][1] < ts


def test_counterfactual_attribution_ws_measurable_with_real_whitespace_tokens(mb, pair_ws):
    mono, mixed = pair_ws
    pc = position_classes(mixed)
    assert len(pc["ws"]) == 2 and all(t >= mixed.annot["tail_start"] for t in pc["ws"])
    site = HookSite(mb.n_layers - 1, "block_output")
    with warnings.catch_warnings():
        warnings.simplefilter("error")        # nothing is unmeasurable here -> no warning
        df = counterfactual_attribution(mb, mono, mixed, [site], ["ws", "delim"], n_random=3, seed=0).set_index("class")
    assert df.loc["ws", "n_positions"] == 2 and df.loc["ws", "measurable"]
    assert np.isfinite(df.loc["ws", "recovery"]) and df.loc["ws", "control_defined"]
    assert np.isfinite(df.loc["ws", "recovery_random_mean"])
    assert df.loc["delim", "n_positions"] >= 3      # ws + the '.' before the tail


def test_grad_x_act(mb, pair):
    mono, mixed = pair
    site = HookSite(mb.n_layers // 2, "block_input")
    ids = torch.tensor([mixed.ids])
    g = grad_x_act(mb, ids, site, mixed.target_id)
    T = ids.shape[1]
    assert g.shape == (T,) and torch.isfinite(g).all() and float(g.abs().sum()) > 0
    # first-order check: scaling the residual at position t by (1+a) changes logp by ~ a*g[t]
    base = float(torch.log_softmax(forward_logits(mb, ids)[0, -1], -1)[mixed.target_id])
    t = T - 1
    a = 1e-3

    def scale(h):
        h2 = h.clone()
        h2[:, t, :] = h2[:, t, :] * (1 + a)
        return h2

    with Injector(mb, site, scale):
        lo = forward_logits(mb, ids)
    new = float(torch.log_softmax(lo[0, -1], -1)[mixed.target_id])
    pred = a * float(g[t])
    assert abs((new - base) - pred) < 0.2 * abs(pred) + 1e-5
