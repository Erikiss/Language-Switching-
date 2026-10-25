"""Tests for lsne.patching (counterfactual position patching) on pythia-14m."""
import types

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.config import HookSite
from lsne.model import Injector, forward_logits, run_with_cache
from lsne.patching import (SWITCH_CONTEXT_WINDOW, align_from_end, align_tail,
                           counterfactual_attribution, grad_x_act, patch_run,
                           position_classes, recovery_fraction)

UNIT1_E = "The soldier reads the report in the morning."
UNIT2_E = " The captain writes the letter at night."
UNIT2_D = " Der Kapitän schreibt den Brief in der Nacht."
TAIL = " The report is about the"
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


def make_variant(tok, segments, name):
    """Duck-typed variant from (text, lang) segments followed by the English tail
    (no lsne.stimuli). Boundary = first token of a segment whose language differs."""
    ids, lang, unit, klass, dist = [], [], [], [], []
    last_switch = -1
    prev_lang = None
    for u, (text, lg) in enumerate(segments + [(TAIL, "E")]):
        seg = _ids(tok, text)
        seg_k = _klass(tok, seg)
        if prev_lang is not None and lg != prev_lang:
            seg_k[0] = "boundary"
            last_switch = len(ids)
        prev_lang = lg
        ids += seg
        lang += [lg] * len(seg)
        unit += [u + 1] * len(seg)
        klass += seg_k
        dist += [(len(ids) - len(seg) + j - last_switch) if last_switch >= 0 else -1 for j in range(len(seg))]
    tail_start = len(ids) - len(_ids(tok, TAIL))
    target_id = _ids(tok, TAIL + TARGET)[-1]
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


def test_position_classes(pair):
    mono, mixed = pair
    pc = position_classes(mixed)
    n = len(mixed.ids)
    for k in ("ws", "punct", "boundary", "word_start", "word_cont", "tail", "all", "switch_context"):
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
    sites = [HookSite(0, "block_input"), HookSite(L - 1, "block_output")]
    classes = ["tail", "all", "boundary", "switch_context", "ws", "punct"]
    df = counterfactual_attribution(mb, mono, mixed, sites, classes, n_random=3, seed=0)
    assert isinstance(df, pd.DataFrame)
    cols = ["site", "class", "n_positions", "logp_mono", "logp_mixed", "logp_patched", "recovery",
            "recovery_random_mean", "recovery_random_sd"]
    assert all(c in df.columns for c in cols)
    assert len(df) == len(sites) * len(classes)
    assert (df["logp_mono"] - df["logp_mixed"]).abs().gt(1e-3).all()
    last = df[df["site"] == sites[1].key].set_index("class")
    first = df[df["site"] == sites[0].key].set_index("class")
    # full tail at the last block output: logits of the last position are the mono ones
    assert last.loc["tail", "recovery"] == pytest.approx(1.0, abs=1e-3)
    assert last.loc["all", "recovery"] == pytest.approx(1.0, abs=1e-3)
    assert last.loc["tail", "n_positions"] == len(align_tail(mono, mixed))
    # block-0 input of the tail is the token embedding -> identical in both variants
    assert abs(first.loc["tail", "recovery"]) < 1e-3
    # whitespace tokens do not occur in this text: nothing patchable
    assert first.loc["ws", "n_positions"] == 0 and np.isnan(first.loc["ws", "recovery"])
    assert np.isnan(first.loc["ws", "recovery_random_mean"])
    # the random control exists wherever something was patched
    ok = df["n_positions"] > 0
    assert np.isfinite(df.loc[ok, "recovery_random_mean"]).all()
    assert (df.loc[ok, "recovery_random_sd"] >= 0).all()
    # boundary/switch_context: the first boundary is in the prefix and only patched if ids match
    b = position_classes(mixed)["boundary"]
    assert last.loc["boundary", "n_positions"] <= len(b)
    assert last.loc["switch_context", "n_positions"] >= last.loc["boundary", "n_positions"]
    # deterministic in the seed
    df2 = counterfactual_attribution(mb, mono, mixed, sites[:1], ["tail"], n_random=3, seed=0)
    pd.testing.assert_frame_equal(df[(df["site"] == sites[0].key) & (df["class"] == "tail")].reset_index(drop=True),
                                  df2.reset_index(drop=True))
    with pytest.raises(ValueError):
        counterfactual_attribution(mb, mono, mixed, sites[:1], ["bogus"], n_random=1)


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
