"""Tests for lsne.ttt (test-time training as structural injection, DESIGN §4.8d).

All training runs here use pythia-14m, snippets of <= 32 tokens and <= 5 steps
(~50 ms per step on CPU), so nothing is marked slow.
"""
import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

from lsne import ttt
from lsne.model import ModelBundle, forward_logits

_TEXT_A = ("In the year of the late flood a stranger came to the hollow, lodged with the miller, "
           "and studied the rules of the book with great patience before the first frost.")
_TEXT_B = ("Below the hollow the river divides around a long island of willows and the two "
           "channels run apart for a mile before they meet again below the weir.")


def _snippet(tok, n, text=_TEXT_A):
    ids = tok(text, add_special_tokens=False)["input_ids"][:n]
    assert len(ids) == n
    return torch.tensor(ids, dtype=torch.long)


def _state(mb):
    return {k: v.detach().clone() for k, v in mb.model.state_dict().items()}


def _bundle_copy(mb):
    return ModelBundle(model=copy.deepcopy(mb.model), tokenizer=mb.tokenizer,
                       spec=replace(mb.spec), device=mb.device)


# --------------------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------------------
def test_load_corpus_placeholder_when_missing(tok, tmp_path):
    corpus = ttt.load_corpus(tmp_path / "does_not_exist", tok, seq_len=64)
    assert isinstance(corpus, ttt.Corpus)
    assert corpus.is_placeholder and corpus.meta["is_placeholder"]
    assert ttt.LAST_LOAD_WAS_PLACEHOLDER and ttt.LAST_LOAD_META["is_placeholder"]
    assert corpus.meta["source"] == str(ttt.PLACEHOLDER_PATH)
    assert corpus.meta["requested"].endswith("does_not_exist")
    assert len(corpus) >= 20 and corpus.meta["n_snippets"] == len(corpus)
    assert all(s.dtype == torch.long and s.dim() == 1 and 2 <= len(s) <= 64 for s in corpus)
    assert ttt.load_corpus(None, tok).is_placeholder
    # an existing but empty directory also falls back
    assert ttt.load_corpus(tmp_path, tok).is_placeholder
    assert ttt.placeholder_corpus(tok, n_chunks=3).is_placeholder


def test_load_corpus_reads_txt_and_jsonl(tok, tmp_path):
    (tmp_path / "a.txt").write_text("first document here.\n---\nsecond document here.\n", encoding="utf-8")
    with (tmp_path / "b.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": "third document from jsonl."}) + "\n")
        fh.write("\n")
        fh.write(json.dumps({"text": "fourth document from jsonl."}) + "\n")
        fh.write(json.dumps({"other": "ignored"}) + "\n")
    corpus = ttt.load_corpus(tmp_path, tok, seq_len=512)
    assert not corpus.is_placeholder and not ttt.LAST_LOAD_WAS_PLACEHOLDER
    assert len(corpus) == 4 and corpus.meta["n_docs"] == 4
    assert corpus.meta["n_tokens"] == sum(len(s) for s in corpus)
    # a single file, max_docs, and chunking at seq_len
    assert len(ttt.load_corpus(tmp_path / "a.txt", tok)) == 2
    assert ttt.load_corpus(tmp_path, tok, max_docs=1).meta["n_docs"] == 1
    small = ttt.load_corpus(tmp_path / "a.txt", tok, seq_len=2)
    assert all(2 <= len(s) <= 2 for s in small) and len(small) > 2


def test_load_corpus_skips_note_and_hidden_files(tok, tmp_path):
    # a drop folder that holds only its note (any extension / case) or hidden files
    # yields no documents -> placeholder; an explicitly given note file is still read
    (tmp_path / "README.txt").write_text("Drop files here; this directory is git-ignored.\n", encoding="utf-8")
    (tmp_path / "readme.jsonl").write_text(json.dumps({"text": "note"}) + "\n", encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    (tmp_path / ".hidden.txt").write_text("hidden document.\n", encoding="utf-8")
    assert all(ttt.is_note_file(p) for p in tmp_path.iterdir())
    assert not ttt.is_note_file(tmp_path / "doc.txt") and not ttt.is_note_file("corpus.jsonl")
    assert ttt.read_documents(tmp_path) == []
    assert ttt.load_corpus(tmp_path, tok).is_placeholder
    assert ttt.read_documents(tmp_path / "README.txt") == ["Drop files here; this directory is git-ignored."]
    # real documents next to the note are read, the note is not
    (tmp_path / "doc.txt").write_text("a real document.\n", encoding="utf-8")
    assert ttt.read_documents(tmp_path) == ["a real document."]
    # the repository's drop folders (data/ttt/{dao_fork,neutral,nofork}) contain only
    # their note: nothing in the repository contains the DAO-fork corpus or its controls
    for d in (ttt.DAO_CORPUS_DIR, ttt.TTT_DATA_DIR / "neutral", ttt.TTT_DATA_DIR / "nofork"):
        assert d.is_dir() and all(ttt.is_note_file(p) for p in d.iterdir()), d
        c = ttt.load_corpus(d, tok)
        assert c.is_placeholder and c.meta["source"] == str(ttt.PLACEHOLDER_PATH), d


def test_permute_corpus_keeps_multiset(tok):
    corpus = ttt.placeholder_corpus(tok, seq_len=64, n_chunks=6)
    perm = ttt.permute_corpus(corpus, seed=1)
    assert isinstance(perm, ttt.Corpus) and perm.meta["control"] == "permuted"
    assert len(perm) == len(corpus)
    changed = 0
    for s, p in zip(corpus, perm):
        assert len(s) == len(p)
        assert torch.equal(torch.sort(s).values, torch.sort(p).values)
        changed += int(not torch.equal(s, p))
    assert changed > 0
    perm2 = ttt.permute_corpus(corpus, seed=1)
    assert all(torch.equal(a, b) for a, b in zip(perm, perm2))
    perm3 = ttt.permute_corpus(corpus, seed=2)
    assert any(not torch.equal(a, b) for a, b in zip(perm, perm3))


def test_collate_and_controls(tok):
    a, b = _snippet(tok, 5), _snippet(tok, 3, _TEXT_B)
    batch = ttt.collate([a, b], pad_id=ttt.pad_id_of(tok))
    assert batch["input_ids"].shape == (2, 5)
    assert batch["attention_mask"].tolist() == [[1] * 5, [1, 1, 1, 0, 0]]
    assert batch["labels"][1, 3:].tolist() == [-100, -100]
    assert torch.equal(batch["labels"][0], a)
    corpus = [a, b]
    assert ttt.corpus_for_control(corpus, None) is corpus
    assert ttt.corpus_for_control(corpus, "permuted").meta["control"] == "permuted"
    other = [b]
    assert ttt.corpus_for_control(corpus, "neutral", {"neutral": other}) is other
    with pytest.raises(ValueError, match="neutral"):
        ttt.corpus_for_control(corpus, "neutral")
    with pytest.raises(ValueError, match="nofork"):
        ttt.corpus_for_control(corpus, "nofork", {"neutral": other})
    with pytest.raises(ValueError, match="unknown control"):
        ttt.corpus_for_control(corpus, "bogus")


# --------------------------------------------------------------------------------------
# memorisation speed
# --------------------------------------------------------------------------------------
def test_memorisation_speed_lowers_nll_and_restores(mb, tok, sample_ids):
    snippet = _snippet(tok, 24)
    before = forward_logits(mb, sample_ids)
    state_before = _state(mb)
    res = ttt.memorisation_speed(mb, snippet, lr=1e-3, steps=5, threshold=0.5)
    assert isinstance(res, ttt.TTTResult)
    assert res.nll_curve.shape == (6,) and np.isfinite(res.nll_curve).all()
    assert res.nll_curve[-1] < res.nll_curve[0]
    assert res.final_nll == res.nll_curve[-1] and res.nll_start == res.nll_curve[0]
    assert set(res.meta) >= {"lr", "steps", "threshold", "n_tokens", "train_params", "optimizer"}
    assert res.meta["n_tokens"] == 24 and res.meta["optimizer"] == "adamw"
    assert res.meta["loss_curve"].shape == (5,) and res.meta["grad_norm_curve"].shape == (5,)
    assert res.meta["restored"] is True
    # the passed bundle is untouched: weights bitwise, logits identical, frozen, eval
    after = forward_logits(mb, sample_ids)
    assert torch.equal(before, after)
    cur = mb.model.state_dict()
    assert all(torch.equal(cur[k], v) for k, v in state_before.items())
    assert not any(p.requires_grad for p in mb.model.parameters())
    assert all(p.grad is None for p in mb.model.parameters())
    assert not mb.model.training


def test_memorisation_speed_threshold_and_train_params(mb, tok):
    snippet = _snippet(tok, 16)
    hi = ttt.memorisation_speed(mb, snippet, lr=1e-3, steps=2, threshold=1e9)
    assert hi.steps_to_threshold == 0
    lo = ttt.memorisation_speed(mb, snippet, lr=1e-3, steps=2, threshold=-1.0)
    assert lo.steps_to_threshold is None
    mid = ttt.memorisation_speed(mb, snippet, lr=1e-3, steps=3, threshold=float(hi.nll_curve[0]) - 1e-6)
    assert mid.steps_to_threshold is not None and 1 <= mid.steps_to_threshold <= 3
    # parameter subsets are smaller and still train; the base model is still restored
    n_all = ttt.memorisation_speed(mb, snippet, lr=1e-3, steps=1).meta["n_trainable"]
    for tp in ("mlp", "attn", f"last_n:{max(1, mb.n_layers // 2)}"):
        r = ttt.memorisation_speed(mb, snippet, lr=3e-4, steps=2, train_params=tp)
        assert 0 < r.meta["n_trainable"] < n_all and r.meta["train_params"] == tp
        assert r.nll_curve[-1] < r.nll_curve[0]
    with pytest.raises(ValueError):
        ttt.memorisation_speed(mb, snippet, steps=1, train_params="nothing")
    with pytest.raises(ValueError):
        ttt.memorisation_speed(mb, snippet, steps=1, optimizer="lion")
    assert not any(p.requires_grad for p in mb.model.parameters())
    # a 2-D snippet (small batch) works too
    two = torch.stack([_snippet(tok, 12), _snippet(tok, 12, _TEXT_B)])
    r2 = ttt.memorisation_speed(mb, two, lr=3e-4, steps=2)
    assert r2.meta["n_tokens"] == 24 and r2.nll_curve[-1] < r2.nll_curve[0]


def test_memorisation_speed_inplace_and_snapshot(mb, tok, sample_ids):
    mbc = _bundle_copy(mb)
    snippet = _snippet(tok, 16)
    before = forward_logits(mbc, sample_ids)
    snap = ttt.WeightSnapshot(mbc)
    res = ttt.memorisation_speed(mbc, snippet, lr=1e-3, steps=2, inplace=True)
    assert res.meta["restored"] is False
    assert not torch.equal(forward_logits(mbc, sample_ids), before)
    assert snap.max_abs_diff() > 0
    assert not any(p.requires_grad for p in mbc.model.parameters())
    snap.restore()
    assert snap.max_abs_diff() == 0.0
    assert torch.equal(forward_logits(mbc, sample_ids), before)
    # context manager restores after an exception as well
    with pytest.raises(RuntimeError):
        with ttt.WeightSnapshot(mbc):
            with torch.no_grad():
                next(mbc.model.parameters()).add_(1.0)
            raise RuntimeError("boom")
    assert torch.equal(forward_logits(mbc, sample_ids), before)


# --------------------------------------------------------------------------------------
# test-time training of a bundle
# --------------------------------------------------------------------------------------
def test_test_time_train_copy_leaves_original(mb, tok, sample_ids):
    corpus = [_snippet(tok, 20), _snippet(tok, 17, _TEXT_B)]
    before = forward_logits(mb, sample_ids)
    state_before = _state(mb)
    mb2 = ttt.test_time_train(mb, corpus, lr=3e-4, steps=3, batch=2, seed=0)
    assert isinstance(mb2, ModelBundle) and mb2 is not mb and mb2.model is not mb.model
    assert mb2.n_layers == mb.n_layers and mb2.d_model == mb.d_model and mb2.device == mb.device
    assert torch.equal(forward_logits(mb, sample_ids), before)
    cur = mb.model.state_dict()
    assert all(torch.equal(cur[k], v) for k, v in state_before.items())
    assert not torch.allclose(forward_logits(mb2, sample_ids), before)
    st = mb2.ttt_stats
    assert st["loss"].shape == (3,) and np.isfinite(st["loss"]).all()
    assert st["grad_norm"].shape == (3,) and st["control"] is None and st["n_snippets"] == 2
    assert len(st["batch_indices"]) == 3 and all(len(b) == 2 for b in st["batch_indices"])
    assert not any(p.requires_grad for p in mb2.model.parameters()) and not mb2.model.training
    assert not any(p.requires_grad for p in mb.model.parameters())
    # the trained copy knows the corpus better
    nll_b = ttt.corpus_nll(mb, corpus)["nll"].mean()
    nll_a = ttt.corpus_nll(mb2, corpus)["nll"].mean()
    assert nll_a < nll_b
    # same seed -> same result; different seed -> different batch order
    mb3 = ttt.test_time_train(mb, corpus, lr=3e-4, steps=3, batch=2, seed=0)
    assert torch.equal(forward_logits(mb3, sample_ids), forward_logits(mb2, sample_ids))
    mb4 = ttt.test_time_train(mb, corpus, lr=3e-4, steps=3, batch=1, seed=1)
    assert mb4.ttt_stats["batch_indices"] != mb2.ttt_stats["batch_indices"]


def test_test_time_train_controls_and_inplace(mb, tok, sample_ids):
    corpus = ttt.Corpus([_snippet(tok, 20), _snippet(tok, 17, _TEXT_B)], dict(is_placeholder=False))
    with pytest.raises(ValueError, match="neutral"):
        ttt.test_time_train(mb, corpus, steps=1, control="neutral")
    with pytest.raises(ValueError, match="unknown control"):
        ttt.test_time_train(mb, corpus, steps=1, control="bogus")
    with pytest.raises(ValueError, match="empty"):
        ttt.test_time_train(mb, [], steps=1)
    mbp = ttt.test_time_train(mb, corpus, lr=1e-3, steps=2, batch=2, control="permuted")
    assert mbp.ttt_stats["control"] == "permuted"
    assert mbp.ttt_stats["corpus_meta"]["control"] == "permuted"
    neutral = [_snippet(tok, 9, _TEXT_B)]
    mbn = ttt.test_time_train(mb, corpus, lr=1e-3, steps=2, batch=2, control="nofork",
                              controls={"nofork": neutral})
    assert mbn.ttt_stats["control"] == "nofork" and mbn.ttt_stats["n_snippets"] == 1
    # inplace on a private copy returns the same object with changed weights
    mbc = _bundle_copy(mb)
    before = forward_logits(mbc, sample_ids)
    out = ttt.test_time_train(mbc, corpus, lr=1e-3, steps=2, batch=2, inplace=True)
    assert out is mbc and out.ttt_stats["inplace"] is True
    assert not torch.equal(forward_logits(mbc, sample_ids), before)


# --------------------------------------------------------------------------------------
# before / after, memorisation table
# --------------------------------------------------------------------------------------
def test_before_after(mb, tok):
    corpus = [_snippet(tok, 14), _snippet(tok, 11, _TEXT_B)]
    mb2 = ttt.test_time_train(mb, corpus, lr=3e-4, steps=3, batch=2)
    df = ttt.before_after(mb, mb2, lambda b: ttt.corpus_nll(b, corpus), label="corpus_nll")
    assert isinstance(df, pd.DataFrame) and list(df.columns[:2]) == ["phase", "label"]
    assert len(df) == 2 * len(corpus)
    assert df["phase"].tolist() == ["before"] * 2 + ["after"] * 2
    assert (df["label"] == "corpus_nll").all()
    assert df[df.phase == "after"]["nll"].mean() < df[df.phase == "before"]["nll"].mean()
    # a dict-valued measure gives one row per phase
    d = ttt.before_after(mb, mb2, lambda b: {"nll": float(ttt.corpus_nll(b, corpus)["nll"].mean())})
    assert len(d) == 2 and set(d["phase"]) == {"before", "after"} and (d["label"] == "").all()


def test_ttt_memorisation_table(mb, tok, sample_ids):
    corpus = [_snippet(tok, 12), _snippet(tok, 10, _TEXT_B)]
    controls = {"permuted": ttt.permute_corpus(corpus, seed=0), "neutral": [_snippet(tok, 8, _TEXT_B)]}
    before = forward_logits(mb, sample_ids)
    df = ttt.ttt_memorisation_table(mb, corpus, controls, checkpoints=None,
                                    lr=3e-4, steps=3, threshold=0.5)
    assert list(df.columns[:3]) == ["checkpoint", "arm", "snippet"]
    assert {"steps_to_threshold", "final_nll", "nll_start", "n_tokens"} <= set(df.columns)
    assert len(df) == 2 + 2 + 1
    assert set(df["arm"]) == {"injection", "permuted", "neutral"}
    assert (df["checkpoint"] == mb.spec.revision).all()
    assert df.groupby("arm")["snippet"].apply(list).to_dict() == {
        "injection": [0, 1], "neutral": [0], "permuted": [0, 1]}
    assert (df["final_nll"] < df["nll_start"]).all()
    assert str(df["steps_to_threshold"].dtype) == "Int64"
    assert torch.equal(forward_logits(mb, sample_ids), before)
    # include_permuted adds the arm, max_snippets limits it; a known checkpoint reuses mb
    df2 = ttt.ttt_memorisation_table(mb, corpus, None, checkpoints=[mb.spec.revision],
                                     include_permuted=True, max_snippets=1, lr=1e-3, steps=1)
    assert len(df2) == 2 and set(df2["arm"]) == {"injection", "permuted"}
    with pytest.raises(ValueError, match="collides"):
        ttt.ttt_memorisation_table(mb, corpus, {"injection": corpus}, steps=1)
