"""Tests for lsne.chain.vocab / train / source / cli (docs/TRACE_SOURCE_SPEC.md §9, sections 5-8).

Fixture: tests/fixtures/xatu (ethPandaOps Xatu, CC BY 4.0; see its README).  Everything runs
on CPU with the tiny configuration; the target model for the alignment is pythia-14m (the
session fixture ``mb`` of conftest.py).
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from lsne.chain import itr, labels, text
from lsne.chain import cli as chain_cli
from lsne.chain import source as S
from lsne.chain import train as T
from lsne.chain import vocab as V
from lsne.config import HookSite, ModelSpec
from lsne.model import load_model, run_with_cache
from lsne.transplant import SourceModel, layer_matching

FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "xatu")
DRAIN = "0x0ec3f2488a93839524add10ea229e773f6bc891b4eb4794c3337d4495263790b"
HELD_OUT = ("0x124532e6", "0xa1fafc6f")          # two mid-sized transactions kept out of training


@pytest.fixture(scope="module")
def recs():
    return itr.load_records(1718000, FIX) + itr.load_records(1920000, FIX)


@pytest.fixture(scope="module")
def drain(recs):
    return next(r for r in recs if r.tx_hash == DRAIN)


@pytest.fixture(scope="module")
def vocab(recs):
    return V.build_vocab(recs, n_addresses=30)


@pytest.fixture(scope="module")
def trained(recs, vocab, tmp_path_factory):
    """Tiny trace LM trained 30 steps on the fixture (two transactions held out for validation)."""
    out = str(tmp_path_factory.mktemp("trace_lm"))
    tok = V.make_tokenizer(vocab)
    tr = [r for r in recs if not r.tx_hash.startswith(HELD_OUT)]
    va = [r for r in recs if r.tx_hash.startswith(HELD_OUT)]
    train_blocks = T.pack(V.encode_records(tr, vocab, tok, max_len=None), block_size=32, eos_id=vocab.eos_id)
    val_blocks = T.pack(V.encode_records(va, vocab, tok, max_len=None), block_size=32, eos_id=vocab.eos_id)
    model = T.make_model(len(vocab), "tiny", seed=0)
    df = T.train(model, train_blocks, val_blocks, out, lr=3e-3, batch_size=8, steps=30, warmup=3, eval_every=10,
                 save_every=0, seed=0, log_fn=None, tokenizer=tok, vocab=vocab, device="cpu")
    return dict(out=out, df=df, n_train=len(train_blocks), n_val=len(val_blocks))


@pytest.fixture(scope="module")
def src(trained):
    return S.TraceLMSource(trained["out"], device="cpu")


# ======================================================================================
# vocab / tokenizer
# ======================================================================================
def test_special_and_closed_tokens():
    assert V.SPECIAL[:3] == ["[PAD]", "[BOS]", "[EOS]"]
    for t in ("[TX]", "[/TX]", "[START]", "[END]", "[CALL]", "[STATE]", "[LOG]", "[INs]", "[OUTs]", "[W]", "[R]",
              "[WORD]", "[A_OOV]", "[F_OOV]", "[T_OOV]", "[UNK]", "[D0]", "[D32]", "[CT:delegate_call]",
              "[ERR:out_of_gas]", "N0", "N10e0", "N99e77", "N10e-1"):
        assert t in V.SPECIAL, t
    closed = V.closed_tokens()
    assert len(closed) == len(set(closed))
    assert closed[: len(V.SPECIAL)] == V.SPECIAL
    assert set(itr.fixed_tokens()) <= set(closed)
    assert not any(t[:2] in ("A:", "F:", "T:") and not t.endswith(":none") for t in closed)


def test_build_vocab_top_n_and_oov_mapping(recs, drain):
    small = V.build_vocab(recs, n_addresses=10, n_selectors=3, n_topics=2)
    assert small.top_addresses[:4] == list(labels.DAO_ADDRESSES.values())      # always included, first
    assert len(small.top_addresses) == 10 and len(small.top_selectors) == 3 and len(small.top_topics) == 2
    assert len(small) == len(V.closed_tokens()) + 15
    assert small.tokens[: len(V.closed_tokens())] == V.closed_tokens()          # stable ids of the closed classes
    # frequency order of the non-forced addresses
    cnt = {}
    for r in recs:
        for t in itr.serialize(r):
            if t.startswith("A:") and t != "A:none":
                cnt[t[2:]] = cnt.get(t[2:], 0) + 1
    rest = small.top_addresses[4:]
    assert [cnt[a] for a in rest] == sorted((cnt[a] for a in rest), reverse=True)
    # mapping a raw serialisation == serialising with the vocabulary
    for r in recs:
        mapped = small.map_tokens(itr.serialize(r))
        assert mapped == itr.serialize(r, small)
        assert all(t in small for t in mapped)
    toks = itr.serialize(drain, small)
    assert "[A_OOV]" in toks and "[F_OOV]" in toks and "[T_OOV]" in toks
    oov = V.oov_rates([itr.serialize(r, small) for r in recs])
    assert 0 < oov["rate_A"] < 1 and 0 < oov["rate_F"] < 1 and oov["rate_unk"] == 0
    full = V.build_vocab(recs)
    oov_full = V.oov_rates([itr.serialize(r, full) for r in recs])
    assert oov_full["rate_open"] == 0 and full.meta["coverage_A"] == 1.0
    assert V.build_vocab(recs, n_addresses=10, n_selectors=3, n_topics=2).tokens == small.tokens   # deterministic
    assert small.map_token("A:0x" + "e" * 40) == "[A_OOV]" and small.map_token("bogus") == "[UNK]"


def test_vocab_save_load_and_from_tokenizer(vocab, tmp_path):
    p = vocab.save(str(tmp_path))
    assert os.path.basename(p) == "vocab.json"
    back = V.TraceVocab.load(str(tmp_path))
    assert back.tokens == vocab.tokens and back.top_addresses == vocab.top_addresses and back.meta == vocab.meta
    rebuilt = V.TraceVocab.from_tokenizer(V.make_tokenizer(vocab))
    assert rebuilt.tokens == vocab.tokens and set(rebuilt.top_addresses) == set(vocab.top_addresses)


def test_tokenizer_roundtrip_and_autotokenizer(recs, drain, vocab, tmp_path):
    from transformers import AutoTokenizer

    out = str(tmp_path / "tok")
    V.make_tokenizer(vocab, out)
    assert {"tokenizer.json", "tokenizer_config.json", "vocab.json"} <= set(os.listdir(out))
    tok = AutoTokenizer.from_pretrained(out)
    assert (tok.pad_token_id, tok.bos_token_id, tok.eos_token_id) == (0, 1, 2)
    assert tok.unk_token_id == vocab.unk_id and len(tok) == len(vocab)
    ids = V.encode_records(recs, vocab, tok, max_len=None)
    for r, i in zip(recs, ids):
        toks = itr.serialize(r, vocab)
        assert tok(" ".join(toks), add_special_tokens=False)["input_ids"] == i
        assert tok.decode(i).split() == toks
        assert vocab.decode(i) == toks
    assert tok("[TX] not-a-token [/TX]", add_special_tokens=False)["input_ids"][1] == vocab.unk_id
    # truncation; token lists and shard-like dicts encode like the record
    d_ids = V.encode_records([drain], vocab, tok)[0]
    assert len(itr.serialize(drain)) > 512 and len(d_ids) == 512
    raw = itr.serialize(drain)
    assert V.encode_records([raw, {"tokens": raw}], vocab, max_len=512) == [d_ids, d_ids]


# ======================================================================================
# model / packing / training
# ======================================================================================
def test_pack_with_eos_separators():
    blocks = T.pack([[5, 6, 7], [8, 9], [10]], block_size=4, eos_id=2)
    assert blocks.dtype == torch.long and blocks.tolist() == [[5, 6, 7, 2], [8, 9, 2, 10]]   # remainder dropped
    assert T.pack([[5]], block_size=4).shape == (0, 4)
    assert T.concat_with_eos([[1, 2], [], [3]], eos_id=9).tolist() == [1, 2, 9, 3, 9]
    assert T.pack([[5, 6, 7], [8, 9]], block_size=2, dtype=torch.int32).dtype == torch.int32


def test_make_model_configs():
    m = T.make_model(1000, "tiny", seed=0)
    cfg = m.config
    assert cfg.model_type == "gpt_neox" and cfg.use_parallel_residual and not cfg.tie_word_embeddings
    assert cfg.rope_parameters["partial_rotary_factor"] == 0.25
    assert (cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.intermediate_size) == (64, 2, 4, 256)
    assert (cfg.pad_token_id, cfg.bos_token_id, cfg.eos_token_id) == (0, 1, 2) and cfg.max_position_embeddings == 512
    m2 = T.make_model(1000, "tiny", seed=0)
    assert all(torch.equal(a, b) for a, b in zip(m.parameters(), m2.parameters()))
    m3 = T.make_model(1000, "tiny", seed=1)
    assert not torch.equal(m.gpt_neox.embed_in.weight, m3.gpt_neox.embed_in.weight)
    pc = T.param_counts("blockgpt", 48195)
    assert pc["embedding"] == pc["unembedding"] == 48195 * 512
    assert 25_000_000 < pc["non_embedding"] < 25_500_000
    assert T.param_counts(m)["total"] == sum(p.numel() for p in m.parameters())
    with pytest.raises(ValueError):
        T.make_model(10, "huge")


def test_lr_schedule_and_batch_order():
    lrs = [T.lr_lambda(s, 100, 10, 0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:])) and lrs[-1] > 0.1 - 1e-9
    assert T.lr_lambda(100, 100, 10, 0.1) == pytest.approx(0.1)
    order = T.BatchOrder(10, 3, seed=0)
    seen = torch.cat([order.batch(s) for s in range(10)]).tolist()      # 30 samples = 3 epochs
    for e in range(3):
        assert sorted(seen[10 * e:10 * e + 10]) == list(range(10))
    assert order.batch(7).tolist() == T.BatchOrder(10, 3, seed=0).batch(7).tolist()


def test_make_model_loads_through_load_model(drain, vocab, tmp_path):
    out = str(tmp_path / "m")
    model = T.make_model(len(vocab), "tiny", seed=0)
    model.save_pretrained(out)
    V.make_tokenizer(vocab, out)
    mb = load_model(ModelSpec(name=out, revision=None, dtype="float32", device="cpu"))
    assert (mb.n_layers, mb.d_model, mb.vocab) == (2, 64, len(vocab))
    ids = torch.tensor([V.encode_records([drain], vocab, mb.tokenizer, max_len=64)[0]])
    co = run_with_cache(mb, ids)
    assert len(co.sites) == 3 and co.resid["L0.block_input"].shape == (1, 64, 64)
    assert co.logits.shape == (1, 64, len(vocab)) and torch.isfinite(co.logits).all()
    with torch.no_grad():
        ref = model.eval()(input_ids=ids).logits
    assert torch.allclose(ref, co.logits, atol=1e-5)
    assert S.load_trace_bundle(out, device="cpu").n_layers == 2


def test_training_lowers_validation_loss(trained):
    df, out = trained["df"], trained["out"]
    assert list(df.columns) == ["step", "train_loss", "val_loss", "lr", "tokens_seen", "elapsed_s"]
    assert df["step"].tolist() == [0, 10, 20, 30]
    assert trained["n_train"] > 20 and trained["n_val"] >= 5
    assert df["val_loss"].iloc[-1] < df["val_loss"].iloc[0] - 2.0, df
    assert np.isfinite(df["train_loss"].iloc[1:]).all()
    assert df["tokens_seen"].iloc[-1] == 30 * 8 * 32
    for f in ("config.json", "model.safetensors", "tokenizer.json", "vocab.json", "train_log.csv", "manifest.json"):
        assert os.path.exists(os.path.join(out, f)), f
    man = json.load(open(os.path.join(out, "manifest.json")))
    assert man["kind"] == "trace_lm" and man["hparams"]["steps"] == 30 and man["params"]["non_embedding"] > 0
    assert pd.read_csv(os.path.join(out, "train_log.csv"))["val_loss"].iloc[-1] == pytest.approx(df["val_loss"].iloc[-1])


def test_training_resume_is_exact(recs, vocab, tmp_path):
    blocks = T.pack(V.encode_records(recs, vocab, max_len=None), block_size=16, eos_id=vocab.eos_id)
    tr, va = blocks[:-6], blocks[-6:]
    kw = dict(lr=2e-3, batch_size=4, steps=6, warmup=2, eval_every=2, save_every=2, seed=3, device="cpu")
    ref_dir, cut_dir = str(tmp_path / "ref"), str(tmp_path / "cut")
    ref_model = T.make_model(len(vocab), "tiny", seed=0)
    ref = T.train(ref_model, tr, va, ref_dir, log_fn=None, **kw)

    class Stop(Exception):
        pass

    def stop_at_4(msg):
        if msg.startswith("step      4"):
            raise Stop

    with pytest.raises(Stop):
        T.train(T.make_model(len(vocab), "tiny", seed=0), tr, va, cut_dir, log_fn=stop_at_4, **kw)
    assert [s for s, _ in T.list_checkpoints(cut_dir)] == [2]
    msgs = []
    model = T.make_model(len(vocab), "tiny", seed=0)
    res = T.train(model, tr, va, cut_dir, log_fn=msgs.append, **kw)
    assert any("resumed" in m and "step 2" in m for m in msgs)
    assert res["step"].tolist() == ref["step"].tolist() == [0, 2, 4, 6]
    np.testing.assert_allclose(res["val_loss"].to_numpy(), ref["val_loss"].to_numpy(), rtol=1e-5)
    for a, b in zip(model.parameters(), ref_model.parameters()):
        assert torch.allclose(a, b, atol=1e-6)
    # a finished run is not repeated (final checkpoint at step 6)
    msgs2 = []
    again = T.train(T.make_model(len(vocab), "tiny", seed=0), tr, va, cut_dir, log_fn=msgs2.append, **kw)
    assert any("at step 6" in m for m in msgs2) and again["step"].tolist() == [0, 2, 4, 6]
    assert [s for s, _ in T.list_checkpoints(cut_dir)] == [4, 6]          # keep_checkpoints=2


def test_eval_loss_matches_hf_loss(recs, vocab):
    blocks = T.pack(V.encode_records(recs[:6], vocab), block_size=16, eos_id=vocab.eos_id)[:4]
    model = T.make_model(len(vocab), "tiny", seed=0).eval()
    with torch.no_grad():
        hf = float(model(input_ids=blocks, labels=blocks).loss)
    assert T.eval_loss(model, blocks, batch_size=3) == pytest.approx(hf, rel=1e-5)


def test_measure_throughput_smoke():
    r = T.measure_throughput("tiny", 500, block_size=16, batch_size=2, n_steps=1, n_warmup=0)
    assert r["tokens_per_s"] > 0 and r["params_total"] == T.param_counts("tiny", 500)["total"]
    assert r["flops_per_token"] == pytest.approx(T.flops_per_token("tiny", 500, block_size=16))


# ======================================================================================
# source adapter
# ======================================================================================
def test_trace_source_implements_protocol(src, recs, drain, trained):
    assert isinstance(src, SourceModel)
    assert (src.d_model, src.n_layers) == (64, 2) and src.name.startswith("trace-lm:")
    ids = src.encode(drain)
    assert ids.shape == (1, 512) and ids.dtype == torch.long
    short = recs[1]
    a = src.encode(short)
    assert torch.equal(a, src.encode(itr.serialize(short)))             # raw token list
    assert torch.equal(a, src.encode({"tokens": itr.serialize(short)}))  # shard row
    assert torch.equal(a, src.encode(a[0].tolist()))                    # ids
    hs = src.residuals(a)
    assert sorted(hs) == [0, 1, 2]
    assert all(h.shape == (a.shape[1], 64) and h.dtype == torch.float32 for h in hs.values())
    co = run_with_cache(src.mb, a, [HookSite(1, "block_input"), HookSite(1, "block_output")])
    assert torch.allclose(hs[1], co.resid["L1.block_input"][0]) and torch.allclose(hs[2], co.resid["L1.block_output"][0])
    assert sorted(src.residuals(a, [2])) == [2]


def test_layer_matching_with_pythia(src, recs, mb):
    pairs = S.paired_items(recs)
    assert len(pairs) == 15 and pairs[0][1] == text.render_text(pairs[0][0])
    df = layer_matching(src, mb, pairs)
    assert len(df) == (src.n_layers + 1) * (mb.n_layers + 1)
    assert set(df["layer_src"]) == {0, 1, 2} and set(df["layer_tgt"]) == set(range(mb.n_layers + 1))
    for c in ("r2_train", "r2_holdout", "r2_permuted"):
        assert np.isfinite(df[c]).all(), c
    assert (df["n_pairs"] == 15).all()


def test_sharing_direction(src, recs):
    pos, neg, mt = labels.probe_sets(recs, None, "dao", n=10, gas_bins=1)
    assert len(pos) >= 2
    d = S.sharing_direction(src, pos, neg, layer=1)
    assert d.site == HookSite(1, "block_input") and d.d == 64
    assert float(d.vec.norm()) == pytest.approx(1.0, abs=1e-5)
    P = S.pooled_states(src, pos + neg, [1])[1]
    ref = (P[: len(pos)].mean(0) - P[len(pos):].mean(0))
    assert torch.allclose(d.vec, ref / ref.norm(), atol=1e-5)
    m = d.meta
    assert m["n_pos"] == m["n_neg"] == len(pos) and m["denoise_pcs"] == 0 and m["readout"] == "mean"
    assert m["proj_pos_mean"] > m["proj_neg_mean"]
    if m["vec_denoised"] is not None:
        assert float(m["vec_denoised"].norm()) == pytest.approx(1.0, abs=1e-5)
        assert -1 <= m["cos_plain_denoised"] <= 1
    last = S.sharing_direction(src, pos, neg, layer=2, readout="call")
    assert last.site == HookSite(1, "block_output") and last.meta["readout"] == "call"
    with pytest.raises(ValueError):
        S.sharing_direction(src, pos, neg, layer=3)


def test_holdout_auc_separable():
    g = torch.Generator().manual_seed(0)
    a = torch.randn(20, 8, generator=g) + torch.tensor([4.0] + [0.0] * 7)
    b = torch.randn(20, 8, generator=g)
    assert S.holdout_auc(a, b) > 0.95
    assert np.isnan(S.holdout_auc(a[:3], b[:3]))
    d = S.direction_from_states(a, b, layer=0, n_layers=2)
    assert d.meta["auc_holdout"] > 0.95 and d.meta["separation_d"] > 2


# ======================================================================================
# round-2 controls: untrained twin, probe criterion, alignment against length
# ======================================================================================
def test_untrained_twin_is_the_initialisation(src, recs, vocab, trained):
    assert S.init_seed_of(trained["out"]) == 0 and S.init_seed_of("/nonexistent", default=5) == 5
    tw = S.untrained_twin(src, seed=0)
    ref = T.make_model(len(vocab), "tiny", seed=0)              # what the fixture model was trained from
    tw_p, ref_p = dict(tw.mb.model.named_parameters()), dict(ref.named_parameters())
    assert set(tw_p) == set(ref_p) and all(torch.equal(tw_p[k], ref_p[k]) for k in ref_p)
    tr_p = dict(src.mb.model.named_parameters())
    assert not all(torch.equal(tw_p[k], tr_p[k]) for k in tr_p)          # the trained weights are untouched
    assert (tw.d_model, tw.n_layers, tw.max_len) == (src.d_model, src.n_layers, src.max_len)
    assert tw.vocab is src.vocab and tw.name.endswith(":untrained(seed=0)") and isinstance(tw, SourceModel)
    ids = src.encode(recs[1])
    assert torch.equal(tw.encode(recs[1]), ids)
    h, h0 = src.residuals(ids), tw.residuals(ids)
    assert sorted(h0) == sorted(h) and not torch.allclose(h[2], h0[2])
    assert not any(p.requires_grad for p in tw.mb.model.parameters())
    other = S.untrained_twin(src, seed=1)
    assert not torch.equal(other.mb.model.gpt_neox.embed_in.weight, tw.mb.model.gpt_neox.embed_in.weight)


def test_masked_items(src, recs, drain):
    items = S.masked_items(src, [drain, recs[1]], "dao")
    assert items[0] == labels.mask_tokens(src.tokens(drain), "dao") and len(items[0]) == 512
    assert not any(t == f"A:{a}" for t in items[0] for a in labels.DAO_ADDRESSES.values())
    assert torch.equal(src.encode(items[1]), src.encode(recs[1])) or "[A_OOV]" in items[1]
    assert src.encode(items[0]).shape == (1, 512)
    assert S.masked_items(src, [recs[1]], "none")[0] == src.tokens(recs[1])


def _states(n, d, shift, seed):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(n, d, generator=g)
    pos[:, 0] += shift
    return pos, torch.randn(n, d, generator=g)


def test_split_half_auc_matches_holdout_auc():
    for seed in range(3):
        a, b = _states(30, 6, 1.0, seed)
        assert S.split_half_auc(a, b, seed=seed) == pytest.approx(S.holdout_auc(a, b, seed=seed), abs=1e-9)
    assert np.isnan(S.split_half_auc(a[:3], b[:3]))
    A, B = S.split_halves(30, 0)
    assert sorted(np.concatenate([A, B]).tolist()) == list(range(30)) and len(A) == 15
    assert S._auc_cols(np.array([3.0, 2.0]), np.array([1.0, 2.0]))[0] == pytest.approx(0.875)


def test_probe_criterion_trained_vs_untrained():
    n = 60
    sp, sq = _states(n, 8, 2.0, 0)                  # "trained": class signal on axis 0
    up, uq = _states(n, 8, 0.0, 1)                  # "untrained": no signal
    kw = dict(n_boot=200, n_perm=50, n_random=50)
    r = S.probe_criterion(sp, sq, up, uq, **kw)
    assert r["auc_trained"] > 0.85 and 0.25 < r["auc_untrained"] < 0.75
    assert r["delta"] == pytest.approx(r["auc_trained"] - r["auc_untrained"])
    assert r["delta_ci_lo"] > 0 and r["delta_ci_lo"] <= r["delta_ci_hi"] and r["counts"]
    assert r["delta_ci_eval_lo"] > 0 and r["auc_trained_ci_lo"] < r["auc_trained"] < r["auc_trained_ci_hi"] + 1e-9
    assert r["perm_auc_p95"] < r["auc_trained"] and r["perm_p"] < 0.05 and r["perm_delta_p"] < 0.05
    assert 0.5 <= r["random_dir_auc_mean"] <= r["random_dir_auc_p95"] <= r["random_dir_auc_max"] <= 1.0
    assert r["auc_trained"] == pytest.approx(S.holdout_auc(sp, sq), abs=1e-9)
    assert S.probe_criterion(sp, sq, up, uq, **kw) == r                          # deterministic
    # identical states for both models: difference exactly 0, the CI contains 0 -> does not count
    same = S.probe_criterion(sp, sq, sp, sq, **kw)
    assert same["delta"] == 0 and same["delta_ci_lo"] == same["delta_ci_hi"] == 0 and not same["counts"]
    # the untrained model separating as well as the trained one (the first run) -> does not count
    tie = S.probe_criterion(sp, sq, *_states(n, 8, 2.0, 5), **kw)
    assert tie["auc_untrained"] > 0.85 and tie["delta_ci_lo"] < 0 < tie["delta_ci_hi"] and not tie["counts"]
    # Bonferroni columns with several tests; NaN with too few pairs
    adj = S.probe_criterion(sp, sq, up, uq, n_tests=5, **kw)
    assert adj["ci_level_adj"] == pytest.approx(0.99) and adj["delta_ci_lo_adj"] <= adj["delta_ci_lo"]
    assert np.isnan(S.probe_criterion(sp[:3], sq[:3], up[:3], uq[:3])["delta"])


def test_criterion_table():
    sp, sq = _states(40, 8, 2.0, 0)
    up, uq = _states(40, 8, 0.0, 1)
    St = {0: torch.cat([sp, sq]), 1: torch.cat(_states(40, 8, 0.0, 2))}   # layer 1: no class signal
    S0 = {0: torch.cat([up, uq]), 1: torch.cat([up, uq])}
    tab = S.criterion_table(St, S0, 40, surface=dict(combined=0.9), n_boot=100, n_perm=20, n_random=20)
    assert tab["layer"].tolist() == [0, 1] and (tab["n_pairs"] == 40).all()
    assert bool(tab.loc[0, "counts"]) and not bool(tab.loc[1, "counts"])
    assert tab.loc[0, "surface_auc"] == 0.9 and bool(tab.loc[0, "beats_surface"]) and not bool(tab.loc[1, "beats_surface"])
    assert tab.loc[0, "ci_level_adj"] == pytest.approx(1 - 0.05 / 2)


def test_alignment_split_residualize_and_covariates(recs):
    from lsne.transplant import fit_alignment

    for n, h in ((50, 0.2), (14, 0.2), (300, 0.2)):
        hold, train = S.alignment_split(n, h, seed=0)
        a = fit_alignment(np.random.default_rng(0).normal(size=(n, 3)), np.random.default_rng(1).normal(size=(n, 2)),
                          holdout=h, seed=0)
        assert hold.tolist() == a.meta["idx_holdout"] and len(train) == a.meta["n_train"]
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(40, 2))
    X = Z @ rng.normal(size=(2, 5)) + 0.1 * rng.normal(size=(40, 5)) + 3.0
    tr = np.arange(30)
    R = S.residualize(X, Z, tr)
    Z1 = np.column_stack([np.ones(30), Z[tr]])
    assert np.abs(Z1.T @ R[tr]).max() < 1e-8                       # orthogonal to [1, Z] on the training rows
    assert np.abs(R).max() < 1.0 and R.shape == X.shape
    cov = S.length_covariates(recs)
    assert list(cov.columns) == ["n_tokens", "n_tokens_raw", "n_nodes", "n_value_calls"]
    for r, (_, row) in zip(recs, cov.iterrows()):
        f = labels.sharing_features(r)
        # the covariate counts what the tokens show (reverted / call_code value calls included), the label
        # feature only the effective value calls
        assert row["n_nodes"] == f["n_nodes"] and row["n_value_calls"] == sum(1 for n in r.nodes() if n.value > 0)
        assert row["n_value_calls"] >= f["n_value_calls"]
        assert row["n_tokens_raw"] == len(itr.serialize(r)) and row["n_tokens"] == min(512, row["n_tokens_raw"])
    Zm, names = S.covariate_matrix(cov)
    assert Zm.shape == (len(recs), 3) and tuple(names) == S.LENGTH_COVARIATES
    with pytest.raises(ValueError):
        S.length_covariates(recs[:2], texts=["a", "b"])


def test_alignment_controls_synthetic():
    rng = np.random.default_rng(0)
    N = 200
    Z = rng.normal(size=(N, 2))                                     # "length"
    content = rng.normal(size=(N, 3))                               # what only the trained source sees
    Tt = {0: Z @ rng.normal(size=(2, 6)) + content @ rng.normal(size=(3, 6)) + 0.05 * rng.normal(size=(N, 6))}
    Sx = {0: np.column_stack([Z, content]) @ rng.normal(size=(5, 8)) + 0.05 * rng.normal(size=(N, 8))}
    S0 = {0: Z @ rng.normal(size=(2, 8)) + 0.05 * rng.normal(size=(N, 8))}
    df = S.alignment_controls(Sx, S0, Tt, Z, alpha=1e-3, n_boot=200)
    r = df.iloc[0]
    assert len(df) == 1 and r["n_pairs"] == N and r["n_holdout"] == 40
    assert r["r2_raw"] > 0.95 and 0.2 < r["r2_raw_untrained"] < 0.9 and 0.2 < r["r2_length_only"] < 0.9
    assert r["r2_raw_untrained"] == pytest.approx(r["r2_length_only"], abs=0.05)
    assert r["r2_resid"] > 0.9 and r["r2_resid_untrained"] < 0.1           # no length-free signal in the untrained one
    assert r["gain_resid"] > 0.8 and r["gain_resid_ci_lo"] > 0.5 and r["gain_raw_ci_lo"] > 0
    assert r["gain_raw_ci_lo"] <= r["gain_raw"] <= r["gain_raw_ci_hi"] + 1e-9
    # a trained source that only knows length gains nothing after residualisation
    df2 = S.alignment_controls(S0, S0, Tt, Z, alpha=1e-3, n_boot=50)
    assert df2.iloc[0]["gain_resid"] == 0 and df2.iloc[0]["gain_resid_ci_lo"] == 0


def test_paired_items_from_rows(recs):
    rows = [{"tokens": itr.serialize(r), "text": "hello"} for r in recs[:2]]
    assert [t for _, t in S.paired_items(rows)] == ["hello", "hello"]
    rec_rows = [{"record": r.to_dict()} for r in recs[:2]]
    assert [t for _, t in S.paired_items(rec_rows)] == [text.render_text(r) for r in recs[:2]]
    with pytest.raises(ValueError):
        S.paired_items([{"tokens": ["[TX]"]}])


# ======================================================================================
# command line (serialize -> vocab -> train -> probe -> pair) on the fixture
# ======================================================================================
def test_cli_pipeline(tmp_path, recs, mb):
    itr_dir, voc, run, probe, pairs = (str(tmp_path / x) for x in ("itr", "vocab", "lm", "probe", "pairs.jsonl"))
    assert chain_cli.main(["serialize", "--cache", FIX, "--from", "1718000", "--to", "1921000", "--out", itr_dir,
                           "--records"]) == 0
    shards = chain_cli.list_shards(itr_dir)
    assert [os.path.basename(s) for s in shards] == ["itr-001718000.jsonl.gz", "itr-001920000.jsonl.gz"]
    rows = list(chain_cli.iter_rows(itr_dir))
    assert len(rows) == 15
    by = {r.tx_hash: r for r in recs}
    for row in rows:
        r = by[row["tx_hash"]]
        assert row["tokens"] == itr.serialize(r) and row["n_tokens"] == len(row["tokens"])
        assert row["era"] == labels.era_of(r.block_number) and row["text"] == text.render_text(r)
        assert row["features"]["fan_out_eth"] == labels.sharing_features(r)["fan_out_eth"]
        assert row["features"]["fan_definition"] == labels.FAN_DEFINITION and row["features"]["visible_len"] == 512
        assert row["features"]["fan_out_eth_visible"] <= row["features"]["fan_out_eth"]
        assert itr.TxRecord.from_dict(row["record"]).tx_hash == r.tx_hash
    mtime = os.path.getmtime(shards[0])
    assert chain_cli.main(["serialize", "--cache", FIX, "--from", "1718000", "--to", "1719000", "--out", itr_dir]) == 0
    assert os.path.getmtime(shards[0]) == mtime                          # resumable: existing shard skipped
    # re-serialising one chunk uncompressed replaces its gz shard; mixed formats are read once each
    assert chain_cli.main(["serialize", "--cache", FIX, "--from", "1718000", "--to", "1719000", "--out", itr_dir,
                           "--records", "--plain", "--overwrite"]) == 0
    assert [os.path.basename(s) for s in chain_cli.list_shards(itr_dir)] == ["itr-001718000.jsonl",
                                                                             "itr-001920000.jsonl.gz"]
    assert sorted(r["tx_hash"] for r in chain_cli.iter_rows(itr_dir)) == sorted(r["tx_hash"] for r in rows)

    assert chain_cli.main(["vocab", "--in", itr_dir, "--out", voc]) == 0
    assert {"vocab.json", "tokenizer.json", "oov.json"} <= set(os.listdir(voc))
    assert json.load(open(os.path.join(voc, "oov.json")))["oov"]["rate_open"] == 0
    assert V.TraceVocab.load(voc).meta["params"]["serialize_kw"]["max_args"] == 4     # shard options carried along

    assert chain_cli.main(["train", "--in", itr_dir, "--vocab", voc, "--size", "tiny", "--out", run, "--steps", "4",
                           "--block-size", "32", "--batch-size", "4", "--eval-every", "2", "--save-every", "2",
                           "--warmup", "1", "--val-frac", "0.3", "--device", "cpu"]) == 0
    man = json.load(open(os.path.join(run, "manifest.json")))
    assert man["corpus"]["n_tx_train"] + man["corpus"]["n_tx_val"] == 15 and man["corpus"]["n_tx_truncated"] == 1

    assert chain_cli.main(["probe", "--model", run, "--in", itr_dir, "--kind", "dao", "--layer", "all",
                           "--out", probe, "--gas-bins", "1", "--device", "cpu"]) == 0
    sm = pd.read_csv(os.path.join(probe, "probe_summary.csv"))
    assert sm["layer"].tolist() == [0, 1, 2] and (sm["n_pos"] >= 2).all()
    saved = torch.load(os.path.join(probe, "sharing_directions.pt"), weights_only=False)
    assert saved["kind"] == "dao" and saved["layers"][2]["vec"].shape == (64,)
    assert len(pd.read_csv(os.path.join(probe, "match_table.csv"))) == saved["n_pairs"]
    # the probe specification and the model fingerprint travel with the direction
    assert saved["k"] == 5 and saved["max_len"] == 512 and saved["split"] == "all"
    assert saved["model_fingerprint"] == S.model_fingerprint(run) and saved["code_sha256"] == chain_cli.code_fingerprint()
    rep0 = json.load(open(os.path.join(probe, "probe_report.json")))
    ls = rep0["matching"]["lm_split"]
    assert ls["val_frac"] == 0.3 and ls["n_items"] == 2 * saved["n_pairs"]
    assert ls["n_items_lm_val"] + ls["n_items_lm_train"] == ls["n_items"]
    assert rep0["matching"]["n_pairs_before_dedup"] - rep0["matching"]["n_pairs_dropped_duplicates"] == saved["n_pairs"]
    assert json.load(open(os.path.join(probe, "manifest.json")))["code_sha256"] == chain_cli.code_fingerprint()
    # held-out split only: every probe item is in the LM's validation split
    probe_h = str(tmp_path / "probe_heldout")
    chain_cli.main(["probe", "--model", run, "--in", itr_dir, "--kind", "dao", "--gas-bins", "1", "--split", "heldout",
                    "--untrained", "none", "--out", probe_h, "--device", "cpu"])
    mt_h = pd.read_csv(os.path.join(probe_h, "match_table.csv"))
    assert all(S.is_lm_val(h, 0.3) for h in list(mt_h["pos_tx"]) + list(mt_h["neg_tx"]))

    assert chain_cli.main(["pair", "--in", itr_dir, "--n", "5", "--out", pairs]) == 0
    loaded = S.load_pairs(pairs)
    assert len(loaded) == 5 and all(isinstance(t, str) and t for _, t in loaded)
    # transactions longer than the window are never paired (their text covers unseen nodes)
    all_pairs = str(tmp_path / "all_pairs.jsonl")
    assert chain_cli.main(["pair", "--in", itr_dir, "--n", "100", "--out", all_pairs]) == 0
    got, hs = S.load_pairs(all_pairs, with_hashes=True)
    assert len(got) == 14 and recs[0].tx_hash not in hs and all(len(t) <= 512 for t, _ in got)     # the drain: 1 409
    assert chain_cli.main(["pair", "--in", itr_dir, "--n", "100", "--max-len", "0", "--out", all_pairs]) == 0
    assert len(S.load_pairs(all_pairs)) == 15
    src = S.TraceLMSource(run, device="cpu")
    assert src.encode(loaded[0][0]).shape[0] == 1
    assert src.serialize_kw["include_reads"] is True
    row = next(r for r in rows if r["tx_hash"] == recs[1].tx_hash)
    assert torch.equal(src.encode(recs[1]), src.encode(row))           # record == its shard tokens
    assert chain_cli.main(["pair", "--in", itr_dir, "--n", "4", "--out", pairs, "--balance", "--rerender",
                           "--address-style", "named"]) == 0

    # default probe: the untrained twin and the criterion are part of the output
    crit = pd.read_csv(os.path.join(probe, "criterion.csv"))
    assert crit["layer"].tolist() == [0, 1, 2]
    rep = json.load(open(os.path.join(probe, "probe_report.json")))
    assert rep["untrained"] == "auto(seed=0)" and rep["settings"]["match"] == "gas"

    # round 2: length-matched probe with the DAO identities masked, criterion trained - untrained
    probe2 = str(tmp_path / "probe2")
    assert chain_cli.main(["probe", "--model", run, "--in", itr_dir, "--kind", "dao", "--match", "length",
                           "--length-ratio", "8", "--mask", "dao", "--out", probe2, "--device", "cpu",
                           "--n-boot", "50", "--n-perm", "20", "--n-random", "20"]) == 0
    crit = pd.read_csv(os.path.join(probe2, "criterion.csv"))
    assert crit["layer"].tolist() == [0, 1, 2]
    assert {"auc_trained", "auc_untrained", "delta", "delta_ci_lo", "delta_ci_hi", "delta_ci_eval_lo", "perm_auc_p95",
            "perm_p", "random_dir_auc_p95", "surface_auc", "beats_surface", "counts", "counts_adj"} <= set(crit.columns)
    assert np.allclose(crit["delta"], crit["auc_trained"] - crit["auc_untrained"], equal_nan=True)
    assert crit["n_pairs"].iloc[0] == 3 and crit["delta"].isna().all()   # < 4 pairs: no estimate, never counts
    assert not crit["counts"].any()
    rep = json.load(open(os.path.join(probe2, "probe_report.json")))
    assert rep["settings"]["mask"] == "dao" and rep["matching"]["matching"] == "era+log_length"
    assert rep["matching"]["n_pairs"] == len(crit.iloc[0:0]) + int(crit["n_pairs"].iloc[0])
    assert "length_smd" in rep["matching"] and "n_tokens" in rep["matching"]["balance"]
    assert "n_a_oov" in rep["surface_baseline"] and rep["verdict"]["rule"].startswith("delta CI lower bound > 0")
    mt = pd.read_csv(os.path.join(probe2, "match_table.csv"))
    assert (np.abs(np.log(mt["pos_n_tokens"] / mt["neg_n_tokens"])) <= np.log(8) + 1e-9).all()
    saved = torch.load(os.path.join(probe2, "sharing_directions.pt"), weights_only=False)
    assert saved["mask"] == "dao" and saved["match"] == "length"
    # an explicit control directory (checked for vocabulary / shape) and no control at all
    probe3 = str(tmp_path / "probe3")
    assert chain_cli.main(["probe", "--model", run, "--in", itr_dir, "--kind", "dao", "--gas-bins", "1",
                           "--untrained", "none", "--out", probe3, "--device", "cpu"]) == 0
    assert not os.path.exists(os.path.join(probe3, "criterion.csv"))
    assert "criterion" not in json.load(open(os.path.join(probe3, "probe_report.json")))
    assert chain_cli.main(["probe", "--model", run, "--in", itr_dir, "--kind", "dao", "--gas-bins", "1",
                           "--untrained", run, "--n-boot", "20", "--n-perm", "5", "--n-random", "5",
                           "--out", probe3, "--device", "cpu"]) == 0
    same = pd.read_csv(os.path.join(probe3, "criterion.csv"))
    assert ((same["delta"] == 0) | same["delta"].isna()).all() and not same["counts"].any()   # never counts
    # too few matchable pairs -> exit code 1, but the match table and the report are written
    probe4 = str(tmp_path / "probe4")
    assert chain_cli.main(["probe", "--model", run, "--in", itr_dir, "--kind", "fanout", "--match", "length",
                           "--out", probe4, "--device", "cpu"]) == 1
    assert json.load(open(os.path.join(probe4, "probe_report.json")))["verdict"]["n_pairs"] == 0

    # the alignment with length / untrained controls against pythia-14m
    pairs14, al = str(tmp_path / "pairs14.jsonl"), str(tmp_path / "align")
    assert chain_cli.main(["pair", "--in", itr_dir, "--n", "14", "--out", pairs14]) == 0
    assert chain_cli.main(["align", "--model", run, "--pairs", pairs14, "--target",
                           f"{mb.spec.name}@{mb.spec.revision}", "--alphas", "1,100", "--n-boot", "20",
                           "--layers-src", "0,2", "--layers-tgt", "0,3", "--out", al, "--device", "cpu"]) == 0
    res = pd.read_csv(os.path.join(al, "align_controls.csv"))
    assert len(res) == 2 * 2 * 2 * 2                    # 2 covariate sets x 2 alphas x 2 x 2 layers
    assert set(res["covariates"]) == {"len3", "len3+text"} and set(res["alpha"]) == {1.0, 100.0}
    for c in ("r2_raw", "r2_raw_untrained", "r2_resid", "r2_resid_untrained", "r2_length_only"):
        assert np.isfinite(res[c]).all(), c
    assert np.allclose(res["gain_resid"], res["r2_resid"] - res["r2_resid_untrained"])
    summ = json.load(open(os.path.join(al, "align_summary.json")))
    assert summ["n_pairs"] == 14 and len(summ["summary"]) == 4 and summ["untrained"] == "auto(seed=0)"
    assert summ["lm_split"]["val_frac"] == 0.3 and summ["lm_split"]["n_holdout"] == 3
    assert summ["code_sha256"] == chain_cli.code_fingerprint()
    # 3 hold-out pairs: no bootstrap CI (NaN) instead of absurd values
    assert res[["gain_raw_ci_lo", "gain_resid_ci_lo"]].isna().all().all()
    assert summ["covariates"]["len3+text"][-1] == "log_n_text_tokens"
    cov = pd.read_csv(os.path.join(al, "length_covariates.csv"))
    assert len(cov) == 14 and (cov["n_text_tokens"] > 0).all()


def test_dedup_pairs_and_degenerate_bootstrap():
    """Repeated visible sequences are not independent pairs; a tiny hold-out gives no CI."""
    pos = [["a", "b"], ["c"], ["a", "b"], ["d"], ["e"], ["x"]]
    neg = [["n1"], ["n2"], ["n3"], ["n1"], ["e"], ["y"]]
    # pair 2 repeats pair 0's positive, pair 3 pair 0's negative, pair 4 has pos == neg
    assert S.dedup_pairs(pos, neg) == [0, 1, 5]
    rng = np.random.default_rng(0)
    Y = rng.normal(size=(3, 4))
    g, r1, r0 = S._boot_r2_gain(Y, Y + 0.1, Y + 0.2, 100, np.random.default_rng(1))
    assert g.size == 0                                           # < MIN_HOLDOUT_BOOT rows: no bootstrap
    Y = rng.normal(size=(40, 4))
    g, r1, r0 = S._boot_r2_gain(Y, Y + 0.1 * rng.normal(size=Y.shape), Y + rng.normal(size=Y.shape), 200,
                                np.random.default_rng(1))
    assert g.shape == (200,) and np.isfinite(g).all() and (g > 0).mean() > 0.9
    assert S.split_filter(["0x" + "0" * 63 + "1", "0x" + "f" * 64], "heldout", 0.1) == [True, False]
    assert S.split_filter(["0x" + "0" * 63 + "1", "0x" + "f" * 64], "train", 0.1) == [False, True]
    with pytest.raises(ValueError):
        S.split_filter(["0x1"], "heldout", None)
