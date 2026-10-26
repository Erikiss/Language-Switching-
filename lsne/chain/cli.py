"""Command line for the trace source model (docs/TRACE_SOURCE_SPEC.md §8).

    python -m lsne.chain.cli download  --from 1428000 --to 2100000 --every 1 \\
        --tables traces,transaction,native_transfers,erc20_transfers,logs,storage_diffs --cache cache/xatu
    python -m lsne.chain.cli serialize --cache cache/xatu --from 1428000 --to 2100000 --out data/traces/itr
    python -m lsne.chain.cli vocab     --in data/traces/itr --out data/traces/vocab
    python -m lsne.chain.cli train     --in data/traces/itr --vocab data/traces/vocab --size blockgpt \\
        --out runs/trace_lm --steps 20000
    python -m lsne.chain.cli probe     --model runs/trace_lm --in data/traces/itr --kind fanout|dao \\
        --layer all --out runs/trace_probe
    python -m lsne.chain.cli pair      --in data/traces/itr --n 2000 --out runs/trace_pairs.jsonl
    python -m lsne.chain.cli align     --model runs/trace_lm --pairs runs/trace_pairs.jsonl \\
        --target EleutherAI/pythia-14m@step98000 --out runs/trace_align
    python -m lsne.chain.cli bench     --sizes tiny,small --block-size 512 --batch-size 8

Controlled probes (round 2, docs/TRACE_SOURCE.md "Kontrollierte Sonden"):

    python -m lsne.chain.cli probe --model runs/trace_lm --in data/traces/itr --kind fanout \\
        --match length --length-ratio 1.25 --k 5 --out runs/trace_probe_fanout_len
    python -m lsne.chain.cli probe --model runs/trace_lm --in data/traces/itr --kind dao \\
        --match length --mask dao --out runs/trace_probe_dao_masked

``probe`` always writes ``match_table.csv``, ``probe_summary.csv``, ``sharing_directions.pt``
and ``probe_report.json`` (matching balance as standardized mean differences, surface
baseline); unless ``--untrained none`` it also encodes the same (masked) sequences with the
untrained twin (same config, vocabulary and initialisation seed) and writes
``criterion.csv``: split-half AUC trained / untrained, their difference with a bootstrap CI
over matched pairs, permutation and random-direction nulls; a layer *counts* only if the
CI of the difference lies above 0.  ``align`` fits the ridge alignment raw, after
residualising both sides on length covariates, for the length covariates alone and for
the untrained source (``align_controls.csv``, ``align_summary.json``).

ITR shards: one gzipped JSON-lines file per 1 000-block chunk, ``<out>/itr-<start>.jsonl.gz``
(``--plain``: uncompressed ``.jsonl``; about 16 MB per DAO-era chunk uncompressed, 1.6 MB
gzipped), one line per transaction with ``tx_hash, block_number, tx_index, era, label, n_tokens, tokens`` (raw
ITR tokens, no vocabulary applied), ``features`` (:func:`lsne.chain.labels.sharing_features`),
``text`` (:func:`lsne.chain.text.render_text`) and, with ``--records``, the full ``record``
(``TxRecord.to_dict``).  Every step is resumable: existing shards / chunks are skipped and
``train`` continues from the newest checkpoint.  Data: ethPandaOps Xatu, CC BY 4.0.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from typing import Iterator, Optional, Sequence

import numpy as np
import pandas as pd

DEFAULT_TABLES = "traces,transaction,native_transfers,erc20_transfers,logs,storage_diffs"
SHARD_RE = re.compile(r"^itr-(\d+)\.jsonl(\.gz)?$")


def _say(msg: str) -> None:
    print(msg, flush=True)


# ======================================================================================
# ITR shards
# ======================================================================================
def shard_path(out_dir: str, start: int, gz: bool = True) -> str:
    return os.path.join(out_dir, f"itr-{int(start):09d}.jsonl" + (".gz" if gz else ""))


def _open_text(fn: str, mode: str = "rt"):
    if fn.endswith(".gz"):
        return gzip.open(fn, mode, encoding="utf-8", compresslevel=1) if "w" in mode else \
            gzip.open(fn, mode, encoding="utf-8")
    return open(fn, mode.replace("t", ""), encoding="utf-8")


def list_shards(path: str) -> list:
    """Shard files below ``path`` (a directory of ``itr-*.jsonl[.gz]`` or a single file), sorted by
    chunk start; when a chunk exists in both formats the gzipped one is used."""
    if os.path.isfile(path):
        return [path]
    if not os.path.isdir(path):
        raise FileNotFoundError(f"no ITR shards at {path}")
    by_start: dict = {}
    for f in os.listdir(path):
        m = SHARD_RE.match(f)
        if m and (int(m.group(1)) not in by_start or f.endswith(".gz")):
            by_start[int(m.group(1))] = os.path.join(path, f)
    return [by_start[s] for s in sorted(by_start)]


def iter_rows(path: str, limit: Optional[int] = None, hashes: Optional[set] = None) -> Iterator[dict]:
    """Rows of every shard (``limit``: stop after that many; ``hashes``: only these tx hashes)."""
    n = 0
    for fn in list_shards(path):
        with _open_text(fn) as fh:
            for line in fh:
                if not line.strip():
                    continue
                if hashes is not None:
                    m = re.search(r'"tx_hash":\s*"(0x[0-9a-f]+)"', line)
                    if m is None or m.group(1) not in hashes:
                        continue
                yield json.loads(line)
                n += 1
                if limit is not None and n >= int(limit):
                    return


def _serialize_chunk(job: tuple) -> dict:
    """Worker: one chunk -> one shard (atomic write)."""
    from .itr import load_records, serialize
    from .labels import sharing_features, sharing_label
    from .text import render_text

    (start, cache, out_dir, block_from, block_to, overwrite, with_records, ser_kw, text_kw, label_k, gz) = job
    path = shard_path(out_dir, start, gz)
    if not overwrite:
        for p in (shard_path(out_dir, start, True), shard_path(out_dir, start, False)):
            if os.path.exists(p):
                return dict(start=start, path=p, n_tx=-1, n_tokens=-1, skipped=True, error=None)
    try:
        recs = [r for r in load_records(start, cache) if block_from <= r.block_number < block_to]
        tmp = f"{path}.tmp.{os.getpid()}" + (".gz" if gz else "")
        n_tok = 0
        with _open_text(tmp, "wt") as fh:
            for r in recs:
                toks = serialize(r, None, **ser_kw)
                feat = sharing_features(r)
                row = dict(tx_hash=r.tx_hash, block_number=r.block_number, tx_index=r.tx_index, era=feat["era"],
                           label=sharing_label(feat, k=label_k), n_tokens=len(toks), tokens=toks, features=feat,
                           text=render_text(r, **text_kw))
                if with_records:
                    row["record"] = r.to_dict()
                n_tok += len(toks)
                fh.write(json.dumps(row) + "\n")
        os.replace(tmp, path)
        if overwrite:                       # a re-serialised chunk must not leave the other format behind
            other = shard_path(out_dir, start, not gz)
            if os.path.exists(other):
                os.remove(other)
        return dict(start=start, path=path, n_tx=len(recs), n_tokens=n_tok, skipped=False, error=None)
    except Exception as exc:  # noqa: BLE001 - reported per chunk
        for p in [f for f in os.listdir(out_dir) if f.startswith(os.path.basename(path) + ".tmp.")]:
            try:
                os.remove(os.path.join(out_dir, p))
            except OSError:
                pass
        return dict(start=start, path=path, n_tx=0, n_tokens=0, skipped=False, error=f"{type(exc).__name__}: {exc}")


def serialize_chunks(cache: str, block_from: int, block_to: int, out_dir: str, overwrite: bool = False,
                     with_records: bool = False, serialize_kw: Optional[dict] = None,
                     text_kw: Optional[dict] = None, label_k: int = 5, workers: int = 1, gz: bool = True,
                     log_fn=_say) -> pd.DataFrame:
    """Write one ITR shard per cached traces chunk covering ``[block_from, block_to)``
    (gzipped JSON lines unless ``gz=False``; existing shards of either format are skipped)."""
    from .xatu import cached_starts, chunk_starts

    wanted = set(chunk_starts(block_from, block_to))
    starts = [s for s in cached_starts(cache, "traces") if s in wanted]
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(s, cache, out_dir, int(block_from), int(block_to), bool(overwrite), bool(with_records),
             dict(serialize_kw or {}), dict(text_kw or {}), int(label_k), bool(gz)) for s in starts]
    rows = []
    if int(workers) > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            for res in ex.map(_serialize_chunk, jobs):
                rows.append(res)
                log_fn(f"serialize {res['start']}: " + ("skipped (exists)" if res["skipped"] else
                                                        res["error"] or f"{res['n_tx']} tx, {res['n_tokens']} tokens"))
    else:
        for j in jobs:
            res = _serialize_chunk(j)
            rows.append(res)
            log_fn(f"serialize {res['start']}: " + ("skipped (exists)" if res["skipped"] else
                                                    res["error"] or f"{res['n_tx']} tx, {res['n_tokens']} tokens"))
    return pd.DataFrame(rows, columns=["start", "path", "n_tx", "n_tokens", "skipped", "error"])


# ======================================================================================
# commands
# ======================================================================================
def cmd_download(a) -> int:
    from .xatu import download_chunks

    df = download_chunks(a.tables, a.block_from, a.block_to, cache_dir=a.cache, every=a.every,
                         max_workers=a.workers, retries=a.retries, log_fn=_say if a.verbose else None)
    n_ok, n_cached = int(df["ok"].sum()), int(df["cached"].sum()) if len(df) else 0
    _say(f"download: {len(df)} chunk files, {n_ok} ok ({n_cached} already cached), "
         f"{len(df) - n_ok} failed, {int(df['bytes'].sum()) / 1e6:.1f} MB")
    os.makedirs(a.cache, exist_ok=True)
    df.to_csv(os.path.join(a.cache, "download_log.csv"), index=False)
    for _, r in df[~df["ok"]].iterrows():
        _say(f"  FAILED {r['table']} {r['start']}: {r['error']}")
    return 0 if n_ok == len(df) else 2


def cmd_serialize(a) -> int:
    ser_kw = dict(max_args=a.max_args, include_state=not a.no_state, include_reads=not a.no_reads,
                  include_logs=not a.no_logs, max_depth_token=a.max_depth_token)
    text_kw = dict(max_nodes=a.text_max_nodes, address_style=a.address_style)
    df = serialize_chunks(a.cache, a.block_from, a.block_to, a.out, overwrite=a.overwrite, with_records=a.records,
                          serialize_kw=ser_kw, text_kw=text_kw, label_k=a.k, workers=a.workers, gz=not a.plain)
    if not len(df):
        _say(f"serialize: no cached traces chunks in [{a.block_from}, {a.block_to}) under {a.cache}")
        return 1
    with open(os.path.join(a.out, "serialize_params.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(serialize_kw=ser_kw, text_kw=text_kw, label_k=a.k, cache=a.cache,
                       block_from=a.block_from, block_to=a.block_to, data_source="ethPandaOps Xatu (CC BY 4.0)"),
                  fh, indent=1)
    new = df[~df["skipped"]]
    bad = df[df["error"].notna()]
    _say(f"serialize: {len(df)} chunks ({int(df['skipped'].sum())} skipped), {int(new['n_tx'].sum())} tx, "
         f"{int(new['n_tokens'].sum())} tokens written, {len(bad)} errors")
    return 0 if bad.empty else 2


def cmd_vocab(a) -> int:
    from .vocab import build_vocab, make_tokenizer, oov_rates

    t0 = time.time()
    # the shards' serialisation options travel with the vocabulary, so that TraceLMSource
    # serialises records exactly like the training corpus
    ser_kw = {}
    pfile = os.path.join(a.inp, "serialize_params.json") if os.path.isdir(a.inp) else None
    if pfile and os.path.exists(pfile):
        with open(pfile, encoding="utf-8") as fh:
            ser_kw = json.load(fh).get("serialize_kw", {})
    vocab = build_vocab((r["tokens"] for r in iter_rows(a.inp, limit=a.limit)), n_addresses=a.n_addresses,
                        n_selectors=a.n_selectors, n_topics=a.n_topics, min_count=a.min_count,
                        serialize_kw=ser_kw)
    make_tokenizer(vocab, a.out, model_max_length=a.max_len)
    oov = oov_rates(vocab.map_tokens(r["tokens"]) for r in iter_rows(a.inp, limit=a.limit))
    with open(os.path.join(a.out, "oov.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(oov=oov, meta=vocab.meta), fh, indent=1)
    _say(f"vocab: {len(vocab)} tokens ({vocab.meta['n_addresses']} addresses, {vocab.meta['n_selectors']} "
         f"selectors, {vocab.meta['n_topics']} topics) from {vocab.meta['n_items']} tx in {time.time() - t0:.0f}s; "
         f"OOV address {oov['rate_A']:.3f}, selector {oov['rate_F']:.3f}, topic {oov['rate_T']:.3f} -> {a.out}")
    return 0


def _is_val(tx_hash: str, frac: float) -> bool:
    return frac > 0 and (int(tx_hash[-8:], 16) / 2 ** 32) < frac


def _merge_oov(parts: list) -> dict:
    keys = ("n_lists", "n_tokens", "n_unk", "n_A", "n_A_oov", "n_F", "n_F_oov", "n_T", "n_T_oov")
    tot = {k: sum(int(p.get(k, 0)) for p in parts) for k in keys}
    nan = float("nan")
    tot["rate_unk"] = tot["n_unk"] / tot["n_tokens"] if tot["n_tokens"] else nan
    for c in "AFT":
        tot[f"rate_{c}"] = tot[f"n_{c}_oov"] / tot[f"n_{c}"] if tot[f"n_{c}"] else nan
    n_open = tot["n_A"] + tot["n_F"] + tot["n_T"]
    tot["rate_open"] = (tot["n_A_oov"] + tot["n_F_oov"] + tot["n_T_oov"]) / n_open if n_open else nan
    return tot


def corpus_blocks(in_path: str, vocab, block_size: int = 512, max_tx_len: Optional[int] = 512,
                  val_frac: float = 0.01, limit: Optional[int] = None, log_fn=_say) -> tuple:
    """Encode the shards with ``vocab``, split by transaction hash (``val_frac``), pack each
    split (``[EOS]`` separators) -> ``(train_blocks int32 [N, T], val_blocks, stats)``."""
    from .train import blocks_from_stream, concat_with_eos
    from .vocab import oov_rates

    import torch

    tr_parts, va_parts, oov_parts = [], [], []
    n_tx = [0, 0]
    n_trunc = 0
    seen = 0
    for fn in list_shards(in_path):
        tr, va, mapped = [], [], []
        for r in iter_rows(fn):
            toks = vocab.map_tokens(r["tokens"])
            mapped.append(toks)
            if max_tx_len is not None and len(toks) > int(max_tx_len):
                toks = toks[: int(max_tx_len)]
                n_trunc += 1
            ids = vocab.ids(toks)
            if _is_val(r["tx_hash"], val_frac):
                va.append(ids)
                n_tx[1] += 1
            else:
                tr.append(ids)
                n_tx[0] += 1
            seen += 1
            if limit is not None and seen >= int(limit):
                break
        oov_parts.append(oov_rates(mapped))
        tr_parts.append(concat_with_eos(tr, vocab.eos_id))
        va_parts.append(concat_with_eos(va, vocab.eos_id))
        if limit is not None and seen >= int(limit):
            break
    tr_stream = np.concatenate(tr_parts) if tr_parts else np.zeros(0, np.int32)
    va_stream = np.concatenate(va_parts) if va_parts else np.zeros(0, np.int32)
    train_blocks = blocks_from_stream(tr_stream, block_size, dtype=torch.int32)
    val_blocks = blocks_from_stream(va_stream, block_size, dtype=torch.int32)
    stats = dict(n_tx_train=n_tx[0], n_tx_val=n_tx[1], n_tx_truncated=n_trunc, max_tx_len=max_tx_len,
                 tokens_train=int(tr_stream.size), tokens_val=int(va_stream.size),
                 n_train_blocks=int(len(train_blocks)), n_val_blocks=int(len(val_blocks)),
                 block_size=int(block_size), val_frac=float(val_frac), oov=_merge_oov(oov_parts))
    log_fn(f"corpus: {n_tx[0]} train / {n_tx[1]} val tx ({n_trunc} truncated to {max_tx_len}), "
           f"{tr_stream.size:,} / {va_stream.size:,} tokens -> {len(train_blocks)} / {len(val_blocks)} blocks of "
           f"{block_size}; OOV address rate {stats['oov']['rate_A']:.3f}")
    return train_blocks, val_blocks, stats


def cmd_train(a) -> int:
    from .train import make_model, param_counts, train
    from .vocab import TraceVocab, load_tokenizer

    vocab = TraceVocab.load(a.vocab)
    tok = load_tokenizer(a.vocab)
    tr, va, stats = corpus_blocks(a.inp, vocab, a.block_size, None if a.max_tx_len <= 0 else a.max_tx_len,
                                  a.val_frac, a.limit)
    if len(tr) == 0:
        _say("train: corpus shorter than one block")
        return 1
    model = make_model(len(vocab), a.size, max_position_embeddings=max(512, a.block_size), seed=a.seed)
    pc = param_counts(model)
    _say(f"train: {a.size} GPT-NeoX, {pc['total']:,} parameters ({pc['non_embedding']:,} non-embedding), "
         f"{a.steps} steps x {a.batch_size} x {a.block_size} tokens -> {a.out}")
    bf16 = {"auto": None, "on": True, "off": False}[a.bf16]
    df = train(model, tr, va if len(va) else None, a.out, lr=a.lr, batch_size=a.batch_size, steps=a.steps,
               warmup=a.warmup, weight_decay=a.weight_decay, grad_clip=a.grad_clip, bf16=bf16,
               eval_every=a.eval_every, save_every=a.save_every, seed=a.seed, tokenizer=tok, vocab=vocab,
               device=a.device, max_eval_blocks=a.eval_blocks, grad_accum=a.grad_accum,
               keep_checkpoints=a.keep_checkpoints, manifest_extra=dict(size=a.size, corpus=stats, data=a.inp))
    _say(f"train: done, val loss {df['val_loss'].iloc[0]:.3f} -> {df['val_loss'].iloc[-1]:.3f}")
    return 0


def _parse_layers(spec: str, n_layers: int) -> list:
    if spec == "all":
        return list(range(n_layers + 1))
    out = [int(x) for x in spec.split(",") if x.strip()]
    bad = [l for l in out if not 0 <= l <= n_layers]
    if bad:
        raise SystemExit(f"layers {bad} outside 0..{n_layers}")
    return out


def _features_frame(in_path: str, limit: Optional[int]) -> pd.DataFrame:
    feats = []
    for r in iter_rows(in_path, limit=limit):
        f = dict(r["features"])
        f["tx_hash"] = r["tx_hash"]
        f["label"] = r.get("label")
        f["n_tokens"] = int(r["n_tokens"]) if r.get("n_tokens") is not None else len(r.get("tokens") or [])
        feats.append(f)
    return pd.DataFrame(feats)


def _jsonable(x):
    """numpy / pandas scalars and containers -> plain JSON types (NaN -> None)."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


def _control_source(src, spec: str, init_seed: Optional[int], model_dir: str, device: str, max_len: int):
    """The untrained control of the same architecture and vocabulary: ``"auto"`` = the
    in-memory twin (:func:`lsne.chain.source.untrained_twin`, seed from the model manifest
    unless ``init_seed``), else a model directory (checked for the same vocabulary / shape)."""
    from .source import TraceLMSource, init_seed_of, untrained_twin

    if spec == "none":
        return None, None
    if spec == "auto":
        seed = init_seed_of(model_dir) if init_seed is None else int(init_seed)
        return untrained_twin(src, seed=seed), f"auto(seed={seed})"
    src0 = TraceLMSource(spec, device=device, max_len=max_len)
    if src0.vocab.tokens != src.vocab.tokens or (src0.d_model, src0.n_layers) != (src.d_model, src.n_layers):
        raise SystemExit(f"untrained control {spec} differs from {model_dir} in vocabulary or shape")
    return src0, spec


def cmd_probe(a) -> int:
    import torch

    from ..io import write_manifest
    from .labels import SURFACE_FEATURES_OOV, balance_table, probe_sets, standardized_mean_difference, \
        surface_baseline_auc
    from .source import TraceLMSource, criterion_table, direction_from_states, masked_items, pooled_states

    df = _features_frame(a.inp, a.limit)
    if df.empty:
        _say("probe: no transactions")
        return 1
    recs = [SimpleNamespace(tx_hash=h) for h in df["tx_hash"]]
    eras = a.eras.split(",") if a.eras else None
    pos, neg, mt = probe_sets(recs, df, a.kind, a.n, seed=a.seed, k=a.k, eras=eras, gas_bins=a.gas_bins,
                              match=a.match, length_ratio=a.length_ratio,
                              length_cap=a.length_cap if a.length_cap and a.length_cap > 0 else None)
    os.makedirs(a.out, exist_ok=True)
    mt.to_csv(os.path.join(a.out, "match_table.csv"), index=False)
    settings = dict(kind=a.kind, k=a.k, match=a.match, length_ratio=a.length_ratio if a.match == "length" else None,
                    gas_bins=a.gas_bins if a.match == "gas" else None, mask=a.mask, readout=a.readout, n_max=a.n,
                    seed=a.seed, eras=eras, model=a.model, data=a.inp, max_len=a.max_len)
    matching = {k: v for k, v in mt.attrs.items()}
    if len(pos) < 2:
        with open(os.path.join(a.out, "probe_report.json"), "w", encoding="utf-8") as fh:
            json.dump(_jsonable(dict(settings=settings, matching=matching, verdict=dict(n_pairs=len(pos)))), fh, indent=1)
        _say(f"probe: only {len(pos)} matched pairs for kind={a.kind} (match={a.match}, pool "
             f"{matching.get('pool_pos')} pos / {matching.get('pool_neg')} neg); need >= 2")
        return 1
    want = {r.tx_hash for r in pos + neg}
    rows = {r["tx_hash"]: r for r in iter_rows(a.inp, hashes=want)}
    src = TraceLMSource(a.model, device=a.device, max_len=a.max_len)
    layers = _parse_layers(a.layer, src.n_layers)
    # vocabulary-mapped, truncated, identity-masked token lists (the same masking for both sets)
    items = masked_items(src, [rows[r.tx_hash] for r in pos] + [rows[r.tx_hash] for r in neg], a.mask)
    n = len(pos)
    t0 = time.time()
    S = pooled_states(src, items, layers, a.readout)
    t_enc = time.time() - t0
    summary, saved = [], {}
    for l in layers:
        d = direction_from_states(S[l][:n], S[l][n:], l, src.n_layers, denoise_pcs=a.denoise_pcs,
                                  name=f"sharing_{a.kind}", seed=a.seed,
                                  meta=dict(kind=a.kind, readout=a.readout, source=src.name, mask=a.mask,
                                            match=a.match))
        m = {k: v for k, v in d.meta.items() if k not in ("vec_denoised", "layer")}
        summary.append(dict(layer=l, site=d.site.key, **m))
        saved[l] = dict(vec=d.vec, vec_denoised=d.meta.get("vec_denoised"), site=d.site.key)
    sm = pd.DataFrame(summary)
    sm.to_csv(os.path.join(a.out, "probe_summary.csv"), index=False)
    torch.save(dict(layers=saved, kind=a.kind, readout=a.readout, model=a.model, n_pairs=n, d_model=src.d_model,
                    n_layers=src.n_layers, mask=a.mask, match=a.match), os.path.join(a.out, "sharing_directions.pt"))
    # balance of the visible (truncated, masked) sequences and the surface baseline on them
    vis_p, vis_q = [len(t) for t in items[:n]], [len(t) for t in items[n:]]
    matching["visible_length_smd"] = standardized_mean_difference(np.log(vis_p), np.log(vis_q))
    matching["visible_n_tokens_mean"] = dict(pos=float(np.mean(vis_p)), neg=float(np.mean(vis_q)))
    surf = surface_baseline_auc(items[:n], items[n:], seed=a.seed, max_len=a.max_len, features=SURFACE_FEATURES_OOV)
    report = dict(settings=settings, matching=matching, surface_baseline=surf, probe_summary=sm.to_dict("records"))
    crit = None
    src0, src0_name = _control_source(src, a.untrained, a.init_seed, a.model, a.device, a.max_len)
    if src0 is not None:
        t1 = time.time()
        S0 = pooled_states(src0, items, layers, a.readout)
        t_enc0 = time.time() - t1
        crit = criterion_table(S, S0, n, layers, surface=surf, seed=a.seed, n_boot=a.n_boot, n_perm=a.n_perm,
                               n_random=a.n_random, ci=a.ci)
        crit.to_csv(os.path.join(a.out, "criterion.csv"), index=False)
        report.update(untrained=src0_name, criterion=crit.to_dict("records"),
                      verdict=dict(n_pairs=n, layers_counting=[int(l) for l in crit.loc[crit["counts"], "layer"]],
                                   layers_counting_bonferroni=[int(l) for l in crit.loc[crit.get("counts_adj", crit["counts"]).astype(bool), "layer"]],
                                   any_counts=bool(crit["counts"].any()),
                                   rule="delta CI lower bound > 0 and trained AUC > permutation p95"),
                      seconds=dict(encode_trained=round(t_enc, 1), encode_untrained=round(t_enc0, 1),
                                   controls=round(time.time() - t1 - t_enc0, 1)))
    with open(os.path.join(a.out, "probe_report.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(report), fh, indent=1)
    write_manifest(a.out, kind="trace_probe", probe_kind=a.kind, model=a.model, data=a.inp, n_pairs=n,
                   layers=layers, readout=a.readout, denoise_pcs=a.denoise_pcs, k=a.k, seed=a.seed, match=a.match,
                   mask=a.mask, untrained=src0_name, seconds=round(time.time() - t0, 1))
    bal = matching.get("balance", {})
    _say(f"probe: {n} matched pairs ({a.kind}, match={matching.get('matching')}, mask={a.mask}; pool "
         f"{matching.get('pool_pos')} pos / {matching.get('pool_neg')} neg, {matching.get('n_matched_total')} matchable); "
         f"SMD log length {matching.get('length_smd', float('nan')):.3f}, log gas {matching.get('log_gas_smd'):.3f}, "
         f"n_nodes {bal.get('n_nodes', {}).get('smd', float('nan')):.3f}; surface AUC {surf['combined']:.3f} "
         f"(best single {surf['best_single']})")
    if crit is not None:
        cols = ["layer", "auc_trained", "auc_untrained", "delta", "delta_ci_lo", "delta_ci_hi", "perm_auc_p95",
                "random_dir_auc_p95", "counts"]
        _say(crit[cols].round(3).to_string(index=False))
    else:
        best = sm.sort_values("auc_holdout", ascending=False, na_position="last").iloc[0]
        _say(f"probe: best split-half AUC {best['auc_holdout']:.3f} at layer {int(best['layer'])}")
    _say(f"probe: -> {a.out}")
    return 0


def cmd_align(a) -> int:
    from ..io import write_manifest
    from ..model import load_model
    from ..transplant import paired_states
    from .source import (TraceLMSource, alignment_controls, covariate_matrix, length_covariates, load_pairs,
                         pooled_states)
    from ..config import ModelSpec

    pairs = load_pairs(a.pairs)
    if a.max_pairs:
        pairs = pairs[: a.max_pairs]
    if len(pairs) < 5:
        _say(f"align: only {len(pairs)} pairs")
        return 1
    t0 = time.time()
    src = TraceLMSource(a.model, device=a.device, max_len=a.max_len)
    src0, src0_name = _control_source(src, a.untrained, a.init_seed, a.model, a.device, a.max_len)
    if src0 is None:
        _say("align: the controls need an untrained source (--untrained auto or a directory)")
        return 1
    name, _, rev = a.target.partition("@")
    mb = load_model(ModelSpec(name=name, revision=a.target_rev or rev or None, device=a.device))
    layers_src = _parse_layers(a.layers_src, src.n_layers)
    layers_tgt = _parse_layers(a.layers_tgt, mb.n_layers)
    S, T = paired_states(src, mb, pairs, layers_src, layers_tgt, a.readout)
    S0 = pooled_states(src0, [p[0] for p in pairs], layers_src, a.readout)
    t_states = time.time() - t0
    cov = length_covariates([p[0] for p in pairs], texts=[p[1] for p in pairs], tokenizer=mb.tokenizer,
                            max_len=src.max_len)
    covsets = {"len3": False, "len3+text": True}
    frames = []
    for cname, with_text in covsets.items():
        Z, znames = covariate_matrix(cov, text=with_text)
        for alpha in [float(x) for x in a.alphas.split(",")]:
            df = alignment_controls(S, S0, T, Z, layers_src, layers_tgt, alpha=alpha, holdout=a.holdout, seed=a.seed,
                                    n_boot=a.n_boot, ci=a.ci)
            frames.append(df.assign(covariates=cname))
    res = pd.concat(frames, ignore_index=True)
    os.makedirs(a.out, exist_ok=True)
    res.to_csv(os.path.join(a.out, "align_controls.csv"), index=False)
    cov.to_csv(os.path.join(a.out, "length_covariates.csv"), index=False)
    summ = []
    for (cname, alpha), g in res.groupby(["covariates", "alpha"], sort=False):
        b = g.sort_values("r2_raw", ascending=False).iloc[0]          # the cell the plain alignment would pick
        r = g.sort_values("r2_resid", ascending=False).iloc[0]
        gr = g.sort_values("gain_resid", ascending=False).iloc[0]
        summ.append(dict(covariates=cname, alpha=alpha,
                         best_raw=b[["layer_src", "layer_tgt", "r2_raw", "r2_raw_untrained", "gain_raw", "gain_raw_ci_lo",
                                     "gain_raw_ci_hi", "r2_permuted", "r2_length_only", "r2_resid", "r2_resid_untrained",
                                     "gain_resid", "gain_resid_ci_lo", "gain_resid_ci_hi"]].to_dict(),
                         best_resid=r[["layer_src", "layer_tgt", "r2_resid", "r2_resid_untrained", "gain_resid",
                                       "gain_resid_ci_lo", "gain_resid_ci_hi"]].to_dict(),
                         best_gain_resid=gr[["layer_src", "layer_tgt", "r2_resid", "r2_resid_untrained", "gain_resid",
                                             "gain_resid_ci_lo", "gain_resid_ci_hi"]].to_dict(),
                         max_r2_length_only=float(g["r2_length_only"].max()),
                         n_cells_gain_resid_ci_above_0=int((g["gain_resid_ci_lo"] > 0).sum()),
                         n_cells=int(len(g))))
    out = dict(pairs=a.pairs, n_pairs=len(pairs), target=a.target, target_rev=a.target_rev, model=a.model,
               untrained=src0_name, readout=a.readout, holdout=a.holdout, seed=a.seed, n_boot=a.n_boot,
               covariates={k: covariate_matrix(cov, text=v)[1] for k, v in covsets.items()},
               covariate_means=cov.mean().to_dict(), summary=summ,
               seconds=dict(states=round(t_states, 1), total=round(time.time() - t0, 1)))
    with open(os.path.join(a.out, "align_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(_jsonable(out), fh, indent=1)
    write_manifest(a.out, kind="trace_align_controls", model=a.model, pairs=a.pairs, n_pairs=len(pairs),
                   target=a.target, untrained=src0_name, alphas=a.alphas, holdout=a.holdout, seed=a.seed)
    for s in summ:
        b = s["best_raw"]
        _say(f"align [{s['covariates']}, alpha={s['alpha']:g}] best raw cell L{int(b['layer_src'])}->L{int(b['layer_tgt'])}: "
             f"r2 raw {b['r2_raw']:.3f} (untrained {b['r2_raw_untrained']:.3f}, gain {b['gain_raw']:+.3f} "
             f"[{b['gain_raw_ci_lo']:+.3f}, {b['gain_raw_ci_hi']:+.3f}]), length-only {b['r2_length_only']:.3f}, "
             f"resid {b['r2_resid']:.3f} (untrained {b['r2_resid_untrained']:.3f}, gain {b['gain_resid']:+.3f} "
             f"[{b['gain_resid_ci_lo']:+.3f}, {b['gain_resid_ci_hi']:+.3f}])")
    _say(f"align: {len(pairs)} pairs, {len(res)} rows -> {a.out} ({time.time() - t0:.0f}s)")
    return 0


def cmd_pair(a) -> int:
    from .source import paired_items

    keys = [(r["tx_hash"], r.get("label")) for r in iter_rows(a.inp, limit=a.limit)]
    if not keys:
        _say("pair: no transactions")
        return 1
    rng = np.random.default_rng(a.seed)
    if a.balance:
        by = {}
        for h, lab in keys:
            by.setdefault(lab, []).append(h)
        per = max(1, a.n // len(by))
        chosen = []
        for lab in sorted(by, key=str):
            hs = by[lab]
            chosen += [hs[i] for i in rng.permutation(len(hs))[:per]]
    else:
        idx = rng.permutation(len(keys))[: a.n]
        chosen = [keys[i][0] for i in idx]
    order = {h: i for i, h in enumerate(chosen)}
    rows = sorted(iter_rows(a.inp, hashes=set(chosen)), key=lambda r: order[r["tx_hash"]])
    if a.rerender and not all("record" in r for r in rows):
        _say("pair: --rerender needs shards written with --records")
        return 1
    if a.rerender:
        texts = [t for _, t in paired_items(rows, max_nodes=a.max_nodes, address_style=a.address_style)]
        pairs = list(zip(rows, texts))
    else:
        pairs = [(r, r["text"]) for r in rows]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r, text in pairs:
            fh.write(json.dumps(dict(tx_hash=r["tx_hash"], block_number=r["block_number"], era=r.get("era"),
                                     label=r.get("label"), tokens=r["tokens"], text=text)) + "\n")
    os.replace(tmp, a.out)
    _say(f"pair: {len(pairs)} (ITR tokens, English text) pairs -> {a.out}")
    return 0


def cmd_bench(a) -> int:
    from .train import measure_throughput
    from .vocab import TraceVocab, closed_tokens

    if a.vocab:
        V = len(TraceVocab.load(a.vocab))
    elif a.vocab_size:
        V = int(a.vocab_size)
    else:
        V = len(closed_tokens()) + 20000 + 5000 + 2000     # the spec's default vocabulary size
    rows = []
    for size in a.sizes.split(","):
        r = measure_throughput(size.strip(), V, block_size=a.block_size, batch_size=a.batch_size,
                               n_steps=a.steps, n_warmup=a.warmup, device=a.device,
                               optimizer_step=a.optimizer_step)
        rows.append(r)
        _say(json.dumps(r))
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
    return 0


# ======================================================================================
# parser
# ======================================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lsne.chain.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download Xatu Parquet chunks (resumable)")
    d.add_argument("--from", dest="block_from", type=int, required=True)
    d.add_argument("--to", dest="block_to", type=int, required=True)
    d.add_argument("--every", type=int, default=1)
    d.add_argument("--tables", default=DEFAULT_TABLES)
    d.add_argument("--cache", default="cache/xatu")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--retries", type=int, default=4)
    d.add_argument("--verbose", action="store_true")
    d.set_defaults(func=cmd_download)

    s = sub.add_parser("serialize", help="cached chunks -> ITR jsonl shards (tokens + features + era + text)")
    s.add_argument("--cache", default="cache/xatu")
    s.add_argument("--from", dest="block_from", type=int, required=True)
    s.add_argument("--to", dest="block_to", type=int, required=True)
    s.add_argument("--out", default="data/traces/itr")
    s.add_argument("--records", action="store_true", help="also store the full TxRecord per line")
    s.add_argument("--plain", action="store_true", help="write uncompressed .jsonl (default: .jsonl.gz, ~10x smaller)")
    s.add_argument("--overwrite", action="store_true")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--max-args", type=int, default=4)
    s.add_argument("--max-depth-token", type=int, default=32)
    s.add_argument("--no-state", action="store_true")
    s.add_argument("--no-reads", action="store_true")
    s.add_argument("--no-logs", action="store_true")
    s.add_argument("--text-max-nodes", type=int, default=12)
    s.add_argument("--address-style", default="short", choices=["short", "full", "named"])
    s.add_argument("--k", type=int, default=5, help="fan threshold of the sharing label")
    s.set_defaults(func=cmd_serialize)

    v = sub.add_parser("vocab", help="build the vocabulary + HF tokenizer from ITR shards")
    v.add_argument("--in", dest="inp", default="data/traces/itr")
    v.add_argument("--out", default="data/traces/vocab")
    v.add_argument("--n-addresses", type=int, default=20000)
    v.add_argument("--n-selectors", type=int, default=5000)
    v.add_argument("--n-topics", type=int, default=2000)
    v.add_argument("--min-count", type=int, default=1)
    v.add_argument("--max-len", type=int, default=512)
    v.add_argument("--limit", type=int, default=None, help="use only the first N transactions")
    v.set_defaults(func=cmd_vocab)

    t = sub.add_parser("train", help="train the trace LM (GPT-NeoX, resumable)")
    t.add_argument("--in", dest="inp", default="data/traces/itr")
    t.add_argument("--vocab", default="data/traces/vocab")
    t.add_argument("--size", default="blockgpt", choices=["tiny", "small", "blockgpt"])
    t.add_argument("--out", default="runs/trace_lm")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--block-size", type=int, default=512)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--bf16", default="auto", choices=["auto", "on", "off"])
    t.add_argument("--eval-every", type=int, default=200)
    t.add_argument("--eval-blocks", type=int, default=512, help="cap on validation blocks per evaluation")
    t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--keep-checkpoints", type=int, default=2)
    t.add_argument("--val-frac", type=float, default=0.01)
    t.add_argument("--max-tx-len", type=int, default=512, help="truncate each transaction (<= 0: no truncation)")
    t.add_argument("--limit", type=int, default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto")
    t.set_defaults(func=cmd_train)

    q = sub.add_parser("probe", help="sharing directions in the trace LM from matched probe sets")
    q.add_argument("--model", default="runs/trace_lm")
    q.add_argument("--in", dest="inp", default="data/traces/itr")
    q.add_argument("--kind", default="fanout", choices=["fanout", "dao"])
    q.add_argument("--layer", default="all", help="'all' or comma-separated residual snapshots 0..L")
    q.add_argument("--out", default="runs/trace_probe")
    q.add_argument("--n", type=int, default=500, help="maximum number of matched pairs")
    q.add_argument("--k", type=int, default=5)
    q.add_argument("--eras", default=None, help="comma-separated eras (default: kind-specific)")
    q.add_argument("--gas-bins", type=int, default=10)
    q.add_argument("--match", default="gas", choices=["gas", "length"],
                   help="gas: era + gas-decile strata (spec); length: era + log ITR length within --length-ratio")
    q.add_argument("--length-ratio", type=float, default=1.25)
    q.add_argument("--length-cap", type=int, default=0, help="match on min(n_tokens, cap) (0: raw length)")
    q.add_argument("--mask", default="none", choices=["none", "dao", "all_addresses"],
                   help="replace identity tokens by [A_OOV] in both sets before encoding")
    q.add_argument("--untrained", default="auto",
                   help="untrained control: 'auto' (in-memory twin, same config/vocab/seed), a model dir, or 'none'")
    q.add_argument("--init-seed", type=int, default=None, help="seed of the auto twin (default: training seed)")
    q.add_argument("--n-boot", type=int, default=1000)
    q.add_argument("--n-perm", type=int, default=200)
    q.add_argument("--n-random", type=int, default=200)
    q.add_argument("--ci", type=float, default=0.95)
    q.add_argument("--readout", default="mean", choices=["mean", "last", "call"])
    q.add_argument("--denoise-pcs", type=int, default=0)
    q.add_argument("--max-len", type=int, default=512)
    q.add_argument("--limit", type=int, default=None)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--device", default="auto")
    q.set_defaults(func=cmd_probe)

    al = sub.add_parser("align", help="alignment trace LM -> target LM with length / untrained controls")
    al.add_argument("--model", default="runs/trace_lm")
    al.add_argument("--pairs", default="runs/trace_pairs.jsonl")
    al.add_argument("--target", default="EleutherAI/pythia-14m", help="HF model name, optionally name@revision")
    al.add_argument("--target-rev", default=None)
    al.add_argument("--untrained", default="auto")
    al.add_argument("--init-seed", type=int, default=None)
    al.add_argument("--alphas", default="1,100", help="comma-separated ridge penalties")
    al.add_argument("--holdout", type=float, default=0.2)
    al.add_argument("--n-boot", type=int, default=1000)
    al.add_argument("--ci", type=float, default=0.95)
    al.add_argument("--layers-src", default="all")
    al.add_argument("--layers-tgt", default="all")
    al.add_argument("--readout", default="mean", choices=["mean", "last"])
    al.add_argument("--max-pairs", type=int, default=None)
    al.add_argument("--max-len", type=int, default=512)
    al.add_argument("--seed", type=int, default=0)
    al.add_argument("--device", default="auto")
    al.add_argument("--out", default="runs/trace_align")
    al.set_defaults(func=cmd_align)

    r = sub.add_parser("pair", help="(ITR, English text) pairs for the alignment with Pythia")
    r.add_argument("--in", dest="inp", default="data/traces/itr")
    r.add_argument("--n", type=int, default=2000)
    r.add_argument("--out", default="runs/trace_pairs.jsonl")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--balance", action="store_true", help="equal numbers per sharing label")
    r.add_argument("--rerender", action="store_true", help="re-render the text from stored records")
    r.add_argument("--max-nodes", type=int, default=12)
    r.add_argument("--address-style", default="short", choices=["short", "full", "named"])
    r.add_argument("--limit", type=int, default=None)
    r.set_defaults(func=cmd_pair)

    b = sub.add_parser("bench", help="training throughput (tokens/s) per model size")
    b.add_argument("--sizes", default="tiny,small")
    b.add_argument("--block-size", type=int, default=512)
    b.add_argument("--batch-size", type=int, default=8)
    b.add_argument("--steps", type=int, default=3)
    b.add_argument("--warmup", type=int, default=1)
    b.add_argument("--vocab", default=None, help="vocabulary dir (default: the spec's full vocabulary size)")
    b.add_argument("--vocab-size", type=int, default=None)
    b.add_argument("--optimizer-step", action="store_true")
    b.add_argument("--device", default="cpu")
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_bench)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":   # pragma: no cover
    sys.exit(main())
