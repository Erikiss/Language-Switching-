#!/usr/bin/env bash
# End-to-end run of the trace source model (Stage L source adapter) on REAL 2016 data at a
# CPU-feasible scale.  Documentation and measured numbers: docs/TRACE_SOURCE.md.
#
#   scripts/trace_e2e.sh                 # every stage in order
#   scripts/trace_e2e.sh probe probe_eval  # selected stages
#
# Stages: download serialize split vocab oov bench train unigram init_model probe probe_eval
#         pair align summary
# Environment: OUT (runs/trace_e2e), CACHE (cache/xatu), TRAIN_MINUTES (35), DEVICE (cpu),
#              TARGET (EleutherAI/pythia-14m), TARGET_REV (step98000)
# Data: ethPandaOps Xatu (https://ethpandaops.io/posts/xatu-execution-layer/), CC BY 4.0.
# Everything under runs/ and cache/ is git-ignored; nothing here is committed.
set -euo pipefail
cd "$(dirname "$0")/.."

export OUT=${OUT:-runs/trace_e2e}
export CACHE=${CACHE:-cache/xatu}
export TRAIN_MINUTES=${TRAIN_MINUTES:-35}
export DEVICE=${DEVICE:-cpu}
export TARGET=${TARGET:-EleutherAI/pythia-14m}
export TARGET_REV=${TARGET_REV:-step98000}
export VAL_FRAC=0.10            # held-out share (by tx hash): vocabulary OOV, validation loss
export BLOCK=512 BATCH=8 LR=1e-3 WARMUP=100 EVAL_EVERY=100 EVAL_BLOCKS=64
FROM=1428000; TO=2100000; EVERY=25
TABLES=traces,transaction,native_transfers,erc20_transfers,logs,storage_diffs
CLI="python -m lsne.chain.cli"
LOG=$OUT/logs
mkdir -p "$LOG"

timed() {  # timed <logname> <command...>: run, tee the log, append the wall time
  local name=$1; shift
  local t0 t1
  t0=$(date +%s.%N)
  "$@" 2>&1 | tee "$LOG/$name.log"
  t1=$(date +%s.%N)
  python -c "print(f'wall {($t1 - $t0):.1f} s')" | tee -a "$LOG/$name.log"
}

# --------------------------------------------------------------------------------------
# 1. download: every 25th chunk of [1 428 000, 2 100 000) plus the full drain and fork chunks
# --------------------------------------------------------------------------------------
st_download() {
  timed 01_download_sampled $CLI download --from $FROM --to $TO --every $EVERY --tables $TABLES --cache "$CACHE"
  cp "$CACHE/download_log.csv" "$LOG/download_log_sampled.csv"
  timed 01_download_drain $CLI download --from 1718000 --to 1720000 --tables $TABLES --cache "$CACHE"
  cp "$CACHE/download_log.csv" "$LOG/download_log_drain.csv"
  timed 01_download_fork $CLI download --from 1920000 --to 1922000 --tables $TABLES --cache "$CACHE"
  cp "$CACHE/download_log.csv" "$LOG/download_log_fork.csv"
}

# --------------------------------------------------------------------------------------
# 2. serialise (ITR shards with tokens, features, era, label, English text)
# --------------------------------------------------------------------------------------
st_serialize() {
  timed 02_serialize $CLI serialize --cache "$CACHE" --from $FROM --to $TO --out "$OUT/itr" --workers 4 --overwrite
}

# training-side copy of the shards (90 %, by the same tx-hash rule the train command uses
# for its validation split) so that the vocabulary never sees the held-out 10 %
st_split() {
  python - <<'PY'
import gzip, json, os, re, shutil
from lsne.chain.cli import _is_val, list_shards
out, frac = os.environ["OUT"], float(os.environ["VAL_FRAC"])
dst = os.path.join(out, "itr_train90")
os.makedirs(dst, exist_ok=True)
shutil.copy(os.path.join(out, "itr", "serialize_params.json"), dst)
n = [0, 0]
for fn in list_shards(os.path.join(out, "itr")):
    with gzip.open(fn, "rt", encoding="utf-8") as fi, \
            gzip.open(os.path.join(dst, os.path.basename(fn)), "wt", encoding="utf-8", compresslevel=1) as fo:
        for line in fi:
            h = re.search(r'"tx_hash":\s*"(0x[0-9a-f]+)"', line).group(1)
            if _is_val(h, frac):
                n[1] += 1
            else:
                fo.write(line)
                n[0] += 1
print(f"split: {n[0]} train tx -> {dst}, {n[1]} held-out tx (val_frac {frac})")
PY
}

# --------------------------------------------------------------------------------------
# 3. vocabulary (built on the 90 % training side only) and held-out OOV
# --------------------------------------------------------------------------------------
st_vocab() {
  timed 03_vocab $CLI vocab --in "$OUT/itr_train90" --out "$OUT/vocab"
}

st_oov() {
  python - <<'PY' 2>&1 | tee "$OUT/logs/03_oov.log"
import json, os
from collections import Counter
from lsne.chain.cli import _is_val, iter_rows
from lsne.chain.vocab import TraceVocab, oov_rates, closed_tokens
out, frac = os.environ["OUT"], float(os.environ["VAL_FRAC"])
v = TraceVocab.load(os.path.join(out, "vocab"))
held, train_counts, held_counts = [], {c: Counter() for c in "AFT"}, {c: Counter() for c in "AFT"}
for r in iter_rows(os.path.join(out, "itr")):
    val = _is_val(r["tx_hash"], frac)
    cnt = held_counts if val else train_counts
    for t in r["tokens"]:
        if t[:2] in ("A:", "F:", "T:") and not t.endswith(":none"):
            cnt[t[0]][t[2:]] += 1
    if val:
        held.append(v.map_tokens(r["tokens"]))
oov = oov_rates(held)
used = Counter(t for toks in held for t in toks)
closed = closed_tokens()
# OOV of the held-out occurrences as a function of the open-class budget (top-N by train frequency)
curve = {}
for c, sizes in (("A", (1000, 5000, 20000, 50000, 100000)), ("F", (100, 500, 1000, 5000)), ("T", (50, 200, 2000))):
    ranked = [k for k, _ in sorted(train_counts[c].items(), key=lambda kv: (-kv[1], kv[0]))]
    tot = sum(held_counts[c].values())
    curve[c] = {int(N): 1 - sum(held_counts[c][k] for k in ranked[:N]) / tot for N in sizes} if tot else {}
res = dict(vocab_size=len(v), n_closed=v.meta["n_closed"], n_addresses=v.meta["n_addresses"],
           n_selectors=v.meta["n_selectors"], n_topics=v.meta["n_topics"],
           distinct_train=dict(A=len(train_counts["A"]), F=len(train_counts["F"]), T=len(train_counts["T"])),
           distinct_heldout=dict(A=len(held_counts["A"]), F=len(held_counts["F"]), T=len(held_counts["T"])),
           heldout_oov=oov, oov_vs_budget=curve,
           closed_tokens_used_in_heldout=sum(1 for t in closed if used.get(t)),
           train_coverage=dict(A=v.meta["coverage_A"], F=v.meta["coverage_F"], T=v.meta["coverage_T"]))
json.dump(res, open(os.path.join(out, "vocab_heldout_oov.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
PY
}

# --------------------------------------------------------------------------------------
# 4. CPU throughput (small for the run, blockgpt for the extrapolation) and training
# --------------------------------------------------------------------------------------
st_bench() {
  timed 04_bench $CLI bench --sizes small,blockgpt --vocab "$OUT/vocab" --block-size $BLOCK --batch-size $BATCH \
      --steps 3 --warmup 1 --optimizer-step --out "$OUT/bench_cpu.csv"
}

st_train() {
  # steps from the measured small-model step time so that training takes ~TRAIN_MINUTES
  local steps
  steps=$(python - <<'PY'
import os, pandas as pd
b = pd.read_csv(os.path.join(os.environ["OUT"], "bench_cpu.csv")).set_index("size")
sps = float(b.loc["small", "s_per_step"])
budget = float(os.environ["TRAIN_MINUTES"]) * 60 * 0.88   # ~12 % for the periodic evaluations and checkpoints
print(max(100, int(budget / sps) // 50 * 50))
PY
)
  echo "train: $steps steps (budget ${TRAIN_MINUTES} min)" | tee "$LOG/05_train_plan.log"
  timed 05_train $CLI train --in "$OUT/itr" --vocab "$OUT/vocab" --size small --out "$OUT/trace_lm" \
      --steps "$steps" --block-size $BLOCK --batch-size $BATCH --lr $LR --warmup $WARMUP \
      --eval-every $EVAL_EVERY --eval-blocks $EVAL_BLOCKS --save-every 250 --val-frac $VAL_FRAC --device $DEVICE
}

# unigram (and untuned interpolated bigram) baseline on exactly the validation blocks the
# training log uses (first EVAL_BLOCKS) and on a strided sample of all validation blocks,
# plus the final model's loss on that larger sample
st_unigram() {
  python - <<'PY' 2>&1 | tee "$OUT/logs/06_unigram.log"
import json, math, os
import numpy as np, torch
from lsne.chain.cli import corpus_blocks
from lsne.chain.train import eval_loss
from lsne.chain.vocab import TraceVocab
from transformers import AutoModelForCausalLM
out = os.environ["OUT"]
v = TraceVocab.load(os.path.join(out, "vocab"))
tr, va, stats = corpus_blocks(os.path.join(out, "itr"), v, int(os.environ["BLOCK"]), 512, float(os.environ["VAL_FRAC"]))
V = len(v)
x = tr.numpy().astype(np.int64)
uni = np.bincount(x[:, 1:].ravel(), minlength=V).astype(np.float64) + 0.5     # add-1/2 smoothing
p_uni = uni / uni.sum()
# bigram counts (context = previous token inside the block, as the LM sees it)
pairs = x[:, :-1].ravel() * V + x[:, 1:].ravel()
keys, cnts = np.unique(pairs, return_counts=True)
ctx_tot = np.bincount(x[:, :-1].ravel(), minlength=V).astype(np.float64)
big = dict(zip(keys.tolist(), cnts.tolist()))
def ce(blocks, lam=0.0):
    b = blocks.numpy().astype(np.int64)
    y, c = b[:, 1:].ravel(), b[:, :-1].ravel()
    pu = p_uni[y]
    if lam == 0.0:
        return float(-np.log(pu).mean())
    pb = np.array([big.get(int(ci) * V + int(yi), 0) for ci, yi in zip(c, y)], np.float64) / np.maximum(ctx_tot[c], 1)
    return float(-np.log(lam * pb + (1 - lam) * pu).mean())
n_eval = int(os.environ["EVAL_BLOCKS"])
idx = torch.linspace(0, len(va) - 1, steps=min(256, len(va))).long().unique()
va_first, va_wide = va[:n_eval], va[idx]
m = AutoModelForCausalLM.from_pretrained(os.path.join(out, "trace_lm")).eval()
res = dict(n_val_blocks=int(len(va)), n_train_blocks=int(len(tr)), corpus=stats, vocab_size=V,
           unigram_first=ce(va_first), unigram_wide=ce(va_wide),
           bigram_first=ce(va_first, 0.9), bigram_wide=ce(va_wide, 0.9),
           model_first=eval_loss(m, va_first, 16), model_wide=eval_loss(m, va_wide, 16),
           n_wide_blocks=int(len(va_wide)), log_V=math.log(V))
json.dump(res, open(os.path.join(out, "baselines.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k != "corpus"}, indent=1, default=float))
PY
}

# --------------------------------------------------------------------------------------
# 5. probes: matched sets, sharing direction per layer (CLI), controls (random directions,
#    surface features, the same probe in an untrained model of the same architecture)
# --------------------------------------------------------------------------------------
st_init_model() {
  python - <<'PY'
import os
from lsne.chain.train import make_model
from lsne.chain.vocab import TraceVocab, load_tokenizer
out = os.environ["OUT"]
d = os.path.join(out, "trace_lm_init")
v = TraceVocab.load(os.path.join(out, "vocab"))
m = make_model(len(v), "small", max_position_embeddings=512, seed=0)
m.save_pretrained(d)
load_tokenizer(os.path.join(out, "vocab")).save_pretrained(d)
v.save(d)
print(f"untrained control model -> {d}")
PY
}

st_probe() {
  for kind in fanout dao; do
    timed 07_probe_$kind $CLI probe --model "$OUT/trace_lm" --in "$OUT/itr" --kind $kind --layer all \
        --out "$OUT/probe_$kind" --n 500 --device $DEVICE
    timed 07_probe_${kind}_init $CLI probe --model "$OUT/trace_lm_init" --in "$OUT/itr" --kind $kind --layer all \
        --out "$OUT/probe_${kind}_init" --n 500 --device $DEVICE
  done
}

st_probe_eval() {
  python - <<'PY' 2>&1 | tee "$OUT/logs/08_probe_eval.log"
import json, os
import numpy as np, pandas as pd, torch
from lsne.chain.cli import iter_rows
from lsne.chain.labels import surface_baseline_auc
from lsne.chain.source import TraceLMSource, holdout_auc, pooled_states
out = os.environ["OUT"]
N_RAND = 500

def oriented_random_auc(Sp, Sn, n_dir, seed=0):
    """Split-half AUC of random unit directions (sign fitted on the training half), same split as holdout_auc."""
    n = min(len(Sp), len(Sn))
    perm = np.random.default_rng(seed).permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    g = torch.Generator().manual_seed(1234)
    R = torch.randn(Sp.shape[1], n_dir, generator=g)
    R = R / R.norm(dim=0, keepdim=True)
    aucs = []
    for a, b in (halves, halves[::-1]):
        s = torch.sign((Sp[a] @ R).mean(0) - (Sn[a] @ R).mean(0))
        s[s == 0] = 1
        p, q = (Sp[b] @ R) * s, (Sn[b] @ R) * s                      # [n_b, n_dir]
        gt = (p[:, None, :] > q[None, :, :]).float().mean((0, 1)) + 0.5 * (p[:, None, :] == q[None, :, :]).float().mean((0, 1))
        aucs.append(gt)
    return torch.stack(aucs).mean(0).numpy()

report = {}
for kind in ("fanout", "dao"):
    mt = pd.read_csv(os.path.join(out, f"probe_{kind}", "match_table.csv"))
    want = set(mt["pos_tx"]) | set(mt["neg_tx"])
    rows = {r["tx_hash"]: r for r in iter_rows(os.path.join(out, "itr"), hashes=want)}
    pos = [rows[h] for h in mt["pos_tx"]]
    neg = [rows[h] for h in mt["neg_tx"]]
    surf = surface_baseline_auc([r["tokens"] for r in pos], [r["tokens"] for r in neg], seed=0, max_len=512)
    surf_full = surface_baseline_auc([r["tokens"] for r in pos], [r["tokens"] for r in neg], seed=0)
    feats = lambda rs, k: [r["features"][k] for r in rs]
    match = dict(n_pairs=int(len(mt)), matching=str(mt["matching"].iloc[0]) if len(mt) else None,
                 gas_smd=float(mt["gas_smd"].iloc[0]) if len(mt) else float("nan"),
                 log_gas_smd=float(mt["log_gas_smd"].iloc[0]) if len(mt) else float("nan"),
                 eras=mt["era"].value_counts().to_dict(),
                 pos_fan_out_eth_median=float(np.median(feats(pos, "fan_out_eth"))),
                 neg_fan_out_eth_median=float(np.median(feats(neg, "fan_out_eth"))),
                 pos_dao_share=float(np.mean(feats(pos, "dao_involved"))),
                 neg_dao_share=float(np.mean(feats(neg, "dao_involved"))),
                 pos_n_tokens_median=float(np.median([r["n_tokens"] for r in pos])),
                 neg_n_tokens_median=float(np.median([r["n_tokens"] for r in neg])),
                 pos_gas_median=float(mt["pos_gas"].median()), neg_gas_median=float(mt["neg_gas"].median()))
    per_model = {}
    for tag, mdir in (("trained", "trace_lm"), ("untrained", "trace_lm_init")):
        sm = pd.read_csv(os.path.join(out, f"probe_{kind}" + ("" if tag == "trained" else "_init"), "probe_summary.csv"))
        src = TraceLMSource(os.path.join(out, mdir), device="cpu")
        layers = list(range(src.n_layers + 1))
        S = pooled_states(src, pos + neg, layers, "mean")
        n = len(pos)
        rows_l = []
        for l in layers:
            Sp, Sn = S[l][:n], S[l][n:]
            auc = holdout_auc(Sp, Sn, seed=0)
            rnd = oriented_random_auc(Sp, Sn, N_RAND)
            rows_l.append(dict(layer=l, auc_holdout=auc, auc_holdout_cli=float(sm.set_index("layer").loc[l, "auc_holdout"]),
                               auc_holdout_denoised=float(sm.set_index("layer").loc[l, "auc_holdout_denoised"]),
                               separation_d=float(sm.set_index("layer").loc[l, "separation_d"]),
                               random_mean=float(rnd.mean()), random_p95=float(np.quantile(rnd, 0.95)),
                               random_max=float(rnd.max()),
                               frac_random_ge=float((rnd >= auc).mean())))
        per_model[tag] = rows_l
        pd.DataFrame(rows_l).to_csv(os.path.join(out, f"probe_{kind}_{tag}_eval.csv"), index=False)
    report[kind] = dict(match=match, surface_baseline_512=surf, surface_baseline_full=surf_full, layers=per_model)
    print(kind, json.dumps(match, default=str))
    print("  surface (first 512 tokens):", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in surf.items()})
    for tag in per_model:
        for r in per_model[tag]:
            print(f"  {tag:9s} L{r['layer']}: AUC {r['auc_holdout']:.3f} (cli {r['auc_holdout_cli']:.3f}, denoised "
                  f"{r['auc_holdout_denoised']:.3f})  random mean {r['random_mean']:.3f} p95 {r['random_p95']:.3f} "
                  f"max {r['random_max']:.3f}  d={r['separation_d']:.2f}")
json.dump(report, open(os.path.join(out, "probe_eval.json"), "w"), indent=1, default=float)
PY
}

# --------------------------------------------------------------------------------------
# 6. pairs (ITR, English) and layer matching trace LM -> pythia-14m (mean pooling)
# --------------------------------------------------------------------------------------
st_pair() {
  timed 09_pair $CLI pair --in "$OUT/itr" --n 300 --out "$OUT/pairs.jsonl" --seed 0
  timed 09_pair_balanced $CLI pair --in "$OUT/itr" --n 300 --out "$OUT/pairs_balanced.jsonl" --seed 0 --balance
}

st_align() {
  python - <<'PY' 2>&1 | tee "$OUT/logs/10_align.log"
import json, os, time
import pandas as pd
from lsne.config import ModelSpec
from lsne.model import load_model
from lsne.transplant import layer_matching
from lsne.chain.source import TraceLMSource, load_pairs
out = os.environ["OUT"]
mb = load_model(ModelSpec(name=os.environ["TARGET"], revision=os.environ["TARGET_REV"], dtype="float32", device="cpu"))
res = {}
for pfile in ("pairs.jsonl", "pairs_balanced.jsonl"):
    pairs = load_pairs(os.path.join(out, pfile))
    for tag, mdir in (("trained", "trace_lm"), ("untrained", "trace_lm_init")):
        src = TraceLMSource(os.path.join(out, mdir), device="cpu")
        t0 = time.time()
        df = layer_matching(src, mb, pairs, readout="mean")
        name = f"align_{pfile.split('.')[0]}_{tag}"
        df.to_csv(os.path.join(out, name + ".csv"), index=False)
        best = df.sort_values("gap", ascending=False).iloc[0]
        res[name] = dict(n_pairs=int(df["n_pairs"].iloc[0]), seconds=round(time.time() - t0, 1),
                         best=best.to_dict(), mean_gap=float(df["gap"].mean()),
                         frac_holdout_gt_permuted=float((df["r2_holdout"] > df["r2_permuted"]).mean()),
                         max_r2_holdout=float(df["r2_holdout"].max()))
        print(name, json.dumps(res[name], default=str))
        print(df.pivot(index="layer_src", columns="layer_tgt", values="r2_holdout").round(3).to_string())
        print(df.pivot(index="layer_src", columns="layer_tgt", values="r2_permuted").round(3).to_string())
json.dump(res, open(os.path.join(out, "align_summary.json"), "w"), indent=1, default=str)
PY
}

st_summary() {
  python - <<'PY'
import json, os
out = os.environ["OUT"]
parts = {}
for f in ("vocab_heldout_oov.json", "baselines.json", "probe_eval.json", "align_summary.json"):
    p = os.path.join(out, f)
    if os.path.exists(p):
        parts[f] = json.load(open(p))
json.dump(parts, open(os.path.join(out, "summary.json"), "w"), indent=1, default=str)
print(f"summary -> {out}/summary.json ({', '.join(parts)})")
PY
}

ALL="download serialize split vocab oov bench train unigram init_model probe probe_eval pair align summary"
for s in ${@:-$ALL}; do
  echo "=== stage $s ($(date -u +%H:%M:%S))"
  "st_$s"
done
