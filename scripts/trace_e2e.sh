#!/usr/bin/env bash
# End-to-end run of the trace source model on REAL 2016 Ethereum data (docs/TRACE_SOURCE.md).
# Defaults = the CPU-scale validation run of 26.10.2026: every 25th chunk of blocks
# 1 428 000-2 100 000 plus the full drain (1 718 000-1 719 999) and fork (1 920 000-1 921 999)
# chunks, a 'small' GPT-NeoX trained for 200 steps x 8 x 512 tokens (~30 min on 4 shared
# CPU cores).  A plumbing validation, not the scientific result.
#
#   scripts/trace_e2e.sh                          # all steps in order
#   scripts/trace_e2e.sh vocab bench              # selected steps (also: vocab_report)
#   SIZE=blockgpt STEPS=20000 BATCH=32 LR=3e-4 WARMUP=200 DEVICE=auto scripts/trace_e2e.sh train
#   BLOCK_FROM=778000 BLOCK_TO=2913000 EVERY=1 EXTRA_RANGES="" scripts/trace_e2e.sh download serialize
#
# Steps: download serialize vocab bench baseline train evaluate untrained probe pair align
#        profile2016 report
# Every step is resumable / idempotent (cached chunks and existing shards are skipped,
# training resumes from its newest checkpoint); a failing step is logged with its exit
# code and the script continues with the next one.  Logs: $OUT/logs/<step>.log, results:
# $OUT/{bench.csv,baselines.json,eval.json,probe_report.json,layer_matching.csv,summary.json}.
# Every setting below can be overridden from the environment.
# Data: ethPandaOps Xatu (https://ethpandaops.io/posts/xatu-execution-layer/), CC BY 4.0.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python}
OUT=${OUT:-runs/trace_e2e}
CACHE=${CACHE:-cache/xatu}
BLOCK_FROM=${BLOCK_FROM:-1428000}       # corpus range [BLOCK_FROM, BLOCK_TO), every EVERY-th chunk
BLOCK_TO=${BLOCK_TO:-2100000}
EVERY=${EVERY:-25}
EXTRA_RANGES=${EXTRA_RANGES-"1718000-1720000 1920000-1922000"}   # fully downloaded sub-ranges (drain, fork)
WORKERS=${WORKERS:-4}
DEVICE=${DEVICE:-cpu}
N_ADDRESSES=${N_ADDRESSES:-20000}
VOCAB_REPORT=${VOCAB_REPORT:-1}         # 0: skip the (single-process, slow on all of 2016) OOV report
ITR=${ITR:-$OUT/itr}                    # all serialised transactions (train + validation split)
ITR_TRAIN=${ITR_TRAIN:-$OUT/itr_train}        # train split only (tokens), the vocabulary is built from it
VOCAB=${VOCAB:-$OUT/vocab}
MODEL=${MODEL:-$OUT/trace_lm_small}
UNTRAINED=${UNTRAINED:-$OUT/trace_lm_untrained}
VAL_FRAC=${VAL_FRAC:-0.1}       # hash-based split (lsne.chain.cli._is_val), identical in every step
SIZE=${SIZE:-small}
STEPS=${STEPS:-200}
BATCH=${BATCH:-8}
LR=${LR:-1e-3}
WARMUP=${WARMUP:-30}
EVAL_EVERY=${EVAL_EVERY:-20}
EVAL_BLOCKS=${EVAL_BLOCKS:-64}
SAVE_EVERY=${SAVE_EVERY:-20}
N_PROBE=${N_PROBE:-500}
PROBE_CONFIGS=${PROBE_CONFIGS:-"fanout:10 fanout:1000 dao:10 dao:1000"}
N_PAIRS=${N_PAIRS:-300}
TARGET=${TARGET:-EleutherAI/pythia-14m}
TARGET_REV=${TARGET_REV:-step98000}
LOG=$OUT/logs
mkdir -p "$LOG"
export OUT CACHE ITR ITR_TRAIN VOCAB MODEL UNTRAINED VAL_FRAC SIZE TARGET TARGET_REV EVAL_BLOCKS PROBE_CONFIGS DEVICE WORKERS

STATUS=0
timed() {   # timed <log name> <command…>: run, tee the log, append exit code and wall time
    local name=$1; shift
    local t0 t1 rc
    t0=$(date +%s.%N)
    "$@" 2>&1 | tee "$LOG/$name.log"
    rc=${PIPESTATUS[0]}
    t1=$(date +%s.%N)
    echo "[$name] exit $rc, wall $(awk "BEGIN{printf \"%.1f\", $t1-$t0}") s" | tee -a "$LOG/$name.log"
    [ "$rc" -eq 0 ] || STATUS=$rc
    return 0
}

# --------------------------------------------------------------------------------------
# 1. download: every EVERY-th chunk of [BLOCK_FROM, BLOCK_TO) + the EXTRA_RANGES in full
#    (default: every 25th chunk 1 428 000-2 100 000 + the drain and fork chunks)
# --------------------------------------------------------------------------------------
step_download() {
    timed 01_download_sampled $PY -m lsne.chain.cli download --from "$BLOCK_FROM" --to "$BLOCK_TO" --every "$EVERY" \
        --cache "$CACHE" --workers "$WORKERS"
    cp "$CACHE/download_log.csv" "$LOG/download_log_sampled.csv"
    local r
    for r in $EXTRA_RANGES; do
        timed "01_download_${r}" $PY -m lsne.chain.cli download --from "${r%-*}" --to "${r#*-}" --cache "$CACHE" \
            --workers "$WORKERS"
        cp "$CACHE/download_log.csv" "$LOG/download_log_${r}.csv"
    done
}

# --------------------------------------------------------------------------------------
# 2. serialise every cached chunk in [1 428 000, 2 100 000) -> ITR shards
# --------------------------------------------------------------------------------------
step_serialize() {
    timed 02_serialize $PY -m lsne.chain.cli serialize --cache "$CACHE" --from "$BLOCK_FROM" --to "$BLOCK_TO" \
        --out "$ITR" --workers "$WORKERS"
}

# --------------------------------------------------------------------------------------
# 3. vocabulary from the TRAIN split only; OOV measured on the held-out 10 %
# --------------------------------------------------------------------------------------
step_vocab() {
    timed 03a_split $PY - "$ITR" "$ITR_TRAIN" "$VAL_FRAC" <<'PY'
import gzip, json, os, re, shutil, sys
from lsne.chain.cli import _is_val, list_shards
src, dst, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
os.makedirs(dst, exist_ok=True)
shutil.copy(os.path.join(src, "serialize_params.json"), os.path.join(dst, "serialize_params.json"))
n = [0, 0]
for fn in list_shards(src):
    out = os.path.join(dst, os.path.basename(fn).replace(".jsonl.gz", "").replace(".jsonl", "") + ".jsonl.gz")
    tmp = out[: -len(".jsonl.gz")] + ".tmp.gz"
    opener = gzip.open if fn.endswith(".gz") else open
    with opener(fn, "rt", encoding="utf-8") as fi, gzip.open(tmp, "wt", encoding="utf-8", compresslevel=1) as fo:
        for line in fi:
            if not line.strip():
                continue
            r = json.loads(line)
            if _is_val(r["tx_hash"], frac):
                n[1] += 1
                continue
            n[0] += 1
            fo.write(json.dumps(dict(tx_hash=r["tx_hash"], tokens=r["tokens"])) + "\n")
    os.replace(tmp, out)
print(f"split: {n[0]} train tx -> {dst}, {n[1]} held-out tx (val_frac {frac})")
PY
    timed 03b_vocab $PY -m lsne.chain.cli vocab --in "$ITR_TRAIN" --out "$VOCAB" --n-addresses "$N_ADDRESSES"
    [ "$VOCAB_REPORT" = "0" ] || step_vocab_report
}

step_vocab_report() {   # composition, held-out OOV (also per era and per vocabulary budget)
    timed 03c_vocab_report $PY - <<'PY'
import collections, json, os
from lsne.chain.cli import _is_val, iter_rows
from lsne.chain.vocab import TraceVocab, closed_tokens, oov_rates
ITR, VOCAB, frac = os.environ["ITR"], os.environ["VOCAB"], float(os.environ["VAL_FRAC"])
v = TraceVocab.load(VOCAB)
closed = set(closed_tokens())
val_lists, by_era, used_closed = [], collections.defaultdict(list), collections.Counter()
n_val_tx = 0
for r in iter_rows(ITR):
    if not _is_val(r["tx_hash"], frac):
        continue
    m = v.map_tokens(r["tokens"])
    val_lists.append(m)
    by_era[r["era"]].append(m)
train_cnt = {c: collections.Counter() for c in "AFT"}
for r in iter_rows(os.environ["ITR_TRAIN"]):
    used_closed.update(t for t in r["tokens"] if t in closed)
    for t in r["tokens"]:
        if t[:2] in ("A:", "F:", "T:") and not t.endswith(":none"):
            train_cnt[t[0]][t[2:]] += 1
val_cnt = {c: collections.Counter() for c in "AFT"}
stream_cls = collections.Counter()
for r in iter_rows(ITR):
    if _is_val(r["tx_hash"], frac):
        for t in r["tokens"]:
            if t[:2] in ("A:", "F:", "T:") and not t.endswith(":none"):
                val_cnt[t[0]][t[2:]] += 1
budgets = dict(A=(1000, 5000, 20000, 50000, 100000), F=(100, 500, 1000, 5000), T=(50, 200, 2000))
oov_budget = {}
for c, bs in budgets.items():
    ranked = [k for k, _ in sorted(train_cnt[c].items(), key=lambda kv: (-kv[1], kv[0]))]
    tot = sum(val_cnt[c].values())
    oov_budget[c] = {str(b): sum(n for k, n in val_cnt[c].items() if k not in set(ranked[:b])) / tot for b in bs}
def cls(t):
    for p, name in (("V:", "value"), ("G:", "gas"), ("S:", "slot"), ("[D", "depth"), ("[CT:", "call_type"),
                    ("[ERR:", "error"), ("N", "numeric_arg")):
        if t.startswith(p):
            return name
    return "structural/other"
closed_by_class = collections.Counter(cls(t) for t in closed)
used_by_class = collections.Counter(cls(t) for t in used_closed)
for ls in val_lists:
    for t in ls:
        stream_cls["address" if t.startswith("A:") or t == "[A_OOV]" else "selector" if t.startswith("F:") or t == "[F_OOV]"
                   else "topic" if t.startswith("T:") or t == "[T_OOV]" else cls(t)] += 1
n_stream = sum(stream_cls.values())
heldout = oov_rates(val_lists)
per_era = {e: {k: oov_rates(ls)[k] for k in ("n_lists", "rate_A", "rate_F", "rate_T", "rate_open")}
           for e, ls in sorted(by_era.items())}
with open(os.path.join(VOCAB, "oov.json"), encoding="utf-8") as fh:
    insample = json.load(fh)["oov"]
m = v.meta
rep = dict(size=len(v), n_closed=m["n_closed"], n_addresses=m["n_addresses"], n_selectors=m["n_selectors"],
           n_topics=m["n_topics"], distinct_train=dict(A=m["distinct_addresses"], F=m["distinct_selectors"],
                                                     T=m["distinct_topics"]),
           coverage_train=dict(A=m["coverage_A"], F=m["coverage_F"], T=m["coverage_T"]),
           closed_by_class=dict(closed_by_class), closed_used_in_train=dict(used_by_class),
           n_closed_used=len(used_closed), oov_train_insample=insample, oov_heldout=heldout,
           oov_heldout_by_era=per_era, train_tx=m["n_items"], train_tokens=m["n_tokens"],
           oov_heldout_vs_budget=oov_budget, distinct_heldout={c: len(val_cnt[c]) for c in "AFT"},
           heldout_token_mix={k: n / n_stream for k, n in stream_cls.most_common()})
with open(os.path.join(VOCAB, "vocab_report.json"), "w", encoding="utf-8") as fh:
    json.dump(rep, fh, indent=1)
print(json.dumps({k: rep[k] for k in ("size", "n_closed", "n_addresses", "n_selectors", "n_topics",
                                      "distinct_train", "n_closed_used")}))
print("held-out OOV: address %.4f selector %.4f topic %.4f open %.4f unk %.5f (n_tx %d)" % (
    heldout["rate_A"], heldout["rate_F"], heldout["rate_T"], heldout["rate_open"], heldout["rate_unk"],
    heldout["n_lists"]))
for e, d in per_era.items():
    print(f"  {e:13s} n={d['n_lists']:6d} A {d['rate_A']:.4f} F {d['rate_F']:.4f} T {d['rate_T']:.4f}")
print("held-out OOV vs. vocabulary budget:", json.dumps(oov_budget))
print("held-out token mix:", json.dumps({k: round(v, 4) for k, v in rep["heldout_token_mix"].items()}))
PY
}

# --------------------------------------------------------------------------------------
# 4. CPU throughput (forward + backward + AdamW) for small and blockgpt with this vocabulary
# --------------------------------------------------------------------------------------
step_bench() {
    timed 04_bench $PY -m lsne.chain.cli bench --sizes small,blockgpt --vocab "$VOCAB" --block-size 512 \
        --batch-size 8 --steps 3 --warmup 1 --optimizer-step --out "$OUT/bench.csv"
}

# --------------------------------------------------------------------------------------
# 5. unigram / bigram baselines on exactly the packed blocks the trainer uses
# --------------------------------------------------------------------------------------
step_baseline() {
    timed 05_baseline $PY - <<'PY'
import json, os
import numpy as np
from lsne.chain.cli import corpus_blocks
from lsne.chain.vocab import TraceVocab
ITR, VOCAB, OUT = os.environ["ITR"], os.environ["VOCAB"], os.environ["OUT"]
frac, nsub = float(os.environ["VAL_FRAC"]), int(os.environ["EVAL_BLOCKS"])
v = TraceVocab.load(VOCAB)
V = len(v)
tr, va, stats = corpus_blocks(ITR, v, 512, 512, frac)
tr, va = tr.numpy().astype(np.int64), va.numpy().astype(np.int64)

def unigram(counts):
    p = (counts + 1.0) / (counts.sum() + V)                 # add-one (Laplace) smoothing
    return np.log(p)

def ce_uni(logp, blocks):
    y = blocks[:, 1:].reshape(-1)                           # eval_loss predicts positions 1..T-1
    return float(-logp[y].mean())

def bigram_tables(blocks):
    a, b = blocks[:, :-1].reshape(-1), blocks[:, 1:].reshape(-1)
    keys, cnt = np.unique(a * V + b, return_counts=True)
    ctx = np.bincount(a, minlength=V).astype(np.float64)
    return keys, cnt.astype(np.float64), ctx

def ce_bi(tables, logp_uni, blocks, lam):
    keys, cnt, ctx = tables
    a, b = blocks[:, :-1].reshape(-1), blocks[:, 1:].reshape(-1)
    q = a * V + b
    i = np.searchsorted(keys, q)
    i = np.minimum(i, len(keys) - 1)
    c = np.where(keys[i] == q, cnt[i], 0.0)
    ml = np.where(ctx[a] > 0, c / np.maximum(ctx[a], 1.0), 0.0)
    lam_eff = np.where(ctx[a] > 0, lam, 0.0)
    return float(-np.log(lam_eff * ml + (1 - lam_eff) * np.exp(logp_uni[b])).mean())

# lambda of the interpolated bigram chosen on the last 5 % of the TRAIN blocks (dev), never on val
n_dev = max(1, len(tr) // 20)
fit, dev = tr[:-n_dev], tr[-n_dev:]
lp_fit = unigram(np.bincount(fit.reshape(-1), minlength=V).astype(np.float64))
tab_fit = bigram_tables(fit)
lams = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
dev_ce = {l: ce_bi(tab_fit, lp_fit, dev, l) for l in lams}
lam = min(dev_ce, key=dev_ce.get)
lp = unigram(np.bincount(tr.reshape(-1), minlength=V).astype(np.float64))
tab = bigram_tables(tr)
yv = va[:, 1:].reshape(-1)
pv = np.bincount(yv, minlength=V) / yv.size
oracle = float(-(pv[pv > 0] * np.log(pv[pv > 0])).sum())
res = dict(corpus=stats, vocab_size=V,
           unigram_val=ce_uni(lp, va), unigram_val_sub=ce_uni(lp, va[:nsub]),
           bigram_val=ce_bi(tab, lp, va, lam), bigram_val_sub=ce_bi(tab, lp, va[:nsub], lam),
           bigram_lambda=lam, bigram_dev_ce=dev_ce, val_unigram_entropy_oracle=oracle,
           uniform=float(np.log(V)), eval_blocks_sub=nsub)
with open(os.path.join(OUT, "baselines.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "corpus"}, indent=1))
PY
}

# --------------------------------------------------------------------------------------
# 6. train the 'small' config on CPU (bounded: STEPS x BATCH x 512 tokens, ~35 min here)
# --------------------------------------------------------------------------------------
step_train() {
    timed 06_train $PY -m lsne.chain.cli train --in "$ITR" --vocab "$VOCAB" --size "$SIZE" --out "$MODEL" \
        --steps "$STEPS" --batch-size "$BATCH" --lr "$LR" --warmup "$WARMUP" --val-frac "$VAL_FRAC" \
        --eval-every "$EVAL_EVERY" --eval-blocks "$EVAL_BLOCKS" --save-every "$SAVE_EVERY" --device "$DEVICE"
}

# --------------------------------------------------------------------------------------
# 7. final validation loss on ALL held-out blocks vs. the baselines
# --------------------------------------------------------------------------------------
step_evaluate() {
    timed 07_evaluate $PY - <<'PY'
import json, os
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from lsne.chain.cli import corpus_blocks
from lsne.chain.vocab import TraceVocab
ITR, VOCAB, MODEL, OUT = (os.environ[k] for k in ("ITR", "VOCAB", "MODEL", "OUT"))
frac = float(os.environ["VAL_FRAC"])
v = TraceVocab.load(VOCAB)
V = len(v)
tr, va, stats = corpus_blocks(ITR, v, 512, 512, frac)
tr, va = tr.numpy().astype(np.int64), va.numpy().astype(np.int64)

def cls(t):
    if t.startswith("A:") or t == "[A_OOV]":
        return "A_in_vocab" if t.startswith("A:") and t != "A:none" else ("A_oov" if t == "[A_OOV]" else "A_none")
    for p, name in (("F:", "selector"), ("T:", "topic"), ("[F_OOV]", "selector"), ("[T_OOV]", "topic"),
                    ("V:", "value"), ("G:", "gas"), ("S:", "slot"), ("[D", "depth"), ("[CT:", "call_type"),
                    ("[ERR:", "error"), ("[EOS]", "eos"), ("N", "numeric_arg")):
        if t.startswith(p):
            return name
    return "structural"
names = sorted({cls(t) for t in v.tokens})
cid = np.asarray([names.index(cls(t)) for t in v.tokens])

# --- model: per-position cross-entropy over ALL validation blocks
dev = torch.device("cuda" if os.environ["DEVICE"] == "auto" and torch.cuda.is_available() else
                   ("cpu" if os.environ["DEVICE"] == "auto" else os.environ["DEVICE"]))
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval().to(dev)
ce_m = np.zeros(len(names)); n_c = np.zeros(len(names))
with torch.no_grad():
    for i in range(0, len(va), 16):
        x = torch.from_numpy(va[i:i + 16]).to(dev)
        lp = torch.log_softmax(model(input_ids=x, use_cache=False).logits[:, :-1].float(), -1)
        y = x[:, 1:]
        nll = -lp.gather(-1, y[..., None])[..., 0].cpu().numpy().reshape(-1)
        c = cid[y.cpu().numpy().reshape(-1)]
        np.add.at(ce_m, c, nll); np.add.at(n_c, c, 1)
# --- unigram / bigram (same smoothing and lambda as step 'baseline')
with open(os.path.join(OUT, "baselines.json"), encoding="utf-8") as fh:
    b = json.load(fh)
lam = float(b["bigram_lambda"])
lp_u = np.log((np.bincount(tr.reshape(-1), minlength=V) + 1.0) / (tr.size + V))
a_, b_ = tr[:, :-1].reshape(-1), tr[:, 1:].reshape(-1)
keys, cnt = np.unique(a_ * V + b_, return_counts=True)
ctx = np.bincount(a_, minlength=V).astype(float)
a, y = va[:, :-1].reshape(-1), va[:, 1:].reshape(-1)
q = a * V + y
k = np.minimum(np.searchsorted(keys, q), len(keys) - 1)
c2 = np.where(keys[k] == q, cnt[k], 0.0)
ml = np.where(ctx[a] > 0, c2 / np.maximum(ctx[a], 1.0), 0.0)
le = np.where(ctx[a] > 0, lam, 0.0)
nll_b = -np.log(le * ml + (1 - le) * np.exp(lp_u[y]))
nll_u = -lp_u[y]
cy = cid[y]
ce_b = np.bincount(cy, weights=nll_b, minlength=len(names)); ce_u = np.bincount(cy, weights=nll_u, minlength=len(names))
per_class = {nm: dict(share=float(n_c[i] / n_c.sum()), model=float(ce_m[i] / n_c[i]), bigram=float(ce_b[i] / n_c[i]),
                      unigram=float(ce_u[i] / n_c[i])) for i, nm in enumerate(names) if n_c[i] > 0}
res = dict(n_val_blocks=int(len(va)), val_tokens_predicted=int(n_c.sum()),
           val_loss_full=float(ce_m.sum() / n_c.sum()), unigram_val=float(nll_u.mean()), bigram_val=float(nll_b.mean()),
           per_class=per_class)
res.update(delta_vs_unigram=res["val_loss_full"] - res["unigram_val"], delta_vs_bigram=res["val_loss_full"] - res["bigram_val"])
with open(os.path.join(OUT, "eval.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))
print(f"{'class':12s} {'share':>6s} {'model':>7s} {'bigram':>7s} {'unigram':>7s}")
for nm, d in sorted(per_class.items(), key=lambda kv: -kv[1]["share"]):
    print(f"{nm:12s} {d['share']:6.3f} {d['model']:7.3f} {d['bigram']:7.3f} {d['unigram']:7.3f}")
PY
}

# --------------------------------------------------------------------------------------
# 8. untrained control model (same architecture, vocabulary and seed, no training)
# --------------------------------------------------------------------------------------
step_untrained() {
    timed 08_untrained $PY - <<'PY'
import os
from lsne.chain.train import make_model
from lsne.chain.vocab import TraceVocab, load_tokenizer
VOCAB, UNTRAINED = os.environ["VOCAB"], os.environ["UNTRAINED"]
v = TraceVocab.load(VOCAB)
m = make_model(len(v), os.environ["SIZE"], seed=0)
m.save_pretrained(UNTRAINED)
load_tokenizer(VOCAB).save_pretrained(UNTRAINED)
v.save(UNTRAINED)
print(f"untrained {os.environ['SIZE']} model -> {UNTRAINED}")
PY
}

# --------------------------------------------------------------------------------------
# 9. probe sets + sharing direction per layer (CLI), then nulls and controls
# --------------------------------------------------------------------------------------
step_probe() {
    # gas_bins 10 = the spec's gas-decile matching; 1000 = finer gas strata (sensitivity)
    for cfg in $PROBE_CONFIGS; do
        kind=${cfg%:*}; bins=${cfg#*:}
        timed 09_probe_${kind}_g$bins $PY -m lsne.chain.cli probe --model "$MODEL" --in "$ITR" --kind $kind --layer all \
            --n "$N_PROBE" --gas-bins $bins --out "$OUT/probe_${kind}_g$bins" --device "$DEVICE"
    done
    timed 09_probe_report $PY - <<'PY'
import json, os
import numpy as np
import pandas as pd
import torch
from lsne.chain.cli import _features_frame, _is_val, iter_rows
from lsne.chain.labels import standardized_mean_difference as smd, surface_baseline_auc
from lsne.chain.source import TraceLMSource, holdout_auc, pooled_states
OUT, ITR, MODEL, UNTRAINED = (os.environ[k] for k in ("OUT", "ITR", "MODEL", "UNTRAINED"))
frac = float(os.environ["VAL_FRAC"])
feats = _features_frame(ITR, None)
ff = feats.set_index("tx_hash")
src, src0 = TraceLMSource(MODEL, device=os.environ["DEVICE"]), TraceLMSource(UNTRAINED, device=os.environ["DEVICE"])
layers = list(range(src.n_layers + 1))

def auc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float((p[:, None] > q[None, :]).mean() + 0.5 * (p[:, None] == q[None, :]).mean())

def halves(n, seed=0):          # exactly the split of lsne.chain.source.holdout_auc
    perm = np.random.default_rng(seed).permutation(n)
    return perm[: n // 2], perm[n // 2:]

def random_dir_auc(Sp, Sq, vecs):
    """AUC of fixed random unit directions on the hold-out halves, folded max(a, 1 - a) (generous to the null)."""
    hs = halves(len(Sp))
    out = []
    for v in vecs:
        a = np.mean([auc((Sp[h] @ v).numpy(), (Sq[h] @ v).numpy()) for h in hs])
        out.append(max(a, 1 - a))
    return np.asarray(out)

def residualize(S, z):
    """Remove the linear effect of the covariates z [N, k] (plus intercept) from the states S [N, d]."""
    Z = np.column_stack([np.ones(len(z)), z])
    beta, *_ = np.linalg.lstsq(Z, S.numpy().astype(np.float64), rcond=None)
    return torch.from_numpy(S.numpy() - (Z @ beta).astype(np.float32))

report = {}
for cfg in os.environ["PROBE_CONFIGS"].split():
    kind, bins = cfg.split(":")
    name = f"{kind}_g{bins}"
    d = os.path.join(OUT, f"probe_{name}")
    if not os.path.exists(os.path.join(d, "match_table.csv")):
        report[name] = dict(error="probe step failed (no match_table.csv)")
        continue
    mt = pd.read_csv(os.path.join(d, "match_table.csv"))
    summ = pd.read_csv(os.path.join(d, "probe_summary.csv"))
    want = set(mt["pos_tx"]) | set(mt["neg_tx"])
    rows = {r["tx_hash"]: r for r in iter_rows(ITR, hashes=want)}
    pos = [rows[h] for h in mt["pos_tx"]]
    neg = [rows[h] for h in mt["neg_tx"]]
    n = len(pos)
    if kind == "fanout":
        pool_pos, pool_neg = int((feats["fan_out_eth"] >= 5).sum()), int((feats["fan_out_eth"] == 1).sum())
    else:
        keep = feats["era"].isin(["drain_crisis", "post_fork"])
        pool_pos = int((keep & feats["dao_involved"].astype(bool)).sum())
        pool_neg = int((keep & ~feats["dao_involved"].astype(bool)).sum())
    ntok_p = [min(r["n_tokens"], 512) for r in pos]; ntok_q = [min(r["n_tokens"], 512) for r in neg]
    nodes_p = [int(ff.at[h, "n_nodes"]) for h in mt["pos_tx"]]; nodes_q = [int(ff.at[h, "n_nodes"]) for h in mt["neg_tx"]]
    lg = np.log10(np.maximum(mt["pos_gas"].to_numpy(float), 1) / np.maximum(mt["neg_gas"].to_numpy(float), 1))
    surf = surface_baseline_auc([r["tokens"] for r in pos], [r["tokens"] for r in neg], seed=0, max_len=512)
    match = dict(n_pairs=n, pool_pos=pool_pos, pool_neg=pool_neg, gas_bins=int(bins),
                 matching=str(mt["matching"].iloc[0]) if "matching" in mt and n else None,
                 eras=mt["era"].value_counts().to_dict(),
                 gas_median_abs_log10_ratio=float(np.median(np.abs(lg))),
                 gas_frac_within_factor2=float((np.abs(lg) <= np.log10(2)).mean()),
                 log_gas_smd=smd(np.log(np.maximum(mt["pos_gas"], 1)), np.log(np.maximum(mt["neg_gas"], 1))),
                 n_tokens_mean_pos=float(np.mean(ntok_p)), n_tokens_mean_neg=float(np.mean(ntok_q)),
                 n_tokens_smd=smd(ntok_p, ntok_q), n_nodes_mean_pos=float(np.mean(nodes_p)),
                 n_nodes_mean_neg=float(np.mean(nodes_q)), auc_n_tokens_insample=auc(ntok_p, ntok_q),
                 surface_baseline=surf,
                 frac_in_val_split=float(np.mean([_is_val(h, frac) for h in list(mt["pos_tx"]) + list(mt["neg_tx"])])),
                 pos_fan_out_median=float(mt["pos_fan_out_eth"].median()),
                 dao_pos_frac=float(mt["pos_dao_involved"].mean()), dao_neg_frac=float(mt["neg_dao_involved"].mean()))
    S = pooled_states(src, pos + neg, layers, "mean")
    S0 = pooled_states(src0, pos + neg, layers, "mean")
    z = np.log(np.asarray(ntok_p + ntok_q, float))[:, None]
    R = torch.randn(200, src.d_model, generator=torch.Generator().manual_seed(1234))
    R = R / R.norm(dim=1, keepdim=True)
    rng = np.random.default_rng(7)
    per_layer = []
    for l in layers:
        Sp, Sq = S[l][:n], S[l][n:]
        rnd = random_dir_auc(Sp, Sq, R)
        null = []
        for _ in range(100):    # label-permutation null for the FITTED direction: swap pos/neg inside random pairs
            sw = torch.from_numpy(rng.random(n) < 0.5)
            null.append(holdout_auc(torch.where(sw[:, None], Sq, Sp), torch.where(sw[:, None], Sp, Sq), seed=0))
        Sr, S0r = residualize(S[l], z), residualize(S0[l], z)
        row = summ[summ["layer"] == l].iloc[0]
        per_layer.append(dict(layer=l, auc_holdout=float(row["auc_holdout"]),
                              auc_holdout_denoised=float(row["auc_holdout_denoised"]),
                              separation_d_insample=float(row["separation_d"]),
                              random_dir_auc_mean=float(rnd.mean()), random_dir_auc_p95=float(np.quantile(rnd, 0.95)),
                              random_dir_auc_max=float(rnd.max()),
                              label_perm_auc_p95=float(np.nanquantile(null, 0.95)),
                              untrained_auc_holdout=holdout_auc(S0[l][:n], S0[l][n:], seed=0),
                              auc_holdout_len_resid=holdout_auc(Sr[:n], Sr[n:], seed=0),
                              untrained_auc_holdout_len_resid=holdout_auc(S0r[:n], S0r[n:], seed=0)))
    pl = pd.DataFrame(per_layer)
    pl.to_csv(os.path.join(d, "separability.csv"), index=False)
    report[name] = dict(matching=match, per_layer=per_layer)
    print(f"== {name}: {n} pairs (pool {pool_pos} pos / {pool_neg} neg)")
    print(json.dumps(match))
    print(pl.round(3).to_string(index=False))
with open(os.path.join(OUT, "probe_report.json"), "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=1)
PY
}

# --------------------------------------------------------------------------------------
# 10. (ITR, render_text) pairs
# --------------------------------------------------------------------------------------
step_pair() {
    timed 10_pair $PY -m lsne.chain.cli pair --in "$ITR" --n "$N_PAIRS" --out "$OUT/pairs.jsonl"
}

# --------------------------------------------------------------------------------------
# 11. layer matching trace LM (source) -> Pythia-14m (target), mean pooling
# --------------------------------------------------------------------------------------
step_align() {
    timed 11_align $PY - <<'PY'
import json, os
import numpy as np
import pandas as pd
from lsne.chain.source import TraceLMSource, load_pairs
from lsne.config import ModelSpec
from lsne.model import load_model
from lsne.transplant import fit_alignment, layer_matching, paired_states
OUT, MODEL, UNTRAINED = os.environ["OUT"], os.environ["MODEL"], os.environ["UNTRAINED"]
pairs = load_pairs(os.path.join(OUT, "pairs.jsonl"))
mb = load_model(ModelSpec(name=os.environ["TARGET"], revision=os.environ["TARGET_REV"], device=os.environ["DEVICE"]))
frames = []
for tag, path in (("trained", MODEL), ("untrained", UNTRAINED)):
    src = TraceLMSource(path, device=os.environ["DEVICE"])
    for alpha in (1.0, 100.0, 10000.0):   # 1.0 = fit_alignment default; larger = stronger ridge (sensitivity)
        df = layer_matching(src, mb, pairs, readout="mean", method="ridge", alpha=alpha, holdout=0.2, seed=0)
        frames.append(df.assign(source=tag, alpha=alpha))
# control: the source replaced by two length features (log #ITR tokens, log #text tokens)
_, Tt = paired_states(src, mb, pairs, [0], list(range(mb.n_layers + 1)), "mean")
L = np.asarray([[np.log(len(t)), np.log(len(mb.tokenizer(x)["input_ids"]))] for t, x in pairs])
for lt in range(mb.n_layers + 1):
    a = fit_alignment(L, Tt[lt], method="ridge", alpha=1e-3, holdout=0.2, seed=0, layer_src=-1, layer_tgt=lt)
    frames.append(pd.DataFrame([dict(layer_src=-1, layer_tgt=lt, r2_train=a.r2_train, r2_holdout=a.r2_holdout,
                                     r2_permuted=a.r2_permuted, gap=a.gap, method="ridge", n_pairs=a.n_pairs,
                                     source="length_only", alpha=1e-3)]))
res = pd.concat(frames, ignore_index=True)
res.to_csv(os.path.join(OUT, "layer_matching.csv"), index=False)
for (tag, alpha), g in res.groupby(["source", "alpha"], sort=False):
    print(f"== source={tag} ridge alpha={alpha}: {len(pairs)} pairs (hold-out 20 %), d_S={src.d_model}, d_T={mb.d_model}")
    print("  r2_holdout (rows: source layer, columns: target layer):")
    print(g.pivot(index="layer_src", columns="layer_tgt", values="r2_holdout").round(3).to_string())
    print("  r2_permuted:")
    print(g.pivot(index="layer_src", columns="layer_tgt", values="r2_permuted").round(3).to_string())
    b = g.sort_values("gap", ascending=False).iloc[0]
    print(f"  best gap {b['gap']:.3f} at src L{int(b['layer_src'])} -> tgt L{int(b['layer_tgt'])} "
          f"(r2_holdout {b['r2_holdout']:.3f}, r2_permuted {b['r2_permuted']:.3f})")
PY
}

# --------------------------------------------------------------------------------------
# 12. 2016 profile for the extrapolation: every 100th chunk of 2016, all six tables
# --------------------------------------------------------------------------------------
step_profile2016() {
    timed 12a_download_profile $PY -m lsne.chain.cli download --from 778000 --to 2913000 --every 100 \
        --cache "$CACHE" --workers 4
    cp "$CACHE/download_log.csv" "$LOG/download_log_profile2016.csv"
    timed 12b_download_blocks $PY -m lsne.chain.cli download --from 778000 --to 779000 --tables block --cache "$CACHE"
    timed 12c_download_blocks $PY -m lsne.chain.cli download --from 2912000 --to 2913000 --tables block --cache "$CACHE"
    timed 12d_serialize_profile $PY -m lsne.chain.cli serialize --cache "$CACHE" --from 778000 --to 2913000 \
        --out "$OUT/itr_profile2016" --workers 2
    timed 12e_profile_report $PY - <<'PY'
import json, os
import numpy as np
import pandas as pd
from lsne.chain.cli import iter_rows, list_shards
from lsne.chain.xatu import read_chunk
OUT, CACHE = os.environ["OUT"], os.environ["CACHE"]
# 2016 block range from the Xatu block table (UTC dates)
bounds = {}
for s in (778000, 2912000):
    rows = read_chunk("block", s, CACHE, columns=["block_number", "block_date_time"])
    bounds[s] = sorted((int(r["block_number"]), str(r["block_date_time"])) for r in rows)
first_2016 = min(b for b, t in bounds[778000] if t.startswith("2016"))
last_2016 = max(b for b, t in bounds[2912000] if t.startswith("2016"))
dl = pd.read_csv(os.path.join(OUT, "logs", "download_log_profile2016.csv"))
grid = sorted(dl["start"].unique())
rows = []
for fn in list_shards(os.path.join(OUT, "itr_profile2016")):
    start = int(os.path.basename(fn)[4:13])
    if start not in grid:
        continue
    n_tx = n_tok = n_tok_trunc = 0
    for r in iter_rows(fn):
        n_tx += 1
        n_tok += r["n_tokens"]
        n_tok_trunc += min(r["n_tokens"], 512) + 1          # truncation to 512 + [EOS] as in training
    mb = dl[dl["start"] == start]["bytes"].sum() / 1e6
    rows.append(dict(start=start, n_tx=n_tx, tokens=n_tok, tokens_train=n_tok_trunc, mb=mb))
pf = pd.DataFrame(rows)
pf.to_csv(os.path.join(OUT, "profile2016.csv"), index=False)
n_chunks = len(range(first_2016 // 1000 * 1000, last_2016 + 1, 1000))
res = dict(first_block_2016=first_2016, last_block_2016=last_2016, first_time=dict(bounds[778000])[first_2016],
           last_time=dict(bounds[2912000])[last_2016], n_chunks_2016=n_chunks, n_profile_chunks=len(pf),
           mean_tx_per_chunk=float(pf["n_tx"].mean()), mean_tokens_per_chunk=float(pf["tokens"].mean()),
           mean_train_tokens_per_chunk=float(pf["tokens_train"].mean()),
           median_train_tokens_per_chunk=float(pf["tokens_train"].median()),
           mean_mb_per_chunk=float(pf["mb"].mean()),
           est_tx_2016=float(pf["n_tx"].mean() * n_chunks), est_train_tokens_2016=float(pf["tokens_train"].mean() * n_chunks),
           est_download_gb_2016=float(pf["mb"].mean() * n_chunks / 1e3))
with open(os.path.join(OUT, "profile2016.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
print(pf.to_string(index=False))
print(json.dumps(res, indent=1))
PY
}

# --------------------------------------------------------------------------------------
# 13. summary + A100 extrapolation for 'blockgpt' over all of 2016 (EXTRAPOLATION)
# --------------------------------------------------------------------------------------
step_report() {
    timed 13_report $PY - <<'PY'
import json, os, re
import pandas as pd
from lsne.chain.train import flops_per_token, param_counts
from lsne.chain.vocab import TraceVocab
OUT, VOCAB, MODEL = os.environ["OUT"], os.environ["VOCAB"], os.environ["MODEL"]
def load(p):
    p = os.path.join(OUT, p)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None
def wall(log):
    m = re.search(r"wall ([0-9.]+) s", open(os.path.join(OUT, "logs", log), encoding="utf-8").read())
    return float(m.group(1)) if m else float("nan")
voc = TraceVocab.load(VOCAB)
V = len(voc)
vpath = os.path.join(VOCAB, "vocab_report.json")
rep = json.load(open(vpath, encoding="utf-8")) if os.path.exists(vpath) else None
V_full = V - int((rep or voc.meta)["n_addresses"]) + 100000   # scenario: 100 000 addresses (BlockGPT: 93 233)
bpath = os.path.join(OUT, "bench.csv")
bench = pd.read_csv(bpath).set_index("size") if os.path.exists(bpath) else pd.DataFrame()
prof, ev, base = load("profile2016.json"), load("eval.json"), load("baselines.json")
man = json.load(open(os.path.join(MODEL, "manifest.json"), encoding="utf-8"))
log = pd.read_csv(os.path.join(MODEL, "train_log.csv"))
A100_PEAK = 312e12                                   # bf16 dense tensor-core peak (NVIDIA datasheet)
def bench_val(size, col):
    return float(bench.loc[size, col]) if size in bench.index else float("nan")
ext = dict(vocab_size=V, vocab_size_100k_addresses=V_full,
           cpu_bench_small_tokens_per_s=bench_val("small", "tokens_per_s"),
           cpu_bench_blockgpt_tokens_per_s=bench_val("blockgpt", "tokens_per_s"),
           cpu_bench_blockgpt_s_per_step_8x512=bench_val("blockgpt", "s_per_step"),
           cpu_train_tokens_per_s_effective=float(man["tokens_per_s"]))
T = prof["est_train_tokens_2016"] if prof else float("nan")
ext.update(tokens_2016_truncated512=T, tokens_spec_run_20000x32x512=20000 * 32 * 512)
for tag, vs in (("V_run", V), ("V_100k_addresses", V_full)):
    fpt = flops_per_token("blockgpt", vs, 512)
    ext[f"blockgpt_{tag}_params"] = param_counts("blockgpt", vs)
    ext[f"blockgpt_{tag}_flops_per_token"] = fpt
    for mfu in (0.15, 0.30):                         # ASSUMED model-FLOP utilisation (extrapolation)
        tps = A100_PEAK * mfu / fpt
        ext[f"a100_{tag}_mfu{int(mfu * 100)}_tokens_per_s"] = tps
        ext[f"a100_{tag}_mfu{int(mfu * 100)}_hours_per_epoch_2016"] = T / tps / 3600
ext["cpu_blockgpt_hours_per_epoch_2016"] = T / ext["cpu_bench_blockgpt_tokens_per_s"] / 3600
# data preparation for all of 2016, scaled from the measured serialisation runs
def ser_cpu_s_per_chunk(name):
    p = os.path.join(OUT, "logs", name)
    if not os.path.exists(p):
        return float("nan")
    txt = open(p, encoding="utf-8").read()
    m = re.search(r"serialize: (\d+) chunks", txt)
    return float(os.environ.get("WORKERS", "4")) * wall(name) / int(m.group(1)) if m else float("nan")
ser_cpu_s = [ser_cpu_s_per_chunk("02_serialize.log"), ser_cpu_s_per_chunk("12d_serialize_profile.log")]
n_chunks = prof["n_chunks_2016"] if prof else float("nan")
ext.update(serialize_cpu_s_per_chunk=ser_cpu_s, serialize_2016_cpu_hours=[x * n_chunks / 3600 for x in ser_cpu_s],
           download_2016_gb=prof["est_download_gb_2016"] if prof else float("nan"))
summary = dict(train=dict(steps=int(log["step"].max()), tokens_seen=int(log["tokens_seen"].max()),
                          wall_s=float(man.get("wall_s", float("nan"))), tokens_per_s=man.get("tokens_per_s"),
                          params=man.get("params"), curve=log.round(4).to_dict(orient="records")),
               eval=ev, baselines={k: base[k] for k in base if k != "corpus"} if base else None,
               corpus=base["corpus"] if base else None, profile2016=prof, extrapolation=ext, vocab=rep,
               probe=load("probe_report.json"))
with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=1, default=float)
print(json.dumps(dict(train={k: v for k, v in summary["train"].items() if k != "curve"},
                      eval={k: v for k, v in (ev or {}).items() if k != "per_class"}, extrapolation=ext),
                 indent=1, default=float))
PY
}

ALL="download serialize vocab bench baseline train evaluate untrained probe pair align profile2016 report"
run_steps() {
    local s
    for s in ${@:-$ALL}; do
        case " $ALL vocab_report " in
            *" $s "*) echo "=== step $s ==="; "step_$s" ;;
            *) echo "unknown step $s (steps: $ALL)" >&2; return 64 ;;
        esac
    done
}
run_steps "$@"; rc=$?; [ "$rc" -eq 0 ] || exit "$rc"; exit "$STATUS"
