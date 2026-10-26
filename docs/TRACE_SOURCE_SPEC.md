# Trace source model (Stage L source adapter): specification

Goal (DESIGN §4.9, docs/BLOCKGPT_FEASIBILITY.md): train a BlockGPT-style causal language
model on 2016 Ethereum execution traces (DAO era) from the public ethPandaOps Xatu Parquet
files, define "sharing" labels on transfer tables, extract a sharing direction inside the
trace model, and expose the model as a `lsne.transplant.SourceModel` so the existing
transplant machinery (alignment, residual substitution, KV planting) can move the feature
into Pythia.

Package: `lsne/chain/` (new; no other package depends on it). Data source licence:
CC BY 4.0, attribution "ethPandaOps Xatu" in docs and fixture README.

## 0. Verified facts about the data (25.10.2026, chunks 1718000 and 1920000)

URL pattern: `https://data.ethpandaops.io/xatu/mainnet/databases/default/<table>/1000/<start>.parquet`
(1 000 blocks per file, `<start>` a multiple of 1 000; chunk 1920000 = 580 KB traces).

| table | key columns (pyarrow types) |
|---|---|
| canonical_execution_traces | block_number u32, transaction_index u32, transaction_hash fixed_binary[66] (ASCII "0x…"), internal_index u32 (1-based node id within tx), action_from/action_to binary (hex str), action_value binary (DECIMAL string), action_gas u32, action_input binary (hex), action_call_type binary ∈ {call, call_code, delegate_call, static_call?, ""}, action_type ∈ {call, create, reward, suicide?}, action_reward_type, action_init, result_gas_used u32, result_output, result_address (create), trace_address binary ("" / None for root, children "1", "1_4", "1_4_0" — separator "_"), subtraces u32, error binary |
| canonical_execution_transaction | block_number, transaction_index, transaction_hash, nonce, from_address, to_address, value fixed_binary[32] (uint256 LITTLE-ENDIAN), input, gas_limit, gas_used, gas_price, success bool |
| canonical_execution_native_transfers | transaction_hash (often UPPER-CASE hex!), internal_index (= trace internal_index; value identical to the trace's action_value in 100 % of the DAO-drain rows), transfer_index, from_address/to_address (may be upper-case), value fixed_binary[32] little-endian |
| canonical_execution_erc20_transfers | transaction_hash, internal_index, log_index, erc20 (token contract), from_address, to_address, value fixed_binary[32] little-endian |
| canonical_execution_logs | transaction_hash, internal_index (per-table running number, NOT a trace pointer: emitter == trace action_to only 19 % of the time), log_index, address, topic0..topic3, data (hex) |
| canonical_execution_storage_diffs | transaction_hash, internal_index (running number, NOT a trace pointer), address, slot, from_value, to_value (hex) |
| canonical_execution_storage_reads | transaction_hash, internal_index, contract_address, slot, value (hex) |
| canonical_execution_block | block_number, block_date_time, extra_data_string ("dao-hard-fork" for 1 920 000–1 920 009) |

Block-level rows (block and uncle rewards) carry a transaction_hash of
"0x" followed by zero bytes; they are not transactions and are excluded from the corpus
(kept only for the fork-block statistics).

DAO drain transaction 0x0ec3f2488a93839524add10ea229e773f6bc891b4eb4794c3337d4495263790b
(block 1 718 497): 81 trace nodes, call-tree depth up to 28 (recursive splitDAO), 81 native
transfers from 6 senders to 9 receivers totalling 2 597.76 ETH, 31 logs, 32 storage diffs.
In chunk 1718000 (7 609 transactions; the block-level reward pseudo-transaction excluded)
the per-transaction count of distinct receivers over ALL native_transfers rows has median 1,
99th percentile 6, maximum 9, and 186 transactions reach ≥ 5. Those rows include zero-value
calls, reverted branches and call_code / delegate_call rows, so this is a structural count.
The economic fan-out used for the labels (round 2, `lsne/chain/labels.py`) counts only
value-carrying, non-reverted transfers of successful transactions; in the sampled 2016
slice of the first end-to-end run only 106 of 225 854 transactions had an economic fan-out
≥ 5 under the first-round definition. The fork's irregular state change does not appear in
the trace table (chunk 1920000 contains only reward rows); fork statistics need
balance_diffs.

## 1. `lsne/chain/xatu.py`

```python
XATU_BASE = "https://data.ethpandaops.io/xatu/mainnet/databases/default"
TABLES = ("traces", "transaction", "native_transfers", "erc20_transfers", "logs", "storage_diffs", "storage_reads", "block")
def chunk_url(table: str, start: int) -> str                      # table without the canonical_execution_ prefix
def chunk_starts(block_from: int, block_to: int) -> list[int]      # multiples of 1000 covering [from, to)
def download_chunks(tables, block_from, block_to, cache_dir="cache/xatu", every: int=1, max_workers: int=4, retries: int=4) -> DataFrame(table, start, path, bytes, ok, error)
    # resumable (skip existing non-empty files), atomic write (tmp + rename), exponential back-off; `every` = take every k-th chunk (sampling across 2016)
def read_chunk(table, start, cache_dir) -> list[dict]               # pyarrow -> python rows with NORMALISED fields:
    # every hex/binary string decoded to str and lower-cased; transaction_hash lower-case; little-endian fixed_binary[32] values -> int;
    # decimal-string action_value -> int; trace_address -> tuple[int,...] (() for root); missing -> None
def is_block_level(tx_hash: str) -> bool                           # "0x" + zeros
```

## 2. `lsne/chain/itr.py` — ITR serialisation (BlockGPT format, adapted)

```python
@dataclass class CallNode: path: tuple[int,...], call_type: str, action_type: str, frm: str, to: str|None, value: int, gas: int, gas_used: int, selector: str|None, args: list[str], error: str|None, created: str|None, children: list["CallNode"], logs: list[dict], state_writes: list[dict], state_reads: list[dict]
@dataclass class TxRecord: tx_hash, block_number, tx_index, frm, to, value, gas_used, success, root: CallNode, native_transfers: list[dict], erc20_transfers: list[dict], n_nodes, max_depth
def build_records(chunk_rows: dict[str, list[dict]]) -> list[TxRecord]
    # group all tables by tx hash (lower-case), build the call tree from trace_address (sorted lexicographically by the int tuple), skip block-level rows;
    # attach logs and storage diffs/reads to a node whose `to` (or `created`) equals the emitting/written address, else to the root
    # (their internal_index is not a node pointer, see §0). Round 2: candidates are taken in the order of the call tree's Euler tour
    # (a parent can emit after a child returns) and nodes that cannot have produced the event (reverted subtrees) are skipped.
def numeric_token(x: int) -> str        # 2 significant figures as in the paper: 0 -> "N0"; else "N{m}e{k}" with m the 2-digit mantissa 10..99 and k = exponent so that x ≈ m·10^k (e.g. 1254 -> "N13e2", 5e18 -> "N50e17")
def serialize(rec: TxRecord, vocab: "TraceVocab|None"=None, max_depth_token: int=32, max_args: int=4, include_state=True, include_logs=True) -> list[str]
    # breadth-first over the call tree (paper); per node:
    # [START] [CALL] [D{depth}] [CT:{call|delegate_call|call_code|static_call|create|suicide}] A:{from} A:{to} F:{selector} V:{numeric_token(value)} G:{numeric_token(gas)} [INs] arg… [OUTs] [ERR:{kind}]? [END]
    # then per attached state write: [START] [STATE] [W] A:{address} S:{slot_bucket} V:{numeric_token(to_value)} [END]; reads with [R] (optional, include_state);
    # per attached log: [START] [LOG] A:{address} T:{topic0} [END]
    # addresses / selectors / topics map to vocab tokens or [A_OOV]/[F_OOV]/[T_OOV] when a vocab is given (raw strings otherwise);
    # args: the first max_args 32-byte words of the input after the selector, each as A:{address} if it is a left-padded 20-byte address, else numeric_token(int) if < 2**64, else [WORD]
    # the whole transaction is wrapped as [TX] … [/TX]
def tree_text(rec) -> str   # human-readable indented dump for debugging
```

## 3. `lsne/chain/text.py` — paired English rendering for Pythia

```python
def render_text(rec: TxRecord, max_nodes: int=12, address_style: str="short") -> str
    # deterministic English description of the same transaction for the TARGET model, e.g.
    # "Account 0x6ebe…27ce sends 2.3 ETH to contract 0x53d2…8a3d. The contract calls 0xbb9b…9413 (function a9059cbb) and passes 1.2 ETH to 0x304a…3490. …";
    # values in ETH with 2 significant figures; depth expressed in words ("inside that call"); at most max_nodes nodes, then "and N further calls".
```

## 4. `lsne/chain/labels.py` — "sharing" labels and eras

```python
DAO_ADDRESSES = {"dao": "0xbb9bc244d798123fde783fcc1c72d3bb8c189413", "extra_balance": "0x807640a13483f8ac783c557fcdf27be11ea4ac7a", "creator": "0x4a574510c7014e4ae985403536074abe582adfc8", "withdraw_dao": "0xbf4ed7b27f1d666546e30d74d50d173d20bca754"}
ERA_BOUNDS = {"pre_dao": (0, 1_428_000), "dao_active": (1_428_000, 1_718_497), "drain_crisis": (1_718_497, 1_920_000), "fork": (1_920_000, 1_920_010), "post_fork": (1_920_010, 10**9)}
def era_of(block: int) -> str
def sharing_features(rec: TxRecord) -> dict(fan_out_eth, fan_in_eth, fan_out_erc20, fan_in_erc20, n_value_calls, n_state_writes, n_distinct_slots_written, dao_involved: bool, withdraw_dao_involved: bool, max_depth, n_nodes, era)
    # fan_out = number of distinct receivers, fan_in = distinct senders, from native (resp. erc20) transfers of the tx (addresses lower-cased)
def sharing_label(feat, k: int=5) -> str   # "distribution" (one sender, fan_out >= k), "pooling" (one receiver, fan_in >= k), "many_to_many" (both >= 2), "single" (fan_out == fan_in == 1), "none" (no value transfer), "other"
def probe_sets(records, labels_df, kind: str, n: int, seed: int=0) -> tuple[list[TxRecord], list[TxRecord], DataFrame]
    # kind "fanout": high (fan_out_eth >= k) vs single (fan_out_eth == 1), matched 1:1 on era and gas_used decile;
    # kind "dao": dao_involved vs not, same era (drain_crisis / post_fork), matched on gas decile; returns (positive, negative, match table)
```

## 5. `lsne/chain/vocab.py` — vocabulary and HF tokenizer

```python
SPECIAL = ["[PAD]", "[BOS]", "[EOS]", "[TX]", "[/TX]", "[START]", "[END]", "[CALL]", "[STATE]", "[LOG]", "[INs]", "[OUTs]", "[W]", "[R]", "[WORD]", "[A_OOV]", "[F_OOV]", "[T_OOV]", "[UNK]"] + depth tokens [D0]..[D32] + call-type, error and numeric tokens (all N{m}e{k} for m 10..99, k 0..77, plus N0)
@dataclass class TraceVocab: tokens: list[str], top_addresses, top_selectors, top_topics, meta
def build_vocab(records_iter, n_addresses=20000, n_selectors=5000, n_topics=2000, always_include=DAO_ADDRESSES.values()) -> TraceVocab
def make_tokenizer(vocab: TraceVocab, out_dir) -> PreTrainedTokenizerFast   # tokenizers.WordLevel + WhitespaceSplit, unk "[UNK]", pad/bos/eos set; save_pretrained(out_dir) so AutoTokenizer.from_pretrained(out_dir) works
def encode_records(records, vocab, tokenizer, max_len=512) -> list[list[int]]
def oov_rates(token_lists) -> dict
```

## 6. `lsne/chain/train.py` — model and training

```python
CONFIGS = {"tiny": dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256),
           "small": dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024),
           "blockgpt": dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048)}
def make_model(vocab_size: int, size: str="blockgpt", max_position_embeddings=512, seed=0) -> GPTNeoXForCausalLM
    # GPT-NeoX architecture ON PURPOSE (rotary_pct=0.25, use_parallel_residual=True, like Pythia) so lsne.model hooks, Injector and transplant work unchanged;
    # deviation from BlockGPT documented: tree-path embeddings are replaced by explicit [D{depth}] tokens + breadth-first order
def pack(token_lists, block_size=512, eos_id) -> Tensor[N, block_size]   # concatenate with [EOS] separators, cut into blocks (standard GPT packing, cross-document attention not masked — documented)
def train(model, train_blocks, val_blocks, out_dir, lr=3e-4, batch_size=32, steps=..., warmup=200, weight_decay=0.1, grad_clip=1.0, bf16: bool|None=None, eval_every=200, save_every=1000, seed=0, log_fn=print) -> DataFrame(step, train_loss, val_loss, lr, tokens_seen)
    # AdamW, cosine schedule, bf16 autocast on CUDA when available, fp32 on CPU; resumable from out_dir/checkpoint-*; final save_pretrained(out_dir) + tokenizer + vocab.json + train_log.csv + manifest
def eval_loss(model, blocks, batch_size) -> float
```

## 7. `lsne/chain/source.py` — SourceModel for the transplant

```python
class TraceLMSource:   # implements lsne.transplant.SourceModel
    def __init__(self, model_dir: str, device="auto")   # loads via lsne.model.load_model(ModelSpec(name=model_dir, revision=None or "main" handled for local dirs))
    name, d_model, n_layers
    def encode(self, item: TxRecord|list[str]) -> Tensor[1,T]     # serialize(item, vocab) -> tokenizer ids, truncated to 512
    def residuals(self, ids, layers=None) -> dict[int, Tensor[T, d]]
def sharing_direction(src: TraceLMSource, pos: list[TxRecord], neg: list[TxRecord], layer: int, readout="mean") -> Direction   # lsne.directions.diff_of_means (denoise_pcs=0 by default; report the denoised variant too)
def paired_items(records) -> list[tuple[TxRecord, str]]   # (record for the source, render_text(record) for Pythia) — input to lsne.transplant.layer_matching / fit_alignment
```

## 8. `lsne/chain/cli.py` and docs

`python -m lsne.chain.cli download --from 1428000 --to 2100000 --every 1 --tables traces,transaction,native_transfers,erc20_transfers,logs,storage_diffs --cache cache/xatu`
`… serialize --cache cache/xatu --from … --to … --out data/traces/itr` (jsonl shards with tokens + features + era)
`… vocab --in data/traces/itr --out data/traces/vocab`
`… train --in data/traces/itr --vocab data/traces/vocab --size blockgpt --out runs/trace_lm --steps 20000`
`… probe --model runs/trace_lm --in data/traces/itr --kind fanout|dao --layer all --out runs/trace_probe`
`… pair --in data/traces/itr --n 2000 --out runs/trace_pairs.jsonl`

`docs/TRACE_SOURCE.md` (German): what is built, data provenance and licence, the commands in order, Colab/A100 time estimates (measured on CPU for tiny/small, extrapolated for blockgpt and marked as such), the probe sets, how the direction enters Stage L, limitations (no tree-path embeddings, no ABI decoding, OOV addresses, cross-document packing).

## 9. Tests (CPU, no network unless marked)

Fixture: `tests/fixtures/xatu/` with small Parquet files cut from chunks 1718000 and 1920000
(the DAO drain transaction, ten ordinary transactions, two transactions touching
WithdrawDAO, one block-level reward row), plus `README` with the CC BY 4.0 attribution.
- read_chunk normalises upper-case hashes and addresses; little-endian values decode to the
  right ETH amount (drain total 2 597.76 ETH ± 0.01); trace_address parsing.
- build_records on the drain transaction: 81 nodes, max depth 28, fan-out 9, fan-in 6;
  every native transfer's internal_index maps to a node with the same value.
- serialize: wrapped in [TX]…[/TX], depth tokens present up to [D28], numeric_token edge
  cases (0, 1, 99, 100, 1254, 5·10^18), OOV mapping with a small vocab.
- render_text deterministic and mentions the root sender and receiver.
- labels: DAO drain is dao_involved, era drain_crisis; WithdrawDAO transactions are
  withdraw_dao_involved, era post_fork; probe_sets matched on era.
- vocab/tokenizer round trip; AutoTokenizer.from_pretrained(out_dir) loads it.
- make_model("tiny") loads through lsne.model.load_model and run_with_cache works;
  training 30 steps on the fixture lowers the validation loss.
- TraceLMSource implements the SourceModel protocol; lsne.transplant.layer_matching between
  the tiny trace LM and pythia-14m on paired items returns finite R² rows.
- network-marked: download_chunks for one tiny chunk, resume behaviour.
