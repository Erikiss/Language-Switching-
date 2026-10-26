"""BlockGPT-style causal LM on ITR token streams: model, packing, training (docs/TRACE_SOURCE_SPEC.md §6).

Architecture: **GPT-NeoX on purpose** (``rotary_pct = 0.25`` i.e. ``rope_parameters
.partial_rotary_factor = 0.25``, ``use_parallel_residual=True``, untied unembedding, like
Pythia), so the trained model loads with :func:`lsne.model.load_model` and the residual
hooks, :class:`lsne.model.Injector` and :mod:`lsne.transplant` work unchanged.

Documented deviations from BlockGPT (arXiv:2304.12749):

* no tree-path positional embeddings: the call-tree position is carried by explicit
  ``[D{depth}]`` tokens and the breadth-first order of the serialisation (rotary positions
  only);
* standard GPT packing (:func:`pack`): transactions are concatenated with ``[EOS]``
  separators and cut into fixed blocks, attention across transaction boundaries inside a
  block is **not** masked;
* the hidden size / head count / MLP width of BlockGPT are not published; ``"blockgpt"``
  uses 8 layers, d = 512, 8 heads, MLP 2 048, context 512 (the published layer count and
  context).

Training (:func:`train`): AdamW (decay on matrices only), linear warm-up + cosine decay to
``min_lr_ratio * lr``, gradient clipping, bf16 autocast on CUDA when available and fp32
on CPU, deterministic batch order (a pure function of ``seed`` and the step, so a resumed
run sees exactly the batches an uninterrupted one would), checkpoints
``out_dir/checkpoint-<step>`` (model, optimiser, scheduler, RNG, log) written atomically,
automatic resume from the newest complete checkpoint, final ``save_pretrained(out_dir)``
plus tokenizer, ``vocab.json``, ``train_log.csv`` and ``manifest.json``.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import time
import warnings
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .vocab import BOS_ID, EOS_ID, PAD_ID

CONFIGS = {
    "tiny": dict(hidden_size=64, num_hidden_layers=2, num_attention_heads=4, intermediate_size=256),
    "small": dict(hidden_size=256, num_hidden_layers=4, num_attention_heads=4, intermediate_size=1024),
    "blockgpt": dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8, intermediate_size=2048),
}
ROPE = dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25)
CKPT_PREFIX = "checkpoint-"
STATE_FILE = "trainer_state.json"
LOG_FILE = "train_log.csv"
LOG_COLUMNS = ["step", "train_loss", "val_loss", "lr", "tokens_seen", "elapsed_s"]


# ======================================================================================
# model
# ======================================================================================
def model_config(vocab_size: int, size: str = "blockgpt", max_position_embeddings: int = 512,
                 pad_token_id: int = PAD_ID, bos_token_id: int = BOS_ID, eos_token_id: int = EOS_ID,
                 attn_implementation: Optional[str] = "sdpa", **overrides):
    """``GPTNeoXConfig`` for one of :data:`CONFIGS` (Pythia conventions, see module doc)."""
    from transformers import GPTNeoXConfig

    if size not in CONFIGS:
        raise ValueError(f"size must be one of {sorted(CONFIGS)}, got {size!r}")
    kw = dict(CONFIGS[size])
    kw.update(vocab_size=int(vocab_size), max_position_embeddings=int(max_position_embeddings),
              rope_parameters=dict(ROPE), use_parallel_residual=True, tie_word_embeddings=False,
              hidden_act="gelu", attention_dropout=0.0, hidden_dropout=0.0, layer_norm_eps=1e-5,
              initializer_range=0.02, pad_token_id=int(pad_token_id), bos_token_id=int(bos_token_id),
              eos_token_id=int(eos_token_id), architectures=["GPTNeoXForCausalLM"])
    kw.update(overrides)
    if attn_implementation is not None:
        kw["attn_implementation"] = attn_implementation
    return GPTNeoXConfig(**kw)


def make_model(vocab_size: int, size: str = "blockgpt", max_position_embeddings: int = 512, seed: int = 0,
               **config_kw):
    """Randomly initialised ``GPTNeoXForCausalLM`` (seeded, the global RNG state is left untouched).
    ``config_kw`` -> :func:`model_config` (e.g. ``pad_token_id``, ``attn_implementation``)."""
    from transformers import GPTNeoXForCausalLM

    cfg = model_config(vocab_size, size, max_position_embeddings, **config_kw)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        model = GPTNeoXForCausalLM(cfg)
    return model


def param_counts(model_or_size, vocab_size: Optional[int] = None) -> dict:
    """``dict(total, embedding (in), unembedding (out), non_embedding)`` of a model or of a
    :data:`CONFIGS` size (``vocab_size`` needed then; counted on the meta device)."""
    if isinstance(model_or_size, str):
        if vocab_size is None:
            raise ValueError("vocab_size is required for a size name")
        with torch.device("meta"):
            model = make_model(int(vocab_size), model_or_size)
    else:
        model = model_or_size
    total = sum(p.numel() for p in model.parameters())
    emb = model.get_input_embeddings().weight.numel()
    out = model.get_output_embeddings().weight.numel()
    return dict(total=int(total), embedding=int(emb), unembedding=int(out), non_embedding=int(total - emb - out))


def flops_per_token(model_or_size, vocab_size: Optional[int] = None, block_size: int = 512) -> float:
    """Training FLOPs per token, ``6 * (non-embedding + unembedding params) + 6 * L * T * d``
    (Kaplan et al. 2020 approximation: forward 2N + attention, backward 2x forward)."""
    pc = param_counts(model_or_size, vocab_size)
    if isinstance(model_or_size, str):
        cfg = CONFIGS[model_or_size]
        L, d = cfg["num_hidden_layers"], cfg["hidden_size"]
    else:
        L, d = model_or_size.config.num_hidden_layers, model_or_size.config.hidden_size
    return float(6 * (pc["non_embedding"] + pc["unembedding"]) + 6 * L * int(block_size) * d)


# ======================================================================================
# packing
# ======================================================================================
def concat_with_eos(token_lists: Iterable[Sequence[int]], eos_id: int = EOS_ID, dtype=np.int32) -> np.ndarray:
    """Flat stream ``ids_1 [EOS] ids_2 [EOS] …`` (every document followed by ``[EOS]``)."""
    parts = []
    eos = np.asarray([int(eos_id)], dtype=dtype)
    for ids in token_lists:
        a = np.asarray(ids, dtype=dtype).reshape(-1)
        if a.size:
            parts.append(a)
            parts.append(eos)
    if not parts:
        return np.zeros(0, dtype=dtype)
    return np.concatenate(parts)


def blocks_from_stream(stream: np.ndarray, block_size: int = 512, dtype=torch.long) -> torch.Tensor:
    """Cut a flat token stream into ``[N, block_size]`` blocks (the remainder is dropped)."""
    bs = int(block_size)
    if bs < 2:
        raise ValueError("block_size must be >= 2")
    s = np.asarray(stream)
    n = s.size // bs
    return torch.from_numpy(np.ascontiguousarray(s[: n * bs].reshape(n, bs))).to(dtype)


def pack(token_lists: Iterable[Sequence[int]], block_size: int = 512, eos_id: int = EOS_ID,
         dtype=torch.long) -> torch.Tensor:
    """Standard GPT packing: concatenate the documents with ``[EOS]`` separators and cut the
    stream into ``[N, block_size]`` blocks (remainder dropped).  Attention across document
    boundaries inside a block is not masked (documented deviation)."""
    return blocks_from_stream(concat_with_eos(token_lists, eos_id), block_size, dtype)


# ======================================================================================
# helpers
# ======================================================================================
def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def _resolve_device(device) -> torch.device:
    if device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _autocast(device: torch.device, bf16: bool):
    if bf16 and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _labels(x: torch.Tensor, pad_id: Optional[int]) -> torch.Tensor:
    if pad_id is None:
        return x
    return x.masked_fill(x == int(pad_id), -100)


def _loss_sum(model, x: torch.Tensor, pad_id: Optional[int]) -> tuple[torch.Tensor, int]:
    """Summed next-token cross-entropy and the number of predicted tokens of a batch."""
    logits = model(input_ids=x, use_cache=False).logits
    y = _labels(x, pad_id)[:, 1:]
    lg = logits[:, :-1].float()
    loss = torch.nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]), y.reshape(-1), ignore_index=-100,
                                             reduction="sum")
    return loss, int((y != -100).sum())


@torch.no_grad()
def eval_loss(model, blocks: torch.Tensor, batch_size: int = 16, max_blocks: Optional[int] = None,
              bf16: bool = False) -> float:
    """Token-weighted mean next-token cross-entropy (nats) over ``blocks [N, T]``
    (the first ``max_blocks`` if given); ``[PAD]`` targets are ignored."""
    if blocks is None or len(blocks) == 0:
        return float("nan")
    dev = _device_of(model)
    was_training = model.training
    model.eval()
    pad = getattr(model.config, "pad_token_id", None)
    n = len(blocks) if max_blocks is None else min(len(blocks), int(max_blocks))
    tot, cnt = 0.0, 0
    try:
        for i in range(0, n, int(batch_size)):
            x = blocks[i:min(i + int(batch_size), n)].to(dev).long()
            with _autocast(dev, bf16):
                s, c = _loss_sum(model, x, pad)
            tot += float(s)
            cnt += c
    finally:
        model.train(was_training)
    return tot / cnt if cnt else float("nan")


class BatchOrder:
    """Deterministic sample order: sample ``p`` (global position) of epoch ``p // n`` is
    ``perm_e[p % n]`` with ``perm_e = randperm(n)`` seeded by ``(seed, epoch)``.  The batch of
    a step is a pure function of the step, which makes resuming exact."""

    def __init__(self, n: int, batch_size: int, seed: int = 0):
        if n <= 0:
            raise ValueError("no training blocks")
        self.n, self.bs, self.seed = int(n), int(batch_size), int(seed)
        self._perms: dict = {}

    def _perm(self, epoch: int) -> torch.Tensor:
        p = self._perms.get(epoch)
        if p is None:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
            p = torch.randperm(self.n, generator=g)
            self._perms = {epoch: p}          # keep one epoch in memory
        return p

    def batch(self, step: int, micro: int = 0, n_micro: int = 1) -> torch.Tensor:
        start = (int(step) * int(n_micro) + int(micro)) * self.bs
        idx = []
        for p in range(start, start + self.bs):
            idx.append(int(self._perm(p // self.n)[p % self.n]))
        return torch.tensor(idx, dtype=torch.long)


def lr_lambda(step: int, steps: int, warmup: int, min_lr_ratio: float = 0.1) -> float:
    """Linear warm-up over ``warmup`` steps, then cosine decay to ``min_lr_ratio``."""
    warmup = max(0, min(int(warmup), int(steps)))
    if warmup and step < warmup:
        return (step + 1) / warmup
    span = max(1, int(steps) - warmup)
    prog = min(1.0, max(0.0, (step - warmup) / span))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog)))


def _param_groups(model, weight_decay: float) -> list:
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    return [dict(params=decay, weight_decay=float(weight_decay)), dict(params=no_decay, weight_decay=0.0)]


def list_checkpoints(out_dir: str) -> list:
    """``[(step, path)]`` of complete checkpoints (with ``trainer_state.json``), ascending."""
    if not os.path.isdir(out_dir):
        return []
    out = []
    for name in os.listdir(out_dir):
        m = re.fullmatch(rf"{CKPT_PREFIX}(\d+)", name)
        if m and os.path.exists(os.path.join(out_dir, name, STATE_FILE)):
            out.append((int(m.group(1)), os.path.join(out_dir, name)))
    return sorted(out)


def _rng_state() -> dict:
    st = dict(torch=torch.get_rng_state(), numpy=np.random.get_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    if "cuda" in st and torch.cuda.is_available():
        with contextlib.suppress(Exception):
            torch.cuda.set_rng_state_all(st["cuda"])


def _save_checkpoint(out_dir: str, step: int, model, opt, sched, log_rows: list, hparams: dict,
                     keep: int) -> str:
    final = os.path.join(out_dir, f"{CKPT_PREFIX}{int(step)}")
    tmp = final + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    model.save_pretrained(tmp)
    torch.save(opt.state_dict(), os.path.join(tmp, "optimizer.pt"))
    torch.save(sched.state_dict(), os.path.join(tmp, "scheduler.pt"))
    torch.save(_rng_state(), os.path.join(tmp, "rng.pt"))
    with open(os.path.join(tmp, STATE_FILE), "w", encoding="utf-8") as fh:
        json.dump(dict(step=int(step), log=log_rows, hparams=hparams), fh)
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    if keep and keep > 0:
        for _, path in list_checkpoints(out_dir)[:-int(keep)]:
            shutil.rmtree(path, ignore_errors=True)
    return final


def _load_checkpoint(path: str, model, opt, sched) -> tuple[int, list, dict]:
    # from_pretrained applies transformers' own key mapping (save_pretrained writes the
    # GPT-NeoX head as the legacy "embed_out"), then the weights are copied into `model`
    saved = type(model).from_pretrained(path, dtype=next(model.parameters()).dtype)
    try:
        model.load_state_dict(saved.state_dict(), strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"checkpoint {path} does not match the model: {exc}") from exc
    del saved
    opt.load_state_dict(torch.load(os.path.join(path, "optimizer.pt"), map_location=_device_of(model),
                                   weights_only=False))
    sched.load_state_dict(torch.load(os.path.join(path, "scheduler.pt"), weights_only=False))
    _set_rng_state(torch.load(os.path.join(path, "rng.pt"), weights_only=False))
    with open(os.path.join(path, STATE_FILE), encoding="utf-8") as fh:
        st = json.load(fh)
    return int(st["step"]), list(st.get("log", [])), dict(st.get("hparams", {}))


def _log_frame(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=LOG_COLUMNS)
    return df.astype({"step": int, "tokens_seen": "int64"}) if len(df) else df


# ======================================================================================
# training
# ======================================================================================
def train(model, train_blocks: torch.Tensor, val_blocks: Optional[torch.Tensor], out_dir: str, lr: float = 3e-4,
          batch_size: int = 32, steps: int = 2000, warmup: int = 200, weight_decay: float = 0.1,
          grad_clip: float = 1.0, bf16: Optional[bool] = None, eval_every: int = 200, save_every: int = 1000,
          seed: int = 0, log_fn: Optional[Callable[[str], None]] = print, tokenizer=None, vocab=None,
          device=None, min_lr_ratio: float = 0.1, betas: tuple = (0.9, 0.95), eval_batch_size: Optional[int] = None,
          max_eval_blocks: Optional[int] = None, keep_checkpoints: int = 2, resume: bool = True,
          grad_accum: int = 1, manifest_extra: Optional[dict] = None) -> pd.DataFrame:
    """Train ``model`` on ``train_blocks [N, T]`` (int tensor, e.g. from :func:`pack`) and
    return ``DataFrame(step, train_loss, val_loss, lr, tokens_seen, elapsed_s)``.

    One row before the first update (``step 0``, validation loss of the initial model) and
    one every ``eval_every`` steps and at the end; ``train_loss`` is the mean training loss
    since the previous row, ``tokens_seen`` counts every position of every block consumed
    (``step * batch_size * grad_accum * T``).  ``bf16=None`` means bf16 autocast iff CUDA
    with bf16 support; on CPU training is always fp32 (``bf16=True`` there warns).
    Checkpoints every ``save_every`` steps and after the last step (``0`` disables; the newest
    ``keep_checkpoints`` are kept); with ``resume=True`` training continues from the newest
    complete checkpoint in ``out_dir`` (a finished run is therefore not repeated; raising
    ``steps`` continues it, with the cosine schedule recomputed for the new ``steps``).  The final model is saved with ``save_pretrained(out_dir)`` (fp32), plus
    ``tokenizer`` / ``vocab`` when given, ``train_log.csv`` and ``manifest.json``.
    """
    from ..io import write_manifest

    os.makedirs(out_dir, exist_ok=True)
    dev = _resolve_device(device) if device is not None else _device_of(model)
    model.to(dev)
    if bf16 is None:
        bf16 = bool(dev.type == "cuda" and torch.cuda.is_bf16_supported())
    elif bf16 and dev.type != "cuda":
        warnings.warn("train: bf16 autocast is only used on CUDA; training in fp32 on CPU")
        bf16 = False
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    steps, grad_accum = int(steps), max(1, int(grad_accum))
    if train_blocks is None or len(train_blocks) == 0:
        raise ValueError("train_blocks is empty (corpus shorter than one block?)")
    T = int(train_blocks.shape[1])
    pad = getattr(model.config, "pad_token_id", None)
    ebs = int(eval_batch_size or batch_size)
    hparams = dict(lr=float(lr), batch_size=int(batch_size), steps=steps, warmup=int(warmup),
                   weight_decay=float(weight_decay), grad_clip=float(grad_clip), bf16=bool(bf16), seed=int(seed),
                   min_lr_ratio=float(min_lr_ratio), betas=list(betas), grad_accum=grad_accum, block_size=T,
                   n_train_blocks=int(len(train_blocks)),
                   n_val_blocks=int(len(val_blocks)) if val_blocks is not None else 0, device=str(dev))

    opt = torch.optim.AdamW(_param_groups(model, weight_decay), lr=float(lr), betas=tuple(betas),
                            fused=(dev.type == "cuda"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_lambda(s, steps, warmup, min_lr_ratio))
    order = BatchOrder(len(train_blocks), batch_size, seed)
    say = log_fn or (lambda s: None)

    start, rows = 0, []
    ckpts = list_checkpoints(out_dir) if resume else []
    if ckpts:
        start, rows, old = _load_checkpoint(ckpts[-1][1], model, opt, sched)
        diff = {k: (old.get(k), hparams[k]) for k in ("batch_size", "block_size", "n_train_blocks", "seed",
                                                      "grad_accum") if old.get(k) != hparams[k]}
        if diff:
            warnings.warn(f"train: resuming from {ckpts[-1][1]} with changed settings {diff}; "
                          f"the batch order differs from an uninterrupted run")
        say(f"resumed from {ckpts[-1][1]} at step {start}")
    else:
        torch.manual_seed(int(seed))
    t0 = time.time() - (float(rows[-1]["elapsed_s"]) if rows else 0.0)

    def tokens_at(step: int) -> int:
        return int(step) * int(batch_size) * grad_accum * T

    def add_row(step: int, tr_loss: float):
        vl = eval_loss(model, val_blocks, ebs, max_eval_blocks, bf16) if val_blocks is not None else float("nan")
        row = dict(step=int(step), train_loss=float(tr_loss), val_loss=float(vl),
                   lr=float(opt.param_groups[0]["lr"]), tokens_seen=tokens_at(step),
                   elapsed_s=round(time.time() - t0, 3))
        rows.append(row)
        say(f"step {step:>6d}  train {row['train_loss']:.4f}  val {row['val_loss']:.4f}  lr {row['lr']:.2e}  "
            f"tokens {row['tokens_seen']:,}  {row['elapsed_s']:.0f}s")

    if not rows:
        add_row(0, float("nan"))
    model.train()
    run_sum, run_n = 0.0, 0
    for step in range(start, steps):
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for micro in range(grad_accum):
            x = train_blocks[order.batch(step, micro, grad_accum)].to(dev, non_blocking=True).long()
            with _autocast(dev, bf16):
                out = model(input_ids=x, labels=_labels(x, pad), use_cache=False)
            loss = out.loss / grad_accum
            loss.backward()
            loss_acc += float(loss.detach())
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
        opt.step()
        sched.step()
        run_sum += loss_acc
        run_n += 1
        done = step + 1
        if done % max(1, int(eval_every)) == 0 or done == steps:
            add_row(done, run_sum / max(run_n, 1))
            run_sum, run_n = 0.0, 0
            model.train()
        if save_every and (done % int(save_every) == 0 or done == steps):
            _save_checkpoint(out_dir, done, model, opt, sched, rows, hparams, keep_checkpoints)
    if rows and rows[-1]["step"] < start:       # resumed from a checkpoint beyond `steps`: report its state
        add_row(start, float("nan"))

    model.eval()
    model.save_pretrained(out_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(out_dir)
    if vocab is not None:
        vocab.save(out_dir)
    df = _log_frame(rows)
    df.to_csv(os.path.join(out_dir, LOG_FILE), index=False)
    wall = float(rows[-1]["elapsed_s"]) if rows else 0.0
    write_manifest(out_dir, kind="trace_lm", architecture="GPTNeoXForCausalLM",
                   config=model.config.to_dict(), params=param_counts(model), hparams=hparams,
                   final_val_loss=float(df["val_loss"].iloc[-1]) if len(df) else float("nan"),
                   initial_val_loss=float(df["val_loss"].iloc[0]) if len(df) else float("nan"),
                   tokens_seen=tokens_at(steps), wall_s=wall,
                   tokens_per_s=(tokens_at(steps) / wall) if wall > 0 else float("nan"),
                   deviations=["no tree-path embeddings ([D{depth}] tokens + breadth-first order instead)",
                               "GPT packing with [EOS] separators; cross-document attention not masked"],
                   data_source="ethPandaOps Xatu (CC BY 4.0)", **(manifest_extra or {}))
    return df


# ======================================================================================
# throughput (CPU measurements for the time estimates)
# ======================================================================================
def measure_throughput(size: str, vocab_size: int, block_size: int = 512, batch_size: int = 8, n_steps: int = 3,
                       n_warmup: int = 1, device: str = "cpu", optimizer_step: bool = False, seed: int = 0,
                       attn_implementation: str = "sdpa") -> dict:
    """Tokens/s of forward + backward (``optimizer_step=True`` adds the AdamW update) on random
    token blocks ``[batch_size, block_size]``; median over ``n_steps`` timed steps after
    ``n_warmup`` untimed ones.  Returns tokens_per_s, s_per_step, params, flops_per_token and the
    achieved FLOP/s (with :func:`flops_per_token`)."""
    dev = _resolve_device(device)
    model = make_model(vocab_size, size, max_position_embeddings=max(512, int(block_size)), seed=seed,
                       attn_implementation=attn_implementation).to(dev)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4) if optimizer_step else None
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, int(vocab_size), (int(batch_size), int(block_size)), generator=g).to(dev)
    times = []
    for i in range(int(n_warmup) + int(n_steps)):
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        out = model(input_ids=x, labels=x, use_cache=False)
        out.loss.backward()
        if opt is not None:
            opt.step()
        model.zero_grad(set_to_none=True)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if i >= int(n_warmup):
            times.append(time.perf_counter() - t)
    sps = float(np.median(times))
    ntok = int(batch_size) * int(block_size)
    fpt = flops_per_token(model, block_size=block_size)
    return dict(size=size, vocab_size=int(vocab_size), block_size=int(block_size), batch_size=int(batch_size),
                device=str(dev), threads=torch.get_num_threads(), optimizer_step=bool(optimizer_step),
                s_per_step=sps, tokens_per_s=ntok / sps, flops_per_token=fpt, achieved_flops=fpt * ntok / sps,
                **{f"params_{k}": v for k, v in param_counts(model).items()})
