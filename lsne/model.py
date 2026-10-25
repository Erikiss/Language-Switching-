"""Model loading, residual-stream hooks, injection, logit lens, entropy gradient and
steered generation for GPT-NeoX (Pythia) models.

Verified conventions (transformers 5.20): see docs/SPEC.md.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import HookSite, ModelSpec

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}

# Pre-registration §4: a run whose written vector deviates from the intended one by more
# than this (relative, FP32) is invalid.  Used for `valid` in the generation audit.
AUDIT_REL_ERR_MAX = 1e-4


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class ModelBundle:
    model: nn.Module
    tokenizer: object
    spec: ModelSpec
    device: torch.device

    # ---- structure -------------------------------------------------------------------
    @property
    def n_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    @property
    def d_model(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def n_heads(self) -> int:
        return int(self.model.config.num_attention_heads)

    @property
    def vocab(self) -> int:
        return int(self.model.config.vocab_size)

    @property
    def layers(self) -> list[nn.Module]:
        return list(self.model.gpt_neox.layers)

    @property
    def final_ln(self) -> nn.Module:
        return self.model.gpt_neox.final_layer_norm

    @property
    def lm_head(self) -> nn.Module:
        head = getattr(self.model, "lm_head", None)
        if head is None:
            head = getattr(self.model, "embed_out")
        return head

    @property
    def embed_in(self) -> nn.Module:
        return self.model.gpt_neox.embed_in

    @property
    def dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype

    # ---- helpers ---------------------------------------------------------------------
    def encode(self, text: str) -> torch.LongTensor:
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids))

    def module_for(self, site: HookSite) -> nn.Module:
        layer = self.layers[site.layer]
        if site.site in ("block_input", "block_output"):
            return layer
        if site.site == "mlp_output":
            return layer.mlp
        if site.site == "attn_output":
            return layer.attention
        raise ValueError(site)


def load_model(spec: ModelSpec) -> ModelBundle:
    device = resolve_device(spec.device)
    tok = AutoTokenizer.from_pretrained(spec.name, revision=spec.revision)
    model = AutoModelForCausalLM.from_pretrained(
        spec.name, revision=spec.revision, dtype=DTYPES[spec.dtype],
        attn_implementation=spec.attn_implementation)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return ModelBundle(model=model, tokenizer=tok, spec=spec, device=device)


# --------------------------------------------------------------------------------------
# low-level hook plumbing
# --------------------------------------------------------------------------------------
def _get_hidden(args, kwargs):
    if args:
        return args[0]
    return kwargs["hidden_states"]


def _set_hidden(args, kwargs, h):
    if args:
        return (h,) + tuple(args[1:]), kwargs
    kwargs = dict(kwargs)
    kwargs["hidden_states"] = h
    return args, kwargs


def _first_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    # transformers ModelOutput-like
    for v in out.values() if hasattr(out, "values") else []:
        if isinstance(v, torch.Tensor):
            return v
    raise TypeError(f"cannot find tensor in hook output of type {type(out)}")


def _replace_first(out, new):
    if isinstance(out, torch.Tensor):
        return new
    if isinstance(out, tuple):
        return (new,) + out[1:]
    if isinstance(out, list):
        return [new] + out[1:]
    raise TypeError(type(out))


def rms(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x.float().pow(2).mean(dim=dim).sqrt()


def make_dose_delta(h_ref: torch.Tensor, direction: torch.Tensor, epsilon: float) -> torch.Tensor:
    """State-2143 dose convention: delta = eps * RMS(h_ref) * dir / RMS(dir)."""
    d = direction.float()
    return (epsilon * rms(h_ref.float()) * d / rms(d).clamp_min(1e-30)).to(direction.dtype)


# --------------------------------------------------------------------------------------
# caching forward
# --------------------------------------------------------------------------------------
@dataclass
class CacheOut:
    logits: torch.Tensor                                   # [B,T,V] float32
    resid: dict[str, torch.Tensor] = field(default_factory=dict)   # key -> [B,T,d]
    attn: Optional[list[torch.Tensor]] = None              # list of [B,H,T,T]
    sites: list[HookSite] = field(default_factory=list)

    def resid_stack(self) -> torch.Tensor:
        """[n_sites, B, T, d] in the order of `self.sites`."""
        return torch.stack([self.resid[s.key] for s in self.sites], 0)


def all_block_inputs(mb: ModelBundle) -> list[HookSite]:
    """L+1 residual snapshots: block_input 0..L-1 and block_output of L-1."""
    L = mb.n_layers
    return [HookSite(l, "block_input") for l in range(L)] + [HookSite(L - 1, "block_output")]


def _resolve_sites(mb: ModelBundle, sites) -> list[HookSite]:
    if isinstance(sites, str):
        if sites == "all_block_inputs":
            return all_block_inputs(mb)
        if sites == "all_block_outputs":
            return [HookSite(l, "block_output") for l in range(mb.n_layers)]
        raise ValueError(sites)
    return list(sites)


@contextlib.contextmanager
def _recording(mb: ModelBundle, sites: list[HookSite], store: dict, detach: bool = True):
    handles = []
    try:
        for s in sites:
            mod = mb.module_for(s)
            if s.site == "block_input":
                def pre(module, args, kwargs, _k=s.key):
                    h = _get_hidden(args, kwargs)
                    store[_k] = h.detach() if detach else h
                handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
            else:
                def post(module, args, out, _k=s.key):
                    h = _first_tensor(out)
                    store[_k] = h.detach() if detach else h
                handles.append(mod.register_forward_hook(post))
        yield
    finally:
        for h in handles:
            h.remove()


def run_with_cache(mb: ModelBundle, input_ids: torch.Tensor, sites="all_block_inputs",
                   need_attn: bool = False, grad: bool = False, attention_mask=None) -> CacheOut:
    """Forward pass recording residual snapshots at `sites` (list of HookSite or a
    string preset). Logits are returned in float32."""
    sites = _resolve_sites(mb, sites)
    store: dict[str, torch.Tensor] = {}
    input_ids = input_ids.to(mb.device)
    ctx = contextlib.nullcontext() if grad else torch.no_grad()
    with ctx, _recording(mb, sites, store, detach=not grad):
        out = mb.model(input_ids=input_ids, attention_mask=attention_mask,
                       output_attentions=need_attn, use_cache=False)
    attn = [a for a in out.attentions] if need_attn else None
    return CacheOut(logits=out.logits.float(), resid=store, attn=attn, sites=sites)


def forward_logits(mb: ModelBundle, input_ids: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return mb.model(input_ids=input_ids.to(mb.device), use_cache=False).logits.float()


# --------------------------------------------------------------------------------------
# injection
# --------------------------------------------------------------------------------------
class Injector:
    """Context manager that modifies the residual stream at a HookSite.

    delta:
      * Tensor[d]            -> added (or written, mode="replace") at `positions`
                                (all positions if positions is None)
      * Tensor[B,T,d]        -> added elementwise to the full chunk (positions ignored)
      * Tensor[P,d]/[B,P,d]  -> with mode="replace": written at `positions`
      * Callable[[Tensor], Tensor] -> h_new = f(h) (full control)
    every_token=True adds a Tensor[d] delta to every position of every chunk the hook sees
    (used for steering during cached generation, where chunks are [B,1,d]).
    `audit` records intended vs. actually written change on the first call (Soll/Ist).
    A callable delta that carries its own `audit` dict (ScheduledSteer) is adopted
    instead as soon as it is non-empty, i.e. on the first *dosed* call.
    """

    def __init__(self, mb: ModelBundle, site: HookSite, delta, positions=None,
                 mode: str = "add", every_token: bool = False):
        self.mb, self.site, self.delta, self.mode = mb, site, delta, mode
        self.positions = positions
        self.every_token = every_token
        self.audit: dict = {}
        self._handle = None
        self.n_calls = 0

    # -- core edit --------------------------------------------------------------------
    def _edit(self, h: torch.Tensor) -> torch.Tensor:
        self.n_calls += 1
        if callable(self.delta) and not isinstance(self.delta, torch.Tensor):
            h_new = self.delta(h)
        else:
            d = self.delta
            d = d.to(device=h.device)
            if self.mode == "add":
                if d.dim() == 1:
                    if self.positions is None or self.every_token:
                        h_new = h + d.to(h.dtype)
                    else:
                        h_new = h.clone()
                        pos = self._pos_index(h)
                        h_new[:, pos, :] = h_new[:, pos, :] + d.to(h.dtype)
                elif d.dim() == 3:
                    h_new = h + d.to(h.dtype)
                else:
                    raise ValueError("add mode expects delta of dim 1 or 3")
            elif self.mode == "replace":
                if self.positions is None:
                    raise ValueError("replace mode needs positions")
                pos = self._pos_index(h)
                h_new = h.clone()
                if d.dim() == 1:
                    h_new[:, pos, :] = d.to(h.dtype)
                elif d.dim() == 2:
                    h_new[:, pos, :] = d.to(h.dtype)
                elif d.dim() == 3:
                    h_new[:, pos, :] = d.to(h.dtype)
                else:
                    raise ValueError("replace mode expects delta of dim 1, 2 or 3")
            else:
                raise ValueError(self.mode)
        if not self.audit:
            self._do_audit(h, h_new)
        return h_new

    def _pos_index(self, h):
        if isinstance(self.positions, slice):
            return self.positions
        return torch.as_tensor(list(self.positions), device=h.device, dtype=torch.long)

    def _do_audit(self, h, h_new):
        with torch.no_grad():
            if callable(self.delta) and not isinstance(self.delta, torch.Tensor):
                own = getattr(self.delta, "audit", None)
                if isinstance(own, dict):
                    # self-auditing callable: adopt its record once a dosed call happened,
                    # otherwise stay empty so the next call is tried again
                    if own:
                        self.audit = dict(own)
                    return
            written = (h_new.float() - h.float())
            if isinstance(self.delta, torch.Tensor) and self.mode == "add" and self.delta.dim() == 1 \
                    and self.positions is not None and not self.every_token:
                pos = self._pos_index(h)
                intended = self.delta.float().to(h.device).expand_as(written[:, pos, :])
                w = written[:, pos, :]
            elif isinstance(self.delta, torch.Tensor) and self.mode == "add" and self.delta.dim() == 1:
                intended = self.delta.float().to(h.device).expand_as(written)
                w = written
            elif isinstance(self.delta, torch.Tensor) and self.mode == "add" and self.delta.dim() == 3:
                intended = self.delta.float().to(h.device)
                w = written
            else:
                self.audit = dict(intended_norm=float("nan"), written_norm=float(written.norm()), rel_err=float("nan"))
                return
            err = (w - intended).norm()
            self.audit = dict(intended_norm=float(intended.norm()), written_norm=float(w.norm()),
                              rel_err=float(err / intended.norm().clamp_min(1e-30)),
                              dtype=str(h.dtype))

    # -- hooks -------------------------------------------------------------------------
    def __enter__(self):
        mod = self.mb.module_for(self.site)
        if self.site.site == "block_input":
            def pre(module, args, kwargs):
                h = _get_hidden(args, kwargs)
                return _set_hidden(args, kwargs, self._edit(h))
            self._handle = mod.register_forward_pre_hook(pre, with_kwargs=True)
        else:
            def post(module, args, out):
                return _replace_first(out, self._edit(_first_tensor(out)))
            self._handle = mod.register_forward_hook(post)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


# --------------------------------------------------------------------------------------
# lens / stats
# --------------------------------------------------------------------------------------
def logit_lens(mb: ModelBundle, resid: torch.Tensor) -> torch.Tensor:
    """Apply final LayerNorm + unembedding to a residual tensor [..., d] -> [..., V]."""
    with torch.no_grad():
        return mb.lm_head(mb.final_ln(resid.to(mb.dtype))).float()


def next_token_stats(logits_last: torch.Tensor, target_id: int,
                     freq_ids: Optional[torch.Tensor] = None) -> dict[str, torch.Tensor]:
    logp = torch.log_softmax(logits_last.float(), dim=-1)
    p = logp.exp()
    out = dict(
        logp_target=logp[:, target_id],
        p_target=p[:, target_id],
        entropy=-(p * logp).sum(-1),
        top1_id=logp.argmax(-1),
    )
    if freq_ids is not None:
        out["p_freq"] = p[:, torch.as_tensor(freq_ids, device=p.device)].sum(-1)
    return out


def entropy_gradient(mb: ModelBundle, input_ids: torch.Tensor, site: HookSite,
                     position: int) -> tuple[torch.Tensor, float]:
    """Gradient of the next-token entropy at the last position with respect to the
    residual at `site`, restricted to `position` (the State-2143 construction).
    Returns (grad[d] float32, entropy)."""
    input_ids = input_ids.to(mb.device)
    leaf: dict[str, torch.Tensor] = {}

    def edit(h):
        t = h.detach().clone().requires_grad_(True)
        leaf["h"] = t
        return t

    with torch.enable_grad():
        with Injector(mb, site, edit):
            out = mb.model(input_ids=input_ids, use_cache=False)
        logp = torch.log_softmax(out.logits[0, -1].float(), -1)
        H = -(logp.exp() * logp).sum()
        H.backward()
    g = leaf["h"].grad[0, position].detach().float().clone()
    return g, float(H.detach())


# --------------------------------------------------------------------------------------
# steered generation
# --------------------------------------------------------------------------------------
class ScheduledSteer:
    """Callable delta for Injector used during cached generation.

    Generated token k is *produced* by the forward position T0-1+k (the last prompt
    position produces token 0; afterwards each new token is fed back and produces the
    next one; the final generated token is never fed back). The dose eps(k) is therefore
    added at absolute position T0-1+k, i.e. "the dose at the step that produces token k".
    Prompt positions < T0-1 are untouched. Positions are tracked by counting the chunk
    lengths the hook sees (full prompt first, then one token per step).

    Soll/Ist audit: on the first *dosed* call (some eps(k) != 0 in the chunk) `audit`
    records, at the dosed positions and in fp32, intended = eps·RMS(h)·dir/RMS(dir) vs.
    written = h_new − h: `intended_norm`, `written_norm`, `rel_err` (= ‖written −
    intended‖/‖intended‖), `dtype` of h, `n_dosed`, absolute `positions` and their `eps`.
    It stays empty while no dose has been written.
    """

    def __init__(self, direction: torch.Tensor, eps_schedule: Callable[[int], float], prompt_len: int):
        self.dir = direction.float() / rms(direction.float()).clamp_min(1e-30)
        self.eps_schedule = eps_schedule
        self.prompt_len = prompt_len
        self.offset = 0
        self.eps_log: list[float] = []
        self.audit: dict = {}

    def reset(self):
        self.offset = 0
        self.eps_log = []
        self.audit = {}

    def __call__(self, h: torch.Tensor) -> torch.Tensor:
        B, n, d = h.shape
        pos = torch.arange(self.offset, self.offset + n, device=h.device)
        self.offset += n
        k = pos - (self.prompt_len - 1)
        eps = torch.tensor([self.eps_schedule(int(kk)) if kk >= 0 else 0.0 for kk in k.tolist()],
                           device=h.device, dtype=torch.float32)
        self.eps_log.extend([float(e) for e, kk in zip(eps.tolist(), k.tolist()) if kk >= 0])
        if float(eps.abs().max()) == 0.0:
            return h
        scale = eps[None, :, None] * rms(h)[..., None]          # [B,n,1] fp32
        intended = scale * self.dir.to(h.device)[None, None, :]  # [B,n,d] fp32
        h_new = (h.float() + intended).to(h.dtype)
        if not self.audit:
            self._do_audit(h, h_new, intended, eps, pos)
        return h_new

    def _do_audit(self, h, h_new, intended, eps, pos):
        with torch.no_grad():
            dosed = eps != 0
            i = intended[:, dosed, :]
            w = (h_new.float() - h.float())[:, dosed, :]
            i_norm = i.norm()
            self.audit = dict(intended_norm=float(i_norm), written_norm=float(w.norm()),
                              rel_err=float((w - i).norm() / i_norm.clamp_min(1e-30)),
                              dtype=str(h.dtype), n_dosed=int(dosed.sum()),
                              positions=[int(p) for p in pos[dosed].tolist()],
                              eps=[float(e) for e in eps[dosed].tolist()])


def generate_steered(mb: ModelBundle, input_ids: torch.Tensor, site: HookSite,
                     direction: Optional[torch.Tensor], eps_schedule: Callable[[int], float],
                     n_new: int = 64, n_samples: int = 8, temperature: float = 1.0,
                     seed: int = 0, top_p: float = 1.0) -> dict:
    """Sample `n_samples` continuations while steering every generated token according
    to `eps_schedule(k)`. Returns tokens [n_samples,n_new], per-token logprob (of the
    sampled token under the raw model distribution) and entropy, eps per step, and
    `audit`: the Soll/Ist record of the first dosed step (see ScheduledSteer) plus
    `valid = rel_err <= AUDIT_REL_ERR_MAX` (pre-registration §4). A run that wrote no
    dose (direction None or an all-zero schedule) has n_dosed = 0, rel_err NaN and
    valid True, since no vector was written."""
    input_ids = input_ids.to(mb.device)
    if input_ids.shape[0] == 1 and n_samples > 1:
        input_ids = input_ids.expand(n_samples, -1).contiguous()
    T0 = input_ids.shape[1]
    gen = torch.Generator(device=mb.device).manual_seed(seed)
    torch.manual_seed(seed)
    steer = ScheduledSteer(direction if direction is not None else torch.zeros(mb.d_model),
                           eps_schedule if direction is not None else (lambda k: 0.0), T0)
    kw = dict(max_new_tokens=n_new, min_new_tokens=n_new, do_sample=True, temperature=temperature,
              top_p=top_p, top_k=0, pad_token_id=mb.tokenizer.pad_token_id or 0,
              output_logits=True, return_dict_in_generate=True, use_cache=True,
              attention_mask=torch.ones_like(input_ids))
    with torch.no_grad(), Injector(mb, site, steer) as inj:
        out = mb.model.generate(input_ids, **kw)
    toks = out.sequences[:, T0:T0 + n_new]
    logits = torch.stack([l.float() for l in out.logits], 1)      # [B,n_new,V] raw
    logp = torch.log_softmax(logits, -1)
    tok_logp = logp.gather(-1, toks[..., None]).squeeze(-1)
    ent = -(logp.exp() * logp).sum(-1)
    eps = torch.tensor(steer.eps_log[:n_new]) if steer.eps_log else torch.zeros(n_new)
    audit = dict(inj.audit) if inj.audit else dict(
        intended_norm=float("nan"), written_norm=float("nan"), rel_err=float("nan"),
        dtype=str(mb.dtype), n_dosed=0, positions=[], eps=[])
    audit["valid"] = bool(audit["rel_err"] <= AUDIT_REL_ERR_MAX) if audit["n_dosed"] else True
    return dict(tokens=toks.cpu(), logprobs=tok_logp.cpu(), entropies=ent.cpu(), eps=eps,
                prompt_len=T0, audit=audit)
