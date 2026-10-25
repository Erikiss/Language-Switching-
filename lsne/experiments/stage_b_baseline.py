"""Stage B - baselines and directions (pilot).

Tables
------
baseline                 next-token statistics of every variant at the last context
                         position: logp_target, p_target, entropy, p_freq, top1 token, rank
direction_meta           one row per direction of the bank (name, family, site, scalar meta)
cosines                  cosine matrix of all directions (row name in column ``name``)
unembed                  tokens promoted / suppressed by each direction through LN + unembedding
pfreq_entropy_tokens     per variant and position: next-token entropy and mass on the 100
                         most frequent tokens (frequent-token-suppression analysis)
pfreq_entropy_by_class   the same aggregated by K, C, token class and tail flag
                         (means, sds, Pearson r between entropy and p_freq)
sae_features / sae_summary   only when the SAE is enabled (``Limits.use_sae``)

The direction bank itself is written to ``directions.npz`` + ``directions_meta.json``
(see :func:`lsne.experiments.common.save_bank`) and shared through ``ctx['bank']``.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch

from ..config import RunConfig
from ..directions import cosine_table, unembed_projection
from ..model import ModelBundle, next_token_stats, run_with_cache
from . import common as C


# --------------------------------------------------------------------------------------
# per-variant statistics
# --------------------------------------------------------------------------------------
def baseline_rows(mb: ModelBundle, variants, freq_ids, enabled: bool = True) -> tuple[list[dict], list[dict]]:
    """Returns (baseline rows, per-position rows)."""
    freq_t = torch.as_tensor(freq_ids, dtype=torch.long, device=mb.device) if freq_ids else None
    base_rows: list[dict] = []
    tok_rows: list[dict] = []
    for v in C.progress(variants, "B baseline", enabled=enabled):
        ids = C.ids_tensor(mb, v)
        with torch.no_grad():
            logits = run_with_cache(mb, ids, []).logits[0]                       # [T, V]
            st = next_token_stats(logits[-1:], int(v.target_id), freq_t)
            logp_all = torch.log_softmax(logits, -1)
            p_all = logp_all.exp()
            ent = -(p_all * logp_all).sum(-1)                                      # [T]
            p_freq = p_all[:, freq_t].sum(-1) if freq_t is not None else torch.full_like(ent, float("nan"))
            nxt = torch.as_tensor(v.ids[1:], device=logits.device)
            logp_next = logp_all[:-1].gather(-1, nxt[:, None]).squeeze(-1)        # [T-1]
            rank = int((logp_all[-1] > logp_all[-1, int(v.target_id)]).sum())
        top1 = int(st["top1_id"][0])
        base_rows.append(dict(C.variant_row(v), target_id=int(v.target_id), target=v.target,
                              logp_target=float(st["logp_target"][0]), p_target=float(st["p_target"][0]),
                              entropy=float(st["entropy"][0]),
                              p_freq=float(st["p_freq"][0]) if "p_freq" in st else float("nan"),
                              top1_id=top1, top1_token=mb.tokenizer.decode([top1]), target_rank=rank,
                              n_words=int(v.annot.get("n_words", 0)), F100=float(v.annot.get("F100", 0.0)),
                              n_tokens_D=int(v.annot.get("n_tokens_D", 0)),
                              last_switch_pos=int(v.annot.get("last_switch_pos", -1))))
        a = v.annot
        ts = int(v.tail_start)
        ent_np, pf_np, ln_np = ent.cpu().numpy(), p_freq.cpu().numpy(), logp_next.cpu().numpy()
        for t in range(v.n_tokens):
            tok_rows.append(dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=int(v.K), C=str(v.C),
                                 position=t, klass=a["klass"][t], lang=a["lang"][t], unit=int(a["unit"][t]),
                                 is_tail=t >= ts, dist_to_last_switch=int(a["dist_to_last_switch"][t]),
                                 entropy=float(ent_np[t]), p_freq=float(pf_np[t]),
                                 logp_next=float(ln_np[t]) if t < v.n_tokens - 1 else float("nan")))
    return base_rows, tok_rows


def aggregate_pfreq(tok: pd.DataFrame) -> pd.DataFrame:
    cols = ["K", "C", "klass", "is_tail"]
    if tok.empty:
        return pd.DataFrame(columns=cols + ["n", "entropy_mean", "entropy_sd", "p_freq_mean", "p_freq_sd", "r_entropy_pfreq"])
    rows = []
    for key, g in tok.groupby(cols, sort=True):
        rows.append(dict(zip(cols, key), n=int(len(g)),
                         entropy_mean=float(g["entropy"].mean()), entropy_sd=float(g["entropy"].std(ddof=0)),
                         p_freq_mean=float(g["p_freq"].mean()), p_freq_sd=float(g["p_freq"].std(ddof=0)),
                         r_entropy_pfreq=C.pearson(g["entropy"], g["p_freq"])))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# direction report
# --------------------------------------------------------------------------------------
def direction_meta_table(bank: C.DirectionBank) -> pd.DataFrame:
    rows = []
    for name, d in bank.items():
        scal = {k: v for k, v in d.meta.items() if isinstance(v, (int, float, str, bool, np.integer, np.floating))}
        rest = {k: v for k, v in d.meta.items() if k not in scal and k != "vec_plain"}
        rows.append(dict(name=name, dir_family=C.dir_family(name), site=d.site.key, d=d.d, **scal,
                         meta_json=json.dumps(C.to_jsonable(rest), ensure_ascii=False)[:2000]))
    return pd.DataFrame(rows)


def cosine_df(bank: C.DirectionBank) -> pd.DataFrame:
    ct = cosine_table(list(bank.values()))
    ct.insert(0, "name", list(ct.index))
    return ct.reset_index(drop=True)


def unembed_table(mb: ModelBundle, bank: C.DirectionBank, top_k: int = 25) -> pd.DataFrame:
    rows = []
    for name, d in bank.items():
        if C.dir_family(name) == "rand" and not name.endswith("00"):
            continue  # one random direction is enough for the report
        up = unembed_projection(mb, d, top_k=top_k)
        for kind in ("promoted", "suppressed"):
            ids = up[f"{kind}_ids"]
            for r, ((tok, score), tid) in enumerate(zip(up[kind], ids)):
                rows.append(dict(direction=name, dir_family=C.dir_family(name), kind=kind, rank=r,
                                 token_id=int(tid), token=tok, score=float(score)))
    return pd.DataFrame(rows, columns=["direction", "dir_family", "kind", "rank", "token_id", "token", "score"])


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Baselines, direction bank, geometry report and the P_freq-vs-entropy relation."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    allv = C.all_variants(variants, original)
    site = C.dossier_site(mb)

    bank = C.direction_bank(cfg, mb, variants, site, original=original, limits=limits, sae=kw.get("sae"),
                            axis_npz=kw.get("axis_npz"))
    sig = C.bank_signature(variants, site, cfg, limits, original)
    if ctx is not None:
        ctx["bank"], ctx["bank_signature"] = bank, sig
    C.save_bank(bank, run_dir)

    base_rows, tok_rows = baseline_rows(mb, allv, bank.freq_ids, enabled=limits.progress)
    tok = pd.DataFrame(tok_rows, columns=["tag", "family", "formulation", "K", "C", "position", "klass", "lang", "unit",
                                          "is_tail", "dist_to_last_switch", "entropy", "p_freq", "logp_next"])
    tables = {
        "baseline": pd.DataFrame(base_rows),
        "direction_meta": direction_meta_table(bank),
        "cosines": cosine_df(bank),
        "unembed": unembed_table(mb, bank),
        "pfreq_entropy_tokens": tok,
        "pfreq_entropy_by_class": aggregate_pfreq(tok),
    }
    if bank.sae_table is not None:
        tables["sae_features"] = bank.sae_table
        info = dict(bank.meta.get("sae", {}))
        tables["sae_summary"] = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                                               for k, v in info.items()}])
    return C.finish_stage(run_dir, "B", cfg, mb, tables, limits=limits, t0=t0, site=site,
                          bank_signature=sig, directions=list(bank.keys()), bank_meta=bank.meta,
                          n_variants=len(variants), original=(original.info() if original is not None else None))
