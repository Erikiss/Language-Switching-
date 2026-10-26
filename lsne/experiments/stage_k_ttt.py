"""Stage K - test-time training as a structural injection (DESIGN section 4.8 d;
PREREGISTRATION contrast 17).

Corpus: ``Limits.k_corpus`` (default ``data/ttt/dao_fork``, the client's DAO-fork texts;
``--ttt-corpus`` on the CLI).  When the folder is missing or empty the loader falls back to
the public-domain placeholder and every row is marked ``is_placeholder=True``: such a run
is a function test, not a finding.  Control arms (DESIGN 4.8 d 3): ``neutral``
(``data/ttt/neutral``), ``nofork`` (``data/ttt/nofork``; both fall back to the placeholder
too, flagged) and ``permuted`` (token-permuted injection corpus).  Training conventions
of :mod:`lsne.ttt` (AdamW, grad clipping 1.0, FP32, seeded); the passed model is never
mutated (deep copies / restored snapshots).

Tables
------
ttt_corpora        one row per arm: source, is_placeholder, n_docs, n_snippets, n_tokens
ttt_memorisation   contrast 17 (i): :func:`lsne.ttt.ttt_memorisation_table` - per
                   checkpoint x arm x snippet the NLL at step 0, ``steps_to_threshold``
                   (first step with NLL <= ``Limits.k_threshold``), final NLL and AUC drop
                   (``Limits.k_mem_steps`` steps at ``Limits.k_mem_lr``; extra checkpoints via
                   ``Limits.k_checkpoints`` / ``--ttt-checkpoints``)
ttt_memorisation_boot  bootstrap (over snippets) of ``steps_to_threshold`` and ``auc_drop``
                   per (checkpoint, arm) and of the injection-minus-control differences
ttt_training       loss and gradient norm per step of every :func:`test_time_train` arm
ttt_before_after   contrast 17 (ii)+(iii): per arm and phase (before = the frozen model,
                   after = the arm's trained copy) the measures on the same stimulus grid:
                   ``logp_target`` per K=0 / K=8-C1 variant, ``T_K8`` per formulation,
                   ``B_lang_dom`` (dose ``Limits.k_eps`` along the *pre-TTT* ``lang_dom`` axis,
                   written into the trained copy), ``sharing_jaccard`` (K0 vs K8C1) and
                   ``network_activity`` (frac_active, mean_mass, participation_ratio; SAE as
                   in stage J, ``sae_source``), ``head_delta`` (mean-ablation effect of stage
                   I's worst head, when selected) and ``log_volume`` / ``participation_ratio``
                   of the K=0 and K=8/C1 clouds at the dossier site (stage G's clouds)
ttt_effects        per arm x measure x condition: after - before, and for the control arms
                   ``delta_injection_minus_control`` (an effect counts only when it exceeds
                   all three controls; ``exceeds_all_controls`` on the injection rows)
"""
from __future__ import annotations

import gc
import time
import warnings
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .. import ttt
from ..config import RunConfig, HookSite
from ..geometry import log_volume, participation_ratio
from ..heads import ablate_heads
from ..model import Injector, ModelBundle, forward_logits, make_dose_delta, run_with_cache
from ..sharing import active_features, jaccard, network_activity
from . import common as C

ARMS = ("injection", "neutral", "nofork", "permuted")
MEASURE_COLS = ["measure", "condition", "value"]


# --------------------------------------------------------------------------------------
# corpora
# --------------------------------------------------------------------------------------
def load_corpora(mb: ModelBundle, limits: C.Limits, corpus_path: Optional[str], seed: int) -> tuple[dict, pd.DataFrame]:
    """Injection corpus and the three control arms (``Corpus`` objects) + the corpora table."""
    path = corpus_path or limits.k_corpus
    base = C._REPO_ROOT / "data" / "ttt"
    corp = {"injection": ttt.load_corpus(path, mb.tokenizer, max_docs=limits.k_max_docs, seq_len=limits.k_seq_len)}
    for arm in ("neutral", "nofork"):
        corp[arm] = ttt.load_corpus(str(base / arm), mb.tokenizer, max_docs=limits.k_max_docs, seq_len=limits.k_seq_len)
    corp["permuted"] = ttt.permute_corpus(corp["injection"], seed=seed)
    rows = []
    for arm, c in corp.items():
        m = dict(getattr(c, "meta", {}) or {})
        rows.append(dict(arm=arm, requested=str(m.get("requested", "")), source=str(m.get("source", "")),
                         is_placeholder=bool(m.get("is_placeholder", arm == "permuted" and corp["injection"].is_placeholder)),
                         n_docs=int(m.get("n_docs", 0)), n_snippets=len(c), n_tokens=int(sum(int(s.numel()) for s in c)),
                         seq_len=int(limits.k_seq_len), control=m.get("control")))
    return corp, pd.DataFrame(rows, columns=["arm", "requested", "source", "is_placeholder", "n_docs", "n_snippets", "n_tokens",
                                             "seq_len", "control"])


# --------------------------------------------------------------------------------------
# (i) memorisation speed
# --------------------------------------------------------------------------------------
def memorisation_tables(mb: ModelBundle, corp: dict, limits: C.Limits, seed: int, n_boot: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    controls = {arm: list(corp[arm]) for arm in ("neutral", "nofork", "permuted")}
    ckpts = list(limits.k_checkpoints) if limits.k_checkpoints else None
    df = ttt.ttt_memorisation_table(mb, corp["injection"], controls, checkpoints=ckpts, arm_name="injection",
                                    max_snippets=limits.k_max_snippets, lr=limits.k_mem_lr, steps=limits.k_mem_steps,
                                    threshold=limits.k_threshold, seed=seed)
    df["is_placeholder"] = [bool(getattr(corp[a], "is_placeholder", False)) for a in df["arm"]]
    df["reached_threshold"] = df["steps_to_threshold"].notna()
    rows = []
    rng = np.random.default_rng(int(seed))
    for (ck, arm), g in df.groupby(["checkpoint", "arm"], sort=True):
        for q in ("steps_to_threshold", "auc_drop", "final_nll"):
            v = pd.to_numeric(g[q], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            n = int(v.size)
            if n == 0:
                rows.append(dict(checkpoint=ck, arm=arm, quantity=q, mean=np.nan, ci_low=np.nan, ci_high=np.nan, n=0, frac_reached=np.nan))
                continue
            boots = v[rng.integers(0, n, size=(int(n_boot), n))].mean(1) if n > 1 else np.repeat(v.mean(), 2)
            rows.append(dict(checkpoint=ck, arm=arm, quantity=q, mean=float(v.mean()), ci_low=float(np.percentile(boots, 2.5)),
                             ci_high=float(np.percentile(boots, 97.5)), n=n,
                             frac_reached=float(g["reached_threshold"].mean())))
    boot = pd.DataFrame(rows, columns=["checkpoint", "arm", "quantity", "mean", "ci_low", "ci_high", "n", "frac_reached"])
    # injection minus control (two-sided; paired by snippet index where both exist)
    drows = []
    for ck, g in df.groupby("checkpoint", sort=True):
        inj = g[g["arm"] == "injection"].set_index("snippet")
        for arm in ("neutral", "nofork", "permuted"):
            con = g[g["arm"] == arm].set_index("snippet")
            idx = inj.index.intersection(con.index)
            for q in ("steps_to_threshold", "auc_drop", "final_nll"):
                a = pd.to_numeric(inj.loc[idx, q], errors="coerce").to_numpy(dtype=float)
                b = pd.to_numeric(con.loc[idx, q], errors="coerce").to_numpy(dtype=float)
                diff = a - b
                diff = diff[np.isfinite(diff)]
                n = int(diff.size)
                if n == 0:
                    drows.append(dict(checkpoint=ck, arm=f"injection-{arm}", quantity=q, mean=np.nan, ci_low=np.nan, ci_high=np.nan, n=0,
                                      frac_reached=np.nan))
                    continue
                boots = diff[rng.integers(0, n, size=(int(n_boot), n))].mean(1) if n > 1 else np.repeat(diff.mean(), 2)
                drows.append(dict(checkpoint=ck, arm=f"injection-{arm}", quantity=q, mean=float(diff.mean()),
                                  ci_low=float(np.percentile(boots, 2.5)), ci_high=float(np.percentile(boots, 97.5)), n=n, frac_reached=np.nan))
    return df, pd.concat([boot, pd.DataFrame(drows, columns=boot.columns)], ignore_index=True)


# --------------------------------------------------------------------------------------
# (ii)/(iii) before / after
# --------------------------------------------------------------------------------------
def make_measure_fn(mb_ref: ModelBundle, variants, site: HookSite, lang_dom, eps: float, sae, sae_source: str, worst: dict,
                    clouds: Optional[dict]):
    """``measure_fn(bundle) -> DataFrame(measure, condition, value)`` on the stimulus grid."""
    k0 = C.k_subset(variants, 0)
    k8 = C.k_subset(variants, 8, "C1")
    vs = k0 + k8
    vec = lang_dom.vec if lang_dom is not None else None
    cloud_k0 = [v for (fid, K, Cc), g in (clouds or {}).items() if int(K) == 0 for v in g]
    cloud_k8 = [v for (fid, K, Cc), g in (clouds or {}).items() if int(K) == 8 and str(Cc) == "C1" for v in g]

    def fn(b: ModelBundle) -> pd.DataFrame:
        rows = []
        logp: dict[str, float] = {}
        states: dict[str, list] = {"K0": [], "K8C1": []}
        for v in vs:
            ids = C.ids_tensor(b, v)
            T = int(ids.shape[1])
            with torch.no_grad():
                cache = run_with_cache(b, ids, [site])
            lp = float(torch.log_softmax(cache.logits[0, -1], -1)[int(v.target_id)])
            logp[v.tag] = lp
            rows.append(dict(measure="logp_target", condition=v.tag, value=lp))
            h = cache.resid[site.key][0, T - 1].detach().float().cpu()
            states["K0" if int(v.K) == 0 else "K8C1"].append(h.numpy())
            if vec is not None:
                delta = make_dose_delta(h, vec, float(eps))
                with torch.no_grad(), Injector(b, site, delta.to(b.dtype), positions=[T - 1], mode="add"):
                    logits = forward_logits(b, ids)
                lp2 = float(torch.log_softmax(logits[0, -1], -1)[int(v.target_id)])
                rows.append(dict(measure="B_lang_dom", condition=v.tag, value=lp2 - lp))
        for v8 in k8:
            ref = next((v for v in k0 if (v.family_id, v.formulation) == (v8.family_id, v8.formulation)), None)
            if ref is not None:
                rows.append(dict(measure="T_K8", condition=f"{v8.family_id}{v8.formulation}", value=logp[v8.tag] - logp[ref.tag]))
        if sae is not None and states["K0"] and states["K8C1"]:
            X0, X8 = np.stack(states["K0"]), np.stack(states["K8C1"])
            s0, s8 = active_features(sae, X0), active_features(sae, X8)
            rows.append(dict(measure="sharing_jaccard", condition="K0|K8C1", value=jaccard(s0, s8)))
            for cond, X in (("K0", X0), ("K8C1", X8)):
                act = network_activity(sae, X)
                for q in ("frac_active", "mean_mass", "participation_ratio", "n_active_mean"):
                    rows.append(dict(measure=f"activity_{q}", condition=cond, value=float(act[q])))
        if worst.get("layer") is not None:
            layer, head = int(worst["layer"]), int(worst["head"])
            deltas = []
            for v in vs:
                ids = C.ids_tensor(b, v)
                with ablate_heads(b, layer, [head], mode="mean"):
                    lpa = float(torch.log_softmax(forward_logits(b, ids)[0, -1], -1)[int(v.target_id)])
                deltas.append(lpa - logp[v.tag])
                rows.append(dict(measure="head_delta", condition=v.tag, value=lpa - logp[v.tag]))
            rows.append(dict(measure="head_delta_mean", condition=f"L{layer}H{head}", value=float(np.mean(deltas))))
        for cond, cv in (("K0", cloud_k0), ("K8C1", cloud_k8)):
            if len(cv) >= 2:
                G = np.stack([C.last_state(b, v, site).numpy() for v in cv], 0)
                rows.append(dict(measure="log_volume", condition=cond, value=log_volume(G)))
                rows.append(dict(measure="participation_ratio", condition=cond, value=participation_ratio(G)))
        return pd.DataFrame(rows, columns=MEASURE_COLS)
    return fn


def before_after_tables(mb: ModelBundle, corp: dict, measure_fn, limits: C.Limits, seed: int, sae_source: str,
                        enabled: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    before = measure_fn(mb)
    before.insert(0, "arm", "")
    before.insert(0, "phase", "before")
    frames = [before]
    train_rows = []
    for arm in C.progress(list(ARMS), "K test-time training", enabled=enabled):
        corpus = corp[arm]
        if len(corpus) == 0:
            warnings.warn(f"arm {arm!r}: empty corpus, skipped")
            continue
        mb2 = ttt.test_time_train(mb, corpus, lr=limits.k_train_lr, steps=limits.k_train_steps, batch=limits.k_train_batch, seed=seed,
                                  control=None, train_params="all")
        st = getattr(mb2, "ttt_stats", {})
        for k, (l, g) in enumerate(zip(st.get("loss", []), st.get("grad_norm", []))):
            train_rows.append(dict(arm=arm, step=k + 1, loss=float(l), grad_norm=float(g), n_snippets=int(st.get("n_snippets", 0)),
                                   n_tokens=int(st.get("n_tokens", 0)), is_placeholder=bool(getattr(corpus, "is_placeholder", False))))
        after = measure_fn(mb2)
        after.insert(0, "arm", arm)
        after.insert(0, "phase", "after")
        frames.append(after)
        del mb2
        gc.collect()
    ba = pd.concat(frames, ignore_index=True)
    ba["sae_source"] = sae_source
    ba["is_placeholder"] = [bool(getattr(corp.get(a), "is_placeholder", False)) if a else False for a in ba["arm"]]
    # effects: after - before per arm x measure x condition; injection minus each control
    b0 = ba[ba["phase"] == "before"].set_index(["measure", "condition"])["value"]
    erows = []
    deltas: dict[tuple, dict[str, float]] = {}
    for arm in ARMS:
        a = ba[(ba["phase"] == "after") & (ba["arm"] == arm)].set_index(["measure", "condition"])["value"]
        for key in a.index:
            if key in b0.index:
                d = float(a[key] - b0[key])
                deltas.setdefault(key, {})[arm] = d
    for key, by_arm in deltas.items():
        inj = by_arm.get("injection", np.nan)
        for arm, d in by_arm.items():
            diff = (inj - d) if arm != "injection" else np.nan
            exceeds = None
            if arm == "injection":
                cons = [by_arm[c] for c in ("neutral", "nofork", "permuted") if c in by_arm]
                exceeds = bool(cons and np.isfinite(inj) and all(abs(inj) > abs(c) for c in cons))
            erows.append(dict(arm=arm, measure=key[0], condition=key[1], before=float(b0[key]), after=float(b0[key] + d), delta=d,
                              delta_injection_minus_control=diff, exceeds_all_controls=exceeds))
    eff = pd.DataFrame(erows, columns=["arm", "measure", "condition", "before", "after", "delta", "delta_injection_minus_control",
                                       "exceeds_all_controls"])
    train = pd.DataFrame(train_rows, columns=["arm", "step", "loss", "grad_norm", "n_snippets", "n_tokens", "is_placeholder"])
    return ba, eff, train


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------
def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Memorisation speed and before/after effects of test-time training; see the module docstring."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    sub = C.Substages("K", verbose=limits.progress)
    variants, _checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    site = C.dossier_site(mb)
    bank = C.get_bank(cfg, mb, variants, site, ctx, limits, original=original, sae=kw.get("sae"))
    sae, sae_source = C.get_sae(mb, bank, limits, kw)
    empty = pd.DataFrame()
    if kw.get("ttt_checkpoints"):
        limits.k_checkpoints = tuple(kw["ttt_checkpoints"])

    corp, corpora = sub.run("corpora", lambda: load_corpora(mb, limits, kw.get("ttt_corpus"), cfg.seed), ({}, empty))
    if not corp:
        raise RuntimeError("no corpus could be loaded (not even the placeholder)")
    mem, mem_boot = sub.run("memorisation", lambda: memorisation_tables(mb, corp, limits, cfg.seed, limits.n_boot), (empty, empty))
    worst = ctx.get("worst_head", {}) if isinstance(ctx, dict) else {}
    clouds = ctx.get("g_groups") if isinstance(ctx, dict) else None
    fn = make_measure_fn(mb, variants, site, bank.get("lang_dom"), limits.k_eps, sae, sae_source, worst or {}, clouds)
    ba, eff, train = sub.run("before_after", lambda: before_after_tables(mb, corp, fn, limits, cfg.seed, sae_source, enabled=limits.progress),
                             (empty, empty, empty))

    tables = {"ttt_corpora": corpora, "ttt_memorisation": mem, "ttt_memorisation_boot": mem_boot, "ttt_training": train,
              "ttt_before_after": ba, "ttt_effects": eff}
    sub.write_tracebacks(run_dir)
    return C.finish_stage(run_dir, "K", cfg, mb, tables, limits=limits, t0=t0, site=site, sae_source=sae_source,
                          corpus=(corpora.to_dict("records") if len(corpora) else None),
                          is_placeholder=bool(corp.get("injection") is not None and corp["injection"].is_placeholder),
                          training=dict(lr=limits.k_train_lr, steps=limits.k_train_steps, batch=limits.k_train_batch, train_params="all",
                                        optimizer="adamw", grad_clip=1.0),
                          memorisation=dict(lr=limits.k_mem_lr, steps=limits.k_mem_steps, threshold=limits.k_threshold,
                                            max_snippets=limits.k_max_snippets, checkpoints=list(limits.k_checkpoints or [])),
                          worst_head=C.to_jsonable(worst), eps=limits.k_eps, substages=sub.report(),
                          prereg=dict(contrast_17i="ttt_memorisation_boot arm=injection-<control>, quantity=steps_to_threshold",
                                      contrast_17ii="ttt_effects measure in {sharing_jaccard, activity_*}: exceeds_all_controls",
                                      contrast_17iii="ttt_effects measure in {T_K8, B_lang_dom}: exceeds_all_controls"))
