"""Stage A - stimuli: families, masks, tokenisation, consistency checks, annotation.

Tables
------
families        one row per family file (ids, title, target, token id from the K=0
                variant - or the K=0 rendering if no variant was kept -, n_units, kind)
variants        one row per variant (family, formulation, K, C, mask, n_tokens, n_words,
                F100, last_switch_pos, tail_start, n_tokens_D, ..., text)
checks          every built variant with its problems (``excluded`` flag) + schema failures
exclusions      the excluded rows of ``checks`` only
token_classes   token-class counts per variant x class x language
tokens          per-token annotation (position, token, lang, unit, seg, klass,
                dist_to_last_switch, is_tail) of every kept variant

The original-2143 variants are included when the anchor is available.
"""
from __future__ import annotations

import time

import pandas as pd

from .. import stimuli
from ..config import RunConfig
from ..model import ModelBundle
from . import common as C


def families_table(fams, original, variants=(), tokenizer=None) -> pd.DataFrame:
    """One row per family; ``target_id`` of a standard family (``Family.target_id`` is
    None) is taken from its K=0 variant in `variants` (the id is identical across the
    variants of a family/formulation, `stimuli.check_variant`; any variant of the group
    is used when no K=0 variant was kept).  For a family without a variant (e.g. cut by
    ``cfg.limit_variants``) the id is computed from the K=0 rendering when a `tokenizer`
    is given (`stimuli.target_token_id`; NaN if the target is not a single token), else
    it is NaN."""
    target_ids: dict[tuple[str, str], int] = {}
    for v in variants:
        key = (str(v.family_id), str(v.formulation))
        if int(v.K) == 0 or key not in target_ids:
            target_ids[key] = int(v.target_id)
    rows = []
    for f in list(fams) + ([original.family] if original is not None else []):
        tid = f.target_id if f.target_id is not None else target_ids.get((f.family_id, f.formulation))
        if tid is None and tokenizer is not None and f.kind == "standard":
            try:
                text = stimuli.render_text(f, stimuli.segment_masks(f.n_units, 0, "C1"))
                tid = stimuli.target_token_id(tokenizer, text, f.target)
            except ValueError:
                tid = None
        rows.append(dict(tag=f.tag, family=f.family_id, formulation=f.formulation, kind=f.kind,
                         title=f.title, topic=f.topic, n_units=f.n_units, n_segments=f.n_segments,
                         target=f.target, target_id=tid, tail_E=f.tail_E,
                         dev=f.family_id.startswith("DEV"), source=f.source or ""))
    return pd.DataFrame(rows, columns=["tag", "family", "formulation", "kind", "title", "topic", "n_units",
                                       "n_segments", "target", "target_id", "tail_E", "dev", "source"])


def variants_table(variants) -> pd.DataFrame:
    if not variants:
        return pd.DataFrame(columns=list(C.ID_COLS))
    df = stimuli.variant_summary(variants)
    df.insert(0, "tag", [v.tag for v in variants])
    df.insert(1, "kind", [C.variant_kind(v) for v in variants])
    df = df.rename(columns={"family_id": "family"})
    df["n_boundary"] = [len(v.annot.get("boundary_positions", [])) for v in variants]
    df["target"] = [v.target for v in variants]
    df["text"] = [v.text for v in variants]
    return df


def token_class_table(variants) -> pd.DataFrame:
    rows = []
    for v in variants:
        counts: dict[tuple[str, str, bool], int] = {}
        ts = int(v.tail_start)
        for t, (k, lang) in enumerate(zip(v.annot["klass"], v.annot["lang"])):
            key = (str(k), str(lang), t >= ts)
            counts[key] = counts.get(key, 0) + 1
        for (k, lang, tail), n in sorted(counts.items()):
            rows.append(dict(C.variant_row(v), klass=k, lang=lang, is_tail=tail, n=n))
    return pd.DataFrame(rows, columns=list(C.ID_COLS) + ["klass", "lang", "is_tail", "n"])


def tokens_table(variants) -> pd.DataFrame:
    rows = []
    for v in variants:
        a = v.annot
        ts = int(v.tail_start)
        for t in range(v.n_tokens):
            rows.append(dict(tag=v.tag, family=v.family_id, formulation=v.formulation, K=int(v.K), C=str(v.C),
                             position=t, token_id=int(v.ids[t]), token=a["tokens"][t], lang=a["lang"][t],
                             unit=int(a["unit"][t]), seg=int(a["seg"][t]), klass=a["klass"][t],
                             dist_to_last_switch=int(a["dist_to_last_switch"][t]), is_tail=t >= ts))
    return pd.DataFrame(rows, columns=["tag", "family", "formulation", "K", "C", "position", "token_id", "token",
                                       "lang", "unit", "seg", "klass", "dist_to_last_switch", "is_tail"])


def run(cfg: RunConfig, mb: ModelBundle, run_dir, **kw) -> dict[str, pd.DataFrame]:
    """Build and check the stimulus bank; see the module docstring for the tables."""
    t0 = time.perf_counter()
    limits = C.get_limits(cfg, kw)
    ctx = kw.get("ctx")
    variants, checks = C.get_variants(cfg, mb, ctx)
    original = C.get_original(cfg, mb, ctx, limits)
    allv = C.all_variants(variants, original)
    fams = C.select_families(cfg)

    families = families_table(fams, original, variants, tokenizer=mb.tokenizer)
    built = {(str(v.family_id), str(v.formulation)) for v in variants}
    missing = [t for t, fid, form, tid in zip(families["tag"], families["family"], families["formulation"], families["target_id"])
               if (fid, form) in built and pd.isna(tid)]
    assert not missing, f"families table without target_id although variants were built: {missing}"
    tables = {
        "families": families,
        "variants": variants_table(allv),
        "checks": checks,
        "exclusions": checks[checks["excluded"]].reset_index(drop=True) if len(checks) else checks,
        "token_classes": token_class_table(allv),
        "tokens": tokens_table(allv),
    }
    return C.finish_stage(run_dir, "A", cfg, mb, tables, limits=limits, t0=t0,
                          n_families=len(fams), n_variants=len(variants),
                          n_excluded=int(checks["excluded"].sum()) if len(checks) else 0,
                          original=(original.info() if original is not None else None))
