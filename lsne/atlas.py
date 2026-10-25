"""Domain compass and local proxy of the susceptibility-atlas query (DESIGN §4.7).

The client's atlas query located State 2143 near *law*; a context pushed towards more
language switching should therefore move "away from law, towards culture/politics",
if necessary counter-steering first (remove the law component) and steering second.
This module provides

* the domain keyword banks ``data/lexicon/domains.json`` (:func:`load_domains`,
  :func:`domain_token_ids` - only single-token words with a leading space are used
  for projections, the dropped ones are reported),
* the **local atlas proxy**: tokens promoted / suppressed by a state or a direction
  under the logit lens (:func:`keyword_readout`), aggregated into a **domain profile**
  (softmax mass per domain, z-standardised against random directions of the same dose,
  :func:`domain_profile`) and the same profile for SAE decoder rows through the
  unembedding (:func:`sae_feature_domain_profile`),
* **domain axes** ``dom_<domain>``: difference of means of last-token residuals on
  template sentences filled with domain words vs. contrast-domain words
  (:func:`domain_axis`), and **counter-steering** (:func:`counter_steer`),
* a tolerant loader for the client's stored atlas extraction (``data/atlas/*.txt``,
  :func:`load_atlas_extraction`) and its overlap with the domain banks
  (:func:`atlas_domain_overlap`).  Without such a file only the proxy runs.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from .config import HookSite
from .directions import Direction, _as_vec, cosine, diff_of_means, random_directions, unembed_projection, unit
from .model import ModelBundle, logit_lens, make_dose_delta, rms, run_with_cache

__all__ = [
    "DEFAULT_DOMAINS_PATH", "load_domains", "domain_names", "DomainIds", "domain_token_ids",
    "keyword_readout", "domain_profile", "sae_feature_domain_profile", "domain_axis",
    "counter_steer", "load_atlas_extraction", "atlas_domain_overlap",
]

DEFAULT_DOMAINS_PATH = Path(__file__).resolve().parent.parent / "data" / "lexicon" / "domains.json"
_EPS = 1e-30
_LANGS = ("E", "D")


# --------------------------------------------------------------------------------------
# lexicon
# --------------------------------------------------------------------------------------
def load_domains(path: Optional[Union[str, Path]] = None) -> dict:
    """Load ``domains.json`` (default: ``data/lexicon/domains.json``) and validate its
    schema: ``{"domains": {name: {"E": [...], "D": [...]}}, "templates": {"E": [...],
    "D": [...]}}`` where every template contains exactly one ``{w}`` slot."""
    p = Path(path) if path is not None else DEFAULT_DOMAINS_PATH
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("domains"), dict):
        raise ValueError(f"{p}: expected a dict with a 'domains' mapping")
    for name, langs in data["domains"].items():
        for lang in _LANGS:
            words = langs.get(lang) if isinstance(langs, dict) else None
            if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
                raise ValueError(f"{p}: domain {name!r} needs a list of strings under {lang!r}")
    templates = data.get("templates", {})
    if not isinstance(templates, dict):
        raise ValueError(f"{p}: 'templates' must be a mapping E/D -> list of strings")
    for lang, ts in templates.items():
        for t in ts:
            if not isinstance(t, str) or t.count("{w}") != 1:
                raise ValueError(f"{p}: template {t!r} ({lang}) must contain exactly one '{{w}}' slot")
    return data


def _domain_map(domains: Optional[dict]) -> dict[str, dict[str, list[str]]]:
    """Accept the full file dict (with a ``domains`` key) or the inner mapping."""
    if domains is None:
        domains = load_domains()
    if isinstance(domains, dict) and "domains" in domains and isinstance(domains["domains"], dict):
        return domains["domains"]
    return domains


def _templates(domains: Optional[dict], lang: str) -> list[str]:
    if isinstance(domains, dict) and isinstance(domains.get("templates"), dict) and domains["templates"].get(lang):
        return list(domains["templates"][lang])
    return list(load_domains()["templates"][lang])


def domain_names(domains: Optional[dict] = None) -> list[str]:
    """Domain names in file order."""
    return list(_domain_map(domains).keys())


def _word_ids(tokenizer, word: str) -> list[int]:
    return list(tokenizer(" " + word, add_special_tokens=False)["input_ids"])


class DomainIds(dict):
    """``{domain: [token ids]}`` with a ``meta`` attribute (``lang``, ``kept`` /
    ``dropped`` words per domain, ``n_kept`` / ``n_dropped``)."""
    meta: dict

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.meta = {}


def domain_token_ids(tokenizer, domains: Optional[dict] = None, lang: str = "E") -> DomainIds:
    """Token ids of the domain words that are a *single* token with a leading space
    (``" " + word``).  Multi-token words are dropped and reported in ``.meta["dropped"]``;
    ids are de-duplicated per domain (order kept)."""
    if lang not in _LANGS:
        raise ValueError(f"lang must be one of {_LANGS}, got {lang!r}")
    dmap = _domain_map(domains)
    out = DomainIds()
    kept: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    for name, langs in dmap.items():
        ids: list[int] = []
        kept[name], dropped[name] = [], []
        for w in langs.get(lang, []):
            t = _word_ids(tokenizer, w)
            if len(t) == 1:
                if t[0] not in ids:
                    ids.append(int(t[0]))
                kept[name].append(w)
            else:
                dropped[name].append(w)
        out[name] = ids
    out.meta = dict(lang=lang, kept=kept, dropped=dropped,
                    n_kept={k: len(v) for k, v in kept.items()},
                    n_dropped={k: len(v) for k, v in dropped.items()})
    return out


# --------------------------------------------------------------------------------------
# local atlas proxy: logit-lens readouts
# --------------------------------------------------------------------------------------
def _state_vec(mb: ModelBundle, h) -> torch.Tensor:
    h = torch.as_tensor(h).detach().float().cpu().reshape(-1)
    if h.numel() != mb.d_model:
        raise ValueError(f"h has {h.numel()} entries, expected d={mb.d_model}")
    return h


def _n_tok(mb: ModelBundle, V: int) -> int:
    try:
        return int(min(len(mb.tokenizer), V))
    except TypeError:          # pragma: no cover - exotic tokenizer
        return int(V)


def _decode(mb: ModelBundle, i: int) -> str:
    try:
        return mb.tokenizer.decode([int(i)])
    except Exception:          # pragma: no cover - defensive
        return f"<{int(i)}>"


def keyword_readout(mb: ModelBundle, h: torch.Tensor, direction=None, eps: float = 0.05,
                    top_k: int = 50) -> dict:
    """Tokens promoted / suppressed under the logit lens.

    With a ``direction`` (``Tensor[d]`` or ``Direction``): ``scores = lens(h + delta) -
    lens(h)`` with ``delta = make_dose_delta(h, direction, eps)`` (delta logits).  With
    ``direction=None``: ``scores = lens(h)`` (the state's own readout).  Returns
    ``dict(promoted=[(token, score)], suppressed=[...], scores=Tensor[V], promoted_ids,
    suppressed_ids, mode, eps)``; only decodable ids are ranked, ``promoted`` in
    descending and ``suppressed`` in ascending score order."""
    hv = _state_vec(mb, h)
    lens0 = logit_lens(mb, hv.to(mb.device)).cpu()                      # [V]
    if direction is None:
        scores, mode = lens0, "lens"
    else:
        vec = _as_vec(direction)
        if vec.numel() != hv.numel():
            raise ValueError(f"direction has {vec.numel()} entries, expected d={hv.numel()}")
        delta = make_dose_delta(hv, vec, float(eps))
        scores = logit_lens(mb, (hv + delta).to(mb.device)).cpu() - lens0
        mode = "delta"
    n_tok = _n_tok(mb, int(scores.numel()))
    k = int(max(0, min(top_k, n_tok)))
    s = scores[:n_tok]
    top_pos = torch.topk(s, k).indices.tolist() if k else []
    top_neg = torch.topk(-s, k).indices.tolist() if k else []
    return dict(promoted=[(_decode(mb, i), float(s[i])) for i in top_pos],
                suppressed=[(_decode(mb, i), float(s[i])) for i in top_neg],
                scores=scores, promoted_ids=[int(i) for i in top_pos],
                suppressed_ids=[int(i) for i in top_neg], mode=mode, eps=float(eps))


def _domain_mass(p: torch.Tensor, domains_ids: dict, n_tok: int) -> torch.Tensor:
    """``p [..., V] -> [..., n_dom]`` softmax mass on each domain's ids (ids >= n_tok ignored)."""
    cols = []
    for name, ids in domains_ids.items():
        idx = torch.as_tensor([int(i) for i in ids if 0 <= int(i) < n_tok], dtype=torch.long)
        cols.append(p[..., idx].sum(-1) if idx.numel() else torch.zeros(p.shape[:-1]))
    return torch.stack(cols, -1)


def domain_profile(mb: ModelBundle, h: torch.Tensor, direction=None, eps: float = 0.05,
                   domains_ids: Optional[dict] = None, n_random: int = 32, seed: int = 0,
                   lang: str = "E") -> pd.DataFrame:
    """Domain profile of a state (``direction=None``) or of a step along ``direction``.

    With a direction: ``mass = sum of lens-softmax mass on the domain's ids after the
    step (h + make_dose_delta(h, direction, eps)) minus before``; the null is the same
    quantity for ``n_random`` random unit directions with the same dose
    (:func:`lsne.directions.random_directions`), ``z = (mass - mean) / sd``.
    Without a direction: ``mass`` is the state's own lens mass on the domain; the null is
    the mass on ``n_random`` random token sets of the same size (seeded), so ``z`` says
    whether the domain is over-represented in the readout.

    ``domains_ids`` defaults to :func:`domain_token_ids` of the shipped lexicon (``lang``).
    Columns: ``domain, mass, mass_before, mass_after, mass_random_mean, mass_random_sd,
    z, n_ids``; ``df.attrs`` records ``mode, eps, n_random, seed``."""
    hv = _state_vec(mb, h)
    if domains_ids is None:
        domains_ids = domain_token_ids(mb.tokenizer, load_domains(), lang=lang)
    names = list(domains_ids.keys())
    lens0 = logit_lens(mb, hv.to(mb.device)).cpu()
    V = int(lens0.numel())
    n_tok = _n_tok(mb, V)
    p0 = torch.softmax(lens0, -1)
    before = _domain_mass(p0, domains_ids, n_tok)                           # [n_dom]
    n_ids = [len([i for i in ids if 0 <= int(i) < n_tok]) for ids in domains_ids.values()]
    n_random = int(max(n_random, 2))
    if direction is None:
        mode = "state"
        after = before.clone()
        mass = before.clone()
        rng = np.random.default_rng(int(seed))
        null = torch.zeros(n_random, len(names))
        for j, n in enumerate(n_ids):
            if n == 0:
                continue
            for r in range(n_random):
                idx = torch.as_tensor(rng.choice(n_tok, size=n, replace=False), dtype=torch.long)
                null[r, j] = p0[idx].sum()
    else:
        mode = "step"
        vec = _as_vec(direction)
        if vec.numel() != hv.numel():
            raise ValueError(f"direction has {vec.numel()} entries, expected d={hv.numel()}")
        delta = make_dose_delta(hv, vec, float(eps))
        p1 = torch.softmax(logit_lens(mb, (hv + delta).to(mb.device)).cpu(), -1)
        after = _domain_mass(p1, domains_ids, n_tok)
        mass = after - before
        R = random_directions(hv.numel(), n_random, seed)                   # [n_random, d]
        deltas = float(eps) * rms(hv) * R / rms(R, dim=-1).clamp_min(_EPS)[:, None]
        pr = torch.softmax(logit_lens(mb, (hv[None] + deltas).to(mb.device)).cpu(), -1)
        null = _domain_mass(pr, domains_ids, n_tok) - before[None]           # [n_random, n_dom]
    mean = null.mean(0)
    sd = null.std(0, unbiased=True)
    z = (mass - mean) / sd.clamp_min(1e-12)
    z = torch.where((mass - mean).abs() < 1e-15, torch.zeros_like(z), z)
    df = pd.DataFrame(dict(domain=names, mass=mass.numpy().astype(np.float64),
                           mass_before=before.numpy().astype(np.float64),
                           mass_after=after.numpy().astype(np.float64),
                           mass_random_mean=mean.numpy().astype(np.float64),
                           mass_random_sd=sd.numpy().astype(np.float64),
                           z=z.numpy().astype(np.float64), n_ids=n_ids))
    df.attrs.update(mode=mode, eps=float(eps), n_random=n_random, seed=int(seed))
    return df


def sae_feature_domain_profile(mb: ModelBundle, sae, feature_idx: Union[int, Sequence[int]],
                               domains_ids: Optional[dict] = None, lang: str = "E") -> pd.DataFrame:
    """Domain profile of SAE decoder rows through the unembedding.

    ``W_dec[feature]`` is projected like :func:`lsne.directions.unembed_projection`
    (final-LN gain times the RMS-normalised row, onto every unembedding row); per domain
    the mean projected logit over the domain's ids is compared with all vocabulary
    logits: ``z = (mean_domain - mean_all) / (sd_all / sqrt(n_ids))`` (z of a mean of
    ``n_ids`` tokens under the null that they are a random vocabulary sample) and
    ``z_token = (mean_domain - mean_all) / sd_all``.  ``feature_idx`` may be a sequence
    (one block of rows per feature).  Columns: ``feature, domain, mean_logit, mean_all,
    sd_all, z, z_token, n_ids``."""
    if domains_ids is None:
        domains_ids = domain_token_ids(mb.tokenizer, load_domains(), lang=lang)
    feats = [int(feature_idx)] if isinstance(feature_idx, (int, np.integer)) else [int(f) for f in feature_idx]
    rows = []
    for f in feats:
        v = sae.W_dec[f].detach().float().cpu()
        scores = unembed_projection(mb, v, top_k=1)["scores"]
        n_tok = _n_tok(mb, int(scores.numel()))
        s = scores[:n_tok].double()
        mean_all, sd_all = float(s.mean()), float(s.std(unbiased=True).clamp_min(1e-12))
        for name, ids in domains_ids.items():
            idx = torch.as_tensor([int(i) for i in ids if 0 <= int(i) < n_tok], dtype=torch.long)
            n = int(idx.numel())
            m = float(s[idx].mean()) if n else float("nan")
            z_tok = (m - mean_all) / sd_all if n else float("nan")
            rows.append(dict(feature=f, domain=name, mean_logit=m, mean_all=mean_all, sd_all=sd_all,
                             z=z_tok * np.sqrt(n) if n else float("nan"), z_token=z_tok, n_ids=n))
    return pd.DataFrame(rows, columns=["feature", "domain", "mean_logit", "mean_all", "sd_all", "z", "z_token", "n_ids"])


# --------------------------------------------------------------------------------------
# domain axes
# --------------------------------------------------------------------------------------
def _split_single_token(tokenizer, words: Sequence[str]) -> tuple[list[str], list[str]]:
    single, multi = [], []
    for w in words:
        (single if len(_word_ids(tokenizer, w)) == 1 else multi).append(w)
    return single, multi


def _last_token_acts(mb: ModelBundle, tokenizer, templates: Sequence[str], words: Sequence[str],
                     site: HookSite) -> tuple[torch.Tensor, int]:
    """Last-token residuals at ``site`` of every ``template.format(w=word)`` sentence
    (one forward per sentence: lengths differ).  Returns ``(acts [N, d], n_mismatch)``
    where ``n_mismatch`` counts sentences whose last token is not the word's last token."""
    acts, n_mis = [], 0
    for t in templates:
        for w in words:
            ids = list(tokenizer(t.format(w=w), add_special_tokens=False)["input_ids"])
            if not ids:
                continue
            if ids[-1] != _word_ids(tokenizer, w)[-1]:
                n_mis += 1
            co = run_with_cache(mb, torch.tensor([ids], dtype=torch.long), [site])
            acts.append(co.resid[site.key][0, -1].detach().float().cpu())
    if not acts:
        return torch.zeros(0, mb.d_model), n_mis
    return torch.stack(acts, 0), n_mis


def _as_list(x) -> list[str]:
    return [x] if isinstance(x, str) else [str(v) for v in x]


def domain_axis(mb: ModelBundle, tokenizer, domain: Union[str, Sequence[str]], site: HookSite,
                contrast: Union[str, Sequence[str]] = "everyday", lang: str = "E",
                n_templates: Optional[int] = None, domains: Optional[dict] = None,
                single_token_only: bool = False, max_words: Optional[int] = None,
                denoise_pcs: int = 1) -> Direction:
    """``dom_<domain>``: unit difference of means (domain minus contrast,
    :func:`lsne.directions.diff_of_means` with ``denoise_pcs``) of the last-token
    residuals at ``site`` on the template sentences of ``lang`` filled with the words
    of ``domain`` vs. the words of ``contrast`` (either may be a list of domains; their
    word lists are pooled; the name joins several domains with ``+``).

    ``single_token_only=True`` keeps only words that are one token (reported under
    ``meta["dropped_*"]``); by default every word is used (the last-token residual sits on
    the word's final sub-token) and the multi-token words are listed under
    ``meta["multi_token_*"]``.  ``n_templates`` / ``max_words`` limit the sentences (smoke
    tests).  meta also records ``n_domain``, ``n_contrast`` (sentences per side),
    ``n_words_*``, ``templates``, ``cos_denoised_plain``, ``diff_norm`` and ``n_mismatch``
    (sentences whose last token is not the word's last token)."""
    tokenizer = tokenizer if tokenizer is not None else mb.tokenizer
    if lang not in _LANGS:
        raise ValueError(f"lang must be one of {_LANGS}, got {lang!r}")
    data = domains if domains is not None else load_domains()
    dmap = _domain_map(data)
    doms, cons = _as_list(domain), _as_list(contrast)
    for n in doms + cons:
        if n not in dmap:
            raise KeyError(f"unknown domain {n!r}; known: {list(dmap)}")
    if set(doms) & set(cons):
        raise ValueError("domain and contrast overlap")
    templates = _templates(data, lang)
    if n_templates is not None:
        templates = templates[: int(n_templates)]
    if not templates:
        raise ValueError("no templates")

    def words_of(names):
        ws = list(dict.fromkeys(w for n in names for w in dmap[n].get(lang, [])))
        return ws[: int(max_words)] if max_words is not None else ws

    w_dom, w_con = words_of(doms), words_of(cons)
    s_dom, m_dom = _split_single_token(tokenizer, w_dom)
    s_con, m_con = _split_single_token(tokenizer, w_con)
    if single_token_only:
        use_dom, use_con, drop_dom, drop_con = s_dom, s_con, m_dom, m_con
    else:
        use_dom, use_con, drop_dom, drop_con = w_dom, w_con, [], []
    if not use_dom or not use_con:
        raise ValueError(f"no usable words (domain: {len(use_dom)}, contrast: {len(use_con)}); "
                         "try single_token_only=False or another language")
    A_dom, mis_d = _last_token_acts(mb, tokenizer, templates, use_dom, site)
    A_con, mis_c = _last_token_acts(mb, tokenizer, templates, use_con, site)
    vec = diff_of_means(A_dom, A_con, denoise_pcs=int(denoise_pcs))
    v_plain = diff_of_means(A_dom, A_con, denoise_pcs=0)
    meta = dict(domain=doms if len(doms) > 1 else doms[0], contrast=cons if len(cons) > 1 else cons[0],
                lang=lang, n_domain=int(A_dom.shape[0]), n_contrast=int(A_con.shape[0]),
                n_words_domain=len(use_dom), n_words_contrast=len(use_con), n_templates=len(templates),
                templates=list(templates), dropped_domain=drop_dom, dropped_contrast=drop_con,
                multi_token_domain=m_dom, multi_token_contrast=m_con,
                single_token_only=bool(single_token_only), denoise_pcs=int(denoise_pcs),
                cos_denoised_plain=cosine(vec, v_plain),
                diff_norm=float((A_dom.mean(0) - A_con.mean(0)).norm()),
                n_mismatch=int(mis_d + mis_c), vec_plain=v_plain)
    return Direction("dom_" + "+".join(doms), vec, site, meta)


# --------------------------------------------------------------------------------------
# counter-steering
# --------------------------------------------------------------------------------------
def counter_steer(h: torch.Tensor, remove, then, eps_remove: Optional[float], eps_then: float,
                  mean: Optional[torch.Tensor] = None, return_info: bool = False):
    """Counter-steer then steer: ``h -> h' = h - removed; out = h' + make_dose_delta(h,
    then, eps_then)``.

    1. ``eps_remove=None``: the ``remove`` component of ``(h - mean)`` is projected out
       (``removed = r (r . (h - mean))`` with ``r`` the unit ``remove`` vector; ``mean``
       defaults to 0, so ``out - dose`` is then exactly orthogonal to ``remove``).
       Otherwise a fixed dose is subtracted, ``removed = make_dose_delta(h, remove,
       eps_remove)`` (relative RMS dose, the State-2143 convention).
    2. ``make_dose_delta(h, then, eps_then)`` is added (dose relative to the *original*
       ``h``).

    ``remove`` / ``then`` are ``Direction`` objects or vectors.  Returns a tensor shaped
    like ``h`` (same device / dtype); with ``return_info=True`` also a dict with
    ``removed_norm, removed_coef, cos_before, cos_after, added_norm``."""
    h_t = torch.as_tensor(h)
    hv = h_t.detach().float().cpu().reshape(-1)
    r = unit(_as_vec(remove))
    t = _as_vec(then)
    if r.numel() != hv.numel() or t.numel() != hv.numel():
        raise ValueError("remove / then must have the same length as h")
    cos_before = cosine(hv, r)
    if eps_remove is None:
        base = hv if mean is None else hv - torch.as_tensor(mean).detach().float().cpu().reshape(-1)
        coef = float(base @ r)
        removed = coef * r
    else:
        removed = make_dose_delta(hv, r, float(eps_remove))
        coef = float(removed @ r)
    h1 = hv - removed
    added = make_dose_delta(hv, t, float(eps_then)) if float(eps_then) != 0.0 else torch.zeros_like(hv)
    out = (h1 + added).reshape(h_t.shape).to(device=h_t.device, dtype=h_t.dtype if h_t.is_floating_point() else torch.float32)
    if not return_info:
        return out
    info = dict(removed_norm=float(removed.norm()), removed_coef=coef, cos_before=cos_before,
                cos_after=cosine(h1, r), added_norm=float(added.norm()),
                rms_h=float(rms(hv)), eps_remove=eps_remove, eps_then=float(eps_then))
    return out, info


# --------------------------------------------------------------------------------------
# atlas extraction import
# --------------------------------------------------------------------------------------
_BULLET = re.compile(r"^\s*[-*•]\s+(.*)$")
_ID_LINE = re.compile(r"^\s*(?:cluster\s*)?#?([A-Za-z0-9_.\-]+)\s*[:：]\s*(.*)$", re.IGNORECASE)
_SPLIT = re.compile(r"[,;|\t]+")


def _split_keywords(s: str) -> list[str]:
    out = []
    for k in _SPLIT.split(s):
        k = k.strip().strip("\"'`[](){}").strip()
        if k:
            out.append(k)
    return out


def _dedupe(kws: Sequence[str]) -> list[str]:
    seen, out = set(), []
    for k in kws:
        key = k.lower()
        if key not in seen:
            seen.add(key)
            out.append(k)
    return out


def _parse_atlas_text(text: str, source: Optional[str] = None) -> list[dict]:
    clusters: list[dict] = []
    blocks = [b for b in re.split(r"\n\s*\n", text.strip()) if b.strip()]
    for bi, block in enumerate(blocks):
        current: Optional[dict] = None
        for line in block.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            mb_ = _BULLET.match(line)
            if mb_:
                if current is None:
                    current = dict(cluster_id=f"block{bi}", keywords=[], raw=block)
                    clusters.append(current)
                current["keywords"].extend(_split_keywords(mb_.group(1)))
                continue
            mi = _ID_LINE.match(line)
            if mi:
                current = dict(cluster_id=mi.group(1), keywords=_split_keywords(mi.group(2)), raw=block)
                clusters.append(current)
                continue
            if current is None:
                current = dict(cluster_id=f"block{bi}", keywords=[], raw=block)
                clusters.append(current)
            current["keywords"].extend(_split_keywords(line))
    for c in clusters:
        c["keywords"] = _dedupe(c["keywords"])
        if source is not None:
            c["source"] = source
    return clusters


def load_atlas_extraction(path: Union[str, Path]) -> list[dict]:
    """Tolerant parser of a stored atlas extraction (``data/atlas/*.txt``).

    Blocks are separated by blank lines; a line ``<id>: kw1, kw2, ...`` (optionally
    ``Cluster <id>:``) starts a cluster, ``- kw`` lines (or plain comma-separated lines)
    add keywords to the current cluster (a block without an id line becomes cluster
    ``block<n>``); ``#`` lines are comments.  Returns ``[dict(cluster_id, keywords,
    raw)]``; ``[]`` if ``path`` does not exist.  A directory is parsed file by file
    (``*.txt`` / ``*.md``, sorted; each cluster gets a ``source`` key)."""
    p = Path(path)
    if not p.exists():
        return []
    if p.is_dir():
        out: list[dict] = []
        for f in sorted(list(p.glob("*.txt")) + list(p.glob("*.md"))):
            out.extend(_parse_atlas_text(f.read_text(encoding="utf-8", errors="replace"), source=f.name))
        return out
    return _parse_atlas_text(p.read_text(encoding="utf-8", errors="replace"))


def atlas_domain_overlap(clusters: Sequence[dict], domains: Optional[dict] = None,
                         langs: Sequence[str] = _LANGS) -> pd.DataFrame:
    """Overlap between cluster keywords and the domain word lists (case-insensitive,
    both languages pooled by default).  One row per ``(cluster, domain)``:
    ``cluster_id, domain, n_overlap, jaccard, n_keywords, n_domain_words, overlap``
    (``jaccard = |K & W| / |K | W|``)."""
    dmap = _domain_map(domains)
    words = {name: {w.strip().lower() for lg in langs for w in d.get(lg, [])} for name, d in dmap.items()}
    rows = []
    for ci, c in enumerate(clusters):
        kws = c.get("keywords", []) if isinstance(c, dict) else list(c)
        cid = c.get("cluster_id", f"cluster{ci}") if isinstance(c, dict) else f"cluster{ci}"
        K = {str(k).strip().lower() for k in kws if str(k).strip()}
        for name, W in words.items():
            inter = K & W
            union = K | W
            rows.append(dict(cluster_id=cid, domain=name, n_overlap=len(inter),
                             jaccard=(len(inter) / len(union)) if union else 0.0,
                             n_keywords=len(K), n_domain_words=len(W), overlap=sorted(inter)))
    return pd.DataFrame(rows, columns=["cluster_id", "domain", "n_overlap", "jaccard", "n_keywords",
                                       "n_domain_words", "overlap"])
