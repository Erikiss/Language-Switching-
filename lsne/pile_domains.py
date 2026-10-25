"""Empirical difficulty ranking of Pile sub-areas (DESIGN §4.8b).

The premise "protein misfolding is the worst-predicted part of the Pile" is *measured*,
not assumed: training sequences are drawn uniformly from the Pythia Pile with the
range-request sampler (:func:`lsne.pile_fetch.fetch_sequence`), each one is assigned to
a subfield by a keyword profile on its decoded text (:data:`PILE_MARKERS`,
:func:`classify_sequence`), and its token NLL under the model is measured
(:func:`sequence_nll`).  :func:`sample_sequence_difficulty` ties the three together into
a per-sequence table, :func:`difficulty_ranking` aggregates it into a ranking (hardest
first, bootstrap CI over sequences) and :func:`hardest_subfield` /
:func:`misfolding_rank_check` read the answer off the ranking.

Marker lists: every subfield has a list of *lowercase substrings*; an entry starting with
``"re:"`` is a regular expression (applied to the lowercased text).  The score of a
subfield is the mean capped hit count per marker (``sum_m min(count_m, cap) / n_markers``,
so one promiscuous marker cannot dominate and subfields with many markers are not
favoured); a subfield is eligible only if its raw hit count reaches ``min_hits``.  The
subfield hierarchy ``pubmed > protein > protein_misfolding`` is resolved towards the most
specific eligible child (a PubMed abstract about amyloid is ``protein_misfolding``).
"""
from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from .model import ModelBundle, forward_logits
from .pile_fetch import fetch_sequence

__all__ = [
    "PILE_MARKERS", "SUBFIELD_CHILDREN", "N_PILE_SEQUENCES", "classify_text", "classify_sequence",
    "lexicon_markers", "sample_pile_sequences", "sequence_nll", "sequence_difficulty_table",
    "sample_sequence_difficulty", "difficulty_ranking", "hardest_subfield", "misfolding_rank_check",
]

N_PILE_SEQUENCES = 146_432_000          # 143_000 batches x 1024 sequences (lsne.pile_fetch)

# --------------------------------------------------------------------------------------
# marker lists
# --------------------------------------------------------------------------------------
PILE_MARKERS: dict[str, list[str]] = {
    "pubmed": [
        "abstract", "patients", "et al", "doi:", "doi.org", "clinical", "in vitro", "in vivo",
        "mg/kg", "expression", "cohort", "placebo", "randomized", "significantly", "pubmed",
        "conclusions", "methods:", "results:", "treatment", r"re:\bp\s*[<=]\s*0?\.\d+",
    ],
    "protein": [
        "protein", "residue", "amino acid", "peptide", "binding", "kinase", "folding", "enzyme",
        "ligand", "receptor", "substrate", "mutant", "crystal structure", "phosphorylation",
        "alpha-helix", "α-helix", "helix", "antibody", "n-terminal", "c-terminal", "cleavage",
        "monomer", "dimer",
    ],
    "protein_misfolding": [
        "prion", "amyloid", "fibril", "misfold", "aggregat", "plaque", "beta-sheet", "β-sheet",
        "beta sheet", "β sheet", r"re:(?<!\\)\btau\b", "synuclein", "chaperone", "oligomer",
        "alzheimer", "parkinson", "huntingt", "scrapie", r"re:\bprp\b", "creutzfeldt",
        "neurodegenerat", "tangle", "proteostasis", "thioflavin", "inclusion bod", "lewy",
    ],
    "code": [
        "def ", "import ", "return ", "#include", ") {", "};", "} else", "==", "!=", "->", "self.",
        "function(", "function ", "const ", "var ", "null", "printf", "std::", "</div>", "static ",
        "void ", ";\n", "});", "===",
    ],
    "math": [
        "\\begin", "\\end", "\\frac", "\\alpha", "\\mathbb", "\\sum", "\\int", "\\left", "\\right",
        "\\cdot", "theorem", "lemma", "proof", "corollary", "proposition", "equation", "integer",
        "polynomial", "matrix", "vector space", r"re:\$[^$\n]{1,80}\$",
    ],
    "law": [
        "court", "plaintiff", "defendant", "statute", "pursuant", r"re:\bv\.\s", "appeal",
        "jurisdiction", "attorney", "judgment", "hereby", "shall be", "tribunal", "petitioner",
        "affirmed", "u.s.c.", "f.2d", "f.3d", "counsel", "legislat", "supreme court",
        "district court", "motion to", "the parties",
    ],
    "news": [
        "said on", "reuters", "associated press", "reported", "spokesman", "spokeswoman",
        "spokesperson", "according to", "told reporters", "officials", "president", "minister",
        "police said", "on monday", "on tuesday", "on wednesday", "on thursday", "on friday",
        "on saturday", "on sunday", "said in a statement", "percent",
    ],
    "web": [
        "click here", "http", "www.", ".com", "subscribe", "email", "posted by", "comments",
        "read more", "cookies", "privacy policy", "login", "sign up", "share this",
        "all rights reserved", "copyright", "blog", "newsletter", "contact us", "follow us",
    ],
    "books": [
        "\" she said", "\" he said", "she said", "he said", "she whispered", "he whispered",
        "he looked", "she looked", "chapter", "his eyes", "her eyes", "she felt", "he felt",
        "the door", "nodded", "smiled", "sighed", "glanced", "shrugged", "muttered",
    ],
    "wikipedia": [
        "born in", "references", "retrieved", "census", "located in", "population of",
        "municipality", "external links", "see also", "category:", "wikipedia", "is a village",
        "is a town", "species of", "genus", "was founded", "district of", "is a genus",
        "is a river", "is a commune", "footballer", "album", "disambiguation",
    ],
    "german": [
        r"re:\bund\b", r"re:\bder\b", r"re:\bdie\b", r"re:\bdas\b", r"re:\bnicht\b", r"re:\bist\b",
        r"re:\bmit\b", r"re:\beine\b", r"re:\bsich\b", r"re:\bwird\b", r"re:\bauch\b",
        r"re:\bwerden\b", r"re:\bfür\b", r"re:\büber\b", r"re:\bdass\b", r"re:\bich\b", "ß",
    ],
}

# hierarchy: a parent's hit is refined towards the most specific eligible child
SUBFIELD_CHILDREN: dict[str, tuple[str, ...]] = {
    "pubmed": ("protein", "protein_misfolding"),
    "protein": ("protein_misfolding",),
}

OTHER = "other"


@functools.lru_cache(maxsize=4096)
def _compile(pattern: str) -> re.Pattern:
    return re.compile(pattern)


def _count(text: str, marker: str) -> int:
    if marker.startswith("re:"):
        return len(_compile(marker[3:]).findall(text))
    return text.count(marker)


def _markers(markers: Optional[Mapping[str, Sequence[str]]]) -> dict[str, list[str]]:
    m = PILE_MARKERS if markers is None else markers
    out = {str(k): [str(x) for x in v] for k, v in m.items()}
    for k, v in out.items():
        if not v:
            raise ValueError(f"subfield {k!r} has no markers")
    return out


def lexicon_markers(domains: Optional[dict] = None, langs: Sequence[str] = ("E", "D"),
                    min_len: int = 4) -> dict[str, list[str]]:
    """Marker lists built from the lexicon domain word banks (``data/lexicon/domains.json``
    or the dict returned by :func:`lsne.atlas.load_domains`): every word of at least
    ``min_len`` characters becomes a word-boundary regex (lowercase) that also accepts the
    common plural / inflection endings ``-e, -en, -n, -s, -es`` (``Prionen``, ``plaques``).
    Use the result on its own (``classify_text(text, markers=lexicon_markers())``) to
    classify into the lexicon domains; because scores are normalised per marker, merging
    the long lexicon lists with the short :data:`PILE_MARKERS` lists favours the latter."""
    from .atlas import _domain_map
    dmap = _domain_map(domains)
    out: dict[str, list[str]] = {}
    for name, langs_words in dmap.items():
        words: list[str] = []
        for lg in langs:
            for w in langs_words.get(lg, []):
                w = str(w).strip().lower()
                if len(w) >= int(min_len) and w not in words:
                    words.append(w)
        if words:
            out[str(name)] = ["re:" + r"\b" + re.escape(w) + r"(?:e|en|n|s|es)?\b" for w in words]
    return out


# --------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------
def classify_text(text: str, markers: Optional[Mapping[str, Sequence[str]]] = None,
                  min_hits: int = 3, cap: int = 25, children: Optional[Mapping[str, Sequence[str]]] = None) -> dict:
    """Keyword-profile classification of a text.

    Returns ``dict(subfield, scores, hits, n_words, n_chars, min_hits)`` where
    ``hits[s]`` is the raw number of marker hits of subfield ``s`` on the lowercased text,
    ``scores[s] = sum_m min(count_m, cap) / n_markers_s`` and ``subfield`` is the argmax of
    ``scores`` among the subfields with ``hits >= min_hits`` (``"other"`` if none), refined
    along ``children`` (default :data:`SUBFIELD_CHILDREN`) towards the best eligible child
    (ties: the first subfield in marker order)."""
    mk = _markers(markers)
    kids = SUBFIELD_CHILDREN if children is None else children
    t = str(text).lower()
    hits: dict[str, int] = {}
    scores: dict[str, float] = {}
    for name, ms in mk.items():
        counts = [_count(t, m) for m in ms]
        hits[name] = int(sum(counts))
        scores[name] = float(sum(min(c, int(cap)) for c in counts) / len(ms))
    eligible = [n for n in mk if hits[n] >= int(min_hits)]
    best = OTHER
    if eligible:
        best = max(eligible, key=lambda n: (scores[n], -list(mk).index(n)))
        seen = {best}
        while best in kids:
            cand = [c for c in kids[best] if c in mk and hits[c] >= int(min_hits) and c not in seen]
            if not cand:
                break
            best = max(cand, key=lambda n: (scores[n], -list(mk).index(n)))
            seen.add(best)
    return dict(subfield=best, scores=scores, hits=hits, n_words=len(t.split()), n_chars=len(t),
                min_hits=int(min_hits))


def _ids_list(token_ids) -> list[int]:
    if isinstance(token_ids, torch.Tensor):
        return [int(i) for i in token_ids.detach().reshape(-1).tolist()]
    return [int(i) for i in np.asarray(token_ids).reshape(-1)]


def classify_sequence(tokenizer, token_ids, markers: Optional[Mapping[str, Sequence[str]]] = None,
                      min_hits: int = 3, cap: int = 25, return_scores: bool = False):
    """Best-matching subfield of a token sequence (``"other"`` when no subfield reaches
    ``min_hits``): the decoded text is scored with :func:`classify_text`.  With
    ``return_scores=True`` the full result dict is returned instead of the name."""
    ids = _ids_list(token_ids)
    try:
        text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    except TypeError:          # pragma: no cover - tokenizers without the kwarg
        text = tokenizer.decode(ids)
    res = classify_text(text, markers=markers, min_hits=min_hits, cap=cap)
    return res if return_scores else res["subfield"]


# --------------------------------------------------------------------------------------
# sampling and NLL
# --------------------------------------------------------------------------------------
def sample_indices(n: int, seed: int, n_total: int = N_PILE_SEQUENCES) -> np.ndarray:
    """``n`` distinct uniformly random sequence indices (``numpy.random.default_rng(seed)``)."""
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(int(n_total), size=int(n), replace=False)).astype(np.int64)


def sample_pile_sequences(n: int, seed: int, cache_dir: Optional[Union[str, Path]] = None,
                          indices: Optional[Sequence[int]] = None,
                          n_total: int = N_PILE_SEQUENCES) -> list[np.ndarray]:
    """Fetch ``n`` uniformly sampled training sequences (``np.uint16 [2049]`` each) by range
    request; every sequence is cached as ``<cache_dir>/seq_<index>.npy`` when ``cache_dir``
    is given.  ``indices`` overrides the sampling."""
    idx = sample_indices(n, seed, n_total) if indices is None else np.asarray(list(indices), dtype=np.int64)
    return [fetch_sequence(int(i), cache_dir=cache_dir) for i in idx]


def sequence_nll(mb: ModelBundle, ids, max_len: int = 512, tail_frac: float = 0.25) -> dict:
    """Teacher-forced token NLL of ``ids`` (truncated to ``max_len`` tokens) under ``mb``.

    Returns ``dict(mean_nll, mean_nll_tail, n_tokens, nll_per_token)`` where ``n_tokens``
    is the number of *predicted* tokens (``T - 1``), ``nll_per_token [T-1]`` the NLL of
    token ``t+1`` given tokens ``<= t`` and ``mean_nll_tail`` the mean over the last
    ``ceil(tail_frac * (T-1))`` predicted tokens (the part with the longest context)."""
    toks = _ids_list(ids)[: int(max_len)]
    if len(toks) < 2:
        raise ValueError("need at least two tokens to compute a next-token NLL")
    x = torch.tensor([toks], dtype=torch.long)
    logits = forward_logits(mb, x)[0]                                   # [T, V]
    logp = torch.log_softmax(logits[:-1], dim=-1)
    tgt = x[0, 1:].to(logp.device)
    nll = -logp.gather(-1, tgt[:, None])[:, 0].cpu().numpy().astype(np.float64)
    n = int(nll.shape[0])
    n_tail = max(1, int(np.ceil(float(tail_frac) * n)))
    return dict(mean_nll=float(nll.mean()), mean_nll_tail=float(nll[-n_tail:].mean()),
                n_tokens=n, nll_per_token=nll)


_COLS = ["seq_index", "subfield", "n_tokens", "mean_nll", "mean_nll_tail", "score", "n_hits"]


def sequence_difficulty_table(mb: ModelBundle, sequences: Sequence, indices: Optional[Sequence] = None,
                              markers: Optional[Mapping[str, Sequence[str]]] = None, max_len: int = 512,
                              tokenizer=None, min_hits: int = 3) -> pd.DataFrame:
    """Classify and score a list of token sequences (no network).  One row per sequence:
    ``seq_index, subfield, n_tokens, mean_nll, mean_nll_tail, score, n_hits`` (``score`` /
    ``n_hits`` of the winning subfield; 0 for ``"other"``)."""
    tok = tokenizer if tokenizer is not None else mb.tokenizer
    idx = list(range(len(sequences))) if indices is None else list(indices)
    if len(idx) != len(sequences):
        raise ValueError("indices and sequences differ in length")
    rows = []
    for i, seq in zip(idx, sequences):
        cls = classify_sequence(tok, seq, markers=markers, min_hits=min_hits, return_scores=True)
        sf = cls["subfield"]
        nll = sequence_nll(mb, seq, max_len=max_len)
        rows.append(dict(seq_index=int(i), subfield=sf, n_tokens=nll["n_tokens"], mean_nll=nll["mean_nll"],
                         mean_nll_tail=nll["mean_nll_tail"],
                         score=float(cls["scores"].get(sf, 0.0)), n_hits=int(cls["hits"].get(sf, 0))))
    return pd.DataFrame(rows, columns=_COLS)


def _model_tag(mb: ModelBundle) -> str:
    tag = mb.spec.tag() if getattr(mb, "spec", None) is not None else "model"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)


def sample_sequence_difficulty(mb: ModelBundle, n_sequences: int = 2000, seed: int = 0,
                               markers: Optional[Mapping[str, Sequence[str]]] = None,
                               cache_dir: Optional[Union[str, Path]] = "cache/pile_nll", max_len: int = 512,
                               indices: Optional[Sequence[int]] = None, n_total: int = N_PILE_SEQUENCES,
                               min_hits: int = 3, progress: bool = False) -> pd.DataFrame:
    """Per-sequence difficulty table of ``n_sequences`` uniformly sampled Pile training
    sequences: ``seq_index, subfield, n_tokens, mean_nll, mean_nll_tail, score, n_hits``.

    Sequences are fetched by range request (cached as npy under ``<cache_dir>/seqs``),
    truncated to ``max_len`` tokens, classified with ``markers`` (default
    :data:`PILE_MARKERS`) and scored with one forward pass each.  The table is cached as
    ``<cache_dir>/pile_nll_n<n>_s<seed>_<model tag>_L<max_len>.csv`` (``indices`` given:
    no csv cache).  ``df.attrs`` records ``n_sequences, seed, model, max_len``."""
    cdir = Path(cache_dir) if cache_dir else None
    tag = _model_tag(mb)
    csv = None
    if cdir is not None and indices is None:
        csv = cdir / f"pile_nll_n{int(n_sequences)}_s{int(seed)}_{tag}_L{int(max_len)}.csv"
        if csv.exists():
            df = pd.read_csv(csv)
            df.attrs.update(n_sequences=int(n_sequences), seed=int(seed), model=tag, max_len=int(max_len), cached=True)
            return df
    idx = sample_indices(n_sequences, seed, n_total) if indices is None else np.asarray(list(indices), dtype=np.int64)
    seq_dir = cdir / "seqs" if cdir is not None else None
    rows = []
    it = enumerate(idx)
    if progress:
        try:
            from tqdm import tqdm
            it = enumerate(tqdm(idx, desc="pile nll"))
        except Exception:       # pragma: no cover
            pass
    for _, i in it:
        seq = fetch_sequence(int(i), cache_dir=seq_dir)
        rows.append(sequence_difficulty_table(mb, [seq], [int(i)], markers=markers, max_len=max_len,
                                              min_hits=min_hits))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=_COLS)
    df.attrs.update(n_sequences=int(len(idx)), seed=int(seed), model=tag, max_len=int(max_len), cached=False)
    if csv is not None:
        cdir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
    return df


# --------------------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------------------
def difficulty_ranking(df: pd.DataFrame, n_boot: int = 1000, seed: int = 0, value_col: str = "mean_nll",
                       min_seqs: int = 1) -> pd.DataFrame:
    """Aggregate a per-sequence table into a ranking of subfields, hardest first.

    Columns: ``subfield, n_seqs, mean_nll, sd, se, ci_low, ci_high, mean_nll_tail, rank``
    (``rank`` 1 = highest mean ``value_col``; ``ci`` = percentile bootstrap over sequences
    with ``n_boot`` resamples, equal to the mean when ``n_boot == 0`` or ``n_seqs == 1``).
    Subfields with fewer than ``min_seqs`` sequences are dropped."""
    if value_col not in df.columns or "subfield" not in df.columns:
        raise ValueError(f"table needs the columns 'subfield' and {value_col!r}")
    rng = np.random.default_rng(int(seed))
    rows = []
    for name, g in df.groupby("subfield", sort=True):
        v = np.asarray(g[value_col], dtype=np.float64)
        v = v[np.isfinite(v)]
        n = int(v.shape[0])
        if n < int(min_seqs) or n == 0:
            continue
        m = float(v.mean())
        sd = float(v.std(ddof=1)) if n > 1 else 0.0
        if int(n_boot) > 0 and n > 1:
            boots = v[rng.integers(0, n, size=(int(n_boot), n))].mean(1)
            lo, hi = (float(x) for x in np.percentile(boots, [2.5, 97.5]))
        else:
            lo = hi = m
        tail = float(np.nanmean(g["mean_nll_tail"])) if "mean_nll_tail" in g.columns else float("nan")
        rows.append(dict(subfield=str(name), n_seqs=n, mean_nll=m, sd=sd, se=sd / np.sqrt(n) if n > 1 else 0.0,
                         ci_low=lo, ci_high=hi, mean_nll_tail=tail))
    cols = ["subfield", "n_seqs", "mean_nll", "sd", "se", "ci_low", "ci_high", "mean_nll_tail", "rank"]
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows).sort_values(["mean_nll", "subfield"], ascending=[False, True], kind="mergesort")
    out = out.reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[cols]


def hardest_subfield(ranking: pd.DataFrame, exclude: Sequence[str] = (OTHER,)) -> str:
    """Name of the hardest subfield (rank 1 after dropping ``exclude``)."""
    r = ranking[~ranking["subfield"].isin(list(exclude))]
    if r.empty:
        raise ValueError("ranking is empty after exclusion")
    return str(r.sort_values("rank").iloc[0]["subfield"])


def misfolding_rank_check(ranking: pd.DataFrame, subfield: str = "protein_misfolding",
                          exclude: Sequence[str] = (OTHER,)) -> dict:
    """Where does ``subfield`` sit in the ranking?  ``dict(subfield, rank, n_subfields,
    in_hardest_third, hardest, present)`` (preregistration item 16: if the subfield is not
    in the hardest third, the counter-pole is replaced by ``hardest``).  ``rank`` is
    recomputed after dropping ``exclude``; ``None`` when the subfield is absent."""
    r = ranking[~ranking["subfield"].isin(list(exclude))].sort_values("rank").reset_index(drop=True)
    n = int(len(r))
    present = bool((r["subfield"] == subfield).any())
    rank = int(r.index[r["subfield"] == subfield][0]) + 1 if present else None
    third = int(np.ceil(n / 3.0)) if n else 0
    return dict(subfield=subfield, rank=rank, n_subfields=n, present=present,
                in_hardest_third=bool(present and rank <= third),
                hardest=str(r.iloc[0]["subfield"]) if n else None)
