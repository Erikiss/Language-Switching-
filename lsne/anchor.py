"""State 2143 anchor: fetch the exact training sequence, verify identity, tokenizer checks."""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from .config import STATE2143
from .pile_fetch import fetch_sequence, sha256_tokens


@dataclass
class AnchorData:
    tokens: np.ndarray            # [2049] uint16, the full training sequence
    context_ids: list[int]        # first 208 tokens
    target_id: int                # 22905
    sha256_sequence: str
    sha256_context: str
    verified: bool
    text: Optional[str] = None    # decoded context (if a tokenizer was given)
    target_text: Optional[str] = None
    continuation_text: Optional[str] = None

    def to_json(self) -> dict:
        d = asdict(self)
        d["tokens"] = self.tokens.tolist()
        return d


def verify_state2143(tokens: np.ndarray) -> dict:
    """Compare a fetched sequence against the dossier hashes and target id."""
    ctx = STATE2143["context_len"]
    sha_seq = sha256_tokens(tokens)
    sha_ctx = sha256_tokens(tokens[:ctx])
    return dict(
        sha256_sequence=sha_seq,
        sha256_context=sha_ctx,
        sequence_ok=(sha_seq == STATE2143["sha256_sequence"]),
        context_ok=(sha_ctx == STATE2143["sha256_context"]),
        context_prefix_ok=sha_ctx.startswith(STATE2143["sha256_context_prefix_dossier"]),
        target_ok=(int(tokens[ctx]) == STATE2143["target_id"]),
    )


def load_anchor(cache_dir: str | Path = "cache/state2143", tokenizer=None,
                strict: bool = True) -> AnchorData:
    """Fetch (or load cached) sequence 5,025,183 and verify it.

    With `strict=True` a hash mismatch raises; the experiment must not run on a wrong text.
    """
    toks = fetch_sequence(STATE2143["sequence_index"], cache_dir=cache_dir)
    v = verify_state2143(toks)
    ok = v["sequence_ok"] and v["context_ok"] and v["target_ok"]
    if strict and not ok:
        raise RuntimeError(f"State 2143 anchor verification failed: {v}")
    ctx = STATE2143["context_len"]
    a = AnchorData(tokens=toks, context_ids=[int(t) for t in toks[:ctx]],
                   target_id=int(toks[ctx]), sha256_sequence=v["sha256_sequence"],
                   sha256_context=v["sha256_context"], verified=ok)
    if tokenizer is not None:
        a.text = tokenizer.decode(a.context_ids)
        a.target_text = tokenizer.decode([a.target_id])
        a.continuation_text = tokenizer.decode([int(t) for t in toks[ctx:ctx + 32]])
    return a


def tokenizer_roundtrip_check(tokenizer, anchor: AnchorData) -> dict:
    """Re-tokenising the decoded context must reproduce the 208 ids exactly (BPE stability);
    the dossier's statement that `bey` has no leading space is checked too."""
    re_ids = tokenizer(anchor.text, add_special_tokens=False)["input_ids"]
    tgt = tokenizer.decode([anchor.target_id])
    return dict(
        roundtrip_ok=(list(re_ids) == list(anchor.context_ids)),
        n_tokens_roundtrip=len(re_ids),
        target_text=tgt,
        target_has_leading_space=tgt.startswith(" "),
        last_context_tokens=[tokenizer.decode([i]) for i in anchor.context_ids[-5:]],
    )


def write_anchor_json(anchor: AnchorData, path: str | Path, include_tokens: bool = False) -> None:
    """Write verification metadata (tokens are excluded by default: the text is fetched, not committed)."""
    d = anchor.to_json()
    if not include_tokens:
        d.pop("tokens")
        d.pop("context_ids")
        d.pop("text", None)
        d.pop("continuation_text", None)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=2))
