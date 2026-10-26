"""Paired English rendering of a :class:`~lsne.chain.itr.TxRecord` for the target model (Pythia).

The text describes the same transaction as the ITR token stream: the root call, then up
to ``max_nodes - 1`` internal calls in depth-first order with the nesting expressed in
words ("Inside that call, …", "Back at level two, …"), values in ETH with two
significant figures, failures, and a closing summary of the value flow, events and
storage writes.  Fully deterministic (no randomness, no dict-order dependence).
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from .itr import CallNode, TxRecord, error_kind, iter_nodes

ADDRESS_STYLES = ("short", "full", "named")
_NAMES = {
    "0xbb9bc244d798123fde783fcc1c72d3bb8c189413": "the DAO",
    "0x807640a13483f8ac783c557fcdf27be11ea4ac7a": "the DAO extraBalance account",
    "0x4a574510c7014e4ae985403536074abe582adfc8": "the DAO creator contract",
    "0xbf4ed7b27f1d666546e30d74d50d173d20bca754": "the WithdrawDAO contract",
}
_ERROR_WORDS = {"out_of_gas": "out of gas", "bad_jump": "bad jump destination",
                "bad_instruction": "bad instruction", "stack_underflow": "stack underflow",
                "stack_overflow": "stack overflow", "out_of_funds": "insufficient funds",
                "reverted": "reverted", "call_depth": "call depth limit", "invalid_opcode": "invalid opcode",
                "other": "error"}
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
         "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_word(n: int) -> str:
    """0..99 in words, larger numbers as digits."""
    n = int(n)
    if 0 <= n < 20:
        return _ONES[n]
    if 20 <= n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + ("" if o == 0 else "-" + _ONES[o])
    return str(n)


def format_eth(wei: int) -> str:
    """Wei -> ETH string with two significant figures (``2597.76 ETH -> "2600"``,
    ``0.01234 -> "0.012"``); below 1e-6 ETH scientific (``"1.0e-18"``)."""
    wei = int(wei)
    if wei == 0:
        return "0"
    d = Decimal(wei) / Decimal(10 ** 18)
    exp = d.adjusted()
    r = d.quantize(Decimal(1).scaleb(exp - 1), rounding=ROUND_HALF_UP)
    if r.adjusted() != exp:                       # 9.96 -> 10: re-quantize at the new magnitude
        exp = r.adjusted()
        r = r.quantize(Decimal(1).scaleb(exp - 1), rounding=ROUND_HALF_UP)
    if exp < -6:
        digits = r.scaleb(-exp)                   # mantissa in [1, 10)
        return f"{digits:.1f}e{exp}"
    return format(r, "f")


def _addr(a: Optional[str], style: str) -> str:
    if not a:
        return "an unknown address"
    if style == "full":
        return a
    if style == "named" and a in _NAMES:
        return _NAMES[a]
    return f"{a[:6]}…{a[-4:]}" if len(a) > 12 else a


def _kind_addr(kind: str, a: Optional[str], style: str) -> str:
    """``"contract 0x53d2…8a3d"``; a known name alone in the ``"named"`` style."""
    if style == "named" and a in _NAMES:
        return _NAMES[a]
    return f"{kind} {_addr(a, style)}"


def _is_contract(n: CallNode) -> bool:
    return bool(n.selector or n.children or n.logs or n.state_writes or n.n_input_bytes > 0
                or n.call_type == "create")


def _fn(n: CallNode) -> str:
    return f" (function {n.selector[2:]})" if n.selector else ""


def _fail(n: CallNode) -> str:
    ek = error_kind(n.error)
    return f" This call fails ({_ERROR_WORDS.get(ek, 'error')})." if ek else ""


def _root_sentence(rec: TxRecord, style: str) -> str:
    n = rec.root
    sender = f"Account {_addr(n.frm or rec.frm, style)}"
    v = format_eth(n.value)
    if n.call_type == "create":
        what = f"contract {_addr(n.created, style)}" if n.created else "a contract"
        s = f"{sender} creates {what}" + (f" and funds it with {v} ETH" if n.value else "") + "."
    else:
        kind = "contract" if _is_contract(n) else "account"
        tgt = _kind_addr(kind, n.to or rec.to, style)
        if n.value and not n.selector:
            s = f"{sender} sends {v} ETH to {tgt}."
        elif n.value:
            s = f"{sender} sends {v} ETH to {tgt}, calling function {n.selector[2:]}."
        elif n.selector:
            s = f"{sender} calls {tgt}{_fn(n)} without sending ETH."
        else:
            s = f"{sender} contacts {tgt} without sending ETH."
    return s


def _actor(n: CallNode, rec: TxRecord, style: str) -> str:
    root_target = rec.root.target
    if n.frm and n.frm == root_target:
        return "the contract"
    return _kind_addr("contract", n.frm, style)


def _node_sentence(n: CallNode, rec: TxRecord, style: str) -> str:
    who = _actor(n, rec, style)
    v = format_eth(n.value)
    tgt = _addr(n.to, style)
    ct = n.call_type
    if ct == "create":
        what = f"contract {_addr(n.created, style)}" if n.created else "a contract"
        s = f"{who} creates {what}" + (f" with {v} ETH" if n.value else "")
    elif ct == "suicide":
        s = (f"{who} self-destructs and sends its remaining {v} ETH to {tgt}" if n.value
             else f"{who} self-destructs in favour of {tgt}")
    elif ct == "delegate_call":
        s = f"{who} runs the code of {tgt}{_fn(n)} in its own context"
    elif ct == "call_code":
        s = f"{who} borrows the code of {tgt}{_fn(n)}" + (f" with {v} ETH" if n.value else "")
    elif ct == "static_call":
        s = f"{who} reads from {tgt}{_fn(n)}"
    elif n.selector:
        s = f"{who} calls {tgt}{_fn(n)}" + (f" and passes {v} ETH" if n.value else "")
    elif n.value:
        s = f"{who} passes {v} ETH to {tgt}"
    else:
        s = f"{who} calls {tgt} with no data"
    return s + "." + _fail(n)


def _relation(n: CallNode, prev: CallNode) -> str:
    if n.path[:-1] == prev.path:
        return "Inside that call, "
    if n.path[:-1] == prev.path[:-1]:
        return "Then "
    return f"Back at level {number_word(n.depth)}, "


def _reverted_indices(nodes) -> frozenset:
    """``internal_index`` of every node inside a failed subtree (its effects are reverted)."""
    failed = [n.path for n in nodes if n.error]
    if not failed:
        return frozenset()
    return frozenset(n.internal_index for n in nodes
                     if n.internal_index is not None and any(n.path[:len(p)] == p for p in failed))


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def render_text(rec: TxRecord, max_nodes: int = 12, address_style: str = "short") -> str:
    """Deterministic English description of ``rec`` (at most ``max_nodes`` call nodes,
    then "and N further calls").  ``address_style``: ``"short"`` (``0x6ebe…27ce``),
    ``"full"`` or ``"named"`` (short, DAO contracts by name)."""
    if address_style not in ADDRESS_STYLES:
        raise ValueError(f"address_style must be one of {ADDRESS_STYLES}, got {address_style!r}")
    max_nodes = max(1, int(max_nodes))
    nodes = list(iter_nodes(rec.root, "dfs"))
    parts = [_root_sentence(rec, address_style)]
    shown = nodes[1:max_nodes]
    prev = rec.root
    for n in shown:
        parts.append(_relation(n, prev) + _node_sentence(n, rec, address_style))
        prev = n
    rest = len(nodes) - 1 - len(shown)
    if rest > 0:
        parts.append(f"… and {rest} further call{'s' if rest != 1 else ''}.")
    # value flow summary: value-carrying native transfers outside failed (reverted) subtrees
    pos = [t for t in rec.native_transfers if t.get("value")]
    reverted = _reverted_indices(nodes)
    eff = [t for t in pos if t.get("internal_index") not in reverted]
    if eff:
        tot = sum(int(t["value"]) for t in eff)
        snd = len({t.get("from_address") for t in eff})
        rcv = len({t.get("to_address") for t in eff})
        parts.append(f"In total {format_eth(tot)} ETH moves in {len(eff)} transfer{'s' if len(eff) != 1 else ''} "
                     f"from {snd} sender{'s' if snd != 1 else ''} to {rcv} receiver{'s' if rcv != 1 else ''}.")
    else:
        parts.append("No ETH moves.")
    if len(pos) > len(eff):
        k = len(pos) - len(eff)
        parts.append(f"{number_word(k).capitalize() if k < 100 else k} attempted transfer{'s are' if k != 1 else ' is'} "
                     f"reverted.")
    if rec.erc20_transfers:
        k = len(rec.erc20_transfers)
        toks = len({t.get("erc20") for t in rec.erc20_transfers})
        parts.append(f"There {'is' if k == 1 else 'are'} {k} token transfer{'s' if k != 1 else ''} "
                     f"of {toks} token{'s' if toks != 1 else ''}.")
    n_logs = sum(len(n.logs) for n in nodes)
    n_writes = sum(len(n.state_writes) for n in nodes)
    effects = []
    if n_logs:
        effects.append(f"emit {n_logs} event{'s' if n_logs != 1 else ''}")
    if n_writes:
        effects.append(f"change {n_writes} storage slot{'s' if n_writes != 1 else ''}")
    if effects:
        parts.append("The contracts " + " and ".join(effects) + ".")
    if not rec.success:
        ek = error_kind(rec.root.error) or "other"
        parts.append(f"The transaction fails ({_ERROR_WORDS.get(ek, 'error')}).")
    return " ".join(_cap(p) for p in parts)
