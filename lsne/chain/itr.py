"""ITR serialisation of Xatu execution traces (BlockGPT format, adapted).

``build_records`` turns the normalised rows of one chunk (``{table: rows}`` from
:func:`lsne.chain.xatu.load_chunk_rows`) into :class:`TxRecord` objects: one call tree per
transaction built from ``trace_address`` (``()`` = root, children sorted by the int tuple,
i.e. depth-first pre-order).  Logs and storage diffs / reads carry no node pointer (their
``internal_index`` is a per-table running number, docs/TRACE_SOURCE_SPEC.md §0), so they are
attached **by address and by feasibility** (:func:`attach_effects`):

* candidates for an item of address ``A`` are the frames whose code acts on ``A``'s storage
  and emits ``A``'s logs: ``call`` / ``create`` frames with target ``A`` and
  ``delegate_call`` / ``call_code`` frames whose *caller* is ``A`` (the code runs in the
  caller's context); frames inside a failed subtree (their effects are reverted, Xatu only
  stores effects that took place), ``static_call`` / ``suicide`` frames and the code
  address of a ``delegate_call`` / ``call_code`` are never candidates;
* a frame can only carry what its own gas paid for: its *self gas* (``gas_used`` minus the
  children's ``gas_used``) must cover the minimum cost of the items attached to it
  (``SSTORE`` change >= 5 000, ``LOG`` >= 375 + 375 per topic + 8 per data byte, ``SLOAD``
  >= 50), consumed in order;
* logs are assigned in ``log_index`` order monotonically along the execution order (the
  Euler tour of the call tree: a frame is "active" again after each child returns), each
  to the first affordable candidate at or after the previous log's position; storage
  diffs (and reads) to the first affordable candidate in depth-first order;
* when no candidate can afford an item it goes to the first non-failed candidate, then to
  the first node with that target (the old rule), then to the root; every attached item
  records how it was placed in ``attach`` (``"feasible"`` / ``"fallback"`` / ``"root"``).

The ``internal_index`` of ``native_transfers`` *is* the trace's ``internal_index`` (one row
per trace node, verified exact) and is kept on every node; every native transfer row is
flagged ``reverted`` (its node lies in a failed subtree, or the transaction failed: no ETH
moved) and ``phantom`` (a ``call_code`` / ``delegate_call`` node: the value is the caller's
own context value, no ETH moves).  :func:`effective_transfers` (value > 0, neither reverted
nor phantom) and :func:`net_flows` are the single definition of "ETH that moved" used by
:mod:`lsne.chain.labels` and :mod:`lsne.chain.text`.

``serialize`` linearises a record breadth-first (as in the BlockGPT paper)::

    [TX]
      [START] [CALL] [D{depth}] [CT:{type}] A:{from} A:{to} F:{selector} V:{N..} G:{N..} [INs] arg… [OUTs] [ERR:{kind}]? [END]
      [START] [STATE] [W] A:{address} S:{slot bucket} V:{N..} [END]      (state writes of that node)
      [START] [STATE] [R] A:{address} S:{slot bucket} V:{N..} [END]      (state reads, optional)
      [START] [LOG] A:{address} T:{topic0} [END]                         (logs of that node)
      …
    [/TX]

Numbers use two significant figures (:func:`numeric_token`).  Call arguments (32-byte
words after the selector): ``A:{address}`` for a left-padded 20-byte value that is either
>= 2**128 or an address the transaction itself touches, else :func:`numeric_token` for any
value below 2**160 (token amounts above 2**64 keep their magnitude), ``[WORD]`` for wider
values (hashes, negative numbers).  :func:`node_spans` gives the token span of every node's
``[CALL]`` block, so features can be restricted to what a ``max_len`` window shows.
Closed-class tokens that are not listed in the spec's SPECIAL set (``[CT:*]``,
``[ERR:*]``, ``S:*``, ``V:N…``, ``G:N…``, ``N10e-1``…``N90e-1`` for 1..9,
``A:none``/``F:none``/``T:none``) are enumerated by :func:`fixed_tokens` so that the
vocabulary can include them.
"""
from __future__ import annotations

import weakref
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

from .xatu import (TABLES, is_block_level, le_uint, load_chunk_rows, normalize_address,
                   normalize_hash, parse_int, parse_trace_address)

CALL_TYPES = ("call", "delegate_call", "call_code", "static_call", "create", "suicide")
ERROR_KINDS = ("out_of_gas", "bad_jump", "bad_instruction", "stack_underflow", "stack_overflow",
               "out_of_funds", "reverted", "call_depth", "invalid_opcode", "other")
SLOT_BUCKETS = tuple(str(i) for i in range(32)) + ("mid", "hash")
STRUCT_TOKENS = ("[TX]", "[/TX]", "[START]", "[END]", "[CALL]", "[STATE]", "[LOG]", "[INs]",
                 "[OUTs]", "[W]", "[R]", "[WORD]", "[A_OOV]", "[F_OOV]", "[T_OOV]")
NONE_TOKENS = ("A:none", "F:none", "T:none")
WORD_HEX = 64          # one 32-byte ABI word in hex characters
_CT_ALIASES = {"call": "call", "delegatecall": "delegate_call", "delegate_call": "delegate_call",
               "callcode": "call_code", "call_code": "call_code", "staticcall": "static_call",
               "static_call": "static_call", "create": "create", "create2": "create",
               "suicide": "suicide", "selfdestruct": "suicide"}


# ======================================================================================
# data model
# ======================================================================================
@dataclass
class CallNode:
    """One trace node.  ``path`` is the trace address (``()`` for the root); ``depth == len(path)``."""
    path: tuple
    call_type: str                 # call | delegate_call | call_code | static_call | create | suicide
    action_type: str               # call | create | suicide
    frm: str
    to: Optional[str]
    value: int
    gas: int
    gas_used: int
    selector: Optional[str]        # "0x" + 8 hex chars, None without >= 4 bytes of call data
    args: list                     # 32-byte words after the selector ("0x" + 64 hex), at most keep_args
    error: Optional[str]
    created: Optional[str]         # address created by a create node
    children: list = field(default_factory=list)
    logs: list = field(default_factory=list)
    state_writes: list = field(default_factory=list)
    state_reads: list = field(default_factory=list)
    internal_index: Optional[int] = None   # Xatu trace internal_index (1-based, = native_transfers')
    n_input_bytes: int = 0

    @property
    def depth(self) -> int:
        return len(self.path)

    @property
    def target(self) -> Optional[str]:
        """``to`` or, for a create node, the created address."""
        return self.to if self.to is not None else self.created

    def _own_dict(self) -> dict:
        return dict(path=list(self.path), call_type=self.call_type, action_type=self.action_type,
                    frm=self.frm, to=self.to, value=str(self.value), gas=self.gas, gas_used=self.gas_used,
                    selector=self.selector, args=list(self.args), error=self.error, created=self.created,
                    internal_index=self.internal_index, n_input_bytes=self.n_input_bytes,
                    logs=[dict(x) for x in self.logs], state_writes=[dict(x) for x in self.state_writes],
                    state_reads=[dict(x) for x in self.state_reads], children=[])

    def to_dict(self) -> dict:
        """Nested dict (``children`` lists); built iteratively, so call depths of 1 024 and
        more (the 2016 call-depth attacks) do not hit Python's recursion limit."""
        out = self._own_dict()
        stack = [(self, out)]
        while stack:
            n, d = stack.pop()
            for c in n.children:
                cd = c._own_dict()
                d["children"].append(cd)
                stack.append((c, cd))
        return out

    @classmethod
    def _from_own(cls, d: dict) -> "CallNode":
        return cls(path=tuple(d["path"]), call_type=d["call_type"], action_type=d["action_type"],
                   frm=d["frm"], to=d.get("to"), value=int(d["value"]), gas=int(d["gas"]),
                   gas_used=int(d["gas_used"]), selector=d.get("selector"), args=list(d.get("args", [])),
                   error=d.get("error"), created=d.get("created"), children=[],
                   logs=list(d.get("logs", [])), state_writes=list(d.get("state_writes", [])),
                   state_reads=list(d.get("state_reads", [])), internal_index=d.get("internal_index"),
                   n_input_bytes=int(d.get("n_input_bytes", 0)))

    @classmethod
    def from_dict(cls, d: dict) -> "CallNode":
        """Inverse of :meth:`to_dict` (iterative as well)."""
        root = cls._from_own(d)
        stack = [(d, root)]
        while stack:
            dd, n = stack.pop()
            for cd in dd.get("children", []):
                c = cls._from_own(cd)
                n.children.append(c)
                stack.append((cd, c))
        return root


@dataclass
class TxRecord:
    tx_hash: str
    block_number: int
    tx_index: int
    frm: str
    to: Optional[str]
    value: int
    gas_used: int
    success: bool
    root: CallNode
    native_transfers: list = field(default_factory=list)
    erc20_transfers: list = field(default_factory=list)
    n_nodes: int = 0
    max_depth: int = 0
    gas_source: str = "transaction"      # "transaction" (table row) or "trace_estimate" (no tx row)
    #: chunk-level evidence for the call endpoints of this transaction: ``"contract"`` (the
    #: address executes code, emits logs, owns storage, issues internal calls or is created
    #: anywhere in the chunk) or ``"account"`` (it signs a transaction or receives a call that
    #: runs no code); addresses without evidence are absent (see :func:`address_kinds`)
    address_kinds: dict = field(default_factory=dict)

    def nodes(self, order: str = "dfs") -> list:
        return list(iter_nodes(self.root, order))

    def to_dict(self) -> dict:
        def _tr(rows):
            return [{k: (str(v) if k == "value" and v is not None else v) for k, v in r.items()} for r in rows]
        return dict(tx_hash=self.tx_hash, block_number=self.block_number, tx_index=self.tx_index,
                    frm=self.frm, to=self.to, value=str(self.value), gas_used=self.gas_used,
                    success=self.success, n_nodes=self.n_nodes, max_depth=self.max_depth,
                    gas_source=self.gas_source, root=self.root.to_dict(),
                    native_transfers=_tr(self.native_transfers), erc20_transfers=_tr(self.erc20_transfers),
                    address_kinds=dict(self.address_kinds))

    @classmethod
    def from_dict(cls, d: dict) -> "TxRecord":
        def _tr(rows):
            return [{k: (int(v) if k == "value" and v is not None else v) for k, v in r.items()} for r in rows]
        return cls(tx_hash=d["tx_hash"], block_number=int(d["block_number"]), tx_index=int(d["tx_index"]),
                   frm=d["frm"], to=d.get("to"), value=int(d["value"]), gas_used=int(d["gas_used"]),
                   success=bool(d["success"]), root=CallNode.from_dict(d["root"]),
                   native_transfers=_tr(d.get("native_transfers", [])),
                   erc20_transfers=_tr(d.get("erc20_transfers", [])), n_nodes=int(d.get("n_nodes", 0)),
                   max_depth=int(d.get("max_depth", 0)), gas_source=d.get("gas_source", "transaction"),
                   address_kinds=dict(d.get("address_kinds") or {}))


def iter_nodes(root: CallNode, order: str = "dfs") -> Iterator[CallNode]:
    """Depth-first pre-order (= trace_address order) or breadth-first traversal."""
    if order == "dfs":
        stack = [root]
        while stack:
            n = stack.pop()
            yield n
            stack.extend(reversed(n.children))
    elif order == "bfs":
        q = deque([root])
        while q:
            n = q.popleft()
            yield n
            q.extend(n.children)
    else:
        raise ValueError(f"order must be 'dfs' or 'bfs', got {order!r}")


# ======================================================================================
# failed subtrees, transfer flags, effective transfers and net flows
# ======================================================================================
PHANTOM_CALL_TYPES = frozenset({"call_code", "delegate_call"})


def failed_nodes(root: CallNode) -> set:
    """``id()`` of every node inside a failed subtree (the node or an ancestor has an error):
    its value transfers, storage changes and logs are reverted."""
    out = set()
    stack = [(root, False)]
    while stack:
        n, bad = stack.pop()
        bad = bad or bool(n.error)
        if bad:
            out.add(id(n))
        stack.extend((c, bad) for c in n.children)
    return out


def reverted_indices(nodes_or_root) -> frozenset:
    """``internal_index`` of every node inside a failed subtree; ``nodes_or_root`` is the
    root :class:`CallNode` or a list of nodes (a failed node's subtree = the nodes whose
    path extends its path)."""
    if isinstance(nodes_or_root, CallNode):
        bad = failed_nodes(nodes_or_root)
        return frozenset(n.internal_index for n in iter_nodes(nodes_or_root, "dfs")
                         if id(n) in bad and n.internal_index is not None)
    nodes = list(nodes_or_root)
    failed = [n.path for n in nodes if n.error]
    if not failed:
        return frozenset()
    return frozenset(n.internal_index for n in nodes
                     if n.internal_index is not None and any(n.path[:len(p)] == p for p in failed))


def node_is_effective(n: CallNode, failed: set, success: bool = True) -> bool:
    """A node's value really moved: the transaction succeeded, the node is outside every
    failed subtree and it is not a ``call_code`` / ``delegate_call`` (context value only)."""
    return bool(success) and id(n) not in failed and n.call_type not in PHANTOM_CALL_TYPES


def flag_transfers(rec: "TxRecord") -> list:
    """Set ``reverted`` / ``phantom`` on every native transfer row of ``rec`` (in place, via
    the exact ``internal_index`` pointer) and return the rows.  ``reverted``: the node lies in
    a failed subtree or the transaction failed; ``phantom``: a ``call_code`` /
    ``delegate_call`` node.  A row whose pointer matches no node is kept (``reverted`` only
    when the transaction failed) and marked ``unmatched``."""
    failed = failed_nodes(rec.root)
    by_ii = {n.internal_index: n for n in iter_nodes(rec.root, "dfs") if n.internal_index is not None}
    for t in rec.native_transfers:
        n = by_ii.get(t.get("internal_index"))
        if n is None:
            t["reverted"] = not rec.success
            t["phantom"] = False
            t["unmatched"] = True
            continue
        t["reverted"] = bool((not rec.success) or id(n) in failed)
        t["phantom"] = n.call_type in PHANTOM_CALL_TYPES
    return rec.native_transfers


def effective_transfers(rec: "TxRecord", internal_indices: Optional[set] = None) -> list:
    """Native transfers that moved ETH: value > 0, not ``reverted``, not ``phantom`` (flags
    from :func:`build_records`; computed on the fly for records without them).  With
    ``internal_indices`` only rows of those nodes (e.g. the nodes visible in a token window)."""
    rows = rec.native_transfers
    if rows and any("reverted" not in t or "phantom" not in t for t in rows):
        rows = flag_transfers(rec)
    out = [t for t in rows if t.get("value") and not t.get("reverted") and not t.get("phantom")]
    if internal_indices is not None:
        out = [t for t in out if t.get("internal_index") in internal_indices]
    return out


def net_flows(transfers: Iterable[dict], key=None) -> dict:
    """Net balance change per address (``to`` +value, ``from`` -value) over ``transfers``;
    addresses with net 0 (pass-through contracts, full refunds) are dropped.  ``key``: an
    optional function ``row -> extra key`` (e.g. the ERC-20 token) giving ``{(key, address): net}``."""
    net: dict = {}
    for t in transfers:
        v = int(t.get("value") or 0)
        if not v:
            continue
        a, b = t.get("from_address"), t.get("to_address")
        a = str(a).lower() if a else a
        b = str(b).lower() if b else b
        k = key(t) if key is not None else None
        ka, kb = ((k, a), (k, b)) if key is not None else (a, b)
        net[ka] = net.get(ka, 0) - v
        net[kb] = net.get(kb, 0) + v
    return {k: v for k, v in net.items() if v != 0}


def flow_counts(net: dict) -> tuple:
    """``(n_recipients, n_sources)`` of a :func:`net_flows` result (distinct addresses with a
    positive / negative net change; for keyed flows an address counts once)."""
    def _addr(k):
        return k[1] if isinstance(k, tuple) else k
    rec = {_addr(k) for k, v in net.items() if v > 0}
    src = {_addr(k) for k, v in net.items() if v < 0}
    return len(rec), len(src)


# ======================================================================================
# record building
# ======================================================================================
def _hex_bytes(h: Optional[str]) -> bytes:
    if not h:
        return b""
    s = h[2:] if h[:2].lower() == "0x" else h
    if len(s) % 2:
        s = s + "0"
    try:
        return bytes.fromhex(s)
    except ValueError:
        return b""


def _call_type(action_type: str, call_type: Optional[str]) -> str:
    at = (action_type or "call").lower()
    if at == "call":
        return _CT_ALIASES.get((call_type or "call").lower(), "call")
    return _CT_ALIASES.get(at, at)


def _node_from_row(r: dict, keep_args: int) -> tuple[CallNode, bytes]:
    action_type = (normalize_address(r.get("action_type")) or "call")
    ct = _call_type(action_type, normalize_address(r.get("action_call_type")))
    data_hex = normalize_address(r.get("action_input"))
    data = _hex_bytes(data_hex)
    selector, args = None, []
    if ct not in ("create", "suicide") and len(data) >= 4:
        selector = "0x" + data[:4].hex()
        rest = data[4:]
        for i in range(0, min(len(rest), 32 * int(keep_args)), 32):
            args.append("0x" + rest[i:i + 32].hex())
    raw = data
    if ct == "create":
        raw = _hex_bytes(normalize_address(r.get("action_init")))
    node = CallNode(path=parse_trace_address(r.get("trace_address")), call_type=ct,
                    action_type=action_type, frm=normalize_address(r.get("action_from")),
                    to=normalize_address(r.get("action_to")), value=parse_int(r.get("action_value")) or 0,
                    gas=int(r.get("action_gas") or 0), gas_used=int(r.get("result_gas_used") or 0),
                    selector=selector, args=args, error=_err_text(r.get("error")),
                    created=normalize_address(r.get("result_address")) if ct == "create" else None,
                    internal_index=(int(r["internal_index"]) if r.get("internal_index") is not None else None),
                    n_input_bytes=len(raw))
    return node, raw


def _err_text(v) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, (bytes, bytearray)):
        v = bytes(v).decode("utf-8", errors="replace")
    s = str(v).strip()
    return s.lower() or None


def _intrinsic_gas(raw: bytes, is_create: bool) -> int:
    """Frontier/Homestead intrinsic gas: 21000 (+32000 create) + 4/68 per zero/non-zero byte."""
    nz = sum(1 for b in raw if b)
    return 21000 + (32000 if is_create else 0) + 4 * (len(raw) - nz) + 68 * nz


def _group(rows: Optional[Iterable[dict]]) -> dict:
    out: dict = {}
    for r in rows or ():
        h = normalize_hash(r.get("transaction_hash"))
        if h is None or is_block_level(h):
            continue
        out.setdefault(h, []).append(r)
    return out


def _trim_log(r: dict) -> dict:
    return dict(log_index=r.get("log_index"), address=normalize_address(r.get("address")),
                topic0=normalize_address(r.get("topic0")), topic1=normalize_address(r.get("topic1")),
                topic2=normalize_address(r.get("topic2")), topic3=normalize_address(r.get("topic3")),
                data=normalize_address(r.get("data")))


def _trim_diff(r: dict) -> dict:
    return dict(internal_index=r.get("internal_index"), address=normalize_address(r.get("address")),
                slot=normalize_address(r.get("slot")), from_value=normalize_address(r.get("from_value")),
                to_value=normalize_address(r.get("to_value")))


def _trim_read(r: dict) -> dict:
    addr = r.get("contract_address", r.get("address"))
    return dict(internal_index=r.get("internal_index"), address=normalize_address(addr),
                slot=normalize_address(r.get("slot")), value=normalize_address(r.get("value")))


def _trim_native(r: dict) -> dict:
    return dict(internal_index=r.get("internal_index"), transfer_index=r.get("transfer_index"),
                from_address=normalize_address(r.get("from_address")),
                to_address=normalize_address(r.get("to_address")), value=le_uint(r.get("value")) or 0)


def _trim_erc20(r: dict) -> dict:
    return dict(internal_index=r.get("internal_index"), log_index=r.get("log_index"),
                erc20=normalize_address(r.get("erc20")), from_address=normalize_address(r.get("from_address")),
                to_address=normalize_address(r.get("to_address")), value=le_uint(r.get("value")) or 0)


# minimum gas a frame must have spent itself for an attached item (Frontier / Homestead):
# SSTORE that changes a slot >= 5 000 (20 000 from zero), LOG = 375 + 375 per topic + 8 per
# data byte (plus memory), SLOAD = 50 (200 after EIP-150)
SSTORE_MIN_GAS = 5000
SLOAD_MIN_GAS = 50
LOG_BASE_GAS, LOG_TOPIC_GAS, LOG_BYTE_GAS = 375, 375, 8


def log_min_gas(lg: dict) -> int:
    """Minimum gas of the ``LOG`` opcode that emitted ``lg`` (topics and data bytes)."""
    n_topics = sum(1 for k in ("topic0", "topic1", "topic2", "topic3") if lg.get(k))
    data = lg.get("data") or ""
    h = data[2:] if str(data)[:2].lower() == "0x" else str(data)
    return LOG_BASE_GAS + LOG_TOPIC_GAS * n_topics + LOG_BYTE_GAS * (len(h) // 2)


def context_address(n: CallNode) -> Optional[str]:
    """The address whose storage / logs the frame's code acts on: the target of a ``call`` /
    ``static_call``, the created address of a ``create``, the *caller* of a
    ``delegate_call`` / ``call_code`` (the code runs in the caller's context); None for
    ``suicide``."""
    if n.call_type in PHANTOM_CALL_TYPES:
        return n.frm
    if n.call_type == "create":
        return n.created
    if n.call_type == "suicide":
        return None
    return n.to


def self_gas(n: CallNode) -> int:
    """Gas the frame spent itself: ``gas_used`` minus the children's ``gas_used`` (>= 0)."""
    return max(0, int(n.gas_used or 0) - sum(int(c.gas_used or 0) for c in n.children))


def _euler_tour(root: CallNode) -> list:
    """Execution order of the frames: a frame appears on entry and again after each child."""
    tour = []
    stack = [(root, 0)]
    while stack:
        n, i = stack.pop()
        tour.append(n)
        if i < len(n.children):
            stack.append((n, i + 1))
            stack.append((n.children[i], 0))
    return tour


def attach_effects(root: CallNode, logs: Sequence[dict] = (), writes: Sequence[dict] = (),
                   reads: Sequence[dict] = ()) -> dict:
    """Attach trimmed log / storage-diff / storage-read rows to the frames that can have
    produced them (rule in the module docstring); appends to ``node.logs`` /
    ``state_writes`` / ``state_reads``, sets ``attach`` on every item and returns the counts
    per placement ``{"feasible": n, "fallback": n, "root": n}``."""
    dfs = list(iter_nodes(root, "dfs"))
    failed = failed_nodes(root)
    budget = {id(n): self_gas(n) for n in dfs}
    first_target: dict = {}
    writers: dict = {}        # address -> candidate frames for writes / logs (DFS order)
    readers: dict = {}        # address -> candidate frames for reads (static calls too)
    for n in dfs:
        for a in (n.to, n.created):
            if a and a not in first_target:
                first_target[a] = n
        a = context_address(n)
        if a and id(n) not in failed:
            readers.setdefault(a, []).append(n)
            if n.call_type != "static_call":
                writers.setdefault(a, []).append(n)
    counts = {"feasible": 0, "fallback": 0, "root": 0}

    def _fallback(addr, cands):
        if cands:
            return cands[0], "fallback"
        n = first_target.get(addr)
        return (n, "fallback") if n is not None else (root, "root")

    # logs: monotone along the execution order (Euler tour), first affordable candidate
    if logs:
        tour = _euler_tour(root)
        pos_of: dict = {}
        for p, n in enumerate(tour):
            a = context_address(n)
            if a and id(n) not in failed and n.call_type != "static_call":
                pos_of.setdefault(a, []).append(p)
        pointer = 0
        for lg in logs:
            addr, cost = lg.get("address"), log_min_gas(lg)
            target, how = None, None
            ps = pos_of.get(addr, [])
            for p in ps[bisect_left(ps, pointer):]:
                if budget[id(tour[p])] >= cost:
                    target, how, pointer = tour[p], "feasible", p
                    break
            if target is None:                    # not monotone, but affordable somewhere
                target = next((n for n in writers.get(addr, ()) if budget[id(n)] >= cost), None)
                how = "fallback" if target is not None else None
            if target is None:
                target, how = _fallback(addr, writers.get(addr))
            else:
                budget[id(target)] -= cost
            lg["attach"] = how
            counts[how] += 1
            target.logs.append(lg)
    for items, cost, pool, attr in ((writes, SSTORE_MIN_GAS, writers, "state_writes"),
                                    (reads, SLOAD_MIN_GAS, readers, "state_reads")):
        for it in items:
            addr = it.get("address")
            target = next((n for n in pool.get(addr, ()) if budget[id(n)] >= cost), None)
            if target is None:
                target, how = _fallback(addr, pool.get(addr))
            else:
                budget[id(target)] -= cost
                how = "feasible"
            it["attach"] = how
            counts[how] += 1
            getattr(target, attr).append(it)
    return counts


def address_kinds(chunk_records: Sequence["TxRecord"], logs: Iterable[dict] = (),
                  writes: Iterable[dict] = ()) -> tuple:
    """``(contracts, accounts)``: chunk-level evidence about addresses.  Contract: created by a
    ``create`` frame, issues an internal call (the ``from`` of a non-root frame), runs code
    when called (a successful frame with ``gas_used > 0`` or children), emits a log or owns a
    storage diff.  Account: signs a transaction (the root sender) or receives a successful
    call that runs no code (``gas_used == 0``, no children).  Code evidence wins."""
    code, eoa = set(), set()
    for r in chunk_records:
        if r.frm:
            eoa.add(r.frm)
        failed = failed_nodes(r.root)
        for n in iter_nodes(r.root, "dfs"):
            if n.created:
                code.add(n.created)
            if n.path and n.frm:
                code.add(n.frm)
            if n.path == () and n.frm:
                eoa.add(n.frm)
            if n.to and id(n) not in failed and n.call_type != "suicide":
                if n.gas_used > 0 or n.children:
                    code.add(n.to)
                elif n.call_type in ("call", "call_code", "delegate_call"):
                    eoa.add(n.to)
    code.update(x.get("address") for x in logs if x.get("address"))
    code.update(x.get("address") for x in writes if x.get("address"))
    code.discard(None)
    return code, eoa - code


def _sort_key(r: dict, *keys):
    return tuple((r.get(k) if r.get(k) is not None else -1) for k in keys)


def build_records(chunk_rows: dict, keep_args: int = 16) -> list:
    """Group all tables of one chunk by (lower-case) transaction hash and build one
    :class:`TxRecord` per transaction; block-level rows are skipped.

    ``chunk_rows`` maps table names (``traces``, ``transaction``, ``native_transfers``,
    ``erc20_transfers``, ``logs``, ``storage_diffs``, ``storage_reads``; the
    ``canonical_execution_`` prefix is accepted) to row lists, normalised by
    :func:`lsne.chain.xatu.read_chunk` or raw pyarrow rows (both are coerced).
    Without a ``transaction`` row the transaction fields come from the root trace and
    ``gas_used`` is estimated as root gas used + intrinsic gas (``gas_source="trace_estimate"``;
    refunds are not visible in the traces).  Records are sorted by (block, tx index).
    """
    rows = {k[len("canonical_execution_"):] if k.startswith("canonical_execution_") else k: v
            for k, v in chunk_rows.items()}
    unknown = set(rows) - set(TABLES) - {"contracts", "balance_diffs"}
    if unknown:
        raise ValueError(f"unknown tables in chunk_rows: {sorted(unknown)}")
    traces = _group(rows.get("traces"))
    txs = {h: rs[0] for h, rs in _group(rows.get("transaction")).items()}
    natives = _group(rows.get("native_transfers"))
    erc20s = _group(rows.get("erc20_transfers"))
    logs = _group(rows.get("logs"))
    diffs = _group(rows.get("storage_diffs"))
    reads = _group(rows.get("storage_reads"))

    records = []
    for h in sorted(set(traces) | set(txs)):
        trow = txs.get(h)
        nodes, raws = [], {}
        for r in traces.get(h, ()):
            n, raw = _node_from_row(r, keep_args)
            nodes.append(n)
            raws[id(n)] = raw
        nodes.sort(key=lambda n: n.path)
        by_path = {}
        for n in nodes:
            by_path.setdefault(n.path, n)          # duplicate paths: keep the first row
        nodes = list(by_path.values())
        root = by_path.get(())
        if root is None:
            if trow is not None:
                data = _hex_bytes(normalize_address(trow.get("input")))
                to = normalize_address(trow.get("to_address"))
                root = CallNode(path=(), call_type="call" if to else "create", action_type="call" if to else "create",
                                frm=normalize_address(trow.get("from_address")), to=to,
                                value=le_uint(trow.get("value")) or 0, gas=int(trow.get("gas_limit") or 0),
                                gas_used=int(trow.get("gas_used") or 0),
                                selector=("0x" + data[:4].hex()) if to and len(data) >= 4 else None,
                                args=[], error=None if trow.get("success", True) else "failed", created=None,
                                n_input_bytes=len(data))
                raws[id(root)] = data
            elif nodes:
                first = nodes[0]
                root = CallNode(path=(), call_type="call", action_type="call", frm=first.frm, to=None,
                                value=0, gas=0, gas_used=0, selector=None, args=[], error=None, created=None)
                raws[id(root)] = b""
            else:
                continue
            nodes.insert(0, root)
            by_path[()] = root
        # tree: parent = longest existing proper prefix (orphans re-attached, never lost)
        for n in nodes:
            if n is root:
                continue
            p = n.path[:-1]
            while p and p not in by_path:
                p = p[:-1]
            by_path[p].children.append(n)
        for n in nodes:
            n.children.sort(key=lambda c: c.path)
        dfs = list(iter_nodes(root, "dfs"))
        attach_effects(root,
                       [_trim_log(r) for r in sorted(logs.get(h, ()), key=lambda r: _sort_key(r, "log_index", "internal_index"))],
                       [_trim_diff(r) for r in sorted(diffs.get(h, ()), key=lambda r: _sort_key(r, "internal_index"))],
                       [_trim_read(r) for r in sorted(reads.get(h, ()), key=lambda r: _sort_key(r, "internal_index"))])
        nat = [_trim_native(r) for r in sorted(natives.get(h, ()),
                                               key=lambda r: _sort_key(r, "transfer_index", "internal_index"))]
        erc = [_trim_erc20(r) for r in sorted(erc20s.get(h, ()),
                                              key=lambda r: _sort_key(r, "log_index", "internal_index"))]
        sample = trow if trow is not None else traces[h][0]
        if trow is not None:
            frm = normalize_address(trow.get("from_address"))
            to = normalize_address(trow.get("to_address"))
            value = le_uint(trow.get("value")) or 0
            gas_used = int(trow.get("gas_used") or 0)
            success = bool(trow.get("success")) if trow.get("success") is not None else root.error is None
            gas_source = "transaction"
        else:
            frm, to, value = root.frm, root.to, root.value
            success = root.error is None
            intr = _intrinsic_gas(raws.get(id(root), b""), root.call_type == "create")
            gas_used = (root.gas if root.error else root.gas_used) + intr
            gas_source = "trace_estimate"
        rec = TxRecord(tx_hash=h, block_number=int(sample.get("block_number") or 0),
                       tx_index=int(sample.get("transaction_index") or 0), frm=frm, to=to,
                       value=value, gas_used=gas_used, success=success, root=root,
                       native_transfers=nat, erc20_transfers=erc, n_nodes=len(dfs),
                       max_depth=max(n.depth for n in dfs), gas_source=gas_source)
        flag_transfers(rec)
        records.append(rec)
    records.sort(key=lambda r: (r.block_number, r.tx_index, r.tx_hash))
    # contract / account evidence over the whole chunk, kept for each record's endpoints
    code, eoa = address_kinds(records, (_trim_log(r) for rs in logs.values() for r in rs),
                              (_trim_diff(r) for rs in diffs.values() for r in rs))
    for rec in records:
        kinds = {}
        for n in iter_nodes(rec.root, "dfs"):
            for a in (n.frm, n.to, n.created):
                if a and a not in kinds:
                    if a in code:
                        kinds[a] = "contract"
                    elif a in eoa:
                        kinds[a] = "account"
        rec.address_kinds = kinds
    return records


def load_records(start: int, cache_dir: str = "cache/xatu", tables=TABLES, keep_args: int = 16) -> list:
    """``build_records(load_chunk_rows(start, cache_dir, tables))``."""
    return build_records(load_chunk_rows(start, cache_dir, tables), keep_args=keep_args)


# ======================================================================================
# tokens
# ======================================================================================
def numeric_token(x: int) -> str:
    """Two significant figures as in BlockGPT: ``0 -> "N0"``, else ``"N{m}e{k}"`` with the
    2-digit mantissa ``m`` in 10..99 and ``x ≈ m·10^k`` (round half up).  1..9 are exact
    with ``k = -1`` (``1 -> "N10e-1"``); ``1254 -> "N13e2"``, ``5e18 -> "N50e17"``."""
    x = int(x)
    if x < 0:
        raise ValueError(f"numeric_token expects a non-negative integer, got {x}")
    if x == 0:
        return "N0"
    if x < 10:
        return f"N{10 * x}e-1"
    k = len(str(x)) - 2
    if k == 0:
        return f"N{x}e0"
    p = 10 ** k
    m = (x + p // 2) // p
    if m >= 100:
        m //= 10
        k += 1
    return f"N{m}e{k}"


def numeric_tokens(max_exp: int = 77) -> list:
    """Every token :func:`numeric_token` can produce for uint256 values."""
    out = ["N0"] + [f"N{10 * d}e-1" for d in range(1, 10)]
    out += [f"N{m}e{k}" for k in range(0, int(max_exp) + 1) for m in range(10, 100)]
    return out


def error_kind(err: Optional[str]) -> Optional[str]:
    """Map a free-text trace error to one of :data:`ERROR_KINDS` (``None`` when no error)."""
    if err is None:
        return None
    e = str(err).lower().replace("_", " ")
    squeezed = e.replace(" ", "")
    if "outofgas" in squeezed or "oog" in squeezed or "gas uint64 overflow" in e:
        return "out_of_gas"
    if "jump" in e:
        return "bad_jump"
    if "outoffunds" in squeezed or "insufficient" in e:
        return "out_of_funds"
    if "revert" in e:
        return "reverted"
    if "underflow" in e:
        return "stack_underflow"
    if "stack limit" in e or "overflow" in e:
        return "stack_overflow"
    if "depth" in e or "toodeep" in squeezed:          # Xatu / Parity: "CallTooDeep"
        return "call_depth"
    if "bad instruction" in e:
        return "bad_instruction"
    if "invalid" in e or "opcode" in e:
        return "invalid_opcode"
    return "other"


def slot_bucket(slot) -> str:
    """Coarse storage-slot class: ``"0"``..``"31"`` for small slots (plain state variables),
    ``"mid"`` below 2**64, ``"hash"`` above (keccak-derived mapping / array slots)."""
    if slot is None:
        return "0"
    v = slot if isinstance(slot, int) else (parse_int(slot) if str(slot).strip() else 0)
    if v < 32:
        return str(v)
    if v < 2 ** 64:
        return "mid"
    return "hash"


def fixed_tokens(max_depth_token: int = 32) -> list:
    """All closed-class tokens :func:`serialize` can emit (everything except ``A:0x…``,
    ``F:0x…``, ``T:0x…``): structural, depth, call-type, error, slot-bucket and numeric
    tokens (bare for args, ``V:``/``G:``-prefixed for values and gas)."""
    nums = numeric_tokens()
    out = list(STRUCT_TOKENS) + [f"[D{d}]" for d in range(int(max_depth_token) + 1)]
    out += [f"[CT:{c}]" for c in CALL_TYPES] + [f"[ERR:{e}]" for e in ERROR_KINDS]
    out += [f"S:{b}" for b in SLOT_BUCKETS] + list(NONE_TOKENS)
    out += nums + [f"V:{n}" for n in nums] + [f"G:{n}" for n in nums]
    return out


class _VocabView:
    """Membership view of a TraceVocab-like object (``tokens`` and optional ``top_*`` lists)."""

    def __init__(self, vocab):
        self.tokens = set(getattr(vocab, "tokens", None) or ())
        self.addresses = set(getattr(vocab, "top_addresses", None) or ())
        self.selectors = set(getattr(vocab, "top_selectors", None) or ())
        self.topics = set(getattr(vocab, "top_topics", None) or ())
        self.size = len(getattr(vocab, "tokens", None) or ())


_VIEWS: dict = {}


def _view(vocab) -> Optional[_VocabView]:
    if vocab is None:
        return None
    if isinstance(vocab, _VocabView):
        return vocab
    key = id(vocab)
    hit = _VIEWS.get(key)
    if hit is not None:
        ref, view = hit
        if ref() is vocab and view.size == len(getattr(vocab, "tokens", None) or ()):
            return view
    view = _VocabView(vocab)
    try:
        ref = weakref.ref(vocab, lambda _r, k=key: _VIEWS.pop(k, None))
        _VIEWS[key] = (ref, view)
    except TypeError:
        pass
    return view


def _open_token(prefix: str, value: Optional[str], V: Optional[_VocabView], members: str, oov: str) -> str:
    tok = f"{prefix}:{value}" if value else f"{prefix}:none"
    if V is None:
        return tok
    if tok in V.tokens or (value and value in getattr(V, members)):
        return tok
    return oov


def _arg_token(word: str, V: Optional[_VocabView], known: Optional[set] = None) -> str:
    """One 32-byte argument word: ``A:{address}`` for a left-padded 20-byte value that is
    >= 2**128 or (>= 2**64 and) an address in ``known`` (addresses the transaction touches),
    :func:`numeric_token` for any other value below 2**160, ``[WORD]`` for wider values."""
    h = word[2:] if word.startswith("0x") else word
    if len(h) != WORD_HEX:
        return "[WORD]"
    v = int(h, 16)
    if h[:24] == "0" * 24:                        # fits in 20 bytes
        addr = "0x" + h[24:]
        if v >= 2 ** 128 or (v >= 2 ** 64 and known is not None and addr in known):
            return _open_token("A", addr, V, "addresses", "[A_OOV]")
        return numeric_token(v)
    return "[WORD]"


def known_addresses(rec: "TxRecord") -> set:
    """Addresses the transaction touches (call endpoints, created contracts, log emitters,
    storage owners, transfer endpoints, ERC-20 tokens): the reference set that lets
    :func:`_arg_token` recognise addresses with leading zero bytes."""
    out = set()
    for n in iter_nodes(rec.root, "dfs"):
        out.update(a for a in (n.frm, n.to, n.created) if a)
        out.update(x.get("address") for x in n.logs if x.get("address"))
        out.update(x.get("address") for x in n.state_writes if x.get("address"))
        out.update(x.get("address") for x in n.state_reads if x.get("address"))
    for t in rec.native_transfers:
        out.update(a for a in (t.get("from_address"), t.get("to_address")) if a)
    for t in rec.erc20_transfers:
        out.update(a for a in (t.get("from_address"), t.get("to_address"), t.get("erc20")) if a)
    return {str(a).lower() for a in out}


def _call_tokens(n: CallNode, V, max_depth_token: int, max_args: int, known: Optional[set] = None) -> list:
    out = ["[START]", "[CALL]", f"[D{min(n.depth, int(max_depth_token))}]", f"[CT:{n.call_type}]",
           _open_token("A", n.frm, V, "addresses", "[A_OOV]"),
           _open_token("A", n.target, V, "addresses", "[A_OOV]"),
           _open_token("F", n.selector, V, "selectors", "[F_OOV]"),
           f"V:{numeric_token(n.value)}", f"G:{numeric_token(n.gas)}", "[INs]"]
    out += [_arg_token(w, V, known) for w in n.args[:int(max_args)]]
    out.append("[OUTs]")
    ek = error_kind(n.error)
    if ek is not None:
        out.append(f"[ERR:{ek}]")
    out.append("[END]")
    return out


def _effect_tokens(n: CallNode, V, include_state: bool, include_reads: bool, include_logs: bool) -> list:
    out = []
    if include_state:
        for w in n.state_writes:
            out += ["[START]", "[STATE]", "[W]", _open_token("A", w.get("address"), V, "addresses", "[A_OOV]"),
                    f"S:{slot_bucket(w.get('slot'))}", f"V:{numeric_token(parse_int(w.get('to_value')) or 0)}",
                    "[END]"]
        if include_reads:
            for r in n.state_reads:
                out += ["[START]", "[STATE]", "[R]", _open_token("A", r.get("address"), V, "addresses", "[A_OOV]"),
                        f"S:{slot_bucket(r.get('slot'))}", f"V:{numeric_token(parse_int(r.get('value')) or 0)}",
                        "[END]"]
    if include_logs:
        for lg in n.logs:
            out += ["[START]", "[LOG]", _open_token("A", lg.get("address"), V, "addresses", "[A_OOV]"),
                    _open_token("T", lg.get("topic0"), V, "topics", "[T_OOV]"), "[END]"]
    return out


def _node_tokens(n: CallNode, V, max_depth_token: int, max_args: int, include_state: bool,
                 include_reads: bool, include_logs: bool, known: Optional[set] = None) -> list:
    return (_call_tokens(n, V, max_depth_token, max_args, known)
            + _effect_tokens(n, V, include_state, include_reads, include_logs))


def _serialize_spans(rec: TxRecord, vocab=None, max_depth_token: int = 32, max_args: int = 4,
                     include_state: bool = True, include_logs: bool = True, include_reads: bool = True) -> tuple:
    V = _view(vocab)
    known = known_addresses(rec)
    out = ["[TX]"]
    spans = []
    for n in iter_nodes(rec.root, "bfs"):
        call = _call_tokens(n, V, max_depth_token, max_args, known)
        spans.append((n, len(out), len(out) + len(call)))
        out += call
        out += _effect_tokens(n, V, include_state, include_reads, include_logs)
    out.append("[/TX]")
    return out, spans


def serialize(rec: TxRecord, vocab=None, max_depth_token: int = 32, max_args: int = 4,
              include_state: bool = True, include_logs: bool = True, include_reads: bool = True) -> list:
    """ITR token list of one transaction (breadth-first over the call tree, see module doc).

    With ``vocab`` (a ``TraceVocab``: ``tokens`` plus optional ``top_addresses`` /
    ``top_selectors`` / ``top_topics``) addresses, selectors and topics outside the
    vocabulary become ``[A_OOV]`` / ``[F_OOV]`` / ``[T_OOV]``; without it raw strings are
    kept.  State reads are emitted when ``include_state`` and ``include_reads`` are set.
    """
    return _serialize_spans(rec, vocab, max_depth_token, max_args, include_state, include_logs, include_reads)[0]


def node_spans(rec: TxRecord, **serialize_kw) -> list:
    """``[(node, start, end)]`` in serialisation order: the half-open token span of every
    node's ``[CALL]`` block in :func:`serialize` (same keywords; index 0 is ``[TX]``)."""
    return _serialize_spans(rec, None, **serialize_kw)[1]


def visible_nodes(rec: TxRecord, max_len: int = 512, **serialize_kw) -> list:
    """Nodes whose whole ``[CALL]`` block lies inside the first ``max_len`` tokens of
    :func:`serialize` (what a model with a ``max_len`` window sees of the call tree)."""
    return [n for n, a, b in node_spans(rec, **serialize_kw) if b <= int(max_len)]


def tree_text(rec: TxRecord) -> str:
    """Human-readable indented dump (debugging)."""
    lines = [f"tx {rec.tx_hash} block {rec.block_number} idx {rec.tx_index} "
             f"{'ok' if rec.success else 'FAILED'} gas_used {rec.gas_used} ({rec.gas_source}) "
             f"nodes {rec.n_nodes} depth {rec.max_depth}"]
    for n in iter_nodes(rec.root, "dfs"):
        ind = "  " * (n.depth + 1)
        path = "_".join(map(str, n.path)) or "root"
        eth = n.value / 1e18
        s = (f"{ind}[{path}] {n.call_type} {n.frm} -> {n.target} value={eth:.6g} ETH gas={n.gas} "
             f"used={n.gas_used}")
        if n.selector:
            s += f" fn={n.selector}"
        if n.error:
            s += f" ERROR({n.error})"
        lines.append(s)
        for w in n.state_writes:
            lines.append(f"{ind}  W {w.get('address')} slot={w.get('slot')} -> {w.get('to_value')}")
        for r in n.state_reads:
            lines.append(f"{ind}  R {r.get('address')} slot={r.get('slot')} = {r.get('value')}")
        for lg in n.logs:
            lines.append(f"{ind}  LOG {lg.get('address')} {lg.get('topic0')}")
    eff = effective_transfers(rec)
    tot = sum(t["value"] for t in eff)
    n_rev = sum(1 for t in rec.native_transfers if t.get("value") and t.get("reverted"))
    n_ph = sum(1 for t in rec.native_transfers if t.get("value") and t.get("phantom"))
    lines.append(f"  native transfers: {len(rec.native_transfers)} rows, {len(eff)} effective moving "
                 f"{tot / 1e18:.6g} ETH ({n_rev} reverted, {n_ph} phantom); "
                 f"erc20 transfers: {len(rec.erc20_transfers)}")
    return "\n".join(lines)
