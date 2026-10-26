"""'Sharing' labels, DAO-era segmentation, matched probe sets and the surface baseline.

**Economic fan-out / fan-in on net flows** (docs/TRACE_SOURCE_SPEC.md §4, DESIGN §4.9):
Xatu writes one ``native_transfers`` row per trace node -- zero-value calls, value rows of
failed subtrees and of failed transactions (no ETH moved) and the context value of
``call_code`` / ``delegate_call`` frames (no ETH moves) included.  The labels use only the
**effective** transfers (:func:`lsne.chain.itr.effective_transfers`: value > 0, not
``reverted``, not ``phantom``) and count **net flows** (:func:`lsne.chain.itr.net_flows`):
per transaction every address's net ETH change, addresses with net 0 dropped (contracts
that pass ETH on, full refunds).  ``fan_out_eth`` = number of addresses with a net inflow
(recipients), ``fan_in_eth`` = number with a net outflow (sources).  A transaction that
routes ETH through one contract to one recipient is ``single`` (1 / 1), not
``many_to_many``; a refund to the sender cancels; a contract that splits one deposit over
several recipients is ``distribution``.  ERC-20 (``fan_out_erc20`` / ``fan_in_erc20``): the
same per (token, address).  The gross counts over the effective transfers are kept as
``fan_out_eth_gross`` / ``fan_in_eth_gross``, the structural counts over *all* rows (one per
trace node) as ``n_call_targets`` / ``n_call_sources`` (``sharing_label(...,
structural=True)``).  ``n_value_calls``, ``total_eth`` (gross ETH of the effective
transfers) and ``net_eth`` (sum of the net inflows) use the same effective transfers.
``"none"`` (no value transfer) means: no net ETH flow and no net ERC-20 flow.

**Visible window** (``visible_len``, default 512 = the trace LM's context): the model sees
only the first ``max_len`` ITR tokens (breadth-first), so long transactions lose most of
their value transfers.  ``fan_out_eth_visible`` / ``fan_in_eth_visible`` /
``n_value_calls_visible`` repeat the counts over the nodes whose ``[CALL]`` block lies inside
the window (:func:`lsne.chain.itr.visible_nodes`); the fan-out probe requires the label on
the visible window too (:func:`probe_sets`, ``visible``).

**DAO involvement**: a transaction touches (call endpoint, log emitter, storage owner,
transfer endpoint, ERC-20 party) an address of EIP-779's list ``L`` (the DAO, its
extraBalance and all child DAOs with their extraBalances, 116 accounts, identical to
go-ethereum's ``DAODrainList``), the DAO creator or the WithdrawDAO contract.  Mentions only
as call arguments (``dao_mentioned``) are ambiguous: such transactions are excluded from the
DAO probe's negatives.  The chunk-1718000 per-transaction statistics of the spec §0 included
the block-level reward pseudo-transaction (1 000 rewards to 51 miners); without it the
structural fan-out (``n_call_targets``) of the 7 609 transactions has median 1, 99th
percentile 6, maximum 9, and 186 transactions with >= 5.

**Probe sets** (:func:`probe_sets`) are matched 1:1 on era and gas_used decile; when that
yields fewer than ``min_pairs`` pairs the matching falls back to era only, and the match
table records which matching was used and the standardized mean difference of gas
(``gas_smd``, ``log_gas_smd``) between the matched positives and negatives.

**Length-matched probe sets** (round 2, docs/TRACE_SOURCE.md "Kontrollierte Sonden"):
``probe_sets(..., match="length")`` pairs every positive with a negative of the same era
whose ITR token count lies within a factor ``length_ratio`` (default 1.25, i.e.
``|log n_pos - log n_neg| <= log 1.25``), nearest length first, without replacement.  The
first run showed that gas-decile matching leaves the high fan-out positives 12x longer than
their negatives, so a fan-out "direction" was a length direction.  Balance is reported as
standardized mean differences (``length_smd`` on log token count, ``n_tokens_smd``,
``gas_smd``, ``log_gas_smd``; all available covariates in ``match_table.attrs["balance"]``).

**Identity masking** (:func:`mask_tokens`): the DAO probe separated the sets through the
DAO address tokens themselves (always in the vocabulary).  ``mask_tokens(tokens, "dao")``
replaces the DAO identities (every address of :data:`DAO_INVOLVED`) by ``[A_OOV]`` before
encoding, in *both* sets, so the separation has to come from context; ``"all_addresses"``
replaces every address token (stricter: also removes the ``[A_OOV]`` count difference the
DAO masking itself creates).

**Surface baseline** (:func:`surface_baseline_auc`): split-half AUC of trivial length /
count statistics of the token sequence (token count, ``[CALL]`` count, ``V:`` token count,
value-carrying calls; optionally the ``[A_OOV]`` count), with exactly the split of
:func:`lsne.chain.source.holdout_auc`, so a sharing direction can be compared with what
length and counts alone achieve.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .itr import (TxRecord, effective_transfers, failed_nodes, flow_counts, iter_nodes, net_flows,
                  node_is_effective, serialize, visible_nodes)

DAO_ADDRESSES = {"dao": "0xbb9bc244d798123fde783fcc1c72d3bb8c189413",
                 "extra_balance": "0x807640a13483f8ac783c557fcdf27be11ea4ac7a",
                 "creator": "0x4a574510c7014e4ae985403536074abe582adfc8",
                 "withdraw_dao": "0xbf4ed7b27f1d666546e30d74d50d173d20bca754"}
#: EIP-779 list ``L`` (= go-ethereum ``params.DAODrainList``): the DAO, its extraBalance and every
#: child DAO of the creator with its extraBalance; balances moved to WithdrawDAO at block 1 920 000
DAO_FORK_LIST = (
    "0xd4fe7bc31cedb7bfb8a345f31e668033056b2728", "0xb3fb0e5aba0e20e5c49d252dfd30e102b171a425",
    "0x2c19c7f9ae8b751e37aeb2d93a699722395ae18f", "0xecd135fa4f61a655311e86238c92adcd779555d2",
    "0x1975bd06d486162d5dc297798dfc41edd5d160a7", "0xa3acf3a1e16b1d7c315e23510fdd7847b48234f6",
    "0x319f70bab6845585f412ec7724b744fec6095c85", "0x06706dd3f2c9abf0a21ddcc6941d9b86f0596936",
    "0x5c8536898fbb74fc7445814902fd08422eac56d0", "0x6966ab0d485353095148a2155858910e0965b6f9",
    "0x779543a0491a837ca36ce8c635d6154e3c4911a6", "0x2a5ed960395e2a49b1c758cef4aa15213cfd874c",
    "0x5c6e67ccd5849c0d29219c4f95f1a7a93b3f5dc5", "0x9c50426be05db97f5d64fc54bf89eff947f0a321",
    "0x200450f06520bdd6c527622a273333384d870efb", "0xbe8539bfe837b67d1282b2b1d61c3f723966f049",
    "0x6b0c4d41ba9ab8d8cfb5d379c69a612f2ced8ecb", "0xf1385fb24aad0cd7432824085e42aff90886fef5",
    "0xd1ac8b1ef1b69ff51d1d401a476e7e612414f091", "0x8163e7fb499e90f8544ea62bbf80d21cd26d9efd",
    "0x51e0ddd9998364a2eb38588679f0d2c42653e4a6", "0x627a0a960c079c21c34f7612d5d230e01b4ad4c7",
    "0xf0b1aa0eb660754448a7937c022e30aa692fe0c5", "0x24c4d950dfd4dd1902bbed3508144a54542bba94",
    "0x9f27daea7aca0aa0446220b98d028715e3bc803d", "0xa5dc5acd6a7968a4554d89d65e59b7fd3bff0f90",
    "0xd9aef3a1e38a39c16b31d1ace71bca8ef58d315b", "0x63ed5a272de2f6d968408b4acb9024f4cc208ebf",
    "0x6f6704e5a10332af6672e50b3d9754dc460dfa4d", "0x77ca7b50b6cd7e2f3fa008e24ab793fd56cb15f6",
    "0x492ea3bb0f3315521c31f273e565b868fc090f17", "0x0ff30d6de14a8224aa97b78aea5388d1c51c1f00",
    "0x9ea779f907f0b315b364b0cfc39a0fde5b02a416", "0xceaeb481747ca6c540a000c1f3641f8cef161fa7",
    "0xcc34673c6c40e791051898567a1222daf90be287", "0x579a80d909f346fbfb1189493f521d7f48d52238",
    "0xe308bd1ac5fda103967359b2712dd89deffb7973", "0x4cb31628079fb14e4bc3cd5e30c2f7489b00960c",
    "0xac1ecab32727358dba8962a0f3b261731aad9723", "0x4fd6ace747f06ece9c49699c7cabc62d02211f75",
    "0x440c59b325d2997a134c2c7c60a8c61611212bad", "0x4486a3d68fac6967006d7a517b889fd3f98c102b",
    "0x9c15b54878ba618f494b38f0ae7443db6af648ba", "0x27b137a85656544b1ccb5a0f2e561a5703c6a68f",
    "0x21c7fdb9ed8d291d79ffd82eb2c4356ec0d81241", "0x23b75c2f6791eef49c69684db4c6c1f93bf49a50",
    "0x1ca6abd14d30affe533b24d7a21bff4c2d5e1f3b", "0xb9637156d330c0d605a791f1c31ba5890582fe1c",
    "0x6131c42fa982e56929107413a9d526fd99405560", "0x1591fc0f688c81fbeb17f5426a162a7024d430c2",
    "0x542a9515200d14b68e934e9830d91645a980dd7a", "0xc4bbd073882dd2add2424cf47d35213405b01324",
    "0x782495b7b3355efb2833d56ecb34dc22ad7dfcc4", "0x58b95c9a9d5d26825e70a82b6adb139d3fd829eb",
    "0x3ba4d81db016dc2890c81f3acec2454bff5aada5", "0xb52042c8ca3f8aa246fa79c3feaa3d959347c0ab",
    "0xe4ae1efdfc53b73893af49113d8694a057b9c0d1", "0x3c02a7bc0391e86d91b7d144e61c2c01a25a79c5",
    "0x0737a6b837f97f46ebade41b9bc3e1c509c85c53", "0x97f43a37f595ab5dd318fb46e7a155eae057317a",
    "0x52c5317c848ba20c7504cb2c8052abd1fde29d03", "0x4863226780fe7c0356454236d3b1c8792785748d",
    "0x5d2b2e6fcbe3b11d26b525e085ff818dae332479", "0x5f9f3392e9f62f63b8eac0beb55541fc8627f42c",
    "0x057b56736d32b86616a10f619859c6cd6f59092a", "0x9aa008f65de0b923a2a4f02012ad034a5e2e2192",
    "0x304a554a310c7e546dfe434669c62820b7d83490", "0x914d1b8b43e92723e64fd0a06f5bdb8dd9b10c79",
    "0x4deb0033bb26bc534b197e61d19e0733e5679784", "0x07f5c1e1bc2c93e0402f23341973a0e043f7bf8a",
    "0x35a051a0010aba705c9008d7a7eff6fb88f6ea7b", "0x4fa802324e929786dbda3b8820dc7834e9134a2a",
    "0x9da397b9e80755301a3b32173283a91c0ef6c87e", "0x8d9edb3054ce5c5774a420ac37ebae0ac02343c6",
    "0x0101f3be8ebb4bbd39a2e3b9a3639d4259832fd9", "0x5dc28b15dffed94048d73806ce4b7a4612a1d48f",
    "0xbcf899e6c7d9d5a215ab1e3444c86806fa854c76", "0x12e626b0eebfe86a56d633b9864e389b45dcb260",
    "0xa2f1ccba9395d7fcb155bba8bc92db9bafaeade7", "0xec8e57756626fdc07c63ad2eafbd28d08e7b0ca5",
    "0xd164b088bd9108b60d0ca3751da4bceb207b0782", "0x6231b6d0d5e77fe001c2a460bd9584fee60d409b",
    "0x1cba23d343a983e9b5cfd19496b9a9701ada385f", "0xa82f360a8d3455c5c41366975bde739c37bfeb8a",
    "0x9fcd2deaff372a39cc679d5c5e4de7bafb0b1339", "0x005f5cee7a43331d5a3d3eec71305925a62f34b6",
    "0x0e0da70933f4c7849fc0d203f5d1d43b9ae4532d", "0xd131637d5275fd1a68a3200f4ad25c71a2a9522e",
    "0xbc07118b9ac290e4622f5e77a0853539789effbe", "0x47e7aa56d6bdf3f36be34619660de61275420af8",
    "0xacd87e28b0c9d1254e868b81cba4cc20d9a32225", "0xadf80daec7ba8dcf15392f1ac611fff65d94f880",
    "0x5524c55fb03cf21f549444ccbecb664d0acad706", "0x40b803a9abce16f50f36a77ba41180eb90023925",
    "0xfe24cdd8648121a43a7c86d289be4dd2951ed49f", "0x17802f43a0137c506ba92291391a8a8f207f487d",
    "0x253488078a4edf4d6f42f113d1e62836a942cf1a", "0x86af3e9626fce1957c82e88cbf04ddf3a2ed7915",
    "0xb136707642a4ea12fb4bae820f03d2562ebff487", "0xdbe9b615a3ae8709af8b93336ce9b477e4ac0940",
    "0xf14c14075d6c4ed84b86798af0956deef67365b5", "0xca544e5c4687d109611d0f8f928b53a25af72448",
    "0xaeeb8ff27288bdabc0fa5ebb731b6f409507516c", "0xcbb9d3703e651b0d496cdefb8b92c25aeb2171f7",
    "0x6d87578288b6cb5549d5076a207456a1f6a63dc0", "0xb2c6f0dfbb716ac562e2d85d6cb2f8d5ee87603e",
    "0xaccc230e8a6e5be9160b8cdf2864dd2a001c28b6", "0x2b3455ec7fedf16e646268bf88846bd7a2319bb2",
    "0x4613f3bca5c44ea06337a9e439fbc6d42e501d0a", "0xd343b217de44030afaa275f54d31a9317c7f441e",
    "0x84ef4b2357079cd7a7c69fd7a37cd0609a679106", "0xda2fef9e4a3230988ff17df2165440f37e8b1708",
    "0xf4c64518ea10f995918a454158c6b61407ea345c", "0x7602b46df5390e432ef1c307d4f2c9ff6d65cc97",
    "0xbb9bc244d798123fde783fcc1c72d3bb8c189413", "0x807640a13483f8ac783c557fcdf27be11ea4ac7a",
)
#: every address that makes a transaction "DAO-involved"
DAO_INVOLVED = frozenset(DAO_FORK_LIST) | {DAO_ADDRESSES["creator"], DAO_ADDRESSES["withdraw_dao"]}
#: the DAO was created in block 1 428 757 (tx 0xe9ebfecc…, sender 0x793ea969…); blocks
#: 1 428 000 - 1 428 756 are pre_dao.  Fork era = the ten "dao-hard-fork" extra-data blocks.
DAO_CREATION_BLOCK = 1_428_757
ERA_BOUNDS = {"pre_dao": (0, DAO_CREATION_BLOCK), "dao_active": (DAO_CREATION_BLOCK, 1_718_497),
              "drain_crisis": (1_718_497, 1_920_000), "fork": (1_920_000, 1_920_010),
              "post_fork": (1_920_010, 10 ** 9)}
ERAS = tuple(ERA_BOUNDS)
LABELS = ("distribution", "pooling", "many_to_many", "single", "none", "other")
FAN_DEFINITION = "net_effective"
VISIBLE_LEN = 512
SURFACE_FEATURES = ("n_tokens", "n_call", "n_value_tokens", "n_value_calls_nonzero")
#: surface features for masked probes: the ``[A_OOV]`` count is a shortcut masking can create
SURFACE_FEATURES_OOV = SURFACE_FEATURES + ("n_a_oov",)
LENGTH_RATIO = 1.25
MATCH_KINDS = ("gas", "length")
MASKS = ("none", "dao", "all_addresses")
MASK_TOKEN = "[A_OOV]"
#: covariates whose balance (SMD) is reported for matched probe sets when available
BALANCE_COVARIATES = ("n_tokens", "gas_used", "n_nodes", "n_value_calls", "max_depth", "n_state_writes")
_DAO_SET = DAO_INVOLVED
_DAO_TOKENS = frozenset(f"A:{a}" for a in DAO_INVOLVED)
_STALE_MSG = ("labels_df comes from an older serialisation (fan definition {!r}, current {!r}); re-run "
              "`python -m lsne.chain.cli serialize ... --overwrite`")


def era_of(block: int) -> str:
    """Era name for a block number (half-open bounds ``[lo, hi)``)."""
    b = int(block)
    for name, (lo, hi) in ERA_BOUNDS.items():
        if lo <= b < hi:
            return name
    raise ValueError(f"block {block} outside every era")


def touched_addresses(rec: TxRecord) -> set:
    """Every address the transaction touches: call endpoints, created contracts, log
    emitters, written / read storage owners, transfer endpoints and ERC-20 token contracts
    (not addresses that only appear as call arguments, see :func:`argument_addresses`)."""
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
    out.update(a for a in (rec.frm, rec.to) if a)
    return {str(a).lower() for a in out}


def argument_addresses(rec: TxRecord) -> set:
    """Addresses among the kept 32-byte call arguments and the log topics 1-3 of the
    transaction: left-padded 20-byte words that are >= 2**128 or exactly an address of
    :data:`DAO_INVOLVED`."""
    out = set()
    words = [w for n in iter_nodes(rec.root, "dfs") for w in n.args]
    words += [lg.get(k) for n in iter_nodes(rec.root, "dfs") for lg in n.logs for k in ("topic1", "topic2", "topic3")]
    for w in words:
        if not w:
            continue
        h = w[2:] if str(w).startswith("0x") else str(w)
        if len(h) != 64 or h[:24] != "0" * 24:
            continue
        a = "0x" + h[24:].lower()
        if int(h, 16) >= 2 ** 128 or a in DAO_INVOLVED:
            out.add(a)
    return out


def _flows(rec: TxRecord, eff: list) -> dict:
    net = net_flows(eff)
    fo, fi = flow_counts(net)
    return dict(fo=fo, fi=fi, net=net)


def sharing_features(rec: TxRecord, visible_len: Optional[int] = VISIBLE_LEN,
                     serialize_kw: Optional[dict] = None) -> dict:
    """Per-transaction sharing features (addresses lower-cased; definitions in the module doc).

    Spec keys: fan_out_eth, fan_in_eth (net flows over effective transfers), fan_out_erc20,
    fan_in_erc20 (net per token), n_value_calls (effective value-carrying nodes),
    n_state_writes, n_distinct_slots_written, dao_involved (EIP-779 list, creator,
    WithdrawDAO), withdraw_dao_involved, max_depth, n_nodes, era.  Gross / structural keys:
    fan_out_eth_gross, fan_in_eth_gross (distinct receivers / senders of the effective
    transfers), n_call_targets, n_call_sources (all rows).  Transfer accounting:
    n_native_transfers, n_native_transfers_pos (value > 0, any status),
    n_transfers_effective, n_transfers_reverted, n_transfers_phantom, total_eth (gross,
    effective), net_eth.  Visible window (``visible_len`` tokens of :func:`serialize` with
    ``serialize_kw``; None skips them): fan_out_eth_visible, fan_in_eth_visible,
    n_value_calls_visible, visible_len.  Extra: dao_mentioned (a DAO address only among
    call arguments / log topics), n_erc20_transfers, has_value_transfer, fan_definition,
    tx_hash, block_number, gas_used, success.
    """
    nat = rec.native_transfers
    erc = rec.erc20_transfers
    lower = lambda a: str(a).lower() if a else a  # noqa: E731
    eff = effective_transfers(rec)
    pos = [t for t in nat if t.get("value")]
    erc_pos = [t for t in erc if t.get("value")]
    flows = _flows(rec, eff)
    erc_net = net_flows(erc_pos, key=lambda t: lower(t.get("erc20")))
    efo, efi = flow_counts(erc_net)
    nodes = list(iter_nodes(rec.root, "dfs"))
    failed = failed_nodes(rec.root)
    eff_nodes = [n for n in nodes if n.value > 0 and node_is_effective(n, failed, rec.success)]
    writes = [w for n in nodes for w in n.state_writes]
    touched = touched_addresses(rec)
    dao_involved = bool(touched & _DAO_SET)
    out = dict(
        fan_out_eth=flows["fo"],
        fan_in_eth=flows["fi"],
        fan_out_erc20=efo,
        fan_in_erc20=efi,
        n_value_calls=len(eff_nodes),
        n_state_writes=len(writes),
        n_distinct_slots_written=len({(lower(w.get("address")), lower(w.get("slot"))) for w in writes}),
        dao_involved=dao_involved,
        withdraw_dao_involved=DAO_ADDRESSES["withdraw_dao"] in touched,
        dao_mentioned=bool((not dao_involved) and (argument_addresses(rec) & _DAO_SET)),
        max_depth=int(rec.max_depth),
        n_nodes=int(rec.n_nodes),
        era=era_of(rec.block_number),
        fan_out_eth_gross=len({lower(t.get("to_address")) for t in eff}),
        fan_in_eth_gross=len({lower(t.get("from_address")) for t in eff}),
        n_call_targets=len({lower(t.get("to_address")) for t in nat}),
        n_call_sources=len({lower(t.get("from_address")) for t in nat}),
        n_native_transfers=len(nat),
        n_native_transfers_pos=len(pos),
        n_transfers_effective=len(eff),
        n_transfers_reverted=sum(1 for t in pos if t.get("reverted") and not t.get("phantom")),
        n_transfers_phantom=sum(1 for t in pos if t.get("phantom")),
        n_erc20_transfers=len(erc),
        total_eth=sum(int(t["value"]) for t in eff) / 1e18,
        net_eth=sum(v for v in flows["net"].values() if v > 0) / 1e18,
        has_value_transfer=bool(flows["fo"] or flows["fi"] or efo or efi),
        fan_definition=FAN_DEFINITION,
        tx_hash=rec.tx_hash,
        block_number=int(rec.block_number),
        gas_used=int(rec.gas_used),
        success=bool(rec.success),
    )
    if visible_len is not None:
        vis = visible_nodes(rec, int(visible_len), **dict(serialize_kw or {}))
        vis_ii = {n.internal_index for n in vis if n.internal_index is not None}
        vflows = _flows(rec, effective_transfers(rec, vis_ii))
        vis_ids = {id(n) for n in vis}
        out.update(fan_out_eth_visible=vflows["fo"], fan_in_eth_visible=vflows["fi"],
                   n_value_calls_visible=sum(1 for n in eff_nodes if id(n) in vis_ids),
                   visible_len=int(visible_len))
    return out


def sharing_label(feat: dict, k: int = 5, structural: bool = False) -> str:
    """"distribution" (one source, fan_out >= k), "pooling" (one recipient, fan_in >= k),
    "many_to_many" (fan_out >= 2 and fan_in >= 2), "single" (fan_out == fan_in == 1),
    "none" (no value transfer), else "other".  Counts from the net ETH flows of the
    effective transfers (``structural=True``: from all native rows, ``n_call_targets`` /
    ``n_call_sources``); without a net ETH flow the net ERC-20 counts."""
    has_value = feat.get("has_value_transfer")
    if structural:
        fo, fi = feat.get("n_call_targets", 0), feat.get("n_call_sources", 0)
    else:
        fo, fi = feat.get("fan_out_eth", 0), feat.get("fan_in_eth", 0)
    if fo == 0 and fi == 0:
        fo, fi = feat.get("fan_out_erc20", 0), feat.get("fan_in_erc20", 0)
    if has_value is None:
        has_value = (fo + fi) > 0
    if not has_value or (fo == 0 and fi == 0):
        return "none"
    k = int(k)
    if fi == 1 and fo >= k:
        return "distribution"
    if fo == 1 and fi >= k:
        return "pooling"
    if fo >= 2 and fi >= 2:
        return "many_to_many"
    if fo == 1 and fi == 1:
        return "single"
    return "other"


def labels_frame(records: Sequence[TxRecord], k: int = 5, structural: bool = False) -> pd.DataFrame:
    """One row per record: :func:`sharing_features` plus ``label`` (input of :func:`probe_sets`)."""
    rows = []
    for r in records:
        f = sharing_features(r)
        f["label"] = sharing_label(f, k=k, structural=structural)
        rows.append(f)
    return pd.DataFrame(rows)


def _gas_deciles(gas: pd.Series, bins: int = 10) -> pd.Series:
    """Quantile bin 0..bins-1 by rank (ties broken by position), robust to repeated values."""
    n = len(gas)
    bins = max(1, int(bins))
    if n == 0:
        return pd.Series([], dtype=int, index=gas.index)
    ranks = gas.rank(method="first").to_numpy() - 1
    return pd.Series(np.minimum((ranks * bins // n).astype(int), bins - 1), index=gas.index)


def standardized_mean_difference(a, b) -> float:
    """``(mean(a) - mean(b)) / sqrt((var(a) + var(b)) / 2)`` (sample variances); NaN when
    either side has fewer than 2 values, 0 when both sides are constant and equal."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    diff = float(a.mean() - b.mean())
    sd = float(np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0))
    if sd == 0.0:
        return 0.0 if diff == 0.0 else float(np.sign(diff) * np.inf)
    return diff / sd


def _match_pairs(pool: pd.DataFrame, gas_bins: int, rng: np.random.Generator) -> list:
    pool = pool.copy()
    pool["gas_decile"] = _gas_deciles(pool["gas_used"].astype(float), gas_bins)
    pairs = []
    for (era, dec), grp in pool.groupby(["era", "gas_decile"], sort=True):
        p = grp[grp["_pos"]].index.to_numpy()
        q = grp[~grp["_pos"]].index.to_numpy()
        m = min(len(p), len(q))
        if m == 0:
            continue
        p = rng.permutation(p)[:m]
        q = rng.permutation(q)[:m]
        # within the stratum pair by gas rank (closest matching for the sampled members)
        p = sorted(p, key=lambda i: (float(pool.at[i, "gas_used"]), int(i)))
        q = sorted(q, key=lambda i: (float(pool.at[i, "gas_used"]), int(i)))
        pairs += [(era, int(dec), int(a), int(b)) for a, b in zip(p, q)]
    return pairs


def _match_pairs_length(pool: pd.DataFrame, ratio: float, rng: np.random.Generator, length_col: str) -> tuple:
    """1:1 nearest-length matching within era with a caliper ``|log n_pos - log n_neg| <= log ratio``.

    Negatives of equal length form one bucket whose members are drawn in a seeded random
    order (many simple transactions share a token count); positives are processed in order
    of increasing number of admissible negatives (hardest first, ties in seeded random
    order), each takes a member of the nearest non-empty length bucket inside the caliper.
    Returns ``(pairs, n_unmatched)`` with pairs ``(era, -1, pos_index, neg_index)``."""
    caliper = float(np.log(float(ratio))) + 1e-12
    pairs, n_unmatched = [], 0
    for era, grp in pool.groupby("era", sort=True):
        P = grp[grp["_pos"]]
        Q = grp[~grp["_pos"]]
        if P.empty:
            continue
        if Q.empty:
            n_unmatched += len(P)
            continue
        qlen = np.maximum(Q[length_col].to_numpy(float), 1.0)
        uniq, inv = np.unique(qlen, return_inverse=True)
        log_u = np.log(uniq)
        qidx = Q.index.to_numpy()
        counts = np.bincount(inv, minlength=len(uniq))
        cum = np.concatenate([[0], np.cumsum(counts)])
        grouped = qidx[np.argsort(inv, kind="stable")]
        buckets = [list(rng.permutation(grouped[cum[j]:cum[j + 1]])) for j in range(len(uniq))]
        plog = np.log(np.maximum(P[length_col].to_numpy(float), 1.0))
        lo = np.searchsorted(log_u, plog - caliper, side="left")
        hi = np.searchsorted(log_u, plog + caliper, side="right")
        n_cand = cum[hi] - cum[lo]
        order = np.lexsort((rng.permutation(len(P)), n_cand))      # fewest candidates first, random ties
        pidx = P.index.to_numpy()
        for i in order:
            js = np.arange(lo[i], hi[i])
            best = None
            for j in js[np.argsort(np.abs(log_u[js] - plog[i]), kind="stable")]:
                if buckets[j]:
                    best = j
                    break
            if best is None:
                n_unmatched += 1
                continue
            pairs.append((era, -1, int(pidx[i]), int(buckets[best].pop())))
    return pairs, n_unmatched


def balance_table(pos_df: pd.DataFrame, neg_df: pd.DataFrame,
                  covariates: Sequence[str] = BALANCE_COVARIATES) -> dict:
    """Standardized mean differences (positives minus negatives) of matched sets for every
    available covariate, plus the log-scale SMD for the heavy-tailed counts
    (``log_<cov>`` = SMD of ``log(max(x, 1))``) and the means of both sides."""
    out = {}
    for c in covariates:
        if c not in pos_df.columns or c not in neg_df.columns:
            continue
        a = pos_df[c].to_numpy(float)
        b = neg_df[c].to_numpy(float)
        out[c] = dict(smd=standardized_mean_difference(a, b),
                      log_smd=standardized_mean_difference(np.log(np.maximum(a, 1.0)), np.log(np.maximum(b, 1.0))),
                      mean_pos=float(np.mean(a)) if len(a) else float("nan"),
                      mean_neg=float(np.mean(b)) if len(b) else float("nan"),
                      median_pos=float(np.median(a)) if len(a) else float("nan"),
                      median_neg=float(np.median(b)) if len(b) else float("nan"))
    return out


def probe_sets(records: Sequence[TxRecord], labels_df: Optional[pd.DataFrame], kind: str, n: int,
               seed: int = 0, k: int = 5, eras: Optional[Sequence[str]] = None, gas_bins: int = 10,
               min_pairs: Optional[int] = None, match: str = "gas", length_ratio: float = LENGTH_RATIO,
               length_col: str = "n_tokens", length_cap: Optional[int] = None, visible: Optional[bool] = None):
    """Matched positive / negative record sets for a sharing probe.

    * ``kind="fanout"``: positives ``fan_out_eth >= k``, negatives ``fan_out_eth == 1``
      (net flows of the effective transfers); with ``visible`` (default: whenever the
      ``fan_out_eth_visible`` column exists) the same must hold on the visible window
      (``fan_out_eth_visible >= k`` resp. ``== 1``), so the model's input contains the
      fan-out the label is about;
    * ``kind="dao"``: positives ``dao_involved``, negatives not and without a DAO address
      among their call arguments (``dao_mentioned``, ambiguous, excluded), restricted to
      ``eras`` (default ``("drain_crisis", "post_fork")``).

    ``match="gas"`` (default, the spec): pairs are matched 1:1 within (era, gas_used decile)
    strata (deciles over the candidate pool; inside a stratum the sampled members are paired
    by gas rank).  When that yields fewer than ``min_pairs`` pairs (default ``min(n, 20)``)
    the matching falls back to era only (``gas_bins=1`` requests era-only matching directly).

    ``match="length"``: 1:1 within era on the log ITR token count (column ``length_col`` of
    ``labels_df``, computed by serialising the records when absent; ``length_cap`` matches
    on ``min(n_tokens, cap)``, the length the model sees), caliper
    ``|log n_pos - log n_neg| <= log(length_ratio)``, nearest length first, without
    replacement (:func:`_match_pairs_length`); positives without an admissible negative stay
    unmatched (counted in ``attrs["n_pos_unmatched"]``) -- there is no silent fallback.

    At most ``n`` pairs, sampled with ``seed``.  Returns ``(positive, negative,
    match_table)``; ``match_table`` columns: pair, era, gas_decile (-1 for length matching),
    pos_tx, neg_tx, pos_gas, neg_gas, pos_fan_out_eth, neg_fan_out_eth, pos_dao_involved,
    neg_dao_involved, pos/neg_fan_out_eth_visible (when available: the recipients the model's
    window shows), matching ("era+gas_decile", "era_only" or "era+log_length"), gas_smd
    and log_gas_smd (standardized mean difference of gas_used resp. log gas_used, positives
    minus negatives, over all matched pairs; the same value in every row), and when token
    counts are available pos_n_tokens, neg_n_tokens, length_smd (SMD of log token count) and
    n_tokens_smd.  ``match_table.attrs`` repeats the summary values and holds ``balance``
    (:func:`balance_table` over the available covariates), the pool sizes, the matching
    parameters, ``visible`` and ``n_excluded`` (fan-out: candidates dropped because the
    visible window does not show the label; dao: ambiguous negatives).
    """
    if match not in MATCH_KINDS:
        raise ValueError(f"match must be one of {MATCH_KINDS}, got {match!r}")
    records = list(records)
    df = labels_frame(records, k=k) if labels_df is None else labels_df.copy()
    if len(df) != len(records):
        raise ValueError("labels_df must have one row per record (same order)")
    if match == "length" and length_col not in df.columns:
        if not all(isinstance(r, TxRecord) for r in records):
            raise ValueError(f"length matching needs a {length_col!r} column in labels_df (or TxRecord records)")
        df[length_col] = [len(serialize(r)) for r in records]
    if "n_call_targets" not in df.columns and "fan_out_eth_pos" in df.columns:
        raise ValueError("labels_df comes from an older serialisation whose fan_out_eth counted zero-value call "
                         "rows; re-run `python -m lsne.chain.cli serialize ... --overwrite`")
    if "fan_definition" in df.columns:
        defs = set(df["fan_definition"].dropna().astype(str))
        if defs - {FAN_DEFINITION}:
            raise ValueError(_STALE_MSG.format(sorted(defs - {FAN_DEFINITION}), FAN_DEFINITION))
    if "tx_hash" in df.columns:
        rec_hashes = [r.tx_hash for r in records]
        if list(df["tx_hash"]) != rec_hashes:
            pos_of = {h: i for i, h in enumerate(rec_hashes)}
            if set(df["tx_hash"]) != set(rec_hashes):
                raise ValueError("labels_df tx_hash values do not match the records")
            df = df.assign(_rec=[pos_of[h] for h in df["tx_hash"]])
    if "_rec" not in df.columns:
        df = df.assign(_rec=np.arange(len(df)))
    df = df.reset_index(drop=True)
    has_vis = "fan_out_eth_visible" in df.columns
    if visible and not has_vis:
        raise ValueError("visible=True needs the fan_out_eth_visible column (re-serialise the shards)")
    use_vis = bool(has_vis if visible is None else visible)
    n_excluded = 0
    if kind == "fanout":
        is_pos = df["fan_out_eth"] >= int(k)
        is_neg = df["fan_out_eth"] == 1
        if use_vis:
            vis = df["fan_out_eth_visible"]
            n_excluded = int((is_pos & ~(vis >= int(k))).sum() + (is_neg & ~(vis == 1)).sum())
            is_pos, is_neg = is_pos & (vis >= int(k)), is_neg & (vis == 1)
        if eras is not None:
            keep = df["era"].isin(list(eras))
            is_pos, is_neg = is_pos & keep, is_neg & keep
    elif kind == "dao":
        eras = ("drain_crisis", "post_fork") if eras is None else tuple(eras)
        keep = df["era"].isin(list(eras))
        is_pos = df["dao_involved"].astype(bool) & keep
        is_neg = ~df["dao_involved"].astype(bool) & keep
        if "dao_mentioned" in df.columns:
            amb = df["dao_mentioned"].fillna(False).astype(bool)
            n_excluded = int((is_neg & amb).sum())
            is_neg = is_neg & ~amb
    else:
        raise ValueError(f"kind must be 'fanout' or 'dao', got {kind!r}")
    pool = df[is_pos | is_neg].copy()
    pool["_pos"] = is_pos[pool.index].to_numpy()
    min_pairs = min(int(n), 20) if min_pairs is None else int(min_pairs)
    rng = np.random.default_rng(int(seed))
    gas_bins = max(1, int(gas_bins))
    n_unmatched = None
    if match == "length":
        matching = "era+log_length"
        if length_cap is not None:
            pool["_len"] = np.minimum(pool[length_col].to_numpy(float), float(length_cap))
        else:
            pool["_len"] = pool[length_col].to_numpy(float)
        pairs, n_unmatched = _match_pairs_length(pool, float(length_ratio), rng, "_len")
    else:
        matching = "era+gas_decile" if gas_bins > 1 else "era_only"
        pairs = _match_pairs(pool, gas_bins, rng)
        if gas_bins > 1 and len(pairs) < min_pairs:
            era_pairs = _match_pairs(pool, 1, np.random.default_rng(int(seed)))
            if len(era_pairs) > len(pairs):
                pairs, matching = era_pairs, "era_only"
    n_matched_total = len(pairs)
    if len(pairs) > int(n):
        idx = np.sort(rng.choice(len(pairs), size=int(n), replace=False))
        pairs = [pairs[i] for i in idx]
    has_len = length_col in df.columns
    rows, pos_recs, neg_recs = [], [], []
    for i, (era, dec, a, b) in enumerate(pairs):
        ra, rb = df.loc[a], df.loc[b]
        pos_recs.append(records[int(ra["_rec"])])
        neg_recs.append(records[int(rb["_rec"])])
        row = dict(pair=i, era=era, gas_decile=dec, pos_tx=ra.get("tx_hash"), neg_tx=rb.get("tx_hash"),
                   pos_gas=int(ra["gas_used"]), neg_gas=int(rb["gas_used"]),
                   pos_fan_out_eth=int(ra["fan_out_eth"]), neg_fan_out_eth=int(rb["fan_out_eth"]),
                   pos_dao_involved=bool(ra["dao_involved"]), neg_dao_involved=bool(rb["dao_involved"]))
        if has_vis:
            row.update(pos_fan_out_eth_visible=int(ra["fan_out_eth_visible"]),
                       neg_fan_out_eth_visible=int(rb["fan_out_eth_visible"]))
        if has_len:
            row.update(pos_n_tokens=int(ra[length_col]), neg_n_tokens=int(rb[length_col]))
        rows.append(row)
    cols = ["pair", "era", "gas_decile", "pos_tx", "neg_tx", "pos_gas", "neg_gas", "pos_fan_out_eth",
            "neg_fan_out_eth", "pos_dao_involved", "neg_dao_involved"]
    vis_cols = ["pos_fan_out_eth_visible", "neg_fan_out_eth_visible"] if has_vis else []
    len_cols = ["pos_n_tokens", "neg_n_tokens"] if has_len else []
    mt = pd.DataFrame(rows, columns=cols + vis_cols + len_cols)
    pg, ng = mt["pos_gas"].to_numpy(float), mt["neg_gas"].to_numpy(float)
    smd = standardized_mean_difference(pg, ng)
    log_smd = standardized_mean_difference(np.log(np.maximum(pg, 1.0)), np.log(np.maximum(ng, 1.0)))
    mt = mt.assign(matching=matching, gas_smd=smd, log_gas_smd=log_smd)
    summary = dict(matching=matching, gas_smd=smd, log_gas_smd=log_smd)
    if has_len:
        pl, nl = mt["pos_n_tokens"].to_numpy(float), mt["neg_n_tokens"].to_numpy(float)
        summary.update(length_smd=standardized_mean_difference(np.log(np.maximum(pl, 1.0)), np.log(np.maximum(nl, 1.0))),
                       n_tokens_smd=standardized_mean_difference(pl, nl))
        mt = mt.assign(length_smd=summary["length_smd"], n_tokens_smd=summary["n_tokens_smd"])
        if len(mt):
            lr = np.abs(np.log(np.maximum(pl, 1.0)) - np.log(np.maximum(nl, 1.0)))
            summary.update(length_abs_log_ratio_median=float(np.median(lr)), length_ratio_max=float(np.exp(lr.max())))
    pos_rows = df.loc[[p[2] for p in pairs]] if pairs else df.iloc[:0]
    neg_rows = df.loc[[p[3] for p in pairs]] if pairs else df.iloc[:0]
    mt.attrs.update(summary)
    mt.attrs.update(n_pairs=len(mt), min_pairs=min_pairs, gas_bins=gas_bins, match=match,
                    length_ratio=float(length_ratio) if match == "length" else None,
                    length_cap=None if length_cap is None else int(length_cap),
                    pool_pos=int(pool["_pos"].sum()), pool_neg=int((~pool["_pos"]).sum()),
                    n_matched_total=int(n_matched_total), n_pos_unmatched=n_unmatched,
                    visible=use_vis if kind == "fanout" else None, n_excluded=n_excluded,
                    eras=mt["era"].value_counts().to_dict(),
                    balance=balance_table(pos_rows, neg_rows))
    return pos_recs, neg_recs, mt


# ======================================================================================
# identity masking
# ======================================================================================
def mask_tokens(tokens: Sequence[str], mask: Union[str, Iterable[str], None] = "dao",
                replacement: str = MASK_TOKEN) -> list:
    """Replace identity tokens of an ITR token list (raw or vocabulary-mapped) by ``[A_OOV]``.

    * ``"none"`` / ``None``: unchanged copy;
    * ``"dao"``: every DAO identity of :data:`DAO_INVOLVED` (EIP-779 list: DAO,
      extraBalance, child DAOs and their extraBalances; creator, WithdrawDAO) wherever it
      occurs (call endpoints, arguments, state writes / reads, log emitters);
    * ``"all_addresses"``: every address token ``A:0x…`` (``A:none`` -- no address -- stays);
    * an iterable of addresses (``0x…``, any case) or ``A:``-tokens: exactly those.

    Masking is applied to positives and negatives alike, before encoding."""
    toks = [str(t) for t in tokens]
    if mask is None or (isinstance(mask, str) and mask == "none"):
        return toks
    if isinstance(mask, str) and mask == "dao":
        targets = _DAO_TOKENS
    elif isinstance(mask, str) and mask == "all_addresses":
        return [replacement if (t.startswith("A:") and t != "A:none") else t for t in toks]
    elif isinstance(mask, str):
        raise ValueError(f"mask must be one of {MASKS} or an iterable of addresses, got {mask!r}")
    else:
        targets = frozenset("A:" + (str(a)[2:] if str(a).startswith("A:") else str(a)).lower() for a in mask)
    return [replacement if t in targets else t for t in toks]


# ======================================================================================
# surface-feature baseline
# ======================================================================================
def surface_features(tokens: Sequence[str], max_len: Optional[int] = None) -> dict:
    """Trivial length / count statistics of an ITR token list (raw or vocabulary-mapped;
    ``max_len``: only the first ``max_len`` tokens, i.e. what the model sees): ``n_tokens``,
    ``n_call`` (``[CALL]``), ``n_value_tokens`` (every ``V:`` token: calls, state writes and
    reads), ``n_value_calls_nonzero`` (``[CALL]`` entries whose ``V:`` token is not ``V:N0``)
    and ``n_a_oov`` (``[A_OOV]`` tokens; 0 for raw tokens unless masked)."""
    toks = list(tokens)[: int(max_len)] if max_len is not None else list(tokens)
    n_call = n_val = n_nz = n_oov = 0
    in_call = False
    for t in toks:
        if t == MASK_TOKEN:
            n_oov += 1
        elif t == "[CALL]":
            n_call += 1
            in_call = True
        elif t == "[END]" or t == "[START]":
            in_call = False
        elif t.startswith("V:"):
            n_val += 1
            if in_call and t != "V:N0":
                n_nz += 1
            in_call = False
    return dict(n_tokens=len(toks), n_call=n_call, n_value_tokens=n_val, n_value_calls_nonzero=n_nz,
                n_a_oov=n_oov)


def _auc(p: np.ndarray, q: np.ndarray) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie)."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float((p[:, None] > q[None, :]).mean() + 0.5 * (p[:, None] == q[None, :]).mean())


def surface_baseline_auc(pos_tokens: Sequence[Sequence[str]], neg_tokens: Sequence[Sequence[str]], seed: int = 0,
                         max_len: Optional[int] = None,
                         features: Sequence[str] = SURFACE_FEATURES) -> dict:
    """Split-half AUC of the surface features of matched positive / negative token lists.

    The split is exactly the one of :func:`lsne.chain.source.holdout_auc` with the same
    ``seed`` (``default_rng(seed).permutation(min(n_pos, n_neg))``, two halves, fit on one,
    evaluate on the other, mean over both directions).  Per feature the orientation (sign)
    is fitted on the training half; ``"combined"`` is the difference of class means of the
    z-scored features (training-half statistics), the same estimator as the sharing
    direction.  Returns ``{feature: auc, ..., "combined": auc, "best_single": name}``
    (NaN with fewer than 4 pairs)."""
    n = min(len(pos_tokens), len(neg_tokens))
    feats = list(features)
    out = {f: float("nan") for f in feats}
    out.update(combined=float("nan"), best_single=None, n_pairs=int(n))
    if n < 4:
        return out
    F_pos = np.array([[surface_features(t, max_len)[f] for f in feats] for t in pos_tokens[:n]], dtype=float)
    F_neg = np.array([[surface_features(t, max_len)[f] for f in feats] for t in neg_tokens[:n]], dtype=float)
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    halves = (perm[: n // 2], perm[n // 2:])
    per = {f: [] for f in feats}
    comb = []
    for a, b in (halves, halves[::-1]):
        mp, mn = F_pos[a].mean(0), F_neg[a].mean(0)
        for j, f in enumerate(feats):
            s = 1.0 if mp[j] >= mn[j] else -1.0
            per[f].append(_auc(s * F_pos[b, j], s * F_neg[b, j]))
        train = np.concatenate([F_pos[a], F_neg[a]])
        mu, sd = train.mean(0), train.std(0)
        sd[sd == 0] = 1.0
        w = (mp - mn) / sd
        if np.linalg.norm(w) > 0:
            comb.append(_auc(((F_pos[b] - mu) / sd) @ w, ((F_neg[b] - mu) / sd) @ w))
        else:
            comb.append(0.5)
    for f in feats:
        out[f] = float(np.mean(per[f]))
    out["combined"] = float(np.mean(comb))
    out["best_single"] = max(feats, key=lambda f: out[f])
    return out
