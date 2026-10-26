"""Cut the test fixture from full Xatu chunks (reproducibility script, not a test).

Usage::

    python -I tests/fixtures/xatu/cut_fixture.py <dir with canonical_execution_<table>_<start>.parquet> [out_dir]

Keeps the original Arrow schemas (pyarrow ``Table.filter``) and selects the rows of the
transactions listed below (hashes compared case-insensitively, since
``native_transfers`` stores upper-case hex), plus the block-level reward row of the fork
block 1 920 000 (``traces`` and ``native_transfers``) and the ``block`` rows of the
selected 1 920 000-chunk blocks.  Data: ethPandaOps Xatu, CC BY 4.0.
"""
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

DRAIN = "0x0ec3f2488a93839524add10ea229e773f6bc891b4eb4794c3337d4495263790b"
SELECTION = {
    1718000: {
        "tables": ("traces", "native_transfers", "logs", "storage_diffs"),
        "txs": [
            DRAIN,                                                                  # DAO drain
            "0x61db76fc3fe109ad548d91d321daf76dc2ef2f683dc3c4006377d1da8629e274",  # plain ETH transfer
            "0x877db25210ca8be928280112b8b3b4a3afced1d35eb8cd795ac730c7e89d88a3",  # plain ETH transfer
            "0x6cfc521f930715518e1aba42d0c414d099bec56bd719649c263c0d2b68be2da2",  # contract call, log + storage
            "0x124532e6294dc3e2ad55d0bdb6039ad562d1874f36cbc4610e9c0dab43e79af0",  # 19 nodes, fan-out 7
        ],
    },
    1920000: {
        "tables": ("traces", "transaction", "native_transfers", "erc20_transfers", "logs", "storage_diffs",
                   "storage_reads", "block"),
        "txs": [
            "0x6f75b64d9364b71b43cde81a889f95df72e6be004b28477f9083ed0ee471a7f9",  # plain transfer, fork block
            "0xf088dda9ba4a859d73a2ab760be508a6859b493685ff58ffd7f7e45b316caf50",  # WithdrawDAO withdraw(), refund
            "0x2fffeca979da48f1cf416b136ebf966c2319e55470700e9271df356c6e9b71c6",  # WithdrawDAO withdraw(), fails
            "0x21d7e78b330072bffe43e26d8d5571bc45691b7d14a454f4d7871c61a58b4414",  # out of gas
            "0x3de72d0e7a6d07ca735b85c0d099f351b8da3d52be0f200417dd9aa3a350e9c5",  # contract creation
            "0x9ab5dcb1ee22a97e6e81987e199939e36576b59ca1d009d79294c66d599b5891",  # ERC-20 transfer
            "0x29c0938e0933990d9965c64130c89189501b6a531d5a53ad4d907c9c698052f6",  # call_code
            "0x798af2597c979e2a218d7a7b8067fa48913c88cfa82e7bafa2dc3120317310e7",  # suicide
            "0xa1fafc6fd9351e7921147d8ddbac1fe169d34cd7baadb2f5cbee69a5f527cd2b",  # 13 nodes, fan-out 5
        ],
        "reward_block": 1920000,
    },
}


def _hash_of(v: bytes) -> str:
    return v.decode("latin-1").lower()


def _is_block_level(h: str) -> bool:
    return all(c in "0\x00" for c in h[2:])


def cut(src: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    for start, sel in SELECTION.items():
        txs = set(sel["txs"])
        blocks = set()
        for table in sel["tables"]:
            if table == "block":
                continue
            tbl = pq.read_table(os.path.join(src, f"canonical_execution_{table}_{start}.parquet"))
            hashes = [_hash_of(h) for h in tbl.column("transaction_hash").to_pylist()]
            bn = tbl.column("block_number").to_pylist()
            mask = [h in txs for h in hashes]
            rb = sel.get("reward_block")
            if rb is not None and table in ("traces", "native_transfers"):
                rows = tbl.to_pylist()
                for i, h in enumerate(hashes):
                    if not (_is_block_level(h) and bn[i] == rb):
                        continue
                    r = rows[i]
                    if table == "traces" and r["action_reward_type"] == b"reward":
                        mask[i] = True
                    if table == "native_transfers" and r["from_address"] == b"0x" + b"0" * 40:
                        mask[i] = True
                rb_rows = [i for i, m in enumerate(mask) if m and _is_block_level(hashes[i])]
                if table == "native_transfers" and len(rb_rows) > 1:   # keep the block reward only
                    trace_reward = 5 * 10 ** 18
                    keep = [i for i in rb_rows if int.from_bytes(rows[i]["value"], "little") == trace_reward][:1]
                    for i in rb_rows:
                        mask[i] = i in keep
            sub = tbl.filter(pa.array(mask))
            blocks.update(b for b, m in zip(bn, mask) if m)
            pq.write_table(sub, os.path.join(out, f"canonical_execution_{table}_{start}.parquet"),
                           compression="zstd")
            print(table, start, sub.num_rows)
        if "block" in sel["tables"]:
            tbl = pq.read_table(os.path.join(src, f"canonical_execution_block_{start}.parquet"))
            mask = [b in blocks for b in tbl.column("block_number").to_pylist()]
            sub = tbl.filter(pa.array(mask))
            pq.write_table(sub, os.path.join(out, f"canonical_execution_block_{start}.parquet"),
                           compression="zstd")
            print("block", start, sub.num_rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    cut(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__)))
