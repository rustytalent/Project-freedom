"""Hash chain for the outcome log — Stream J integrity layer.

Every partition file the outcome log writes gets a SHA-256 sidecar
that commits to its bytes. Each sidecar also records the hash of the
PREVIOUS partition's sidecar, creating a Merkle-style chain. A
customer auditing yesterday's published probability can:

  1. Read the partition file off disk (or fetch via API).
  2. Hash its bytes with SHA-256.
  3. Compare against the sidecar value — if they don't match, the
     partition was edited after we wrote it.
  4. Walk the chain backward — if any partition's "previous_hash"
     doesn't match the next partition's sidecar, the chain was
     forked or rewritten.

The whole chain is anchored by publishing the CURRENT HEAD hash to
a public location (a static URL on the marketing site, a tweet,
an OP_RETURN — operationally choose one). Once anchored, the entire
history is committed to: tampering with any historical partition
invalidates every later sidecar.

Sidecar file format (JSON, sibling to the parquet file):

    predictions/trading_date_ist=2026-06-09/predictions.parquet
    predictions/trading_date_ist=2026-06-09/predictions.parquet.sha256.json

    {
      "file_sha256": "abc123...",          # SHA-256 of the parquet bytes
      "previous_chain_hash": "def456...",  # SHA-256 of the previous sidecar JSON
      "chain_hash": "789xyz...",           # SHA-256 of (file_sha256 || previous_chain_hash)
      "table": "predictions",
      "partition_key": "trading_date_ist=2026-06-09",
      "written_at_utc": "2026-06-09T03:01:23Z",
      "schema_version": "1.0"
    }

The "chain_hash" is what later partitions point at. The latest such
hash across the whole log directory is the chain HEAD that gets
published externally.

Design constraints:
  * Append-only. Recomputing a historical sidecar requires also
    recomputing every later one — the chain refuses to silently
    fork.
  * Independent of the writer. The chain layer reads what's already
    on disk and lays down sidecars; it never modifies the parquet
    files themselves.
  * Verifiable in a few lines of Python by any third party (see
    `verify_chain`).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


SCHEMA_VERSION = "1.0"
SIDECAR_SUFFIX = ".sha256.json"
EMPTY_HASH = "0" * 64  # genesis "previous" for the first partition


# ---------------------------------------------------------------------------
# Hashing primitives
# ---------------------------------------------------------------------------

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            h.update(blk)
    return h.hexdigest()


def link_hash(file_sha: str, previous_chain_hash: str) -> str:
    """The chain hash for one sidecar = SHA-256 of (file_sha || prev_chain_hash).

    Order matters; never reverse. This is what later partitions point
    at to extend the chain.
    """
    return sha256_bytes(f"{file_sha}{previous_chain_hash}".encode("utf-8"))


# ---------------------------------------------------------------------------
# Sidecar record
# ---------------------------------------------------------------------------

@dataclass
class ChainSidecar:
    """One sidecar entry; mirrors the JSON written to disk."""
    file_sha256: str
    previous_chain_hash: str
    chain_hash: str
    table: str
    partition_key: str
    written_at_utc: str
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps({
            "file_sha256": self.file_sha256,
            "previous_chain_hash": self.previous_chain_hash,
            "chain_hash": self.chain_hash,
            "table": self.table,
            "partition_key": self.partition_key,
            "written_at_utc": self.written_at_utc,
            "schema_version": self.schema_version,
        }, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "ChainSidecar":
        d = json.loads(raw)
        return cls(
            file_sha256=d["file_sha256"],
            previous_chain_hash=d["previous_chain_hash"],
            chain_hash=d["chain_hash"],
            table=d["table"],
            partition_key=d["partition_key"],
            written_at_utc=d["written_at_utc"],
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )


# ---------------------------------------------------------------------------
# Building the chain
# ---------------------------------------------------------------------------

def _table_partitions(root: Path, table: str) -> List[Path]:
    """Return parquet files for `table` in chronological (lex) order.

    Layout: `<root>/<table>/trading_date_ist=YYYY-MM-DD/<table>.parquet`.
    Lex order of `trading_date_ist=...` directories matches calendar
    order — that's the chain's ordering convention.
    """
    table_dir = root / table
    if not table_dir.exists():
        return []
    out = []
    for d in sorted(table_dir.glob("trading_date_ist=*")):
        f = d / f"{table}.parquet"
        if f.exists():
            out.append(f)
    return out


def _sidecar_path(parquet_path: Path) -> Path:
    return parquet_path.with_name(parquet_path.name + SIDECAR_SUFFIX)


def build_or_extend_chain(
    root: Path, table: str,
    written_at_utc: Optional[str] = None,
) -> Dict[str, str]:
    """Walk every partition for `table` in chronological order; lay
    down a sidecar for each one that doesn't already have a valid
    one. Returns {"head": chain_head_hex, "n_partitions": int,
    "n_newly_written": int}.

    Existing sidecars are left in place IF they're consistent with
    the parquet file and the prior chain link. If a parquet file's
    bytes don't match its existing sidecar (the file was edited),
    we raise — the chain layer never silently rewrites history. The
    fix path is: undo the edit, or write an explicit correction
    record into the log and rebuild.
    """
    written_at_utc = written_at_utc or (
        pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    partitions = _table_partitions(root, table)
    prev_chain_hash = EMPTY_HASH
    n_newly_written = 0
    for pq in partitions:
        file_sha = sha256_file(pq)
        sidecar = _sidecar_path(pq)
        if sidecar.exists():
            existing = ChainSidecar.from_json(sidecar.read_text())
            if existing.file_sha256 != file_sha:
                raise ChainBroken(
                    f"sidecar mismatch at {pq}: parquet hash {file_sha} "
                    f"!= sidecar {existing.file_sha256}; partition was "
                    f"edited after sealing. To proceed: roll back the "
                    f"edit OR record a correction event and rebuild "
                    f"the chain from the corrected partition forward."
                )
            if existing.previous_chain_hash != prev_chain_hash:
                raise ChainBroken(
                    f"sidecar at {pq} points at previous hash "
                    f"{existing.previous_chain_hash} but the chain "
                    f"says previous = {prev_chain_hash}; earlier "
                    f"history was tampered with."
                )
            prev_chain_hash = existing.chain_hash
            continue
        # No sidecar yet — extend the chain.
        ch = link_hash(file_sha, prev_chain_hash)
        sc = ChainSidecar(
            file_sha256=file_sha,
            previous_chain_hash=prev_chain_hash,
            chain_hash=ch,
            table=table,
            partition_key=pq.parent.name,  # "trading_date_ist=YYYY-MM-DD"
            written_at_utc=written_at_utc,
        )
        _atomic_write_text(sidecar, sc.to_json())
        prev_chain_hash = ch
        n_newly_written += 1
    return {
        "head": prev_chain_hash,
        "n_partitions": len(partitions),
        "n_newly_written": n_newly_written,
    }


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(content)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Verification — what a third party runs
# ---------------------------------------------------------------------------

class ChainBroken(ValueError):
    """Raised when the chain detects tampering. The message names the
    partition and the kind of break (file mismatch vs link mismatch).
    """


def verify_chain(root: Path, table: str,
                 expected_head: Optional[str] = None) -> Dict[str, str]:
    """Read-only verification. Walks every partition in chronological
    order, recomputes its file_sha256 from the parquet bytes,
    reconstructs the chain hash, and checks it matches the sidecar.

    If `expected_head` is provided, the function asserts the final
    chain head equals it — this is the "verify against the publicly
    anchored head" path.

    Returns {"head": ..., "n_partitions": ...} on success. Raises
    ChainBroken on any inconsistency.
    """
    partitions = _table_partitions(root, table)
    prev_chain_hash = EMPTY_HASH
    for pq in partitions:
        sidecar = _sidecar_path(pq)
        if not sidecar.exists():
            raise ChainBroken(f"no sidecar at {sidecar}")
        sc = ChainSidecar.from_json(sidecar.read_text())
        file_sha = sha256_file(pq)
        if sc.file_sha256 != file_sha:
            raise ChainBroken(
                f"file mismatch at {pq}: parquet hash {file_sha} vs "
                f"sidecar {sc.file_sha256}"
            )
        if sc.previous_chain_hash != prev_chain_hash:
            raise ChainBroken(
                f"link mismatch at {pq}: sidecar previous "
                f"{sc.previous_chain_hash} vs chain {prev_chain_hash}"
            )
        expected_ch = link_hash(file_sha, prev_chain_hash)
        if sc.chain_hash != expected_ch:
            raise ChainBroken(
                f"chain hash recomputation mismatch at {pq}: "
                f"sidecar {sc.chain_hash} vs recomputed {expected_ch}"
            )
        prev_chain_hash = sc.chain_hash
    if expected_head is not None and prev_chain_hash != expected_head:
        raise ChainBroken(
            f"chain head mismatch: computed {prev_chain_hash} vs "
            f"expected {expected_head}"
        )
    return {
        "head": prev_chain_hash,
        "n_partitions": str(len(partitions)),
    }
