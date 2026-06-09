"""Stream J — outcome-log hash chain tests.

Pins:
  * sha256 sidecar is written next to each partition file on build
  * build_or_extend_chain is idempotent (re-running adds nothing new)
  * verify_chain catches file-tampering (parquet bytes changed)
  * verify_chain catches link-tampering (sidecar previous_hash forged)
  * verify_chain catches chain-recomputation tampering (sidecar's
    own chain_hash doesn't match its declared inputs)
  * expected_head check fails when the published head disagrees
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from liqpool.products.outcome_hash_chain import (
    ChainBroken,
    ChainSidecar,
    build_or_extend_chain,
    link_hash,
    sha256_file,
    verify_chain,
)


def _seed_two_partitions(root: Path) -> None:
    """Lay down two partition parquet files in the canonical layout."""
    base = root / "predictions"
    p1_dir = base / "trading_date_ist=2026-06-08"
    p1_dir.mkdir(parents=True)
    pd.DataFrame({"a": [1, 2, 3]}).to_parquet(p1_dir / "predictions.parquet")
    p2_dir = base / "trading_date_ist=2026-06-09"
    p2_dir.mkdir(parents=True)
    pd.DataFrame({"a": [4, 5]}).to_parquet(p2_dir / "predictions.parquet")


def test_build_writes_one_sidecar_per_partition(tmp_path: Path):
    _seed_two_partitions(tmp_path)
    summary = build_or_extend_chain(tmp_path, "predictions")
    assert summary["n_partitions"] == 2
    assert summary["n_newly_written"] == 2
    p1 = tmp_path / "predictions" / "trading_date_ist=2026-06-08"
    p2 = tmp_path / "predictions" / "trading_date_ist=2026-06-09"
    assert (p1 / "predictions.parquet.sha256.json").exists()
    assert (p2 / "predictions.parquet.sha256.json").exists()


def test_build_is_idempotent(tmp_path: Path):
    """Running the builder twice writes nothing the second time."""
    _seed_two_partitions(tmp_path)
    first = build_or_extend_chain(tmp_path, "predictions")
    second = build_or_extend_chain(tmp_path, "predictions")
    assert first["head"] == second["head"]
    assert second["n_newly_written"] == 0


def test_verify_passes_on_a_clean_chain(tmp_path: Path):
    _seed_two_partitions(tmp_path)
    summary = build_or_extend_chain(tmp_path, "predictions")
    verified = verify_chain(tmp_path, "predictions",
                            expected_head=summary["head"])
    assert verified["head"] == summary["head"]


def test_verify_detects_parquet_tampering(tmp_path: Path):
    """If the parquet file is edited after sealing, verify_chain
    raises with a clear 'file mismatch' message."""
    _seed_two_partitions(tmp_path)
    build_or_extend_chain(tmp_path, "predictions")
    # Tamper: overwrite the 2026-06-08 partition with different data.
    p1 = (tmp_path / "predictions" / "trading_date_ist=2026-06-08"
          / "predictions.parquet")
    pd.DataFrame({"a": [99]}).to_parquet(p1)
    with pytest.raises(ChainBroken, match="file mismatch"):
        verify_chain(tmp_path, "predictions")


def test_verify_detects_link_tampering(tmp_path: Path):
    """If a sidecar's previous_chain_hash is forged, verify_chain
    raises with a 'link mismatch'."""
    _seed_two_partitions(tmp_path)
    build_or_extend_chain(tmp_path, "predictions")
    # Tamper: forge the second sidecar's previous_chain_hash to all-zeros.
    s2 = (tmp_path / "predictions" / "trading_date_ist=2026-06-09"
          / "predictions.parquet.sha256.json")
    sidecar = ChainSidecar.from_json(s2.read_text())
    forged = ChainSidecar(
        file_sha256=sidecar.file_sha256,
        previous_chain_hash="0" * 64,  # forged
        chain_hash=sidecar.chain_hash,  # NOT recomputed -> chain-hash check will also catch
        table=sidecar.table,
        partition_key=sidecar.partition_key,
        written_at_utc=sidecar.written_at_utc,
    )
    s2.write_text(forged.to_json())
    with pytest.raises(ChainBroken, match="link mismatch|chain hash recomputation"):
        verify_chain(tmp_path, "predictions")


def test_build_refuses_to_silently_reseal_a_tampered_partition(tmp_path: Path):
    """If a partition was sealed and then edited, the BUILDER (not
    just the verifier) must refuse to silently re-seal it. This is
    what stops a hostile operator from quietly relinking the chain
    around a forged partition."""
    _seed_two_partitions(tmp_path)
    build_or_extend_chain(tmp_path, "predictions")
    p1 = (tmp_path / "predictions" / "trading_date_ist=2026-06-08"
          / "predictions.parquet")
    pd.DataFrame({"a": [99]}).to_parquet(p1)
    with pytest.raises(ChainBroken, match="sidecar mismatch"):
        build_or_extend_chain(tmp_path, "predictions")


def test_verify_fails_on_wrong_expected_head(tmp_path: Path):
    _seed_two_partitions(tmp_path)
    build_or_extend_chain(tmp_path, "predictions")
    with pytest.raises(ChainBroken, match="head mismatch"):
        verify_chain(tmp_path, "predictions",
                     expected_head="0" * 64)


def test_link_hash_is_deterministic_and_order_sensitive():
    """sha256(file || prev) is intentionally order-sensitive."""
    a = link_hash("abc", "def")
    b = link_hash("abc", "def")
    c = link_hash("def", "abc")
    assert a == b
    assert a != c


def test_chain_extends_when_a_new_partition_lands(tmp_path: Path):
    """The expected normal path: seal what's there, then write a new
    partition, then re-build and extend the chain."""
    _seed_two_partitions(tmp_path)
    s1 = build_or_extend_chain(tmp_path, "predictions")
    # New partition lands.
    p3 = tmp_path / "predictions" / "trading_date_ist=2026-06-10"
    p3.mkdir()
    pd.DataFrame({"a": [6, 7, 8, 9]}).to_parquet(p3 / "predictions.parquet")
    s2 = build_or_extend_chain(tmp_path, "predictions")
    assert s2["n_partitions"] == 3
    assert s2["n_newly_written"] == 1
    # New head differs from old head.
    assert s2["head"] != s1["head"]
    # And verify still passes against the new head.
    verify_chain(tmp_path, "predictions", expected_head=s2["head"])
