"""Daily journal backup — tarballs the journal dir for compliance + DR.

What lands in the tarball:
  ledger_<date>.jsonl, suggestions.jsonl + .rates.json,
  trust_promotions.jsonl, mind_reports.jsonl, audit.jsonl,
  paper_orders.jsonl, knobs.json, leakage_history.json,
  liqpool_live_signals.jsonl + rotated archives.

Layout:
  <out_root>/sentinel-backup-<session_date>.tar.gz

Usage (cron at 23:55 IST):
  python -m sentinel.scripts.backup_journal \\
      --journal ~/.sentinel \\
      --session 2026-06-14 \\
      --out /var/lib/sentinel/backups \\
      --keep 30

Optional --upload S3://bucket/path uploads via aws CLI (no boto3 dep).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--journal", type=Path, required=True,
                   help="<journal_dir> the cockpit writes to")
    p.add_argument("--session", default=datetime.utcnow().strftime("%Y-%m-%d"),
                   help="YYYY-MM-DD session label for the tarball")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory for the tarball")
    p.add_argument("--keep", type=int, default=30,
                   help="prune older-than-N-day tarballs in --out")
    p.add_argument("--upload", default=None,
                   help="optional s3://bucket/path destination "
                        "(requires `aws` CLI on PATH)")
    args = p.parse_args()

    if not args.journal.exists():
        print(f"journal dir does not exist: {args.journal}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    tar_name = f"sentinel-backup-{args.session}.tar.gz"
    tar_path = args.out / tar_name
    manifest = {"session": args.session,
                 "created_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "files": []}
    with tarfile.open(tar_path, "w:gz") as tar:
        for f in sorted(args.journal.iterdir()):
            if not f.is_file():
                continue
            tar.add(str(f), arcname=f.name)
            h = hashlib.sha256()
            with f.open("rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            manifest["files"].append({
                "name": f.name, "bytes": f.stat().st_size,
                "sha256": h.hexdigest(),
            })
    (args.out / f"{tar_name}.manifest.json").write_text(
        json.dumps(manifest, indent=2))
    print(f"wrote {tar_path} ({len(manifest['files'])} files)")

    # Prune
    if args.keep > 0:
        backups = sorted(args.out.glob("sentinel-backup-*.tar.gz"))
        for old in backups[:-args.keep]:
            try:
                old.unlink()
                man = old.with_suffix(old.suffix + ".manifest.json")
                if man.exists():
                    man.unlink()
                print(f"pruned {old}")
            except Exception as exc:
                print(f"prune failed for {old}: {exc}", file=sys.stderr)

    # Optional upload
    if args.upload:
        try:
            subprocess.check_call(
                ["aws", "s3", "cp", str(tar_path), args.upload],
                stdout=subprocess.DEVNULL)
            print(f"uploaded to {args.upload}")
        except FileNotFoundError:
            print("--upload requested but `aws` CLI not on PATH; "
                  "skipping upload", file=sys.stderr)
        except subprocess.CalledProcessError as exc:
            print(f"upload failed: {exc}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
