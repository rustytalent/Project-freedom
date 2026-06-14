"""RunManifest — training/inference run metadata.

Codex's report listed RunManifest as a missing core contract. Every
serious training run should record the git sha, command line, data
roots used, model bundle path, stage statuses + warnings — so a
future audit can answer "what produced this artifact?"
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class StageStatus:
    name: str
    status: str = "pending"          # pending / ok / warn / fail / skipped
    note: str = ""
    started_at_utc: Optional[str] = None
    ended_at_utc: Optional[str] = None


@dataclass
class RunManifest:
    run_tag: str                              # human label
    mode: str = ""                            # train / predict / live / brief
    git_commit: str = ""
    branch: str = ""
    command: str = ""
    started_at_utc: str = ""
    ended_at_utc: str = ""
    data_root: str = ""
    latest_data_timestamp_utc: str = ""
    model_dir: str = ""
    feature_store_dir: str = ""
    output_dir: str = ""
    stages: List[StageStatus] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stages"] = [asdict(s) for s in self.stages]
        return d

    def save_json(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_row(), indent=2))
        return p

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "RunManifest":
        stages = [StageStatus(**s) for s in row.get("stages") or []
                  if isinstance(s, dict)]
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in row.items()
              if k in known and k != "stages"}
        kw["stages"] = stages
        kw.setdefault("run_tag", row.get("run_tag", ""))
        return cls(**kw)


def detect_git_commit() -> str:
    """Best-effort git sha at the cwd. Empty string if not a repo."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            timeout=2).decode().strip()
    except Exception:
        return ""


def detect_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=2).decode().strip()
    except Exception:
        return ""


def fresh_manifest(run_tag: str, mode: str, command: str = "") -> RunManifest:
    now = _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    return RunManifest(
        run_tag=run_tag,
        mode=mode,
        command=command,
        git_commit=detect_git_commit(),
        branch=detect_git_branch(),
        started_at_utc=now,
    )
