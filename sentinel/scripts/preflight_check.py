"""Pre-launch verification — a single command an SRE runs before going live.

What it checks (each is an explicit assertion with a clear failure
message; exit 0 = launchable, exit non-zero = stop and read the output):

  1. Both halves are importable (sentinel + liqpool)
  2. Every IO-declared module loads cleanly
  3. The full Sentinel test suite passes
  4. The full liquidity_backtester test suite passes
  5. Critical files exist (auth, byok, audit_log, trust_spine,
     orchestration, shadow_ledger, contracts package)
  6. /healthz + /readyz respond on a freshly-spun demo instance
  7. The launch README + Codex backlog + Mother's audit docs exist
  8. No accidental ``SENTINEL_CONFIRM_REAL=1`` in the running env
  9. The journal dir is writable

Usage:
    python -m sentinel.scripts.preflight_check
    python -m sentinel.scripts.preflight_check --strict  (fail on warnings)
"""
from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class CheckResult:
    OK = "✓"
    WARN = "⚠"
    FAIL = "✗"

    def __init__(self) -> None:
        self.passes: List[str] = []
        self.warnings: List[str] = []
        self.failures: List[str] = []

    def passed(self, msg: str) -> None:
        self.passes.append(msg)
        print(f"  {self.OK} {msg}")

    def warned(self, msg: str) -> None:
        self.warnings.append(msg)
        print(f"  {self.WARN} {msg}")

    def failed(self, msg: str) -> None:
        self.failures.append(msg)
        print(f"  {self.FAIL} {msg}")

    def section(self, name: str) -> None:
        print(f"\n── {name} ──")

    def summary(self, strict: bool) -> int:
        print("\n" + "=" * 62)
        print(f"  passes:   {len(self.passes)}")
        print(f"  warnings: {len(self.warnings)}")
        print(f"  failures: {len(self.failures)}")
        if self.failures:
            print("\nFailures (must fix before launch):")
            for f in self.failures:
                print(f"  ✗ {f}")
            return 1
        if strict and self.warnings:
            print("\nWarnings escalated to failure (--strict mode)")
            return 1
        print("\n  ALL CHECKS GREEN — launchable")
        return 0


def check_modules_import(r: CheckResult) -> None:
    r.section("Module imports")
    critical = [
        "sentinel.server", "sentinel.auth", "sentinel.byok",
        "sentinel.audit_log", "sentinel.orchestration",
        "sentinel.shadow_ledger", "sentinel.live_publisher",
        "sentinel.psychology", "sentinel.crux_signal",
        "sentinel.replay", "sentinel.paper", "sentinel.notify",
        "sentinel.rate_limit",
        "liqpool.contracts", "liqpool.live_inference",
        "liqpool.sentinel_adapter",
    ]
    for mod in critical:
        try:
            importlib.import_module(mod)
            r.passed(f"import {mod}")
        except Exception as exc:
            r.failed(f"import {mod}: {exc}")


def check_io_registry(r: CheckResult) -> None:
    r.section("IO registry self-declaration")
    try:
        import sentinel.reports  # noqa: F401  triggers all declare() calls
        from sentinel.io_decl import REGISTRY
        n = len(REGISTRY)
        if n >= 30:
            r.passed(f"{n} modules self-declared in IO map")
        else:
            r.warned(f"only {n} modules self-declared (expected >= 30)")
        # spot-check the high-value ones
        for must_have in ("sentinel.auth", "sentinel.byok",
                           "sentinel.audit_log", "sentinel.orchestration",
                           "sentinel.replay", "sentinel.paper"):
            if must_have in REGISTRY:
                r.passed(f"IO declares {must_have}")
            else:
                r.failed(f"IO map missing {must_have}")
    except Exception as exc:
        r.failed(f"could not load IO registry: {exc}")


def check_critical_files(r: CheckResult) -> None:
    r.section("Critical files present")
    must_exist = [
        "sentinel/auth.py", "sentinel/byok.py", "sentinel/audit_log.py",
        "sentinel/orchestration.py", "sentinel/shadow_ledger.py",
        "sentinel/replay.py", "sentinel/paper.py", "sentinel/notify.py",
        "sentinel/rate_limit.py", "sentinel/server.py",
        "sentinel/static/index.html", "sentinel/static/console.html",
        "sentinel/static/sentinel.css",
        "sentinel/scripts/backup_journal.py",
        "sentinel/docs/LAUNCH_README.md",
        "sentinel/docs/CODEX_BACKLOG.md",
        "sentinel/docs/HANDOFF_FOR_CODEX.md",
        "liquidity_backtester/liqpool/contracts/__init__.py",
        "liquidity_backtester/liqpool/contracts/events.py",
        "liquidity_backtester/liqpool/contracts/signals.py",
        "liquidity_backtester/liqpool/live_inference.py",
        "liquidity_backtester/liqpool/sentinel_adapter.py",
        "liquidity_backtester/docs/MOTHERS_AUDIT_2026_06_14.md",
    ]
    for rel in must_exist:
        p = PROJECT_ROOT / rel
        if p.exists() and p.stat().st_size > 0:
            r.passed(rel)
        else:
            r.failed(f"missing or empty: {rel}")


def check_test_suites(r: CheckResult) -> None:
    r.section("Test suites")
    for suite, expected_min in (("sentinel/tests", 370),
                                  ("liquidity_backtester/tests", 800)):
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", str(PROJECT_ROOT / suite),
                  "-q", "--tb=no"],
                capture_output=True, text=True, timeout=180,
                cwd=str(PROJECT_ROOT))
        except Exception as exc:
            r.failed(f"{suite}: failed to run pytest: {exc}")
            continue
        if result.returncode != 0:
            tail = result.stdout.splitlines()[-3:]
            r.failed(f"{suite}: tests failed — {' | '.join(tail)}")
            continue
        # parse last line "X passed in Ys"
        n_passed = 0
        for line in result.stdout.splitlines()[::-1]:
            if " passed" in line:
                try:
                    n_passed = int(line.split()[0])
                except Exception:
                    pass
                break
        if n_passed >= expected_min:
            r.passed(f"{suite}: {n_passed} tests passed (expected ≥{expected_min})")
        else:
            r.warned(f"{suite}: only {n_passed} tests passed "
                      f"(expected ≥{expected_min})")


def check_env_safety(r: CheckResult) -> None:
    r.section("Environment safety")
    real = os.environ.get("SENTINEL_CONFIRM_REAL", "")
    demo = os.environ.get("SENTINEL_DEMO", "")
    if real == "1" and demo != "1":
        r.failed("SENTINEL_CONFIRM_REAL=1 in the running env — "
                  "real-money mode would activate. Set SENTINEL_DEMO=1 "
                  "until launch sign-off.")
    else:
        r.passed("SENTINEL_CONFIRM_REAL not set (real-money path safe)")
    if os.environ.get("SENTINEL_ALLOW_HEADER_AUTH") == "1":
        r.warned("SENTINEL_ALLOW_HEADER_AUTH=1 — dev header backdoor "
                  "open; unset before public launch")
    journal_env = os.environ.get("SENTINEL_JOURNAL_DIR")
    if journal_env:
        p = Path(journal_env)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".preflight_probe"
            probe.touch()
            probe.unlink()
            r.passed(f"journal dir writable: {p}")
        except Exception as exc:
            r.failed(f"journal dir not writable ({journal_env}): {exc}")
    else:
        r.warned("SENTINEL_JOURNAL_DIR unset — default ~/.sentinel will be used")


def check_demo_server_responds(r: CheckResult) -> None:
    r.section("Demo server smoke")
    port = 8801
    env = dict(os.environ)
    env["SENTINEL_DEMO"] = "1"
    env["SENTINEL_JOURNAL_DIR"] = str(PROJECT_ROOT / ".preflight_journal")
    Path(env["SENTINEL_JOURNAL_DIR"]).mkdir(parents=True, exist_ok=True)
    proc = None
    try:
        proc = subprocess.Popen(
            ["python", "-m", "uvicorn", "sentinel.server:app",
              "--host", "127.0.0.1", "--port", str(port),
              "--log-level", "error"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT), env=env)
        # wait for the server to come up
        for _ in range(50):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/healthz", timeout=0.5) as resp:
                    if resp.getcode() == 200:
                        break
            except Exception:
                time.sleep(0.2)
        else:
            r.failed("/healthz did not respond within 10s")
            return
        # smoke through the key endpoints
        for path, expect in (("/healthz", 200), ("/readyz", 200),
                              ("/", 200), ("/console", 200),
                              ("/sentinel.css", 200)):
            try:
                with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}{path}", timeout=1) as resp:
                    if resp.getcode() == expect:
                        r.passed(f"GET {path} -> {expect}")
                    else:
                        r.failed(f"GET {path} -> {resp.getcode()} "
                                  f"(expected {expect})")
            except Exception as exc:
                r.failed(f"GET {path}: {exc}")
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.kill()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--strict", action="store_true",
                   help="fail (non-zero) on warnings too")
    p.add_argument("--skip-tests", action="store_true",
                   help="skip the full pytest run (CI shortcut)")
    p.add_argument("--skip-smoke", action="store_true",
                   help="skip the live demo-server smoke")
    args = p.parse_args()

    # Ensure both halves are importable from the project root
    for p_ in (str(PROJECT_ROOT), str(PROJECT_ROOT / "liquidity_backtester")):
        if p_ not in sys.path:
            sys.path.insert(0, p_)

    print("=" * 62)
    print(" Sentinel pre-launch check")
    print("=" * 62)

    r = CheckResult()
    check_modules_import(r)
    check_io_registry(r)
    check_critical_files(r)
    check_env_safety(r)
    if not args.skip_tests:
        check_test_suites(r)
    if not args.skip_smoke:
        check_demo_server_responds(r)
    return r.summary(args.strict)


if __name__ == "__main__":
    sys.exit(main())
