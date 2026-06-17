#!/usr/bin/env bash
set -u

ROOT="${SENTINEL_ROOT:-/root/Project-freedom}"
JOURNAL_DIR="${SENTINEL_JOURNAL_DIR:-/root/.sentinel}"
PORT="${SENTINEL_PORT:-8800}"
JSONL="$JOURNAL_DIR/liqpool_live_signals.jsonl"
NOW="$(date +%s)"
FAIL=0

say() { printf '%s\n' "$*"; }
section() { printf '\n==== %s ====\n' "$*"; }
bad() { say "FAIL: $*"; FAIL=1; }
warn() { say "WARN: $*"; }
ok() { say "OK: $*"; }

section "Premium Belief Doctor"
say "time: $(date)"
say "root: $ROOT"
say "journal_dir: $JOURNAL_DIR"
say "jsonl: $JSONL"
say "port: $PORT"

section "Sentinel server"
SERVER_PIDS="$(pgrep -f 'uvicorn sentinel.server:app' || true)"
if [ -z "$SERVER_PIDS" ]; then
  bad "no uvicorn sentinel.server:app process is running"
else
  ps -o pid,ppid,stat,pcpu,pmem,rss,etime,args -p $SERVER_PIDS 2>/dev/null || true
  for pid in $SERVER_PIDS; do
    say ""
    say "process env flags for pid=$pid:"
    if [ -r "/proc/$pid/environ" ]; then
      tr '\0' '\n' < "/proc/$pid/environ" \
        | grep -E '^(SENTINEL_TOKEN|SENTINEL_DEMO|SENTINEL_JOURNAL_DIR|PYTHONPATH)=' \
        | sed -E 's/^(SENTINEL_TOKEN)=.*/\1=<SET>/' || say "  no sentinel env flags visible"
    else
      warn "cannot read /proc/$pid/environ"
    fi
  done
fi

section "HTTP auth and API"
HEALTH_CODE="$(curl -s -o /tmp/sentinel_healthz.out -w '%{http_code}' "http://127.0.0.1:$PORT/healthz" || true)"
if [ "$HEALTH_CODE" = "200" ]; then
  ok "/healthz returned 200"
else
  bad "/healthz returned ${HEALTH_CODE:-curl_failed}"
fi

API_CODE="$(curl -s -o /tmp/sentinel_belief.out -w '%{http_code}' "http://127.0.0.1:$PORT/api/premium_belief?limit=1" || true)"
say "premium belief status without token: ${API_CODE:-curl_failed}"
if [ "$API_CODE" = "401" ]; then
  bad "Sentinel is token-gated. Either enter the exact token in browser prompt or restart server with env -u SENTINEL_TOKEN."
elif [ "$API_CODE" = "200" ]; then
  ok "/api/premium_belief is reachable without token"
else
  warn "/api/premium_belief returned ${API_CODE:-curl_failed}; response:"
  sed -n '1,12p' /tmp/sentinel_belief.out 2>/dev/null || true
fi

section "Belief runner"
RUNNER_PIDS="$(pgrep -f 'scripts/run_belief_live.py' || true)"
if [ -z "$RUNNER_PIDS" ]; then
  bad "no scripts/run_belief_live.py process is running"
else
  ps -o pid,ppid,stat,pcpu,pmem,rss,etime,args -p $RUNNER_PIDS 2>/dev/null || true
  if pgrep -af 'scripts/run_belief_live.py' | grep -q -- '--max-ticks'; then
    warn "runner was started with --max-ticks; it will stop by design"
  fi
fi

section "JSONL feed freshness"
if [ ! -e "$JSONL" ]; then
  bad "JSONL feed file does not exist at $JSONL"
else
  SIZE="$(wc -c < "$JSONL" 2>/dev/null || echo 0)"
  MTIME="$(stat -c %Y "$JSONL" 2>/dev/null || echo 0)"
  AGE=$((NOW - MTIME))
  say "size_bytes: $SIZE"
  say "age_seconds: $AGE"
  if [ "$SIZE" -le 0 ]; then
    bad "JSONL feed exists but is empty"
  elif [ "$AGE" -gt 5 ]; then
    bad "JSONL feed is stale by ${AGE}s; UI will show old data"
  else
    ok "JSONL feed is fresh"
  fi
  say ""
  say "last row summary:"
  python3 - "$JSONL" <<'PY' 2>/dev/null || tail -1 "$JSONL"
import json, sys
path = sys.argv[1]
line = ""
with open(path, "rb") as f:
    for raw in f:
        if raw.strip():
            line = raw.decode("utf-8", "replace")
if not line:
    print("no non-empty rows")
    raise SystemExit
row = json.loads(line)
extras = row.get("extras") or {}
snap = extras.get("snapshot") or row.get("snapshot") or {}
decision = snap.get("decision") or {}
print("asset=", row.get("asset") or extras.get("underlying") or snap.get("underlying"))
print("ts_ist=", row.get("ts_ist") or snap.get("ts"))
print("signal=", row.get("signal") or decision.get("action"))
print("confidence=", row.get("confidence") or decision.get("confidence"))
print("spot=", extras.get("spot") or snap.get("spot"))
print("bars_seen=", snap.get("bars_seen"))
PY
fi

section "Verdict"
if [ "$FAIL" -eq 0 ]; then
  ok "Premium Belief live chain looks healthy: server reachable, API allowed, runner alive, JSONL fresh."
else
  bad "Premium Belief chain is broken. Fix the FAIL lines above first."
fi
exit "$FAIL"
