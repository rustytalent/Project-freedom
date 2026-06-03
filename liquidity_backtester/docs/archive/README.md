# Archived documentation

These are historical handoffs and Phase 4 era runbooks preserved for
provenance. They were authored BEFORE the project adopted
`AGENTS.md` + `COMPANY_MAP.md` + `COORDINATION.md` as the canonical
agent-coordination + architecture + tactical-state trio.

**Do not edit these files.** They are frozen as-is. If you are looking
for current information, start from one of:

  * `docs/SYSTEM_OVERVIEW.md`   — what the project IS and how it works.
  * `docs/COMPANY_MAP.md`       — architecture facts.
  * `docs/COORDINATION.md`      — what's in flight RIGHT NOW.
  * `docs/daily_brief_schema.md`         — Daily Brief data contract.
  * `docs/outcome_logging_schema.md`     — Outcome log data contract.
  * `AGENTS.md` (repo root)     — agent operating rules.

## What's here

* `CODEX_HANDOFF.md` (1266 lines)
  Master handoff from the pre-coordination era. Contained the
  R-MEASURE / R-NULLS / R-R3-VERIFY / R-ARSENAL-VERIFY task taxonomy.
  Most tasks have either landed or been superseded by the
  decision-log in `COORDINATION.md`. Preserved for cross-referencing
  the original task names if they appear in old commit messages.

* `next_chat_handoff.md` (3364 lines)
  Phase 4 Track A handoff that pre-dates the moment-null pivot and
  the strategic memo on three revenue lines. The "journey-to-
  liquidity" framing it introduces is correct and survives in
  `SYSTEM_OVERVIEW.md`; the per-Track tactical content is stale.

* `phase4_v2_terminal_runbook.md` (89 lines)
  Phase 4 V2 simulator runbook. The V2 simulator itself still
  exists in `liqpool/execution_simulator_v2.py`; the runbook's
  step-by-step instructions reference paths and configs that no
  longer match the current `analysis/` and `output_*/` layout.

* `track_a_null_decision.md`
  Opus decision memo on the Track A POOL_LEVEL_FAIL interpretation.
  Conclusions survived into the three-revenue-lines pivot
  (research is the lead product, not private alpha). The memo itself
  is historical now.
