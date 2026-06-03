# Agent onboarding

If you are Claude (Opus), Codex, or any other automated assistant opening
this repo: read these two documents BEFORE making any changes or proposing
plans. They are the shared context that prior sessions agreed on, and they
prevent the cross-session drift that has previously caused broken commits.

1. `docs/COMPANY_MAP.md` — the architectural map of the project, the
   two-axis taxonomy (Stack × Horizon), and which layers feed which
   commercial product. This is the long-lived "what we are building"
   reference.

2. `docs/COORDINATION.md` — the live coordination board. Sections:
   `IN FLIGHT` (work currently owned by an agent), `RECENTLY DECIDED`
   (decisions made in the last few sessions with rationale), `NEXT UP`
   (the priority queue with owner + acceptance criteria), `OPEN
   QUESTIONS FOR USER`. Update it on every commit you make so the next
   agent knows the state.

## Operating rules between Opus and Codex

- **Opus** is the strategy + diagnosis + design partner. Opus writes
  code only when (a) the user explicitly asks for it, or (b) the work
  is small enough to ship in one self-contained commit without blocking
  Codex's parallel work.
- **Codex** is the primary implementer + retraining + sweep runner.
  Codex owns long-running infrastructure work, multi-file refactors,
  cluster jobs, and anything that touches the live training pipeline.
- **Before starting work, every agent reads the IN FLIGHT section of
  `docs/COORDINATION.md`**. If your planned change overlaps with what
  another agent owns, hand it to that agent via a coordination-doc
  entry instead of editing the file.
- **After every commit, append a one-line entry to RECENTLY DECIDED** in
  the coordination doc. Format: `YYYY-MM-DD HH:MM [agent] commit_sha
  one-line summary`. Use the same format that already exists in the
  file. Do NOT include real-time timestamps that change between
  sessions — use the commit date.
- **Never overwrite the coordination doc's IN FLIGHT entries for work
  you don't own.** Add new entries, don't replace existing ones.

## Repository scope

- The Python package lives in `liqpool/`.
- Tests in `tests/`.
- Long-running analysis scripts in `analysis/`.
- Operational docs in `docs/`.
- The active development branch is `claude/liquidity-pool-backtester-1uskb`.

## What you must NEVER do

- Push to `main` without explicit user permission.
- Force-push.
- Delete a coordination-doc entry that isn't yours.
- Hide test failures with skips or guards — fix the root cause.
- Add tipster-style "buy X at Y stop Z" outputs to any product. The
  product is research context, not trade instructions. This is both a
  legal (SEBI) requirement and a deliberate moat decision.
