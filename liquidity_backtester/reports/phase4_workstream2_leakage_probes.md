# Phase 4 Workstream 2 Leakage Probes v1

## Verdict

- Fast leakage probe foundation is implemented.
- Fast probes can now run locally and in CI once the workflow template is installed.
- This is not the full Workstream 2 finish line yet. Full model-retrain shuffle, full MTF replay, and full pool replay still require the local parquet warehouse and feature-store artifacts.

## Implemented

- Added `liqpool/leakage_probes.py`.
- Added fast tests in `tests/leakage/test_fast_leakage_probes.py`.
- Added GitHub workflow template at `docs/workflows/leakage_tests.yml`.
- Note: the active `.github/workflows/` file could not be pushed from this environment because the current GitHub token lacks `workflow` scope.

Fast probes cover:

- Label-shuffle association destruction on persisted-style score/label pairs.
- Feature timestamp future-mask validation using `as_of_ts <= decision_ts`.
- Snapshot `distance_atr` recomputation from decision-time close and ATR.
- Regression test that mutating bars after a snapshot does not change snapshot pool distance.

## Not Yet Complete

These remain the next Workstream 2 tasks:

- Retrain quality/direction/proximity/reaction models on shuffled labels and assert OOS AUC collapses near 0.50.
- Build feature metadata with `as_of_ts` and `decision_ts` for every feature-store row.
- Run full MTF replay for every OOS decision timestamp.
- Run full pool availability replay for every pool.
- Add an artifact-backed local/nightly runner for the slow probes.

## Interpretation

The fast probes are a release-safety starting point, not a proof of no leakage. They make the highest-risk assumptions testable locally and are ready to be wired into CI once a token/user with workflow scope installs the template under `.github/workflows/`. The heavier probes are still engineered around the real Core25 parquet data and saved feature store.

## Next Step

Continue Workstream 2 by adding artifact-backed slow probes:

1. Generate or load compact feature metadata for proximity rows.
2. Add `reports/leakage_findings.md` output for any slow-probe failure.
3. Run the distance-at-decision audit on real OOS proximity rows before Track A.
