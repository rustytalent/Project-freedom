# Phase 4 Track A Synthetic Null Preflight

## Verdict

- Decision: `SYNTHETIC_PREFLIGHT_CORE_PASS_DIRECTION_WEAK`
- `winning_combo` beats matched-random/time/sector nulls, but the direction hard gate does not beat direction-label shuffle.
- Trials per pocket/null: `1000`
- Workers requested: `1`
- Scope: artifact-backed preflight using existing v2 pre-touch trade rows.
- This does **not** replace full generated random-pool or ATR-offset replay through the simulator.

## Primary Pocket: `winning_combo`

| null | n | actual_R | actual_CI | PF | null_R_med | null_R_95 | p | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_random_rows | 7776 | 0.349 | 0.316/0.383 | 1.728 | -0.176 | -0.150 | 0.0010 | yes |
| time_bucket_shuffle | 7776 | 0.349 | 0.316/0.383 | 1.728 | -0.024 | -0.001 | 0.0010 | yes |
| sector_shuffle | 7776 | 0.349 | 0.316/0.383 | 1.728 | 0.256 | 0.270 | 0.0010 | yes |
| direction_shuffle | 7776 | 0.349 | 0.316/0.383 | 1.728 | 0.349 | 0.359 | 0.5125 | no |

## Other Pockets

### `morning`

| null | n | actual_R | actual_CI | PF | null_R_med | null_R_95 | p | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_random_rows | 8748 | 0.177 | 0.145/0.208 | 1.314 | -0.166 | -0.141 | 0.0010 | yes |
| time_bucket_shuffle | 8748 | 0.177 | 0.145/0.208 | 1.314 | -0.125 | -0.102 | 0.0010 | yes |

### `midday`

| null | n | actual_R | actual_CI | PF | null_R_med | null_R_95 | p | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_random_rows | 1584 | 0.295 | 0.218/0.372 | 1.569 | -0.177 | -0.118 | 0.0010 | yes |
| time_bucket_shuffle | 1584 | 0.295 | 0.218/0.372 | 1.569 | -0.126 | -0.068 | 0.0010 | yes |

### `morning_midday`

| null | n | actual_R | actual_CI | PF | null_R_med | null_R_95 | p | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_random_rows | 10332 | 0.195 | 0.166/0.224 | 1.351 | -0.168 | -0.145 | 0.0010 | yes |
| time_bucket_shuffle | 10332 | 0.195 | 0.166/0.224 | 1.351 | -0.125 | -0.105 | 0.0010 | yes |

### `auto_only`

| null | n | actual_R | actual_CI | PF | null_R_med | null_R_95 | p | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| matched_random_rows | 8856 | 0.201 | 0.170/0.233 | 1.372 | -0.093 | -0.067 | 0.0010 | yes |
| sector_shuffle | 8856 | 0.201 | 0.170/0.233 | 1.372 | -0.124 | -0.103 | 0.0010 | yes |

## Interpretation

- Passing here means the pocket beats cheap label/matched-row nulls and is worth full synthetic replay.
- A direction-shuffle failure is a component warning: direction should become a soft feature, not necessarily a hard gate.
- Failing a core null means the apparent edge is likely explained by time, sector, or geometry/distance sampling effects.
- If this passes, the next step is the heavier pool-level null suite: random pools, ATR-offset pools, sector-neutral random pools, and full V2 shuffled direction.

## Outputs

- Summary CSV: `output_audit/track_a_synthetic_nulls/track_a_synthetic_nulls.csv`
- Trial CSV: `output_audit/track_a_synthetic_nulls/track_a_synthetic_null_trials.csv`