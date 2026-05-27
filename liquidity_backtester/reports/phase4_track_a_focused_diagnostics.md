# Phase 4 Track A Focused Diagnostics

## Verdict

- Decision: `FOCUSED_POCKET_READY_FOR_CONSTRAINED_VALIDATION`
- At least one filtered pocket is broad enough for the next constrained validation run.
- Best broad diagnostic pocket: `long_ex_banking_it` / `all` with `440` trades, mean R `+0.360`, PF `1.89`, second-half R `+0.340`, and 1.5x-cost R `+0.185`.
- This is still not live-trade approval; these filters were discovered after the sweep.

## Base Cell

- Gate: `T>=0.75|D>=0.60|3-8ATR`
- Geometry: target_fraction `1.0`, stop `2.0 ATR`, hold `60` bars
- Base trades: `600`

## Strict-Ready Filtered Pockets

| scenario | Q | n | win | mean_R | PF | cost1.5x | 1st/2nd | L/S | sectors | survives | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_ex_banking_it | all | 440 | 57.0% | 0.360 | 1.890 | 0.185 | 0.380/0.340 | 440/0 | 3 | True | True |
| long_auto_pharma_fmcg | all | 440 | 57.0% | 0.360 | 1.890 | 0.185 | 0.380/0.340 | 440/0 | 3 | True | True |
| ex_banking_it | all | 456 | 56.8% | 0.341 | 1.831 | 0.162 | 0.374/0.308 | 440/16 | 3 | True | True |
| auto_pharma_fmcg | all | 456 | 56.8% | 0.341 | 1.831 | 0.162 | 0.374/0.308 | 440/16 | 3 | True | True |
| long_only | all | 541 | 55.6% | 0.307 | 1.752 | 0.130 | 0.363/0.253 | 541/0 | 5 | True | True |

## Focused Survivors

| scenario | Q | n | win | mean_R | PF | cost1.5x | 1st/2nd | L/S | sectors | survives | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| long_morning_midday_ex_banking_it | all | 188 | 68.1% | 0.762 | 3.465 | 0.614 | 1.097/0.613 | 188/0 | 2 | True | False |
| long_morning_ex_banking_it | all | 167 | 65.3% | 0.690 | 3.094 | 0.546 | 1.097/0.474 | 167/0 | 2 | True | False |
| long_morning_midday | all | 231 | 62.3% | 0.640 | 2.791 | 0.496 | 0.905/0.507 | 231/0 | 2 | True | False |
| long_morning | all | 209 | 59.3% | 0.557 | 2.465 | 0.416 | 0.905/0.354 | 209/0 | 2 | True | False |
| morning_midday | all | 246 | 59.3% | 0.541 | 2.340 | 0.395 | 0.905/0.375 | 231/15 | 3 | True | False |
| morning_only | all | 218 | 57.3% | 0.496 | 2.207 | 0.354 | 0.905/0.272 | 209/9 | 3 | True | False |
| all | top_25pct | 150 | 60.0% | 0.447 | 2.105 | 0.250 | 0.688/0.205 | 133/17 | 3 | True | False |
| long_ex_banking_it | top_50pct | 220 | 59.5% | 0.424 | 2.031 | 0.250 | 0.370/0.467 | 220/0 | 2 | True | False |
| long_auto_pharma_fmcg | top_50pct | 220 | 59.5% | 0.424 | 2.031 | 0.250 | 0.370/0.467 | 220/0 | 2 | True | False |
| auto_pharma_fmcg | top_50pct | 228 | 58.3% | 0.392 | 1.927 | 0.213 | 0.363/0.414 | 217/11 | 2 | True | False |
| ex_banking_it | top_50pct | 228 | 58.3% | 0.392 | 1.927 | 0.213 | 0.363/0.414 | 217/11 | 2 | True | False |
| long_only | top_50pct | 271 | 58.3% | 0.372 | 1.920 | 0.196 | 0.420/0.334 | 271/0 | 3 | True | False |
| long_ex_banking_it | all | 440 | 57.0% | 0.360 | 1.890 | 0.185 | 0.380/0.340 | 440/0 | 3 | True | True |
| long_auto_pharma_fmcg | all | 440 | 57.0% | 0.360 | 1.890 | 0.185 | 0.380/0.340 | 440/0 | 3 | True | True |
| auto_pharma_fmcg | all | 456 | 56.8% | 0.341 | 1.831 | 0.162 | 0.374/0.308 | 440/16 | 3 | True | True |
| ex_banking_it | all | 456 | 56.8% | 0.341 | 1.831 | 0.162 | 0.374/0.308 | 440/16 | 3 | True | True |
| long_only | all | 541 | 55.6% | 0.307 | 1.752 | 0.130 | 0.363/0.253 | 541/0 | 5 | True | True |
| all | top_50pct | 300 | 55.0% | 0.283 | 1.644 | 0.099 | 0.408/0.192 | 272/28 | 3 | True | False |

## Top Diagnostics By Mean R

| scenario | Q | n | win | mean_R | PF | cost1.5x | 1st/2nd | L/S | sectors | survives | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| auto_only | top_5pct | 10 | 90.0% | 1.164 | 10.375 | 0.979 | 1.605/0.723 | 10/0 | 0 | False | False |
| midday_only | top_50pct | 14 | 78.6% | 1.104 | 4.663 | 0.930 | 0.063/1.278 | 11/3 | 0 | False | False |
| long_morning_midday | top_5pct | 12 | 75.0% | 1.059 | 7.762 | 0.936 | 0.910/2.704 | 12/0 | 0 | False | False |
| midday_only | top_25pct | 7 | 71.4% | 1.052 | 3.428 | 0.866 | -1.581/1.491 | 5/2 | 0 | False | False |
| long_morning_midday | top_25pct | 58 | 70.7% | 1.004 | 4.663 | 0.854 | 1.325/0.777 | 58/0 | 0 | False | False |
| long_morning_midday_ex_banking_it | top_25pct | 47 | 70.2% | 0.997 | 4.372 | 0.843 | 1.483/0.769 | 47/0 | 0 | False | False |
| long_morning | top_5pct | 11 | 72.7% | 0.975 | 6.706 | 0.852 | 0.802/2.704 | 11/0 | 0 | False | False |
| morning_only | top_5pct | 11 | 72.7% | 0.975 | 6.706 | 0.852 | 0.802/2.704 | 11/0 | 0 | False | False |
| fmcg_only | top_10pct | 5 | 80.0% | 0.965 | 21.321 | 0.706 | 1.393/0.323 | 1/4 | 0 | False | False |
| morning_midday | top_25pct | 62 | 71.0% | 0.954 | 4.384 | 0.799 | 1.298/0.722 | 60/2 | 0 | False | False |
| auto_only | top_10pct | 20 | 80.0% | 0.942 | 4.622 | 0.751 | 1.015/0.723 | 20/0 | 0 | False | False |
| fmcg_only | top_5pct | 3 | 66.7% | 0.931 | 12.768 | 0.673 | 1.516/-0.237 | 0/3 | 0 | False | False |
| long_morning | top_25pct | 53 | 69.8% | 0.917 | 4.366 | 0.769 | 1.332/0.518 | 53/0 | 0 | False | False |
| midday_only | all | 28 | 75.0% | 0.889 | 3.565 | 0.712 | -0.496/1.055 | 22/6 | 0 | False | False |
| long_morning_ex_banking_it | top_25pct | 42 | 69.0% | 0.887 | 3.993 | 0.734 | 1.475/0.487 | 42/0 | 0 | False | False |
| morning_only | top_25pct | 55 | 69.1% | 0.871 | 4.055 | 0.724 | 1.332/0.457 | 55/0 | 0 | False | False |
| auto_only | top_50pct | 97 | 78.4% | 0.858 | 4.765 | 0.677 | 0.650/1.227 | 97/0 | 1 | False | False |
| auto_only | top_25pct | 49 | 77.6% | 0.813 | 4.035 | 0.616 | 0.768/0.966 | 49/0 | 1 | False | False |
| long_morning_midday_ex_banking_it | top_50pct | 94 | 66.0% | 0.800 | 3.525 | 0.649 | 1.233/0.675 | 94/0 | 1 | False | False |
| fmcg_only | top_25pct | 12 | 75.0% | 0.764 | 4.720 | 0.512 | 1.045/-0.079 | 7/5 | 0 | False | False |
| long_morning_midday_ex_banking_it | all | 188 | 68.1% | 0.762 | 3.465 | 0.614 | 1.097/0.613 | 188/0 | 2 | True | False |
| morning_midday | top_5pct | 13 | 69.2% | 0.714 | 3.684 | 0.572 | 0.802/0.421 | 11/2 | 0 | False | False |
| long_morning_midday_ex_banking_it | top_5pct | 10 | 50.0% | 0.700 | 3.003 | 0.564 | 0.226/0.818 | 10/0 | 0 | False | False |
| long_morning_midday | top_50pct | 116 | 62.9% | 0.693 | 2.972 | 0.548 | 1.120/0.537 | 116/0 | 1 | False | False |
| long_morning_ex_banking_it | all | 167 | 65.3% | 0.690 | 3.094 | 0.546 | 1.097/0.474 | 167/0 | 2 | True | False |

## Base Breakdown By Sector

| sector | n | win | mean_R | PF |
| --- | --- | --- | --- | --- |
| AUTO | 194 | 65.5% | 0.572 | 2.879 |
| FMCG | 48 | 56.2% | 0.220 | 1.397 |
| PHARMA | 214 | 49.1% | 0.159 | 1.336 |
| IT | 94 | 42.6% | -0.155 | 0.707 |
| BANKING | 50 | 36.0% | -0.162 | 0.755 |

## Base Breakdown By Direction

| direction | n | win | mean_R | PF |
| --- | --- | --- | --- | --- |
| UP | 541 | 55.6% | 0.307 | 1.752 |
| DOWN | 59 | 27.1% | -0.560 | 0.328 |

## Base Breakdown By Time Bucket

| time_bucket | n | win | mean_R | PF |
| --- | --- | --- | --- | --- |
| midday | 28 | 75.0% | 0.889 | 3.565 |
| morning | 218 | 57.3% | 0.496 | 2.207 |
| afternoon | 354 | 48.3% | -0.000 | 1.000 |

## Interpretation

- The short side is actively harmful in the base cell.
- BANKING and IT are negative in the base cell; AUTO/PHARMA/FMCG carry the useful signal.
- Blocking afternoon entries improves the result, but can reduce sector breadth.
- The cleanest next experiment is a constrained validation run, not live trading: long-only, AUTO/PHARMA/FMCG, same Track A gate family, and explicit no-afternoon sensitivity.
- Triple-barrier/final-stage trade modeling remains the right model upgrade after this pocket is validated out-of-sample or on the full universe.
