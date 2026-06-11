# Premium Experiment Harness — Stream R specification

> The founder's delta-implied limit-order idea, formalised. Origin:
> founder's live options trading observations, 2026-06.

## 1. The founder's insight (preserved, math corrected)

When you predict the underlying will move, the Greeks let you
translate that prediction into an exact premium level — and a LIMIT
ORDER at that level converts your prediction into a falsifiable
experiment the market grades for free.

Worked example (corrected units): put premium ₹20. Prediction:
underlying rises 30 points. Put delta ≈ −0.5 → premium should fall
~15 → target ₹5. Place a **limit buy at ₹5**.

- **Filled** → the underlying did (at least) what you predicted.
  You now also OWN a cheap put — either as the reversal leg of the
  thesis or as a hedge acquired at your own predicted bottom.
- **Not filled** → prediction wrong, and the minimum premium reached
  tells you exactly *by how much* (in delta-translated points).

Every such order is a self-labeling data point. Accumulate them and
you can (a) train an entry model on what actually fills, and (b)
**reverse-engineer calibrated underlying-move predictions** from the
premium path — if you can predict the premium change, inverting
through the Greeks gives you the move magnitude, calibrated.

## 2. Why this is archit