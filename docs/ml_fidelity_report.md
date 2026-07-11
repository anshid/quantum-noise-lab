# Part 6: ML Fidelity Prediction vs. Physics Baseline

## Hypothesis

An ML regressor trained only on Part 3's swept data (`experiments/results/noise_sweep.csv`, 670
rows: `circuit, qubit_count, channel, param, depth -> fidelity`) can fit the observed grid well,
but — because it has no access to the mechanism Part 5 derived by hand — its error relative to
the exact physics closed form (available for `phase_flip`/`phase_damping`) should grow, not
shrink, when evaluated on rows extrapolated past the edge of its training range. A high-bias
linear model should additionally fail to represent the multiplicative, non-additive way fidelity
compounds over `depth`.

## Methodology

1. **Reused Part 3's committed sweep** directly, no new simulation — `circuit`/`channel`
   one-hot-encoded, `qubit_count`/`param`/`depth` passed through as numeric features, `fidelity`
   as the single regression target (`src/ml_fidelity.py::build_feature_target`).
2. **Two model families**, bracketing the bias-variance tradeoff: `LinearRegression` (high bias)
   and `RandomForestRegressor` (low bias, `n_estimators=200`, `max_depth=8`).
3. **Three evaluation splits**: one interpolation split (random, stratified by `channel`,
   `test_size=0.2`) and two extrapolation splits (train on `qubit_count<=4` / test on
   `qubit_count in {5,6}`; train on `depth in {1,2}` / test on `depth in {4,8}`) —
   `src/ml_fidelity.py::interpolation_split`, `extrapolation_split`.
4. **Physics baseline** (`src/ml_fidelity.py::physics_baseline_fidelity`): for
   `phase_flip`/`phase_damping` rows, the Part 5 closed form $F = (1+r^{n\cdot\text{depth}})/2$,
   generalized to `depth`>1 by noting each of `depth` sequential noise layers multiplies a
   qubit's coherence contribution by the same per-layer factor $r$ (confirmed exactly against
   the committed sweep at `depth`$\in\{1,2,4\}$ before use). Scored against the same test rows as
   the ML models, restricted to the two eligible channels.
5. **Feature importance** on the interpolation split only (extrapolation splits deliberately
   unbalance the feature distribution, which would confound importance with the split itself):
   Random Forest Gini importance (aggregated from one-hot columns back to original features) and
   `sklearn.inspection.permutation_importance` (computed directly on original features) as a
   cross-check.

Reproducible end-to-end via `python experiments/run_ml_fidelity.py`
(config: `experiments/configs/ml_fidelity.toml`) →
`experiments/results/ml_fidelity_metrics.csv` and `..._feature_importance.csv`; walked through
interactively in `notebooks/06_ml_fidelity_prediction.ipynb`.

## Results

**Regression metrics (test set), by split:**

| Split | Model | R² | MAE | RMSE |
|---|---|---|---|---|
| interpolation | linear | 0.553 | 0.149 | 0.193 |
| interpolation | random_forest | 0.961 | 0.038 | 0.057 |
| extrapolation_qubit_count | linear | 0.423 | 0.180 | 0.235 |
| extrapolation_qubit_count | random_forest | 0.934 | 0.061 | 0.080 |
| extrapolation_depth | linear | **−0.369** | 0.217 | 0.306 |
| extrapolation_depth | random_forest | 0.875 | 0.066 | 0.092 |

**The random forest fits and interpolates well, and degrades monotonically under
extrapolation** — test R² falls from 0.961 (interpolation) to 0.934 (`qubit_count` held out) to
0.875 (`depth` held out). This is the expected signature of a tree ensemble asked to extrapolate:
past the maximum feature value seen in training, it can only predict the nearest training leaf's
average, not continue the underlying decay trend, and the effect compounds most when the held-out
axis (`depth`) has the most nonlinear/multiplicative relationship to fidelity.

**Linear regression's `depth`-extrapolation R² is negative (−0.369)** — worse than predicting the
training-set mean for every test row. This is the sharpest single result in the study: fidelity
compounds multiplicatively over `depth` (each noise layer multiplies surviving
population/coherence by a fixed per-layer factor), so a model that is linear in `depth` is fitting
the wrong functional form outright, and the mismatch gets worse, not better, as `depth` is
extrapolated further from the training range.

**Physics baseline vs. random forest, MAE on the two dephasing channels' rows only:**

| Split | Random forest MAE | Physics baseline MAE |
|---|---|---|
| interpolation | 0.0314 | 6.5×10⁻⁹ |
| extrapolation_qubit_count | 0.0305 | 6.5×10⁻⁹ |
| extrapolation_depth | 0.0566 | 7.1×10⁻⁹ |

The physics baseline is exact to floating-point precision on every split, by construction — it is
the actual analytic fidelity, not a fit. The random forest's error, while small in absolute terms,
is roughly 4–7 orders of magnitude larger throughout, and specifically worsens further under
`depth` extrapolation (MAE 0.031 → 0.057, R² 0.92 → 0.81); it does *not* worsen under
`qubit_count` extrapolation on this dephasing-only subset (MAE actually ticks down slightly to
0.031, R² up to 0.95) — consistent with `depth` being the axis where the full-test-set random
forest degradation (see table above) was also sharpest.

**Feature importance** (Gini, aggregated to original features, and permutation, agreeing in
ranking): `param` dominates by a wide margin, `channel` is second, `qubit_count` third, `depth`
smaller but nonzero, and `circuit` is essentially negligible — consistent with the 2-qubit Bell
state used here being itself a 2-qubit GHZ state, so once `qubit_count` is known `circuit` carries
almost no additional information.

## Limitations

- **Physics baseline covers only 2 of 5 channels.** Bit-flip, depolarizing, and amplitude damping
  have no closed form derived in this project (Part 5's own limitation); whether the random
  forest is relatively better or worse on those channels' functional shapes is not checked here.
- **Dataset size (670 rows) is small by ML standards**, and the target is exact/near-exact
  simulated fidelity, not measurement-noise-corrupted data — results may not transfer to a
  sparser or noisier real-world sweep.
- **Extrapolation was only tested one axis at a time.** The compounded case (both `qubit_count`
  and `depth` extrapolated simultaneously) was not evaluated and would likely degrade further.
- **No neural network was tried**, deliberately — a few hundred rows and five features is not
  where a neural net's capacity advantage over a tree ensemble would show up, and skipping it
  keeps the interpretability story (Gini + permutation importance) clean.
- **Noise-type classification and inverse (error-probability) estimation were considered and
  explicitly deferred.** Classification in particular is provably ill-posed using only this
  dataset's Z-basis aggregate features: `success_probability`/`error_rate` are invariant to
  `param` for `phase_flip`/`phase_damping`, since both channels commute with computational-basis
  measurement (same mechanism as Part 5's population/coherence argument) — resolving this would
  need new multi-basis measurement data, a natural stretch goal rather than something built here.

## Conclusion

A generic ML model can fit and interpolate Part 3's swept fidelity data well, but its
generalization degrades measurably past the edge of the training grid, most sharply for a linear
model asked to extrapolate a multiplicative decay linearly (negative R²), and more gently but
still monotonically for a random forest, which structurally cannot extrapolate a trend at all. Set
against the two channels with an exact physics closed form, the ML model's error — though small —
is consistently orders of magnitude worse than the mechanistic prediction, and that gap widens
further specifically under depth extrapolation. The practical takeaway for
an adaptive-noise-modelling pipeline: prefer a derived, mechanistic model wherever one is
available, and treat a data-driven fit as a fallback for the regime it has no shortcut for
(bit-flip, depolarizing, amplitude damping here) — not as a general replacement for physics
understanding, especially outside the range of the data it was trained on.
