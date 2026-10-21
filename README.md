# Turbofan RUL — predictive maintenance service

Predicts **remaining useful life (RUL)** of turbofan engines from multivariate sensor
histories (NASA C-MAPSS, all four subsets: 709 training engines, 707 test engines, 265K rows),
and serves predictions with calibrated intervals and early-failure alerts behind a REST API.

The core problem is the same one real fleets have: **sensor readings swing with operating
conditions far more than with wear.** In FD002/FD004, engines fly in six different
regimes, so a reading that looks alarming may just mean the engine is at altitude.

## Results (official test set, last cycle per engine, RUL capped at 125)

| Model | Regime-normalized | Temporal features | CV RMSE | Test RMSE | Test MAE | NASA score |
|---|:-:|:-:|--:|--:|--:|--:|
| Ridge, raw sensors | | | 21.0 | 23.4 | 18.8 | 11,993 |
| LightGBM, raw sensors | | | 16.1 | 16.7 | 12.1 | 4,547 |
| LightGBM | | ✓ | 15.6 | 15.5 | 11.4 | 3,557 |
| LightGBM | ✓ | | 15.9 | 16.8 | 12.2 | 4,652 |
| **LightGBM (production)** | ✓ | ✓ | **10.9** | **11.3** | **7.7** | **1,641** |

Per subset (production model): FD001 10.8 · FD002 11.0 · FD003 11.3 · FD004 11.9.

**The ablation is the finding.** Neither component helps much alone (16.7 → 15.5–16.8),
but together they cut RMSE 32% versus the same model on raw sensors. Temporal features
on raw readings mostly capture regime switching, not wear. Once regime effects are
removed, rolling trends and degradation slopes become informative. The gain is largest
on the multi-regime subsets (FD004: 17.1 → 11.9).

**Uncertainty.** 80% prediction intervals are calibrated from grouped-CV residuals,
binned by predicted RUL (split-conformal style). Test coverage is 77% at a median width
of 17 cycles (11 cycles for engines predicted within 50 cycles of failure). An earlier
version used two quantile-regression models; it covered 84% but at a 57-cycle median
width, with crossing bounds.

**Alerting** (alert when predicted RUL < 30), evaluated on out-of-fold run-to-failure
trajectories across all 709 training engines: 0% of engines reached failure without an
alert, 0.8% alerted more than 60 cycles early, and the median lead time was 30 cycles.
The API requires 3 consecutive low-RUL cycles before alerting. On this data that left
the premature rate unchanged and cost 4 cycles of median lead time. It's kept as a
guard against single-cycle sensor glitches, which the benchmark doesn't contain.

## Design

```
raw cycles ──► regime clustering (KMeans on 3 operating settings, 6 regimes)
           ──► per-regime sensor standardization; drop sensors with no signal (21 → 17)
           ──► per-engine backward-looking features: rolling mean/std (5, 15),
               15-cycle least-squares slope, expanding mean  (120 features)
           ──► LightGBM regressor on capped RUL
           ──► binned residual intervals + persistence-gated alert
```

- **No leakage.** Cross-validation is grouped by engine, and the feature pipeline is re-fit
  inside every fold. Tests assert that features are causal: truncating an engine's future
  doesn't change its past features. They also assert that engines never mix.
- **Train/serve parity.** The API runs the exact fitted pipeline object on the posted
  history, so features at serve time match training by construction.
- **Drift monitoring that doesn't false-alarm on wear.** PSI over all incoming features
  flagged 37 of 120 as shifted on the clean test set (11 significant). That "drift" was really a fleet of
  younger engines. Comparing only each engine's healthy window (first 30 cycles) of
  regime-normalized sensors removes the confound: 0 of 17 sensors flagged on clean data
  (max PSI 0.009). A simulated 1-σ calibration offset on one sensor is caught with PSI
  1.19, and only that sensor is flagged.
- **Experiment tracking.** Every variant logs params and metrics to MLflow.
- **CI.** GitHub Actions runs lint, trains, applies a quality gate that fails the build
  if test RMSE regresses past 13, runs the tests, and builds the serving image.

## Run it

```bash
make setup          # install deps
make train          # download data, train all variants (~7 min), log to MLflow
make test           # 9 tests incl. causality + API contract
make monitor        # drift report, clean and with a simulated sensor offset
make serve          # API at http://localhost:8000/docs
make docker         # build + run the serving image
make mlflow         # compare runs in the MLflow UI
```

```bash
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "engine_id": "E1",
  "cycles": [{"cycle": 1, "settings": [0.0, 0.0, 100.0], "sensors": [/* 21 readings */]}, ...]
}'
# FD004 test engine 1 (true RUL 22):
# → {"rul": 24.4, "rul_p10": 18.6, "rul_p90": 30.0, "alert": true, ...}
```

## Layout

```
src/rul/      config, data, features, metrics, intervals, train, monitor, serve
tests/        pytest suite
scripts/      dataset download
reports/      results.json, feature importance, drift reports (generated)
```

Data: Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure
Simulation," PHM 2008 (NASA Prognostics Center of Excellence).
