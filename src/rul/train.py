"""Train and compare RUL models, log everything to MLflow, save the best model for serving.

Usage: python -m rul.train  (from src/, or `make train`)
"""
from __future__ import annotations

import json
import time

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import ALERT_HORIZON, MODEL_DIR, RANDOM_STATE, REPORT_DIR, ROOT, RUL_CAP
from .data import load_test, load_train
from .features import FeaturePipeline
from .intervals import fit_interval_table, predict_interval
from .metrics import alert_quality, first_alerts, mae, nasa_score, rmse

LGB_PARAMS = dict(
    n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
    random_state=RANDOM_STATE, verbose=-1,
)

EXPERIMENTS = {
    # name: (model factory, regime_normalize, temporal)
    "ridge_raw": (lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0)), False, False),
    "lgbm_raw": (lambda: lgb.LGBMRegressor(**LGB_PARAMS), False, False),
    "lgbm_temporal_only": (lambda: lgb.LGBMRegressor(**LGB_PARAMS), False, True),
    "lgbm_regime_only": (lambda: lgb.LGBMRegressor(**LGB_PARAMS), True, False),
    "lgbm_full": (lambda: lgb.LGBMRegressor(**LGB_PARAMS), True, True),
}


def clip(p):
    return np.clip(p, 0, RUL_CAP)


def cross_validate(train: pd.DataFrame, factory, regime: bool, temporal: bool, n_splits=5):
    """Grouped CV by engine (no engine appears in both train and validation folds).

    The feature pipeline is re-fit inside each fold so normalization stats never see
    validation engines. Returns out-of-fold predictions aligned with `train`.
    """
    oof = np.zeros(len(train))
    for tr_idx, va_idx in GroupKFold(n_splits=n_splits).split(train, groups=train["engine_id"]):
        tr, va = train.iloc[tr_idx], train.iloc[va_idx]
        fp = FeaturePipeline(regime, temporal).fit(tr)
        model = factory().fit(fp.transform(tr), tr["rul"])
        oof[va_idx] = clip(model.predict(fp.transform(va)))
    return oof


def evaluate_test(test: pd.DataFrame, preds: np.ndarray) -> dict:
    last = test["is_last"].to_numpy()
    out = {
        "test_rmse": rmse(test["rul"][last], preds[last]),
        "test_mae": mae(test["rul"][last], preds[last]),
        "test_nasa_score": nasa_score(test["rul"][last], preds[last]),
    }
    for sub, d in test[last].assign(p=preds[last]).groupby("subset"):
        out[f"test_rmse_{sub}"] = rmse(d["rul"], d["p"])
    return out


def main():
    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    mlflow.set_experiment("turbofan-rul")

    train, test = load_train(), load_test()
    results = {}
    for name, (factory, regime, temporal) in EXPERIMENTS.items():
        t0 = time.time()
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "regime_normalize": regime, "temporal": temporal,
                               "rul_cap": RUL_CAP, **(LGB_PARAMS if "lgbm" in name else {})})
            oof = cross_validate(train, factory, regime, temporal)
            fp = FeaturePipeline(regime, temporal).fit(train)
            model = factory().fit(fp.transform(train), train["rul"])
            test_pred = clip(model.predict(fp.transform(test)))

            m = {"cv_rmse": rmse(train["rul"], oof), "cv_mae": mae(train["rul"], oof),
                 **evaluate_test(test, test_pred)}
            for k in (1, 3):
                q = alert_quality(first_alerts(train["engine_id"], train["rul"], oof, ALERT_HORIZON, k),
                                  ALERT_HORIZON)
                m.update({f"alert_p{k}_{kk}": v for kk, v in q.items()})
            m["n_features"] = len(fp.feature_names_)
            m["train_seconds"] = time.time() - t0
            mlflow.log_metrics(m)
            results[name] = m
            print(f"{name:20s} cv_rmse={m['cv_rmse']:.2f} test_rmse={m['test_rmse']:.2f} "
                  f"nasa={m['test_nasa_score']:.0f} ({m['train_seconds']:.0f}s)")
            if name == "lgbm_full":
                best = (fp, model, test_pred, oof)

    # 80% prediction intervals for the production model, calibrated on grouped-CV residuals.
    fp, model, test_pred, oof = best
    table = fit_interval_table(train["rul"], oof, coverage=0.8)
    last = test["is_last"].to_numpy()
    lo, hi = predict_interval(table, test_pred[last], RUL_CAP)
    y = test["rul"][last].to_numpy()
    results["lgbm_full"]["interval_80_coverage"] = float(np.mean((y >= lo) & (y <= hi)))
    results["lgbm_full"]["interval_80_median_width"] = float(np.median(hi - lo))
    near = test_pred[last] < 50
    results["lgbm_full"]["interval_80_median_width_pred_lt50"] = float(np.median((hi - lo)[near]))

    joblib.dump({"pipeline": fp, "model": model, "intervals": table,
                 "rul_cap": RUL_CAP, "alert_horizon": ALERT_HORIZON},
                MODEL_DIR / "rul_model.joblib")
    imp = pd.Series(model.feature_importances_, index=fp.feature_names_).sort_values(ascending=False)
    imp.head(25).to_csv(REPORT_DIR / "feature_importance.csv")
    (REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["lgbm_full"], indent=2))


if __name__ == "__main__":
    main()
