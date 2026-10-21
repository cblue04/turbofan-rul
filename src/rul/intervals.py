"""Prediction intervals from out-of-fold residuals, binned by predicted RUL.

Error grows with distance from failure, so one global width would be too wide near
failure and too narrow early in life. We take the empirical residual quantiles of
grouped-CV predictions within each predicted-RUL bin (split-conformal style). Intervals
are ordered by construction, unlike two independently trained quantile models.
"""
from __future__ import annotations

import numpy as np

EDGES = np.array([0, 15, 30, 50, 75, 100, 115, np.inf])


def fit_interval_table(y_true, y_pred, coverage: float = 0.8) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    resid = y_true - y_pred
    a = (1 - coverage) / 2
    bins = np.clip(np.digitize(y_pred, EDGES) - 1, 0, len(EDGES) - 2)
    lo, hi = [], []
    for b in range(len(EDGES) - 1):
        r = resid[bins == b]
        lo.append(np.quantile(r, a) if len(r) else np.quantile(resid, a))
        hi.append(np.quantile(r, 1 - a) if len(r) else np.quantile(resid, 1 - a))
    return {"edges": EDGES, "lo": np.minimum(lo, 0), "hi": np.maximum(hi, 0), "coverage": coverage}


def predict_interval(table: dict, y_pred, cap: float):
    y_pred = np.asarray(y_pred, float)
    bins = np.clip(np.digitize(y_pred, table["edges"]) - 1, 0, len(table["edges"]) - 2)
    lo = np.clip(y_pred + table["lo"][bins], 0, cap)
    hi = np.clip(y_pred + table["hi"][bins], 0, cap)
    return lo, hi
