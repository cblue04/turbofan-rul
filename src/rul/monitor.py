"""Sensor-drift monitoring with the Population Stability Index (PSI).

Naively comparing all incoming features to training data flags "drift" that is really
just a different mix of engine ages (a fleet of mostly-new engines looks shifted versus
run-to-failure training data). To isolate sensor/calibration drift from wear, we compare
only each engine's *healthy window* (first HEALTHY_CYCLES cycles), using regime-normalized
instantaneous sensor readings.

PSI < 0.1: stable, 0.1-0.25: moderate shift, > 0.25: significant shift (investigate/retrain).

Usage: python -m rul.monitor [--shift SENSOR]   (simulate a +1 std calibration offset)
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR, REPORT_DIR
from .data import load_test, load_train

BINS = 10
EPS = 1e-4
HEALTHY_CYCLES = 30


def psi(reference: np.ndarray, current: np.ndarray, bins: int = BINS) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def healthy_window(fp, df: pd.DataFrame) -> pd.DataFrame:
    return fp.normalized_sensors(df[df["cycle"] <= HEALTHY_CYCLES])


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    rows = [{"sensor": c, "psi": psi(reference[c].to_numpy(), current[c].to_numpy())}
            for c in reference.columns]
    df = pd.DataFrame(rows).sort_values("psi", ascending=False)
    df["status"] = pd.cut(df["psi"], [-np.inf, 0.1, 0.25, np.inf],
                          labels=["stable", "moderate", "significant"]).astype(str)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shift", metavar="SENSOR", help="simulate a +1 std offset on SENSOR, e.g. s11")
    args = ap.parse_args()

    fp = joblib.load(MODEL_DIR / "rul_model.joblib")["pipeline"]
    reference = healthy_window(fp, load_train())
    batch = load_test()
    if args.shift:
        batch[args.shift] = batch[args.shift] + batch.groupby(fp.regimes(batch))[args.shift].transform("std")
    report = drift_report(reference, healthy_window(fp, batch))
    summary = {
        "sensors_checked": len(report),
        "significant": report.loc[report.status == "significant", "sensor"].tolist(),
        "moderate": report.loc[report.status == "moderate", "sensor"].tolist(),
        "max_psi": round(float(report["psi"].max()), 3),
        "investigate": bool((report.status == "significant").any()),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    name = f"drift_shift_{args.shift}" if args.shift else "drift"
    report.to_csv(REPORT_DIR / f"{name}.csv", index=False)
    (REPORT_DIR / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
