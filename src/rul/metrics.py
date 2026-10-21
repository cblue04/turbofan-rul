import numpy as np


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def nasa_score(y, p) -> float:
    """PHM08 asymmetric score: late predictions (over-estimating RUL) are penalized harder."""
    d = np.asarray(p) - np.asarray(y)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def first_alerts(engine_ids, rul_true, rul_pred, horizon: int, persistence: int = 1):
    """For each engine, the true RUL at the first cycle an alert fires.

    An alert fires when predicted RUL < horizon for `persistence` consecutive cycles.
    Returns an array (NaN if the engine never alerted). Rows must be in cycle order per engine.
    """
    out = []
    engine_ids = np.asarray(engine_ids)
    for eid in dict.fromkeys(engine_ids):
        m = engine_ids == eid
        below = np.asarray(rul_pred)[m] < horizon
        run = 0
        fired = np.nan
        for i, b in enumerate(below):
            run = run + 1 if b else 0
            if run >= persistence:
                fired = np.asarray(rul_true)[m][i]
                break
        out.append(fired)
    return np.array(out, dtype=float)


def alert_quality(first_alert_rul, horizon: int, early_tolerance: int = 30) -> dict:
    """Summarize run-to-failure alert behaviour.

    - missed: engine reached failure without an alert
    - premature: alert fired while true RUL > horizon + early_tolerance (alarm fatigue)
    - median_lead: median true RUL remaining when the alert fired (for non-premature alerts)
    """
    a = np.asarray(first_alert_rul)
    missed = np.isnan(a)
    premature = ~missed & (a > horizon + early_tolerance)
    good = ~missed & ~premature
    return {
        "missed_rate": float(missed.mean()),
        "premature_rate": float(premature.mean()),
        "median_lead_cycles": float(np.median(a[good])) if good.any() else float("nan"),
    }
