import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rul.config import MODEL_DIR, SENSOR_COLS, SETTING_COLS  # noqa: E402
from rul.data import load_test, load_train  # noqa: E402
from rul.features import FeaturePipeline  # noqa: E402
from rul.metrics import alert_quality, first_alerts, nasa_score, rmse  # noqa: E402
from rul.monitor import psi  # noqa: E402


@pytest.fixture(scope="module")
def train_small():
    return load_train(["FD001", "FD002"])


def test_train_labels_capped_and_end_at_zero(train_small):
    assert train_small["rul"].max() == 125
    assert (train_small.groupby("engine_id")["rul"].last() == 0).all()


def test_test_set_has_one_last_row_per_engine():
    te = load_test(["FD001"])
    assert te["is_last"].sum() == te["engine_id"].nunique() == 100


def test_features_have_no_nans(train_small):
    fp = FeaturePipeline().fit(train_small)
    assert not fp.transform(train_small).isna().any().any()


def test_features_are_causal(train_small):
    """Features at cycle t must not change when future cycles are removed (no look-ahead)."""
    fp = FeaturePipeline().fit(train_small)
    eng = train_small[train_small["engine_id"] == "FD002_7"].reset_index(drop=True)
    full = fp.transform(eng)
    truncated = fp.transform(eng.iloc[:60])
    pd.testing.assert_frame_equal(full.iloc[:60], truncated, check_exact=False, atol=1e-8)


def test_features_do_not_mix_engines(train_small):
    fp = FeaturePipeline().fit(train_small)
    a = train_small[train_small["engine_id"] == "FD001_1"]
    both = train_small[train_small["engine_id"].isin(["FD001_1", "FD001_2"])]
    pd.testing.assert_frame_equal(fp.transform(a), fp.transform(both).loc[a.index],
                                  check_exact=False, atol=1e-8)


def test_metrics():
    assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt(12.5))
    # Late predictions cost more than early ones of the same size.
    assert nasa_score([50], [60]) > nasa_score([50], [40])


def test_alert_persistence():
    ids = ["a"] * 7
    true = [60, 50, 40, 30, 20, 10, 0]
    pred = [50, 20, 45, 25, 20, 15, 5]  # one noisy dip at index 1
    assert first_alerts(ids, true, pred, 30, persistence=1)[0] == 50
    assert first_alerts(ids, true, pred, 30, persistence=3)[0] == 10
    q = alert_quality(np.array([np.nan, 100, 20]), horizon=30)
    assert q["missed_rate"] == pytest.approx(1 / 3) and q["premature_rate"] == pytest.approx(1 / 3)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi(ref, rng.normal(size=5000)) < 0.1
    assert psi(ref, rng.normal(loc=1.0, size=5000)) > 0.25


@pytest.mark.skipif(not (MODEL_DIR / "rul_model.joblib").exists(), reason="model not trained")
def test_api_predict():
    from fastapi.testclient import TestClient

    from rul.serve import app

    eng = load_test(["FD001"]).query("engine_id == 'FD001_1'")
    payload = {"engine_id": "FD001_1", "cycles": [
        {"cycle": int(r.cycle), "settings": [float(r[c]) for c in SETTING_COLS],
         "sensors": [float(r[c]) for c in SENSOR_COLS]} for _, r in eng.iterrows()]}
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    out = client.post("/predict", json=payload).json()
    assert 0 <= out["rul_p10"] <= out["rul"] <= out["rul_p90"] <= 125
    assert out["last_cycle"] == int(eng.cycle.max())

    bad = dict(payload, cycles=payload["cycles"][::-1])
    assert client.post("/predict", json=bad).status_code == 422
