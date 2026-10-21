"""FastAPI service: POST an engine's cycle history, get RUL with an 80% interval and an alert flag.

Run: uvicorn rul.serve:app --app-dir src --port 8000
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .config import MODEL_DIR, SENSOR_COLS, SETTING_COLS
from .intervals import predict_interval

MODEL_PATH = Path(os.environ.get("RUL_MODEL_PATH", MODEL_DIR / "rul_model.joblib"))
ALERT_PERSISTENCE = 3  # consecutive low-RUL cycles required before alerting


class Cycle(BaseModel):
    cycle: int = Field(ge=1)
    settings: list[float] = Field(min_length=3, max_length=3)
    sensors: list[float] = Field(min_length=21, max_length=21)


class EngineHistory(BaseModel):
    engine_id: str = "engine"
    cycles: list[Cycle] = Field(min_length=1)

    @field_validator("cycles")
    @classmethod
    def strictly_increasing(cls, v):
        c = [x.cycle for x in v]
        if any(b <= a for a, b in zip(c, c[1:])):
            raise ValueError("cycles must be strictly increasing")
        return v


class Prediction(BaseModel):
    engine_id: str
    last_cycle: int
    rul: float
    rul_p10: float
    rul_p90: float
    alert: bool
    model_version: str


@lru_cache
def bundle():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"model not found at {MODEL_PATH}; run `make train` first")
    return joblib.load(MODEL_PATH)


app = FastAPI(title="Turbofan RUL service", version="1.0.0")


@app.get("/health")
def health():
    try:
        bundle()
    except RuntimeError as e:
        raise HTTPException(503, str(e)) from e
    return {"status": "ok"}


def _to_frame(h: EngineHistory) -> pd.DataFrame:
    rows = [{"cycle": c.cycle, **dict(zip(SETTING_COLS, c.settings)), **dict(zip(SENSOR_COLS, c.sensors))}
            for c in h.cycles]
    df = pd.DataFrame(rows)
    df["engine_id"] = h.engine_id
    return df


@app.post("/predict", response_model=Prediction)
def predict(h: EngineHistory):
    b = bundle()
    X = b["pipeline"].transform(_to_frame(h))
    cap = b["rul_cap"]
    rul = np.clip(b["model"].predict(X), 0, cap)
    lo, hi = predict_interval(b["intervals"], rul[-1:], cap)
    recent = rul[-ALERT_PERSISTENCE:]
    alert = len(recent) == ALERT_PERSISTENCE and bool((recent < b["alert_horizon"]).all())
    return Prediction(
        engine_id=h.engine_id, last_cycle=h.cycles[-1].cycle, rul=round(float(rul[-1]), 1),
        rul_p10=round(float(lo[0]), 1), rul_p90=round(float(hi[0]), 1),
        alert=alert, model_version=app.version,
    )
