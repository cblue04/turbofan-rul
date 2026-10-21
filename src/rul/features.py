"""Feature pipeline: operating-regime normalization + per-engine temporal features.

Sensor readings shift heavily with flight condition (altitude, Mach, throttle), which
masks degradation. We cluster the three operating settings into regimes and standardize
each sensor *within* its regime, so what remains is mostly wear signal. Temporal features
are computed strictly per engine and only look backward, so serving on a partial
history produces exactly the same features as training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .config import N_REGIMES, RANDOM_STATE, ROLL_WINDOWS, SENSOR_COLS, SETTING_COLS, SLOPE_WINDOW


class FeaturePipeline:
    def __init__(self, regime_normalize: bool = True, temporal: bool = True):
        self.regime_normalize = regime_normalize
        self.temporal = temporal

    # ---- fitting -------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        if self.regime_normalize:
            self.kmeans_ = KMeans(n_clusters=N_REGIMES, n_init=10, random_state=RANDOM_STATE)
            regimes = self.kmeans_.fit_predict(df[SETTING_COLS].to_numpy())
        else:
            regimes = np.zeros(len(df), dtype=int)
        grouped = df[SENSOR_COLS].groupby(regimes)
        self.mean_ = grouped.mean()
        self.std_ = grouped.std().replace(0, np.nan)
        normed = self._normalize(df, regimes)
        # Drop sensors that carry no information once regime effects are removed.
        spread = normed.std()
        self.sensors_ = [c for c in SENSOR_COLS if spread[c] > 1e-3]
        self.feature_names_ = self._build(df, regimes).columns.tolist()
        return self

    # ---- transforming --------------------------------------------------------
    def regimes(self, df: pd.DataFrame) -> np.ndarray:
        if self.regime_normalize:
            return self.kmeans_.predict(df[SETTING_COLS].to_numpy())
        return np.zeros(len(df), dtype=int)

    def normalized_sensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regime-normalized instantaneous readings for the informative sensors."""
        return self._normalize(df, self.regimes(df))[self.sensors_]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._build(df, self.regimes(df))
        return X[self.feature_names_]

    def _normalize(self, df: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        mu = self.mean_.reindex(regimes).to_numpy()
        sd = self.std_.reindex(regimes).to_numpy()
        out = (df[SENSOR_COLS].to_numpy() - mu) / sd
        return pd.DataFrame(np.nan_to_num(out), columns=SENSOR_COLS, index=df.index)

    def _build(self, df: pd.DataFrame, regimes: np.ndarray) -> pd.DataFrame:
        sensors = getattr(self, "sensors_", SENSOR_COLS)
        normed = self._normalize(df, regimes)[sensors]
        feats = {"cycle": df["cycle"].to_numpy()}
        for c in sensors:
            feats[c] = normed[c].to_numpy()
        if not self.temporal:
            return pd.DataFrame(feats, index=df.index)

        g_key = df["engine_id"].to_numpy()
        work = normed.copy()
        work["_g"] = g_key
        work["_t"] = df["cycle"].to_numpy().astype(float)
        grp = work.groupby("_g", sort=False)
        for w in ROLL_WINDOWS:
            roll = grp[sensors].rolling(w, min_periods=1)
            m = roll.mean().reset_index(level=0, drop=True).loc[df.index]
            s = roll.std().reset_index(level=0, drop=True).loc[df.index].fillna(0)
            for c in sensors:
                feats[f"{c}_mean{w}"] = m[c].to_numpy()
                feats[f"{c}_std{w}"] = s[c].to_numpy()
        # Rolling least-squares slope of each sensor against cycle: the degradation rate.
        w = SLOPE_WINDOW
        t = work["_t"]
        mt = grp["_t"].rolling(w, min_periods=2).mean().reset_index(level=0, drop=True).loc[df.index]
        mtt = (t * t).groupby(g_key).rolling(w, min_periods=2).mean().reset_index(level=0, drop=True)
        mtt = mtt.loc[df.index]
        var_t = mtt - mt**2
        for c in sensors:
            x = work[c]
            mx = grp[c].rolling(w, min_periods=2).mean().reset_index(level=0, drop=True).loc[df.index]
            mtx = (t * x).groupby(g_key).rolling(w, min_periods=2).mean()
            mtx = mtx.reset_index(level=0, drop=True).loc[df.index]
            slope = (mtx - mt * mx) / var_t.replace(0, np.nan)
            feats[f"{c}_slope{w}"] = slope.fillna(0).to_numpy()
        # Cumulative drift since the engine's first observed cycles (health-index style).
        for c in sensors:
            cm = grp[c].expanding().mean().reset_index(level=0, drop=True)
            feats[f"{c}_cummean"] = cm.loc[df.index].to_numpy()
        return pd.DataFrame(feats, index=df.index)
