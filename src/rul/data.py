import numpy as np
import pandas as pd

from .config import COLUMNS, RAW_DIR, RUL_CAP, SUBSETS


def _read(path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def load_train(subsets=SUBSETS) -> pd.DataFrame:
    """Run-to-failure trajectories with a capped RUL label for every cycle."""
    frames = []
    for sub in subsets:
        df = _read(RAW_DIR / f"train_{sub}.txt")
        df.insert(0, "subset", sub)
        last = df.groupby("unit")["cycle"].transform("max")
        df["rul"] = np.minimum(last - df["cycle"], RUL_CAP)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["engine_id"] = out["subset"] + "_" + out["unit"].astype(str)
    return out


def load_test(subsets=SUBSETS) -> pd.DataFrame:
    """Truncated trajectories; true RUL is only known at each engine's last cycle."""
    frames = []
    for sub in subsets:
        df = _read(RAW_DIR / f"test_{sub}.txt")
        df.insert(0, "subset", sub)
        true_rul = pd.read_csv(RAW_DIR / f"RUL_{sub}.txt", header=None).iloc[:, 0].to_numpy()
        last = df.groupby("unit")["cycle"].transform("max")
        rul_at_end = df["unit"].map(dict(zip(range(1, len(true_rul) + 1), true_rul)))
        df["rul"] = np.minimum(rul_at_end + (last - df["cycle"]), RUL_CAP)
        df["is_last"] = df["cycle"] == last
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["engine_id"] = out["subset"] + "_" + out["unit"].astype(str)
    return out
