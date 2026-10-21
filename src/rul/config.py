from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

SETTING_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTING_COLS, *SENSOR_COLS]

# Piecewise-linear target: degradation is not observable early in life,
# so RUL is capped (standard practice for C-MAPSS).
RUL_CAP = 125
# Early-warning horizon: alert when predicted RUL falls below this.
ALERT_HORIZON = 30

N_REGIMES = 6
ROLL_WINDOWS = (5, 15)
SLOPE_WINDOW = 15
RANDOM_STATE = 42
