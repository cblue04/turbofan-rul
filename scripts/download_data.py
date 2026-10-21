"""Download the NASA C-MAPSS turbofan degradation dataset (FD001-FD004).

Source: NASA Prognostics Center of Excellence (Saxena et al., 2008), mirrored on GitHub.
"""
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/hankroark/Turbofan-Engine-Degradation/master/CMAPSSData"
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for sub in SUBSETS:
        for kind in ("train", "test", "RUL"):
            name = f"{kind}_{sub}.txt"
            dest = OUT / name
            if dest.exists():
                continue
            print(f"downloading {name}")
            urllib.request.urlretrieve(f"{BASE}/{name}", dest)
    print(f"data in {OUT}")


if __name__ == "__main__":
    sys.exit(main())
