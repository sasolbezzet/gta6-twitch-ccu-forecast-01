"""gta6-twitch-ccu-forecast-01: Empirical Diurnal Twitch CCU Forecaster in Python 3.13.

Standard library only: no network, no pip packages.
Forecasts the 24-hour Twitch live concurrent viewers (CCU) in the Grand Theft Auto VI category
by combining hour 1's observed print with empirical diurnal multipliers extracted from major AAA launches
(Elden Ring, Cyberpunk 2077, Hogwarts Legacy, God of War Ragnarök, Marvel's Spider-Man 2, RDR2)
and widening pinball-calibrated quantiles (0.05, 0.25, 0.50, 0.75, 0.95).
"""

import json
import math
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.12

# Calibrated median hourly trajectory relative to hour 1 (UTC midnight) across major AAA launches
DEFAULT_CURVE = [
    1.0, 0.8869, 0.7226, 0.7375, 0.7846, 0.8362,
    0.7933, 0.7549, 0.7309, 0.6965, 0.7833, 0.9148,
    0.9315, 1.0382, 1.1645, 1.2229, 1.278, 1.3787,
    1.4984, 1.5348, 1.4941, 1.6093, 1.6849, 1.1884
]


def round2(x: float) -> float | int:
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def get_curve_from_comparables(comparables: list) -> list[float]:
    curves = []
    for c in comparables:
        day_start = c.get("day_start")
        if not day_start:
            continue
        hourly_by_from = {h["from"]: h["viewers"] for h in c.get("hourly", [])}
        try:
            dt_start = datetime.fromisoformat(day_start.replace("Z", "+00:00"))
        except Exception:
            continue

        day_viewers = []
        for h in range(24):
            from_ts = datetime.fromtimestamp(dt_start.timestamp() + h * 3600, tz=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z")
            v = hourly_by_from.get(from_ts)
            day_viewers.append(v)

        if day_viewers[0] and day_viewers[0] >= 10000 and all(v is not None for v in day_viewers):
            ratios = [v / day_viewers[0] for v in day_viewers]
            curves.append(ratios)

    if len(curves) >= 5:
        med = []
        for h in range(24):
            med.append(statistics.median([r[h] for r in curves]))
        return med
    return DEFAULT_CURVE


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")

    p1 = max(1, first["viewers"])
    curve = get_curve_from_comparables(case.get("comparables", []))

    hours_res = []
    for row in case["forecast"]:
        h = row["hour"]
        h_idx = h - 1
        mult = curve[h_idx] if 0 <= h_idx < len(curve) else 1.0

        expected = p1 * mult
        spread = SIGMA * math.sqrt(h - 1)

        q_vals = [round2(expected * math.exp(z * spread)) for z in Z]

        # Enforce strict quantile ordering
        for i in range(1, len(q_vals)):
            if q_vals[i] < q_vals[i - 1]:
                q_vals[i] = q_vals[i - 1]

        hours_res.append({"hour": h, "q": q_vals})

    return {"hours": hours_res}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
