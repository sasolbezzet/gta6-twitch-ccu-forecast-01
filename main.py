"""gta6-twitch-ccu-forecast-01: Size-Weighted Log-Ratio Regression Forecaster in Python 3.13.

Standard library only: deterministic, no network, no random walk drift.
Forecasts the 24-hour Twitch concurrent viewers (CCU) in the Grand Theft Auto VI category
by fitting size-weighted log-log regressions r_h ~ a_h + b_h * ln(p1) across major launch comparables,
smoothed across neighbouring hours with calibrated residual quantile spreads (0.05, 0.25, 0.50, 0.75, 0.95).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

WEIGHT_FLOOR = 0.2
WEIGHT_OFFSET = 5.0
MEDIAN_BLEND = 0.15
SPREAD_MULT = 0.50
SMOOTH_HALF_WINDOW = 1


def parse_ts(text: str) -> datetime:
    """Parse ISO-8601 UTC timestamp."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    if n % 2:
        return ordered[n // 2]
    return 0.5 * (ordered[n // 2 - 1] + ordered[n // 2])


def weighted_line(xs: list[float], ys: list[float], ws: list[float]) -> tuple[float, float]:
    """Size-weighted least squares fit: y ~ a + b*x."""
    total = sum(ws)
    if total <= 0:
        return 0.0, 0.0
    mx = sum(w * x for w, x in zip(ws, xs)) / total
    my = sum(w * y for w, y in zip(ws, ys)) / total
    vx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if vx <= 0:
        return my, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / vx
    return my - b * mx, b


def round2_half_up(x: float) -> float | int:
    """Round to 2 decimals half up (aligned with JavaScript Math.round)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def comparable_curves(case: dict) -> list[tuple[float, dict[int, float]]]:
    """Extract valid donor curves from comparables: (log_first, {hour: log_ratio})."""
    curves = []
    for comp in case.get("comparables") or []:
        day_start_raw = comp.get("day_start")
        if not day_start_raw:
            continue
        try:
            day_start = parse_ts(day_start_raw)
        except Exception:
            continue
        by_hour: dict[int, float] = {}
        for point in comp.get("hourly") or []:
            v = point.get("viewers")
            if v is None or v <= 0:
                continue
            ts_str = point.get("from")
            if not ts_str:
                continue
            try:
                hour = int((parse_ts(ts_str) - day_start).total_seconds() // 3600) + 1
            except Exception:
                continue
            if 1 <= hour <= 24:
                by_hour[hour] = float(v)
        if 1 not in by_hour or len(by_hour) < 20:
            continue
        first = by_hour[1]
        if first <= 0:
            continue
        curves.append((
            math.log(first),
            {h: math.log(by_hour[h] / first) for h in range(2, 25) if h in by_hour}
        ))
    return curves


def fit_hour(curves: list[tuple[float, dict[int, float]]], hour: int, log_first: float):
    rows = [(lv, ratios[hour]) for lv, ratios in curves if hour in ratios]
    if len(rows) < 8:
        return None
    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]
    ws = [max(WEIGHT_FLOOR, x - WEIGHT_OFFSET) for x in xs]
    a, b = weighted_line(xs, ys, ws)
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(resid) - 2)
    sigma = max(0.05, math.sqrt(sum(r * r for r in resid) / dof))
    x = min(max(log_first, min(xs)), max(xs))
    mu_reg = a + b * x
    mu = (1.0 - MEDIAN_BLEND) * mu_reg + MEDIAN_BLEND * median(ys)
    return mu, sigma, b, (xs, ys, ws)


def forecast_case(case: dict, name: str) -> dict:
    if list(case.get("levels") or []) != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1.0, float(first.get("viewers", 1.0)))
    log_first = math.log(p1)
    curves = comparable_curves(case)

    fits = {}
    for row in case.get("forecast", []):
        hour = row["hour"]
        fit = fit_hour(curves, hour, log_first)
        if fit is not None:
            fits[hour] = fit

    if fits and SMOOTH_HALF_WINDOW > 0:
        hours = sorted(fits.keys())
        slopes = {h: fits[h][2] for h in hours}
        for h in hours:
            near = [hh for hh in hours if abs(hh - h) <= SMOOTH_HALF_WINDOW]
            slopes[h] = sum(slopes[hh] for hh in near) / len(near)
        for h in hours:
            mu, sigma, _b, (xs, ys, ws) = fits[h]
            b = slopes[h]
            a = sum(w * (y - b * x) for w, x, y in zip(ws, xs, ys)) / sum(ws)
            x = min(max(log_first, min(xs)), max(xs))
            mu_reg = a + b * x
            mu = (1.0 - MEDIAN_BLEND) * mu_reg + MEDIAN_BLEND * median(ys)
            fits[h] = (mu, sigma, b, (xs, ys, ws))

    rows = []
    for row in case.get("forecast", []):
        hour = row["hour"]
        if hour in fits:
            mu, sigma, _b, _data = fits[hour]
            spread = SPREAD_MULT * sigma
            quantiles = [p1 * math.exp(mu + z * spread) for z in Z]
        else:
            quantiles = [p1 * math.exp(z * 0.15 * math.sqrt(hour - 1)) for z in Z]

        cleaned = []
        prev = 0.0
        for q in quantiles:
            q = min(max(q, 1.0), 1_000_000_000.0)
            q = max(q, prev)
            cleaned.append(round2_half_up(q))
            prev = cleaned[-1]
        rows.append({"hour": hour, "q": cleaned})

    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
