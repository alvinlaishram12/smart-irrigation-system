"""
weather_api.py
--------------
Live weather integration via Open-Meteo (https://open-meteo.com) - a free,
no-API-key weather service. Fetches today's observed-ish conditions and
tomorrow's rainfall forecast, in the same shape the models expect
(temperature_c, humidity_pct, rainfall_mm, rainfall_forecast_next_day_mm).

Why hourly data, aggregated in Python, instead of Open-Meteo's own "daily"
endpoint: Open-Meteo's daily block reliably supports temperature and
precipitation aggregates, but daily-mean relative humidity support has
varied across API versions. Requesting the hourly arrays and aggregating
them ourselves (mean for temperature/humidity, sum for precipitation) only
depends on parameters that have been stable in the API for years.

OFFLINE FALLBACK: if the live request fails for any reason (no internet,
timeout, campus WiFi down mid-viva), this falls back to a bundled sample
response instead of crashing, and clearly flags the result as
`"is_live": False` so calling code (the Day 6 dashboard) can show a warning
badge rather than silently presenting stale data as current.

Usage:
    python src/weather_api.py --lat 30.9010 --lon 75.8573
"""

import argparse
import json
import os
from datetime import datetime, timezone

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "sample_weather_response.json")


class WeatherAPIError(Exception):
    pass


def _aggregate_hourly(times, values, target_date):
    """Mean of hourly `values` whose timestamp falls on `target_date` (YYYY-MM-DD)."""
    day_values = [v for t, v in zip(times, values) if t.startswith(target_date) and v is not None]
    return sum(day_values) / len(day_values) if day_values else None


def _sum_hourly(times, values, target_date):
    day_values = [v for t, v in zip(times, values) if t.startswith(target_date) and v is not None]
    return sum(day_values) if day_values else None


def _parse_response(payload):
    hourly = payload["hourly"]
    times = hourly["time"]
    dates = sorted({t[:10] for t in times})
    today, tomorrow = dates[0], dates[1]

    return {
        "date_today": today,
        "date_tomorrow": tomorrow,
        "temperature_c": round(_aggregate_hourly(times, hourly["temperature_2m"], today), 1),
        "humidity_pct": round(_aggregate_hourly(times, hourly["relative_humidity_2m"], today), 1),
        "rainfall_mm": round(_sum_hourly(times, hourly["precipitation"], today), 1),
        "rainfall_forecast_next_day_mm": round(_sum_hourly(times, hourly["precipitation"], tomorrow), 1),
    }


def fetch_weather(lat, lon, timeout=8):
    """
    Returns a dict with today's temperature/humidity/rainfall and tomorrow's
    rainfall forecast, plus an `is_live` flag. Never raises - falls back to
    bundled sample data (clearly flagged) if the live call fails.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation",
        "forecast_days": 2,
        "timezone": "auto",
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        result = _parse_response(resp.json())
        result["is_live"] = True
        result["source"] = "Open-Meteo (live)"
        return result
    except Exception as exc:
        print(f"[weather_api] Live request failed ({exc}); falling back to offline sample data.")
        return _load_fallback()


def _load_fallback():
    with open(FIXTURE_PATH) as f:
        payload = json.load(f)
    result = _parse_response(payload)
    result["is_live"] = False
    result["source"] = "Offline sample data (NOT live - check internet connection)"
    return result


def main():
    parser = argparse.ArgumentParser(description="Fetch live weather for irrigation decisions.")
    parser.add_argument("--lat", type=float, default=30.9010, help="Latitude (default: Ludhiana, Punjab)")
    parser.add_argument("--lon", type=float, default=75.8573, help="Longitude (default: Ludhiana, Punjab)")
    args = parser.parse_args()

    result = fetch_weather(args.lat, args.lon)
    print(f"\nFetched at {datetime.now(timezone.utc).isoformat()}Z")
    print(f"Source: {result['source']}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
