"""Fetch historical hourly weather data for Giralia from Open-Meteo archive API.

Saves to mardie/data/giralia_weather_YYYY.csv (one file per year) and a
combined giralia_weather.csv spanning the full hydro-data period (2016-2020).
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import openmeteo_requests

LAT = -22.1
LON = 114.6
START = "2016-01-01"
END = "2020-12-31"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "precipitation",
    "dew_point_2m",
    "cloud_cover",
]

URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_year(client, year):
    """Fetch one calendar year of hourly data."""
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": HOURLY_VARS,
        "timezone": "UTC",
    }
    responses = client.weather_api(URL, params=params)
    r = responses[0]
    hourly = r.Hourly()
    t0 = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    t1 = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    times = pd.date_range(start=t0, end=t1,
                          freq=f"{hourly.Interval()}s", inclusive="left")

    data = {"time": times}
    for i, name in enumerate(HOURLY_VARS):
        data[name] = hourly.Variables(i).ValuesAsNumpy()
    return pd.DataFrame(data)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    client = openmeteo_requests.Client()

    frames = []
    for year in range(2016, 2021):
        print(f"  Fetching {year} ...", end=" ", flush=True)
        df = fetch_year(client, year)
        outf = os.path.join(OUT_DIR, f"giralia_weather_{year}.csv")
        df.to_csv(outf, index=False)
        print(f"{len(df)} rows -> {os.path.basename(outf)}")
        frames.append(df)
        time.sleep(1.0)

    combined = pd.concat(frames, ignore_index=True)
    combined_path = os.path.join(OUT_DIR, "giralia_weather.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\nCombined: {len(combined)} rows -> {os.path.basename(combined_path)}")

    print(f"\nSummary ({START} to {END}):")
    for col in HOURLY_VARS:
        vals = combined[col].dropna()
        print(f"  {col:30s}: mean={vals.mean():8.2f}  min={vals.min():8.2f}  max={vals.max():8.2f}")


if __name__ == "__main__":
    main()
