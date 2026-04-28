"""Site configuration for Giralia intertidal soil modelling.

Defines soil parameters, macropore gradient, and per-site metadata
for the 12 monitoring points across the Giralia tidal flat.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HYDRO_DIR = os.path.join(os.path.dirname(__file__), "hydro")

# --- Base soil parameters (sandy-clay tidal flat) --------------------------
BASE_SOIL = dict(
    n_layers=20,
    max_depth=1.0,
    psi_e=-1.5,          # J/kg, sandy-clay air entry
    K_s=2e-4,            # kg s/m^3, moderate-low matrix conductivity
    b=4.0,               # sandy-clay retention shape
    bulk_density=1.55,    # Mg/m^3, compacted tidal flat
    Sy=0.15,             # specific yield
    deep_temp=26.0,       # degC, stable deep temperature (tropical)
    lai=0.1,              # sparse mangrove fringe
)

# --- Macropore gradient by inundation category -----------------------------
MACRO_PARAMS = {
    "subtidal":       dict(f_macro=0.030, k_macro=0.010, alpha_exchange=5e-5),
    "low_intertidal": dict(f_macro=0.025, k_macro=0.010, alpha_exchange=5e-5),
    "mid_intertidal": dict(f_macro=0.015, k_macro=0.008, alpha_exchange=4e-5),
    "high_intertidal": dict(f_macro=0.008, k_macro=0.005, alpha_exchange=3e-5),
    "supratidal":     dict(f_macro=0.002, k_macro=0.002, alpha_exchange=1e-5),
}

# --- Per-site definitions --------------------------------------------------
SITES = {
    1:  dict(category="mid_intertidal"),
    2:  dict(category="supratidal"),
    3:  dict(category="mid_intertidal"),
    4:  dict(category="subtidal"),
    5:  dict(category="low_intertidal"),
    6:  dict(category="high_intertidal"),
    7:  dict(category="low_intertidal"),
    8:  dict(category="supratidal"),
    9:  dict(category="supratidal"),
    10: dict(category="supratidal"),
    11: dict(category="supratidal"),
    12: dict(category="subtidal"),
}


def get_soil_params(site_id):
    """Return a dict of SoilParams kwargs for a given site."""
    cat = SITES[site_id]["category"]
    macro = MACRO_PARAMS[cat]
    return {**BASE_SOIL, **macro}


def load_hydro(site_id):
    """Load the hydro forcing for a site as a DataFrame.

    Columns: water_depth (m), sea_tem (degC).
    Index: DatetimeIndex (hourly, 2016-01-01 to 2020-12-31).
    """
    path = os.path.join(HYDRO_DIR, f"hydro_info_at_Point{site_id}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.columns = [c.strip() for c in df.columns]
    out = pd.DataFrame(index=df.index)
    out["water_depth"] = df.iloc[:, 0].values  # water_depth (m)
    out["sea_tem"] = df.iloc[:, 2].values      # sea_tem
    return out


def load_weather():
    """Load the combined Giralia weather data.

    Returns DataFrame with columns: temperature_2m, relative_humidity_2m,
    wind_speed_10m, shortwave_radiation, precipitation, cloud_cover, etc.
    Index: DatetimeIndex (hourly UTC).
    """
    path = os.path.join(DATA_DIR, "giralia_weather.csv")
    df = pd.read_csv(path, parse_dates=["time"])
    df = df.set_index("time")
    df.index = df.index.tz_localize(None)
    return df
