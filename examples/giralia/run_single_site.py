"""Run a single Giralia site for 3 months and produce diagnostic plots.

Usage: python mardie/scripts/run_single_site.py [site_id] [months]
Default: site 1 (mid-intertidal, 32% wet), 3 months from 2016-07-01.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from giralia_config import get_soil_params, load_hydro, load_weather
from intertidal_soil.drivers import GiraliaSoilColumn

SITE = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MONTHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
START = sys.argv[3] if len(sys.argv) > 3 else "2017-01-01"
DT = 3600.0  # 1 hour timestep (matches data resolution)

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def main():
    print(f"=== Giralia single-site test: Point {SITE}, {MONTHS} months from {START} ===")

    # --- Load data ---
    hydro = load_hydro(SITE)
    weather = load_weather()

    end = pd.Timestamp(START) + pd.DateOffset(months=MONTHS)
    hydro = hydro[START:str(end)]
    weather = weather[START:str(end)]

    # Align on common hourly index
    idx = hydro.index.intersection(weather.index)
    hydro = hydro.loc[idx]
    weather = weather.loc[idx]
    n_steps = len(idx)
    print(f"  {n_steps} hourly steps ({idx[0]} to {idx[-1]})")

    # --- Soil column setup ---
    soil_kw = get_soil_params(SITE)
    print(f"  Soil: f_macro={soil_kw.get('f_macro', 0):.3f}, "
          f"K_s={soil_kw['K_s']:.1e}, b={soil_kw['b']}")
    col = GiraliaSoilColumn(SITE, soil_kw, dt=DT, wt_depth=0.8)
    col.initialise(air_temp=25.0, deep_temp=26.0,
                   init_moisture=0.12, spin_up_days=10.0)

    # --- Run ---
    times = []
    temp_profiles = []
    moist_profiles = []
    wt_depths = []
    surf_temps = []
    evaps = []
    submerged = []
    G_arr = []
    sw_arr = []

    print("  Running ...", end=" ", flush=True)
    for i in range(n_steps):
        wd = hydro["water_depth"].iloc[i]
        wt = hydro["sea_tem"].iloc[i]
        ta = weather["temperature_2m"].iloc[i]
        sw = weather["shortwave_radiation"].iloc[i]
        rh = weather["relative_humidity_2m"].iloc[i]
        wind = weather["wind_speed_10m"].iloc[i] / 3.6  # km/h -> m/s
        rain_mm = weather["precipitation"].iloc[i]
        rain_m_day = rain_mm / 1000.0 * 24.0  # mm/hr -> m/day
        cc = weather["cloud_cover"].iloc[i] / 100.0

        state, diag = col.step(wd, wt, ta, sw, rh, wind, rain_m_day, cc)

        times.append(idx[i])
        temp_profiles.append(state.temperature.copy())
        moist_profiles.append(state.moisture.copy())
        wt_depths.append(diag.get("water_table_depth", np.nan))
        surf_temps.append(diag.get("surface_temp", ta))
        evaps.append(diag.get("evap", 0.0))
        submerged.append(diag.get("is_submerged", False))
        G_arr.append(diag.get("heatflux", 0.0))
        sw_arr.append(sw)

    print("done.")

    times = np.array(times, dtype="datetime64[ns]")
    temp_arr = np.array(temp_profiles)
    moist_arr = np.array(moist_profiles)
    wt_arr = np.array(wt_depths)
    surf_t = np.array(surf_temps)
    evap_arr = np.array(evaps)
    subm_arr = np.array(submerged)
    G_arr = np.array(G_arr)
    sw_arr = np.array(sw_arr)

    depths_cm = col.params.layer_centres() * 100.0
    days = (times - times[0]).astype("timedelta64[h]").astype(float) / 24.0

    wet_frac = subm_arr.sum() / len(subm_arr) * 100
    cum_evap = evap_arr.sum() * 1000
    print(f"  Wet fraction: {wet_frac:.1f}%")
    print(f"  Cumulative ET: {cum_evap:.1f} mm")
    print(f"  WT: {wt_arr[0]*100:.1f} -> {wt_arr[-1]*100:.1f} cm")
    print(f"  Surface T range: {surf_t.min():.1f} to {surf_t.max():.1f} degC")

    # --- Plot ---
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14, 18))
    gs = GridSpec(5, 2, figure=fig,
                  width_ratios=[1, 0.02],
                  height_ratios=[0.3, 0.5, 1.0, 1.0, 0.4],
                  hspace=0.30, wspace=0.03)

    ax0 = fig.add_subplot(gs[0, 0])
    axes = [ax0] + [fig.add_subplot(gs[r, 0], sharex=ax0) for r in range(1, 5)]

    xlim = (days[0], days[-1])

    # Row 0: submergence + water depth
    ax = axes[0]
    ax.fill_between(days, 0, hydro["water_depth"].values,
                    alpha=0.4, color="blue", label="Water depth")
    ax.set_ylabel("Water depth (m)")
    ax.set_xlim(xlim)
    ax.set_title(f"Giralia Point {SITE} — {MONTHS}-month test from {START}",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # Row 1: surface temperature + air temp
    ax = axes[1]
    ax.plot(days, weather["temperature_2m"].values, "grey", lw=0.5,
            alpha=0.6, label="Air temp")
    ax.plot(days, surf_t, "r-", lw=0.5, label="Surface temp")
    ax.set_ylabel("Temperature (degC)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    # Row 2: temperature heatmap
    ax = axes[2]
    im_t = ax.pcolormesh(days, depths_cm, temp_arr.T, shading="auto",
                         cmap="RdYlBu_r", vmin=15, vmax=50)
    ax.set_ylim(50, 0)
    ax.set_ylabel("Depth (cm)")
    ax.set_title("Soil temperature", fontsize=10)
    cax_t = fig.add_subplot(gs[2, 1])
    fig.colorbar(im_t, cax=cax_t, label="degC")

    # Row 3: moisture heatmap + WT
    ax = axes[3]
    im_m = ax.pcolormesh(days, depths_cm, moist_arr.T, shading="auto",
                         cmap="YlGnBu", vmin=0.05,
                         vmax=col.params.porosity)
    if not np.all(np.isnan(wt_arr)):
        ax.plot(days, wt_arr * 100, "r-", lw=1.5, label="Water table")
        ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(50, 0)
    ax.set_ylabel("Depth (cm)")
    ax.set_title("Soil moisture (VWC)", fontsize=10)
    cax_m = fig.add_subplot(gs[3, 1])
    fig.colorbar(im_m, cax=cax_m, label="m3/m3")

    # Row 4: ground heat flux
    ax = axes[4]
    ax.plot(days, G_arr, "orange", lw=0.5, label="G (into soil)")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("W/m2")
    ax.set_xlabel("Days")
    ax.set_title("Ground heat flux", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Hide x-tick labels on all but the bottom panel
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)

    outfile = os.path.join(PLOT_DIR, f"giralia_point{SITE}_{MONTHS}mo.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {outfile}")


if __name__ == "__main__":
    main()
