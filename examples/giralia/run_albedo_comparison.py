"""Compare bare mudflat (albedo=0.25) vs dark algal mat (albedo=0.10).

Usage: python mardie/scripts/run_albedo_comparison.py [site_id] [months] [start]
Default: site 6 (high intertidal), 3 months from 2017-01-01.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


from giralia_config import SITES, get_soil_params, load_hydro, load_weather
from intertidal_soil.drivers import GiraliaSoilColumn

SITE = int(sys.argv[1]) if len(sys.argv) > 1 else 6
MONTHS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
START = sys.argv[3] if len(sys.argv) > 3 else "2017-01-01"
DT = 3600.0

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

ALBEDOS = {"Bare mudflat (a=0.25)": 0.25, "Dark algal mat (a=0.10)": 0.10}


def run_with_albedo(site_id, albedo, hydro, weather, idx, n_steps):
    soil_kw = get_soil_params(site_id)
    col = GiraliaSoilColumn(site_id, soil_kw, dt=DT, wt_depth=0.8)
    col.initialise(air_temp=25.0, deep_temp=26.0,
                   init_moisture=0.12, spin_up_days=10.0)

    surf_temps = np.full(n_steps, np.nan)
    evaps = np.full(n_steps, np.nan)
    wt_depths = np.full(n_steps, np.nan)
    submerged = np.zeros(n_steps, dtype=bool)
    temp_profiles = []
    moist_profiles = []
    G_arr = np.full(n_steps, np.nan)
    H_arr = np.full(n_steps, np.nan)
    LE_arr = np.full(n_steps, np.nan)
    sw_net_arr = np.full(n_steps, np.nan)

    for i in range(n_steps):
        wd = hydro["water_depth"].iloc[i]
        wt = hydro["sea_tem"].iloc[i]
        ta = weather["temperature_2m"].iloc[i]
        sw = weather["shortwave_radiation"].iloc[i]
        rh = weather["relative_humidity_2m"].iloc[i]
        wind = weather["wind_speed_10m"].iloc[i] / 3.6
        rain_mm = weather["precipitation"].iloc[i]
        rain_m_day = rain_mm / 1000.0 * 24.0
        cc = weather["cloud_cover"].iloc[i] / 100.0

        state, diag = col.step(wd, wt, ta, sw, rh, wind, rain_m_day, cc,
                               albedo=albedo)

        surf_temps[i] = diag.get("surface_temp", ta)
        evaps[i] = diag.get("evap", 0.0)
        wt_depths[i] = diag.get("water_table_depth", np.nan)
        submerged[i] = diag.get("is_submerged", False)
        temp_profiles.append(state.temperature.copy())
        moist_profiles.append(state.moisture.copy())
        G_arr[i] = diag.get("heatflux", 0.0)
        H_arr[i] = diag.get("H", 0.0)
        LE_arr[i] = diag.get("LE", 0.0)
        sw_net_arr[i] = diag.get("sw_net", 0.0)

    return dict(
        col=col, surf_temps=surf_temps, evaps=evaps, wt_depths=wt_depths,
        submerged=submerged, temp_profiles=np.array(temp_profiles),
        moist_profiles=np.array(moist_profiles),
        G=G_arr, H=H_arr, LE=LE_arr, sw_net=sw_net_arr,
    )


def main():
    cat = SITES[SITE]["category"]
    print(f"=== Albedo comparison: Site {SITE} ({cat}), {MONTHS} months from {START} ===\n")

    hydro = load_hydro(SITE)
    weather = load_weather()
    end = pd.Timestamp(START) + pd.DateOffset(months=MONTHS)
    hydro = hydro[START:str(end)]
    weather = weather[START:str(end)]
    idx = hydro.index.intersection(weather.index)
    hydro = hydro.loc[idx]
    weather = weather.loc[idx]
    n_steps = len(idx)

    results = {}
    for label, alb in ALBEDOS.items():
        print(f"  Running {label} ...", end=" ", flush=True)
        results[label] = run_with_albedo(SITE, alb, hydro, weather, idx, n_steps)
        r = results[label]
        dry = ~r["submerged"]
        print(f"done. Ts: {r['surf_temps'][dry].min():.1f}–{r['surf_temps'][dry].max():.1f}°C, "
              f"ET={r['evaps'].sum()*1000:.0f}mm")

    days = (np.array(idx, dtype="datetime64[ns]") - np.array(idx[0], dtype="datetime64[ns]"))
    days = days.astype("timedelta64[h]").astype(float) / 24.0

    depths_cm = results[list(ALBEDOS.keys())[0]]["col"].params.layer_centres() * 100.0

    # --- Summary ---
    print(f"\n  {'':30s} {'Bare (0.25)':>14} {'Algal (0.10)':>14} {'Difference':>14}")
    labels = list(ALBEDOS.keys())
    r1, r2 = results[labels[0]], results[labels[1]]
    dry1, dry2 = ~r1["submerged"], ~r2["submerged"]

    d_ts_max = r2["surf_temps"][dry2].max() - r1["surf_temps"][dry1].max()
    d_ts_mean = np.nanmean(r2["surf_temps"][dry2]) - np.nanmean(r1["surf_temps"][dry1])
    d_et = r2["evaps"].sum()*1000 - r1["evaps"].sum()*1000
    d_sw = np.nanmean(r2["sw_net"][dry2]) - np.nanmean(r1["sw_net"][dry1])

    print(f"  {'Peak surface T (dry)':30s} {r1['surf_temps'][dry1].max():14.1f} "
          f"{r2['surf_temps'][dry2].max():14.1f} {d_ts_max:+14.1f}")
    print(f"  {'Mean surface T (dry)':30s} {np.nanmean(r1['surf_temps'][dry1]):14.1f} "
          f"{np.nanmean(r2['surf_temps'][dry2]):14.1f} {d_ts_mean:+14.1f}")
    print(f"  {'Cumulative ET (mm)':30s} {r1['evaps'].sum()*1000:14.0f} "
          f"{r2['evaps'].sum()*1000:14.0f} {d_et:+14.0f}")
    print(f"  {'Mean SW_net (dry, W/m2)':30s} {np.nanmean(r1['sw_net'][dry1]):14.1f} "
          f"{np.nanmean(r2['sw_net'][dry2]):14.1f} {d_sw:+14.1f}")

    # --- Plot ---
    fig = plt.figure(figsize=(16, 22))
    gs = GridSpec(6, 3, figure=fig,
                  width_ratios=[1, 1, 0.02],
                  height_ratios=[0.3, 0.5, 1.0, 1.0, 0.4, 0.4],
                  hspace=0.30, wspace=0.08)

    xlim = (days[0], days[-1])
    col_labels = list(ALBEDOS.keys())

    # Create axes with shared x per column
    ax_left_0 = fig.add_subplot(gs[0, 0])
    ax_right_0 = fig.add_subplot(gs[0, 1], sharex=ax_left_0, sharey=ax_left_0)
    left_axes = [ax_left_0] + [fig.add_subplot(gs[r, 0], sharex=ax_left_0) for r in range(1, 6)]
    right_axes = [ax_right_0] + [fig.add_subplot(gs[r, 1], sharex=ax_left_0, sharey=left_axes[r])
                                  for r in range(1, 6)]

    for ci, (label, axes) in enumerate([(col_labels[0], left_axes), (col_labels[1], right_axes)]):
        r = results[label]

        # Row 0: water depth
        ax = axes[0]
        ax.fill_between(days, 0, hydro["water_depth"].values, alpha=0.4, color="blue")
        ax.set_xlim(xlim)
        if ci == 0:
            ax.set_ylabel("Water depth (m)")
        ax.set_title(label, fontsize=12, fontweight="bold")

        # Row 1: surface + air temp
        ax = axes[1]
        ax.plot(days, weather["temperature_2m"].values, "grey", lw=0.5, alpha=0.6, label="Air")
        ax.plot(days, r["surf_temps"], "r-", lw=0.5, label="Surface")
        if ci == 0:
            ax.set_ylabel("Temp (degC)")
            ax.legend(fontsize=7, ncol=2, loc="upper right")
        ax.grid(True, alpha=0.2)

        # Row 2: temperature heatmap
        ax = axes[2]
        im_t = ax.pcolormesh(days, depths_cm, r["temp_profiles"].T,
                             shading="auto", cmap="RdYlBu_r", vmin=15, vmax=55)
        ax.set_ylim(50, 0)
        if ci == 0:
            ax.set_ylabel("Depth (cm)")

        # Row 3: moisture heatmap + WT
        ax = axes[3]
        porosity = r["col"].params.porosity
        im_m = ax.pcolormesh(days, depths_cm, r["moist_profiles"].T,
                             shading="auto", cmap="YlGnBu", vmin=0.05, vmax=porosity)
        if not np.all(np.isnan(r["wt_depths"])):
            ax.plot(days, r["wt_depths"] * 100, "r-", lw=1.5, label="WT")
            if ci == 0:
                ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(50, 0)
        if ci == 0:
            ax.set_ylabel("Depth (cm)")

        # Row 4: energy fluxes
        ax = axes[4]
        ax.plot(days, r["sw_net"], "gold", lw=0.5, label="SW_net")
        ax.plot(days, -r["H"], "red", lw=0.5, label="-H")
        ax.plot(days, -r["LE"], "green", lw=0.5, label="-LE")
        ax.plot(days, -r["G"], "orange", lw=0.5, label="-G")
        ax.axhline(0, color="grey", lw=0.3)
        if ci == 0:
            ax.set_ylabel("W/m2")
            ax.legend(fontsize=7, ncol=4, loc="upper left")
        ax.grid(True, alpha=0.2)

        # Row 5: cumulative ET
        ax = axes[5]
        cum_et = np.cumsum(r["evaps"]) * 1000
        ax.plot(days, cum_et, "b-", lw=1.0)
        if ci == 0:
            ax.set_ylabel("Cum. ET (mm)")
        ax.set_xlabel("Days")
        ax.grid(True, alpha=0.2)

        # Hide x-tick labels except bottom
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        # Hide y-tick labels on right column
        if ci == 1:
            for ax in axes:
                plt.setp(ax.get_yticklabels(), visible=False)

    # Colorbars
    cax_t = fig.add_subplot(gs[2, 2])
    fig.colorbar(im_t, cax=cax_t, label="degC")
    cax_m = fig.add_subplot(gs[3, 2])
    fig.colorbar(im_m, cax=cax_m, label="m3/m3")

    fig.suptitle(f"Giralia Point {SITE} ({cat}) — albedo comparison, {MONTHS} months from {START}",
                 fontsize=14, fontweight="bold", y=0.995)

    outfile = os.path.join(PLOT_DIR, f"albedo_comparison_site{SITE}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {outfile}")


if __name__ == "__main__":
    main()
