"""Run all 12 Giralia sites and produce comparison plots.

Usage: python mardie/scripts/run_multi_site.py [months] [start_date]
Default: 12 months from 2017-01-01.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from giralia_config import SITES, MACRO_PARAMS, get_soil_params, load_hydro, load_weather
from intertidal_soil.drivers import GiraliaSoilColumn

MONTHS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
START = sys.argv[2] if len(sys.argv) > 2 else "2017-01-01"
DT = 3600.0

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

CATEGORY_ORDER = [
    "subtidal", "low_intertidal", "mid_intertidal",
    "high_intertidal", "supratidal",
]
CATEGORY_COLORS = {
    "subtidal": "#1f77b4",
    "low_intertidal": "#2ca02c",
    "mid_intertidal": "#ff7f0e",
    "high_intertidal": "#d62728",
    "supratidal": "#9467bd",
}


def run_site(site_id, weather, start, months):
    """Run one site and return summary timeseries."""
    hydro = load_hydro(site_id)
    end = pd.Timestamp(start) + pd.DateOffset(months=months)
    hydro = hydro[start:str(end)]
    w = weather[start:str(end)]

    idx = hydro.index.intersection(w.index)
    hydro = hydro.loc[idx]
    w = w.loc[idx]
    n_steps = len(idx)

    soil_kw = get_soil_params(site_id)
    cat = SITES[site_id]["category"]
    col = GiraliaSoilColumn(site_id, soil_kw, dt=DT, wt_depth=0.8)
    col.initialise(air_temp=25.0, deep_temp=26.0,
                   init_moisture=0.12, spin_up_days=10.0)

    wt_depths = np.full(n_steps, np.nan)
    surf_temps = np.full(n_steps, np.nan)
    evaps = np.full(n_steps, np.nan)
    submerged = np.zeros(n_steps, dtype=bool)
    soil_temp_5cm = np.full(n_steps, np.nan)
    soil_temp_20cm = np.full(n_steps, np.nan)
    soil_moist_5cm = np.full(n_steps, np.nan)
    soil_moist_20cm = np.full(n_steps, np.nan)

    depths = col.params.layer_centres()
    i5 = np.argmin(np.abs(depths - 0.05))
    i20 = np.argmin(np.abs(depths - 0.20))

    for i in range(n_steps):
        wd = hydro["water_depth"].iloc[i]
        wt = hydro["sea_tem"].iloc[i]
        ta = w["temperature_2m"].iloc[i]
        sw = w["shortwave_radiation"].iloc[i]
        rh_val = w["relative_humidity_2m"].iloc[i]
        wind = w["wind_speed_10m"].iloc[i] / 3.6
        rain_mm = w["precipitation"].iloc[i]
        rain_m_day = rain_mm / 1000.0 * 24.0
        cc = w["cloud_cover"].iloc[i] / 100.0

        state, diag = col.step(wd, wt, ta, sw, rh_val, wind, rain_m_day, cc)

        wt_depths[i] = diag.get("water_table_depth", np.nan)
        surf_temps[i] = diag.get("surface_temp", ta)
        evaps[i] = diag.get("evap", 0.0)
        submerged[i] = diag.get("is_submerged", False)
        soil_temp_5cm[i] = state.temperature[i5]
        soil_temp_20cm[i] = state.temperature[i20]
        soil_moist_5cm[i] = state.moisture[i5]
        soil_moist_20cm[i] = state.moisture[i20]

    return dict(
        site_id=site_id,
        category=cat,
        times=idx,
        wt_depths=wt_depths,
        surf_temps=surf_temps,
        evaps=evaps,
        submerged=submerged,
        soil_temp_5cm=soil_temp_5cm,
        soil_temp_20cm=soil_temp_20cm,
        soil_moist_5cm=soil_moist_5cm,
        soil_moist_20cm=soil_moist_20cm,
        wet_frac=submerged.sum() / len(submerged) * 100,
        cum_evap_mm=evaps.sum() * 1000,
        f_macro=get_soil_params(site_id)["f_macro"],
    )


def daily_mean(times, values):
    """Compute daily means from hourly data."""
    s = pd.Series(values, index=times)
    return s.resample("D").mean()


def main():
    print(f"=== Giralia multi-site run: {MONTHS} months from {START} ===\n")
    weather = load_weather()

    results = {}
    for sid in sorted(SITES.keys()):
        t0 = time.time()
        print(f"  Site {sid:2d} ({SITES[sid]['category']:17s}) ...", end=" ", flush=True)
        results[sid] = run_site(sid, weather, START, MONTHS)
        r = results[sid]
        elapsed = time.time() - t0
        print(f"done ({elapsed:.1f}s) | wet={r['wet_frac']:.0f}% "
              f"ET={r['cum_evap_mm']:.0f}mm "
              f"WT={r['wt_depths'][-1]*100:.0f}cm")

    # --- Summary table ---
    print(f"\n{'Site':>5} {'Category':>17} {'f_macro':>8} {'Wet%':>6} "
          f"{'ET(mm)':>8} {'WT_end(cm)':>10} {'Tsurf_mean':>10}")
    for sid in sorted(SITES.keys()):
        r = results[sid]
        print(f"{sid:5d} {r['category']:>17} {r['f_macro']:8.3f} "
              f"{r['wet_frac']:6.1f} {r['cum_evap_mm']:8.0f} "
              f"{r['wt_depths'][-1]*100:10.1f} "
              f"{np.nanmean(r['surf_temps']):10.1f}")

    # --- Comparison plots ---
    _plot_comparison(results)
    _plot_per_site_panels(results)

    print("\nDone.")


def _plot_comparison(results):
    """Summary comparison across all sites."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 16),
                             gridspec_kw=dict(hspace=0.35))
    fig.suptitle(f"Giralia multi-site comparison — {MONTHS} months from {START}",
                 fontsize=14, fontweight="bold")

    plotted_cats = set()
    for sid in sorted(SITES.keys()):
        r = results[sid]
        cat = r["category"]
        col = CATEGORY_COLORS[cat]
        label = f"{cat}" if cat not in plotted_cats else None
        plotted_cats.add(cat)
        days = daily_mean(r["times"], np.ones(len(r["times"])))
        d_idx = days.index

        # Water table depth
        wt_d = daily_mean(r["times"], r["wt_depths"] * 100)
        axes[0].plot(d_idx, wt_d.values, color=col, lw=0.8, alpha=0.7, label=label)

        # Surface temperature
        ts_d = daily_mean(r["times"], r["surf_temps"])
        axes[1].plot(d_idx, ts_d.values, color=col, lw=0.5, alpha=0.6)

        # 5cm soil moisture
        sm_d = daily_mean(r["times"], r["soil_moist_5cm"])
        axes[2].plot(d_idx, sm_d.values, color=col, lw=0.8, alpha=0.7)

        # Cumulative evaporation
        cum_e = np.cumsum(r["evaps"]) * 1000
        cum_d = daily_mean(r["times"], cum_e)
        axes[3].plot(d_idx, cum_d.values, color=col, lw=0.8, alpha=0.7)

    axes[0].set_ylabel("Water table depth (cm)")
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8, ncol=3, loc="lower left")
    axes[0].grid(True, alpha=0.2)

    axes[1].set_ylabel("Surface temp (degC)")
    axes[1].grid(True, alpha=0.2)

    axes[2].set_ylabel("Moisture at 5cm (m3/m3)")
    axes[2].grid(True, alpha=0.2)

    axes[3].set_ylabel("Cumulative ET (mm)")
    axes[3].set_xlabel("Date")
    axes[3].grid(True, alpha=0.2)

    outfile = os.path.join(PLOT_DIR, f"giralia_multisite_{MONTHS}mo.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {outfile}")


def _plot_per_site_panels(results):
    """Individual site panels: water table + moisture at 5cm."""
    fig, axes = plt.subplots(4, 3, figsize=(20, 16),
                             gridspec_kw=dict(hspace=0.40, wspace=0.25))
    fig.suptitle(f"Giralia per-site water table and moisture — {MONTHS} months from {START}",
                 fontsize=14, fontweight="bold")

    site_ids = sorted(SITES.keys())
    for idx, sid in enumerate(site_ids):
        r = results[sid]
        row, col_idx = divmod(idx, 3)
        ax = axes[row, col_idx]

        d_idx_wt = daily_mean(r["times"], r["wt_depths"] * 100)
        d_idx_sm = daily_mean(r["times"], r["soil_moist_5cm"])

        ax.plot(d_idx_wt.index, d_idx_wt.values, "b-", lw=0.8, label="WT (cm)")
        ax.set_ylabel("WT depth (cm)", color="blue", fontsize=8)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelcolor="blue", labelsize=7)
        ax.tick_params(axis="x", labelsize=7, rotation=30)

        ax2 = ax.twinx()
        ax2.plot(d_idx_sm.index, d_idx_sm.values, "g-", lw=0.8, label="VWC 5cm")
        ax2.set_ylabel("VWC 5cm", color="green", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="green", labelsize=7)

        cat = r["category"]
        ax.set_title(f"Pt {sid} ({cat}, wet={r['wet_frac']:.0f}%)",
                     fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.15)

    outfile = os.path.join(PLOT_DIR, f"giralia_persites_{MONTHS}mo.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"  Saved {outfile}")


if __name__ == "__main__":
    main()
