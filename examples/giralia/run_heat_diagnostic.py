"""Diagnostic: heat budget components for an exposed supratidal site.

Runs site 2 (supratidal, ~6% wet) for 14 days in summer and dumps
the SEB components to understand why surface temperatures are too low.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


from giralia_config import SITES, get_soil_params, load_hydro, load_weather
from intertidal_soil.drivers import GiraliaSoilColumn
from intertidal_soil.surface_energy import surface_energy_balance

SITE = int(sys.argv[1]) if len(sys.argv) > 1 else 2
START = sys.argv[2] if len(sys.argv) > 2 else "2017-01-15"
DAYS = int(sys.argv[3]) if len(sys.argv) > 3 else 14
DT = 3600.0

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def main():
    print(f"=== Heat budget diagnostic: Site {SITE}, {DAYS} days from {START} ===")

    hydro = load_hydro(SITE)
    weather = load_weather()
    end = pd.Timestamp(START) + pd.Timedelta(days=DAYS)
    hydro = hydro[START:str(end)]
    weather = weather[START:str(end)]
    idx = hydro.index.intersection(weather.index)
    hydro = hydro.loc[idx]
    weather = weather.loc[idx]
    n_steps = len(idx)

    soil_kw = get_soil_params(SITE)
    col = GiraliaSoilColumn(SITE, soil_kw, dt=DT, wt_depth=0.8)
    col.initialise(air_temp=30.0, deep_temp=26.0,
                   init_moisture=0.12, spin_up_days=10.0)

    records = []
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

        # Capture SEB inputs/outputs before the step
        vwc_top = col.state.moisture[0]
        T_soil_top = col.state.temperature[0]
        k_soil = col.params.thermal_conductivity(np.array([vwc_top]))[0]
        dz_top = 0.5 * (col.params.layer_depths[1] + col.params.layer_depths[2])

        state, diag = col.step(wd, wt, ta, sw, rh, wind, rain_m_day, cc)

        fc = col.params.porosity - col.params.Sy
        beta = min(vwc_top / max(fc, 0.01), 1.0)

        rec = dict(
            time=idx[i],
            water_depth=wd,
            submerged=diag.get("is_submerged", False),
            T_air=ta,
            sw_down=sw,
            rh=rh,
            wind=wind,
            cloud_cover=cc,
            vwc_top=vwc_top,
            T_soil_top=T_soil_top,
            k_soil_top=k_soil,
            dz_top=dz_top,
            beta=beta,
            Ts=diag.get("surface_temp", ta),
            G=diag.get("heatflux", 0.0),
            sw_net=diag.get("sw_net", 0.0),
            lw_down=diag.get("lw_down", 0.0),
            lw_out=diag.get("lw_out", 0.0),
            H=diag.get("H", 0.0),
            LE=diag.get("LE", 0.0),
            wt_depth=diag.get("water_table_depth", np.nan),
            T_5cm=state.temperature[np.argmin(np.abs(col.params.layer_centres() - 0.05))],
        )
        records.append(rec)

    df = pd.DataFrame(records)

    # --- Print summary ---
    dry = df[~df["submerged"]]
    print(f"\n  Total steps: {len(df)}, dry steps: {len(dry)}")
    print(f"\n  --- Dry-step flux means (W/m2) ---")
    print(f"  SW_net:  {dry['sw_net'].mean():7.1f}")
    print(f"  LW_down: {dry['lw_down'].mean():7.1f}")
    print(f"  LW_out:  {dry['lw_out'].mean():7.1f}")
    print(f"  H:       {dry['H'].mean():7.1f}")
    print(f"  LE:      {dry['LE'].mean():7.1f}")
    print(f"  G:       {dry['G'].mean():7.1f}")
    print(f"  Residual:{(dry['sw_net'] + dry['lw_down'] - dry['lw_out'] - dry['H'] - dry['LE'] - dry['G']).mean():7.1f}")

    print(f"\n  --- Dry-step peak daytime (sw>400) means ---")
    daytime = dry[dry["sw_down"] > 400]
    if len(daytime) > 0:
        print(f"  SW_net:  {daytime['sw_net'].mean():7.1f}")
        print(f"  LW_down: {daytime['lw_down'].mean():7.1f}")
        print(f"  LW_out:  {daytime['lw_out'].mean():7.1f}")
        print(f"  H:       {daytime['H'].mean():7.1f}")
        print(f"  LE:      {daytime['LE'].mean():7.1f}")
        print(f"  G:       {daytime['G'].mean():7.1f}")

    print(f"\n  --- Temperature ---")
    print(f"  Surface T: min={dry['Ts'].min():.1f}, max={dry['Ts'].max():.1f}, "
          f"mean={dry['Ts'].mean():.1f} degC")
    print(f"  Air T:     min={dry['T_air'].min():.1f}, max={dry['T_air'].max():.1f}, "
          f"mean={dry['T_air'].mean():.1f} degC")

    print(f"\n  --- Evap beta factor ---")
    print(f"  beta: min={dry['beta'].min():.3f}, max={dry['beta'].max():.3f}, "
          f"mean={dry['beta'].mean():.3f}")
    fc = col.params.porosity - col.params.Sy
    print(f"  Field capacity: {fc:.3f}")

    print(f"\n  --- Soil surface moisture ---")
    print(f"  VWC top: min={dry['vwc_top'].min():.4f}, max={dry['vwc_top'].max():.4f}, "
          f"mean={dry['vwc_top'].mean():.4f}")
    print(f"  Porosity: {col.params.porosity:.3f}")

    # --- Plot ---
    fig, axes = plt.subplots(5, 1, figsize=(14, 18),
                             gridspec_kw=dict(height_ratios=[0.3, 0.6, 1.0, 1.0, 0.5],
                                              hspace=0.35))
    cat = SITES[SITE]["category"] if SITE in SITES else "?"
    fig.suptitle(f"Heat budget diagnostic — Site {SITE} ({cat}), {DAYS} days from {START}",
                 fontsize=13, fontweight="bold")

    hours = np.arange(len(df))

    # Row 0: submergence
    ax = axes[0]
    ax.fill_between(hours, 0, df["water_depth"].values, alpha=0.5, color="blue")
    ax.set_ylabel("Water depth (m)")
    ax.set_xlim(0, len(df))

    # Row 1: temperatures
    ax = axes[1]
    ax.plot(hours, df["T_air"], "grey", lw=0.8, label="T_air")
    ax.plot(hours, df["Ts"], "r-", lw=1.0, label="T_surface")
    ax.plot(hours, df["T_soil_top"], "brown", lw=0.8, label="T_soil[0]")
    ax.plot(hours, df["T_5cm"], "orange", lw=0.8, label="T_5cm")
    ax.set_ylabel("Temperature (degC)")
    ax.legend(fontsize=8, ncol=4)
    ax.grid(True, alpha=0.2)

    # Row 2: flux components
    ax = axes[2]
    ax.plot(hours, df["sw_net"], "gold", lw=0.8, label="SW_net")
    ax.plot(hours, df["lw_down"], "skyblue", lw=0.8, label="LW_down")
    ax.plot(hours, -df["lw_out"], "navy", lw=0.8, label="-LW_out")
    ax.plot(hours, -df["H"], "red", lw=0.8, label="-H")
    ax.plot(hours, -df["LE"], "green", lw=1.2, label="-LE")
    ax.plot(hours, -df["G"], "orange", lw=0.8, label="-G")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_ylabel("Flux (W/m2)")
    ax.set_title("Energy balance components (positive = warming surface)", fontsize=10)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.2)

    # Row 3: LE + beta
    ax = axes[3]
    ax.plot(hours, df["LE"], "g-", lw=1.2, label="LE (moisture-limited)")
    ax.set_ylabel("LE (W/m2)", color="green")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax2 = ax.twinx()
    ax2.plot(hours, df["beta"], "m-", lw=1.0, alpha=0.7, label="beta")
    ax2.set_ylabel("beta (0-1)", color="purple")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8, loc="upper right")
    ax.set_title("Latent heat and soil moisture beta factor", fontsize=10)

    # Row 4: soil moisture + WT
    ax = axes[4]
    ax.plot(hours, df["vwc_top"], "g-", lw=1.0, label="VWC top layer")
    ax.set_ylabel("VWC (m3/m3)", color="green")
    ax2 = ax.twinx()
    ax2.plot(hours, df["wt_depth"] * 100, "b-", lw=1.0, label="WT depth")
    ax2.set_ylabel("WT depth (cm)", color="blue")
    ax2.invert_yaxis()
    ax.set_xlabel("Hours")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)

    outfile = os.path.join(PLOT_DIR, f"heat_diagnostic_site{SITE}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n  Saved {outfile}")


if __name__ == "__main__":
    main()
