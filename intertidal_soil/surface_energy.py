"""Surface energy balance for exposed (dry) soil.

Computes the surface temperature from the balance of radiative, sensible,
latent, and ground heat fluxes.  Replaces the simple Dirichlet BC
(T_surface = T_air) with a physically-based surface temperature.

    Q_sw * (1 - albedo) + Q_lw_down - eps * sigma * Ts^4
        - H_sensible - H_latent = G

where G = -k * dT/dz  at z = 0  (into the soil, handled by the heat solver).

When submerged, the water temperature from the hydrodynamic model is used
directly as a Dirichlet BC (the hydro model already solved its own energy
balance on the water column).
"""

import numpy as np

_SIGMA = 5.67e-8   # Stefan-Boltzmann constant, W/(m^2 K^4)
_CP_AIR = 1004.0   # specific heat of air, J/(kg K)
_LV = 2.45e6       # latent heat of vaporisation, J/kg
_RHO_AIR = 1.2     # approximate air density, kg/m^3
_K_VON = 0.4       # von Karman constant
_Z_REF = 10.0      # reference height for met observations, m (ERA5 10m wind)
_Z0 = 0.001        # roughness length for bare tidal flat, m


def saturation_vapour_pressure(T_C):
    """Saturation vapour pressure (Pa) from temperature (degC).
    Tetens formula.
    """
    return 610.78 * np.exp(17.27 * T_C / (T_C + 237.3))


def surface_energy_balance(
    sw_down,          # W/m^2, incoming shortwave radiation
    T_air,            # degC, air temperature at reference height
    rh,               # %, relative humidity
    wind,             # m/s, wind speed at reference height
    T_soil_top,       # degC, current temperature of the top soil layer
    k_soil_top,       # W/(m K), thermal conductivity of top layer
    dz_top,           # m, distance from surface to top layer centre
    vwc_surface=0.2,  # m3/m3, volumetric water content of surface layer
    field_capacity=0.25, # m3/m3, field capacity (porosity - Sy)
    albedo=0.25,      # bare soil albedo
    emissivity=0.95,  # surface emissivity
    cloud_cover=0.0,  # fraction 0-1
    max_Ts=75.0,      # degC, upper clamp for stability
) -> dict:
    """Solve the surface energy balance for surface temperature.

    Uses Newton iteration to find Ts that balances:
        R_net - H - LE - G = 0

    Latent heat is moisture-limited: a beta factor (0-1) scales the
    potential evaporation based on surface soil moisture relative to
    field capacity.  When dry (vwc << fc), LE → 0 and the surface
    heats up under insolation.  When wet (vwc >= fc), full potential
    evaporation is allowed.

    Returns dict with Ts, G (ground heat flux into soil, W/m^2),
    and diagnostic fluxes.
    """
    T_air_K = T_air + 273.15

    # --- Soil moisture limitation on evaporation ---
    beta = min(vwc_surface / max(field_capacity, 0.01), 1.0)
    beta = max(beta, 0.0)

    # --- Incoming longwave (Brutsaert 1975 clear-sky + cloud correction) ---
    e_a = saturation_vapour_pressure(T_air) * rh / 100.0
    e_a_kPa = e_a / 1000.0
    eps_clear = 1.24 * (e_a_kPa / T_air_K) ** (1.0 / 7.0) if T_air_K > 0 else 0.8
    eps_sky = eps_clear * (1.0 - cloud_cover) + cloud_cover
    lw_down = eps_sky * _SIGMA * T_air_K ** 4

    # --- Bulk transfer coefficient for sensible/latent heat ---
    wind_eff = max(wind, 0.5)
    Ch = (_K_VON ** 2) / (np.log(_Z_REF / _Z0)) ** 2

    # --- Newton iteration for surface temperature ---
    Ts = T_air  # initial guess
    for _it in range(20):
        Ts_K = Ts + 273.15

        sw_net = sw_down * (1.0 - albedo)
        lw_out = emissivity * _SIGMA * Ts_K ** 4
        R_net = sw_net + lw_down - lw_out

        H = _RHO_AIR * _CP_AIR * Ch * wind_eff * (Ts - T_air)

        e_s = saturation_vapour_pressure(Ts)
        e_air = saturation_vapour_pressure(T_air) * rh / 100.0
        LE_pot = _RHO_AIR * _LV * Ch * wind_eff * 0.622 / 101325.0 * (e_s - e_air)
        LE = beta * max(LE_pot, 0.0)

        G = k_soil_top / dz_top * (Ts - T_soil_top)

        residual = R_net - H - LE - G

        # Derivative of residual w.r.t. Ts
        d_lw_out = 4.0 * emissivity * _SIGMA * Ts_K ** 3
        d_H = _RHO_AIR * _CP_AIR * Ch * wind_eff
        d_es = e_s * 17.27 * 237.3 / (Ts + 237.3) ** 2
        d_LE = beta * _RHO_AIR * _LV * Ch * wind_eff * 0.622 / 101325.0 * d_es
        if LE_pot < 0:
            d_LE = 0.0
        d_G = k_soil_top / dz_top
        d_residual = -d_lw_out - d_H - d_LE - d_G

        if abs(d_residual) < 1e-10:
            break
        Ts_new = Ts - residual / d_residual
        Ts_new = np.clip(Ts_new, T_air - 20.0, max_Ts)

        if abs(Ts_new - Ts) < 0.01:
            Ts = Ts_new
            break
        Ts = Ts_new

    Ts_K = Ts + 273.15
    G = k_soil_top / dz_top * (Ts - T_soil_top)

    return dict(
        Ts=Ts,
        G=G,
        sw_net=sw_down * (1.0 - albedo),
        lw_down=lw_down,
        lw_out=emissivity * _SIGMA * Ts_K ** 4,
        H=_RHO_AIR * _CP_AIR * Ch * wind_eff * (Ts - T_air),
        LE=LE,
        beta=beta,
    )
