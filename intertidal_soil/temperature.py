"""1-D soil heat conduction solver.

Implements the SoilTemp and InitialTemp functionality that aed_land.F90
calls from an external module.  Uses an implicit (backward-Euler) finite-
difference scheme on the same layer grid as the moisture solver.
"""

import numpy as np
from .soil_params import SoilParams


def soil_temp(
    params: SoilParams,
    vwc: np.ndarray,
    surface_temp: float,
    temp: np.ndarray,
    dt: float = 900.0,
    surface_flux: float = None,
) -> tuple[np.ndarray, float]:
    """Advance the soil temperature profile by one time-step.

    Parameters
    ----------
    params       : SoilParams
    vwc          : (n_layers,) volumetric water content per layer
    surface_temp : degC, temperature at the soil surface (air or water).
                   Used as Dirichlet BC when surface_flux is None.
    temp         : (n_layers,) current temperature profile (degC)
    dt           : time-step in seconds (default 900 = 15 min)
    surface_flux : W/m^2, prescribed ground heat flux at surface (positive =
                   into soil).  When set, replaces the Dirichlet BC with a
                   Neumann BC: the top-layer RHS gets this flux directly,
                   and surface_temp is ignored.

    Returns
    -------
    temp_new  : (n_layers,) updated temperature profile
    heatflux  : W/m^2, heat flux into the soil surface (positive = into soil)
    """
    n = params.n_layers
    d = params.layer_depths  # size n+2; d[0]=0, d[1..n+1]

    dz = np.diff(d[1:n + 2])            # thickness of each layer (n,)
    zc = 0.5 * (d[1:n + 1] + d[2:n + 2])  # centre depths (n,)

    lam = params.thermal_conductivity(vwc)  # (n,)
    C = params.heat_capacity(vwc)           # (n,)

    # Inter-layer distances and interface conductivities (harmonic mean)
    dzc = np.zeros(n + 1)
    ki = np.zeros(n + 1)

    # Top interface: surface -> layer 0
    dzc[0] = zc[0]  # distance from surface to first cell centre
    ki[0] = lam[0]  # use top-layer conductivity for surface interface

    for i in range(1, n):
        dzc[i] = zc[i] - zc[i - 1]
        ki[i] = 2.0 * lam[i] * lam[i - 1] / (lam[i] + lam[i - 1] + 1e-30)

    # Bottom interface
    dzc[n] = d[n + 1] - zc[n - 1]
    ki[n] = lam[n - 1]

    # Build tridiagonal system: C_i * dz_i * (T_new - T_old)/dt
    #   = ki_{i-1}/dzc_{i-1} * (T_{i-1} - T_i) + ki_i/dzc_i * (T_{i+1} - T_i)
    a = np.zeros(n)  # sub-diagonal
    b = np.zeros(n)  # diagonal
    c = np.zeros(n)  # super-diagonal
    rhs = np.zeros(n)

    use_flux_bc = surface_flux is not None

    for i in range(n):
        cap = C[i] * dz[i] / dt

        flux_below = ki[i + 1] / dzc[i + 1] if dzc[i + 1] > 0 else 0.0

        if i == 0 and use_flux_bc:
            b[i] = cap + flux_below
            rhs[i] = cap * temp[i] + surface_flux
        else:
            flux_above = ki[i] / dzc[i] if dzc[i] > 0 else 0.0
            b[i] = cap + flux_above + flux_below
            if i > 0:
                a[i] = -flux_above
            rhs[i] = cap * temp[i]
            if i == 0:
                rhs[i] += flux_above * surface_temp

        if i < n - 1:
            c[i] = -flux_below

        if i == n - 1:
            rhs[i] += flux_below * params.deep_temp

    temp_new = thomas_solve(a, b, c, rhs)

    if use_flux_bc:
        heatflux = surface_flux
    else:
        heatflux = ki[0] / dzc[0] * (surface_temp - temp_new[0]) if dzc[0] > 0 else 0.0

    return temp_new, heatflux


def initial_temp(
    params: SoilParams,
    vwc_init: float,
    air_temp: float,
    deep_temp: float,
    spin_up_days: float = 30.0,
    dt: float = 900.0,
) -> np.ndarray:
    """Spin up soil temperature to quasi-equilibrium.

    Runs the heat equation repeatedly with constant surface and deep
    temperatures until spin_up_days of model time have elapsed.
    """
    n = params.n_layers
    params_local = params
    params_local.deep_temp = deep_temp

    vwc = np.full(n, vwc_init)
    temp = np.linspace(air_temp, deep_temp, n)

    n_steps = int(spin_up_days * 86400.0 / dt)
    for _ in range(n_steps):
        temp, _ = soil_temp(params_local, vwc, air_temp, temp, dt)

    return temp


def thomas_solve(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system Ax = d using the Thomas algorithm.

    a : sub-diagonal  (a[0] unused)
    b : main diagonal
    c : super-diagonal (c[n-1] unused)
    d : right-hand side
    """
    n = len(d)
    cp = np.zeros(n)
    dp = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m if i < n - 1 else 0.0
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m

    x = np.zeros(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x
