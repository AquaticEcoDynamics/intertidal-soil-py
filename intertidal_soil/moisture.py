"""Campbell (1985) soil moisture redistribution solver.

Translated from the `infil` subroutine in aed_land.F90.  Solves the
Richards equation for vertical water movement through a layered soil column
using an implicit finite-difference scheme with the Thomas algorithm.

Includes:
  - Matric-potential-based water flow (Darcy)
  - Vapour diffusion between nodes
  - Root water uptake / transpiration
  - Evaporation from the soil surface

Indexing convention (matching the Fortran, but with one extra solved node
so all n_layers outputs are properly solved):

  Internal grid has nodes 2..mi where mi = n_layers + 1.
  z[1] = -1e10 (phantom upper), z[mi+1] = 1e20 (phantom lower).
  wn[2..mi] are solved;  wn[mi+1] = saturated boundary.
  Output: moisture_out[0..n_layers-1] = wn[2..mi+1],
          but wn[mi+1] is now one node beyond the deepest real layer.
"""

import numpy as np
from .soil_params import SoilParams


# Physical constants (matching the Fortran)
_G = 9.8          # gravitational acceleration, m/s^2
_MW = 0.018       # molar mass of water, kg/mol
_R_GAS = 8.31     # universal gas constant, J/(mol K)
_WD = 1000.0      # water density, kg/m^3
_DV = 2.4e-5      # binary diffusion coefficient of water vapour in air
_VP = 0.017       # reference vapour density (used in simple vapour flow)
_RW = 2.5e10      # resistance per unit length of root, m^3 kg^-1 s^-1
_PC = -1500.0     # critical leaf water potential for stomatal closure, J/kg
_RL = 2.0e6       # resistance per unit length of leaf, m^3 kg^-1 s^-1
_SP = 10.0        # stability parameter for stomatal closure
_IM = 1.0e-6      # maximum allowable mass-balance error, kg
_MAX_ITER = 500


def campbell_moisture(
    params: SoilParams,
    moisture: np.ndarray,
    et_rate: float,
    temp: np.ndarray,
    dt: int = 3600,
    ha: float = 0.5,
    ponding: bool = False,
    bottom_bc: str = "fixed_head",
) -> dict:
    """Solve one time-step of the Campbell (1985) soil moisture model.

    Parameters
    ----------
    params   : SoilParams
    moisture : (n_layers,) volumetric water content per layer
    et_rate  : potential evapotranspiration rate (m/s)
    temp     : (n_layers,) soil temperature per layer (degC)
    dt       : time-step in seconds (default 3600)
    ha       : air relative humidity as fraction (default 0.5)
    ponding  : if True, pin the surface node at saturation each iteration
               (Dirichlet BC representing standing water above the soil)
    bottom_bc   : bottom boundary condition —
                  "fixed_head" saturated node at real depth (water table),
                  "free_drain" phantom depth for free-draining bottom,
                  "no_flux" sealed base allowing internal water table to rise

    Returns
    -------
    dict with moisture, evap, soil_water, humidity, potential
    """
    n = params.n_layers
    depth = params.layer_depths  # size n+2, indices 0..n+1

    # Use n+1 internal nodes so all n output layers are fully solved.
    # Solved nodes:  2 .. mi   where mi = n + 1
    # Boundary node: mi + 1    (always saturated)
    mi = n + 1         # last solved node index
    sz = mi + 2        # total array size (0 .. mi+1)

    pe  = np.full(sz, abs(params.psi_e) * -1.0)
    ks  = np.full(sz, params.K_s)
    bb  = np.full(sz, params.b)
    bd  = np.full(sz, params.bulk_density)
    root_l = np.zeros(sz)
    for i in range(2, mi + 1):
        idx = i - 2
        root_l[i] = params.root_density[idx] if idx < len(params.root_density) else 0.1

    ws = 1.0 - bd / params.mineral_density
    b1 = 1.0 / bb
    nn = 2.0 + 3.0 / bb
    n1 = 1.0 - nn

    # ------------------------------------------------------------------
    # Step 1: depth nodes (real values first, phantoms set later)
    # depth has n+2 entries (0..n+1).  We need sz entries (0..mi+1 = 0..n+2).
    # Extrapolate one extra node beyond depth[n+1].
    # ------------------------------------------------------------------
    z = np.zeros(sz)
    z[0:n + 2] = depth[0:n + 2]
    # Extrapolate the boundary node one layer-thickness beyond the last depth
    dz_last = depth[n + 1] - depth[n]
    z[mi + 1] = depth[n + 1] + dz_last  # one extra step beyond column

    # Temperature in Kelvin
    tk = np.zeros(sz)
    for i in range(2, mi + 1):
        idx = i - 2
        tk[i] = (temp[idx] if idx < n else temp[n - 1]) + 273.15
    tk[1] = tk[2]
    tk[mi + 1] = tk[mi]

    # ------------------------------------------------------------------
    # Step 2: initial water content and derived quantities
    # moisture[0..n-1] -> wn[2..n+1]  (all n layers)
    # ------------------------------------------------------------------
    wn = np.zeros(sz)
    w  = np.zeros(sz)
    p  = np.zeros(sz)
    h  = np.zeros(sz)
    k  = np.zeros(sz)
    v  = np.zeros(sz)

    for i in range(2, mi + 1):
        idx = i - 2
        wn[i] = np.clip(moisture[idx] if idx < n else moisture[n - 1], 1e-7, ws[i] - 1e-7)
        p[i] = pe[i] * (ws[i] / wn[i]) ** bb[i]
        h[i] = np.exp(_MW * p[i] / (_R_GAS * tk[i]))
        k[i] = ks[i] * (pe[i] / p[i]) ** nn[i]
        w[i] = wn[i]

    # ------------------------------------------------------------------
    # Step 3: volumes (using real z, BEFORE phantom assignment)
    # ------------------------------------------------------------------
    for i in range(2, mi + 1):
        v[i] = _WD * (z[i + 1] - z[i - 1]) / 2.0

    # ------------------------------------------------------------------
    # Step 4: lower boundary (saturated, at node mi+1)
    # ------------------------------------------------------------------
    p[mi + 1] = pe[mi]
    h[mi + 1] = 1.0
    w[mi + 1] = ws[mi + 1]
    wn[mi + 1] = ws[mi + 1]
    k[mi + 1] = ks[mi] * (pe[mi] / p[mi + 1]) ** nn[mi + 1]

    p[1] = p[2]
    k[1] = 0.0

    # ------------------------------------------------------------------
    # Step 5: phantom boundaries for the convergence loop
    # ------------------------------------------------------------------
    z[1] = -1e10
    if bottom_bc == "free_drain":
        z[mi + 1] = 1e20

    # ------------------------------------------------------------------
    # Root water uptake initialisation
    # ------------------------------------------------------------------
    rr = np.full(sz, 1e20)
    bz = np.zeros(sz)
    for i in range(2, mi + 1):
        if root_l[i] > 0:
            dz_node = z[i + 1] - z[i - 1]
            if dz_node > 0:
                rr[i] = 2.0 * _RW / (root_l[i] * dz_node)
                bz[i] = (
                    (1 - mi)
                    * np.log(np.pi * params.root_radius ** 2 * root_l[i])
                    / (2.0 * np.pi * root_l[i] * dz_node)
                )

    # Evapotranspiration partitioning
    ep = np.exp(-0.82 * params.lai) * et_rate
    tp = et_rate - ep

    # Plant water uptake: iterative leaf-potential solve
    rs = np.zeros(sz)
    pb = 0.0
    rb = 0.0
    for i in range(2, mi + 1):
        rs[i] = bz[i] / k[i] if k[i] > 1e-30 else 1e20
        pb += p[i] / (rr[i] + rs[i])
        rb += 1.0 / (rs[i] + rr[i])
    pb = pb / rb if rb > 0 else 0.0
    rb = 1.0 / rb if rb > 0 else 1e20

    pl = 0.0
    xp = 0.0
    for _it in range(_MAX_ITER):
        if pl > pb:
            pl = pb - tp * (_RL + rb)
        xp = (pl / _PC) ** _SP if abs(_PC) > 0 else 0.0
        denom = pl * (1.0 + xp) ** 2
        sl = tp * (_RL + rb) * _SP * xp / denom - 1.0 if abs(denom) > 1e-30 else -1.0
        ff = pb - pl - tp * (_RL + rb) / (1.0 + xp)
        if abs(sl) > 1e-30:
            pl = pl - ff / sl
        if abs(ff) <= 10.0:
            break

    tr = tp / (1.0 + xp) if (1.0 + xp) > 0 else 0.0

    # Root extraction (disabled, matching Fortran "MH" comment)
    e = np.zeros(sz)

    # ------------------------------------------------------------------
    # Main convergence loop
    # ------------------------------------------------------------------
    a_arr  = np.zeros(sz)
    b_arr  = np.zeros(sz)
    c_arr  = np.zeros(sz)
    f_arr  = np.zeros(sz)
    cp_arr = np.zeros(sz)
    jv     = np.zeros(sz)
    dj     = np.zeros(sz)
    dp_arr = np.zeros(sz)

    for _outer in range(_MAX_ITER):
        se = 0.0

        for i in range(2, mi + 1):
            k[i] = ks[i] * (pe[i] / p[i]) ** nn[i]

        # Surface vapour flux
        jv[1] = ep * (h[2] - ha) / (1.0 - ha) if (1.0 - ha) > 1e-10 else 0.0
        dj[1] = ep * _MW * h[2] / (_R_GAS * tk[2] * (1.0 - ha)) if (1.0 - ha) > 1e-10 else 0.0

        for i in range(2, mi + 1):
            i1 = min(i + 1, mi + 1)
            avg_wn = (wn[i] + wn[i1]) / 2.0
            dz_below = z[i + 1] - z[i]
            kv = 0.66 * _DV * _VP * max(ws[i] - avg_wn, 0.0) / dz_below if dz_below > 0 else 0.0
            jv[i] = kv * (h[i1] - h[i])
            dj[i] = _MW * h[i] * kv / (_R_GAS * tk[i]) if tk[i] > 0 else 0.0
            cp_arr[i] = -v[i] * wn[i] / (bb[i] * p[i] * dt) if abs(p[i]) > 1e-30 else 0.0

            dz_above = z[i] - z[i - 1]
            dz_below = z[i + 1] - z[i]

            if dz_above > 0 and abs(p[i - 1]) > 1e-30:
                a_arr[i] = -k[i - 1] / dz_above + _G * nn[i] * k[i - 1] / p[i - 1]
            else:
                a_arr[i] = 0.0

            c_arr[i] = -k[i1] / dz_below if dz_below > 0 else 0.0

            if dz_above > 0 and dz_below > 0:
                b_arr[i] = (
                    k[i] / dz_above + k[i] / dz_below + cp_arr[i]
                    - _G * nn[i] * k[i] / p[i]
                    + dj[i - 1] + dj[i]
                )
            else:
                b_arr[i] = 1.0

            flux_above = (p[i] * k[i] - p[i - 1] * k[i - 1]) / dz_above if dz_above > 0 else 0.0
            flux_below = (p[i1] * k[i1] - p[i] * k[i]) / dz_below if dz_below > 0 else 0.0
            if abs(n1[i]) > 1e-30:
                f_arr[i] = (
                    (flux_above - flux_below) / n1[i]
                    + v[i] * (wn[i] - w[i]) / dt
                    - _G * (k[i - 1] - k[i])
                    + jv[i - 1] - jv[i]
                    + e[i]
                )
            else:
                f_arr[i] = 0.0
            se += abs(f_arr[i])

        # No-flux bottom BC: seal the base of the last solved node so
        # no water leaves the column; allows an internal water table to rise.
        if bottom_bc == "no_flux":
            se -= abs(f_arr[mi])
            jv[mi] = 0.0
            dj[mi] = 0.0
            c_arr[mi] = 0.0
            dz_a = z[mi] - z[mi - 1]
            b_arr[mi] = (
                k[mi] / dz_a + cp_arr[mi]
                - _G * nn[mi] * k[mi] / p[mi]
                + dj[mi - 1]
            ) if dz_a > 0 else 1.0
            fl_above = (p[mi] * k[mi] - p[mi - 1] * k[mi - 1]) / dz_a if dz_a > 0 else 0.0
            if abs(n1[mi]) > 1e-30:
                f_arr[mi] = (
                    fl_above / n1[mi]
                    + v[mi] * (wn[mi] - w[mi]) / dt
                    - _G * (k[mi - 1] - k[mi])
                    + jv[mi - 1]
                    + e[mi]
                )
            else:
                f_arr[mi] = 0.0
            se += abs(f_arr[mi])

        # Ponding Dirichlet BC: force surface node to saturation potential
        # and exclude it from the convergence residual so the solver
        # converges on the interior nodes that receive infiltration.
        if ponding:
            se -= abs(f_arr[2])
            p_sat = pe[2] * (ws[2] / (ws[2] - 1e-7)) ** bb[2]
            a_arr[2] = 0.0
            b_arr[2] = 1.0
            c_arr[2] = 0.0
            f_arr[2] = p[2] - p_sat

        # Thomas algorithm (forward sweep on nodes 2..mi-1)
        for i in range(2, mi):
            if abs(b_arr[i]) > 1e-30:
                c_arr[i] = c_arr[i] / b_arr[i]
                if abs(c_arr[i]) < 1e-8:
                    c_arr[i] = 0.0
            f_arr[i] = f_arr[i] / b_arr[i] if abs(b_arr[i]) > 1e-30 else 0.0
            b_arr[i + 1] -= a_arr[i + 1] * c_arr[i]
            f_arr[i + 1] -= a_arr[i + 1] * f_arr[i]

        # Back substitution
        dp_arr[mi] = f_arr[mi] / b_arr[mi] if abs(b_arr[mi]) > 1e-30 else 0.0
        p[mi] -= dp_arr[mi]
        if p[mi] > pe[mi]:
            p[mi] = pe[mi]

        for i in range(mi - 1, 1, -1):
            dp_arr[i] = f_arr[i] - c_arr[i] * dp_arr[i + 1]
            p[i] -= dp_arr[i]
            if p[i] > pe[i]:
                p[i] = (p[i] + dp_arr[i] + pe[i]) / 2.0

        # Update water content from potential
        for i in range(2, mi + 1):
            wn[i] = max(ws[i] * (pe[i] / p[i]) ** b1[i], 1e-7)
            p[i] = pe[i] * (ws[i] / wn[i]) ** bb[i]
            h[i] = np.exp(_MW * p[i] / (_R_GAS * tk[i]))

        # Ponding: pin surface node at saturation (Dirichlet BC)
        if ponding:
            wn[2] = ws[2] - 1e-7
            p[2] = pe[2] * (ws[2] / wn[2]) ** bb[2]
            h[2] = np.exp(_MW * p[2] / (_R_GAS * tk[2]))

        h[mi + 1] = h[mi]

        if se <= _IM:
            break

    # Update w after convergence (Fortran line 1699, outside loop)
    w[:] = wn[:]

    # ------------------------------------------------------------------
    # Outputs: wn[2..mi] -> moisture_out[0..n-1]  (all n layers solved)
    # ------------------------------------------------------------------
    moisture_out = np.array([wn[i] for i in range(2, mi + 1)])  # n elements
    fl = ep * (h[2] - ha) / (1.0 - ha) * dt if (1.0 - ha) > 1e-10 else 0.0
    humidity_out = np.array([h[i] for i in range(2, mi + 1)])
    potential_out = np.array([p[i] for i in range(2, mi + 1)])

    dz_surf = z[3] - z[2]
    sw = 0.0
    if abs(n1[2]) > 1e-30 and dz_surf > 0:
        sw = ((p[2] * k[2] - p[3] * k[3]) / (n1[2] * dz_surf) + _G * k[2] + tr) * dt

    # Bottom boundary flux (m of water this timestep, positive = downward/out)
    # Darcy flux has units kg/(m²·s); × dt → kg/m²; ÷ _WD → m of water
    dz_bot = z[mi + 1] - z[mi]
    if dz_bot > 0 and bottom_bc != "no_flux" and abs(n1[mi]) > 1e-30:
        bottom_flux = (
            (p[mi + 1] * k[mi + 1] - p[mi] * k[mi]) / (n1[mi] * dz_bot)
            + _G * k[mi]
        ) * dt / _WD
    else:
        bottom_flux = 0.0

    return dict(
        moisture=moisture_out,
        evap=fl,
        soil_water=sw,
        humidity=humidity_out,
        potential=potential_out,
        bottom_flux=bottom_flux,
    )
