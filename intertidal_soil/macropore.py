"""Macropore (preferential flow) cascade model.

Simple dual-domain representation for structured/cracked soils:
  - Macropore domain: fast gravity-driven cascade (kinematic wave)
  - Matrix domain: Campbell (1985) Richards equation (existing solver)
  - Exchange: first-order transfer from macropores into matrix
"""

import numpy as np
from .soil_params import SoilParams


def macropore_step(
    params: SoilParams,
    macro_moisture: np.ndarray,
    matrix_moisture: np.ndarray,
    dt: float,
    ponding: bool = False,
    n_active: int = None,
) -> dict:
    """Advance the macropore domain by one timestep.

    Parameters
    ----------
    params          : SoilParams (uses f_macro, k_macro, alpha_exchange)
    macro_moisture  : (n_layers,) macropore volumetric water content
    matrix_moisture : (n_layers,) matrix volumetric water content
    dt              : timestep in seconds
    ponding         : True if surface is submerged
    n_active        : number of unsaturated layers above the water table.
                      When set, the cascade and exchange operate only on
                      these layers; water exiting the bottom is returned
                      as bottom_drain (direct macropore recharge to WT).
                      None = all layers (no WT coupling).

    Returns
    -------
    dict with macro_moisture (updated), exchange (n_layers, m³/m³ added
    to matrix this step), and bottom_drain (m of water exiting the base
    of the macropore column — direct recharge to the water table).
    """
    n = params.n_layers
    n_calc = n_active if n_active is not None else n
    f_macro = params.f_macro
    k_macro = params.k_macro
    alpha = params.alpha_exchange
    depths = params.layer_depths

    macro = macro_moisture.copy()
    exchange = np.zeros(n)

    layer_dz = np.array([depths[i + 2] - depths[i + 1] for i in range(n)])

    # --- Phase 1: gravity cascade (top-down kinematic wave) ---
    # k_macro is a macropore hydraulic conductivity (m/s).  During ponding,
    # water enters the macropore surface; the input is capped at available
    # macropore volume.  During dry periods, no external input but stored
    # macropore water still drains under gravity toward the water table.
    if ponding:
        empty_vol = sum(max(f_macro - macro[i], 0) * layer_dz[i]
                        for i in range(n_calc))
        flux_in = min(k_macro * f_macro * dt, empty_vol)
    else:
        flux_in = 0.0

    for i in range(n_calc):
        dz = layer_dz[i]
        if dz <= 0:
            continue
        macro[i] += flux_in / dz
        if macro[i] > f_macro:
            overflow = (macro[i] - f_macro) * dz
            macro[i] = f_macro
        else:
            overflow = 0.0
        drain = min(macro[i], k_macro * f_macro * dt / dz)
        macro[i] -= drain
        flux_in = drain * dz + overflow

    # Water exiting the base of the unsaturated macropore column.
    # Only meaningful when n_active > 0 (there are layers to drain through).
    bottom_drain = flux_in if n_calc > 0 else 0.0


    # --- Phase 2: exchange with matrix ---
    matrix_cap = params.porosity - f_macro

    for i in range(n_calc):
        Sr_macro = macro[i] / f_macro if f_macro > 0 else 0.0
        Sr_matrix = matrix_moisture[i] / matrix_cap if matrix_cap > 0 else 1.0

        xfer = alpha * f_macro * (Sr_macro - Sr_matrix) * dt

        if xfer > 0:
            xfer = min(xfer, macro[i])
            xfer = min(xfer, matrix_cap - matrix_moisture[i])
        else:
            xfer = max(xfer, -matrix_moisture[i] * 0.5)
            xfer = max(xfer, -(f_macro - macro[i]))

        macro[i] -= xfer
        exchange[i] = xfer

    return dict(macro_moisture=macro, exchange=exchange,
                bottom_drain=bottom_drain)
