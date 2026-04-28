"""Soil column driver: callable forward-step interface.

Wraps IntertidalSoilModel with surface energy balance for exposed steps
and water-temperature Dirichlet BC for submerged steps.  Designed to be
called from a larger coupled model at each timestep.
"""

import numpy as np

from intertidal_soil import SoilParams, IntertidalSoilModel, ModelState
from intertidal_soil.surface_energy import surface_energy_balance
from intertidal_soil.temperature import soil_temp


class GiraliaSoilColumn:
    """Single soil column with energy-balance surface BC.

    Parameters
    ----------
    site_id       : int, site identifier
    soil_kwargs   : dict passed to SoilParams
    dt            : timestep in seconds (default 3600 = 1 hour)
    wt_depth      : initial water table depth (m below surface)
    """

    def __init__(self, site_id, soil_kwargs, dt=3600.0,
                 wt_depth=0.8):
        self.site_id = site_id
        self.dt = dt
        self.params = SoilParams(**soil_kwargs)

        self.model = IntertidalSoilModel(
            self.params,
            surface_elevation=0.0,
            dt=dt,
            water_table_depth=wt_depth,
            fixed_et=None,
            evap_scale=0.0,
            evap_extinction_depth=0.15,
        )
        self.state = None

    def initialise(self, air_temp=30.0, deep_temp=26.0,
                   init_moisture=0.15, spin_up_days=10.0):
        """Create initial state with temperature spin-up."""
        self.params.deep_temp = deep_temp
        self.state = self.model.initialise(
            air_temp=air_temp, deep_temp=deep_temp,
            init_moisture=init_moisture, spin_up_days=spin_up_days,
        )
        return self.state

    def step(
        self,
        water_depth,    # m, depth of water above soil surface (0 = dry)
        water_temp,     # degC, temperature of overlying water (used when wet)
        air_temp,       # degC, air temperature
        sw_down,        # W/m^2, incoming shortwave radiation
        rh,             # %, relative humidity
        wind,           # m/s, wind speed
        rain,           # m/day, rainfall rate
        cloud_cover=0.0,  # fraction 0-1
        albedo=0.25,    # surface albedo (can vary per step)
    ):
        """Advance the soil column by one timestep.

        Returns (state, diagnostics) where state is the updated ModelState
        and diagnostics is a dict of scalar outputs.
        """
        submerged = water_depth > 0.01

        if submerged:
            new_state, diag = self._step_wet(water_temp)
        else:
            new_state, diag = self._step_dry(
                air_temp, sw_down, rh, wind, rain, cloud_cover, albedo)

        new_state.is_submerged = submerged
        diag["is_submerged"] = submerged
        diag["water_depth"] = water_depth
        self.state = new_state
        return new_state, diag

    def _step_wet(self, water_temp):
        """Submerged: use water temperature as Dirichlet BC."""
        return self.model._step_wet(self.state, water_temp)

    def _step_dry(self, air_temp, sw_down, rh, wind, rain, cloud_cover, albedo):
        """Exposed: energy balance determines surface temperature."""
        from intertidal_soil.macropore import macropore_step
        from intertidal_soil.moisture import campbell_moisture

        matrix_moist = self.state.moisture.copy()
        macro_out = None
        macro_drain = 0.0
        params = self.params
        model = self.model

        if model.dual_domain:
            n_act = (model.wt_tracker.find_active_layers(
                         params.layer_depths, params.n_layers)
                     if model.wt_tracker else None)
            mp = macropore_step(
                params, self.state.macro_moisture, matrix_moist,
                model.dt, ponding=False, n_active=n_act,
            )
            matrix_moist += mp["exchange"]
            macro_out = mp["macro_moisture"]
            macro_drain = mp["bottom_drain"]
            cap = params.porosity - params.f_macro
            matrix_moist = np.clip(matrix_moist, 1e-7, cap)

        vwc_top = matrix_moist[0]
        k_soil = params.thermal_conductivity(np.array([vwc_top]))[0]
        dz_top = 0.5 * (params.layer_depths[1] + params.layer_depths[2])
        fc = params.porosity - params.Sy

        seb = surface_energy_balance(
            sw_down=sw_down,
            T_air=air_temp,
            rh=rh,
            wind=wind,
            T_soil_top=self.state.temperature[0],
            k_soil_top=k_soil,
            dz_top=max(dz_top, 0.003),
            vwc_surface=vwc_top,
            field_capacity=fc,
            albedo=albedo,
            cloud_cover=cloud_cover,
        )
        Ts = seb["Ts"]
        G = seb["G"]

        temp_new, heatflux = soil_temp(
            params, matrix_moist, Ts, self.state.temperature, model.dt,
        )

        LE = seb["LE"]
        et_actual_m_day = max(LE / (1000.0 * 2.45e6), 0.0) * 86400.0

        wt_depth = None
        if model.wt_tracker is not None:
            et_demand = et_actual_m_day / 86400.0 * model.dt
            n_active = model.wt_tracker.find_active_layers(
                params.layer_depths, params.n_layers)

            f_direct = model._wt_evap_fraction(model.wt_tracker.depth)
            direct_wt_evap = et_demand * f_direct
            unsat_et_demand = et_demand * (1.0 - f_direct)

            new_moist, _, w_before, _ = model._solve_moisture_wt(
                matrix_moist, temp_new, 0.0, ponding=False)

            if unsat_et_demand > 1e-12:
                unsat_et_m_day = unsat_et_demand / model.dt * 86400.0
                new_moist, actual_unsat_evap = model._apply_et(
                    new_moist, unsat_et_m_day, n_active)
            else:
                actual_unsat_evap = 0.0

            evap = actual_unsat_evap + direct_wt_evap

            if model.dual_domain:
                cap = params.porosity - params.f_macro
                new_moist[:n_active] = np.clip(
                    new_moist[:n_active], 1e-7, cap)

            w_after = model._unsat_water(new_moist, n_active)
            flux_from_wt = (w_after - w_before) + actual_unsat_evap
            net_wt_flux = -flux_from_wt - direct_wt_evap + macro_drain

            wt_state = model.wt_tracker.update(net_wt_flux, model.dt)
            wt_depth = wt_state.depth
        else:
            result = campbell_moisture(
                params, matrix_moist,
                et_actual_m_day / 86400.0, temp_new,
                dt=int(model.dt), bottom_bc=model.bottom_bc,
            )
            new_moist = result["moisture"]
            evap = result["evap"]

        if model.dual_domain:
            macro_out = model._zero_saturated_macropores(macro_out)

        diag = dict(
            surface_temp=Ts, heatflux=G, evap=evap,
            sw_net=seb["sw_net"], lw_down=seb["lw_down"],
            lw_out=seb["lw_out"], H=seb["H"], LE=seb["LE"],
        )
        if model.wt_tracker is not None:
            diag["recharge"] = wt_state.recharge
            diag["water_table_depth"] = wt_state.depth
            diag["direct_wt_evap"] = direct_wt_evap
            diag["f_direct"] = f_direct
            diag["macro_drain"] = macro_drain

        return (
            ModelState(temperature=temp_new, moisture=new_moist,
                       macro_moisture=macro_out, water_table_depth=wt_depth),
            diag,
        )
