"""Inter-tidal soil temperature and moisture model.

Drives the coupled temperature–moisture system with periodic wetting and
drying, switching boundary conditions depending on whether the soil surface
is submerged (wet) or exposed (dry).
"""

import numpy as np
from dataclasses import dataclass, field
from .soil_params import SoilParams
from .temperature import soil_temp, initial_temp
from .moisture import campbell_moisture
from .atmosphere import potential_evaporation
from .macropore import macropore_step
from .water_table import WaterTableTracker


@dataclass
class TidalForcing:
    """Time-series forcing for an inter-tidal simulation.

    All arrays must have the same length (n_times).  Times are in seconds
    from the start of the simulation.
    """
    time: np.ndarray          # seconds from t=0
    water_level: np.ndarray   # m, water surface elevation
    air_temp: np.ndarray      # degC
    water_temp: np.ndarray    # degC
    rain: np.ndarray          # m/day


@dataclass
class ModelState:
    """Snapshot of the soil column state."""
    temperature: np.ndarray   # (n_layers,) degC
    moisture: np.ndarray      # (n_layers,) volumetric water content (matrix)
    is_submerged: bool = False
    macro_moisture: np.ndarray = None  # (n_layers,) macropore water content
    water_table_depth: float = None    # m below surface (None = not tracked)


@dataclass
class ModelOutput:
    """Storage for time-series results."""
    time: list = field(default_factory=list)
    temperature: list = field(default_factory=list)
    moisture: list = field(default_factory=list)
    macro_moisture: list = field(default_factory=list)
    surface_temp: list = field(default_factory=list)
    heatflux: list = field(default_factory=list)
    evap: list = field(default_factory=list)
    is_submerged: list = field(default_factory=list)
    water_table_depth: list = field(default_factory=list)
    recharge: list = field(default_factory=list)
    direct_wt_evap: list = field(default_factory=list)
    f_direct: list = field(default_factory=list)

    def to_arrays(self) -> dict:
        d = dict(
            time=np.array(self.time),
            temperature=np.array(self.temperature),
            moisture=np.array(self.moisture),
            surface_temp=np.array(self.surface_temp),
            heatflux=np.array(self.heatflux),
            evap=np.array(self.evap),
            is_submerged=np.array(self.is_submerged),
        )
        if self.macro_moisture:
            d["macro_moisture"] = np.array(self.macro_moisture)
        if self.water_table_depth:
            d["water_table_depth"] = np.array(self.water_table_depth)
        if self.recharge:
            d["recharge"] = np.array(self.recharge)
        if self.direct_wt_evap:
            d["direct_wt_evap"] = np.array(self.direct_wt_evap)
        if self.f_direct:
            d["f_direct"] = np.array(self.f_direct)
        return d


class IntertidalSoilModel:
    """Coupled soil temperature–moisture model for inter-tidal flats.

    The model switches boundary conditions based on submergence:

    **Dry (exposed):**
      - Surface temperature BC = air temperature
      - Moisture evolves via Campbell (1985) Richards equation
      - Evaporation removes water from the surface

    **Wet (submerged):**
      - Surface temperature BC = water temperature
      - Moisture is set to near-saturation (porosity-based)
      - No evaporation
    """

    def __init__(
        self,
        params: SoilParams,
        surface_elevation: float,
        dt: float = 900.0,
        bottom_bc: str = "no_flux",
        evap_scale: float = 1.0,
        fixed_et: float = None,
        water_table_depth: float = None,
        evap_extinction_depth: float = 0.10,
    ):
        """
        Parameters
        ----------
        params            : SoilParams — soil column properties
        surface_elevation : m, elevation of the soil surface (datum-referenced)
        dt                : model time-step in seconds (default 900 = 15 min)
        bottom_bc         : bottom boundary — "no_flux" (sealed, water table
                            rises), "fixed_head" (saturated at real depth),
                            "free_drain" (free-draining bottom).
                            Ignored when water_table_depth is set (uses
                            fixed_head on the truncated unsaturated column).
        evap_scale        : multiplier on evapotranspiration (0 = no evap)
        fixed_et          : m/day, constant ET rate. When set, overrides the
                            PET calculation (evap_scale is ignored).
        water_table_depth : m below surface. If set, enables coupled WT
                            tracking: the Campbell solver operates only on
                            the unsaturated zone above the WT, and recharge
                            raises/lowers the WT each step.
        evap_extinction_depth : m, depth below which direct evaporation from
                            the saturated zone is zero. Exponential decay from
                            surface (f=1) to this depth (f~0).
        """
        self.params = params
        self.surface_elevation = surface_elevation
        self.dt = dt
        self.bottom_bc = bottom_bc
        self.evap_scale = evap_scale
        self.fixed_et = fixed_et
        self.d_ext = evap_extinction_depth
        self.wt_tracker = None
        self._prev_n_active = 0
        if water_table_depth is not None:
            self.wt_tracker = WaterTableTracker(
                initial_depth=water_table_depth,
                Sy=params.Sy,
                min_depth=0.0,
                max_depth=params.max_depth,
            )
            self._prev_n_active = self.wt_tracker.find_active_layers(
                params.layer_depths, params.n_layers)

    @property
    def dual_domain(self) -> bool:
        return self.params.f_macro > 0

    def _enforce_matrix_cap(self, matrix_moist, macro_moist, macro_drain,
                            n_active=None):
        """Clip matrix to its capacity, returning excess to macropores.

        Any water above the matrix cap (porosity - f_macro) is transferred
        back to the macropore domain. If macropores are also full, the
        excess is added to macro_drain (direct WT recharge).

        Only operates on the unsaturated layers (0..n_active-1) when
        n_active is set.  Layers below the WT are at full porosity by
        definition and must not be clipped.
        """
        cap = self.params.porosity - self.params.f_macro
        f_macro = self.params.f_macro
        depths = self.params.layer_depths
        n = n_active if n_active is not None else self.params.n_layers

        for i in range(n):
            excess = matrix_moist[i] - cap
            if excess > 0:
                matrix_moist[i] = cap
                space = f_macro - macro_moist[i]
                if space >= excess:
                    macro_moist[i] += excess
                else:
                    macro_moist[i] = f_macro
                    dz = depths[i + 2] - depths[i + 1]
                    macro_drain += (excess - space) * dz

        matrix_moist[:n] = np.clip(matrix_moist[:n], 1e-7, cap)
        return matrix_moist, macro_moist, macro_drain

    def _zero_saturated_macropores(self, macro_moist):
        """Zero macropore moisture in layers below the water table.

        Below the WT, all pore space (matrix + macro) is part of the
        saturated zone tracked by the WT reservoir.  Macropore moisture
        must be zero there to avoid double-counting.
        """
        if self.wt_tracker is None or macro_moist is None:
            return macro_moist
        n_active = self.wt_tracker.find_active_layers(
            self.params.layer_depths, self.params.n_layers)
        macro_moist[n_active:] = 0.0
        return macro_moist

    def initialise(
        self,
        air_temp: float = 20.0,
        deep_temp: float = 18.0,
        init_moisture: float = 0.2,
        spin_up_days: float = 30.0,
    ) -> ModelState:
        """Create an initial soil state by spinning up temperature."""
        self.params.deep_temp = deep_temp

        temp = initial_temp(
            self.params,
            vwc_init=init_moisture,
            air_temp=air_temp,
            deep_temp=deep_temp,
            spin_up_days=spin_up_days,
            dt=self.dt,
        )
        moisture = np.full(self.params.n_layers, init_moisture)
        macro = np.zeros(self.params.n_layers) if self.dual_domain else None

        wt_depth = None
        if self.wt_tracker is not None:
            wt_depth = self.wt_tracker.depth
            n_active = self.wt_tracker.find_active_layers(
                self.params.layer_depths, self.params.n_layers)
            moisture[n_active:] = self.params.porosity

        return ModelState(temperature=temp, moisture=moisture,
                          is_submerged=False, macro_moisture=macro,
                          water_table_depth=wt_depth)

    def is_submerged(self, water_level: float) -> bool:
        return water_level >= self.surface_elevation

    def step(
        self,
        state: ModelState,
        water_level: float,
        air_temp: float,
        water_temp: float,
        rain: float,
    ) -> tuple[ModelState, dict]:
        """Advance the model by one time-step.

        Parameters
        ----------
        state       : current ModelState
        water_level : m, current water surface elevation
        air_temp    : degC
        water_temp  : degC
        rain        : m/day

        Returns
        -------
        new_state : updated ModelState
        diagnostics : dict of scalar diagnostics for this step
        """
        submerged = self.is_submerged(water_level)

        if submerged:
            new_state, diag = self._step_wet(state, water_temp)
        else:
            new_state, diag = self._step_dry(state, air_temp, rain)

        new_state.is_submerged = submerged
        diag["is_submerged"] = submerged
        return new_state, diag

    def _wt_evap_fraction(self, wt_depth: float) -> float:
        """Fraction of ET demand taken directly from the saturated zone.

        Exponential decay: 1.0 at surface, ~0 at extinction depth.
        k chosen so exp(-k * d_ext) ~ 0.007.
        """
        if self.d_ext <= 0:
            return 0.0
        k = 5.0 / self.d_ext
        return np.exp(-k * wt_depth)

    def _solve_moisture_wt(self, matrix_moist, temp, et_rate,
                           ponding, ha=0.5):
        """Run Campbell solver on the unsaturated zone above the WT.

        Returns (full_moisture, bottom_flux, w_before, evap).
        bottom_flux : solver's Darcy flux at the WT boundary (m, positive=down)
        w_before    : total water in unsaturated zone after fc cap, before
                      solver (m, for mass-balance WT tracking in dry steps)
        """
        n = self.params.n_layers
        n_active = self.wt_tracker.find_active_layers(
            self.params.layer_depths, n)

        prev = self._prev_n_active
        self._prev_n_active = n_active

        if n_active == 0:
            return (np.full(n, self.params.porosity), 0.0, 0.0, 0.0)

        active_moist = matrix_moist[:n_active].copy()
        fc = self.params.porosity - self.params.Sy
        for i in range(prev, n_active):
            if active_moist[i] >= self.params.porosity - 1e-6:
                active_moist[i] = fc

        w_before = self._unsat_water(active_moist, n_active)

        trunc = self.params.truncated_view(n_active,
                                           wt_depth=self.wt_tracker.depth)
        result = campbell_moisture(
            trunc, active_moist, et_rate, temp[:n_active],
            dt=int(self.dt), ha=ha, ponding=ponding,
            bottom_bc="fixed_head",
        )

        full_moist = matrix_moist.copy()
        full_moist[:n_active] = result["moisture"]
        full_moist[n_active:] = self.params.porosity
        return (full_moist, result["bottom_flux"], w_before, result["evap"])

    def _step_wet(self, state: ModelState, water_temp: float) -> tuple[ModelState, dict]:
        """Submerged: water temperature drives surface; wetting front infiltrates."""
        matrix_moist = state.moisture.copy()
        macro_out = None
        macro_drain = 0.0

        if self.dual_domain:
            n_act = (self.wt_tracker.find_active_layers(
                         self.params.layer_depths, self.params.n_layers)
                     if self.wt_tracker else None)
            mp = macropore_step(
                self.params, state.macro_moisture, matrix_moist,
                self.dt, ponding=True, n_active=n_act,
            )
            matrix_moist += mp["exchange"]
            macro_out = mp["macro_moisture"]
            macro_drain = mp["bottom_drain"]
            matrix_moist, macro_out, macro_drain = self._enforce_matrix_cap(
                matrix_moist, macro_out, macro_drain, n_active=n_act)

        wt_depth = None
        if self.wt_tracker is not None:
            new_moist, bot_flux, _, evap = self._solve_moisture_wt(
                matrix_moist, state.temperature, 0.0, ponding=True, ha=1.0)
            cap_overflow = 0.0
            if self.dual_domain:
                n_act_now = self.wt_tracker.find_active_layers(
                    self.params.layer_depths, self.params.n_layers)
                new_moist, macro_out, cap_overflow = self._enforce_matrix_cap(
                    new_moist, macro_out, 0.0, n_active=n_act_now)
            wt_recharge = bot_flux + macro_drain + cap_overflow
            wt_state = self.wt_tracker.update(wt_recharge, self.dt)
            wt_depth = wt_state.depth
        else:
            result = campbell_moisture(
                self.params, matrix_moist, 0.0, state.temperature,
                dt=int(self.dt), ha=1.0, ponding=True,
                bottom_bc=self.bottom_bc,
            )
            new_moist = result["moisture"]
            bot_flux = 0.0
            if self.dual_domain:
                new_moist, macro_out, _ = self._enforce_matrix_cap(
                    new_moist, macro_out, 0.0)

        if self.dual_domain:
            macro_out = self._zero_saturated_macropores(macro_out)

        temp_new, heatflux = soil_temp(
            self.params, new_moist, water_temp, state.temperature, self.dt
        )

        diag = dict(surface_temp=water_temp, heatflux=heatflux, evap=0.0)
        if self.wt_tracker is not None:
            diag["recharge"] = wt_state.recharge
            diag["water_table_depth"] = wt_state.depth
            diag["direct_wt_evap"] = 0.0
            diag["f_direct"] = 0.0
            diag["macro_drain"] = macro_drain

        return (
            ModelState(temperature=temp_new, moisture=new_moist,
                       macro_moisture=macro_out, water_table_depth=wt_depth),
            diag,
        )

    def _apply_et(self, moisture: np.ndarray, et_m_day: float,
                  n_active: int = None) -> tuple[np.ndarray, float]:
        """Remove water from the top of the unsaturated zone at a fixed rate.

        Parameters
        ----------
        moisture  : full column moisture array (modified in-place)
        et_m_day  : evapotranspiration rate in m/day
        n_active  : number of unsaturated layers (None = all layers)

        Returns (moisture, actual_evap_m) where actual_evap_m is the water
        actually removed this timestep (m), which may be less than demanded
        if the soil dries out.
        """
        demand = et_m_day / 86400.0 * self.dt  # m of water to remove
        residual = 0.02
        remaining = demand
        n = n_active if n_active is not None else self.params.n_layers
        depths = self.params.layer_depths
        wt = self.wt_tracker.depth if self.wt_tracker else None

        for i in range(min(n, len(moisture))):
            if remaining <= 0:
                break
            top = depths[i + 1]
            bot = depths[i + 2]
            if wt is not None and bot > wt:
                bot = wt
            dz = bot - top
            if dz <= 0:
                continue
            available = (moisture[i] - residual) * dz
            if available <= 0:
                continue
            take = min(remaining, available)
            moisture[i] -= take / dz
            remaining -= take

        return moisture, demand - remaining

    def _unsat_water(self, moisture, n_active):
        """Total water stored in the unsaturated zone (m).

        Truncates the last active layer at the WT depth so the mass
        balance only counts the unsaturated portion.
        """
        depths = self.params.layer_depths
        wt = self.wt_tracker.depth if self.wt_tracker else None
        total = 0.0
        for i in range(n_active):
            top = depths[i + 1]
            bot = depths[i + 2]
            if wt is not None and bot > wt:
                bot = wt
            total += moisture[i] * (bot - top)
        return total

    def _step_dry(self, state: ModelState, air_temp: float, rain: float) -> tuple[ModelState, dict]:
        """Exposed: air temperature drives surface; moisture evolves."""
        temp_new, heatflux = soil_temp(
            self.params, state.moisture, air_temp, state.temperature, self.dt
        )

        matrix_moist = state.moisture.copy()
        macro_out = None
        macro_drain = 0.0

        if self.dual_domain:
            n_act = (self.wt_tracker.find_active_layers(
                         self.params.layer_depths, self.params.n_layers)
                     if self.wt_tracker else None)
            mp = macropore_step(
                self.params, state.macro_moisture, matrix_moist,
                self.dt, ponding=False, n_active=n_act,
            )
            matrix_moist += mp["exchange"]
            macro_out = mp["macro_moisture"]
            macro_drain = mp["bottom_drain"]
            cap = self.params.porosity - self.params.f_macro
            matrix_moist = np.clip(matrix_moist, 1e-7, cap)

        if self.fixed_et is not None:
            et = self.fixed_et
        else:
            et = potential_evaporation(rain, air_temp) * self.evap_scale

        wt_depth = None
        if self.wt_tracker is not None:
            et_demand = et / 86400.0 * self.dt  # m of water this timestep
            n_active = self.wt_tracker.find_active_layers(
                self.params.layer_depths, self.params.n_layers)

            # (c) Direct evaporation from the saturated zone — exponential
            #     decay with WT depth, zero beyond extinction depth.
            f_direct = self._wt_evap_fraction(self.wt_tracker.depth)
            direct_wt_evap = et_demand * f_direct
            unsat_et_demand = et_demand * (1.0 - f_direct)

            # Solver handles redistribution + capillary rise via fixed_head BC.
            # w_before is computed after the fc cap (excludes Sy release).
            new_moist, _, w_before, _ = self._solve_moisture_wt(
                matrix_moist, temp_new, 0.0, ponding=False)

            # Apply unsaturated ET to the top layers
            if unsat_et_demand > 1e-12:
                unsat_et_m_day = unsat_et_demand / self.dt * 86400.0
                new_moist, actual_unsat_evap = self._apply_et(
                    new_moist, unsat_et_m_day, n_active)
            else:
                actual_unsat_evap = 0.0

            evap = actual_unsat_evap + direct_wt_evap

            if self.dual_domain:
                cap = self.params.porosity - self.params.f_macro
                new_moist[:n_active] = np.clip(new_moist[:n_active], 1e-7, cap)

            w_after = self._unsat_water(new_moist, n_active)

            # (a+b) from column mass balance (after fc cap):
            # flux_from_wt = net water entering unsat from the WT via solver
            flux_from_wt = (w_after - w_before) + actual_unsat_evap
            # For the WT: capillary rise is a loss, drainage is a gain.
            # (d) macropore bottom drain is direct recharge (positive into WT)
            net_wt_flux = -flux_from_wt - direct_wt_evap + macro_drain

            wt_state = self.wt_tracker.update(net_wt_flux, self.dt)
            wt_depth = wt_state.depth
        else:
            result = campbell_moisture(
                self.params, matrix_moist,
                et / 86400.0, temp_new,
                dt=int(self.dt), bottom_bc=self.bottom_bc,
            )
            new_moist = result["moisture"]
            evap = result["evap"]
            if self.fixed_et is not None:
                new_moist, evap = self._apply_et(new_moist, et)

        if self.dual_domain:
            macro_out = self._zero_saturated_macropores(macro_out)

        diag = dict(surface_temp=air_temp, heatflux=heatflux, evap=evap)
        if self.wt_tracker is not None:
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

    def run(self, state: ModelState, forcing: TidalForcing) -> ModelOutput:
        """Run the model over a full forcing time-series.

        The forcing is interpolated to the model time-step internally.
        """
        t_start = forcing.time[0]
        t_end = forcing.time[-1]
        times = np.arange(t_start, t_end, self.dt)

        wl_interp = np.interp(times, forcing.time, forcing.water_level)
        at_interp = np.interp(times, forcing.time, forcing.air_temp)
        wt_interp = np.interp(times, forcing.time, forcing.water_temp)
        rn_interp = np.interp(times, forcing.time, forcing.rain)

        output = ModelOutput()
        current = state

        for i, t in enumerate(times):
            current, diag = self.step(
                current, wl_interp[i], at_interp[i], wt_interp[i], rn_interp[i]
            )
            output.time.append(t)
            output.temperature.append(current.temperature.copy())
            output.moisture.append(current.moisture.copy())
            if current.macro_moisture is not None:
                output.macro_moisture.append(current.macro_moisture.copy())
            output.surface_temp.append(diag["surface_temp"])
            output.heatflux.append(diag["heatflux"])
            output.evap.append(diag["evap"])
            output.is_submerged.append(diag["is_submerged"])
            if "water_table_depth" in diag:
                output.water_table_depth.append(diag["water_table_depth"])
            if "recharge" in diag:
                output.recharge.append(diag["recharge"])
            if "direct_wt_evap" in diag:
                output.direct_wt_evap.append(diag["direct_wt_evap"])
            if "f_direct" in diag:
                output.f_direct.append(diag["f_direct"])

        return output
