"""Inter-tidal soil temperature and moisture model.

A Python implementation of the core soil dynamics from AED's aed_land.F90,
focused on coupled heat conduction and moisture redistribution in soil
columns subject to periodic tidal wetting and drying.

Quick start
-----------
>>> from intertidal_soil import SoilParams, IntertidalSoilModel, TidalForcing
>>> params = SoilParams(n_layers=20, b=3.0, K_s=5e-4, psi_e=-2.0)
>>> model = IntertidalSoilModel(params, surface_elevation=0.5)
>>> state = model.initialise(air_temp=20.0, deep_temp=18.0)
>>> output = model.run(state, forcing)
"""

from .soil_params import SoilParams
from .model import IntertidalSoilModel, TidalForcing, ModelState, ModelOutput
from .temperature import soil_temp, initial_temp
from .moisture import campbell_moisture
from .atmosphere import wetair, vapprs, potential_evaporation
from .macropore import macropore_step
from .water_table import WaterTableTracker, WaterTableState
from .surface_energy import surface_energy_balance
from .drivers import GiraliaSoilColumn

__all__ = [
    "SoilParams",
    "IntertidalSoilModel",
    "GiraliaSoilColumn",
    "TidalForcing",
    "ModelState",
    "ModelOutput",
    "soil_temp",
    "initial_temp",
    "campbell_moisture",
    "macropore_step",
    "WaterTableTracker",
    "WaterTableState",
    "wetair",
    "vapprs",
    "potential_evaporation",
    "surface_energy_balance",
]
