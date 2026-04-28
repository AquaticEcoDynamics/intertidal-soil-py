# Intertidal Soil Model

Coupled heat conduction and moisture redistribution model for soil columns
subject to periodic tidal wetting and drying. Translated from the AED
`aed_land.F90` Fortran module into Python.

## Features

- **Campbell (1985) infiltration solver** -- implicit finite-difference solution
  with Thomas algorithm, Fortran-style indexing
- **Implicit heat conduction** -- backward-Euler with moisture-dependent thermal
  properties (Johansen model)
- **Surface energy balance** -- Newton-iteration solver for exposed surface
  temperature, with moisture-limited latent heat (beta factor)
- **Macropore dual-domain** -- gravity cascade + matrix exchange for bioturbated
  soils (crab burrows, root channels)
- **Coupled water table tracking** -- external WT dynamics with specific yield,
  capillary rise, and direct evaporation
- **Callable step interface** -- `GiraliaSoilColumn.step()` advances one timestep
  given external forcing, designed for coupling with hydrodynamic models

## Installation

```bash
pip install -e .

# With plotting support:
pip install -e ".[plot]"

# For development (includes pytest):
pip install -e ".[dev]"
```

Requires Python >= 3.9, NumPy, and Pandas.

## Quick Start

```python
from intertidal_soil import SoilParams, IntertidalSoilModel, TidalForcing
import numpy as np

params = SoilParams(n_layers=20, b=4.0, K_s=2e-4, psi_e=-1.5)
model = IntertidalSoilModel(params, surface_elevation=0.0, dt=3600)
state = model.initialise(air_temp=25.0, deep_temp=22.0)

# Or use the step-based driver for coupling:
from intertidal_soil import GiraliaSoilColumn

col = GiraliaSoilColumn(1, dict(n_layers=20, b=4.0, K_s=2e-4, psi_e=-1.5))
col.initialise(air_temp=25.0, deep_temp=26.0)

state, diag = col.step(
    water_depth=0.0,    # m
    water_temp=28.0,    # degC
    air_temp=35.0,      # degC
    sw_down=800.0,      # W/m2
    rh=50.0,            # %
    wind=3.0,           # m/s
    rain=0.0,           # m/day
    albedo=0.25,        # surface albedo (can vary per step)
)
```

## Tests

```bash
pytest
```

## Examples

- `examples/` -- demo and diagnostic scripts (single-domain, dual-domain,
  animations, coupled water table)
- `examples/giralia/` -- full case study for the Giralia arid-zone mangrove
  system, Exmouth Gulf, Western Australia. Runs 12 monitoring sites with
  hourly forcing from a calibrated Delft-FM hydrodynamic model and ERA5
  reanalysis weather data.

## Documentation

Technical manuals covering the mathematical formulation, numerical methods,
and implementation details are in `docs/`:

- `technical_manual_moisture.md` -- Campbell infiltration solver, macropore
  dual-domain, coupled water table tracking
- `technical_manual_temperature.md` -- heat conduction, surface energy balance,
  thermal properties, albedo sensitivity

## Architecture

The `intertidal_soil` package contains:

| Module | Purpose |
|--------|---------|
| `soil_params.py` | `SoilParams` dataclass -- layer geometry, hydraulic and thermal properties |
| `moisture.py` | Campbell (1985) implicit infiltration solver |
| `temperature.py` | Implicit backward-Euler heat conduction |
| `surface_energy.py` | Newton-iteration surface energy balance |
| `macropore.py` | Dual-domain gravity cascade + exchange |
| `water_table.py` | Water table tracker with specific yield |
| `atmosphere.py` | Vapour pressure, humid air properties, PET |
| `model.py` | `IntertidalSoilModel` orchestrator |
| `drivers.py` | `GiraliaSoilColumn` callable step interface |

## License

MIT License. See [LICENSE](LICENSE).
