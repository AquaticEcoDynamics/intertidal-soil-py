# Giralia Intertidal Soil Modelling Workflow

## Site Description

Giralia is an arid-zone mangrove system on the eastern shore of Exmouth Gulf,
Western Australia (~22.1S, 114.6E). The tidal flat transitions from subtidal
channels through low/mid/high intertidal zones to supratidal salt flats over
a distance of several kilometres. Soil is sandy-clay with macropore structure
(bioturbation, crab burrows, root channels) that varies along the inundation
gradient: abundant near creeks, negligible on the dry supratidal flats.

## Data Sources

### Hydrodynamic forcing (12 sites)

Located in `hydro/`. Each CSV provides hourly data from
2016-01-01 to 2020-12-31:

| Column | Units | Description |
|--------|-------|-------------|
| water_depth | m | Depth of water above the soil surface (0 = dry) |
| shear_stress | Pa | Bed shear stress |
| sea_tem | degC | Water column temperature (valid when wet) |
| x_velocity, y_velocity | m/s | Current speed components |

Sites span a wide inundation gradient:

| Category | Sites | Wet fraction | f_macro |
|----------|-------|-------------|---------|
| Subtidal | 4, 12 | 79-100% | 0.030 |
| Low intertidal | 5, 7 | 49-56% | 0.025 |
| Mid intertidal | 1, 3 | 32-33% | 0.015 |
| High intertidal | 6 | 20% | 0.008 |
| Supratidal | 2, 8, 9, 10, 11 | 5-10% | 0.002 |

### Meteorological forcing

Located in `data/giralia_weather.csv`. Hourly ERA5 reanalysis data from
Open-Meteo archive API for 2016-2020 at (-22.1, 114.6):

| Variable | Units | Mean | Description |
|----------|-------|------|-------------|
| temperature_2m | degC | 25.8 | Air temperature |
| relative_humidity_2m | % | 50 | Relative humidity |
| wind_speed_10m | km/h | 17.0 | 10m wind speed (convert to m/s: /3.6) |
| shortwave_radiation | W/m2 | 263 | Global horizontal irradiance |
| diffuse_radiation | W/m2 | 53 | Diffuse component |
| direct_normal_irradiance | W/m2 | 323 | Direct normal irradiance |
| precipitation | mm | 0.02 | Hourly rainfall |
| cloud_cover | % | 22 | Total cloud cover |
| dew_point_2m | degC | 12.9 | Dew point temperature |

## Model Architecture

### Callable forward step

The model is designed for integration into a larger coupled system. The
`GiraliaSoilColumn` class (in `intertidal_soil.drivers`) wraps the
intertidal soil model with a callable `.step()` method:

```python
from intertidal_soil.drivers import GiraliaSoilColumn
from giralia_config import get_soil_params

col = GiraliaSoilColumn(site_id=1, soil_kwargs=get_soil_params(1),
                         dt=3600.0, wt_depth=0.8)
col.initialise(air_temp=30.0, deep_temp=26.0, init_moisture=0.15)

# Each timestep:
state, diag = col.step(
    water_depth=0.15,       # m (0 = dry)
    water_temp=28.0,        # degC (from hydro model)
    air_temp=32.0,          # degC
    sw_down=800.0,          # W/m2
    rh=50.0,                # %
    wind=5.0,               # m/s
    rain=0.0,               # m/day
    cloud_cover=0.2,        # fraction 0-1
    albedo=0.25,            # surface albedo (can vary per step)
)
# state.temperature  -> (n_layers,) soil temp profile
# state.moisture     -> (n_layers,) VWC profile
# diag keys: surface_temp, heatflux, evap, water_table_depth, ...
```

### Surface temperature boundary condition

**When submerged**: the hydrodynamic model's water temperature (`sea_tem`) is
used as a Dirichlet BC. The hydro model has already solved its own energy
balance on the water column.

**When exposed**: a surface energy balance solves for surface temperature Ts:

    SW_net + LW_down - LW_out - H - LE = G

where G = k/dz * (Ts - T_soil_top) couples to the soil heat equation. The
Newton-iteration SEB (`surface_energy.py`) uses:

- **Shortwave**: `sw_down * (1 - albedo)`, albedo default 0.25 (per-step argument)
- **Longwave down**: Brutsaert (1975) clear-sky emissivity + cloud correction
- **Longwave out**: `eps * sigma * Ts^4`, emissivity = 0.95
- **Sensible heat**: bulk aerodynamic, `H = rho * Cp * Ch * u * (Ts - Ta)`
- **Latent heat**: bulk aerodynamic, moisture-limited via `beta = min(VWC_top / fc, 1)`

The resulting Ts is passed as a Dirichlet BC to the implicit heat solver
(unconditionally stable; the Neumann flux approach was unstable with the
thin 3mm surface layer).

### Soil parameters

Base parameters for sandy-clay tidal flat:

| Parameter | Value | Units |
|-----------|-------|-------|
| n_layers | 20 | - |
| max_depth | 1.0 | m |
| psi_e | -1.5 | J/kg |
| K_s | 2e-4 | kg s/m^3 |
| b | 4.0 | - |
| bulk_density | 1.55 | Mg/m^3 |
| Sy | 0.15 | - |
| deep_temp | 26.0 | degC |

Macropore parameters vary with inundation category (see table above).

## Scripts

| Script | Purpose |
|--------|---------|
| `fetch_weather.py` | Download Open-Meteo weather data to `data/` |
| `giralia_config.py` | Site definitions, soil parameters, data loaders |
| `run_single_site.py` | Single-site test with diagnostic plot |
| `run_multi_site.py` | All 12 sites for a specified period |
| `run_heat_diagnostic.py` | SEB component breakdown |
| `run_albedo_comparison.py` | Bare vs algal mat albedo comparison |

## Known Issues and Notes

1. **Wind speed units**: Open-Meteo returns km/h; must divide by 3.6 for m/s.
2. **sea_tem fill values**: When water_depth = 0, sea_tem may contain fill
   values (6.0 degC at simulation start). These are never used since the
   model checks submergence before applying the water temperature BC.
3. **ERA5 wind bias**: Reanalysis winds for coastal grid cells may over-
   represent open-ocean conditions. A sheltering factor could be applied
   for more protected sites.
4. **Surface layer stability**: The SEB computes Ts then passes it as
   Dirichlet (not Neumann flux) to avoid numerical oscillation with the
   thin surface layer (dz ~ 3mm).
