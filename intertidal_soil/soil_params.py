"""Soil parameter definitions and defaults."""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SoilParams:
    """Physical and hydraulic properties of a soil column.

    Campbell (1985) parameterisation for hydraulic functions:
        psi = psi_e * (theta_s / theta) ^ b
        K   = K_s   * (psi_e / psi) ^ n       where n = 2 + 3/b
    """

    # Layer geometry
    n_layers: int = 20
    max_depth: float = 1.0    # m, total column depth
    layer_depths: np.ndarray = None  # m, depth to bottom of each layer

    # Campbell hydraulic parameters
    psi_e: float = -2.0       # air-entry potential, J/kg (negative)
    K_s: float = 5e-4         # saturated hydraulic conductivity, kg s / m^3
    b: float = 3.0            # Campbell 'b' parameter
    bulk_density: float = 1.5 # Mg/m^3
    mineral_density: float = 2.6  # Mg/m^3

    # Porosity (derived from bulk_density / mineral_density if not set)
    porosity: float = None

    # Root / vegetation
    root_density: np.ndarray = None  # m/m^3 per layer
    root_radius: float = 0.001      # m
    lai: float = 0.1                 # leaf area index

    # Thermal properties of mineral fraction
    k_mineral: float = 2.5    # W/(m K) thermal conductivity of mineral
    k_water: float = 0.57     # W/(m K) thermal conductivity of water
    k_air: float = 0.025      # W/(m K) thermal conductivity of air
    c_mineral: float = 2.0e6  # J/(m^3 K) volumetric heat capacity mineral
    c_water: float = 4.18e6   # J/(m^3 K) volumetric heat capacity water
    c_air: float = 1.25e3     # J/(m^3 K) volumetric heat capacity air

    # Macropore (dual-domain) parameters
    f_macro: float = 0.0         # macropore porosity fraction (0 = single domain)
    k_macro: float = 0.01        # m/s, macropore gravity drainage conductivity
    alpha_exchange: float = 5e-5  # 1/s, macro-matrix exchange rate (~5 h timescale)

    # Water table parameters
    Sy: float = 0.2              # specific yield (drainable porosity)

    # Deep boundary
    deep_temp: float = 18.0   # degC, temperature at bottom boundary

    def __post_init__(self):
        if self.porosity is None:
            self.porosity = 1.0 - self.bulk_density / self.mineral_density

        if self.layer_depths is None:
            self.layer_depths = self._default_layer_depths()

        if self.root_density is None:
            self.root_density = np.full(self.n_layers, 0.1)

    def _default_layer_depths(self) -> np.ndarray:
        """Geometric spacing: fine near surface, total depth = max_depth.

        Finds a growth ratio so that the sum of layer thicknesses
        (dz_0, dz_0*r, dz_0*r^2, ...) equals max_depth.
        """
        n = self.n_layers
        dz_0 = 0.005  # 5 mm thinnest layer
        target = self.max_depth - 0.001  # subtract the surface offset

        # Solve for growth ratio: dz_0 * (r^n - 1)/(r - 1) = target
        def residual(r):
            if abs(r - 1.0) < 1e-8:
                return dz_0 * n - target
            return dz_0 * (r ** n - 1.0) / (r - 1.0) - target

        # Bisection
        r_lo, r_hi = 1.0, 3.0
        for _ in range(100):
            r_mid = 0.5 * (r_lo + r_hi)
            if residual(r_mid) < 0:
                r_lo = r_mid
            else:
                r_hi = r_mid
        growth = 0.5 * (r_lo + r_hi)

        depths = np.zeros(n + 2)
        depths[1] = 0.001
        for i in range(1, n + 1):
            depths[i + 1] = depths[i] + dz_0 * growth ** (i - 1)
        return depths

    def layer_centres(self) -> np.ndarray:
        d = self.layer_depths
        return 0.5 * (d[1:self.n_layers + 1] + d[2:self.n_layers + 2])

    def theta_sat(self) -> float:
        return self.porosity

    def thermal_conductivity(self, theta: np.ndarray) -> np.ndarray:
        """Johansen-style effective thermal conductivity as f(moisture)."""
        Sr = np.clip(theta / self.porosity, 0.0, 1.0)
        Ke = np.log10(np.maximum(Sr, 1e-10)) + 1.0
        Ke = np.clip(Ke, 0.0, 1.0)
        k_dry = (0.135 * self.bulk_density * 1e3 + 64.7) / (
            self.mineral_density * 1e3 - 0.947 * self.bulk_density * 1e3
        )
        k_sat = (
            self.k_mineral ** (1.0 - self.porosity)
            * self.k_water ** self.porosity
        )
        return k_dry + (k_sat - k_dry) * Ke

    def heat_capacity(self, theta: np.ndarray) -> np.ndarray:
        """Volumetric heat capacity as f(moisture)."""
        return (
            (1.0 - self.porosity) * self.c_mineral
            + theta * self.c_water
            + (self.porosity - theta) * self.c_air
        )

    def truncated_view(self, n_active: int, wt_depth: float = None) -> 'SoilParams':
        """Return a lightweight copy with only the top n_active layers.

        Used to run the Campbell solver on the unsaturated zone above the
        water table.  The bottom of the truncated column is positioned at
        *wt_depth* (if given) so the fixed_head BC sits at the true WT.
        """
        from dataclasses import replace
        depths = self.layer_depths[:n_active + 2].copy()
        if wt_depth is not None and len(depths) > n_active + 1:
            depths[n_active + 1] = wt_depth
        roots = self.root_density[:n_active].copy()
        return replace(self, n_layers=n_active,
                       layer_depths=depths, root_density=roots)
