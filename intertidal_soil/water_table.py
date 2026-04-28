"""External water table tracker for coupling with the Campbell unsaturated solver.

The Campbell infiltration model handles the unsaturated zone; this module
tracks the water table position from the recharge flux leaving the base
of the unsaturated column.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class WaterTableState:
    """Snapshot of the water table after an update."""
    depth: float               # m below soil surface (positive downward)
    recharge: float            # m of water this timestep (positive = into WT)
    cumulative_recharge: float # m total since start


class WaterTableTracker:
    """Track water table depth from recharge flux.

    Parameters
    ----------
    initial_depth : m below soil surface
    Sy            : specific yield (drainable porosity, ~0.1-0.3)
    min_depth     : shallowest WT allowed (0 = surface)
    max_depth     : deepest WT allowed (column base)
    """

    def __init__(
        self,
        initial_depth: float,
        Sy: float,
        min_depth: float = 0.0,
        max_depth: float = 1.0,
    ):
        self.depth = initial_depth
        self.Sy = Sy
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.cumulative_recharge = 0.0

    def find_active_layers(self, layer_depths: np.ndarray, n_layers: int) -> int:
        """Number of unsaturated layers above the current water table.

        Includes any layer that straddles the WT (top above, bottom below).
        That layer is truncated at the WT depth by truncated_view().
        """
        n_active = 0
        for i in range(n_layers):
            layer_top = layer_depths[i + 1]
            layer_bot = layer_depths[i + 2]
            if layer_bot <= self.depth:
                n_active = i + 1
            elif layer_top < self.depth:
                n_active = i + 1
                break
            else:
                break
        return n_active

    def update(self, bottom_flux: float, dt: float) -> WaterTableState:
        """Update water table from the recharge flux.

        Parameters
        ----------
        bottom_flux : m of water leaving the unsaturated column this step
                      (positive = downward = recharge into the water table)
        dt          : timestep in seconds (for diagnostics only; bottom_flux
                      is already integrated over the timestep)
        """
        recharge = bottom_flux
        self.cumulative_recharge += recharge

        # WT rises when recharge is positive (water entering from above)
        self.depth -= recharge / self.Sy
        self.depth = np.clip(self.depth, self.min_depth, self.max_depth)

        return WaterTableState(
            depth=self.depth,
            recharge=recharge,
            cumulative_recharge=self.cumulative_recharge,
        )
