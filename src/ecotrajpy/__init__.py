"""ecotrajpy: clean-room Python port of Ecological Trajectory Analysis (ETA).

Method credit: De Cáceres et al. (2019) Ecological Monographs 89:e01350;
Sturbois et al. (2021) Ecological Modelling 440:109400; Djeghri et al. (2026)
Ecological Monographs 96:e70058; and the R package ecotraj (GPL), which this
package uses only as a black-box test oracle. No ecotraj source code is copied.
"""
__version__ = "0.1.0"

from .core import Trajectories, define_trajectories, is_metric
from .metrics import (
    trajectory_angles,
    trajectory_directionality,
    trajectory_internal_variation,
    trajectory_lengths,
    trajectory_metrics,
    trajectory_speeds,
)
from .comparison import segment_distances, trajectory_convergence, trajectory_distances
from .compositional import clr
from .nulls import directionality_coords, signflip_path, signflip_test
from .split import increment_split

__all__ = [
    "Trajectories",
    "define_trajectories",
    "is_metric",
    "trajectory_angles",
    "trajectory_directionality",
    "trajectory_internal_variation",
    "trajectory_lengths",
    "trajectory_metrics",
    "trajectory_speeds",
    "segment_distances",
    "trajectory_convergence",
    "trajectory_distances",
    "clr",
    "directionality_coords",
    "signflip_path",
    "signflip_test",
    "increment_split",
    "__version__",
]
