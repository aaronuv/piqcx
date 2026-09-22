"""piqcx: JAX Path integral Quantum Control."""

from .annealing import SCHEDULES, AnnealingConfig, schedule_diffusion
from .config import configure, devices, runtime
from .problem import ControlProblem, make_problem
from .smoothing import SmoothingConfig
from .solver import PiQC, PiQCResult, infer_time_grid, optimize

__version__ = "0.1.0"

__all__ = [
    "SCHEDULES",
    "AnnealingConfig",
    "ControlProblem",
    "PiQC",
    "PiQCResult",
    "SmoothingConfig",
    "configure",
    "devices",
    "infer_time_grid",
    "make_problem",
    "optimize",
    "runtime",
    "schedule_diffusion",
    "__version__",
]
