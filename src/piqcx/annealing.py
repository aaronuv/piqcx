"""Diffusion annealing schedules for closed-system PiQC."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ScheduleName = Literal[
    "constant",
    "linear",
    "inverse",
    "inverse_square",
    "exponential",
    "steps",
    "log_steps",
    "const_plus_exp",
]

SCHEDULES: tuple[ScheduleName, ...] = (
    "constant",
    "linear",
    "inverse",
    "inverse_square",
    "exponential",
    "steps",
    "log_steps",
    "const_plus_exp",
)


@dataclass(frozen=True)
class AnnealingConfig:
    """Toggle and shape the synthetic-noise schedule ``D_p``.

    ``R`` is held fixed; ``lambda_p = R * D_p`` is updated each IS step,
    matching the PI condition ``R = lambda D^{-1}``.
    """

    enabled: bool = False
    schedule: ScheduleName = "exponential"
    d_init: float = 1e-1
    d_final: float = 1e-10
    n_plateaus: int = 20
    floor: float = 1e-16

    def resolve_bounds(self, diffusion: float) -> tuple[float, float]:
        if not self.enabled:
            return float(diffusion), float(diffusion)
        return float(self.d_init), float(self.d_final)


def schedule_diffusion(
    n_iterations: int,
    *,
    d_init: float,
    d_final: float,
    schedule: ScheduleName = "exponential",
    n_plateaus: int = 20,
    floor: float = 1e-16,
) -> np.ndarray:
    """Return a length-``n_iterations`` array of Ito diffusion scales."""
    if n_iterations < 1:
        raise ValueError("n_iterations must be >= 1")
    d0 = max(float(d_init), floor)
    d1 = max(float(d_final), floor)
    if n_iterations == 1:
        return np.array([d0], dtype=np.float64)

    j = np.arange(1, n_iterations + 1, dtype=np.float64)
    n = float(n_iterations)

    if schedule in ("constant", "ct"):
        values = np.full(n_iterations, d0)
    elif schedule == "linear":
        a = (d1 - d0) / (n - 1.0)
        b = d0 - a
        values = a * j + b
    elif schedule in ("inverse", "1/j"):
        a = (1.0 / d1 - 1.0 / d0) / (n - 1.0)
        b = 1.0 / d0 - a
        values = 1.0 / (a * j + b)
    elif schedule in ("inverse_square", "1/j^2"):
        a = (1.0 / np.sqrt(d1) - 1.0 / np.sqrt(d0)) / (n - 1.0)
        b = 1.0 / np.sqrt(d0) - a
        values = 1.0 / (a * j + b) ** 2
    elif schedule in ("exponential", "exp"):
        a = (np.log(d1) - np.log(d0)) / (n - 1.0)
        b = np.log(d0) - a
        values = np.exp(a * j + b)
    elif schedule == "steps":
        nsteps = max(1, int(n_plateaus))
        values = np.empty(n_iterations, dtype=np.float64)
        length = n_iterations / nsteps
        ratio = d1 / d0
        for k in range(1, nsteps + 1):
            start = int(round((k - 1) * length))
            stop = int(round(k * length))
            values[start:stop] = d0 * (ratio ** (k / nsteps))
    elif schedule in ("log_steps", "logsteps"):
        nsteps = max(1, int(n_plateaus))
        values = np.empty(n_iterations, dtype=np.float64)
        remaining = n_iterations
        end = 0
        ratio = d1 / d0
        for k in range(nsteps):
            start = end
            chunk = max(1, int(np.ceil(remaining / 2.0))) if k < nsteps - 1 else n_iterations - start
            end = min(n_iterations, start + chunk)
            remaining = n_iterations - end
            level = 0.0 if nsteps == 1 else k / (nsteps - 1)
            values[start:end] = d0 * (ratio**level)
        if end < n_iterations:
            values[end:] = d1
    elif schedule in ("const_plus_exp", "const + exp"):
        # Offset exponential: approaches d_final from above.
        values = d1 + (d0 - d1) * np.exp(-4.0 * (j - 1.0) / (n - 1.0))
    else:
        raise ValueError(f"Unknown annealing schedule '{schedule}'. Choose from {SCHEDULES}.")

    return np.maximum(values, floor)


def lambda_from_diffusion(diffusion: float, control_cost: float) -> float:
    """PI consistency ``R = lambda / D`` for scalar isotropic noise."""
    return float(control_cost) * float(diffusion)
