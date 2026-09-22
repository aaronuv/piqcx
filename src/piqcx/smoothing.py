"""Control smoothing after each importance-sampling update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SmoothName = Literal["none", "window", "ema"]


@dataclass(frozen=True)
class SmoothingConfig:
    kind: SmoothName = "window"
    window: int = 1
    window_after: int | None = None
    window_late: int = 10
    ema_decay: float = 0.3

    def __post_init__(self):
        object.__setattr__(self, "window", max(1, int(self.window)))
        late_at = self.window if self.window_after is None else int(self.window_after)
        object.__setattr__(self, "window_after", late_at)
        object.__setattr__(self, "window_late", max(1, int(self.window_late)))


class ControlSmoother:
    """Keep a rolling average of pulse iterates (MATLAB window smoother)."""

    def __init__(self, n_controls: int, n_pulses: int, config: SmoothingConfig):
        self.config = config
        self._buffer: list[np.ndarray] = []
        self._ema: np.ndarray | None = None

    def _window_size(self, iteration: int) -> int:
        if iteration + 1 > self.config.window_after:
            return self.config.window_late
        return self.config.window

    def update(self, pulses: np.ndarray, iteration: int) -> np.ndarray:
        kind = self.config.kind
        u = np.asarray(pulses)
        if kind == "none":
            return u
        if kind == "ema":
            if self._ema is None:
                self._ema = u
            else:
                a = float(self.config.ema_decay)
                self._ema = (1.0 - a) * self._ema + a * u
            return self._ema
        win = self._window_size(iteration)
        self._buffer.append(u)
        self._buffer = self._buffer[-win:]
        return np.mean(np.stack(self._buffer, axis=0), axis=0)
