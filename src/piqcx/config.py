"""Runtime configuration: precision and JAX platform (CPU/GPU)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

Precision = Literal["float32", "float64"]
Platform = Literal["cpu", "gpu", "tpu"]


@dataclass(frozen=True)
class RuntimeConfig:
    precision: Precision = "float32"
    platform: Platform | None = None

    @property
    def real_dtype(self) -> jnp.dtype:
        return jnp.float64 if self.precision == "float64" else jnp.float32

    @property
    def complex_dtype(self) -> jnp.dtype:
        return jnp.complex128 if self.precision == "float64" else jnp.complex64


_RUNTIME = RuntimeConfig()


def configure(
    *,
    precision: Precision | None = None,
    platform: Platform | None = None,
    enable_x64: bool | None = None,
) -> RuntimeConfig:
    """Set global precision and optionally the JAX platform.

    ``platform`` must be chosen before JAX backends are initialized. If JAX
    has already started, only precision is updated.
    """
    global _RUNTIME
    prec = precision or _RUNTIME.precision
    plat = platform if platform is not None else _RUNTIME.platform
    if enable_x64 is None:
        enable_x64 = prec == "float64"
    jax.config.update("jax_enable_x64", bool(enable_x64))
    if platform is not None:
        try:
            jax.config.update("jax_platform_name", platform)
        except RuntimeError:
            pass
    _RUNTIME = RuntimeConfig(precision=prec, platform=plat)
    return _RUNTIME


def runtime() -> RuntimeConfig:
    return _RUNTIME


def devices() -> list:
    return list(jax.devices())
