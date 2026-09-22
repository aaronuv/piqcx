"""Single-qubit and generic local-drive factories matching the MATLAB demos."""

from __future__ import annotations

import jax.numpy as jnp

from ..pauli import pauli, product_state
from ..problem import ControlProblem, make_problem


def single_qubit(
    *,
    time: float = 1.0,
    diffusion: float = 5e-3,
    control_cost: float = 0.01,
    terminal_weight: float = 10.0,
    terminal_cost: str = "fidelity",
    psi0=None,
    target=None,
    h0=None,
) -> ControlProblem:
    """Dissipative one-qubit state preparation (MATLAB ``one_qubit_adaptive``).

    Default states: ``psi0 = |+>``, ``target = (|0> + i|1>)/sqrt(2)``.
    ``diffusion`` is the Ito scale (MATLAB ``Dtilde``); the scripts use
    Lindblad ``D`` and set ``Dtilde = D/2``.
    """
    plus = jnp.array([1.0, 1.0], dtype=jnp.complex64)
    plus = plus / jnp.linalg.norm(plus)
    phi = jnp.array([1.0, 1.0j])
    phi = phi / jnp.linalg.norm(phi)
    if h0 is None:
        h0 = jnp.zeros((2, 2), dtype=jnp.complex64)
    return make_problem(
        n_qubits=1,
        h0=h0,
        control_hamiltonians=jnp.stack([pauli("x"), pauli("y")]),
        psi0=plus if psi0 is None else psi0,
        target=phi if target is None else target,
        time=time,
        diffusion=diffusion,
        control_cost=control_cost,
        terminal_weight=terminal_weight,
        terminal_cost=terminal_cost,
    )


def local_drive(
    n_qubits: int,
    *,
    h0=None,
    axes: str = "xy",
    **kwargs,
) -> ControlProblem:
    """Zero or user drift with local Pauli drives on every qubit."""
    return make_problem(n_qubits=n_qubits, h0=h0, control_axes=axes, **kwargs)


def plus_state(n_qubits: int):
    plus = jnp.array([1.0, 1.0])
    plus = plus / jnp.linalg.norm(plus)
    return product_state(plus, n_qubits)
