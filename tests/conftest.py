"""Shared fixtures for the piqcx test suite."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from piqcx import AnnealingConfig, PiQC, SmoothingConfig, configure, make_problem
from piqcx.systems import nmr_problem, single_qubit


@pytest.fixture(scope="session", autouse=True)
def _jax_runtime():
    configure(precision="float32")
    return True


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def tiny_qubit_problem():
    return single_qubit(
        time=1.0,
        diffusion=5e-3,
        control_cost=0.01,
        terminal_weight=10.0,
        terminal_cost="fidelity",
    )


@pytest.fixture
def abstract_two_qubit_problem():
    return make_problem(
        n_qubits=2,
        time=0.5,
        diffusion=1e-2,
        control_cost=0.25,
        terminal_weight=8.0,
    )


@pytest.fixture
def tiny_nmr_problem():
    return nmr_problem(
        2,
        time=1.0,
        diffusion=1e-3,
        control_weight=1.0,
        beta=20.0,
        terminal_cost="log_infidelity",
    )


@pytest.fixture
def solver_kwargs():
    return dict(
        n_traj=16,
        n_steps=8,
        n_pulses=4,
        n_iterations=3,
        smoothing=SmoothingConfig(kind="window", window=1),
        seed=1,
    )


@pytest.fixture
def tiny_solver(tiny_qubit_problem, solver_kwargs):
    return PiQC(tiny_qubit_problem, **solver_kwargs)


@pytest.fixture
def annealing_exponential():
    return AnnealingConfig(
        enabled=True,
        schedule="exponential",
        d_init=1e-2,
        d_final=1e-6,
        n_plateaus=4,
    )


@pytest.fixture
def paulis():
    from piqcx.pauli import pauli

    return {name: pauli(name) for name in ("i", "x", "y", "z")}


@pytest.fixture
def identity2():
    return jnp.eye(2, dtype=jnp.complex64)
