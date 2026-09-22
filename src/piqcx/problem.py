"""Abstract control problem: H0, control Hamiltonians, states, costs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import jax.numpy as jnp
import numpy as np

from .config import runtime
from .pauli import computational_basis, infer_n_qubits, local_controls, product_state

TerminalCost = Literal["fidelity", "log_infidelity", "energy"]


def _as_hermitian(mat, dtype):
    arr = jnp.asarray(mat, dtype=dtype)
    return 0.5 * (arr + jnp.conjugate(arr.T))


def _normalize(vec, dtype):
    psi = jnp.asarray(vec, dtype=dtype).reshape(-1)
    nrm = jnp.linalg.norm(psi)
    if float(jnp.real(nrm)) == 0.0:
        raise ValueError("State has zero norm")
    return psi / nrm


@dataclass(frozen=True)
class ControlProblem:
    """Finite-dimensional PiQC instance.

    Parameters
    ----------
    h0
        Drift Hamiltonian, shape ``(dim, dim)``.
    control_hamiltonians
        Hermitian generators ``H_a``, shape ``(n_controls, dim, dim)``.
    psi0, target
        Normalized state vectors of length ``dim``.
    n_qubits
        Optional; inferred when ``dim`` is a power of two.
    time
        Horizon ``T``.
    diffusion
        Isotropic Ito scale ``D`` in ``<dW dW^T> = D I dt``.
        This is MATLAB ``Dtilde = D_lindblad / 2``.
    control_cost
        Scalar ``R`` in the quadratic fluence term.
    terminal_weight
        Scalar ``Q`` in the terminal cost.
    terminal_cost
        ``fidelity``: ``-Q/2 F``;
        ``log_infidelity``: ``Q/2 log(1-F)``;
        ``energy``: ``Q <H_cost>``.
    """

    h0: jnp.ndarray
    control_hamiltonians: jnp.ndarray
    psi0: jnp.ndarray
    target: jnp.ndarray
    n_qubits: int
    time: float = 1.0
    diffusion: float = 5e-3
    control_cost: float = 1.0
    terminal_weight: float = 10.0
    terminal_cost: TerminalCost = "fidelity"
    energy_hamiltonian: jnp.ndarray | None = None

    @property
    def dim(self) -> int:
        return int(self.h0.shape[0])

    @property
    def n_controls(self) -> int:
        return int(self.control_hamiltonians.shape[0])

    @property
    def dissipator(self) -> jnp.ndarray:
        """``(1/2) sum_a H_a @ H_a`` used in the SSE drift."""
        ha = self.control_hamiltonians
        return 0.5 * jnp.einsum("aij,ajk->ik", ha, ha)

    def with_updates(self, **kwargs) -> "ControlProblem":
        return replace(self, **kwargs)


def make_problem(
    *,
    n_qubits: int | None = None,
    h0=None,
    control_hamiltonians=None,
    control_axes: str = "xy",
    psi0=None,
    target=None,
    time: float = 1.0,
    diffusion: float = 5e-3,
    control_cost: float | None = None,
    terminal_weight: float | None = None,
    terminal_cost: TerminalCost = "fidelity",
    energy_hamiltonian=None,
) -> ControlProblem:
    """Build a problem, filling unspecified pieces from a qubit default.

    If ``h0`` / ``control_hamiltonians`` are omitted, they default to a zero
    drift and local ``X,Y`` controls on each qubit. Missing states default to
    ``|0...0>`` and ``|+...+>``.
    """
    dtype = runtime().complex_dtype

    if h0 is None and n_qubits is None and control_hamiltonians is None:
        raise ValueError("Specify n_qubits and/or h0 / control_hamiltonians")

    if control_hamiltonians is not None:
        ha = jnp.asarray(control_hamiltonians, dtype=dtype)
        if ha.ndim == 2:
            ha = ha[None, ...]
        ha = jnp.stack([_as_hermitian(h, dtype) for h in ha])
        dim = int(ha.shape[-1])
    elif h0 is not None:
        dim = int(np.asarray(h0).shape[0])
        n_qubits = n_qubits if n_qubits is not None else infer_n_qubits(dim)
        ha = local_controls(n_qubits, control_axes)
    else:
        ha = local_controls(n_qubits, control_axes)
        dim = 2**n_qubits

    if n_qubits is None:
        try:
            n_qubits = infer_n_qubits(dim)
        except ValueError:
            n_qubits = 0

    if h0 is None:
        h0_arr = jnp.zeros((dim, dim), dtype=dtype)
    else:
        h0_arr = _as_hermitian(h0, dtype)
        if h0_arr.shape != (dim, dim):
            raise ValueError("h0 and control Hamiltonians have incompatible shapes")

    if psi0 is None:
        if n_qubits > 0:
            psi0_arr = computational_basis(n_qubits, 0)
        else:
            psi0_arr = jnp.zeros((dim,), dtype=dtype).at[0].set(1)
    else:
        psi0_arr = _normalize(psi0, dtype)

    if target is None:
        plus = jnp.asarray([1, 1], dtype=dtype) / jnp.sqrt(jnp.asarray(2.0))
        if n_qubits > 0:
            target_arr = product_state(plus, n_qubits)
        else:
            target_arr = jnp.ones((dim,), dtype=dtype)
            target_arr = target_arr / jnp.linalg.norm(target_arr)
    else:
        target_arr = _normalize(target, dtype)

    if psi0_arr.shape[-1] != dim or target_arr.shape[-1] != dim:
        raise ValueError("State dimension does not match the Hamiltonians")

    n_c = int(ha.shape[0])
    r = 1.0 / n_c if control_cost is None else float(control_cost)
    q = 10.0 if terminal_weight is None else float(terminal_weight)
    h_e = None if energy_hamiltonian is None else _as_hermitian(energy_hamiltonian, dtype)

    return ControlProblem(
        h0=h0_arr,
        control_hamiltonians=ha,
        psi0=psi0_arr,
        target=target_arr,
        n_qubits=int(n_qubits),
        time=float(time),
        diffusion=float(diffusion),
        control_cost=r,
        terminal_weight=q,
        terminal_cost=terminal_cost,
        energy_hamiltonian=h_e,
    )
