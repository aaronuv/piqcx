"""Multi-qubit NMR Hamiltonians from the MATLAB ``nmr_adaptive`` demos."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..config import runtime
from ..pauli import computational_basis, pauli_on
from ..problem import ControlProblem, make_problem

# Coupling tables are the MATLAB ``params`` arrays *without* the ``scale * pi``
# prefactor; ``nmr_params`` applies ``pi * scale``.
_NMR_RAW: dict[int, np.ndarray] = {
    2: np.array(
        [
            [400e6, 0.0],
            [47.6, 376e6],
        ]
    ),
    4: np.array(
        [
            [15479.88, 0.0, 0.0, 0.0],
            [-297.71, -3132.45, 0.0, 0.0],
            [-275.56, 64.74, -42682.97, 0.0],
            [39.17, 51.5, -129.08, -56445.71],
        ]
    ),
    7: np.array(
        [
            [1750.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [40.8, 14930.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.6, 69.5, 12199.9, 0.0, 0.0, 0.0, 0.0],
            [8.47, 1.4, 71.04, 17173.7, 0.0, 0.0, 0.0],
            [4.0, 155.6, -1.8, 6.5, 2785.85, 0.0, 0.0],
            [6.64, -0.7, 162.9, 3.3, 15.81, 2320.25, 0.0],
            [128.0, -7.1, 6.6, -0.9, 6.9, -1.7, 718.487],
        ]
    ),
}

_DEFAULT_T = {2: 8.8, 4: 8.8, 7: 8.8}
_DEFAULT_D_LINDBLAD = {2: 0.01 * 1e-3 * 0.84, 4: 1e-3 * 0.23, 7: 1e-3 * 0.23}


def nmr_params(n_qubits: int, *, scale: float = 1e-3) -> np.ndarray:
    if n_qubits not in _NMR_RAW:
        raise ValueError(f"No tabulated NMR molecule for n={n_qubits}; use 2, 4, or 7")
    return np.pi * scale * _NMR_RAW[n_qubits]


def zz_hamiltonian(params: np.ndarray):
    """``H0 = (1/2) sum_{i>j} J_{ij} Z_i Z_j`` (MATLAB ``build_S`` / ``h2``)."""
    n = int(params.shape[0])
    dim = 2**n
    h2 = jnp.zeros((dim, dim), dtype=runtime().complex_dtype)
    for i in range(n):
        for j in range(i):
            zz = pauli_on("z", j, n) @ pauli_on("z", i, n)
            h2 = h2 + jnp.asarray(params[i, j], dtype=h2.dtype) * zz
    return 0.5 * h2


def ghz_like_target(n_qubits: int, params: np.ndarray, time: float, scale: float = 1e-3):
    """MATLAB NMR target: phased ``(|0...0> + |1...1>)/sqrt(2)``."""
    shift_sum = float(np.sum(np.diag(params))) / scale
    phase = np.exp(1j * np.mod(shift_sum * time, 2.0 * np.pi))
    dim = 2**n_qubits
    phi = np.zeros((dim,), dtype=np.complex128)
    phi[0] = phase / np.sqrt(2.0)
    phi[-1] = np.conjugate(phase) / np.sqrt(2.0)
    return jnp.asarray(phi)


def nmr_problem(
    n_qubits: int = 2,
    *,
    time: float | None = None,
    scale: float = 1e-3,
    diffusion: float | None = None,
    control_weight: float = 1.0,
    beta: float = 200.0,
    terminal_cost: str = "log_infidelity",
    include_chemical_shifts: bool = False,
) -> ControlProblem:
    """NMR molecule with local ``X,Y`` drives.

    ``diffusion`` is Ito ``Dtilde``. If omitted, MATLAB's ``D_lindblad / 2``
    for that molecule is used.
    """
    params = nmr_params(n_qubits, scale=scale)
    t = _DEFAULT_T.get(n_qubits, 8.8) if time is None else float(time)
    h0 = zz_hamiltonian(params)
    if include_chemical_shifts:
        for q in range(n_qubits):
            h0 = h0 + 0.5 * jnp.asarray(params[q, q]) * pauli_on("z", q, n_qubits)
    psi0 = computational_basis(n_qubits, 0)
    target = ghz_like_target(n_qubits, params, t, scale=scale)
    n_c = 2 * n_qubits
    r = control_weight / n_c
    q = beta * r
    if diffusion is None:
        d_lind = _DEFAULT_D_LINDBLAD.get(n_qubits, 1e-3)
        diffusion = 0.5 * d_lind
    return make_problem(
        n_qubits=n_qubits,
        h0=h0,
        control_axes="xy",
        psi0=psi0,
        target=target,
        time=t,
        diffusion=float(diffusion),
        control_cost=r,
        terminal_weight=q,
        terminal_cost=terminal_cost,
    )
