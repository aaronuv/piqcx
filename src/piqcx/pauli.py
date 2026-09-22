"""Pauli operators and n-qubit embeddings."""

from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .config import runtime


def _cdtype():
    return runtime().complex_dtype


@lru_cache(maxsize=4)
def _pauli_numpy() -> dict[str, np.ndarray]:
    i2 = np.eye(2, dtype=np.complex128)
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return {"i": i2, "x": x, "y": y, "z": z}


def pauli(name: str):
    key = name.lower()
    if key not in {"i", "x", "y", "z", "0", "1", "2", "3"}:
        raise ValueError(f"Unknown Pauli '{name}'")
    alias = {"0": "i", "1": "x", "2": "y", "3": "z"}
    key = alias.get(key, key)
    return jnp.asarray(_pauli_numpy()[key], dtype=_cdtype())


def embed(op, site: int, n_qubits: int):
    """Embed a single-qubit operator on ``site`` (0-based) of ``n_qubits``."""
    if not 0 <= site < n_qubits:
        raise ValueError(f"site={site} out of range for n_qubits={n_qubits}")
    ident = pauli("i")
    out = jnp.array([[1]], dtype=_cdtype())
    for q in range(n_qubits):
        out = jnp.kron(out, op if q == site else ident)
    return out


def pauli_on(axis: str, site: int, n_qubits: int):
    return embed(pauli(axis), site, n_qubits)


def local_controls(n_qubits: int, axes: str = "xy"):
    """Stack local control Hamiltonians, shape ``(n_controls, dim, dim)``."""
    ops = [pauli_on(ax, q, n_qubits) for q in range(n_qubits) for ax in axes.lower()]
    return jnp.stack(ops, axis=0)


def computational_basis(n_qubits: int, bits: int = 0):
    dim = 2**n_qubits
    psi = jnp.zeros((dim,), dtype=_cdtype())
    return psi.at[bits].set(1)


def product_state(one_qubit, n_qubits: int):
    psi = jnp.asarray(one_qubit, dtype=_cdtype())
    psi = psi / jnp.linalg.norm(psi)
    out = psi
    for _ in range(n_qubits - 1):
        out = jnp.kron(out, psi)
    return out


def infer_n_qubits(dim: int) -> int:
    n = int(np.log2(dim))
    if 2**n != dim:
        raise ValueError(f"Hilbert dimension {dim} is not a power of two")
    return n
