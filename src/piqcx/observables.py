"""Expectation values for diagnostics (Bloch vectors, etc.)."""

from __future__ import annotations

import jax.numpy as jnp

from .pauli import pauli


def expectation(psi, op):
    """``<psi|op|psi>`` for ``psi`` of shape ``(..., dim)``."""
    op_psi = jnp.einsum("ij,...j->...i", op, psi)
    return jnp.real(jnp.einsum("...i,...i->...", jnp.conjugate(psi), op_psi))


def bloch_vector(psi):
    """Single-qubit Bloch coordinates, ``psi`` shape ``(..., 2)``."""
    return jnp.stack(
        [expectation(psi, pauli("x")), expectation(psi, pauli("y")), expectation(psi, pauli("z"))],
        axis=-1,
    )
