"""Path-integral costs and importance-sampling weights."""

from __future__ import annotations

import jax.numpy as jnp


def fidelity(psi, target):
    """``|<phi|psi>|^2`` for a batch ``(n_traj, dim)``."""
    overlap = psi @ jnp.conjugate(target)
    return jnp.real(overlap * jnp.conjugate(overlap))


def terminal_cost(psi, target, *, kind: str, q: float, energy_hamiltonian=None):
    if kind == "fidelity":
        f = fidelity(psi, target)
        return -0.5 * q * f, f
    if kind == "log_infidelity":
        f = jnp.clip(fidelity(psi, target), 0.0, 1.0 - 1e-12)
        return 0.5 * q * jnp.log1p(-f), f
    if kind == "energy":
        if energy_hamiltonian is None:
            raise ValueError("energy terminal cost requires energy_hamiltonian")
        e = jnp.real(jnp.einsum("ti,ij,tj->t", jnp.conjugate(psi), energy_hamiltonian, psi))
        f = fidelity(psi, target)
        return q * e, f
    raise ValueError(f"Unknown terminal cost '{kind}'")


def path_cost(controls_time, dW, *, r: float, dt: float):
    """``S_u + R int u dW`` with ``S_u = (R/2) int |u|^2 dt`` (scalar R)."""
    su = 0.5 * r * dt * jnp.sum(controls_time**2)
    ito = r * jnp.einsum("ct,nct->n", controls_time, dW)
    return su + ito, su


def softmax_weights(cost, lam):
    shifted = cost - jnp.min(cost)
    unnorm = jnp.exp(-shifted / lam)
    mean_un = jnp.mean(unnorm)
    j = jnp.min(cost) - lam * jnp.log(mean_un)
    w = unnorm / jnp.sum(unnorm)
    ess = 1.0 / (jnp.sum(w**2) * w.shape[0])
    return w, j, ess
