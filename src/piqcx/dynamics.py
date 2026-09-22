"""Euler–Maruyama unraveling of the controlled SSE (paper Eq. 19)."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp


def expand_pulses(pulses, n_steps: int):
    """Repeat piecewise-constant pulses onto the Euler grid."""
    n_pulses = pulses.shape[-1]
    spp = n_steps // n_pulses
    return jnp.repeat(pulses, spp, axis=-1)


def sample_increments(key, *, n_traj: int, n_controls: int, n_steps: int, diffusion: float, dt: float, dtype):
    scale = jnp.sqrt(jnp.asarray(diffusion * dt, dtype=dtype))
    return scale * jax.random.normal(key, (n_traj, n_controls, n_steps), dtype=dtype)


def _euler_step(psi, h0, dissipator, ha, u_dt_dw, dt, diffusion):
    # psi: (n_traj, dim), ha: (n_c, dim, dim), u_dt_dw: (n_traj, n_c)
    drift = (-1j * (psi @ h0.T) - diffusion * (psi @ dissipator.T)) * dt
    ha_psi = jnp.einsum("cij,tj->tci", ha, psi)
    stoch = -1j * jnp.einsum("tci,tc->ti", ha_psi, u_dt_dw)
    nxt = psi + drift + stoch
    return nxt / jnp.linalg.norm(nxt, axis=-1, keepdims=True)


@partial(jax.jit, static_argnames=("store_states",))
def propagate(
    psi0,
    h0,
    dissipator,
    ha,
    controls_time,
    dW,
    dt,
    diffusion,
    *,
    store_states: bool = False,
):
    """Integrate a batch of trajectories.

    Parameters
    ----------
    psi0
        ``(dim,)`` initial state, broadcast over trajectories.
    controls_time
        ``(n_controls, n_steps)`` piecewise-constant fields on the Euler grid.
    dW
        ``(n_traj, n_controls, n_steps)`` Wiener increments.

    Returns
    -------
    psi_T
        Final states ``(n_traj, dim)``.
    states
        If ``store_states``, ``(n_steps, n_traj, dim)``, else empty.
    """
    n_traj = dW.shape[0]
    psi = jnp.broadcast_to(psi0, (n_traj, psi0.shape[-1]))

    def body(psi_t, inputs):
        u_t, dw_t = inputs
        inc = u_t[None, :] * dt + dw_t
        psi_next = _euler_step(psi_t, h0, dissipator, ha, inc, dt, diffusion)
        return psi_next, psi_next if store_states else psi_next[:0]

    xs = (controls_time.T, jnp.moveaxis(dW, -1, 0))
    psi_T, states = jax.lax.scan(body, psi, xs)
    return psi_T, states
