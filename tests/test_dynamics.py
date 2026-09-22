import jax
import jax.numpy as jnp
import numpy as np

from piqcx.costs import fidelity, path_cost, softmax_weights
from piqcx.dynamics import expand_pulses, propagate, sample_increments
from piqcx.pauli import pauli


def test_expand_pulses():
    pulses = jnp.arange(6.0).reshape(2, 3)
    u = expand_pulses(pulses, 6)
    assert u.shape == (2, 6)
    np.testing.assert_allclose(u[:, 0], u[:, 1])
    np.testing.assert_allclose(u[:, 2], pulses[:, 1])


def test_unitary_limit_preserves_state(rng_key):
    h0 = jnp.zeros((2, 2), dtype=jnp.complex64)
    ha = jnp.stack([pauli("x"), pauli("y")])
    diss = 0.5 * jnp.einsum("aij,ajk->ik", ha, ha)
    psi0 = jnp.array([1.0, 0.0], dtype=jnp.complex64)
    n_steps, n_traj, n_c = 8, 7, 2
    dt = 0.05
    pulses = jnp.zeros((n_c, 4), dtype=jnp.float32)
    u_t = expand_pulses(pulses, n_steps)
    dW = sample_increments(
        rng_key, n_traj=n_traj, n_controls=n_c, n_steps=n_steps, diffusion=0.0, dt=dt, dtype=jnp.float32
    )
    psi_T, _ = propagate(psi0, h0, diss, ha, u_t, dW, dt, 0.0, store_states=False)
    assert psi_T.shape == (n_traj, 2)
    assert jnp.allclose(jnp.linalg.norm(psi_T, axis=-1), 1.0, atol=1e-5)
    assert jnp.allclose(jnp.abs(psi_T[:, 0]), 1.0, atol=1e-5)


def test_fidelity_and_weights():
    target = jnp.array([1.0, 0.0], dtype=jnp.complex64)
    psi = jnp.stack([target, jnp.array([0.0, 1.0], dtype=jnp.complex64)])
    f = fidelity(psi, target)
    np.testing.assert_allclose(np.asarray(f), [1.0, 0.0], atol=1e-6)
    w, j, ess = softmax_weights(jnp.array([0.0, 10.0]), 1.0)
    assert jnp.isclose(jnp.sum(w), 1.0)
    assert 0.0 < float(ess) <= 1.0
    assert w[0] > w[1]
    assert jnp.isfinite(j)


def test_path_cost_zero_control(rng_key):
    u = jnp.zeros((2, 5), dtype=jnp.float32)
    dW = jax.random.normal(rng_key, (4, 2, 5), dtype=jnp.float32)
    s, su = path_cost(u, dW, r=2.0, dt=0.1)
    assert jnp.allclose(su, 0.0)
    assert jnp.allclose(s, 0.0)
