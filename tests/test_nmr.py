import jax.numpy as jnp
import numpy as np

from piqcx import PiQC
from piqcx.systems import nmr_params, nmr_problem, zz_hamiltonian


def test_nmr_params_shape():
    for n in (2, 4, 7):
        p = nmr_params(n)
        assert p.shape == (n, n)


def test_zz_hamiltonian_hermitian():
    params = nmr_params(2)
    h0 = zz_hamiltonian(params)
    assert h0.shape == (4, 4)
    assert jnp.allclose(h0, jnp.conjugate(h0.T), atol=1e-8)


def test_nmr_problem_controls(tiny_nmr_problem):
    p = tiny_nmr_problem
    assert p.n_qubits == 2
    assert p.n_controls == 4
    assert p.terminal_cost == "log_infidelity"
    assert p.dim == 4


def test_nmr_two_qubit_short_run(tiny_nmr_problem, solver_kwargs, rng_key):
    result = PiQC(tiny_nmr_problem, **solver_kwargs).run(rng_key)
    assert result.pulses.shape == (4, 4)
    assert np.all(np.isfinite(result.fidelity))
    assert np.all((result.fidelity >= -1e-4) & (result.fidelity <= 1.0 + 1e-4))


def test_nmr_four_qubit_hamiltonian_only():
    problem = nmr_problem(4, time=1.0, diffusion=1e-3, beta=10.0)
    assert problem.dim == 16
    assert problem.n_controls == 8
    assert jnp.allclose(problem.h0, jnp.conjugate(problem.h0.T))
