import jax.numpy as jnp
import numpy as np

from piqcx import AnnealingConfig, PiQC, infer_time_grid, make_problem, optimize
from piqcx.smoothing import ControlSmoother, SmoothingConfig


def test_infer_time_grid():
    assert infer_time_grid(None, None) == (100, 50)
    assert infer_time_grid(100, 50) == (100, 50)
    assert infer_time_grid(10, 4) == (8, 4)
    assert infer_time_grid(None, 7) == (14, 7)


def test_make_problem_defaults():
    p = make_problem(n_qubits=2)
    assert p.dim == 4
    assert p.n_controls == 4
    assert p.h0.shape == (4, 4)
    assert jnp.allclose(jnp.linalg.norm(p.psi0), 1.0)
    assert jnp.allclose(jnp.linalg.norm(p.target), 1.0)


def test_tiny_open_run(tiny_solver, rng_key):
    result = tiny_solver.run(rng_key)
    assert result.pulses.shape == (2, 4)
    assert result.controls.shape == (2, 8)
    assert result.fidelity.shape == (3,)
    assert np.all((result.fidelity >= -1e-4) & (result.fidelity <= 1.0 + 1e-4))
    assert np.all((result.ess > 0.0) & (result.ess <= 1.0 + 1e-6))
    assert np.all(np.isfinite(result.pulses))


def test_annealed_run(tiny_qubit_problem, solver_kwargs, annealing_exponential, rng_key):
    solver = PiQC(tiny_qubit_problem, annealing=annealing_exponential, **solver_kwargs)
    result = solver.run(rng_key)
    sched = result.diffusion_schedule
    assert sched[0] > sched[-1]
    np.testing.assert_allclose(sched[0], annealing_exponential.d_init, rtol=1e-5)
    np.testing.assert_allclose(sched[-1], annealing_exponential.d_final, rtol=1e-4)


def test_optimize_wrapper(abstract_two_qubit_problem, solver_kwargs):
    result = optimize(abstract_two_qubit_problem, **solver_kwargs)
    assert result.pulses.shape[0] == abstract_two_qubit_problem.n_controls


def test_self_overlap_near_one(solver_kwargs, rng_key):
    plus = jnp.array([1.0, 1.0], dtype=jnp.complex64)
    plus = plus / jnp.linalg.norm(plus)
    problem = make_problem(
        n_qubits=1,
        psi0=plus,
        target=plus,
        diffusion=1e-4,
        time=0.2,
        terminal_cost="fidelity",
    )
    result = PiQC(problem, **solver_kwargs).run(rng_key)
    assert result.fidelity[0] > 0.9


def test_window_smoother():
    cfg = SmoothingConfig(kind="window", window=1, window_after=0, window_late=2)
    sm = ControlSmoother(1, 2, cfg)
    a = np.array([[1.0, 3.0]])
    b = np.array([[5.0, 7.0]])
    out0 = sm.update(a, 0)
    np.testing.assert_allclose(out0, a)
    out1 = sm.update(b, 1)
    np.testing.assert_allclose(out1, 0.5 * (a + b))
