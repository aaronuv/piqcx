# piqcx

JAX implementation of **Path integral Quantum Control (PiQC)** for closed and open quantum systems, following [Villanueva & Kappen, *Path integral control of open quantum systems* (2025)](https://arxiv.org/abs/2410.18635) and the MATLAB reference [aaronuv/piqc](https://github.com/aaronuv/piqc).

The package is **system-agnostic**: you instantiate a problem by drift Hamiltonian `H0`, control Hamiltonians `H_a`, states, and a time / sampling grid. NMR molecules are optional helpers for the paper demos.

## Algorithm

Each importance-sampling iteration:

1. Samples an ensemble of Wiener paths and integrates the controlled stochastic Schrödinger equation (Euler–Maruyama, renormalized).
2. Scores trajectories with the path-integral cost (fluence + Itô term + terminal cost).
3. Updates piecewise-constant pulses by a weighted average of the noise, \(A \leftarrow A + \langle \omega\,\Delta W / \Delta\tau\rangle\).
4. Optionally **anneals** the Ito diffusion \(D\) toward a small floor so the dynamics collapse onto a closed-system (unitary) trajectory.

Trajectory batches are `vmap`/`scan`/`jit` friendly and run on **CPU or GPU** through JAX.

## Install

```bash
pip install -e ".[dev,notebooks]"
```

GPU (CUDA 12):

```bash
pip install -e ".[gpu]"
```

Select the platform **before** the first JAX call:

```python
from piqcx import configure
configure(precision="float32", platform="gpu")
```

## Minimal example

```python
import jax
from piqcx import AnnealingConfig, PiQC, make_problem

problem = make_problem(
    n_qubits=1,
    # H0 defaults to 0; H_a default to local X,Y
    time=1.0,
    diffusion=5e-3,
)
result = PiQC(
    problem,
    n_traj=256,
    n_steps=80,
    n_pulses=40,
    n_iterations=50,
    annealing=AnnealingConfig(enabled=True, schedule="exponential", d_init=1e-1, d_final=1e-8),
).run(jax.random.PRNGKey(0))
print(result.fidelity[-1])
```

### Annealing schedules

Toggle `AnnealingConfig.enabled` and set `schedule` to one of:

`constant`, `linear`, `inverse`, `inverse_square`, `exponential`, `steps`, `log_steps`, `const_plus_exp`.

`R` is held fixed; \(\lambda_p = R D_p\) is updated each iteration.

### Inferred defaults

| omitted | default |
|---|---|
| `H0` | zero operator |
| `H_a` | local `X,Y` on every qubit |
| `psi0` | \(\lvert 0\dots 0\rangle\) |
| `target` | \(\lvert +\dots +\rangle\) |
| `n_steps`, `n_pulses` | `100`, `50` (steps made divisible by pulses) |
| `control_cost` `R` | `1 / n_controls` |
| `terminal_weight` `Q` | `10` |

Terminal costs: `fidelity` (\(-Q/2\,F\)), `log_infidelity` (\(Q/2\log(1-F)\), NMR), `energy`.

## Demos and tests

- `notebooks/01_single_qubit.ipynb` — dissipative qubit and annealed closed-system run
- `notebooks/02_nmr_multiqubit.ipynb` — 2- and 4-qubit NMR Hamiltonians
- `tests/` — `pytest` suite with shared `conftest.py` fixtures

```bash
pytest
```

## Citation

If you use PiQC, please cite the paper and this repository.

## License

MIT
