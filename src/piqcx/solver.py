"""Adaptive importance-sampling PiQC solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from .annealing import AnnealingConfig, schedule_diffusion
from .config import runtime
from .costs import path_cost, softmax_weights, terminal_cost
from .dynamics import expand_pulses, propagate, sample_increments
from .problem import ControlProblem
from .smoothing import ControlSmoother, SmoothingConfig


def infer_time_grid(n_steps: int | None, n_pulses: int | None) -> tuple[int, int]:
    """Choose a compatible ``(n_steps, n_pulses)`` pair.

    ``n_steps`` is rounded down to a multiple of ``n_pulses`` so each pulse
    occupies the same number of Euler steps.
    """
    if n_steps is None and n_pulses is None:
        return 100, 50
    if n_pulses is None:
        assert n_steps is not None
        n_pulses = n_steps if n_steps <= 32 else max(1, n_steps // 2)
    if n_steps is None:
        n_steps = int(n_pulses) * 2
    n_pulses = int(n_pulses)
    n_steps = int(n_steps)
    if n_pulses < 1 or n_steps < 1:
        raise ValueError("n_steps and n_pulses must be positive")
    spp = max(1, n_steps // n_pulses)
    return spp * n_pulses, n_pulses


def _compile_step(n_traj: int, n_steps: int, terminal_kind: str, store_states: bool):
    def step(pulses, key, h0, dissipator, ha, psi0, target, energy_h, diffusion, dt, r, q):
        dtype = pulses.dtype
        key, sub = jax.random.split(key)
        u_t = expand_pulses(pulses, n_steps)
        dW = sample_increments(
            sub,
            n_traj=n_traj,
            n_controls=pulses.shape[0],
            n_steps=n_steps,
            diffusion=diffusion,
            dt=dt,
            dtype=dtype,
        )
        psi_T, states = propagate(
            psi0,
            h0,
            dissipator,
            ha,
            u_t,
            dW,
            dt,
            diffusion,
            store_states=store_states,
        )
        send, fid = terminal_cost(
            psi_T,
            target,
            kind=terminal_kind,
            q=q,
            energy_hamiltonian=energy_h,
        )
        s_path, su = path_cost(u_t, dW, r=r, dt=dt)
        cost = s_path + send
        lam = jnp.maximum(r * diffusion, jnp.asarray(1e-16, dtype=dtype))
        w, j, ess = softmax_weights(cost, lam)
        n_pulses = pulses.shape[-1]
        spp = n_steps // n_pulses
        dw_pulse = dW.reshape((n_traj, pulses.shape[0], n_pulses, spp)).sum(-1)
        delta = jnp.einsum("b,bcp->cp", w, dw_pulse) / (dt * spp)
        pulses_new = pulses + delta
        metrics = (jnp.mean(fid), jnp.min(fid), jnp.mean(cost), j, ess, su)
        return pulses_new, metrics, states, key

    return jax.jit(step)


@dataclass
class PiQCResult:
    pulses: np.ndarray
    controls: np.ndarray
    fidelity: np.ndarray
    fidelity_min: np.ndarray
    expected_cost: np.ndarray
    free_energy: np.ndarray
    ess: np.ndarray
    fluence: np.ndarray
    diffusion_schedule: np.ndarray
    n_steps: int
    n_pulses: int
    time: float
    extras: dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return self.time / self.n_steps


@dataclass
class PiQC:
    """Path integral Quantum Control with optional diffusion annealing.

    Typical workflow::

        problem = make_problem(n_qubits=1, h0=0, ...)
        result = PiQC(problem, n_traj=400, n_steps=100, n_pulses=50,
                      annealing=AnnealingConfig(enabled=True, schedule="exponential")).run()
    """

    problem: ControlProblem
    n_traj: int = 400
    n_steps: int | None = None
    n_pulses: int | None = None
    n_iterations: int = 200
    annealing: AnnealingConfig = field(default_factory=AnnealingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    store_states: bool = False
    seed: int = 0

    def __post_init__(self):
        steps, pulses = infer_time_grid(self.n_steps, self.n_pulses)
        object.__setattr__(self, "n_steps", steps)
        object.__setattr__(self, "n_pulses", pulses)

    @property
    def dt(self) -> float:
        return self.problem.time / self.n_steps

    def diffusion_schedule(self) -> np.ndarray:
        d0, d1 = self.annealing.resolve_bounds(self.problem.diffusion)
        name = self.annealing.schedule if self.annealing.enabled else "constant"
        return schedule_diffusion(
            self.n_iterations,
            d_init=d0,
            d_final=d1,
            schedule=name,
            n_plateaus=self.annealing.n_plateaus,
            floor=self.annealing.floor,
        )

    def run(
        self,
        key=None,
        pulses0=None,
        callback: Callable[[int, dict], None] | None = None,
    ) -> PiQCResult:
        problem = self.problem
        rdtype = runtime().real_dtype
        cdtype = runtime().complex_dtype
        if key is None:
            key = jax.random.PRNGKey(self.seed)
        if pulses0 is None:
            pulses = jnp.zeros((problem.n_controls, self.n_pulses), dtype=rdtype)
        else:
            pulses = jnp.asarray(pulses0, dtype=rdtype)
            if pulses.shape != (problem.n_controls, self.n_pulses):
                raise ValueError(
                    f"pulses0 shape {pulses.shape} != {(problem.n_controls, self.n_pulses)}"
                )

        h0 = jnp.asarray(problem.h0, dtype=cdtype)
        ha = jnp.asarray(problem.control_hamiltonians, dtype=cdtype)
        diss = jnp.asarray(problem.dissipator, dtype=cdtype)
        psi0 = jnp.asarray(problem.psi0, dtype=cdtype)
        target = jnp.asarray(problem.target, dtype=cdtype)
        energy_h = h0 if problem.energy_hamiltonian is None else jnp.asarray(
            problem.energy_hamiltonian, dtype=cdtype
        )

        step = _compile_step(self.n_traj, self.n_steps, problem.terminal_cost, self.store_states)
        smoother = ControlSmoother(problem.n_controls, self.n_pulses, self.smoothing)
        schedule = self.diffusion_schedule()
        u_avg = np.asarray(pulses)

        hist = {k: [] for k in ("f", "fmin", "c", "j", "ess", "su")}
        last_states = None
        dt = self.dt
        r = float(problem.control_cost)
        q = float(problem.terminal_weight)

        for p, d_p in enumerate(schedule):
            pulses_j = jnp.asarray(u_avg, dtype=rdtype)
            pulses_new, metrics, states, key = step(
                pulses_j, key, h0, diss, ha, psi0, target, energy_h, float(d_p), dt, r, q
            )
            u_avg = smoother.update(np.asarray(pulses_new), p)
            f, fmin, c, j, ess, su = (float(x) for x in metrics)
            hist["f"].append(f)
            hist["fmin"].append(fmin)
            hist["c"].append(c)
            hist["j"].append(j)
            hist["ess"].append(ess)
            hist["su"].append(su)
            last_states = states
            if callback is not None:
                callback(
                    p,
                    {
                        "fidelity": f,
                        "ess": ess,
                        "diffusion": float(d_p),
                        "pulses": u_avg,
                    },
                )

        u_final = np.asarray(u_avg, dtype=np.float64)
        controls = np.repeat(u_final, self.n_steps // self.n_pulses, axis=-1)
        extras = {}
        if self.store_states and last_states is not None:
            extras["states"] = np.asarray(last_states)
        return PiQCResult(
            pulses=u_final,
            controls=controls,
            fidelity=np.asarray(hist["f"]),
            fidelity_min=np.asarray(hist["fmin"]),
            expected_cost=np.asarray(hist["c"]),
            free_energy=np.asarray(hist["j"]),
            ess=np.asarray(hist["ess"]),
            fluence=np.asarray(hist["su"]),
            diffusion_schedule=np.asarray(schedule),
            n_steps=self.n_steps,
            n_pulses=self.n_pulses,
            time=problem.time,
            extras=extras,
        )


def optimize(problem: ControlProblem, **kwargs) -> PiQCResult:
    return PiQC(problem, **kwargs).run()
