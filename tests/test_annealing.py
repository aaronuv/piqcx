import numpy as np
import pytest

from piqcx.annealing import SCHEDULES, schedule_diffusion


@pytest.mark.parametrize("name", SCHEDULES)
def test_schedule_length_and_positivity(name):
    d = schedule_diffusion(11, d_init=1e-1, d_final=1e-6, schedule=name, n_plateaus=4)
    assert d.shape == (11,)
    assert np.all(d > 0)


def test_linear_endpoints():
    d = schedule_diffusion(5, d_init=0.1, d_final=0.02, schedule="linear")
    np.testing.assert_allclose(d[0], 0.1)
    np.testing.assert_allclose(d[-1], 0.02)


def test_exponential_is_log_linear():
    d = schedule_diffusion(8, d_init=1e-1, d_final=1e-5, schedule="exponential")
    np.testing.assert_allclose(d[0], 1e-1, rtol=1e-6)
    np.testing.assert_allclose(d[-1], 1e-5, rtol=1e-6)
    logs = np.log(d)
    np.testing.assert_allclose(np.diff(logs), np.diff(logs)[0], rtol=1e-5)


def test_constant_schedule():
    d = schedule_diffusion(6, d_init=0.03, d_final=1e-9, schedule="constant")
    np.testing.assert_allclose(d, 0.03)


def test_unknown_schedule():
    with pytest.raises(ValueError):
        schedule_diffusion(4, d_init=1.0, d_final=0.1, schedule="not-a-schedule")  # type: ignore[arg-type]


def test_single_iteration():
    d = schedule_diffusion(1, d_init=0.2, d_final=1e-8, schedule="steps")
    assert d.shape == (1,)
    np.testing.assert_allclose(d[0], 0.2)
