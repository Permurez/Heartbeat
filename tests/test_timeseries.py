import numpy as np
import pytest

from rppg.timeseries import clean_timestamps, resample_uniform


def test_duplicates_are_averaged():
    t = np.array([0.0, 0.1, 0.1, 0.2])
    x = np.array([1.0, 2.0, 4.0, 5.0])
    t_c, x_c, rep = clean_timestamps(t, x)
    assert list(t_c) == [0.0, 0.1, 0.2]
    assert list(x_c) == [1.0, 3.0, 5.0]
    assert rep["duplicates_merged"] == 1


def test_nan_dropped_and_backward_sorted():
    t = np.array([0.0, 0.2, 0.1, 0.3])
    x = np.array([[1, 1], [3, 3], [2, 2], [np.nan, 4]], dtype=float)
    t_c, x_c, rep = clean_timestamps(t, x)
    assert rep["nan_dropped"] == 1 and rep["backward_steps"] == 1
    assert list(t_c) == [0.0, 0.1, 0.2]
    assert x_c.shape == (3, 2) and x_c[1, 0] == 2


def test_multidimensional_shape_preserved():
    t = np.linspace(0, 1, 31)
    x = np.random.rand(31, 3, 3)
    t_new, x_new, valid, _ = resample_uniform(t, x, fs=30)
    assert x_new.shape == (len(t_new), 3, 3)
    assert valid.all()


@pytest.mark.parametrize("method", ["spline", "linear"])
def test_resampling_recovers_sine_and_flags_gap(method):
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.025, 0.041, 600))       # jittery ~30 FPS
    keep = (t < 10) | (t > 10.3)                          # 300 ms of dropped frames
    t = t[keep]
    x = np.sin(2 * np.pi * 1.5 * t)
    t_new, x_new, valid, rep = resample_uniform(t, x, fs=30, method=method, max_gap=0.1)
    assert np.allclose(np.diff(t_new), 1 / 30)
    assert rep["gaps_over_max"] == 1
    assert not valid[(t_new > 10.05) & (t_new < 10.25)].any()
    err = np.abs(x_new - np.sin(2 * np.pi * 1.5 * t_new))[valid]
    assert err.max() < (0.01 if method == "spline" else 0.05)


def test_shared_grid_marks_outside_range_invalid():
    t = np.linspace(1, 2, 31)
    grid = np.arange(0, 3, 1 / 30)
    _, x_new, valid, _ = resample_uniform(t, t, t_grid=grid)
    assert not valid[grid < 1].any() and not valid[grid > 2].any()
    assert np.isfinite(x_new).all()
