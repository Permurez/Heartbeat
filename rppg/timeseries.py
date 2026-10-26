"""Cleaning and resampling of irregularly sampled signals.

Camera frames do not arrive at perfectly regular intervals (jitter, dropped
frames, duplicated log entries). Filters and the FFT assume a uniform time
grid, so every signal is first cleaned and then resampled onto one.
"""
import numpy as np
from scipy.interpolate import CubicSpline


def clean_timestamps(t, x):
    """Drop NaN rows, sort by time and merge duplicate timestamps by averaging.

    x can have any shape whose first axis matches t, e.g. (N,), (N, 3) or (N, R, 3).
    Returns (t_clean, x_clean, report).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    if x.shape[0] != t.shape[0]:
        raise ValueError(f"t has {t.shape[0]} samples but x has {x.shape[0]}")
    tail_shape = x.shape[1:]
    x2d = x.reshape(len(t), -1)                       # flatten everything except time
    report = {}

    ok = np.isfinite(t) & np.all(np.isfinite(x2d), axis=1)
    report["nan_dropped"] = int((~ok).sum())
    t, x2d = t[ok], x2d[ok]
    if len(t) < 2:
        raise ValueError("Fewer than 2 valid samples after dropping NaNs")

    report["backward_steps"] = int((np.diff(t) < 0).sum())
    order = np.argsort(t, kind="stable")
    t, x2d = t[order], x2d[order]

    t_unique, inverse, counts = np.unique(t, return_inverse=True, return_counts=True)
    report["duplicates_merged"] = int(len(t) - len(t_unique))
    x_unique = np.stack(
        [np.bincount(inverse, weights=col) / counts for col in x2d.T], axis=1
    )
    return t_unique, x_unique.reshape((len(t_unique),) + tail_shape), report


def uniform_grid(t_start, t_end, fs):
    """Uniform time grid from t_start (inclusive) to t_end (exclusive) at fs Hz."""
    return np.arange(t_start, t_end, 1.0 / fs)


def resample_uniform(t, x, fs=30.0, method="spline", max_gap=0.1, t_grid=None):
    """Clean timestamps, then resample x onto a uniform grid.

    Pass t_grid to put several signals of the same recording on one shared grid.
    Returns (t_new, x_new, valid, report). valid[i] is False where t_new[i] is
    interpolated across a gap longer than max_gap seconds or lies outside the
    measured time range (these values are guesses, not measurements).
    """
    t_c, x_c, report = clean_timestamps(t, x)
    tail_shape = x_c.shape[1:]
    x2d = x_c.reshape(len(t_c), -1)

    t_new = uniform_grid(t_c[0], t_c[-1], fs) if t_grid is None else np.asarray(t_grid, float)
    inside = (t_new >= t_c[0]) & (t_new <= t_c[-1])
    t_eval = np.clip(t_new, t_c[0], t_c[-1])          # never extrapolate

    if method == "spline":
        x_new = CubicSpline(t_c, x2d, axis=0)(t_eval)
    elif method == "linear":
        x_new = np.stack([np.interp(t_eval, t_c, col) for col in x2d.T], axis=1)
    else:
        raise ValueError(f"Unknown method '{method}'")

    gaps = np.diff(t_c)
    valid = inside.copy()
    big = np.where(gaps > max_gap)[0]
    for i in big:
        valid[(t_new > t_c[i]) & (t_new < t_c[i + 1])] = False

    report["gaps_over_max"] = int(len(big))
    report["largest_gap_ms"] = round(float(gaps.max()) * 1000, 1)
    report["invalid_fraction"] = round(float((~valid).mean()), 4)
    return t_new, x_new.reshape((len(t_new),) + tail_shape), valid, report
