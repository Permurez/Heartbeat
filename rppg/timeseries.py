"""Cleaning and resampling of irregularly sampled signals.

Camera frames do not arrive at perfectly regular intervals (jitter, dropped
frames, duplicated log entries). Filters and the FFT assume a uniform time
grid, so every signal is first cleaned and then resampled onto one.
"""
import numpy as np
from scipy.interpolate import CubicSpline


def clean_timestamps(timestamps_s, values):
    """Drop NaN rows, sort by time and merge duplicate timestamps by averaging.

    values: one row per timestamp. A row can be:
        - one number                  shape (n_frames,)               e.g. reference PPG
        - R, G, B                     shape (n_frames, 3)             e.g. mean skin colour
        - R, G, B for every region    shape (n_frames, n_regions, 3)  e.g. forehead + 2 cheeks

    Returns (timestamps_s_clean, values_clean, report).
    """
    timestamps_s = np.asarray(timestamps_s, dtype=float)
    values = np.asarray(values, dtype=float)
    if values.shape[0] != timestamps_s.shape[0]:
        raise ValueError(f"timestamps_s has {timestamps_s.shape[0]} samples but values has {values.shape[0]}")
    tail_shape = values.shape[1:]
    values_2d = values.reshape(len(timestamps_s), -1)        # flatten everything except time
    report = {}

    ok = np.isfinite(timestamps_s) & np.all(np.isfinite(values_2d), axis=1)
    report["nan_dropped"] = int((~ok).sum())
    timestamps_s, values_2d = timestamps_s[ok], values_2d[ok]
    if len(timestamps_s) < 2:
        raise ValueError("Fewer than 2 valid samples after dropping NaNs")

    report["backward_steps"] = int((np.diff(timestamps_s) < 0).sum())
    order = np.argsort(timestamps_s, kind="stable")
    timestamps_s, values_2d = timestamps_s[order], values_2d[order]

    timestamps_unique, inverse, counts = np.unique(timestamps_s, return_inverse=True, return_counts=True)
    report["duplicates_merged"] = int(len(timestamps_s) - len(timestamps_unique))
    values_unique = np.stack(
        [np.bincount(inverse, weights=col) / counts for col in values_2d.T], axis=1
    )
    return timestamps_unique, values_unique.reshape((len(timestamps_unique),) + tail_shape), report


def uniform_grid(start_s, end_s, sample_rate_hz):
    """Uniform time grid from start_s (inclusive) to end_s (exclusive) at sample_rate_hz Hz."""
    return np.arange(start_s, end_s, 1.0 / sample_rate_hz)


def resample_uniform(timestamps_s, values, sample_rate_hz=30.0, method="spline", max_gap_s=0.1, time_grid_s=None):
    """Clean timestamps, then resample values onto an evenly spaced grid.

    values can be multi-dimensional (e.g. RGB, or RGB per ROI) - see clean_timestamps.
    method: "spline" (recommended, smoother) or "linear". Pass time_grid_s to put several
    signals of the same recording on one shared grid.

    Returns:
        new_timestamps_s: the evenly spaced grid.
        new_values: values resampled onto that grid, same row shape as the input.
        valid: False where the value is a guess (interpolated across a gap longer than
            max_gap_s, or outside the measured time range), not a real measurement.
        report: see clean_timestamps, plus gap statistics.
    """
    clean_t, clean_values, report = clean_timestamps(timestamps_s, values)
    tail_shape = clean_values.shape[1:]
    values_2d = clean_values.reshape(len(clean_t), -1)

    new_timestamps_s = uniform_grid(clean_t[0], clean_t[-1], sample_rate_hz) if time_grid_s is None \
        else np.asarray(time_grid_s, float)
    inside = (new_timestamps_s >= clean_t[0]) & (new_timestamps_s <= clean_t[-1])
    eval_timestamps_s = np.clip(new_timestamps_s, clean_t[0], clean_t[-1])   # never extrapolate

    if method == "spline":
        new_values = CubicSpline(clean_t, values_2d, axis=0)(eval_timestamps_s)
    elif method == "linear":
        new_values = np.stack([np.interp(eval_timestamps_s, clean_t, col) for col in values_2d.T], axis=1)
    else:
        raise ValueError(f"Unknown method '{method}'")

    gaps = np.diff(clean_t)
    valid = inside.copy()
    big_gaps = np.where(gaps > max_gap_s)[0]
    for i in big_gaps:
        valid[(new_timestamps_s > clean_t[i]) & (new_timestamps_s < clean_t[i + 1])] = False

    report["gaps_over_max"] = int(len(big_gaps))
    report["largest_gap_ms"] = round(float(gaps.max()) * 1000, 1)
    report["invalid_fraction"] = round(float((~valid).mean()), 4)
    return new_timestamps_s, new_values.reshape((len(new_timestamps_s),) + tail_shape), valid, report
