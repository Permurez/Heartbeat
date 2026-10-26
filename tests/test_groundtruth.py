import numpy as np

from rppg.groundtruth import hr_peaks, hr_spectral, reference_hr


def ppg_like(hr_bpm, fs=30.0, seconds=8.0, noise=0.05, seed=0):
    """Pulse-like wave: fundamental + harmonics (sharp upstroke) + noise + slow drift."""
    t = np.arange(0, seconds, 1 / fs)
    f = hr_bpm / 60
    x = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t + 0.5)
         + 0.15 * np.sin(6 * np.pi * f * t + 1.0) + 0.5 * t / seconds)
    return x + noise * np.random.default_rng(seed).standard_normal(len(t))


def test_both_estimators_agree_on_clean_signal():
    for hr in (55, 75, 102, 140):
        x = ppg_like(hr)
        fft_hr, peak_hr = reference_hr(x, 30.0)
        assert abs(fft_hr - hr) < 1.5, (hr, fft_hr)
        assert abs(peak_hr - hr) < 4.0, (hr, peak_hr)


def test_spectral_ignores_out_of_band_drift():
    t = np.arange(0, 8, 1 / 30)
    x = 5 * np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 1.2 * t)
    assert abs(hr_spectral(x, 30.0) - 72) < 1.5


def test_peaks_returns_nan_without_beats():
    assert np.isnan(hr_peaks(np.zeros(240), 30.0))
