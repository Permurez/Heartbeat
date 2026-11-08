import numpy as np

from rppg.groundtruth import hr_peaks, hr_spectral, reference_hr


def ppg_like(heart_rate_bpm, sample_rate_hz=30.0, seconds=8.0, noise=0.05, seed=0):
    """Pulse-like wave: fundamental + harmonics (sharp upstroke) + noise + slow drift."""
    t = np.arange(0, seconds, 1 / sample_rate_hz)
    f = heart_rate_bpm / 60
    x = (np.sin(2 * np.pi * f * t) + 0.4 * np.sin(4 * np.pi * f * t + 0.5)
         + 0.15 * np.sin(6 * np.pi * f * t + 1.0) + 0.5 * t / seconds)
    return x + noise * np.random.default_rng(seed).standard_normal(len(t))


def test_both_estimators_agree_on_clean_signal():
    for heart_rate_bpm in (55, 75, 102, 140):
        signal = ppg_like(heart_rate_bpm)
        spectral_bpm, peaks_bpm = reference_hr(signal, 30.0)
        assert abs(spectral_bpm - heart_rate_bpm) < 1.5, (heart_rate_bpm, spectral_bpm)
        assert abs(peaks_bpm - heart_rate_bpm) < 4.0, (heart_rate_bpm, peaks_bpm)


def test_spectral_ignores_out_of_band_drift():
    t = np.arange(0, 8, 1 / 30)
    signal = 5 * np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 1.2 * t)
    assert abs(hr_spectral(signal, 30.0) - 72) < 1.5


def test_peaks_returns_nan_without_beats():
    assert np.isnan(hr_peaks(np.zeros(240), 30.0))
