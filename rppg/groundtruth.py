"""Heart rate from a pulse waveform (used for the reference PPG; reusable for rPPG output).

Two independent estimators are provided:
  * spectral: frequency of the strongest peak in the heart-rate band,
  * temporal: median interval between detected beats.
If they disagree on a clean reference signal, that window of the reference
itself is unreliable and must not be used to judge an algorithm.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, detrend


def bandpass(x, fs, lo_hz, hi_hz, order=3):
    """Zero-phase Butterworth band-pass (filtfilt runs the filter forward and backward)."""
    b, a = butter(order, [lo_hz, hi_hz], btype="bandpass", fs=fs)
    return filtfilt(b, a, x, padlen=min(3 * max(len(a), len(b)), len(x) - 1))


def hr_spectral(x, fs, hr_min_bpm=42.0, hr_max_bpm=180.0, nfft=2 ** 14):
    """HR [BPM] from the dominant spectral peak inside the HR band.

    Zero-padding to nfft does not add information; it only samples the spectrum
    more densely so the peak position can be read with sub-BPM precision.
    """
    x = detrend(np.asarray(x, float))
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)), n=nfft))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    band = (freqs >= hr_min_bpm / 60) & (freqs <= hr_max_bpm / 60)
    return float(freqs[band][np.argmax(spec[band])] * 60.0)


def hr_peaks(x, fs, hr_max_bpm=180.0):
    """HR [BPM] from the median beat-to-beat interval. Returns NaN if < 3 beats."""
    x = np.asarray(x, float)
    min_dist = max(1, int(fs * 60.0 / hr_max_bpm))      # beats cannot be closer than this
    prominence = 0.3 * np.std(x)
    peaks, _ = find_peaks(x, distance=min_dist, prominence=prominence)
    if len(peaks) < 3:
        return float("nan")
    return float(60.0 / np.median(np.diff(peaks) / fs))


def reference_hr(ppg_window, fs, hr_min_bpm=42.0, hr_max_bpm=180.0):
    """Filter a reference PPG window and return (hr_spectral, hr_peaks)."""
    filt = bandpass(ppg_window, fs, hr_min_bpm / 60, hr_max_bpm / 60)
    return (hr_spectral(filt, fs, hr_min_bpm, hr_max_bpm),
            hr_peaks(filt, fs, hr_max_bpm))
