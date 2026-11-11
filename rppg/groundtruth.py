"""Heart rate from a pulse waveform (used for the reference PPG; reusable for rPPG output).

Two independent estimators are provided:
  * spectral: frequency of the strongest peak in the heart-rate band,
  * temporal: median interval between detected beats.
If they disagree on a clean reference signal, that window of the reference
itself is unreliable and must not be used to judge an algorithm.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, detrend


def bandpass(signal, sample_rate_hz, low_cutoff_hz, high_cutoff_hz, order=3):
    """Zero-phase Butterworth band-pass (filtfilt runs the filter forward and backward)."""
    filter_numerator, filter_denominator = butter(order, [low_cutoff_hz, high_cutoff_hz], btype="bandpass", fs=sample_rate_hz) #
    filter_length = max(len(filter_numerator), len(filter_denominator))
    edge_padding = min(3 * filter_length, len(signal) - 1)   # filtfilt needs room to pad both edges
    return filtfilt(filter_numerator, filter_denominator, signal, padlen=edge_padding)


def hr_spectral(signal, sample_rate_hz, heart_rate_min_bpm=42.0, heart_rate_max_bpm=180.0, fft_length=2 ** 14):
    """HR [BPM] from the dominant spectral peak inside the HR band.

    Zero-padding to fft_length does not add information; it only samples the spectrum
    more densely so the peak position can be read with sub-BPM precision.
    """
    signal = detrend(np.asarray(signal, float))
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)), n=fft_length))
    freqs = np.fft.rfftfreq(fft_length, d=1.0 / sample_rate_hz)
    band = (freqs >= heart_rate_min_bpm / 60) & (freqs <= heart_rate_max_bpm / 60)
    return float(freqs[band][np.argmax(spectrum[band])] * 60.0)


def hr_peaks(signal, sample_rate_hz, heart_rate_max_bpm=180.0):
    """HR [BPM] from the median beat-to-beat interval. Returns NaN if < 3 beats."""
    signal = np.asarray(signal, float)
    min_distance = max(1, int(sample_rate_hz * 60.0 / heart_rate_max_bpm))   # beats cannot be closer than this
    prominence = 0.3 * np.std(signal)
    peaks, _ = find_peaks(signal, distance=min_distance, prominence=prominence)
    if len(peaks) < 3:
        return float("nan")
    return float(60.0 / np.median(np.diff(peaks) / sample_rate_hz))


def reference_hr(ppg_window, sample_rate_hz, heart_rate_min_bpm=42.0, heart_rate_max_bpm=180.0):
    """Filter a reference PPG window and return (hr_spectral, hr_peaks)."""
    low_cutoff_hz = heart_rate_min_bpm / 60
    high_cutoff_hz = heart_rate_max_bpm / 60
    filtered = bandpass(ppg_window, sample_rate_hz, low_cutoff_hz, high_cutoff_hz)
    hr_from_spectrum_bpm = hr_spectral(filtered, sample_rate_hz, heart_rate_min_bpm, heart_rate_max_bpm)
    hr_from_beats_bpm = hr_peaks(filtered, sample_rate_hz, heart_rate_max_bpm)
    return hr_from_spectrum_bpm, hr_from_beats_bpm
