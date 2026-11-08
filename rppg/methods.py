"""rPPG pulse-extraction methods: turn a per-frame RGB trace (ROI colour means
over time) into a single pulse signal, from which heart rate can be read
(see groundtruth.hr_spectral / hr_peaks).

References:
  GREEN : Verkruysse, Svaasand & Nelson, "Remote plethysmographic imaging
          using ambient light", Optics Express, 2008.
  CHROM : de Haan & Jeanne, "Robust Pulse Rate From Chrominance-Based rPPG",
          IEEE Trans. Biomed. Eng., 2013.
  POS   : Wang, den Brinker, Stuijk & de Haan, "Algorithmic Principles of
          Remote PPG", IEEE Trans. Biomed. Eng., 2017.
"""
import numpy as np
from scipy.signal import detrend

from .groundtruth import bandpass


def green(rgb_trace):
    """Pulse signal = detrended green channel (Verkruysse et al., 2008).

    Haemoglobin absorbs green light more strongly than red, so the green
    channel carries the largest pulse-related variation. Simple and fast,
    but has no defence against brightness changes from motion or lighting:
    those show up in the green channel too and can swamp the pulse.

    rgb_trace: (T, 3) array of R, G, B means over time, any units.
    """
    return detrend(np.asarray(rgb_trace, float)[:, 1])


def chrom(rgb_trace, sample_rate_hz, low_cutoff_hz, high_cutoff_hz, order=3):
    """CHROM pulse signal (de Haan & Jeanne, 2013).

    Builds two "chrominance" combinations of the normalised R, G, B traces
    that, under a standard skin-reflectance model, respond to the blood
    volume pulse but carry any remaining specular/intensity artefact in a
    fixed, known ratio to each other. Subtracting them in that ratio removes
    the artefact and keeps the pulse. rgb_trace: (T, 3) means over time.
    """
    rgb_trace = np.asarray(rgb_trace, float)
    red_norm, green_norm, blue_norm = (rgb_trace[:, i] / rgb_trace[:, i].mean() for i in range(3))
    chroma_x = 3 * red_norm - 2 * green_norm
    chroma_y = 1.5 * red_norm + green_norm - 1.5 * blue_norm
    chroma_x_filtered = bandpass(chroma_x, sample_rate_hz, low_cutoff_hz, high_cutoff_hz, order)
    chroma_y_filtered = bandpass(chroma_y, sample_rate_hz, low_cutoff_hz, high_cutoff_hz, order)
    amplitude_ratio = chroma_x_filtered.std() / chroma_y_filtered.std() if chroma_y_filtered.std() > 0 else 0.0
    return chroma_x_filtered - amplitude_ratio * chroma_y_filtered


def pos(rgb_trace, sample_rate_hz, sub_window_s=1.6):
    """POS (Plane-Orthogonal-to-Skin) pulse signal (Wang et al., 2017).

    In normalised RGB space, overall brightness/specular changes move the
    signal along the (1, 1, 1) direction (the "skin-tone" axis); the blood
    volume pulse must therefore live in the plane orthogonal to it. For every
    short sliding sub-window the RGB trace is normalised by its own temporal
    mean, projected onto that plane (S1 = G - B, S2 = -2R + G + B), and the
    two axes combined with a ratio that equalises their amplitude. Sub-windows
    are then overlapped and added (each de-meaned first), which reinforces
    the shared pulse component and cancels slow drift without an explicit
    filter cutoff. rgb_trace: (T, 3) means over time.
    """
    rgb_trace = np.asarray(rgb_trace, float)
    n_samples = len(rgb_trace)
    sub_window_len = max(3, int(round(sub_window_s * sample_rate_hz)))
    pulse_signal = np.zeros(n_samples)
    for start in range(0, max(1, n_samples - sub_window_len + 1)):
        segment = rgb_trace[start:start + sub_window_len]
        segment_mean = segment.mean(axis=0)
        segment_norm = segment / np.where(segment_mean == 0, 1.0, segment_mean)
        s1 = segment_norm[:, 1] - segment_norm[:, 2]
        s2 = -2 * segment_norm[:, 0] + segment_norm[:, 1] + segment_norm[:, 2]
        amplitude_ratio = s1.std() / s2.std() if s2.std() > 0 else 0.0
        combined = s1 + amplitude_ratio * s2
        pulse_signal[start:start + len(segment)] += combined - combined.mean()
    return pulse_signal
