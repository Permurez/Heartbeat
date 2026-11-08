"""Tests for the GREEN / CHROM / POS pulse-extraction methods on synthetic
ROI colour traces (no video, no face detection: these methods only see
three numbers per frame)."""
import numpy as np

from rppg.groundtruth import hr_spectral
from rppg.methods import chrom, green, pos

SAMPLE_RATE_HZ = 30.0
HEART_RATE_BPM = 72.0


def make_rgb_trace(duration_s=12.0, brightness_amplitude=0.0, brightness_bpm=110.0):
    """Synthetic ROI trace: a skin-tone baseline with a pulsatile absorption
    dip that is strongest in G and weakest in B (matching real skin optics,
    see methods.py docstrings), plus an optional common-mode brightness
    oscillation (the same fractional change in every channel, as from
    lighting or head motion) at a different, still in-band, frequency.
    """
    t = np.arange(0, duration_s, 1 / SAMPLE_RATE_HZ)
    pulse = np.sin(2 * np.pi * HEART_RATE_BPM / 60 * t)
    brightness = brightness_amplitude * np.sin(2 * np.pi * brightness_bpm / 60 * t)
    red = 200 * (1 + 0.003 * pulse + brightness)
    green_channel = 150 * (1 + 0.012 * pulse + brightness)
    blue = 100 * (1 + 0.002 * pulse + brightness)
    return np.stack([red, green_channel, blue], axis=1)


def test_green_recovers_hr():
    assert abs(hr_spectral(green(make_rgb_trace()), SAMPLE_RATE_HZ) - HEART_RATE_BPM) < 1.5


def test_chrom_recovers_hr():
    sig = chrom(make_rgb_trace(), SAMPLE_RATE_HZ, 0.7, 3.0)
    assert abs(hr_spectral(sig, SAMPLE_RATE_HZ) - HEART_RATE_BPM) < 1.5


def test_pos_recovers_hr():
    sig = pos(make_rgb_trace(), SAMPLE_RATE_HZ)
    assert abs(hr_spectral(sig, SAMPLE_RATE_HZ) - HEART_RATE_BPM) < 1.5


def test_green_is_fooled_by_common_mode_brightness():
    """A brightness/motion artefact that changes every channel by the same
    fraction dominates the plain green channel: this is exactly the failure
    mode CHROM and POS are designed to fix (see the next two tests)."""
    rgb_trace = make_rgb_trace(brightness_amplitude=0.03, brightness_bpm=110.0)
    assert abs(hr_spectral(green(rgb_trace), SAMPLE_RATE_HZ) - 110.0) < 1.5


def test_chrom_rejects_common_mode_brightness():
    rgb_trace = make_rgb_trace(brightness_amplitude=0.03, brightness_bpm=110.0)
    sig = chrom(rgb_trace, SAMPLE_RATE_HZ, 0.7, 3.0)
    assert abs(hr_spectral(sig, SAMPLE_RATE_HZ) - HEART_RATE_BPM) < 1.5


def test_pos_rejects_common_mode_brightness():
    rgb_trace = make_rgb_trace(brightness_amplitude=0.03, brightness_bpm=110.0)
    sig = pos(rgb_trace, SAMPLE_RATE_HZ)
    assert abs(hr_spectral(sig, SAMPLE_RATE_HZ) - HEART_RATE_BPM) < 1.5
