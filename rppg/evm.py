"""Eulerian Video Magnification (Wu et al., "Eulerian Video Magnification for
Revealing Subtle Changes in the World", SIGGRAPH 2012).

Idea: instead of tracking pixels, look at every pixel's value over time,
keep only the frequencies of interest (here: the heart-rate band) and add
them back to the video multiplied by a gain magnification_gain.

* Colour magnification: done on a heavily blurred, downsampled video (a Gaussian
  pyramid level), because the pulse colour change is spatially smooth and
  blurring removes pixel noise that would otherwise be amplified too.
* Motion magnification: done on a Laplacian pyramid (band-pass images that keep
  edges). Small shifts of edges appear as intensity changes at each level;
  amplifying them makes the motion visible. The gain is limited per level so
  fine levels (noise, aliasing) are not over-amplified.

Both operate offline on a clip held in memory and assume roughly uniform frame
timing, which is fine for visualisation (not used for measurements).
"""
import cv2
import numpy as np
from scipy.signal import butter, sosfiltfilt


def temporal_bandpass(signal, sample_rate_hz, low_cutoff_hz, high_cutoff_hz, order=2):
    """Zero-phase Butterworth band-pass along the time axis (axis 0)."""
    filter_sections = butter(order, [low_cutoff_hz, high_cutoff_hz], btype="bandpass",
                             fs=sample_rate_hz, output="sos")
    return sosfiltfilt(filter_sections, signal, axis=0).astype(np.float32)


def gaussian_down(img, levels):
    for _ in range(levels):
        img = cv2.pyrDown(img)
    return img


def magnify_color(frames, sample_rate_hz, low_cutoff_hz, high_cutoff_hz, magnification_gain=80.0, levels=4):
    """frames: (T, H, W, 3) float32 in [0, 1]. Returns magnified frames, same shape."""
    n_frames, frame_height_px, frame_width_px, _ = frames.shape
    downsampled_frames = np.stack([gaussian_down(f, levels) for f in frames])
    magnified_boost = temporal_bandpass(downsampled_frames, sample_rate_hz, low_cutoff_hz,
                                        high_cutoff_hz) * magnification_gain
    magnified_frames = np.empty_like(frames)
    for i in range(n_frames):
        magnified_frames[i] = frames[i] + cv2.resize(
            magnified_boost[i], (frame_width_px, frame_height_px), interpolation=cv2.INTER_LINEAR)
    return np.clip(magnified_frames, 0.0, 1.0)


def laplacian_pyramid(img, levels):
    """List of `levels` band-pass images (finest first) followed by the low-pass residual."""
    pyramid, current_level = [], img
    for _ in range(levels):
        downsampled_level = cv2.pyrDown(current_level)
        upsampled_level = cv2.pyrUp(downsampled_level, dstsize=(current_level.shape[1], current_level.shape[0]))
        pyramid.append(current_level - upsampled_level)
        current_level = downsampled_level
    pyramid.append(current_level)
    return pyramid


def collapse(pyramid):
    current_level = pyramid[-1]
    for band in reversed(pyramid[:-1]):
        current_level = cv2.pyrUp(current_level, dstsize=(band.shape[1], band.shape[0])) + band
    return current_level


def level_gains(frame_height_px, frame_width_px, levels, magnification_gain, lambda_c=16.0):
    """Per-level gain following Wu et al.: magnification_gain is reduced for fine spatial
    scales where (1 + magnification_gain) * max_detail_amplitude would exceed lambda / 8
    and produce artefacts. The finest level and the residual get 0."""
    max_detail_amplitude = lambda_c / 8.0 / (1.0 + magnification_gain)
    representative_wavelength = np.sqrt(frame_height_px ** 2 + frame_width_px ** 2) / 3.0
    level_wavelengths = []
    for level_index in range(levels):      # computed coarse -> fine, stored fine -> coarse
        level_wavelengths.append(representative_wavelength / (2 ** (levels - 1 - level_index)))
    gains = [min(magnification_gain, level_wavelength / max_detail_amplitude / 8.0 - 1.0)
             for level_wavelength in level_wavelengths]
    gains = [max(0.0, gain) for gain in gains]
    gains[0] = 0.0                         # finest band: mostly noise
    return gains + [0.0]                   # residual: overall brightness, leave untouched


def magnify_motion(frames, sample_rate_hz, low_cutoff_hz, high_cutoff_hz, magnification_gain=15.0,
                   levels=4, lambda_c=16.0):
    """Linear Eulerian motion magnification on the luminance channel.
    frames: (T, H, W, 3) float32 RGB in [0, 1]. Returns magnified frames."""
    n_frames, frame_height_px, frame_width_px, _ = frames.shape
    ycc_frames = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb) for f in frames])
    gains = level_gains(frame_height_px, frame_width_px, levels, magnification_gain, lambda_c)

    frame_pyramids = [laplacian_pyramid(ycc_frames[i, :, :, 0], levels) for i in range(n_frames)]
    boosted_levels = []
    for level_index, level_gain in enumerate(gains):
        if level_gain == 0:
            boosted_levels.append(None)
            continue
        level_stack = np.stack([pyramid[level_index] for pyramid in frame_pyramids])
        boosted_levels.append(temporal_bandpass(level_stack, sample_rate_hz, low_cutoff_hz,
                                                 high_cutoff_hz) * level_gain)

    magnified_frames = np.empty_like(frames)
    for i in range(n_frames):
        delta = [np.zeros_like(frame_pyramids[i][level_index]) if boosted is None else boosted[i]
                 for level_index, boosted in enumerate(boosted_levels)]
        luma_channel = ycc_frames[i, :, :, 0] + collapse(delta)
        merged_ycc_frame = np.dstack(
            [luma_channel, ycc_frames[i, :, :, 1], ycc_frames[i, :, :, 2]]).astype(np.float32)
        magnified_frames[i] = cv2.cvtColor(merged_ycc_frame, cv2.COLOR_YCrCb2RGB)
    return np.clip(magnified_frames, 0.0, 1.0)
