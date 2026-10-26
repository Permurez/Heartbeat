"""Eulerian Video Magnification (Wu et al., "Eulerian Video Magnification for
Revealing Subtle Changes in the World", SIGGRAPH 2012).

Idea: instead of tracking pixels, look at every pixel's value over time,
keep only the frequencies of interest (here: the heart-rate band) and add
them back to the video multiplied by a gain alpha.

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


def temporal_bandpass(x, fs, lo_hz, hi_hz, order=2):
    """Zero-phase Butterworth band-pass along the time axis (axis 0)."""
    sos = butter(order, [lo_hz, hi_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0).astype(np.float32)


def gaussian_down(img, levels):
    for _ in range(levels):
        img = cv2.pyrDown(img)
    return img


def magnify_color(frames, fs, lo_hz, hi_hz, alpha=80.0, levels=4):
    """frames: (T, H, W, 3) float32 in [0, 1]. Returns magnified frames, same shape."""
    t, h, w, _ = frames.shape
    small = np.stack([gaussian_down(f, levels) for f in frames])
    boost = temporal_bandpass(small, fs, lo_hz, hi_hz) * alpha
    out = np.empty_like(frames)
    for i in range(t):
        out[i] = frames[i] + cv2.resize(boost[i], (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(out, 0.0, 1.0)


def laplacian_pyramid(img, levels):
    """List of `levels` band-pass images (finest first) followed by the low-pass residual."""
    pyr, cur = [], img
    for _ in range(levels):
        down = cv2.pyrDown(cur)
        up = cv2.pyrUp(down, dstsize=(cur.shape[1], cur.shape[0]))
        pyr.append(cur - up)
        cur = down
    pyr.append(cur)
    return pyr


def collapse(pyr):
    cur = pyr[-1]
    for band in reversed(pyr[:-1]):
        cur = cv2.pyrUp(cur, dstsize=(band.shape[1], band.shape[0])) + band
    return cur


def level_gains(h, w, levels, alpha, lambda_c=16.0):
    """Per-level gain following Wu et al.: alpha is reduced for fine spatial scales
    where (1 + alpha) * delta would exceed lambda / 8 and produce artefacts.
    The finest level and the residual get 0."""
    delta = lambda_c / 8.0 / (1.0 + alpha)
    lam = np.sqrt(h * h + w * w) / 3.0            # representative wavelength of the coarsest band
    gains = []
    for lvl in range(levels):                      # computed coarse -> fine, stored fine -> coarse
        lam_l = lam / (2 ** (levels - 1 - lvl))
        gains.append(lam_l)
    gains = [min(alpha, lam_l / delta / 8.0 - 1.0) for lam_l in gains]
    gains = [max(0.0, g) for g in gains]
    gains[0] = 0.0                                 # finest band: mostly noise
    return gains + [0.0]                           # residual: overall brightness, leave untouched


def magnify_motion(frames, fs, lo_hz, hi_hz, alpha=15.0, levels=4, lambda_c=16.0):
    """Linear Eulerian motion magnification on the luminance channel.
    frames: (T, H, W, 3) float32 RGB in [0, 1]. Returns magnified frames."""
    t, h, w, _ = frames.shape
    ycc = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2YCrCb) for f in frames])
    gains = level_gains(h, w, levels, alpha, lambda_c)

    pyrs = [laplacian_pyramid(ycc[i, :, :, 0], levels) for i in range(t)]
    boosted = []
    for lvl, g in enumerate(gains):
        if g == 0:
            boosted.append(None)
            continue
        stack = np.stack([p[lvl] for p in pyrs])
        boosted.append(temporal_bandpass(stack, fs, lo_hz, hi_hz) * g)

    out = np.empty_like(frames)
    for i in range(t):
        delta = [np.zeros_like(pyrs[i][lvl]) if b is None else b[i] for lvl, b in enumerate(boosted)]
        y = ycc[i, :, :, 0] + collapse(delta)
        merged = np.dstack([y, ycc[i, :, :, 1], ycc[i, :, :, 2]]).astype(np.float32)
        out[i] = cv2.cvtColor(merged, cv2.COLOR_YCrCb2RGB)
    return np.clip(out, 0.0, 1.0)
