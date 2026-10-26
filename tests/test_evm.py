import numpy as np

from rppg.evm import collapse, laplacian_pyramid, magnify_color, magnify_motion

FS = 30.0


def test_laplacian_pyramid_reconstructs_exactly():
    img = np.random.default_rng(0).random((96, 128)).astype(np.float32)
    assert np.allclose(collapse(laplacian_pyramid(img, 4)), img, atol=1e-5)


def test_color_magnification_amplifies_in_band_signal_only():
    t = np.arange(150) / FS
    pulse = 0.002 * np.sin(2 * np.pi * 1.5 * t)           # 90 BPM, invisible amplitude
    drift = 0.002 * np.sin(2 * np.pi * 0.1 * t)           # out of band
    frames = np.full((150, 64, 64, 3), 0.5, np.float32)
    frames += (pulse + drift)[:, None, None, None].astype(np.float32)
    out = magnify_color(frames, FS, 1.2, 1.8, alpha=50, levels=3)
    trace = out[:, 32, 32, 1] - frames[:, 32, 32, 1]
    core = slice(30, 120)                                  # ignore filter edge effects
    gain = np.std(trace[core]) / np.std(pulse[core])
    assert 30 < gain < 60


def test_motion_magnification_enlarges_small_oscillation():
    t = np.arange(150) / FS
    shift = 0.3 * np.sin(2 * np.pi * 1.5 * t)              # sub-pixel horizontal wobble
    xx = np.arange(128)[None, :]
    frames = []
    for s in shift:
        row = 1 / (1 + np.exp(-(xx - 64 - s) / 3.0))       # soft vertical edge
        img = np.repeat(row, 96, axis=0).astype(np.float32) * 0.6 + 0.2
        frames.append(np.dstack([img] * 3))
    frames = np.stack(frames)
    out = magnify_motion(frames, FS, 1.2, 1.8, alpha=10, levels=4)

    def edge_pos(f):
        prof = f[48, :, 0]
        return np.interp(0.5, prof[40:90], np.arange(40, 90))
    core = slice(30, 120)
    orig = np.array([edge_pos(f) for f in frames])[core]
    mag = np.array([edge_pos(f) for f in out])[core]
    assert np.std(mag) > 3 * np.std(orig)
