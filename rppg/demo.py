"""Three-panel demo video for presentations.

  [ ROIs + live heart rate ] [ motion magnified (EVM) ] [ colour magnified (EVM) ]

All panels show the same square crop around the face. The EVM frequency band is
centred on the heart rate estimated FROM THE VIDEO (not from the reference), so
the demo does not use information a real system would not have.
"""
import os
import subprocess

import cv2
import numpy as np

from . import evm, roi, ubfc
from .groundtruth import bandpass, hr_spectral
from .pipeline import load_subject, resample_subject


def rolling_hr(res, cfg):
    """HR estimates for 8 s windows ending every second: (t_end, hr_camera, hr_reference)."""
    w, step = int(cfg.window_s * cfg.fs), int(cfg.fs)
    green = res["rgb_lin"][:, :, 1]
    green = (green / green.mean(axis=0)).mean(axis=1)           # average of normalised ROIs
    t_end, cam, ref = [], [], []
    for s in range(0, len(res["t"]) - w + 1, step):
        sl = slice(s, s + w)
        t_end.append(res["t"][s + w - 1])
        cam.append(hr_spectral(bandpass(green[sl], cfg.fs, 0.7, 3.0), cfg.fs))
        ref.append(hr_spectral(bandpass(res["ppg"][sl], cfg.fs, 0.7, 3.0), cfg.fs))
    return np.array(t_end), np.array(cam), np.array(ref)


def _label(img, text, y, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def make_demo_video(cfg, subject="subject1", start_s=20.0, duration_s=10.0, out_path=None,
                    panel=360, alpha_motion=15.0, alpha_color=80.0, landmarker=None):
    """Render the demo clip to an .mp4 file and return its path."""
    out_path = out_path or os.path.join(cfg.cache_dir, f"demo_{subject}.mp4")
    raw = load_subject(cfg, subject)
    res = resample_subject(raw, cfg)
    t_end, hr_cam, hr_ref = rolling_hr(res, cfg)

    t_raw = raw["t"]
    idx = np.where((t_raw >= start_s) & (t_raw < start_s + duration_s))[0]
    if len(idx) == 0:
        raise ValueError("Clip is outside the recording")
    first, last = idx[0], idx[-1]

    # 1) Read the clip and run landmarks on it
    own_landmarker = landmarker is None
    if own_landmarker:
        roi.ensure_model(cfg.model_path, cfg.model_url)
        landmarker = roi.create_landmarker(cfg.model_path)
    import mediapipe as mp
    frames, lms = [], []
    try:
        with ubfc.extracted_video(cfg.zip_path, subject, cfg.tmp_dir) as video_path:
            w, h, fps, _ = roi.probe_video(video_path)
            for i, frame in enumerate(roi.iter_frames_rgb(video_path, w, h)):
                if i < first:
                    continue
                if i > last:
                    break
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame), int(round(i * 1000 / fps)))
                lm = (np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]], np.float32)
                      if result.face_landmarks else None)
                frames.append(frame)
                lms.append(lm)
    finally:
        if own_landmarker:
            landmarker.close()

    # 2) Square crop around the face (fixed for the whole clip, so motion stays visible)
    found = [lm for lm in lms if lm is not None]
    if not found:
        raise RuntimeError("No face found in the clip")
    centre = np.median([lm.mean(axis=0) for lm in found], axis=0)
    face_w = np.median([np.linalg.norm(lm[roi.FACE_EDGE[0]] - lm[roi.FACE_EDGE[1]]) for lm in found])
    side = int(min(1.6 * face_w, h, w))
    x0 = int(np.clip(centre[0] - side / 2, 0, w - side))
    y0 = int(np.clip(centre[1] - side / 2, 0, h - side))
    crop = np.stack([f[y0:y0 + side, x0:x0 + side] for f in frames]).astype(np.float32) / 255.0

    # 3) EVM in a band around the camera-estimated heart rate
    in_clip = (t_end >= start_s) & (t_end <= start_s + duration_s)
    f0 = np.median(hr_cam[in_clip] if in_clip.any() else hr_cam) / 60.0
    lo, hi = max(0.5, f0 - 0.25), f0 + 0.25
    eff_fs = len(idx) / (t_raw[last] - t_raw[first]) if last > first else cfg.fs
    motion = evm.magnify_motion(crop, eff_fs, lo, hi, alpha=alpha_motion)
    color = evm.magnify_color(crop, eff_fs, lo, hi, alpha=alpha_color)

    # 4) Compose frames and stream them to FFmpeg (H.264 mp4)
    header = 40
    width, height = 3 * panel, panel + header + 30
    width += width % 2
    height += height % 2
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", f"{eff_fs:.3f}", "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", out_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for k, i in enumerate(idx):
            canvas = np.zeros((height, width, 3), np.uint8)
            p1 = (crop[k] * 255).astype(np.uint8).copy()
            if lms[k] is not None:
                polys = roi.roi_polygons(lms[k] - [x0, y0], cfg.roi_shrink)
                cv2.polylines(p1, list(polys.values()), True, (0, 255, 0), 1, cv2.LINE_AA)
            panels = [p1, (motion[k] * 255).astype(np.uint8), (color[k] * 255).astype(np.uint8)]
            for j, p in enumerate(panels):
                canvas[header:header + panel, j * panel:(j + 1) * panel] = cv2.resize(
                    p, (panel, panel), interpolation=cv2.INTER_AREA)

            _label(canvas, "Skin ROIs (MediaPipe)", 26)
            cv2.putText(canvas, f"Motion x{alpha_motion:.0f} (EVM)", (panel + 10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Colour x{alpha_color:.0f} (EVM)", (2 * panel + 10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            j = max(0, np.searchsorted(t_end, t_raw[i], side="right") - 1)
            _label(canvas, f"t={t_raw[i]:5.1f}s   camera: {hr_cam[j]:5.1f} BPM   "
                           f"reference: {hr_ref[j]:5.1f} BPM   "
                           f"(EVM band {lo * 60:.0f}-{hi * 60:.0f} BPM)",
                   height - 10, 0.55, (255, 255, 120))
            proc.stdin.write(canvas.tobytes())
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed to write the video")
    return out_path
