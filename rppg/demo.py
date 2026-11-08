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
    """HR estimates for 8 s windows ending every second: (window_end_s, camera_heart_rate_bpm,
    reference_heart_rate_bpm)."""
    window_length_samples = int(cfg.window_s * cfg.sample_rate_hz)
    step_samples = int(cfg.sample_rate_hz)
    green_trace = res["rgb_lin"][:, :, 1]
    green_trace = (green_trace / green_trace.mean(axis=0)).mean(axis=1)    # average of normalised ROIs
    window_end_s, camera_heart_rate_bpm, reference_heart_rate_bpm = [], [], []
    for start in range(0, len(res["timestamps_s"]) - window_length_samples + 1, step_samples):
        window_slice = slice(start, start + window_length_samples)
        window_end_s.append(res["timestamps_s"][start + window_length_samples - 1])
        camera_heart_rate_bpm.append(
            hr_spectral(bandpass(green_trace[window_slice], cfg.sample_rate_hz, 0.7, 3.0), cfg.sample_rate_hz))
        reference_heart_rate_bpm.append(
            hr_spectral(bandpass(res["ppg"][window_slice], cfg.sample_rate_hz, 0.7, 3.0), cfg.sample_rate_hz))
    return np.array(window_end_s), np.array(camera_heart_rate_bpm), np.array(reference_heart_rate_bpm)


def _label(img, text, y, scale=0.6, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def make_demo_video(cfg, subject="subject1", start_s=20.0, duration_s=10.0, out_path=None,
                    panel=360, magnification_gain_motion=15.0, magnification_gain_color=80.0, landmarker=None):
    """Render the demo clip to an .mp4 file and return its path."""
    out_path = out_path or os.path.join(cfg.cache_dir, f"demo_{subject}.mp4")
    raw = load_subject(cfg, subject)
    res = resample_subject(raw, cfg)
    window_end_s, camera_heart_rate_bpm, reference_heart_rate_bpm = rolling_hr(res, cfg)

    raw_timestamps_s = raw["timestamps_s"]
    clip_frame_indices = np.where((raw_timestamps_s >= start_s) & (raw_timestamps_s < start_s + duration_s))[0]
    if len(clip_frame_indices) == 0:
        raise ValueError("Clip is outside the recording")
    first, last = clip_frame_indices[0], clip_frame_indices[-1]

    # 1) Read the clip and run landmarks on it
    own_landmarker = landmarker is None
    if own_landmarker:
        roi.ensure_model(cfg.model_path, cfg.model_url)
        landmarker = roi.create_landmarker(cfg.model_path)
    import mediapipe as mp
    frames, landmarks_per_frame = [], []
    try:
        with ubfc.extracted_video(cfg.zip_path, subject, cfg.tmp_dir) as video_path:
            frame_width_px, frame_height_px, sample_rate_hz, _ = roi.probe_video(video_path)
            for frame_index, frame in enumerate(roi.iter_frames_rgb(video_path, frame_width_px, frame_height_px)):
                if frame_index < first:
                    continue
                if frame_index > last:
                    break
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),
                    int(round(frame_index * 1000 / sample_rate_hz)))
                landmarks_px = (np.array([(p.x * frame_width_px, p.y * frame_height_px)
                                          for p in result.face_landmarks[0]], np.float32)
                               if result.face_landmarks else None)
                frames.append(frame)
                landmarks_per_frame.append(landmarks_px)
    finally:
        if own_landmarker:
            landmarker.close()

    # 2) Square crop around the face (fixed for the whole clip, so motion stays visible)
    found_landmarks = [lm for lm in landmarks_per_frame if lm is not None]
    if not found_landmarks:
        raise RuntimeError("No face found in the clip")
    centre = np.median([lm.mean(axis=0) for lm in found_landmarks], axis=0)
    face_width_px = np.median(
        [np.linalg.norm(lm[roi.FACE_EDGE[0]] - lm[roi.FACE_EDGE[1]]) for lm in found_landmarks])
    side = int(min(1.6 * face_width_px, frame_height_px, frame_width_px))
    x0 = int(np.clip(centre[0] - side / 2, 0, frame_width_px - side))
    y0 = int(np.clip(centre[1] - side / 2, 0, frame_height_px - side))
    face_crop = np.stack([f[y0:y0 + side, x0:x0 + side] for f in frames]).astype(np.float32) / 255.0

    # 3) EVM in a band around the camera-estimated heart rate
    in_clip = (window_end_s >= start_s) & (window_end_s <= start_s + duration_s)
    evm_center_freq_hz = np.median(
        camera_heart_rate_bpm[in_clip] if in_clip.any() else camera_heart_rate_bpm) / 60.0
    low_cutoff_hz = max(0.5, evm_center_freq_hz - 0.25)
    high_cutoff_hz = evm_center_freq_hz + 0.25
    effective_sample_rate_hz = (len(clip_frame_indices) / (raw_timestamps_s[last] - raw_timestamps_s[first])
                                if last > first else cfg.sample_rate_hz)
    motion_panel = evm.magnify_motion(face_crop, effective_sample_rate_hz, low_cutoff_hz, high_cutoff_hz,
                                      magnification_gain=magnification_gain_motion)
    color_panel = evm.magnify_color(face_crop, effective_sample_rate_hz, low_cutoff_hz, high_cutoff_hz,
                                    magnification_gain=magnification_gain_color)

    # 4) Compose frames and stream them to FFmpeg (H.264 mp4)
    header = 40
    width, height = 3 * panel, panel + header + 30
    width += width % 2
    height += height % 2
    ffmpeg_command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                      "-s", f"{width}x{height}", "-r", f"{effective_sample_rate_hz:.3f}", "-i", "-",
                      "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", out_path]
    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)
    try:
        for clip_index, frame_index in enumerate(clip_frame_indices):
            canvas = np.zeros((height, width, 3), np.uint8)
            roi_panel_image = (face_crop[clip_index] * 255).astype(np.uint8).copy()
            if landmarks_per_frame[clip_index] is not None:
                roi_shapes = roi.roi_polygons(landmarks_per_frame[clip_index] - [x0, y0], cfg.roi_shrink)
                cv2.polylines(roi_panel_image, list(roi_shapes.values()), True, (0, 255, 0), 1, cv2.LINE_AA)
            panels = [roi_panel_image, (motion_panel[clip_index] * 255).astype(np.uint8),
                     (color_panel[clip_index] * 255).astype(np.uint8)]
            for j, p in enumerate(panels):
                canvas[header:header + panel, j * panel:(j + 1) * panel] = cv2.resize(
                    p, (panel, panel), interpolation=cv2.INTER_AREA)

            _label(canvas, "Skin ROIs (MediaPipe)", 26)
            cv2.putText(canvas, f"Motion x{magnification_gain_motion:.0f} (EVM)", (panel + 10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"Colour x{magnification_gain_color:.0f} (EVM)", (2 * panel + 10, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            window_at_frame = max(0, np.searchsorted(
                window_end_s, raw_timestamps_s[frame_index], side="right") - 1)
            _label(canvas, f"t={raw_timestamps_s[frame_index]:5.1f}s   "
                           f"camera: {camera_heart_rate_bpm[window_at_frame]:5.1f} BPM   "
                           f"reference: {reference_heart_rate_bpm[window_at_frame]:5.1f} BPM   "
                           f"(EVM band {low_cutoff_hz * 60:.0f}-{high_cutoff_hz * 60:.0f} BPM)",
                   height - 10, 0.55, (255, 255, 120))
            ffmpeg_process.stdin.write(canvas.tobytes())
    finally:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
    if ffmpeg_process.returncode != 0:
        raise RuntimeError("ffmpeg failed to write the video")
    return out_path
