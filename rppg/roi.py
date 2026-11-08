"""Face landmarks (MediaPipe Face Landmarker) -> skin ROIs -> per-frame colour statistics.

For every frame and every ROI (forehead, left cheek, right cheek) we store:
  * mean R, G, B of the raw camera values (gamma-encoded sRGB),
  * mean R, G, B after undoing the sRGB gamma (linear light, proportional to
    the amount of light reaching the sensor, which is what the optical models
    behind CHROM/POS assume),
  * number of ROI pixels and share of clipped (under/over-exposed) pixels.
Plus head-motion data: a subset of landmark positions and the face width.

Landmark indices refer to the 478-point MediaPipe face mesh. ROIs avoid eyes,
eyebrows, nostrils, lips and the hairline; check them visually with the debug
images written by the pipeline before trusting them.
"""
import json
import os
import shutil
import subprocess
import urllib.request

import cv2
import numpy as np

ROI_LANDMARKS = {
    "forehead": [10, 338, 297, 299, 337, 9, 108, 69, 67, 109],
    "left_cheek": [117, 118, 101, 36, 205, 187, 123, 50],     # image-left = subject's right
    "right_cheek": [346, 347, 330, 266, 425, 411, 352, 280],
}
ROI_NAMES = list(ROI_LANDMARKS)

# Stable points kept for head-motion analysis (nose, forehead, eye corners, cheeks, chin, face edges)
MOTION_LANDMARKS = [1, 4, 5, 6, 9, 10, 151, 168, 195, 197,
                    33, 133, 362, 263, 50, 280, 205, 425, 152, 234, 454]
FACE_EDGE = (234, 454)   # used to measure face width in pixels


def srgb_to_linear_lut():
    """Lookup table: 8-bit sRGB value -> linear light in [0, 1] (IEC 61966-2-1)."""
    c = np.arange(256) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def shrink_polygon(points, factor):
    """Move polygon vertices toward their centroid; factor=1 keeps it unchanged."""
    centre = points.mean(axis=0)
    return centre + (points - centre) * factor


def roi_polygons(landmarks_px, shrink=0.85):
    """Convex hull polygons (int32, OpenCV format) for every ROI from (478, 2) pixel landmarks."""
    roi_shapes = {}
    for name, landmark_indices in ROI_LANDMARKS.items():
        polygon_points = shrink_polygon(landmarks_px[landmark_indices], shrink)
        roi_shapes[name] = cv2.convexHull(np.round(polygon_points).astype(np.int32))
    return roi_shapes


def roi_stats(frame_rgb, roi_shapes, lut, clip_low=5, clip_high=250):
    """Colour statistics of each ROI polygon in one RGB frame.

    Returns arrays ordered like ROI_NAMES:
      mean_raw (R, 3), mean_lin (R, 3), pixel_count (R,), clipped_fraction (R,)
    """
    frame_height_px, frame_width_px = frame_rgb.shape[:2]
    n_roi = len(ROI_NAMES)
    mean_raw = np.full((n_roi, 3), np.nan, np.float32)
    mean_lin = np.full((n_roi, 3), np.nan, np.float32)
    pixel_count = np.zeros(n_roi, np.int32)
    clipped_fraction = np.full(n_roi, np.nan, np.float32)

    for i, name in enumerate(ROI_NAMES):
        roi_mask = np.zeros((frame_height_px, frame_width_px), np.uint8)
        cv2.fillConvexPoly(roi_mask, roi_shapes[name], 1)
        roi_pixel_values = frame_rgb[roi_mask.astype(bool)]        # (P, 3) uint8
        if len(roi_pixel_values) == 0:
            continue
        pixel_count[i] = len(roi_pixel_values)
        mean_raw[i] = roi_pixel_values.mean(axis=0)
        mean_lin[i] = lut[roi_pixel_values].mean(axis=0)
        is_clipped = np.any((roi_pixel_values <= clip_low) | (roi_pixel_values >= clip_high), axis=1)
        clipped_fraction[i] = is_clipped.mean()
    return mean_raw, mean_lin, pixel_count, clipped_fraction


def probe_video(path):
    """Width, height, sample rate [Hz] and frame count of the first video stream (via ffprobe)."""
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found: install FFmpeg (apt-get install ffmpeg)")
    ffprobe_output = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    stream_info = json.loads(ffprobe_output)["streams"][0]

    def rate(txt):
        num, _, den = txt.partition("/")
        return float(num) / float(den or 1) if txt and float(den or 1) else 0.0

    sample_rate_hz = (rate(stream_info.get("avg_frame_rate", ""))
                      or rate(stream_info.get("r_frame_rate", "")) or 30.0)
    n_frames = stream_info.get("nb_frames")
    return (int(stream_info["width"]), int(stream_info["height"]), sample_rate_hz,
            int(n_frames) if n_frames and n_frames.isdigit() else -1)


def iter_frames_rgb(path, frame_width_px, frame_height_px):
    """Yield decoded frames as (H, W, 3) uint8 RGB arrays.

    FFmpeg decodes the file and streams raw pixels through a pipe. This avoids
    OpenCV's video backend, which crashes on UBFC files in some OpenCV builds.
    """
    ffmpeg_command = ["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    frame_bytes = frame_width_px * frame_height_px * 3
    # stderr is silenced: stopping early (we only need part of a video) makes FFmpeg
    # print a harmless 'Broken pipe' message.
    ffmpeg_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    try:
        while True:
            raw_frame_bytes = ffmpeg_process.stdout.read(frame_bytes)
            if len(raw_frame_bytes) < frame_bytes:
                break
            # copy(): frombuffer gives a read-only view; MediaPipe and OpenCV drawing need a normal array
            yield np.frombuffer(raw_frame_bytes, np.uint8).reshape(frame_height_px, frame_width_px, 3).copy()
    finally:
        ffmpeg_process.stdout.close()
        ffmpeg_process.kill()
        ffmpeg_process.wait()


def ensure_model(model_path, model_url):
    """Download the Face Landmarker model once (~4 MB) if it is not on disk yet."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
    return model_path


def create_landmarker(model_path, num_faces=1):
    """MediaPipe Face Landmarker in VIDEO mode (uses tracking between frames)."""
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=num_faces,
    )
    return vision.FaceLandmarker.create_from_options(options)


def process_video(video_path, landmarker, n_expected=None, shrink=0.85,
                  clip_low=5, clip_high=250, debug_frame_idx=None, log_every=500):
    """Run landmarks + ROI statistics over every frame of a video.

    Returns (data, debug_image). data is a dict of per-frame numpy arrays;
    frames without a detected face are NaN. debug_image is an RGB frame with ROIs
    drawn on it (or None).
    """
    import mediapipe as mp

    frame_width_px, frame_height_px, sample_rate_hz, n_frames = probe_video(video_path)
    if debug_frame_idx is None:
        debug_frame_idx = max(n_frames, n_expected or 0) // 2
    lut = srgb_to_linear_lut()

    rows = {k: [] for k in ("mean_raw", "mean_lin", "n_pixels", "clipped_fraction",
                            "face_found", "motion_xy", "face_width")}
    debug_image = None
    frame_index = 0
    for frame in iter_frames_rgb(video_path, frame_width_px, frame_height_px):

        # VIDEO mode needs strictly increasing timestamps; frame index based ones are fine here,
        # the real (irregular) timestamps are applied later from the ground-truth file.
        result = landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame),
            int(round(frame_index * 1000 / sample_rate_hz)))

        if result.face_landmarks:
            landmarks_px = np.array([(p.x * frame_width_px, p.y * frame_height_px)
                                     for p in result.face_landmarks[0]], np.float32)
            roi_shapes = roi_polygons(landmarks_px, shrink)
            mean_raw, mean_lin, pixel_count, clipped_fraction = roi_stats(
                frame, roi_shapes, lut, clip_low, clip_high)
            rows["face_found"].append(True)
            rows["motion_xy"].append(landmarks_px[MOTION_LANDMARKS])
            rows["face_width"].append(
                np.linalg.norm(landmarks_px[FACE_EDGE[0]] - landmarks_px[FACE_EDGE[1]]))
            if frame_index == debug_frame_idx:
                debug_image = frame.copy()
                cv2.polylines(debug_image, list(roi_shapes.values()), True, (0, 255, 0), 1)
                for x, y in landmarks_px[MOTION_LANDMARKS]:
                    cv2.circle(debug_image, (int(x), int(y)), 2, (255, 0, 0), -1)
        else:
            n_roi = len(ROI_NAMES)
            mean_raw = mean_lin = np.full((n_roi, 3), np.nan, np.float32)
            pixel_count = np.zeros(n_roi, np.int32)
            clipped_fraction = np.full(n_roi, np.nan, np.float32)
            rows["face_found"].append(False)
            rows["motion_xy"].append(np.full((len(MOTION_LANDMARKS), 2), np.nan, np.float32))
            rows["face_width"].append(np.nan)

        rows["mean_raw"].append(mean_raw)
        rows["mean_lin"].append(mean_lin)
        rows["n_pixels"].append(pixel_count)
        rows["clipped_fraction"].append(clipped_fraction)
        frame_index += 1
        if log_every and frame_index % log_every == 0:
            print(f"    frame {frame_index}/{n_frames}")
        if n_expected is not None and frame_index >= n_expected:
            break

    data = {k: np.asarray(v) for k, v in rows.items()}
    data["video_fps_meta"] = np.float32(sample_rate_hz)
    data["video_frames_meta"] = np.int32(n_frames)
    return data, debug_image
