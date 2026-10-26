"""End-to-end data preparation for UBFC-rPPG.

Stage 1  build_cache():        video -> per-frame ROI colour traces, saved as one .npz per subject.
                               Resumable: subjects already in the cache are skipped, so a Colab
                               disconnect only costs the subject that was in progress.
Stage 2  build_window_table(): cache -> resampled signals -> 8 s windows with reference HR and
                               quality flags, saved as windows.csv.
Algorithms later read the cache + windows.csv only; they never touch the 32 GB of video again.
"""
import os
import time

import cv2
import numpy as np
import pandas as pd

from . import roi, ubfc
from .config import Config
from .quality import build_windows
from .timeseries import clean_timestamps, resample_uniform, uniform_grid


def cache_path(cfg, subject):
    return os.path.join(cfg.cache_dir, f"{subject}.npz")


def build_cache(cfg: Config, subjects=None, overwrite=False):
    """Extract ROI traces for every subject that is not cached yet. Returns a summary DataFrame."""
    os.makedirs(cfg.cache_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.cache_dir, "debug"), exist_ok=True)
    os.makedirs(cfg.tmp_dir, exist_ok=True)
    roi.ensure_model(cfg.model_path, cfg.model_url)
    subjects = subjects or ubfc.list_subjects(cfg.zip_path)

    summary = []
    for n, subject in enumerate(subjects, 1):
        out = cache_path(cfg, subject)
        if os.path.exists(out) and not overwrite:
            print(f"[{n}/{len(subjects)}] {subject}: cached, skipping")
            continue
        print(f"[{n}/{len(subjects)}] {subject}: processing")
        t0 = time.time()

        ppg, hr, t = ubfc.load_ground_truth(cfg.zip_path, subject)
        # A new landmarker per video: VIDEO mode requires timestamps to keep increasing.
        landmarker = roi.create_landmarker(cfg.model_path)
        try:
            with ubfc.extracted_video(cfg.zip_path, subject, cfg.tmp_dir) as video_path:
                data, debug_img = roi.process_video(
                    video_path, landmarker, n_expected=len(t), shrink=cfg.roi_shrink,
                    clip_low=cfg.clip_low, clip_high=cfg.clip_high)
        finally:
            landmarker.close()

        n_video = len(data["face_found"])
        n = min(n_video, len(t))
        per_frame = ("mean_raw", "mean_lin", "n_pixels", "clipped", "face_found", "motion_xy", "face_width")
        payload = {k: data[k][:n] for k in per_frame}
        payload.update(
            t=t[:n], ppg=ppg[:n], hr_oximeter=hr[:n],
            frames_gt=len(t), frames_video=n_video,
            video_fps_meta=data["video_fps_meta"], video_frames_meta=data["video_frames_meta"],
            roi_names=np.array(roi.ROI_NAMES), motion_landmarks=np.array(roi.MOTION_LANDMARKS),
        )
        tmp = out + ".tmp.npz"              # write then rename: a crash never leaves a half-written file
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, out)
        if debug_img is not None:
            cv2.imwrite(os.path.join(cfg.cache_dir, "debug", f"{subject}.png"),
                        cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

        row = {"subject": subject, "frames_gt": len(t), "frames_video": n_video,
               "face_found": round(float(payload["face_found"].mean()), 4),
               "seconds": round(time.time() - t0, 1)}
        summary.append(row)
        print(f"    done: {row}")
    return pd.DataFrame(summary)


def load_subject(cfg, subject):
    """Load one cached subject as a dict of numpy arrays."""
    with np.load(cache_path(cfg, subject), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def resample_subject(raw, cfg):
    """Put every signal of one subject on a shared uniform time grid.

    Frames without a face are NaN in the video signals; they are dropped and
    therefore show up as gaps (valid_video=False) if longer than cfg.max_gap.
    """
    t_c, _, _ = clean_timestamps(raw["t"], raw["ppg"])
    grid = uniform_grid(t_c[0], t_c[-1], cfg.fs)

    def rs(x):
        return resample_uniform(raw["t"], x, cfg.fs, "spline", cfg.max_gap, t_grid=grid)

    _, ppg, valid_gt, report = rs(raw["ppg"])
    _, rgb_raw, valid_video, report_video = rs(raw["mean_raw"])
    _, rgb_lin, _, _ = rs(raw["mean_lin"])
    _, motion_xy, _, _ = rs(raw["motion_xy"])
    _, face_width, _, _ = rs(raw["face_width"])
    return {"t": grid, "ppg": ppg, "rgb_raw": rgb_raw, "rgb_lin": rgb_lin,
            "motion_xy": motion_xy, "face_width": face_width,
            "valid_gt": valid_gt, "valid_video": valid_video,
            "report_gt": report, "report_video": report_video}


def build_window_table(cfg: Config, subjects=None):
    """Windows with reference HR + quality flags for every cached subject -> windows.csv."""
    subjects = subjects or sorted(
        (f[:-4] for f in os.listdir(cfg.cache_dir) if f.startswith("subject") and f.endswith(".npz")),
        key=lambda s: int(s[len("subject"):]))
    rows = []
    for subject in subjects:
        try:
            raw = load_subject(cfg, subject)
            res = resample_subject(raw, cfg)
            rows.extend(build_windows(subject, raw, res, cfg))
        except Exception as exc:            # one broken subject must not stop the whole table
            print(f"{subject}: skipped ({type(exc).__name__}: {exc})")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(cfg.cache_dir, "windows.csv"), index=False)
    return df
