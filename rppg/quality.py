"""Sliding windows + data-quality rules.

Rules look ONLY at input quality (timestamps, face detection, exposure, the
reference signal itself), never at how well an rPPG algorithm performed.
Rejecting windows because an algorithm failed on them would be cherry-picking.
Every report must state the coverage (share of accepted windows).
"""
import numpy as np

from .groundtruth import reference_hr


def head_motion(motion_xy, face_width):
    """Relative head motion in a window: std of the landmark centroid displacement
    divided by the median face width (dimensionless, comparable across subjects)."""
    centroid = np.nanmean(motion_xy, axis=1)                  # (T, 2)
    if np.all(np.isnan(centroid)):
        return float("nan")
    disp = centroid - np.nanmean(centroid, axis=0)
    return float(np.nanstd(np.linalg.norm(disp, axis=1)) / np.nanmedian(face_width))


def build_windows(subject, raw, res, cfg):
    """Cut one subject into overlapping windows and evaluate quality rules.

    raw: per-frame cache (dict from the .npz file), res: resampled dict (see pipeline.resample_subject).
    Returns a list of dicts, one per window.
    """
    t, t_raw = res["t"], raw["t"]
    rows = []
    starts = np.arange(t[0], t[-1] - cfg.window_s + 1e-9, cfg.step_s)
    for k, s in enumerate(starts):
        e = s + cfg.window_s
        g = (t >= s) & (t < e)                  # samples on the uniform grid
        r = (t_raw >= s) & (t_raw < e)          # original frames

        gap_frac = float(1 - res["valid_video"][g].mean())
        gt_gap_frac = float(1 - res["valid_gt"][g].mean())
        face_missing = float(1 - raw["face_found"][r].mean())
        clipped = float(np.nanmean(raw["clipped"][r])) if raw["face_found"][r].any() else float("nan")
        motion = head_motion(res["motion_xy"][g], res["face_width"][g])
        hr_fft, hr_pk = reference_hr(res["ppg"][g], cfg.fs, cfg.hr_min_bpm, cfg.hr_max_bpm)
        gt_diff = abs(hr_fft - hr_pk)

        reasons = []
        if gap_frac > cfg.max_gap_fraction:
            reasons.append("video_gaps")
        if gt_gap_frac > cfg.max_gap_fraction:
            reasons.append("gt_gaps")
        if face_missing > cfg.max_face_missing:
            reasons.append("face_missing")
        if not (clipped <= cfg.max_clipped):          # also catches NaN
            reasons.append("clipped")
        if not (gt_diff <= cfg.max_gt_disagreement_bpm):
            reasons.append("gt_unreliable")

        rows.append({
            "subject": subject, "window": k,
            "t_start": round(float(s), 3), "t_end": round(float(e), 3),
            "hr_gt": round(hr_fft, 2), "hr_gt_peaks": round(hr_pk, 2),
            "gap_frac": round(gap_frac, 4), "face_missing": round(face_missing, 4),
            "clipped": round(clipped, 4), "motion": round(motion, 5),
            "accepted": not reasons, "reject_reasons": ";".join(reasons),
        })
    return rows
