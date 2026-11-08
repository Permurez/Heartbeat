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
    displacement = centroid - np.nanmean(centroid, axis=0)
    return float(np.nanstd(np.linalg.norm(displacement, axis=1)) / np.nanmedian(face_width))


def build_windows(subject, raw, res, cfg):
    """Cut one subject into overlapping windows and evaluate quality rules.

    raw: per-frame cache (dict from the .npz file), res: resampled dict (see pipeline.resample_subject).
    Returns a list of dicts, one per window.
    """
    timestamps_s, raw_timestamps_s = res["timestamps_s"], raw["timestamps_s"]
    rows = []
    window_starts_s = np.arange(timestamps_s[0], timestamps_s[-1] - cfg.window_s + 1e-9, cfg.step_s)
    for window_index, window_start_s in enumerate(window_starts_s):
        window_end_s = window_start_s + cfg.window_s
        in_window_grid_mask = (timestamps_s >= window_start_s) & (timestamps_s < window_end_s)
        in_window_frame_mask = (raw_timestamps_s >= window_start_s) & (raw_timestamps_s < window_end_s)

        gap_fraction = float(1 - res["valid_video"][in_window_grid_mask].mean())
        ground_truth_gap_fraction = float(1 - res["valid_gt"][in_window_grid_mask].mean())
        face_missing_fraction = float(1 - raw["face_found"][in_window_frame_mask].mean())
        clipped_fraction = (float(np.nanmean(raw["clipped_fraction"][in_window_frame_mask]))
                           if raw["face_found"][in_window_frame_mask].any() else float("nan"))
        head_motion_value = head_motion(res["motion_xy"][in_window_grid_mask], res["face_width"][in_window_grid_mask])
        reference_hr_spectral_bpm, reference_hr_peaks_bpm = reference_hr(
            res["ppg"][in_window_grid_mask], cfg.sample_rate_hz, cfg.heart_rate_min_bpm, cfg.heart_rate_max_bpm)
        reference_hr_disagreement_bpm = abs(reference_hr_spectral_bpm - reference_hr_peaks_bpm)

        reasons = []
        if gap_fraction > cfg.max_gap_fraction:
            reasons.append("video_gaps")
        if ground_truth_gap_fraction > cfg.max_gap_fraction:
            reasons.append("gt_gaps")
        if face_missing_fraction > cfg.max_face_missing:
            reasons.append("face_missing")
        if not (clipped_fraction <= cfg.max_clipped):          # also catches NaN
            reasons.append("clipped")
        if not (reference_hr_disagreement_bpm <= cfg.max_gt_disagreement_bpm):
            reasons.append("gt_unreliable")

        rows.append({
            "subject": subject, "window_index": window_index,
            "window_start_s": round(float(window_start_s), 3), "window_end_s": round(float(window_end_s), 3),
            "reference_heart_rate_bpm": round(reference_hr_spectral_bpm, 2),
            "reference_heart_rate_peaks_bpm": round(reference_hr_peaks_bpm, 2),
            "gap_fraction": round(gap_fraction, 4), "face_missing_fraction": round(face_missing_fraction, 4),
            "clipped_fraction": round(clipped_fraction, 4), "head_motion": round(head_motion_value, 5),
            "accepted": not reasons, "reject_reasons": ";".join(reasons),
        })
    return rows
