"""Central configuration: every path and tunable parameter lives here.

Changing a value in one place changes it for the whole pipeline, and the
config used for a run can be printed into the report.
"""
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # --- Paths -------------------------------------------------------------
    zip_path: str = "/content/drive/MyDrive/Datasets_rPPG/ubfc_rppg.zip"
    cache_dir: str = "/content/drive/MyDrive/Datasets_rPPG/cache"   # persistent (Drive)
    tmp_dir: str = "/content/ubfc_tmp"                              # fast local disk, wiped per subject
    model_path: str = "/content/face_landmarker.task"
    model_url: str = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                      "face_landmarker/float16/1/face_landmarker.task")

    # --- Resampling --------------------------------------------------------
    fs: float = 30.0          # uniform sampling rate after resampling [Hz]
    max_gap: float = 0.1      # gaps longer than this [s] are marked as "interpolated, not measured"

    # --- Heart-rate band ---------------------------------------------------
    hr_min_bpm: float = 42.0  # 0.7 Hz
    hr_max_bpm: float = 180.0 # 3.0 Hz

    # --- ROI extraction ----------------------------------------------------
    roi_shrink: float = 0.85  # shrink ROI polygons toward their centre to stay away from edges
    clip_low: int = 5         # pixel value <= clip_low counts as under-exposed
    clip_high: int = 250      # pixel value >= clip_high counts as over-exposed

    # --- Windowing and quality rules (decided BEFORE looking at algorithm results) ---
    window_s: float = 8.0
    step_s: float = 1.0
    max_gap_fraction: float = 0.10      # max share of window that is interpolated across gaps
    max_face_missing: float = 0.05      # max share of frames without a detected face
    max_clipped: float = 0.05           # max share of ROI pixels that are clipped
    max_gt_disagreement_bpm: float = 5.0  # FFT vs peak-count HR on the reference PPG

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
