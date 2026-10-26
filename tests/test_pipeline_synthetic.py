"""End-to-end test of the data pipeline on a synthetic UBFC-like zip.

MediaPipe is replaced by a fake landmarker returning fixed landmarks, so the
test runs offline. The synthetic "skin" region pulses at a known heart rate.
"""
import os
import zipfile
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

from rppg import Config, pipeline, roi

HR_BPM = 84.0
FPS = 30.0
N_FRAMES = 450                # 15 s
H, W = 240, 320


class FakeLandmarker:
    """Returns the same 478 landmarks every frame, except for frames listed in `miss`."""

    def __init__(self, miss=()):
        rng = np.random.default_rng(1)
        pts = np.column_stack([rng.uniform(0.35, 0.65, 478), rng.uniform(0.3, 0.7, 478)])
        self.landmarks = [SimpleNamespace(x=float(x), y=float(y)) for x, y in pts]
        self.miss, self.calls, self.last_ts = set(miss), 0, -1

    def detect_for_video(self, image, timestamp_ms):
        assert timestamp_ms > self.last_ts, "timestamps must increase"
        self.last_ts = timestamp_ms
        i, self.calls = self.calls, self.calls + 1
        return SimpleNamespace(face_landmarks=[] if i in self.miss else [self.landmarks])

    def close(self):
        pass


def make_dataset(tmp_path):
    rng = np.random.default_rng(0)
    t = np.cumsum(np.r_[0, rng.uniform(0.028, 0.039, N_FRAMES - 1)])
    t[200] = t[199]                                    # duplicated log entry
    t[300:] += 0.25                                    # dropped frames -> gap
    pulse = np.sin(2 * np.pi * HR_BPM / 60 * t)

    sdir = tmp_path / "subject1"
    sdir.mkdir()
    vw = cv2.VideoWriter(str(sdir / "vid.avi"), cv2.VideoWriter_fourcc(*"FFV1"), FPS, (W, H))
    for p in pulse:
        frame = np.full((H, W, 3), (90, 120, 160), np.uint8)       # BGR background
        skin = np.clip(np.array([110, 140 + 4 * p, 190]), 0, 255).astype(np.uint8)
        frame[60:180, 100:220] = skin
        vw.write(frame)
    vw.release()
    np.savetxt(sdir / "ground_truth.txt", np.vstack([pulse, np.full_like(t, HR_BPM), t]))

    zpath = tmp_path / "ubfc.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.write(sdir / "vid.avi", "subject1/vid.avi")
        zf.write(sdir / "ground_truth.txt", "subject1/ground_truth.txt")
    return zpath


def test_pipeline_end_to_end(tmp_path, monkeypatch):
    zpath = make_dataset(tmp_path)
    cfg = Config(zip_path=str(zpath), cache_dir=str(tmp_path / "cache"),
                 tmp_dir=str(tmp_path / "tmp"), model_path=str(tmp_path / "model.task"))
    monkeypatch.setattr(roi, "ensure_model", lambda *a, **k: None)
    monkeypatch.setattr(roi, "create_landmarker", lambda *a, **k: FakeLandmarker(miss=range(100, 103)))

    summary = pipeline.build_cache(cfg)
    assert summary.loc[0, "frames_video"] == N_FRAMES
    assert not os.path.exists(os.path.join(cfg.tmp_dir, "subject1")), "video must be deleted"
    assert os.path.exists(os.path.join(cfg.cache_dir, "debug", "subject1.png"))

    # Second run must skip the cached subject (resumability)
    assert pipeline.build_cache(cfg).empty

    raw = pipeline.load_subject(cfg, "subject1")
    assert raw["mean_raw"].shape == (N_FRAMES, 3, 3)
    assert raw["face_found"].sum() == N_FRAMES - 3

    res = pipeline.resample_subject(raw, cfg)
    assert res["report_gt"]["duplicates_merged"] == 1
    assert res["report_gt"]["gaps_over_max"] == 1

    # Green channel of the ROI must carry the pulse at the right frequency
    from rppg.groundtruth import hr_spectral
    green = res["rgb_raw"][:, 0, 1]
    assert abs(hr_spectral(green, cfg.fs) - HR_BPM) < 1.5

    df = pipeline.build_window_table(cfg)
    assert os.path.exists(os.path.join(cfg.cache_dir, "windows.csv"))
    duration = res["t"][-1] - res["t"][0]
    assert len(df) == int(np.floor((duration - cfg.window_s) / cfg.step_s)) + 1
    print(df.to_string())
    assert (abs(df["hr_gt"] - HR_BPM) < 1.5).all()
    assert df["accepted"].any()
    assert pd.api.types.is_bool_dtype(df["accepted"])


def test_windows_without_face_are_rejected(tmp_path, monkeypatch):
    zpath = make_dataset(tmp_path)
    cfg = Config(zip_path=str(zpath), cache_dir=str(tmp_path / "cache"),
                 tmp_dir=str(tmp_path / "tmp"), model_path=str(tmp_path / "model.task"))
    monkeypatch.setattr(roi, "ensure_model", lambda *a, **k: None)
    monkeypatch.setattr(roi, "create_landmarker", lambda *a, **k: FakeLandmarker(miss=range(0, 90)))
    pipeline.build_cache(cfg)
    df = pipeline.build_window_table(cfg)
    first = df.iloc[0]
    assert not first["accepted"] and "face_missing" in first["reject_reasons"]
    assert df["accepted"].iloc[-1]
