import os
import subprocess

from rppg import Config, demo, pipeline, roi
from test_pipeline_synthetic import FakeLandmarker, make_dataset


def test_demo_video_is_written(tmp_path, monkeypatch):
    zpath = make_dataset(tmp_path)
    cfg = Config(zip_path=str(zpath), cache_dir=str(tmp_path / "cache"),
                 tmp_dir=str(tmp_path / "tmp"), model_path=str(tmp_path / "model.task"))
    monkeypatch.setattr(roi, "ensure_model", lambda *a, **k: None)
    monkeypatch.setattr(roi, "create_landmarker", lambda *a, **k: FakeLandmarker())
    pipeline.build_cache(cfg)
    out = demo.make_demo_video(cfg, "subject1", start_s=2.0, duration_s=6.0, panel=120)
    assert os.path.getsize(out) > 1000
    n = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", out],
                       capture_output=True, text=True).stdout.strip()
    assert int(n) > 150
