# Heartbeat — heart rate from video (rPPG)

Remote photoplethysmography: estimating heart rate from subtle skin-colour changes in ordinary
video, using several methods (green channel, CHROM, POS, head-motion) and fusing them.
Semester project for *Grafika i GUI* (G2UI), Wrocław University of Science and Technology.

## Status

- [x] Data layer: UBFC-rPPG loading, timestamp cleaning, resampling, ROI extraction, quality-controlled windows
- [ ] rPPG methods (GREEN, CHROM, POS, head motion)
- [ ] Fusion + evaluation (MAE, RMSE, Pearson r, Bland-Altman)
- [ ] Multi-person tracking
- [x] Eulerian Video Magnification (colour + motion) and a 3-panel demo video (`rppg/evm.py`, `rppg/demo.py`)
- [ ] GUI (live preview, per-face BPM, signal plots, EVM view)

## Data pipeline

```
ubfc_rppg.zip (32 GB, 42 subjects)
   │  rppg.pipeline.build_cache()        one video at a time, resumable
   ▼
cache/subjectN.npz                       per-frame ROI colour means (raw + linear), face/motion data, reference PPG
cache/debug/subjectN.png                 frame with ROIs drawn, for visual checking
   │  rppg.pipeline.build_window_table()
   ▼
cache/windows.csv                        8 s windows, reference HR, quality flags, accepted yes/no
```

Design decisions:

- **Timestamps are irregular** (jitter, ~1 duplicated entry per 4 subjects, gaps of up to ~1 s).
  Signals are cleaned and resampled to a uniform 30 Hz grid with a cubic spline; samples
  interpolated across gaps > 100 ms are flagged as not measured.
- **Reference HR is computed from the reference PPG waveform** with the same spectral method
  used for rPPG, and cross-checked with a beat-interval estimate. Windows where the two
  disagree by > 5 BPM are considered unreliable ground truth.
- **Quality rules use input quality only** (gaps, face detection, clipped pixels, reference
  reliability) and are fixed before any algorithm is evaluated. Reports always state coverage.
- **sRGB gamma is undone** (exact IEC 61966-2-1 curve) for the linear-light colour means, which is
  what optical skin models behind CHROM/POS assume. Raw means are kept for comparison.

## Usage (Google Colab)

Open `notebooks/02_build_cache.ipynb` in Colab.

Progress figure: `%run scripts/progress_figure.py`.
Demo video: `rppg.demo.make_demo_video(Config(), 'subject1')`.

## Tests

```
pytest -q
```

The end-to-end test builds a synthetic UBFC-like dataset (pulsing skin patch, jittery timestamps,
duplicated entry, dropped frames, missing face detections) and runs the whole pipeline offline.

## Dataset

UBFC-rPPG, DATASET_2: S. Bobbia, R. Macwan, Y. Benezeth, A. Mansouri, J. Dubois,
*Unsupervised skin tissue segmentation for remote photoplethysmography*,
Pattern Recognition Letters, 2019. The dataset is not redistributed in this repository.

## AI assistance

In line with university guidelines: the code in `rppg/` and `tests/` of the data layer was
drafted with an AI assistant (Claude, Anthropic) and reviewed, run and validated by the author.
Commits containing AI-generated code are marked in their commit messages.
