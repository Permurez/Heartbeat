"""Access to the UBFC-rPPG DATASET_2 stored as a single zip (subjectN/vid.avi + ground_truth.txt).

Videos are ~1.5-1.9 GB each, so they are extracted one at a time to local disk
and deleted right after processing.

Reference: S. Bobbia et al., "Unsupervised skin tissue segmentation for remote
photoplethysmography", Pattern Recognition Letters, 2019.
"""
import os
import shutil
import zipfile
from contextlib import contextmanager

import numpy as np


def list_subjects(zip_path):
    """Subject folder names sorted numerically (subject9 before subject10)."""
    with zipfile.ZipFile(zip_path) as zf:
        names = {n.split("/")[0] for n in zf.namelist() if n.startswith("subject")}
    return sorted(names, key=lambda s: int(s[len("subject"):]))


def load_ground_truth(zip_path, subject):
    """Return (ppg, hr, t): reference waveform, oximeter HR [BPM], timestamps [s]."""
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read(f"{subject}/ground_truth.txt").decode()
    rows = [np.array(line.split(), dtype=float) for line in text.strip().splitlines()]
    if len(rows) != 3:
        raise ValueError(f"{subject}: expected 3 rows in ground_truth.txt, got {len(rows)}")
    ppg, hr, t = rows
    return ppg, hr, t


@contextmanager
def extracted_video(zip_path, subject, tmp_dir):
    """Extract subjectN/vid.avi to tmp_dir for the duration of a `with` block, then delete it."""
    target_dir = os.path.join(tmp_dir, subject)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(f"{subject}/vid.avi", tmp_dir)
        yield os.path.join(target_dir, "vid.avi")
    finally:
        shutil.rmtree(target_dir, ignore_errors=True)
