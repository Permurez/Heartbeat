"""One-time migration: rename fields inside already-built cache/subjectN.npz files
to match the renamed keys in rppg/pipeline.py (t -> timestamps_s, hr_oximeter ->
oximeter_heart_rate_bpm, clipped -> clipped_fraction). Does not touch the videos.

Run once after pulling the variable-naming update, if you already built a cache:

    python scripts/migrate_cache_keys.py                    # uses Config().cache_dir
    python scripts/migrate_cache_keys.py /path/to/cache

Safe to run twice: a file whose keys are already renamed is left untouched.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rppg.config import Config

RENAMED_KEYS = {"t": "timestamps_s", "hr_oximeter": "oximeter_heart_rate_bpm", "clipped": "clipped_fraction"}


def migrate_file(path):
    with np.load(path, allow_pickle=False) as cache_file:
        payload = {k: cache_file[k] for k in cache_file.files}
    if not (RENAMED_KEYS.keys() & payload.keys()):
        return False       # already migrated (or not ours), nothing to do
    for old_key, new_key in RENAMED_KEYS.items():
        if old_key in payload:
            payload[new_key] = payload.pop(old_key)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)
    return True


def main():
    cache_dir = sys.argv[1] if len(sys.argv) > 1 else Config().cache_dir
    if not os.path.isdir(cache_dir):
        sys.exit(f"Cache directory not found: {cache_dir}")

    names = sorted(f for f in os.listdir(cache_dir) if f.startswith("subject") and f.endswith(".npz"))
    if not names:
        print(f"No subjectN.npz files in {cache_dir}, nothing to migrate.")
        return
    migrated = 0
    for name in names:
        path = os.path.join(cache_dir, name)
        if migrate_file(path):
            migrated += 1
            print(f"{name}: migrated")
        else:
            print(f"{name}: already up to date, skipped")
    print(f"Done: {migrated}/{len(names)} files migrated.")


if __name__ == "__main__":
    main()
