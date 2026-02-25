"""
Scans all images in processed/train, val, test folders and removes corrupted ones.
Run this ONCE before training:
    python utils/clean_dataset.py   (from backend/ folder)
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = False  # We WANT it to fail so we can catch and delete

from config import TRAIN_DIR, VAL_DIR, TEST_DIR, RAW_DATA_DIR

DIRS_TO_CLEAN = [TRAIN_DIR, VAL_DIR, TEST_DIR, RAW_DATA_DIR]


def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()          # checks file integrity
        # verify() resets the file pointer — reopen to actually decode
        with Image.open(path) as img:
            img.convert("RGB")    # ensures full decode works
        return True
    except Exception:
        return False


def clean_directory(root_dir):
    removed = 0
    checked = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            fpath = os.path.join(dirpath, fname)
            checked += 1
            if not is_valid_image(fpath):
                print(f"  Removing corrupt: {fpath}")
                os.remove(fpath)
                removed += 1

    return checked, removed


if __name__ == "__main__":
    total_checked = 0
    total_removed = 0

    for d in DIRS_TO_CLEAN:
        if not os.path.exists(d):
            continue
        print(f"\nScanning: {d}")
        checked, removed = clean_directory(d)
        print(f"  Checked: {checked} | Removed: {removed}")
        total_checked += checked
        total_removed += removed

    print(f"\n{'='*40}")
    print(f"Total checked : {total_checked}")
    print(f"Total removed : {total_removed}")
    print(f"{'='*40}")
    if total_removed == 0:
        print("All images are clean.")
    else:
        print("Corrupted images removed. You can now run train.py.")