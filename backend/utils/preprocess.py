"""
Run this before every training run to (re)split raw data into train/val/test.
Automatically clears old processed data first to avoid duplicates.

Usage: python utils/preprocess.py   (from backend/ folder)
"""

import os
import sys
import shutil
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASSES, TRAIN_RATIO, VAL_RATIO
)


def clear_processed():
    """Delete and recreate all train/val/test class subdirectories."""
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(split):
            shutil.rmtree(split)
        for cls in CLASSES:
            os.makedirs(os.path.join(split, cls), exist_ok=True)
    print("Cleared and recreated processed/ directories.")


def find_raw_class_folder(cls):
    """
    Locate the raw folder for a class case-insensitively.
    e.g. 'heart' will match 'Heart' or 'HEART' in raw/.
    """
    for name in os.listdir(RAW_DATA_DIR):
        if name.lower() == cls.lower() and os.path.isdir(os.path.join(RAW_DATA_DIR, name)):
            return os.path.join(RAW_DATA_DIR, name)
    return None


def split_dataset():
    """Randomly split each class's images into train/val/test."""
    total_images = 0
    for cls in CLASSES:
        src = find_raw_class_folder(cls)
        if src is None:
            print(f"  WARNING: No raw folder found for class '{cls}' — skipping.")
            continue

        images = [
            f for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
        ]
        random.seed(42)
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR:   images[n_train : n_train + n_val],
            TEST_DIR:  images[n_train + n_val:],
        }

        for dest_dir, files in splits.items():
            for fname in files:
                src_path = os.path.join(src, fname)
                dst_path = os.path.join(dest_dir, cls, fname)
                shutil.copy2(src_path, dst_path)

        n_test = n - n_train - n_val
        print(f"  {cls:10s}: {n:4d} total → {n_train} train | {n_val} val | {n_test} test")
        total_images += n

    print(f"\nDone. {total_images} images split across train/val/test.")


if __name__ == "__main__":
    print("=" * 50)
    print("Step 1: Clearing old processed data…")
    clear_processed()

    print("\nStep 2: Splitting dataset…")
    split_dataset()
    print("=" * 50)
    print("Ready to train. Run: python model/train.py")