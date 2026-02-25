"""
Run this ONCE before training to split raw data into train/val/test folders.
Usage: python utils/preprocess.py   (from backend/ folder)
   OR: python preprocess.py         (from utils/ folder)
"""

import os
import sys
import shutil
import random

# Add backend/ to path so config.py can be found regardless of where you run from
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    RAW_DATA_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR,
    CLASSES, TRAIN_RATIO, VAL_RATIO
)


def create_dirs():
    """Create train/val/test subdirectories for each class."""
    for split in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        for cls in CLASSES:
            os.makedirs(os.path.join(split, cls), exist_ok=True)
    print("Directories created.")


def split_dataset():
    """Randomly split each class's images into train/val/test."""
    for cls in CLASSES:
        src = os.path.join(RAW_DATA_DIR, cls)
        images = [f for f in os.listdir(src) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        splits = {
            TRAIN_DIR: images[:n_train],
            VAL_DIR: images[n_train:n_train + n_val],
            TEST_DIR: images[n_train + n_val:]
        }

        for dest_dir, files in splits.items():
            for fname in files:
                src_path = os.path.join(src, fname)
                dst_path = os.path.join(dest_dir, cls, fname)
                shutil.copy2(src_path, dst_path)

        print(f"  {cls}: {n_train} train | {n_val} val | {n - n_train - n_val} test")

    print("Dataset split complete.")


if __name__ == "__main__":
    print("Splitting dataset...")
    create_dirs()
    split_dataset()