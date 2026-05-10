import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping

from config import (
    TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE,
    NUM_EPOCHS, MODEL_PATH, MODEL_SAVE_DIR, CLASS_NAMES_PATH, LEARNING_RATE
)
from model.cnn_model import build_model
from utils.helpers import plot_training_curves, print_training_summary


# ── Augmentation helpers ───────────────────────────────────────────────────
def _add_gaussian_noise(x, stddev=0.02):
    """Additive Gaussian noise — improves robustness to photo compression."""
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=stddev)
    return tf.clip_by_value(x + noise, 0.0, 1.0)


def _random_grayscale(x, prob=0.15):
    """Randomly convert to grayscale — stops model relying on skin/hair colour."""
    if tf.random.uniform(()) < prob:
        gray = tf.image.rgb_to_grayscale(x)
        x = tf.repeat(gray, 3, axis=-1)
    return x


def _augment(x, y):
    """Rich augmentation pipeline applied only to training data."""
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_brightness(x, max_delta=0.15)
    x = tf.image.random_contrast(x, lower=0.85, upper=1.15)
    x = tf.image.random_hue(x, max_delta=0.05)
    x = tf.image.random_saturation(x, lower=0.8, upper=1.2)
    x = _add_gaussian_noise(x, stddev=0.02)
    x = _random_grayscale(x, prob=0.15)
    x = tf.clip_by_value(x, 0.0, 1.0)
    return x, y


def _normalize(x, y):
    return tf.cast(x, tf.float32) / 255.0, y


def _to_categorical(x, y):
    """Convert integer labels to one-hot vectors (required for label smoothing)."""
    return x, tf.one_hot(tf.cast(y, tf.int32), depth=len(class_names_global))


# Module-level placeholder filled in get_datasets()
class_names_global = []


def get_datasets():
    global class_names_global
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_raw = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=True, seed=42, label_mode='int'
    )
    val_ds_raw = keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, label_mode='int'
    )

    class_names_global = train_ds_raw.class_names
    print(f"Classes : {class_names_global}")
    print(f"Train batches: {len(train_ds_raw)} | Val batches: {len(val_ds_raw)}")

    # ── Compute class weights to counter the square-class imbalance ──
    counts = {}
    for _, labels in train_ds_raw.unbatch():
        lbl = int(labels.numpy())
        counts[lbl] = counts.get(lbl, 0) + 1
    total = sum(counts.values())
    n_classes = len(class_names_global)
    class_weight = {
        cls: total / (n_classes * cnt)
        for cls, cnt in counts.items()
    }
    print(f"Class weights: { {class_names_global[k]: round(v, 3) for k, v in class_weight.items()} }")

    # ── Build tf.data pipelines ──────────────────────────────────────
    # Training: normalize → one-hot → augment → cache → shuffle → prefetch
    train_ds = (train_ds_raw
                .map(_normalize,      num_parallel_calls=AUTOTUNE)
                .map(_to_categorical, num_parallel_calls=AUTOTUNE)
                .map(_augment,        num_parallel_calls=AUTOTUNE)
                .cache()
                .shuffle(1000)
                .prefetch(AUTOTUNE))

    # Validation: normalize → one-hot only
    val_ds = (val_ds_raw
              .map(_normalize,      num_parallel_calls=AUTOTUNE)
              .map(_to_categorical, num_parallel_calls=AUTOTUNE)
              .cache()
              .prefetch(AUTOTUNE))

    return train_ds, val_ds, class_names_global, class_weight


def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    train_ds, val_ds, class_names, class_weight = get_datasets()

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved: {class_names}")

    model = build_model()
    model.summary()

    # ── Cosine Decay LR schedule ─────────────────────────────────────
    # Decays smoothly from LEARNING_RATE to ~0 over NUM_EPOCHS.
    # Restarts twice (3 cycles total) to escape local minima.
    steps_per_epoch = len(train_ds)
    cosine_decay = keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate=LEARNING_RATE,
        first_decay_steps=steps_per_epoch * (NUM_EPOCHS // 3),
        t_mul=1.0,          # equal-length cycles
        m_mul=0.9,          # each restart starts at 90% of previous peak LR
        alpha=1e-6           # minimum LR floor
    )
    model.optimizer.learning_rate = cosine_decay

    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,           # more patience — cosine restarts can cause temporary dips
            restore_best_weights=True,
            verbose=1
        ),
    ]

    print(f"\nStarting training — {NUM_EPOCHS} epochs max...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight   # compensates for square-class imbalance
    )

    print_training_summary(history)
    plot_training_curves(history, save_path=os.path.join(MODEL_SAVE_DIR, "training_curves.png"))


if __name__ == "__main__":
    main()