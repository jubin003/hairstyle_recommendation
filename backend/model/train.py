"""
train.py — Two-phase transfer learning with proper Phase 2 initialization.

Key fixes:
  - Phase 1 best weights saved to a separate checkpoint file
  - Phase 2 loads Phase 1 best weights before fine-tuning
  - Phase 2 ModelCheckpoint starts fresh without overwriting Phase 1 best
  - Fewer unfrozen layers (10) with very low LR (5e-5) to prevent Phase 2 overfitting
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import (
    TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE,
    NUM_EPOCHS, MODEL_PATH, MODEL_SAVE_DIR, CLASS_NAMES_PATH
)
from model.cnn_model import build_model
from utils.helpers import plot_training_curves, print_training_summary

PHASE1_PATH = os.path.join(MODEL_SAVE_DIR, "phase1_best.keras")


def get_datasets():
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=True, seed=42, label_mode="int"
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, label_mode="int"
    )

    class_names = train_ds.class_names
    print(f"Classes      : {class_names}")
    print(f"Train batches: {len(train_ds)} | Val batches: {len(val_ds)}")

    # MobileNetV2 preprocessing: pixels to [-1, 1]
    def preprocess(x, y):
        x = tf.cast(x, tf.float32)
        x = keras.applications.mobilenet_v2.preprocess_input(x)
        return x, y

    # Mild augmentation only — face shape is geometric, heavy distortion destroys signal
    def augment(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.1)
        x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
        x = tf.clip_by_value(x, -1.0, 1.0)
        return x, y

    train_ds = (train_ds
                .map(preprocess, num_parallel_calls=AUTOTUNE)
                .map(augment,    num_parallel_calls=AUTOTUNE)
                .cache()
                .shuffle(1000)
                .prefetch(AUTOTUNE))

    val_ds = (val_ds
              .map(preprocess, num_parallel_calls=AUTOTUNE)
              .cache()
              .prefetch(AUTOTUNE))

    return train_ds, val_ds, class_names


def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    train_ds, val_ds, class_names = get_datasets()

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved: {class_names}")

    model, base = build_model()
    model.summary()

    # ── Phase 1: Head only, backbone fully frozen ─────────────────────────────
    print("\n" + "="*55)
    print("PHASE 1 — Head-only training  (backbone frozen, LR=1e-3)")
    print("="*55)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[
            # Save Phase 1 best to its OWN file
            ModelCheckpoint(
                filepath=PHASE1_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=7,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ],
        verbose=1
    )

    phase1_best_val = max(history1.history["val_accuracy"])
    print(f"\nPhase 1 best val accuracy: {phase1_best_val*100:.2f}%")

    # ── Load Phase 1 best weights before fine-tuning ──────────────────────────
    print(f"\nLoading Phase 1 best weights from {PHASE1_PATH}")
    model.load_weights(PHASE1_PATH)

    # ── Phase 2: Unfreeze only top 10 layers — conservative fine-tuning ───────
    print("\n" + "="*55)
    print("PHASE 2 — Fine-tuning top 10 MobileNetV2 layers  (LR=5e-5)")
    print("="*55)

    base.trainable = True
    for layer in base.layers[:-10]:
        layer.trainable = False

    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"Unfrozen backbone layers: {trainable_count}")

    # Very low LR — minimizes risk of destroying pretrained features
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[
            # Final model saved here
            ModelCheckpoint(
                filepath=MODEL_PATH,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor="val_accuracy",
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ],
        verbose=1
    )

    phase2_best_val = max(history2.history["val_accuracy"])

    # If Phase 2 didn't beat Phase 1, keep Phase 1 weights as final model
    if phase2_best_val < phase1_best_val:
        print(f"\nPhase 2 ({phase2_best_val*100:.2f}%) did not beat Phase 1 ({phase1_best_val*100:.2f}%)")
        print("Saving Phase 1 best weights as final model.")
        import shutil
        shutil.copy(PHASE1_PATH, MODEL_PATH)
    else:
        print(f"\nPhase 2 improved to {phase2_best_val*100:.2f}% — final model saved.")

    print_training_summary(history2)

    class _History:
        def __init__(self, h): self.history = h

    combined = _History({
        "accuracy":     history1.history["accuracy"]     + history2.history["accuracy"],
        "val_accuracy": history1.history["val_accuracy"] + history2.history["val_accuracy"],
        "loss":         history1.history["loss"]         + history2.history["loss"],
        "val_loss":     history1.history["val_loss"]     + history2.history["val_loss"],
    })

    plot_training_curves(
        combined,
        save_path=os.path.join(MODEL_SAVE_DIR, "training_curves.png")
    )

    print(f"\nFinal model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()