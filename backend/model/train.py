import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tensorflow as tf
import keras
from keras import layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import (
    TRAIN_DIR, VAL_DIR, IMAGE_SIZE, BATCH_SIZE,
    NUM_EPOCHS, MODEL_PATH, MODEL_SAVE_DIR, CLASS_NAMES_PATH
)
from model.cnn_model import build_model
from utils.helpers import plot_training_curves, print_training_summary


def get_datasets():
    AUTOTUNE  = tf.data.AUTOTUNE

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=True, seed=42, label_mode='int'
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, label_mode='int'
    )

    class_names = train_ds.class_names
    print(f"Classes : {class_names}")
    print(f"Train batches: {len(train_ds)} | Val batches: {len(val_ds)}")

    # Normalize FIRST, then augment — order matters
    def normalize(x, y):
        return tf.cast(x, tf.float32) / 255.0, y

    def augment(x, y):
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.1)
        x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
        return x, y

    # Apply to train: normalize then mild augment
    train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(augment,    num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)

    # Val: normalize only, no augmentation
    val_ds = val_ds.map(normalize, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names


def main():
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    train_ds, val_ds, class_names = get_datasets()

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"Class names saved: {class_names}")

    model = build_model()
    model.summary()

    callbacks = [
        ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]

    print(f"\nStarting training — {NUM_EPOCHS} epochs max...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=NUM_EPOCHS,
        callbacks=callbacks
    )

    print_training_summary(history)
    plot_training_curves(history, save_path=os.path.join(MODEL_SAVE_DIR, "training_curves.png"))


if __name__ == "__main__":
    main()