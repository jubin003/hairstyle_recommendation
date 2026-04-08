"""
cnn_model.py — CNN built entirely from scratch.
No pretrained weights, no external backbone.
All weights randomly initialized and trained solely on the face shape dataset.

Architecture:
    4x ConvBlock (Conv2D → BatchNorm → ReLU → MaxPooling)
    GlobalAveragePooling2D  (replaces Flatten — prevents overfitting)
    Dense(256) → BN → Dropout(0.5)
    Dense(128) → Dropout(0.4)
    Dense(NUM_CLASSES, softmax)
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
from keras import layers, models, regularizers
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE

L2 = 1e-5  # mild L2 regularization on Dense layers


def build_model():
    """
    Returns:
        model : compiled Keras Sequential model trained from scratch
    """

    model = models.Sequential([
        layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),

        # ── Block 1: 224x224 → 112x112 ──────────────────────────
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        # ── Block 2: 112x112 → 56x56 ────────────────────────────
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        # ── Block 3: 56x56 → 28x28 ──────────────────────────────
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        # ── Block 4: 28x28 → 14x14 ──────────────────────────────
        layers.Conv2D(256, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D(2, 2),

        # ── Classifier head ─────────────────────────────────────
        # GlobalAveragePooling: 14x14x256 → 256  (no pretrained features)
        layers.GlobalAveragePooling2D(),

        layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(L2)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(L2)),
        layers.Dropout(0.4),

        layers.Dense(NUM_CLASSES, activation="softmax")

    ], name="FaceShapeCNN_FromScratch")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"\nTotal params: {model.count_params():,}")
    print("No pretrained weights — trained from scratch on face shape dataset only.")