"""
cnn_model.py — CNN built entirely from scratch.
No pretrained weights, no external backbone.
All weights randomly initialized and trained solely on the face shape dataset.

Architecture (ResNet-inspired VGG-style, fully from scratch):
    5x ConvBlock with double Conv2D → BatchNorm → ReLU → MaxPooling
    Blocks 3-5 have a residual 1×1 projection shortcut (skip connection)
    GlobalAveragePooling2D
    Dense(512) → BN → Dropout(0.5)
    Dense(256) → BN → Dropout(0.4)
    Dense(NUM_CLASSES, softmax)

Augmentation is handled entirely in the tf.data pipeline (train.py),
keeping this file clean and inference-safe.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
from keras import layers, regularizers
import tensorflow as tf
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE

L2 = 1e-5  # mild L2 regularization on Dense layers


def _conv_block(x, filters, pool=True):
    """
    Double Conv2D → BN → ReLU block (VGG-style).
    Optionally applies MaxPooling2D at the end.
    Returns the output tensor.
    """
    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if pool:
        x = layers.MaxPooling2D(2, 2)(x)
    return x


def _residual_block(x, filters):
    """
    Double Conv2D block with a 1×1 projection shortcut (skip connection).
    Enables residual learning on deeper blocks without pretrained weights.
    Always applies MaxPooling after the addition.
    """
    shortcut = x

    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # 1×1 projection shortcut to match channel dimensions
    shortcut = layers.Conv2D(filters, (1, 1), padding="same", use_bias=False)(shortcut)
    shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2, 2)(x)
    return x


def build_model():
    """
    Builds and compiles the CNN using the Keras functional API.

    Returns:
        model : compiled Keras Model trained from scratch
    """
    inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    # ── Block 1: 224×224 → 112×112  (plain, shallow features) ───────
    x = _conv_block(inputs, filters=32)

    # ── Block 2: 112×112 → 56×56  (plain) ───────────────────────────
    x = _conv_block(x, filters=64)

    # ── Block 3: 56×56 → 28×28  (residual) ──────────────────────────
    x = _residual_block(x, filters=128)

    # ── Block 4: 28×28 → 14×14  (residual) ──────────────────────────
    x = _residual_block(x, filters=256)

    # ── Block 5: 14×14 → 7×7   (residual) ───────────────────────────
    x = _residual_block(x, filters=512)

    # ── Classifier head ──────────────────────────────────────────────
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(512, activation="relu",
                     kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs,
                        name="FaceShapeCNN_ResidualFromScratch")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        # label_smoothing=0.1 acts as regularization — penalises overconfident predictions
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()
    print(f"\nTotal params: {model.count_params():,}")
    print("No pretrained weights — trained from scratch on face shape dataset only.")