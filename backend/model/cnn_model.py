"""
cnn_model.py — MobileNetV2 transfer learning.
Reduced L2 from 1e-4 to 1e-5 (was over-penalizing the small head).
Dropout kept at 0.5/0.4 to prevent overfitting from Phase 2.
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
from keras import layers, regularizers
from keras.applications import MobileNetV2
from config import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE

L2 = 1e-5   # reduced from 1e-4 — was suppressing learning in Phase 1


def build_model():
    """
    Returns:
        model : full compiled Keras model (backbone frozen for Phase 1)
        base  : reference to the MobileNetV2 backbone
    """

    base = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )
    base.trainable = False

    inputs  = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation="relu",
                           kernel_regularizer=regularizers.l2(L2))(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(0.5)(x)
    x       = layers.Dense(128, activation="relu",
                           kernel_regularizer=regularizers.l2(L2))(x)
    x       = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="FaceShape_MobileNetV2")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base


if __name__ == "__main__":
    model, base = build_model()
    model.summary()
    print(f"\nBackbone layers   : {len(base.layers)}")
    print(f"Total params      : {model.count_params():,}")