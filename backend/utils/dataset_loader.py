import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import keras
from keras import layers
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE


def get_datasets():
    AUTOTUNE  = tf.data.AUTOTUNE
    normalize = layers.Rescaling(1.0 / 255)

    train_ds = keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=True, seed=42, label_mode='int'
    )
    val_ds = keras.utils.image_dataset_from_directory(
        VAL_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, label_mode='int'
    )
    test_ds = keras.utils.image_dataset_from_directory(
        TEST_DIR, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
        shuffle=False, label_mode='int'
    )

    class_names = train_ds.class_names
    print(f"Classes : {class_names}")
    print(f"Train: {len(train_ds)} batches | Val: {len(val_ds)} batches | Test: {len(test_ds)} batches")

    train_ds = train_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)\
                       .cache().shuffle(1000).prefetch(AUTOTUNE)
    val_ds   = val_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)\
                     .cache().prefetch(AUTOTUNE)
    test_ds  = test_ds.map(lambda x, y: (normalize(x), y), num_parallel_calls=AUTOTUNE)\
                      .prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


if __name__ == "__main__":
    get_datasets()
    print("Dataset loading OK.")