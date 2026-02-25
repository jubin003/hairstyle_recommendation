import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR       = os.path.join(BASE_DIR, "dataset", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed")
TRAIN_DIR          = os.path.join(PROCESSED_DATA_DIR, "train")
VAL_DIR            = os.path.join(PROCESSED_DATA_DIR, "val")
TEST_DIR           = os.path.join(PROCESSED_DATA_DIR, "test")

MODEL_SAVE_DIR   = os.path.join(BASE_DIR, "model", "saved_model")
MODEL_PATH       = os.path.join(MODEL_SAVE_DIR, "face_cnn.keras")
CLASS_NAMES_PATH = os.path.join(MODEL_SAVE_DIR, "class_names.json")

IMAGE_SIZE    = (224, 224)
TRAIN_RATIO   = 0.7
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15

CLASSES       = ["heart", "oblong", "oval", "round", "square"]
NUM_CLASSES   = len(CLASSES)

BATCH_SIZE    = 32
NUM_EPOCHS    = 50          # total budget across both phases
LEARNING_RATE = 1e-3        # Phase 1 head LR (fine-tune uses 1e-4 automatically)