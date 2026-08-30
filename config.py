from pathlib import Path

# ============================================================
# AGROFRESH CONFIGURATION
# ============================================================

# Project directory
PROJECT_DIR = Path(__file__).resolve().parent

# Dataset location
DATASET_DIR = Path(
    r"C:\Users\ravip\Desktop\DataSet\New folder (6)\dataset"
)

# Dataset folders
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Project output folders
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

# Image settings
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Training settings
EPOCHS = 10
LEARNING_RATE = 0.0001

# Create output directories
MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)