from pathlib import Path

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"

MODEL_DIR = BASE_DIR / "models"

EXP_DIR = BASE_DIR / "experiments_results"

DATASET_PATH = DATA_DIR / "cleaned_data.csv"

MODEL_PATH = MODEL_DIR / "house_price_pipeline.pkl"

METADATA_PATH = MODEL_DIR / "metadata.json"


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

CV_FOLDS = 5


# ============================================================
# TARGET
# ============================================================

TARGET = "price"


# ============================================================
# FEATURES
# ============================================================

NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    # "yr_renovated"
]


CATEGORICAL_FEATURES = [
    "city",
    "statezip",
]


ALL_FEATURES = (
    NUMERIC_FEATURES +
    CATEGORICAL_FEATURES
)
