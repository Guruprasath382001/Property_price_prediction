import json
import platform
from datetime import datetime

import joblib
import pandas as pd
import sklearn

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CV_FOLDS,
    DATASET_PATH,
    METADATA_PATH,
    MODEL_DIR,
    MODEL_PATH,
    EXP_DIR,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE
)

from src.preprocessing import create_preprocessor
from src.utils import create_model


def Train_model():

    # ========================================================
    # 1. CREATE MODEL DIRECTORY
    # ========================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_path = (
        EXP_DIR /
        "model_tuning_results.json"
    )

    if not selection_path.exists():

        raise FileNotFoundError(

            "model_tuning_results.json not found.\n"
            "Run tune_models.py first."
        )

    with open(
        selection_path,
        "r",
        encoding="utf-8",
    ) as file:

        selection = json.load(
            file
        )

    algorithm_name = selection[
        "selected_model"
    ]

    best_parameters = selection[
        "selected_parameters"
    ]

    selected_cv_rmse = selection[
        "selected_cv_rmse"
    ]

    model = create_model(

        algorithm_name,

        best_parameters,
    )

    # ========================================================
    # 2. LOAD DATASET
    # ========================================================

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
        "Dataset shape:",
        df.shape,
    )

    df = df[df["price"] > 0]


    # ========================================================
    # 3. VALIDATE DATASET
    # ========================================================

    required_columns = (
        ALL_FEATURES +
        [TARGET]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    # ========================================================
    # 4. SEPARATE FEATURES AND TARGET
    # ========================================================

    X = df[
        ALL_FEATURES
    ]

    y = df[
        TARGET
    ]

    # ========================================================
    # 5. TRAIN / TEST SPLIT
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    X_development, X_test, y_development, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
    )

    print(
        "Development samples:",
        len(X_development),
    )

    print(
        "Test samples:",
        len(X_test),
    )

    # ========================================================
    # 6. CREATE MODEL SELECTION PIPELINE
    # ========================================================

    final_pipeline = Pipeline([
        (
            "preprocessor",
            create_preprocessor(),
        ),
        (
            "model",
            model
        ),
    ])



    # ========================================================
    # 8. FINAL MODEL TRAINING
    # ========================================================


    final_pipeline.fit(
        X_development,
        y_development,
    )

    print(
        "Final model trained on",
        len(X_development),
        "development samples.",
    )

    # ========================================================
    # 9. FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    y_pred = final_pipeline.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        y_pred,
    )

    rmse = mse ** 0.5

    mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    r2 = r2_score(
        y_test,
        y_pred,
    )

    print(
        "MSE :",
        mse,
    )

    print(
        "RMSE:",
        rmse,
    )

    print(
        "MAE :",
        mae,
    )

    print(
        "R2  :",
        r2,
    )

    # ========================================================
    # 10. SAVE MODEL
    # ========================================================

    joblib.dump(
        final_pipeline,
        MODEL_PATH,
    )

    print("\nFinal model saved to:")
    print(MODEL_PATH)

    # ========================================================
    # 11. SAVE METADATA
    # ========================================================

    metadata = {
        "model_name": algorithm_name,

        "model_type": "regression",

        "target": TARGET,

        "features": ALL_FEATURES,

        "numeric_features": NUMERIC_FEATURES,

        "categorical_features": CATEGORICAL_FEATURES,

        "hyperparameters": best_parameters,

        "training": {
            "dataset_rows": int(len(df)),
            "development_rows": int(len(X_development)),
            "test_rows": int(len(X_test)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "cross_validation_folds": CV_FOLDS,
        },

        "metrics": {
            "cv_rmse": float(selected_cv_rmse),
            "test_mse": float(mse),
            "test_rmse": float(rmse),
            "test_mae": float(mae),
            "test_r2": float(r2),
        },

        "preprocessing": {
            "numeric_missing_values": "median",
            "numeric_scaling": "StandardScaler",
            "categorical_missing_values": "most_frequent",
            "categorical_encoding": "OneHotEncoder",
            "unknown_categories": "ignore",
        },

        "environment": {
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
        },

        "created_at": datetime.now().isoformat(),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
        )

    print("\nMetadata saved to:")
    print(METADATA_PATH)

    print("\n" + "=" * 70)
    print("FINAL PRODUCTION TRAINING COMPLETE")
    print("=" * 70)

    return metadata