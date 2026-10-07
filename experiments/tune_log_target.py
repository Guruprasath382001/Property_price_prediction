# experiments/tune_log_target.py

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import (
    DATASET_PATH,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    ALL_FEATURES,
)

from src.preprocessing import create_preprocessor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "experiments_results"
    / "log_target_results.json"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATASET_PATH)

    print("=" * 70)
    print("LOG-TARGET EXPERIMENT")
    print("=" * 70)

    print(
        f"Dataset path : {DATASET_PATH}"
    )

    print(
        f"Dataset shape: {df.shape}"
    )

    print()

    return df


# ============================================================
# CREATE LOG-TARGET MODEL
# ============================================================

def create_model():

    # --------------------------------------------------------
    # Normal preprocessing
    # --------------------------------------------------------

    preprocessor = create_preprocessor()

    # --------------------------------------------------------
    # Ridge regression
    # --------------------------------------------------------

    ridge = Ridge()

    pipeline = Pipeline([
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            ridge,
        ),
    ])

    # --------------------------------------------------------
    # Transform target:
    #
    # price
    #   ↓
    # log1p(price)
    #
    # prediction:
    #
    # log-price
    #   ↓
    # expm1()
    #   ↓
    # price
    # --------------------------------------------------------

    model = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
    )

    return model


# ============================================================
# MAIN
# ============================================================

def tune_log_target():

    df = load_data()

    # --------------------------------------------------------
    # Features / target
    # --------------------------------------------------------

    X = df[ALL_FEATURES]

    y = df[TARGET]

    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Development samples : "
        f"{len(X_train):,}"
    )

    print(
        f"Test samples        : "
        f"{len(X_test):,}"
    )

    print()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # Hyperparameter tuning
    #
    # Notice the parameter name:
    #
    # regressor__model__alpha
    #
    # because:
    #
    # TransformedTargetRegressor
    #       ↓
    # Pipeline
    #       ↓
    # model
    # --------------------------------------------------------

    param_grid = {
        "regressor__model__alpha": [
            0.01,
            0.1,
            1,
            10,
            100,
        ]
    }

    cv = KFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        return_train_score=False,
    )

    print("=" * 70)
    print("TUNING LOG-TARGET RIDGE")
    print("=" * 70)

    grid.fit(
        X_train,
        y_train,
    )

    # --------------------------------------------------------
    # Best CV result
    # --------------------------------------------------------

    best_model = grid.best_estimator_

    cv_rmse = -grid.best_score_

    print()
    print(
        f"Best Parameters : "
        f"{grid.best_params_}"
    )

    print(
        f"Best CV RMSE    : "
        f"{cv_rmse:,.4f}"
    )

    # --------------------------------------------------------
    # Test prediction
    #
    # predict() automatically applies expm1()
    # because of TransformedTargetRegressor.
    # --------------------------------------------------------

    predictions = best_model.predict(
        X_test
    )

    # Safety check
    predictions = np.maximum(
        predictions,
        0,
    )

    # --------------------------------------------------------
    # Test metrics
    # --------------------------------------------------------

    mse = mean_squared_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    r2 = r2_score(
        y_test,
        predictions,
    )

    print()
    print("=" * 70)
    print("LOG-TARGET TEST RESULTS")
    print("=" * 70)

    print(
        f"MSE  : {mse:,.4f}"
    )

    print(
        f"RMSE : {rmse:,.4f}"
    )

    print(
        f"MAE  : {mae:,.4f}"
    )

    print(
        f"R²   : {r2:,.4f}"
    )

    # --------------------------------------------------------
    # Compare with current raw-target model
    # --------------------------------------------------------

    raw_model = {
        "model": "Ridge Regression",
        "target": "price",
        "cv_rmse": 174865.8570636789,
        "test_rmse": 166481.17780377754,
        "test_mae": 105153.73251435775,
        "test_r2": 0.7774843230447084,
    }

    log_model = {
        "model": "Ridge Regression",
        "target": "log1p(price)",
        "best_parameters": grid.best_params_,
        "cv_rmse": cv_rmse,
        "test_mse": mse,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
    }

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RAW TARGET vs LOG TARGET")
    print("=" * 70)

    print()
    print(
        f"{'Metric':<15}"
        f"{'Raw Price':>20}"
        f"{'Log Price':>20}"
    )

    print("-" * 55)

    print(
        f"{'CV RMSE':<15}"
        f"{raw_model['cv_rmse']:>20,.2f}"
        f"{cv_rmse:>20,.2f}"
    )

    print(
        f"{'Test RMSE':<15}"
        f"{raw_model['test_rmse']:>20,.2f}"
        f"{rmse:>20,.2f}"
    )

    print(
        f"{'Test MAE':<15}"
        f"{raw_model['test_mae']:>20,.2f}"
        f"{mae:>20,.2f}"
    )

    print(
        f"{'Test R²':<15}"
        f"{raw_model['test_r2']:>20,.4f}"
        f"{r2:>20,.4f}"
    )

    # --------------------------------------------------------
    # Select winner based on TEST RMSE
    #
    # NOTE:
    # This is for experiment comparison only.
    # We are not changing the production model yet.
    # --------------------------------------------------------

    if rmse < raw_model["test_rmse"]:

        winner = "LOG_TARGET"

    else:

        winner = "RAW_TARGET"

    print()
    print(
        f"Experiment winner: {winner}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "raw_target_model": raw_model,
        "log_target_model": log_model,
        "winner": winner,
    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
        )

    print()
    print(
        f"Results saved to:\n"
        f"{RESULTS_PATH}"
    )
