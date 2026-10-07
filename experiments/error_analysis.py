from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.config import (
    DATASET_PATH,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    EXP_DIR,
)
from src.preprocessing import create_preprocessor

def error_analysis():
    # ============================================================
    # 1. LOAD DATA
    # ============================================================

    df = pd.read_csv(DATASET_PATH)

    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    print(f"\nDataset shape: {df.shape}")


    # ============================================================
    # 2. PREPARE FEATURES AND TARGET
    # ============================================================

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]


    # ============================================================
    # 3. TRAIN / TEST SPLIT
    # ============================================================

    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print(f"Development samples: {len(X_dev)}")
    print(f"Test samples: {len(X_test)}")


    # ============================================================
    # 4. CREATE SELECTED MODEL
    # ============================================================

    preprocessor = create_preprocessor()

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Ridge(alpha=1)),
    ])


    # ============================================================
    # 5. TRAIN MODEL
    # ============================================================

    model.fit(X_dev, y_dev)


    # ============================================================
    # 6. MAKE PREDICTIONS
    # ============================================================

    y_pred = model.predict(X_test)


    # ============================================================
    # 7. OVERALL MODEL METRICS
    # ============================================================

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 70)
    print("OVERALL TEST PERFORMANCE")
    print("=" * 70)

    print(f"MAE  : ${mae:,.2f}")
    print(f"RMSE : ${rmse:,.2f}")
    print(f"R²   : {r2:.4f}")


    # ============================================================
    # 8. CREATE ERROR DATAFRAME
    # ============================================================

    error_df = X_test.copy()

    error_df["actual_price"] = y_test.values
    error_df["predicted_price"] = y_pred

    error_df["signed_error"] = (
        error_df["predicted_price"]
        - error_df["actual_price"]
    )

    error_df["absolute_error"] = (
        error_df["signed_error"].abs()
    )

    error_df["percentage_error"] = (
        error_df["absolute_error"]
        / error_df["actual_price"]
    ) * 100

    error_df["error_direction"] = error_df["signed_error"].apply(
        lambda x: "OVERPREDICTION"
        if x > 0
        else "UNDERPREDICTION"
    )


    # ============================================================
    # 9. TOP 20 LARGEST ERRORS
    # ============================================================

    top_errors = (
        error_df
        .sort_values("absolute_error", ascending=False)
        .head(20)
    )

    print("\n" + "=" * 70)
    print("TOP 20 LARGEST ABSOLUTE ERRORS")
    print("=" * 70)

    columns_to_show = [
        "actual_price",
        "predicted_price",
        "absolute_error",
        "percentage_error",
        "error_direction",
        "city",
        "statezip",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "waterfront",
        "view",
    ]

    print(
        top_errors[columns_to_show]
        .to_string(index=False)
    )


    # ============================================================
    # 10. UNDERPREDICTION / OVERPREDICTION SUMMARY
    # ============================================================

    print("\n" + "=" * 70)
    print("ERROR DIRECTION")
    print("=" * 70)

    direction_summary = (
        error_df["error_direction"]
        .value_counts()
    )

    print(direction_summary.to_string())


    # ============================================================
    # 11. PRICE RANGE ANALYSIS
    # ============================================================

    bins = [
        0,
        300_000,
        500_000,
        1_000_000,
        2_000_000,
        float("inf"),
    ]

    labels = [
        "< $300K",
        "$300K - $500K",
        "$500K - $1M",
        "$1M - $2M",
        "> $2M",
    ]

    error_df["price_range"] = pd.cut(
        error_df["actual_price"],
        bins=bins,
        labels=labels,
        right=False,
    )

    price_range_analysis = (
        error_df
        .groupby("price_range", observed=True)
        .agg(
            samples=("actual_price", "count"),
            mae=("absolute_error", "mean"),
            mean_percentage_error=("percentage_error", "mean"),
            max_error=("absolute_error", "max"),
        )
        .reset_index()
    )

    print("\n" + "=" * 70)
    print("ERROR BY ACTUAL PRICE RANGE")
    print("=" * 70)

    print(
        price_range_analysis.to_string(
            index=False,
            formatters={
                "mae": "${:,.2f}".format,
                "mean_percentage_error": "{:.2f}%".format,
                "max_error": "${:,.2f}".format,
            },
        )
    )


    # ============================================================
    # 12. CITY-LEVEL ERROR ANALYSIS
    # ============================================================

    city_analysis = (
        error_df
        .groupby("city")
        .agg(
            samples=("actual_price", "count"),
            mae=("absolute_error", "mean"),
            mean_percentage_error=("percentage_error", "mean"),
            max_error=("absolute_error", "max"),
        )
        .reset_index()
    )

    # Ignore cities with very few test samples.
    city_analysis = city_analysis[
        city_analysis["samples"] >= 10
    ]

    city_analysis = city_analysis.sort_values(
        "mae",
        ascending=False,
    )

    print("\n" + "=" * 70)
    print("TOP 15 CITIES BY MAE")
    print("=" * 70)

    print(
        city_analysis
        .head(15)
        .to_string(
            index=False,
            formatters={
                "mae": "${:,.2f}".format,
                "mean_percentage_error": "{:.2f}%".format,
                "max_error": "${:,.2f}".format,
            },
        )
    )


    # ============================================================
    # 13. ERROR DISTRIBUTION
    # ============================================================

    print("\n" + "=" * 70)
    print("ERROR DISTRIBUTION")
    print("=" * 70)

    print(
        error_df["absolute_error"]
        .describe()
        .to_string()
    )


    # ============================================================
    # 14. SAVE DETAILED ERROR DATA
    # ============================================================

    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_output_path = (
        EXP_DIR
        / "error_analysis.csv"
    )

    error_df = error_df.sort_values(
        "absolute_error",
        ascending=False,
    )

    error_df.to_csv(
        error_output_path,
        index=False,
    )


    # ============================================================
    # 15. SAVE SUMMARY
    # ============================================================

    summary = {
        "model": "Ridge Regression",
        "alpha": 1,
        "dataset_samples": len(df),
        "development_samples": len(X_dev),
        "test_samples": len(X_test),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        },
        "error_direction": {
            key: int(value)
            for key, value in direction_summary.items()
        },
        "largest_absolute_error": float(
            error_df["absolute_error"].max()
        ),
        "mean_absolute_error": float(
            error_df["absolute_error"].mean()
        ),
    }

    summary_output_path = (
        EXP_DIR
        / "error_analysis_summary.json"
    )

    with open(
        summary_output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )


    # ============================================================
    # 16. FINAL MESSAGE
    # ============================================================

    print("\n" + "=" * 70)
    print("ERROR ANALYSIS COMPLETED")
    print("=" * 70)

    print(f"\nDetailed results:")
    print(error_output_path)

    print(f"\nSummary:")
    print(summary_output_path)