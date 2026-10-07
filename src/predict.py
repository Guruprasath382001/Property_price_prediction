import joblib
import pandas as pd

from src.config import (
    ALL_FEATURES,
    MODEL_PATH,
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained ML pipeline.
    """

    return joblib.load(
        MODEL_PATH
    )


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input(data):
    """
    Validate prediction input.
    """

    # --------------------------------------------------------
    # Check required features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in ALL_FEATURES
        if feature not in data
    ]

    if missing_features:

        raise ValueError(
            f"Missing features: "
            f"{missing_features}"
        )

    # --------------------------------------------------------
    # Numeric validation
    # --------------------------------------------------------

    if data["bedrooms"] < 0:
        raise ValueError(
            "bedrooms cannot be negative."
        )

    if data["bathrooms"] < 0:
        raise ValueError(
            "bathrooms cannot be negative."
        )

    if data["sqft_living"] <= 0:
        raise ValueError(
            "sqft_living must be greater than 0."
        )

    if data["sqft_lot"] <= 0:
        raise ValueError(
            "sqft_lot must be greater than 0."
        )

    if data["floors"] <= 0:
        raise ValueError(
            "floors must be greater than 0."
        )

    if data["waterfront"] not in [0, 1]:
        raise ValueError(
            "waterfront must be either 0 or 1."
        )

    if data["view"] < 0:
        raise ValueError(
            "view cannot be negative."
        )

    if data["condition"] < 1 or data["condition"] > 5:
        raise ValueError(
            "condition must be between 1 and 5."
        )

    if data["sqft_above"] <= 0:
        raise ValueError(
            "sqft_above must be greater than 0."
        )

    if data["sqft_basement"] < 0:
        raise ValueError(
            "sqft_basement cannot be negative."
        )

    if data["yr_built"] <= 0:
        raise ValueError(
            "yr_built must be greater than 0."
        )

    if data["yr_renovated"] < 0:
        raise ValueError(
            "yr_renovated cannot be negative."
        )

    # --------------------------------------------------------
    # Logical validation
    # --------------------------------------------------------

    if data["sqft_above"] > data["sqft_living"]:
        raise ValueError(
            "sqft_above cannot be greater than sqft_living."
        )

    if data["sqft_basement"] > data["sqft_living"]:
        raise ValueError(
            "sqft_basement cannot be greater than sqft_living."
        )

    if data["yr_renovated"] != 0:

        if data["yr_renovated"] < data["yr_built"]:
            raise ValueError(
                "yr_renovated cannot be earlier than yr_built."
            )

    # --------------------------------------------------------
    # Categorical validation
    # --------------------------------------------------------

    if not isinstance(data["city"], str):
        raise ValueError(
            "city must be a string."
        )

    if not isinstance(data["statezip"], str):
        raise ValueError(
            "statezip must be a string."
        )


# ============================================================
# PREDICT PRICE
# ============================================================

def predict_price(data):
    """
    Predict property price for one property.
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_input(
        data
    )

    # --------------------------------------------------------
    # Convert input dictionary into DataFrame
    # --------------------------------------------------------

    input_df = pd.DataFrame(
        [data],
        columns=ALL_FEATURES,
    )

    # --------------------------------------------------------
    # Load trained pipeline
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(
        input_df
    )

    return float(
        prediction[0]
    )

