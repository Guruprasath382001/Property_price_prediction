from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)
from src.config import RANDOM_STATE


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(
    algorithm_name,
    best_parameters,
):
    """
    Create the selected algorithm using the
    hyperparameters chosen by choose_algorithm.py.
    """

    if algorithm_name == "Ridge Regression":

        alpha = best_parameters[
            "model__alpha"
        ]

        return Ridge(
            alpha=alpha
        )


    if algorithm_name == "Random Forest":

        n_estimators = best_parameters[
            "model__n_estimators"
        ]

        max_depth = best_parameters[
            "model__max_depth"
        ]

        min_samples_split = best_parameters[
            "model__min_samples_split"
        ]

        return RandomForestRegressor(

            n_estimators=n_estimators,

            max_depth=max_depth,

            min_samples_split=min_samples_split,

            random_state=RANDOM_STATE,

            n_jobs=-1,
        )


    if algorithm_name == "Gradient Boosting":

        n_estimators = best_parameters[
            "model__n_estimators"
        ]

        learning_rate = best_parameters[
            "model__learning_rate"
        ]

        max_depth = best_parameters[
            "model__max_depth"
        ]

        return GradientBoostingRegressor(

            n_estimators=n_estimators,

            learning_rate=learning_rate,

            max_depth=max_depth,

            random_state=RANDOM_STATE,
        )

    if algorithm_name == "Linear Regression":

        return LinearRegression()


    raise ValueError(
        f"Unknown algorithm: {algorithm_name}"
    )

# import pandas as pd

# df = pd.read_csv(
#         DATASET_PATH
#     )

# print(df.columns.tolist())

# print(df[df["price"] == 0].to_string())
