import json
from datetime import datetime

import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)
from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
)

from sklearn.pipeline import Pipeline

from src.config import (
    ALL_FEATURES,
    CV_FOLDS,
    DATASET_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)

from src.preprocessing import create_preprocessor
from src.data_cleaning import clean_data


# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_PATH = (
    "experiments_results/model_tuning_results.json"
)



def Tune_and_choose_right_model():

    print("=" * 80)
    print("MODEL TUNING AND COMPARISON")
    print("=" * 80)


    # ========================================================
    # 1. LOAD DATASET
    # ========================================================

    print("\nLoading dataset...")

    df = pd.read_csv(
        DATASET_PATH
    )

    print(
            "Original Dataset shape:",
            df.shape
        )

    df = df[df["price"] > 0]

    print(
        "Dataset shape after removing 0:",
        df.shape
    )

    # ========================================================
    # 2. PREPARE DATA
    # ========================================================

    X = df[
        ALL_FEATURES
    ]

    y = df[
        TARGET
    ]


    # ========================================================
    # 3. DEVELOPMENT / TEST SPLIT
    # ========================================================

    X_development, X_test, y_development, y_test = (
        train_test_split(

            X,

            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE
        )
    )


    print(
        "\nDevelopment samples:",
        len(X_development)
    )

    print(
        "Test samples:",
        len(X_test)
    )


    # ========================================================
    # 4. DEFINE CANDIDATE MODELS
    # ========================================================

    models = {

        "Linear Regression": {

            "model": LinearRegression(),

            "parameters": {}
        },


        "Ridge Regression": {

            "model": Ridge(),

            "parameters": {

                "model__alpha": [

                    0.01,
                    0.1,
                    1,
                    10,
                    100
                ]
            }
        },


        "Random Forest": {

            "model": RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1
            ),

            "parameters": {

                "model__n_estimators": [

                    100,
                    200
                ],

                "model__max_depth": [

                    5,
                    10,
                    None
                ],

                "model__min_samples_split": [

                    2,
                    5
                ]
            }
        },


        "Gradient Boosting": {

            "model": GradientBoostingRegressor(
                random_state=RANDOM_STATE
            ),

            "parameters": {

                "model__n_estimators": [

                    100,
                    200
                ],

                "model__learning_rate": [

                    0.05,
                    0.1
                ],

                "model__max_depth": [

                    2,
                    3,
                    5
                ]
            }
        }
    }


    # ========================================================
    # 5. STORE RESULTS
    # ========================================================

    results = []


    # ========================================================
    # 6. TUNE EACH MODEL
    # ========================================================

    for model_name, configuration in models.items():

        print("\n" + "=" * 80)

        print(
            f"TUNING: {model_name}"
        )

        print("=" * 80)


        model = configuration[
            "model"
        ]

        parameters = configuration[
            "parameters"
        ]


        # ----------------------------------------------------
        # Create pipeline
        # ----------------------------------------------------

        pipeline = Pipeline([

            (
                "preprocessor",
                create_preprocessor()
            ),

            (
                "model",
                model
            )
        ])


        # ----------------------------------------------------
        # GridSearchCV
        # ----------------------------------------------------

        grid_search = GridSearchCV(

            estimator=pipeline,

            param_grid=parameters,

            cv=CV_FOLDS,

            # scoring="neg_root_mean_squared_error",
            scoring = {
                        "rmse": "neg_root_mean_squared_error",
                        "mae": "neg_mean_absolute_error",
                        "mse": "neg_mean_squared_error",
                        "r2": "r2"
                    },

            n_jobs=-1,

            verbose=1,

            refit = 'rmse'      ## RMSE remains the selection metric.  (means Choose the model with the lowest cross-validation RMSE.)
        )


        # ----------------------------------------------------
        # Train using DEVELOPMENT data only
        # ----------------------------------------------------

        grid_search.fit(

            X_development,

            y_development
        )


        # ----------------------------------------------------
        # Best score
        # ----------------------------------------------------

        best_cv_rmse = (
            -grid_search.best_score_
        )


        best_parameters = (
            grid_search.best_params_
        )

        best_index = grid_search.best_index_                 ## the index(row) in grid_search table which have the best RMSE(rmse is the one we gave in refit)

        cv_rmse = -grid_search.cv_results_["mean_test_rmse"][best_index]
        

        cv_mae = -grid_search.cv_results_["mean_test_mae"][best_index]
        

        cv_mse = -grid_search.cv_results_["mean_test_mse"][best_index]
        

        cv_r2 = grid_search.cv_results_["mean_test_r2"][best_index]


        print(
            "\nBest Parameters:"
        )

        print(
            best_parameters
        )


        print(
            "\nBest CV RMSE:"
        )

        print(
            best_cv_rmse
        )


        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        results.append({

            "model": model_name,

            "cv_rmse": float(
                best_cv_rmse
            ),

            "cv_metrics": {
                    "rmse": cv_rmse,
                    "mae": cv_mae,
                    "mse": cv_mse,
                    "r2": cv_r2,
                },

            "best_parameters":
                best_parameters
        })


    # ========================================================
    # 7. CREATE RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )


    # ========================================================
    # 8. SORT MODELS
    # ========================================================

    results_df = results_df.sort_values(
        by="cv_rmse",
        ascending=True
    ).reset_index(
        drop=True
    )


    # ========================================================
    # 9. DISPLAY FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "FINAL MODEL COMPARISON"
    )

    print("=" * 80)


    print(
        results_df.to_string(
            index=False
        )
    )


    # ========================================================
    # 10. SELECT BEST MODEL
    # ========================================================

    best_model = results_df.iloc[0]


    print("\n" + "=" * 80)

    print(
        "SELECTED MODEL"
    )

    print("=" * 80)


    print(
        "Model:",
        best_model["model"]
    )

    print(
        "CV RMSE:",
        best_model["cv_rmse"]
    )

    print(
        "Parameters:",
        best_model["best_parameters"]
    )


    # ========================================================
    # 11. SAVE RESULTS
    # ========================================================

    output = {

        "created_at":
            datetime.now().isoformat(),

        "dataset": str(
            DATASET_PATH
        ),

        "target":
            TARGET,

        "cross_validation_folds":
            CV_FOLDS,

        "random_state":
            RANDOM_STATE,

        "selection_metric":
            "RMSE",

        "selection_rule":
            "Lowest cross-validation RMSE",

        "selected_model":
            best_model["model"],

        "selected_cv_rmse":
            float(
                best_model["cv_rmse"]
            ),

        "selected_parameters":
            best_model["best_parameters"],

        "models": results
    }


    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )


    print("\nResults saved to:")

    print(
        RESULTS_PATH
    )


    # ========================================================
    # 12. IMPORTANT NOTE
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "MODEL SELECTION COMPLETE"
    )

    print("=" * 80)

    print(
        "\nThe model with the LOWEST CV RMSE "
        "was selected."
    )

    print(
        "The test set was NOT used for model selection."
    )

    return output
