import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    ALL_FEATURES,
    DATASET_PATH,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def data_distribution():

    print("=" * 80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(DATASET_PATH)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    (
        X_development,
        X_test,
        y_development,
        y_test
    ) = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\n" + "=" * 80)
    print("DATASET PRICE DISTRIBUTION")
    print("=" * 80)

    print(y.describe())

    print("\n" + "=" * 80)
    print("DEVELOPMENT PRICE DISTRIBUTION")
    print("=" * 80)

    print(y_development.describe())

    print("\n" + "=" * 80)
    print("TEST PRICE DISTRIBUTION")
    print("=" * 80)

    print(y_test.describe())

    print("\n" + "=" * 80)
    print("EXTREME PRICE COUNTS")
    print("=" * 80)

    thresholds = [
        0,
        250_000,
        500_000,
        1_000_000,
        2_000_000,
        5_000_000,
        10_000_000,
        20_000_000
    ]

    for threshold in thresholds:

        total = (y > threshold).sum()

        development = (
            y_development > threshold
        ).sum()

        test = (
            y_test > threshold
        ).sum()

        print(
            f"\nPrice > {threshold:,.0f}"
        )

        print(
            f"Total       : {total}"
        )

        print(
            f"Development : {development}"
        )

        print(
            f"Test        : {test}"
        )

    print("\n" + "=" * 80)
    print("ZERO-PRICE RECORDS")
    print("=" * 80)

    zero_total = (y == 0).sum()

    zero_development = (
        y_development == 0
    ).sum()

    zero_test = (
        y_test == 0
    ).sum()

    print(
        "Total zero-price records:",
        zero_total
    )

    print(
        "Development zero-price records:",
        zero_development
    )

    print(
        "Test zero-price records:",
        zero_test
    )

    print("\n" + "=" * 80)
    print("TOP 20 MOST EXPENSIVE PROPERTIES")
    print("=" * 80)

    top_properties = (
        df
        .sort_values(
            TARGET,
            ascending=False
        )
        .head(20)
    )

    print(
        top_properties[
            [TARGET] + ALL_FEATURES
        ].to_string(
            index=False
        )
    )
