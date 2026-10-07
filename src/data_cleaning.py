from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "data.csv"

PRICE_OUTLIER_ANALYSIS_PATH = (
    PROJECT_ROOT
    / "experiments_results"
    / "price_outlier_analysis.csv"
)

CLEANED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "cleaned_data.csv"
)


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_data():

    if not RAW_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Raw dataset not found:\n"
            f"{RAW_DATA_PATH}"
        )

    df = pd.read_csv(
        RAW_DATA_PATH
    )

    print("=" * 70)
    print("RAW DATA")
    print("=" * 70)

    print(
        f"Rows    : {len(df):,}"
    )

    print(
        f"Columns : {len(df.columns):,}"
    )

    print()

    return df


# ============================================================
# LOAD PRICE OUTLIER ANALYSIS
# ============================================================

def load_price_outlier_analysis():

    if not PRICE_OUTLIER_ANALYSIS_PATH.exists():

        raise FileNotFoundError(
            "Price outlier analysis file not found:\n"
            f"{PRICE_OUTLIER_ANALYSIS_PATH}\n\n"
            "Run price_outlier_analysis.py first."
        )

    analysis_df = pd.read_csv(
        PRICE_OUTLIER_ANALYSIS_PATH
    )

    print("=" * 70)
    print("PRICE OUTLIER ANALYSIS")
    print("=" * 70)

    print(
        f"Rows : {len(analysis_df):,}"
    )

    print(
        f"Columns : {len(analysis_df.columns):,}"
    )

    print()

    return analysis_df


# ============================================================
# REMOVE NON-NORMAL RECORDS
# ============================================================

def remove_non_normal_records(
    df,
    analysis_df
):

    print("=" * 70)
    print("PRICE OUTLIER FILTERING")
    print("=" * 70)

    # --------------------------------------------------------
    # Make sure the classification column exists
    # --------------------------------------------------------

    classification_column = "price_anomaly_class"

    if classification_column not in analysis_df.columns:

        raise KeyError(
            f"Column '{classification_column}' "
            "was not found in price_outlier_analysis.csv.\n\n"
            f"Available columns:\n"
            f"{list(analysis_df.columns)}"
        )

    # --------------------------------------------------------
    # Display classification distribution
    # --------------------------------------------------------

    print("\nOutlier classification:")

    print(
        analysis_df[
            classification_column
        ]
        .value_counts()
        .to_string()
    )

    print()

    # --------------------------------------------------------
    # Keep only NORMAL records
    # --------------------------------------------------------

    normal_analysis = analysis_df[
        analysis_df[classification_column]
        == "NORMAL"
    ].copy()

    print(
        f"Normal records : {len(normal_analysis):,}"
    )

    print(
        f"Non-normal records : "
        f"{len(analysis_df) - len(normal_analysis):,}"
    )

    # --------------------------------------------------------
    # Identify columns shared between the two files
    # --------------------------------------------------------

    common_columns = [
        column
        for column in df.columns
        if column in normal_analysis.columns
    ]

    if not common_columns:

        raise ValueError(
            "No common columns found between "
            "data.csv and price_outlier_analysis.csv."
        )

    # --------------------------------------------------------
    # Create a key from the common columns
    #
    # We do this instead of relying on dataframe row numbers.
    # --------------------------------------------------------

    df = df.copy()

    normal_analysis = normal_analysis.copy()

    df["_record_key"] = (
        df[common_columns]
        .astype(str)
        .agg("|".join, axis=1)
    )

    normal_analysis["_record_key"] = (
        normal_analysis[common_columns]
        .astype(str)
        .agg("|".join, axis=1)
    )

    normal_keys = set(
        normal_analysis["_record_key"]
    )

    before = len(df)

    df = df[
        df["_record_key"].isin(normal_keys)
    ].copy()

    removed = before - len(df)

    # --------------------------------------------------------
    # Remove temporary key
    # --------------------------------------------------------

    df.drop(
        columns=["_record_key"],
        inplace=True
    )

    print()
    print(
        f"Rows before : {before:,}"
    )

    print(
        f"Removed     : {removed:,}"
    )

    print(
        f"Rows after  : {len(df):,}"
    )

    print()

    return df


# ============================================================
# REMOVE UNUSABLE COLUMNS
# ============================================================

def remove_unusable_columns(df):

    columns_to_remove = [
        "street",
        "country",
    ]

    existing_columns = [
        column
        for column in columns_to_remove
        if column in df.columns
    ]

    df = df.drop(
        columns=existing_columns
    )

    print("=" * 70)
    print("REMOVED COLUMNS")
    print("=" * 70)

    for column in existing_columns:

        print(
            f"- {column}"
        )

    print()

    return df


# ============================================================
# CLEAN RENOVATION YEAR
# ============================================================

def clean_renovation_year(df):

    df = df.copy()

    invalid_renovation = (
        (df["yr_renovated"] > 0)
        &
        (
            df["yr_renovated"]
            < df["yr_built"]
        )
    )

    count = invalid_renovation.sum()

    print("=" * 70)
    print("RENOVATION YEAR")
    print("=" * 70)

    print(
        f"Invalid renovation years : {count:,}"
    )

    # --------------------------------------------------------
    # Do NOT remove these rows.
    #
    # 0 means "no valid renovation year available".
    # --------------------------------------------------------

    df.loc[
        invalid_renovation,
        "yr_renovated"
    ] = 0

    print(
        "Invalid renovation years "
        "were replaced with 0."
    )

    print()

    return df


# ============================================================
# SAVE CLEAN DATA
# ============================================================

def save_data(df):

    CLEANED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        CLEANED_DATA_PATH,
        index=False
    )

    print("=" * 70)
    print("CLEANED DATA SAVED")
    print("=" * 70)

    print(
        f"Path : {CLEANED_DATA_PATH}"
    )

    print(
        f"Rows : {len(df):,}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def clean_data():

    # --------------------------------------------------------
    # 1. Load original dataset
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Load results of price outlier investigation
    # --------------------------------------------------------

    analysis_df = (
        load_price_outlier_analysis()
    )

    # --------------------------------------------------------
    # 3. Keep only NORMAL records
    # --------------------------------------------------------

    df = remove_non_normal_records(
        df,
        analysis_df
    )

    # --------------------------------------------------------
    # 4. Remove useless columns
    # --------------------------------------------------------

    df = remove_unusable_columns(
        df
    )

    # --------------------------------------------------------
    # 5. Clean invalid renovation years
    # --------------------------------------------------------

    df = clean_renovation_year(
        df
    )

    # --------------------------------------------------------
    # 6. Save final cleaned dataset
    # --------------------------------------------------------

    save_data(
        df
    )

