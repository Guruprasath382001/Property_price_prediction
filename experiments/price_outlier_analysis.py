# experiments/outlier_detection.py

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = PROJECT_ROOT / "data" / "data.csv"
RESULTS_DIR = PROJECT_ROOT / "experiments_results"

FULL_REPORT_PATH = RESULTS_DIR / "price_outlier_analysis.csv"
SUSPICIOUS_REPORT_PATH = RESULTS_DIR / "suspicious_price_records.csv"


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "price"

LOCATION_FEATURE = "statezip"
SIZE_FEATURE = "sqft_living"

# Minimum sqft required before calculating price/sqft.
# Avoids division by tiny/invalid values.
MIN_SQFT = 300

# Extremely large deviation from local market.
EXTREME_HIGH_RATIO = 10.0
EXTREME_LOW_RATIO = 0.10

# Strongly suspicious deviation.
SUSPICIOUS_HIGH_RATIO = 5.0
SUSPICIOUS_LOW_RATIO = 0.25


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("=" * 70)
    print("DATASET")
    print("=" * 70)

    print(f"Dataset path : {DATASET_PATH}")
    print(f"Shape        : {df.shape}")
    print()

    return df


# ============================================================
# BASIC DATA QUALITY
# ============================================================

def basic_quality_report(df):

    print("=" * 70)
    print("BASIC DATA QUALITY")
    print("=" * 70)

    print(f"Rows                  : {len(df):,}")
    print(f"Columns               : {len(df.columns):,}")
    print(
        f"Missing values        : "
        f"{df.isna().sum().sum():,}"
    )
    print(
        f"Duplicate rows        : "
        f"{df.duplicated().sum():,}"
    )

    zero_price = (df[TARGET] <= 0).sum()

    print(
        f"Price <= 0            : "
        f"{zero_price:,}"
    )

    print()


# ============================================================
# CREATE PRICE PER SQFT
# ============================================================

def calculate_price_per_sqft(df):

    df = df.copy()

    df["price_per_sqft"] = np.nan             ### Initializes a new column named price_per_sqft in the DataFrame, filling every row with NaN (Not a Number/missing value) as a default starting state

    
    valid_sqft = df[SIZE_FEATURE] >= MIN_SQFT   ### Creates a boolean mask (a series of True and False values) named valid_sqft. if SIZE_FEATURE (sqft_living) if > than MIN_SQFT it is True else False.

    df.loc[valid_sqft, "price_per_sqft"] = (      ### Uses pandas .loc indexing to select only the rows where valid_sqft is True. 
        df.loc[valid_sqft, TARGET]                ### For those valid rows, it divides the target column (TARGET, which is "price") by the sqft_living column (SIZE_FEATURE) and assigns the result to price_per_sqft. Rows where valid_sqft is False retain their initial NaN value.
        / df.loc[valid_sqft, SIZE_FEATURE]
    )

    return df


# ============================================================
# CALCULATE LOCAL MARKET STATISTICS
# ============================================================

def calculate_local_statistics(df):

    df = df.copy()

    valid_price = (
        (df[TARGET] > 0)
        & (df[SIZE_FEATURE] >= MIN_SQFT)
        & df["price_per_sqft"].notna()
    )

    valid_df = df.loc[valid_price].copy()

    # --------------------------------------------------------
    # Median price/sqft for each ZIP code
    # --------------------------------------------------------

    zip_median_ppsf = (
        valid_df
        .groupby(LOCATION_FEATURE)["price_per_sqft"]        ### Groups valid_df by ZIP code (LOCATION_FEATURE, which is "statezip"), 
        .median()                                           ### calculates the median price_per_sqft for each ZIP code, and 
        .rename("zip_median_price_per_sqft")                ### renames the resulting Series to "zip_median_price_per_sqft".
    )

    df = df.merge(
        zip_median_ppsf,                ### Merges the ZIP medians back into the primary df using a left join on statezip. 
        left_on=LOCATION_FEATURE,       ### Every row in df receives a new column, zip_median_price_per_sqft, representing its ZIP code's median price per sqft.
        right_index=True,
        how="left"
    )

    # --------------------------------------------------------
    # City fallback
    # --------------------------------------------------------

    city_median_ppsf = (
        valid_df
        .groupby("city")["price_per_sqft"]      ### Groups valid_df by "city", 
        .median()                               ### calculates the median price_per_sqft for each city, and 
        .rename("city_median_price_per_sqft")   ### renames the Series to "city_median_price_per_sqft".
    )

    df = df.merge(
        city_median_ppsf,       ### Merges the city medians back into df using a left join on city, 
        left_on="city",         ### creating the column city_median_price_per_sqft.
        right_index=True,
        how="left"
    )

    # --------------------------------------------------------
    # Global fallback
    # --------------------------------------------------------

    global_median_ppsf = valid_df["price_per_sqft"].median()    ### Calculates a single global median price per sqft across all valid records in the entire dataset to serve as a final safety net.

    # Use ZIP median first.
    # If unavailable, use city median.
    # If unavailable, use global median.
    df["expected_price_per_sqft"] = (
        df["zip_median_price_per_sqft"]
        .fillna(df["city_median_price_per_sqft"])
        .fillna(global_median_ppsf)
    )

    # --------------------------------------------------------
    # Expected price based on local price/sqft
    # --------------------------------------------------------

    df["expected_local_price"] = (
        df["expected_price_per_sqft"]           ### Calculates expected total price by multiplying the property's square footage (sqft_living) by its localized expected_price_per_sqft.
        * df[SIZE_FEATURE]
    )

    # --------------------------------------------------------
    # Actual / expected ratio
    # --------------------------------------------------------

    df["price_ratio"] = np.nan

    valid_expected = (
        (df["expected_local_price"] > 0)
        & (df[TARGET] > 0)
    )

    df.loc[valid_expected, "price_ratio"] = (                   
        df.loc[valid_expected, TARGET]
        / df.loc[valid_expected, "expected_local_price"]
    )

    '''
        Computes the ratio actual price/expected price for valid rows:
        
        A ratio of 1.0 means the property is priced exactly at its local market average.
        A ratio of 10.0 means the property is listed at 10× its expected local market price.
        A ratio of 0.1 means the property is listed at 1/10th its expected local market price.
    '''

    return df


# ============================================================
# CLASSIFY PRICE ANOMALIES
# ============================================================

def classify_records(df):

    df = df.copy()

    df["price_anomaly_class"] = "NORMAL"

    # --------------------------------------------------------
    # Invalid target
    # --------------------------------------------------------

    invalid_price = df[TARGET] <= 0

    df.loc[
        invalid_price,
        "price_anomaly_class"
    ] = "INVALID_PRICE"

    # --------------------------------------------------------
    # Extreme high price
    # --------------------------------------------------------

    extreme_high = (
        (df["price_ratio"] >= EXTREME_HIGH_RATIO)
        & (df[TARGET] > 0)
    )

    df.loc[
        extreme_high,
        "price_anomaly_class"
    ] = "EXTREME_PRICE_ANOMALY"

    # --------------------------------------------------------
    # Extreme low price
    # --------------------------------------------------------

    extreme_low = (
        (df["price_ratio"] <= EXTREME_LOW_RATIO)
        & (df[TARGET] > 0)
    )

    df.loc[
        extreme_low,
        "price_anomaly_class"
    ] = "EXTREME_PRICE_ANOMALY"

    # --------------------------------------------------------
    # Suspicious high
    # --------------------------------------------------------

    suspicious_high = (
        (df["price_ratio"] >= SUSPICIOUS_HIGH_RATIO)
        & (df["price_ratio"] < EXTREME_HIGH_RATIO)
        & (df[TARGET] > 0)
    )

    df.loc[
        suspicious_high,
        "price_anomaly_class"
    ] = "SUSPICIOUS_HIGH_PRICE"

    # --------------------------------------------------------
    # Suspicious low
    # --------------------------------------------------------

    suspicious_low = (
        (df["price_ratio"] <= SUSPICIOUS_LOW_RATIO)
        & (df["price_ratio"] > EXTREME_LOW_RATIO)
        & (df[TARGET] > 0)
    )

    df.loc[
        suspicious_low,
        "price_anomaly_class"
    ] = "SUSPICIOUS_LOW_PRICE"

    return df


# ============================================================
# PRINT REPORT
# ============================================================

def print_summary(df):

    print("=" * 70)
    print("PRICE ANOMALY SUMMARY")
    print("=" * 70)

    counts = (
        df["price_anomaly_class"]
        .value_counts()
    )

    for category, count in counts.items():

        percentage = (
            count / len(df)
        ) * 100

        print(
            f"{category:<30}"
            f"{count:>6,}"
            f" ({percentage:>6.2f}%)"
        )

    print()


# ============================================================
# PRINT EXTREME RECORDS
# ============================================================

def print_extreme_records(df):

    suspicious = df[
        df["price_anomaly_class"].isin(
            [
                "INVALID_PRICE",
                "EXTREME_PRICE_ANOMALY",
                "SUSPICIOUS_HIGH_PRICE",
                "SUSPICIOUS_LOW_PRICE",
            ]
        )
    ].copy()

    suspicious = suspicious.sort_values(
        by="price_ratio",
        ascending=False
    )

    print("=" * 70)
    print("SUSPICIOUS RECORDS")
    print("=" * 70)

    columns = [
        "price",
        "price_per_sqft",
        "expected_price_per_sqft",
        "expected_local_price",
        "price_ratio",
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "waterfront",
        "view",
        "city",
        "statezip",
        "price_anomaly_class",
    ]

    available_columns = [
        col for col in columns
        if col in suspicious.columns
    ]

    print(
        suspicious[
            available_columns
        ].head(50).to_string(index=False)
    )

    print()


# ============================================================
# SAVE REPORT
# ============================================================

def save_reports(df):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        FULL_REPORT_PATH,
        index=False
    )

    suspicious = df[
        df["price_anomaly_class"] != "NORMAL"
    ].copy()

    suspicious.to_csv(
        SUSPICIOUS_REPORT_PATH,
        index=False
    )

    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Full report       : {FULL_REPORT_PATH}"
    )

    print(
        f"Suspicious report : {SUSPICIOUS_REPORT_PATH}"
    )

    print()



def outlier_detection():

    df = load_data()

    basic_quality_report(df)

    df = calculate_price_per_sqft(df)

    df = calculate_local_statistics(df)

    df = classify_records(df)

    print_summary(df)

    print_extreme_records(df)

    save_reports(df)

