from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)


def create_preprocessor():
    """
    Create the preprocessing pipeline.

    Numerical:
        Missing values -> median
        Scaling -> StandardScaler

    Categorical:
        Missing values -> most frequent
        Encoding -> OneHotEncoder
    """

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        ),
    ])

    preprocessor = ColumnTransformer([
        (
            "num",
            numeric_pipeline,
            NUMERIC_FEATURES,
        ),
        (
            "cat",
            categorical_pipeline,
            CATEGORICAL_FEATURES,
        ),
    ])

    return preprocessor