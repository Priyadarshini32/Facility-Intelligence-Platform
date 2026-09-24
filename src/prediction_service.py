"""Task 2 inference using the saved predictive-maintenance model.

The dashboard loads the complete sklearn Pipeline saved by the notebook.
Feature lists are read from the saved pipeline itself, so inference does
not depend on a manually maintained model_config.json feature list.
"""

from pathlib import Path
import json

import joblib
import pandas as pd


SENSOR_COLS = [
    "temperature",
    "humidity",
    "pressure",
    "vibration",
    "power_consumption",
    "occupancy_count",
]

ROLLING_SPECS = {
    "vibration": [4, 24, 48],
    "power_consumption": [4, 24, 48],
    "temperature": [4, 24],
    "pressure": [4, 24],
}

CHANGE_COLS = [
    "vibration",
    "power_consumption",
    "temperature",
    "pressure",
]

DEFAULT_THRESHOLD = 0.76


def load_model(model_path):
    """Load the complete saved sklearn Pipeline."""
    return joblib.load(Path(model_path))


def load_model_config(config_path):
    """Load the saved model configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_pipeline_feature_lists(model):
    """
    Extract the exact raw input feature columns from the saved
    ColumnTransformer inside the sklearn Pipeline.
    """
    if not hasattr(model, "named_steps"):
        raise ValueError(
            "Saved predictive-maintenance model is not the expected "
            "sklearn Pipeline."
        )

    preprocessor = model.named_steps.get("preprocessor")

    if preprocessor is None:
        raise ValueError(
            "Saved model does not contain a 'preprocessor' step."
        )

    numeric_features = []
    categorical_features = []

    for name, _, columns in preprocessor.transformers_:
        if name == "num":
            numeric_features = list(columns)
        elif name == "cat":
            categorical_features = list(columns)

    if not numeric_features and not categorical_features:
        raise ValueError(
            "Could not extract feature columns from the saved "
            "ColumnTransformer."
        )

    return numeric_features, categorical_features


def load_model_bundle(model_dir):
    """
    Load the saved model and configuration.

    If model_config.json does not contain feature lists, they are
    extracted directly from the saved preprocessing pipeline.
    """
    model_dir = Path(model_dir)

    model = load_model(
        model_dir / "predictive_maintenance_model.joblib"
    )

    config_path = model_dir / "model_config.json"

    if config_path.exists():
        config = load_model_config(config_path)
    else:
        config = {}

    numeric_features, categorical_features = (
        _get_pipeline_feature_lists(model)
    )

    # The saved pipeline is the source of truth for feature columns.
    config["numeric_features"] = numeric_features
    config["categorical_features"] = categorical_features

    if "threshold" not in config:
        config["threshold"] = DEFAULT_THRESHOLD

    if "model_name" not in config:
        config["model_name"] = "Saved Predictive Maintenance Model"

    return model, config


def _add_time_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)
    return df


def _add_historical_features(df):
    """Reproduce the leakage-safe notebook features."""
    df = df.copy()
    df = df.sort_values(
        ["asset_id", "timestamp"]
    ).reset_index(drop=True)

    for col, windows in ROLLING_SPECS.items():
        shifted = df.groupby("asset_id")[col].shift(1)

        for window in windows:
            suffix = {
                4: "1h",
                24: "6h",
                48: "12h",
            }[window]

            grouped = shifted.groupby(
                df["asset_id"]
            )

            df[f"{col}_mean_{suffix}"] = (
                grouped.transform(
                    lambda x: x.rolling(
                        window,
                        min_periods=1,
                    ).mean()
                )
            )

            if col in [
                "vibration",
                "power_consumption",
            ]:
                df[f"{col}_max_{suffix}"] = (
                    grouped.transform(
                        lambda x: x.rolling(
                            window,
                            min_periods=1,
                        ).max()
                    )
                )

    for col in CHANGE_COLS:
        df[f"{col}_change_1h"] = (
            df[col]
            - df.groupby("asset_id")[col].shift(4)
        )

        df[f"{col}_change_6h"] = (
            df[col]
            - df.groupby("asset_id")[col].shift(24)
        )

    return df


def _prepare_features(
    history,
    metadata,
    asset_id,
    numeric_features,
    categorical_features,
):
    history = history.copy()
    history["timestamp"] = pd.to_datetime(
        history["timestamp"]
    )

    asset_metadata = metadata[
        metadata["asset_id"] == asset_id
    ].copy()

    if asset_metadata.empty:
        raise ValueError(
            f"Unknown asset_id: {asset_id}"
        )

    history = history.merge(
        asset_metadata[
            [
                "asset_id",
                "asset_type",
                "manufacturer",
                "installation_date",
                "capacity",
                "parent_asset_id",
            ]
        ],
        on="asset_id",
        how="left",
    )

    history["installation_date"] = pd.to_datetime(
        history["installation_date"]
    )

    history["asset_age_days"] = (
        history["timestamp"]
        - history["installation_date"]
    ).dt.days

    history["asset_age_years"] = (
        history["asset_age_days"] / 365.25
    )

    history["has_parent_asset"] = (
        history["parent_asset_id"]
        .notna()
        .astype(int)
    )

    history = _add_time_features(history)
    history = _add_historical_features(history)

    for col in SENSOR_COLS:
        history[f"{col}_missing"] = (
            history[col].isna().astype(int)
        )

    feature_columns = list(
        dict.fromkeys(
            list(numeric_features)
            + list(categorical_features)
        )
    )

    missing_features = [
        c for c in feature_columns
        if c not in history.columns
    ]

    if missing_features:
        raise ValueError(
            "The following model features could not be created: "
            + ", ".join(missing_features)
        )

    return history, feature_columns


def predict_failure(
    asset_id,
    current_telemetry,
    telemetry_history,
    metadata,
    model,
    threshold=DEFAULT_THRESHOLD,
    numeric_features=None,
    categorical_features=None,
):
    """Generate a 24-hour failure-risk prediction."""

    # If the caller does not pass feature lists, read them from
    # the saved pipeline itself.
    if (
        numeric_features is None
        or categorical_features is None
    ):
        (
            numeric_features,
            categorical_features,
        ) = _get_pipeline_feature_lists(model)

    current = pd.DataFrame([current_telemetry])
    current["asset_id"] = asset_id

    history = telemetry_history[
        telemetry_history["asset_id"] == asset_id
    ].copy()

    history = pd.concat(
        [history, current],
        ignore_index=True,
    )

    history = (
        history
        .drop_duplicates(
            subset=["timestamp", "asset_id"],
            keep="last",
        )
        .sort_values("timestamp")
    )

    prepared, feature_columns = _prepare_features(
        history,
        metadata,
        asset_id,
        numeric_features,
        categorical_features,
    )

    latest = prepared.iloc[[-1]]
    X_latest = latest[feature_columns]

    probability = float(
        model.predict_proba(X_latest)[0, 1]
    )

    prediction = int(
        probability >= threshold
    )

    return {
        "asset_id": asset_id,
        "failure_probability": round(
            probability,
            4,
        ),
        "prediction": prediction,
        "predicted_failure": prediction,
        "threshold": threshold,
        "risk_level": (
            "High"
            if prediction == 1
            else "Normal"
        ),
    }
