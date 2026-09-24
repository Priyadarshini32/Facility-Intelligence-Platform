"""Predictive-maintenance inference service.

The service reproduces the feature engineering used during Task 2 and
loads the complete saved sklearn Pipeline.  The saved pipeline is the
source of truth for preprocessing and model features.
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
    "vibration": [4, 24, 48],          # 1h, 6h, 12h
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
    """Load saved model configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_pipeline_feature_lists(model):
    """Read the exact raw feature columns from the saved pipeline."""

    if not hasattr(model, "named_steps"):
        raise ValueError(
            "Saved model is not the expected sklearn Pipeline."
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
    """Load model + configuration.

    Feature lists are always obtained from the saved Pipeline so the API
    cannot silently drift from the training preprocessing.
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

    config["numeric_features"] = numeric_features
    config["categorical_features"] = categorical_features
    config.setdefault("threshold", DEFAULT_THRESHOLD)
    config.setdefault(
        "model_name",
        "Saved Predictive Maintenance Model",
    )

    return model, config


def _add_time_features(df):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def _add_historical_features(df):
    """Reproduce the leakage-safe notebook feature engineering."""

    df = df.copy()

    df["asset_id"] = df["asset_id"].astype(str)

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

            grouped = shifted.groupby(df["asset_id"])

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
    """Build exactly the features expected by the saved model."""

    history = history.copy()

    history["asset_id"] = (
        history["asset_id"].astype(str).str.strip()
    )
    asset_id = str(asset_id).strip()

    history["timestamp"] = pd.to_datetime(
        history["timestamp"]
    )

    asset_metadata = metadata.copy()
    asset_metadata["asset_id"] = (
        asset_metadata["asset_id"]
        .astype(str)
        .str.strip()
    )

    asset_metadata = asset_metadata[
        asset_metadata["asset_id"] == asset_id
    ].copy()

    if asset_metadata.empty:
        raise ValueError(
            f"Unknown asset_id: {asset_id}"
        )

    metadata_columns = [
        "asset_id",
        "asset_type",
        "manufacturer",
        "installation_date",
        "capacity",
        "parent_asset_id",
    ]

    history = history.drop(
        columns=[
            c for c in [
                "asset_type",
                "manufacturer",
                "installation_date",
                "capacity",
                "parent_asset_id",
            ]
            if c in history.columns
        ],
        errors="ignore",
    )

    history = history.merge(
        asset_metadata[metadata_columns],
        on="asset_id",
        how="left",
        validate="many_to_one",
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
    """Generate a 24-hour-ahead failure-risk prediction.

    `telemetry_history` must contain observations before the current
    observation. The current observation is appended and is the row
    passed to the model.
    """

    if (
        numeric_features is None
        or categorical_features is None
    ):
        (
            numeric_features,
            categorical_features,
        ) = _get_pipeline_feature_lists(model)

    asset_id = str(asset_id).strip()

    # Normalize the incoming API observation before concatenating
    # it with the pandas telemetry history. JSON sends timestamps
    # as strings, while the loaded telemetry may contain pandas
    # Timestamp objects. Mixing those types causes:
    # TypeError: '<' not supported between instances of
    # 'str' and 'Timestamp'
    current = pd.DataFrame([current_telemetry])
    current["asset_id"] = asset_id

    if "timestamp" not in current.columns:
        raise ValueError(
            "current_telemetry must contain a timestamp."
        )

    current["timestamp"] = pd.to_datetime(
        current["timestamp"],
        errors="coerce",
    )

    if current["timestamp"].isna().any():
        raise ValueError(
            "current_telemetry contains an invalid timestamp."
        )

    history = telemetry_history.copy()

    history["asset_id"] = (
        history["asset_id"].astype(str).str.strip()
    )

    history = history[
        history["asset_id"] == asset_id
    ].copy()

    # Always use one datetime type before concat/sort.
    history["timestamp"] = pd.to_datetime(
        history["timestamp"],
        errors="coerce",
    )

    history = history.dropna(
        subset=["timestamp"]
    )

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
        .sort_values(
            ["asset_id", "timestamp"]
        )
        .reset_index(drop=True)
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
        probability >= float(threshold)
    )

    return {
        "asset_id": asset_id,
        "timestamp": str(
            latest["timestamp"].iloc[0]
        ),
        "failure_probability": round(
            probability,
            6,
        ),
        "failure_probability_percent": round(
            probability * 100,
            4,
        ),
        "prediction": prediction,
        "predicted_failure": prediction,
        "threshold": float(threshold),
        "risk_level": (
            "Maintenance Alert"
            if prediction == 1
            else "Normal"
        ),
        "model": (
            model.named_steps["model"].__class__.__name__
            if hasattr(model, "named_steps")
            and "model" in model.named_steps
            else "Saved Model"
        ),
    }
