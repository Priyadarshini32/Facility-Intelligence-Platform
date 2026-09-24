"""Task 3 inference using the saved building-specific XGBoost models.

The feature names and order are taken from the saved XGBoost model.
This prevents inference/training feature mismatches.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd


# These are the features actually used by the saved Task 3 models.
BASE_FEATURES = [
    "hour",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "lag_1",
    "lag_6",
    "lag_24",
    "lag_168",
    "rolling_mean_6h",
    "rolling_mean_24h",
    "rolling_std_24h",
]


def load_energy_models(model_path):
    """Load the dictionary of saved building-specific models."""
    return joblib.load(Path(model_path))


def load_energy_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_model_feature_names(model):
    """
    Get the exact feature names expected by the saved XGBoost model.

    XGBoost stores feature_names_in_ when trained from a pandas DataFrame.
    The fallback uses the known 12-feature Task 3 configuration.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "get_booster"):
        booster_names = model.get_booster().feature_names
        if booster_names:
            return list(booster_names)

    return BASE_FEATURES.copy()


def create_features(
    history,
    timestamp,
    feature_names=None,
):
    """Create exactly the features expected by the saved model."""

    values = history.astype(float)

    if len(values) < 168:
        raise ValueError(
            "At least 168 hourly observations are required "
            "for the 7-day lag."
        )

    row = {
        "hour": timestamp.hour,
        "day_of_week": timestamp.dayofweek,
        "is_weekend": int(
            timestamp.dayofweek >= 5
        ),
        "hour_sin": np.sin(
            2 * np.pi * timestamp.hour / 24
        ),
        "hour_cos": np.cos(
            2 * np.pi * timestamp.hour / 24
        ),
        "lag_1": values.iloc[-1],
        "lag_6": values.iloc[-6],
        "lag_24": values.iloc[-24],
        "lag_168": values.iloc[-168],
        "rolling_mean_6h": values.iloc[-6:].mean(),
        "rolling_mean_24h": values.iloc[-24:].mean(),
        "rolling_std_24h": values.iloc[-24:].std(),
    }

    expected = (
        list(feature_names)
        if feature_names is not None
        else BASE_FEATURES
    )

    missing = [
        feature
        for feature in expected
        if feature not in row
    ]

    if missing:
        raise ValueError(
            "Saved model expects unsupported features: "
            + ", ".join(missing)
        )

    # IMPORTANT:
    # Use the exact training feature order.
    return pd.DataFrame(
        [[row[name] for name in expected]],
        columns=expected,
    )


def forecast_energy(
    building_id,
    hourly_history,
    model,
    horizon=24,
):
    """Recursively forecast future hourly building energy."""

    if horizon <= 0:
        raise ValueError(
            "horizon must be greater than zero."
        )

    data = hourly_history.copy()

    data["timestamp"] = pd.to_datetime(
        data["timestamp"]
    )

    data = (
        data[
            data["building_id"] == building_id
        ]
        .sort_values("timestamp")
        .dropna(
            subset=["power_consumption"]
        )
    )

    if data.empty:
        raise ValueError(
            f"No energy history found for {building_id}."
        )

    hourly = (
        data.set_index("timestamp")[
            "power_consumption"
        ]
        .resample("1h")
        .sum()
        .dropna()
    )

    if len(hourly) < 168:
        raise ValueError(
            f"{building_id} needs at least "
            "168 hourly observations."
        )

    feature_names = get_model_feature_names(
        model
    )

    history = hourly.copy()
    predictions = []

    next_timestamp = (
        history.index[-1]
        + pd.Timedelta(hours=1)
    )

    for _ in range(horizon):

        X = create_features(
            history,
            next_timestamp,
            feature_names=feature_names,
        )

        prediction = float(
            model.predict(X)[0]
        )

        predictions.append(
            {
                "building_id": building_id,
                "timestamp": next_timestamp,
                "predicted_energy": prediction,
            }
        )

        # Recursive forecasting:
        # the prediction becomes the next lag value.
        history.loc[next_timestamp] = prediction

        next_timestamp += pd.Timedelta(hours=1)

    return pd.DataFrame(predictions)
