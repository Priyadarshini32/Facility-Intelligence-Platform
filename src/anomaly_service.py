"""Task 4 dashboard service.

Task 4 produces a hybrid anomaly result in the notebook.
The dashboard therefore loads that saved result instead of
retraining the anomaly detectors on every request.

The trained anomaly models are also available as a saved artifact
for future batch/inference extensions.
"""

from pathlib import Path

import joblib
import pandas as pd


def load_anomaly_models(path):
    """Load the anomaly-model bundle saved by Task 4."""
    return joblib.load(Path(path))


def load_anomaly_results(path):
    """Load the precomputed hybrid anomaly results."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Anomaly results file not found: {path}"
        )

    return pd.read_csv(path)


def get_anomalies(
    anomaly_df,
    severity=None,
    asset_type=None,
    asset_id=None,
):
    """Filter saved anomaly results."""

    result = anomaly_df.copy()

    if severity:
        result = result[
            result["severity"]
            .astype(str)
            .str.lower()
            == severity.lower()
        ]

    if (
        asset_type
        and "asset_type" in result.columns
    ):
        result = result[
            result["asset_type"]
            .astype(str)
            .str.lower()
            == asset_type.lower()
        ]

    if (
        asset_id
        and "asset_id" in result.columns
    ):
        result = result[
            result["asset_id"].astype(str)
            == str(asset_id)
        ]

    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(
            result["timestamp"],
            errors="coerce",
        )
        result = result.sort_values(
            "timestamp",
            ascending=False,
        )

    return result.reset_index(drop=True)


def anomaly_summary(anomaly_df):
    """Return counts from the saved hybrid result."""

    if "severity" not in anomaly_df.columns:
        return {}

    counts = (
        anomaly_df["severity"]
        .value_counts()
        .to_dict()
    )

    return {
        "Normal": int(
            counts.get("Normal", 0)
        ),
        "Watch": int(
            counts.get("Watch", 0)
        ),
        "Anomaly": int(
            counts.get("Anomaly", 0)
        ),
        "Critical": int(
            counts.get("Critical", 0)
        ),
    }
