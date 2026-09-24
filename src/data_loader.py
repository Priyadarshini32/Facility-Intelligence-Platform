"""Load the raw Nectar datasets used by the challenge notebooks."""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def find_data_dir() -> Path:
    """Return the project-level data/raw directory."""
    candidates = [
        DEFAULT_DATA_DIR,
        Path("data/raw"),
        Path("../data/raw"),
    ]

    for data_dir in candidates:
        data_dir = Path(data_dir)
        required = [
            "sensor_telemetry.csv",
            "asset_metadata.csv",
            "asset_connectivity.csv",
        ]

        if all((data_dir / name).exists() for name in required):
            return data_dir.resolve()

    raise FileNotFoundError(
        "Could not find data/raw containing "
        "sensor_telemetry.csv, asset_metadata.csv and "
        "asset_connectivity.csv."
    )


def load_data(data_dir=None):
    """Load telemetry, metadata and connectivity from data/raw."""

    data_path = (
        Path(data_dir).resolve()
        if data_dir is not None
        else find_data_dir()
    )

    telemetry = pd.read_csv(
        data_path / "sensor_telemetry.csv"
    )
    metadata = pd.read_csv(
        data_path / "asset_metadata.csv"
    )
    connectivity = pd.read_csv(
        data_path / "asset_connectivity.csv"
    )

    telemetry["timestamp"] = pd.to_datetime(
        telemetry["timestamp"]
    )

    metadata["installation_date"] = pd.to_datetime(
        metadata["installation_date"]
    )

    return telemetry, metadata, connectivity
