"""Flask backend for the Nectar Intelligent Facilities application.

Training/analysis is performed by the challenge notebooks.
This API only loads saved models/results from outputs/ and
performs inference or exposes already-computed connectivity data.
"""

import sys
from pathlib import Path


# Nectar Task/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request
import pandas as pd
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parents[1]

# Make project-root imports work when starting with:
# python backend/app.py
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_loader import load_data
from src.prediction_service import (
    load_model_bundle,
    predict_failure,
)
from src.energy_service import (
    load_energy_models,
    load_energy_config,
    forecast_energy,
)
from src.anomaly_service import (
    load_anomaly_results,
    anomaly_summary,
    get_anomalies,
)
from src.graph_service import (
    build_graph,
    load_saved_connectivity,
    connected_assets,
    downstream_assets,
    assets_under_site,
    isolated_assets,
)

try:
    from backend.graphql_schema import create_schema
    from graphql import graphql_sync
except ImportError:
    create_schema = None
    graphql_sync = None


# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUTS_DIR = BASE_DIR / "outputs"

PM_DIR = (
    OUTPUTS_DIR
    / "models"
    / "predictive_maintenance"
)

ENERGY_DIR = (
    OUTPUTS_DIR
    / "models"
    / "energy_forecasting"
)

ANOMALY_DIR = (
    OUTPUTS_DIR
    / "models"
    / "anomaly_detection"
)

CONNECTIVITY_DIR = (
    OUTPUTS_DIR
    / "connectivity"
)


app = Flask(__name__)
CORS(app)


# ------------------------------------------------------------------
# Raw data used to create inference features
# ------------------------------------------------------------------

telemetry, metadata, raw_connectivity = load_data(
    DATA_DIR
)

# Normalize identifiers and timestamps once at API startup.
# This prevents mixed string/Timestamp comparisons during inference.
telemetry["asset_id"] = (
    telemetry["asset_id"].astype(str).str.strip()
)
telemetry["timestamp"] = pd.to_datetime(
    telemetry["timestamp"],
    errors="coerce",
)
telemetry = telemetry.dropna(
    subset=["timestamp"]
).copy()

metadata["asset_id"] = (
    metadata["asset_id"].astype(str).str.strip()
)


# ------------------------------------------------------------------
# Load the saved connectivity result when available.
# This keeps Task 5 aligned with the notebook output.
# ------------------------------------------------------------------

saved_connectivity_path = (
    CONNECTIVITY_DIR
    / "asset_connectivity.csv"
)

if saved_connectivity_path.exists():
    connectivity = load_saved_connectivity(
        saved_connectivity_path
    )
else:
    connectivity = raw_connectivity.copy()

graph = build_graph(connectivity)


# ------------------------------------------------------------------
# GraphQL
# ------------------------------------------------------------------

graph_services = {
    "connected_assets": connected_assets,
    "downstream_assets": downstream_assets,
    "assets_under_site": assets_under_site,
    "isolated_assets": isolated_assets,
}

graphql_schema = (
    create_schema(graph, metadata)
    if create_schema is not None
    else None
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def missing_artifact(path):
    return jsonify(
        {
            "error": "Required saved artifact is missing.",
            "path": str(path),
            "message": (
                "Run the corresponding challenge notebook "
                "before using this endpoint."
            ),
        }
    ), 503


# ------------------------------------------------------------------
# Health / overview
# ------------------------------------------------------------------

@app.get("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": (
                "Nectar Intelligent Facilities API"
            ),
        }
    )


@app.get("/api/overview")
def overview():
    return jsonify(
        {
            "sites": int(
                metadata["site_id"].nunique()
            ),
            "buildings": int(
                metadata["building_id"].nunique()
            ),
            "assets": int(
                metadata["asset_id"].nunique()
            ),
            "telemetry_records": int(
                len(telemetry)
            ),
            "connectivity_relationships": int(
                len(connectivity)
            ),
        }
    )


@app.get("/api/artifacts")
def artifacts():
    """Show whether notebook-generated artifacts exist."""

    paths = {
        "predictive_maintenance_model":
            PM_DIR
            / "predictive_maintenance_model.joblib",

        "predictive_maintenance_config":
            PM_DIR / "model_config.json",

        "energy_models":
            ENERGY_DIR
            / "xgboost_energy_models.joblib",

        "energy_config":
            ENERGY_DIR / "model_config.json",

        "anomaly_models":
            ANOMALY_DIR
            / "anomaly_detection_models.joblib",

        "anomaly_results":
            ANOMALY_DIR / "anomaly_results.csv",

        "connectivity":
            CONNECTIVITY_DIR / "asset_connectivity.csv",
    }

    return jsonify(
        {
            name: {
                "available": path.exists(),
                "path": str(path),
            }
            for name, path in paths.items()
        }
    )


# ------------------------------------------------------------------
# Asset endpoints
# ------------------------------------------------------------------

@app.get("/api/assets")
def assets():
    result = metadata.copy()

    site_id = request.args.get("site_id")
    building_id = request.args.get("building_id")
    asset_type = request.args.get("asset_type")

    if site_id:
        result = result[
            result["site_id"].astype(str)
            == site_id
        ]

    if building_id:
        result = result[
            result["building_id"].astype(str)
            == building_id
        ]

    if asset_type:
        result = result[
            result["asset_type"].astype(str)
            == asset_type
        ]

    result = result.where(
        result.notna(),
        None,
    )

    return jsonify(
        result.to_dict(
            orient="records"
        )
    )


# ------------------------------------------------------------------
# Task 2 — Predictive Maintenance
# ------------------------------------------------------------------

@app.get("/predict_failure/status")
def predict_failure_status():
    """Return the readiness of the predictive-maintenance service."""

    model_path = (
        PM_DIR
        / "predictive_maintenance_model.joblib"
    )

    config_path = (
        PM_DIR
        / "model_config.json"
    )

    model_available = model_path.exists()
    config_available = config_path.exists()

    ready = (
        model_available
        and config_available
    )

    return jsonify(
        {
            "service": "predictive-maintenance",
            "status": (
                "ready"
                if ready
                else "not_ready"
            ),
            "model": "XGBoost",
            "prediction_horizon": "24 hours",
            "threshold": 0.76,
            "model_artifact_available": model_available,
            "config_available": config_available,
        }
    )


@app.post("/predict_failure")
@app.post("/api/predict_failure")
def predict_failure_api():
    model_path = (
        PM_DIR
        / "predictive_maintenance_model.joblib"
    )
    config_path = (
        PM_DIR / "model_config.json"
    )

    if not model_path.exists():
        return missing_artifact(model_path)

    if not config_path.exists():
        return missing_artifact(config_path)

    payload = (
        request.get_json(
            silent=True
        )
        or {}
    )

    asset_id = payload.get("asset_id")
    current_telemetry = payload.get(
        "current_telemetry"
    )

    if asset_id is not None:
        asset_id = str(asset_id).strip()

    if not asset_id:
        return jsonify(
            {
                "error": (
                    "asset_id is required."
                )
            }
        ), 400

    if not current_telemetry:
        return jsonify(
            {
                "error": (
                    "current_telemetry is required."
                )
            }
        ), 400

    asset_history = telemetry[
        telemetry["asset_id"]
        == asset_id
    ].sort_values("timestamp")

    if asset_history.empty:
        return jsonify(
            {
                "error": (
                    f"No telemetry found for {asset_id}."
                )
            }
        ), 404

    try:
        model, config = load_model_bundle(
            PM_DIR
        )

        result = predict_failure(
            asset_id=asset_id,
            current_telemetry=current_telemetry,
            telemetry_history=asset_history,
            metadata=metadata,
            model=model,
            threshold=float(
                config["threshold"]
            ),
            numeric_features=config[
                "numeric_features"
            ],
            categorical_features=config[
                "categorical_features"
            ],
        )

        return jsonify(result)

    except Exception as exc:
        app.logger.exception(
            "Predictive-maintenance inference failed"
        )

        return jsonify(
            {
                "error": (
                    "Predictive-maintenance "
                    "inference failed."
                ),
                "details": str(exc),
            }
        ), 500


# ------------------------------------------------------------------
# Task 3 — Energy Forecasting
# ------------------------------------------------------------------

@app.get("/api/energy/forecast/<building_id>")
def energy_forecast_api(building_id):

    model_path = (
        ENERGY_DIR
        / "xgboost_energy_models.joblib"
    )
    config_path = (
        ENERGY_DIR / "model_config.json"
    )

    if not model_path.exists():
        return missing_artifact(model_path)

    if not config_path.exists():
        return missing_artifact(config_path)

    try:
        models = load_energy_models(
            model_path
        )
        config = load_energy_config(
            config_path
        )

        if building_id not in models:
            return jsonify(
                {
                    "error": (
                        f"No saved energy model "
                        f"for {building_id}."
                    ),
                    "available_buildings": list(
                        models.keys()
                    ),
                }
            ), 404

        # Match Task 3: Energy Meter is the
        # building-level energy target.
        energy_assets = metadata[
            metadata["asset_type"]
            == "Energy Meter"
        ][
            [
                "asset_id",
                "building_id",
            ]
        ]

        energy_history = (
            telemetry
            .merge(
                energy_assets,
                on=[
                    "asset_id",
                    "building_id",
                ],
                how="inner",
            )
            [
                [
                    "timestamp",
                    "building_id",
                    "power_consumption",
                ]
            ]
        )

        forecast = forecast_energy(
            building_id=building_id,
            hourly_history=energy_history,
            model=models[building_id],
            horizon=24,
        )

        return jsonify(
            {
                "building_id": building_id,
                "horizon_hours": 24,
                "forecast": (
                    forecast
                    .assign(
                        timestamp=lambda x:
                            x["timestamp"]
                            .astype(str)
                    )
                    .to_dict(
                        orient="records"
                    )
                ),
                "model": config.get(
                    "model_name",
                    "XGBoost",
                ),
            }
        )

    except Exception as exc:
        return jsonify(
            {
                "error": (
                    "Energy forecasting "
                    "failed."
                ),
                "details": str(exc),
            }
        ), 500


# ------------------------------------------------------------------
# Task 4 — Anomaly Detection
# ------------------------------------------------------------------

@app.get("/api/anomalies")
def anomalies():

    result_path = (
        ANOMALY_DIR
        / "anomaly_results.csv"
    )

    if not result_path.exists():
        return missing_artifact(
            result_path
        )

    try:
        anomaly_df = load_anomaly_results(
            result_path
        )

        severity = request.args.get(
            "severity"
        )
        asset_type = request.args.get(
            "asset_type"
        )
        asset_id = request.args.get(
            "asset_id"
        )

        result = get_anomalies(
            anomaly_df,
            severity=severity,
            asset_type=asset_type,
            asset_id=asset_id,
        )

        return jsonify(
            result.where(
                result.notna(),
                None,
            ).to_dict(
                orient="records"
            )
        )

    except Exception as exc:
        return jsonify(
            {
                "error": (
                    "Could not load "
                    "anomaly results."
                ),
                "details": str(exc),
            }
        ), 500


@app.get("/api/anomalies/summary")
def anomaly_summary_api():

    result_path = (
        ANOMALY_DIR
        / "anomaly_results.csv"
    )

    if not result_path.exists():
        return missing_artifact(
            result_path
        )

    anomaly_df = load_anomaly_results(
        result_path
    )

    return jsonify(
        anomaly_summary(
            anomaly_df
        )
    )


# ------------------------------------------------------------------
# Task 5 — Connectivity
# ------------------------------------------------------------------

@app.get("/api/connectivity/<asset_id>")
def asset_connectivity(asset_id):
    return jsonify(
        connected_assets(
            graph,
            asset_id,
        )
    )


@app.get("/api/downstream/<asset_id>")
def asset_downstream(asset_id):
    return jsonify(
        downstream_assets(
            graph,
            asset_id,
        )
    )


@app.get("/api/sites/<site_id>/assets")
def site_assets(site_id):

    result = assets_under_site(
        metadata,
        site_id,
    )

    return jsonify(
        result.where(
            result.notna(),
            None,
        ).to_dict(
            orient="records"
        )
    )


@app.get("/api/isolated-assets")
def isolated():

    result = isolated_assets(
        graph,
        metadata,
    )

    return jsonify(
        result.where(
            result.notna(),
            None,
        ).to_dict(
            orient="records"
        )
    )


# ------------------------------------------------------------------
# Bonus GraphQL
# ------------------------------------------------------------------

@app.post("/graphql")
def graphql():

    if graphql_sync is None:
        return jsonify(
            {
                "error": (
                    "graphql-core is not installed."
                )
            }
        ), 500

    if graphql_schema is None:
        return jsonify(
            {
                "error": (
                    "GraphQL schema could not "
                    "be created."
                )
            }
        ), 500

    payload = (
        request.get_json(
            silent=True
        )
        or {}
    )

    query = payload.get("query")
    variables = payload.get(
        "variables"
    )

    if not query:
        return jsonify(
            {
                "error": (
                    "Missing GraphQL query."
                )
            }
        ), 400

    result = graphql_sync(
        graphql_schema,
        query,
        context_value={
            "services": graph_services
        },
        variable_values=variables,
    )

    response = {}

    if result.errors:
        response["errors"] = [
            str(error)
            for error in result.errors
        ]

    if result.data is not None:
        response["data"] = result.data

    return jsonify(response), (
        400
        if result.errors
        else 200
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
    )
