"""Streamlit dashboard for the Nectar Data Scientist Challenge.

The notebooks train/save the artifacts. This dashboard only loads
those saved artifacts and performs inference/visualization.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import requests
import streamlit as st
import sys 

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_data
# Predictive maintenance is intentionally called through the Flask
# deployment API rather than running the model directly in Streamlit.

from src.energy_service import forecast_energy
from src.anomaly_service import (
    load_anomaly_results,
    anomaly_summary,
    get_anomalies,
)
from src.graph_service import (
    build_graph,
    connected_assets,
    downstream_assets,
    assets_under_site,
    isolated_assets,
)


# ------------------------------------------------------------
# Project paths
# ------------------------------------------------------------

DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

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


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Nectar Intelligent Facilities",
    page_icon="🏢",
    layout="wide",
)

# ------------------------------------------------------------
# Dashboard styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    [data-testid="stMetric"] {
        background: linear-gradient(
            135deg,
            rgba(31, 41, 55, 0.85),
            rgba(17, 24, 39, 0.85)
        );
        border: 1px solid rgba(148, 163, 184, 0.15);
        padding: 16px;
        border-radius: 14px;
    }

    .section-card {
        padding: 18px 20px;
        border-radius: 14px;
        background: rgba(31, 41, 55, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.12);
        margin-bottom: 12px;
    }

    .hero {
        padding: 22px 26px;
        border-radius: 18px;
        background: linear-gradient(
            120deg,
            rgba(30, 64, 175, 0.25),
            rgba(15, 118, 110, 0.20)
        );
        border: 1px solid rgba(96, 165, 250, 0.18);
        margin-bottom: 20px;
    }

    .small-muted {
        color: #94a3b8;
        font-size: 0.9rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.14);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Cached data/model loading
# ------------------------------------------------------------

@st.cache_data
def load_project_data():
    return load_data(DATA_DIR)


@st.cache_data
def load_pm_config():
    with open(
        PM_DIR / "model_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


@st.cache_resource
def load_energy_models():
    return joblib.load(
        ENERGY_DIR / "xgboost_energy_models.joblib"
    )


@st.cache_data
def load_energy_config():
    with open(
        ENERGY_DIR / "model_config.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


@st.cache_data
def load_anomaly_data():
    return load_anomaly_results(
        ANOMALY_DIR / "anomaly_results.csv"
    )


@st.cache_data
def load_connectivity_data():
    metadata = pd.read_csv(
        DATA_DIR / "asset_metadata.csv"
    )
    connectivity = pd.read_csv(
        CONNECTIVITY_DIR / "asset_connectivity.csv"
    )
    return metadata, connectivity


# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def artifact_exists(path):
    return Path(path).exists()


def artifact_message(path):
    st.warning(
        "The required artifact has not been created yet.\n\n"
        f"Run the corresponding dashboard-ready notebook first:\n"
        f"`{path}`"
    )


# ------------------------------------------------------------
# Load raw data
# ------------------------------------------------------------

try:
    telemetry, metadata, connectivity = load_project_data()
except Exception as exc:
    st.error(f"Could not load project data: {exc}")
    st.stop()


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("Nectar Facilities")

page = st.sidebar.radio(
    "Dashboard",
    [
        "Overview",
        "Task 1 - EDA",
        "Task 2 - Predictive Maintenance",
        "Task 3 - Energy Forecasting",
        "Task 4 - Anomaly Detection",
        "Task 5 - Connectivity",
        "GraphQL",
    ],
)


st.markdown(
    """
    <div class="hero">
        <h1 style="margin-bottom:6px;">
            🏢 Intelligent Facilities Analytics
        </h1>
        <div class="small-muted">
            IoT telemetry • Predictive maintenance • Energy forecasting
            • Anomaly detection • Multi-asset connectivity
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="section-card">
            <h2 style="margin-top:0;">
                Nectar Intelligent Facilities Analytics
            </h2>
            <h4 style="margin-top:4px; color:#94a3b8;">
                End-to-End IoT Analytics, Machine Learning, Forecasting,
                Anomaly Detection & Connectivity
            </h4>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Executive Summary")

    st.markdown(
        """
        This project develops an end-to-end analytics solution for
        connected commercial-facility assets. The workflow uses IoT
        telemetry, asset metadata, and asset connectivity information
        to understand equipment behavior, predict near-term failures,
        forecast building energy consumption, detect abnormal behavior,
        and analyze dependency relationships.

        """
    )

    st.subheader("Project Summary")

    overview_df = pd.DataFrame(
        [
            {
                "Area": "Dataset",
                "Key Outcome": (
                    "259,200 telemetry readings | 30 assets | "
                    "6 buildings | 3 sites | Jan–Mar 2026"
                ),
            },
            {
                "Area": "Task 1",
                "Key Outcome": (
                    "EDA, data-quality validation, temporal/asset behavior "
                    "and failure/energy factor analysis"
                ),
            },
            {
                "Area": "Task 2",
                "Key Outcome": (
                    "24-hour failure prediction using a saved XGBoost pipeline"
                ),
            },
            {
                "Area": "Task 3",
                "Key Outcome": (
                    "Building-level energy forecasting using hourly "
                    "recursive models"
                ),
            },
            {
                "Area": "Task 4",
                "Key Outcome": (
                    "Hybrid anomaly detection using five complementary signals"
                ),
            },
            {
                "Area": "Task 5",
                "Key Outcome": (
                    "Directed asset graph, failure-impact analysis and "
                    "connectivity data-quality checks"
                ),
            },
            {
                "Area": "Application",
                "Key Outcome": (
                    "Streamlit dashboard + Flask REST API + GraphQL interface"
                ),
            },
        ]
    )

    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Area": st.column_config.TextColumn(
                "Area",
                width="small",
            ),
            "Key Outcome": st.column_config.TextColumn(
                "Key Outcome",
                width="large",
            ),
        },
    )

    st.subheader("Dataset & Domain")

    st.markdown(
        """
        - **Telemetry records** contain timestamp, site/building/asset
          identifiers, temperature, humidity, pressure, vibration,
          power consumption, occupancy, operating mode and fault flag.

        - **Asset metadata** provides asset type, manufacturer,
          installation date, capacity and parent-asset relationships.

        - **Connectivity data** provides source asset, target asset,
          connection type and relationship strength.

        - The application treats **telemetry as asset behavior,
          metadata as asset context, and connectivity as dependency
          information**.
        """
    )

    st.subheader("Overall Architecture")

    st.markdown(
        """
        <div class="section-card">
            <div style="font-size:1.05rem; text-align:center; padding:8px;">
                Raw data
                <b>→</b>
                Jupyter notebooks
                <b>→</b>
                saved models / results / graphs
                <b>→</b>
                service layer
                <b>→</b>
                Flask REST / GraphQL
                <b>→</b>
                Streamlit dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.info(
        "The dashboard does not retrain models; it consumes the saved "
        "artifacts produced by the notebooks."
    )

    st.subheader("Artifact Status")

    status = {
        "Predictive Maintenance Model":
            PM_DIR / "predictive_maintenance_model.joblib",
        "Energy Forecasting Models":
            ENERGY_DIR / "xgboost_energy_models.joblib",
        "Anomaly Detection Models":
            ANOMALY_DIR / "anomaly_detection_models.joblib",
        "Anomaly Results":
            ANOMALY_DIR / "anomaly_results.csv",
        "Connectivity Data":
            CONNECTIVITY_DIR / "asset_connectivity.csv",
    }

    status_df = pd.DataFrame(
        [
            {
                "Artifact": name,
                "Status": (
                    "Available"
                    if artifact_exists(path)
                    else "Run notebook"
                ),
            }
            for name, path in status.items()
        ]
    )

    st.dataframe(
        status_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# TASK 1
# ============================================================

elif page == "Task 1 - EDA":

    st.subheader("📊 Exploratory Data Analysis")
    st.caption(
        "Curated evidence from the Task 1 notebook, selected to directly "
        "cover the challenge requirements: distributions, data quality, "
        "temporal behavior, site and asset-type differences, failure signals, "
        "and energy drivers."
    )

    figures_dir = OUTPUTS_DIR / "eda" / "dashboard_figures"
    tables_dir = OUTPUTS_DIR / "eda" / "dashboard_tables"

    # --------------------------------------------------------
    # Dataset summary
    # --------------------------------------------------------
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Telemetry Records", f"{len(telemetry):,}")
    e2.metric("Assets", f"{metadata['asset_id'].nunique():,}")
    e3.metric("Buildings", f"{metadata['building_id'].nunique():,}")
    e4.metric("Sites", f"{metadata['site_id'].nunique():,}")

    st.divider()

    # --------------------------------------------------------
    # Key findings
    # --------------------------------------------------------
    st.subheader("Key Findings & Evidence")

    finding_files = [
        ("01_data_quality.png", "01_data_quality.csv",
         "1. Data Quality",
         "Missingness is limited and the telemetry stream has a reliable sampling structure."),
        ("02_temporal_energy.png", "02_temporal_energy.csv",
         "2. Temporal Pattern",
         "Energy demand changes strongly by hour of day, supporting time-aware forecasting."),
        ("03_site_behavior.png", "03_site_behavior.csv",
         "3. Asset Behavior Across Sites",
         "Site-level energy differences are measurable but should be interpreted with building and operating context."),
        ("04_asset_type_performance.png", "04_asset_type_performance.csv",
         "4. Performance Across Asset Types",
         "Different equipment classes operate in different physical regimes, so health baselines should be asset-type specific."),
        ("05_failure_signals.png", "05_failure_signals.csv",
         "5. Equipment Failure Signals",
         "Fault readings show higher temperature, vibration and power than normal readings."),
        ("06_prefault_behavior.png", "06_prefault_behavior.csv",
         "6. Pre-Fault Behavior",
         "Vibration changes substantially within 24 hours before recorded faults."),
        ("07_energy_drivers.png", "07_energy_drivers.csv",
         "7. Energy Consumption Drivers",
         "Occupancy and temperature show the strongest positive numeric associations with energy."),
    ]

    finding_text = {
        "1. Data Quality": (
            "**Key observation:** humidity 1.028%, temperature 0.984%, "
            "vibration 0.788%, and pressure 0.772% are missing. "
            "No duplicate telemetry rows or non-15-minute intervals were identified."
        ),
        "2. Temporal Pattern": (
            "**Key observation:** average energy rises sharply during the "
            "operating period and falls again after the operating window. "
            "Time-of-day is therefore important for energy modeling."
        ),
        "3. Asset Behavior Across Sites": (
            "**Key observation:** SITE_02 has the highest average energy "
            "reading (29.712 kWh), while SITE_03 has the lowest (28.768 kWh)."
        ),
        "4. Performance Across Asset Types": (
            "**Key observation:** Energy Meters average 62.692 kWh, Chillers "
            "42.486 kWh, AHUs 16.892 kWh and Pumps 7.600 kWh. "
            "Pumps also have the highest average vibration (0.249) and pressure (4.502)."
        ),
        "5. Equipment Failure Signals": (
            "**Key observation:** fault readings have higher mean temperature "
            "(24.084 vs 22.513), vibration (0.540 vs 0.135), and power "
            "(31.734 vs 29.310). Vibration has the strongest simple association "
            "with the fault flag (≈0.162)."
        ),
        "6. Pre-Fault Behavior": (
            "**Key observation:** 5,112 readings occur within 24 hours before "
            "a recorded fault. Mean vibration increases by about 75.65% in this "
            "pre-fault comparison, making recent vibration behavior an important "
            "candidate predictive signal."
        ),
        "7. Energy Consumption Drivers": (
            "**Key observation:** the notebook reports positive associations "
            "between occupancy and energy (≈0.470) and temperature and energy "
            "(≈0.317). Energy analysis should also account for operating mode, "
            "asset type, building and time."
        ),
    }

    for fig_name, table_name, title, description in finding_files:
        st.markdown(f"### {title}")
        st.caption(description)

        left, right = st.columns([1.45, 1])

        with left:
            fig_path = figures_dir / fig_name
            if fig_path.exists():
                st.image(
                    str(fig_path),
                    use_container_width=True,
                )
            else:
                st.warning(
                    f"{fig_name} not found. Run the updated Task 1 notebook."
                )

        with right:
            st.markdown(finding_text[title])

            table_path = tables_dir / table_name
            if table_path.exists():
                try:
                    table_df = pd.read_csv(table_path)
                    st.dataframe(
                        table_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                except Exception as exc:
                    st.error(f"Could not read {table_name}: {exc}")
            else:
                st.info(
                    f"{table_name} not found. Run the updated Task 1 notebook."
                )

        st.divider()

    # --------------------------------------------------------
    # Compact statistical summary
    # --------------------------------------------------------
    st.subheader("EDA Summary")

    summary_path = tables_dir / "00_eda_findings.csv"
    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "area": "Area",
                "key_finding": "Key Finding",
                "business_insight": "Business Insight",
            },
        )
    else:
        st.info(
            "Run the updated Task 1 notebook to create the dashboard summary."
        )

    # --------------------------------------------------------
    # Optional detailed exploratory plots
    # --------------------------------------------------------
    with st.expander("View Supporting Distribution / Exploratory Plots"):
        st.caption(
            "These are the broader exploratory plots generated by the notebook. "
            "The sections above contain the curated evidence used to explain the challenge requirements."
        )

        if (OUTPUTS_DIR / "eda" / "figures").exists():
            image_files = sorted(
                list((OUTPUTS_DIR / "eda" / "figures").glob("*.png"))
                + list((OUTPUTS_DIR / "eda" / "figures").glob("*.jpg"))
                + list((OUTPUTS_DIR / "eda" / "figures").glob("*.jpeg"))
            )

            if image_files:
                for start_idx in range(0, len(image_files), 2):
                    cols = st.columns(2)
                    for col, image in zip(
                        cols,
                        image_files[start_idx:start_idx + 2],
                    ):
                        with col:
                            st.image(
                                str(image),
                                caption=image.stem.replace(
                                    "_", " "
                                ).title(),
                                use_container_width=True,
                            )
            else:
                st.info("No supporting EDA figures found.")

# ============================================================
# TASK 2
# ============================================================

elif page == "Task 2 - Predictive Maintenance":

    st.subheader("Predictive Maintenance")

    model_path = (
        PM_DIR
        / "predictive_maintenance_model.joblib"
    )

    config_path = PM_DIR / "model_config.json"

    if not model_path.exists() or not config_path.exists():
        artifact_message(model_path)
        st.stop()

    config = load_pm_config()

    # --------------------------------------------------------
    # Deployment status
    # --------------------------------------------------------

    backend_url = st.text_input(
        "Prediction API Endpoint",
        value="http://127.0.0.1:5000/predict_failure",
        key="pm_api_url",
    )

    # Construct the status endpoint from the selected prediction URL.
    if backend_url.endswith("/predict_failure"):
        status_url = (
            backend_url[
                : -len("/predict_failure")
            ]
            + "/predict_failure/status"
        )
    elif backend_url.endswith("/api/predict_failure"):
        status_url = (
            backend_url[
                : -len("/api/predict_failure")
            ]
            + "/predict_failure/status"
        )
    else:
        status_url = (
            backend_url.rstrip("/")
            + "/predict_failure/status"
        )

    try:
        status_response = requests.get(
            status_url,
            timeout=5,
        )

        if status_response.ok:
            status = status_response.json()

            if status.get("status") == "ready":
                st.success(
                    "Prediction API is running and the saved model is loaded."
                )
            else:
                st.warning(
                    "Prediction API is reachable, but the model is not ready."
                )
        else:
            st.warning(
                "Prediction API is not ready. Start Flask with: "
                "`python backend/app.py`"
            )

    except requests.exceptions.RequestException:
        st.warning(
            "Prediction API is not running. Start Flask with:\n\n"
            "`python backend/app.py`"
        )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model",
        config.get(
            "model_name",
            "XGBoost",
        ),
    )

    c2.metric(
        "Alert Threshold",
        f"{float(config['threshold']):.2f}",
    )

    c3.metric(
        "Prediction Horizon",
        "24 Hours",
    )

    st.caption(
        "This page sends telemetry to the Flask prediction API. "
        "The API loads the saved Task 2 pipeline and returns the "
        "24-hour failure probability."
    )

    # --------------------------------------------------------
    # Select asset
    # --------------------------------------------------------

    asset_ids = sorted(
        metadata["asset_id"]
        .astype(str)
        .unique()
    )

    selected_asset = st.selectbox(
        "Select Asset",
        asset_ids,
        key="pm_asset",
    )

    asset_history = telemetry[
        telemetry["asset_id"].astype(str)
        == selected_asset
    ].sort_values("timestamp").copy()

    if asset_history.empty:
        st.warning(
            "No telemetry found for this asset."
        )
        st.stop()

    # --------------------------------------------------------
    # Choose which telemetry point to score
    # --------------------------------------------------------

    prediction_mode = st.radio(
        "Prediction Input",
        [
            "Latest Reading",
            "Highest-Risk Test Observation",
            "Select Historical Reading",
        ],
        horizontal=True,
    )

    selected_row = None

    if prediction_mode == "Latest Reading":

        selected_row = asset_history.iloc[-1]

    elif prediction_mode == "Highest-Risk Test Observation":

        test_predictions_path = (
            PM_DIR / "test_predictions.csv"
        )

        if not test_predictions_path.exists():

            st.info(
                "test_predictions.csv is not available. "
                "Run Task 2 notebook first."
            )

            selected_row = asset_history.iloc[-1]

        else:

            test_predictions = pd.read_csv(
                test_predictions_path
            )

            test_predictions["asset_id"] = (
                test_predictions["asset_id"]
                .astype(str)
            )

            asset_test = test_predictions[
                test_predictions["asset_id"]
                == selected_asset
            ].copy()

            if asset_test.empty:

                st.info(
                    "This asset has no test-period prediction row. "
                    "Using its latest reading."
                )

                selected_row = asset_history.iloc[-1]

            else:

                best_row = asset_test.sort_values(
                    "failure_probability",
                    ascending=False,
                ).iloc[0]

                timestamp = pd.to_datetime(
                    best_row["timestamp"]
                )

                matching = asset_history[
                    pd.to_datetime(
                        asset_history["timestamp"]
                    ) == timestamp
                ]

                if matching.empty:

                    st.info(
                        "Saved test timestamp was not found "
                        "in telemetry. Using latest reading."
                    )

                    selected_row = asset_history.iloc[-1]

                else:

                    selected_row = matching.iloc[-1]

                    st.info(
                        "Using the highest-risk saved test-period "
                        "observation for this asset."
                    )

    else:

        timestamps = pd.to_datetime(
            asset_history["timestamp"]
        ).tolist()

        selected_timestamp = st.select_slider(
            "Historical Timestamp",
            options=timestamps,
            value=timestamps[-1],
            format_func=lambda x: x.strftime(
                "%Y-%m-%d %H:%M"
            ),
        )

        selected_rows = asset_history[
            pd.to_datetime(
                asset_history["timestamp"]
            ) == selected_timestamp
        ]

        selected_row = selected_rows.iloc[-1]

    # --------------------------------------------------------
    # Call the deployed Flask API
    # --------------------------------------------------------

    current_telemetry = selected_row.to_dict()

    # Convert pandas timestamps/numpy values into JSON-safe values.
    current_telemetry["timestamp"] = str(
        pd.to_datetime(
            current_telemetry["timestamp"]
        )
    )

    for key, value in list(
        current_telemetry.items()
    ):
        if pd.isna(value):
            current_telemetry[key] = None
        elif hasattr(value, "item"):
            current_telemetry[key] = value.item()

    payload = {
        "asset_id": selected_asset,
        "current_telemetry": current_telemetry,
    }

    if st.button(
        "Run Prediction API",
        type="primary",
        use_container_width=True,
    ):

        try:

            response = requests.post(
                backend_url,
                json=payload,
                timeout=30,
            )

            response.raise_for_status()

            result = response.json()

            if "error" in result:

                st.error(
                    result["error"]
                )

                if result.get("details"):
                    st.code(
                        result["details"]
                    )

            else:

                probability = float(
                    result["failure_probability"]
                )

                threshold = float(
                    result["threshold"]
                )

                p1, p2, p3, p4 = st.columns(4)

                p1.metric(
                    "Failure Probability",
                    f"{probability:.2%}",
                )

                p2.metric(
                    "Alert Threshold",
                    f"{threshold:.0%}",
                )

                p3.metric(
                    "Prediction",
                    (
                        "Maintenance Alert"
                        if result["predicted_failure"]
                        else "No Alert"
                    ),
                )

                p4.metric(
                    "Risk Level",
                    result["risk_level"],
                )

                st.success(
                    "Prediction generated by the deployed Flask API."
                )

                st.dataframe(
                    pd.DataFrame([result]),
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander(
                    "API Request Payload"
                ):
                    st.json(payload)

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the Flask prediction API."
            )

            st.code(
                "python backend/app.py",
                language="powershell",
            )

        except requests.exceptions.HTTPError as exc:

            st.error(
                f"Prediction API returned an HTTP error: {exc}"
            )

            try:
                st.json(response.json())
            except Exception:
                pass

        except Exception as exc:

            st.error(
                f"Prediction could not be generated: {exc}"
            )

    # --------------------------------------------------------
    # Saved test-period summary
    # --------------------------------------------------------

    summary_path = (
        PM_DIR / "asset_alert_summary.csv"
    )

    if summary_path.exists():

        st.subheader(
            "Test-Period Asset Risk Summary"
        )

        summary = pd.read_csv(
            summary_path
        )

        st.dataframe(
            summary.head(20),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# TASK 3
# ============================================================

elif page == "Task 3 - Energy Forecasting":

    st.subheader("24-Hour Energy Forecast")

    models_path = (
        ENERGY_DIR
        / "xgboost_energy_models.joblib"
    )

    config_path = (
        ENERGY_DIR / "model_config.json"
    )

    if not models_path.exists() or not config_path.exists():
        artifact_message(models_path)
        st.stop()

    models = load_energy_models()
    config = load_energy_config()

    building_ids = sorted(
        metadata["building_id"].unique()
    )

    selected_building = st.selectbox(
        "Select Building",
        building_ids,
    )

    # Use only the building's Energy Meter,
    # matching Task 3 target construction.
    energy_assets = metadata[
        metadata["asset_type"].eq("Energy Meter")
    ][["asset_id", "building_id"]]

    energy_history = (
        telemetry
        .merge(
            energy_assets,
            on=["asset_id", "building_id"],
            how="inner",
        )
        [
            [
                "timestamp",
                "building_id",
                "power_consumption",
            ]
        ]
        .rename(
            columns={
                "power_consumption":
                    "power_consumption"
            }
        )
    )

    try:
        forecast = forecast_energy(
            building_id=selected_building,
            hourly_history=energy_history,
            model=models[selected_building],
            horizon=24,
        )

        st.metric(
            "Forecast Horizon",
            "Next 24 Hours",
        )

        st.dataframe(
            forecast,
            use_container_width=True,
            hide_index=True,
        )

        st.line_chart(
            forecast.set_index("timestamp")[
                ["predicted_energy"]
            ]
        )

        result_path = (
            ENERGY_DIR
            / "exact_24h_results.csv"
        )

        if result_path.exists():
            results = pd.read_csv(result_path)

            st.subheader(
                "Notebook Evaluation Results"
            )

            st.dataframe(
                results[
                    results["Building"]
                    == selected_building
                ],
                use_container_width=True,
                hide_index=True,
            )

    except Exception as exc:
        st.error(
            f"Energy forecast could not be generated: {exc}"
        )


# ============================================================
# TASK 4
# ============================================================

elif page == "Task 4 - Anomaly Detection":

    st.subheader("Anomaly Detection")

    anomaly_path = (
        ANOMALY_DIR
        / "anomaly_results.csv"
    )

    if not anomaly_path.exists():
        artifact_message(anomaly_path)
        st.stop()

    anomaly_df = load_anomaly_data()

    summary = anomaly_summary(anomaly_df)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Watch", summary["Watch"])
    c2.metric("Anomaly", summary["Anomaly"])
    c3.metric("Critical", summary["Critical"])
    c4.metric(
        "Total Flagged",
        sum(
            [
                summary["Watch"],
                summary["Anomaly"],
                summary["Critical"],
            ]
        ),
    )

    severity = st.selectbox(
        "Severity",
        ["All", "Watch", "Anomaly", "Critical"],
    )

    asset_types = [
        "All"
    ] + sorted(
        anomaly_df["asset_type"]
        .dropna()
        .unique()
        .tolist()
    )

    asset_type = st.selectbox(
        "Asset Type",
        asset_types,
    )

    filtered = get_anomalies(
        anomaly_df,
        severity=None
        if severity == "All"
        else severity,
        asset_type=None
        if asset_type == "All"
        else asset_type,
    )

    st.dataframe(
        filtered.head(100),
        use_container_width=True,
        hide_index=True,
    )

    asset_summary_path = (
        ANOMALY_DIR
        / "asset_anomaly_summary.csv"
    )

    if asset_summary_path.exists():
        st.subheader(
            "Asset-Level Anomaly Summary"
        )

        st.dataframe(
            pd.read_csv(
                asset_summary_path
            ).head(20),
            use_container_width=True,
            hide_index=True,
        )



# ============================================================
# GRAPHQL BONUS
# ============================================================

elif page == "GraphQL":

    st.subheader("GraphQL - Connectivity Queries")

    st.caption(
        "Run the four GraphQL connectivity queries required "
        "by the challenge through the Flask backend."
    )

    backend_url = st.text_input(
        "GraphQL Endpoint",
        value="http://127.0.0.1:5000/graphql",
    )

    asset_ids = sorted(
        metadata["asset_id"].astype(str).unique()
    )

    site_ids = sorted(
        metadata["site_id"].astype(str).unique()
    )

    query_type = st.selectbox(
        "Query",
        [
            "Connected Assets",
            "Downstream Assets",
            "Assets Under Site",
            "Isolated Assets",
        ],
    )

    if query_type == "Connected Assets":

        asset_name = st.selectbox(
            "Asset",
            asset_ids,
        )

        query = """
        query($assetName: String!) {
          connectedAssets(assetName: $assetName) {
            asset_id
            direction
            connection_type
            relationship_strength
          }
        }
        """

        variables = {
            "assetName": asset_name
        }

    elif query_type == "Downstream Assets":

        asset_name = st.selectbox(
            "Asset",
            asset_ids,
        )

        query = """
        query($assetName: String!) {
          downstreamAssets(assetName: $assetName) {
            asset_id
            depth
            connection_type
            relationship_strength
          }
        }
        """

        variables = {
            "assetName": asset_name
        }

    elif query_type == "Assets Under Site":

        site_id = st.selectbox(
            "Site",
            site_ids,
        )

        query = """
        query($siteId: String!) {
          assetsUnderSite(siteId: $siteId) {
            asset_id
            site_id
            building_id
            asset_name
            asset_type
          }
        }
        """

        variables = {
            "siteId": site_id
        }

    else:

        query = """
        query {
          isolatedAssets {
            asset_id
            asset_name
            site_id
            building_id
            asset_type
          }
        }
        """

        variables = {}

    with st.expander("GraphQL Query"):
        st.code(
            query,
            language="graphql",
        )

    if st.button(
        "Run GraphQL Query",
        type="primary",
    ):

        try:

            response = requests.post(
                backend_url,
                json={
                    "query": query,
                    "variables": variables,
                },
                timeout=15,
            )

            response.raise_for_status()

            payload = response.json()

            if payload.get("errors"):

                st.error(
                    "GraphQL returned an error."
                )

                st.json(
                    payload["errors"]
                )

            else:

                result = payload.get(
                    "data",
                    {},
                )

                st.success(
                    "GraphQL query executed successfully."
                )

                st.json(result)

                if result:

                    result_value = next(
                        iter(result.values())
                    )

                    if isinstance(
                        result_value,
                        list,
                    ):

                        st.dataframe(
                            pd.DataFrame(
                                result_value
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the Flask backend. "
                "Start it first with:\n\n"
                "python backend/app.py"
            )

        except Exception as exc:

            st.error(
                f"GraphQL request failed: {exc}"
            )


# ============================================================
# TASK 5
# ============================================================

elif page == "Task 5 - Connectivity":

    st.subheader("🔗 Multi-Asset Connectivity")
    st.caption(
        "Connectivity views use the relationships saved by the "
        "Task 5 notebook."
    )

    connectivity_path = CONNECTIVITY_DIR / "asset_connectivity.csv"

    if not connectivity_path.exists():
        artifact_message(connectivity_path)
        st.stop()

    graph_metadata, graph_connectivity = load_connectivity_data()
    graph = build_graph(graph_connectivity)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Assets", graph.number_of_nodes())
    c2.metric("Connections", graph.number_of_edges())
    c3.metric(
        "Isolated Assets",
        len(isolated_assets(graph, graph_metadata)),
    )
    c4.metric(
        "Sites",
        graph_metadata["site_id"].nunique(),
    )

    st.divider()

    graph_tab, hierarchy_tab, impact_tab, analysis_tab = st.tabs(
        [
            "🌐 Connectivity Graph",
            "🏢 Asset Hierarchy",
            "⚠️ Failure Impact",
            "📋 Connectivity Analysis",
        ]
    )

    with graph_tab:

        graph_image = CONNECTIVITY_DIR / "asset_connectivity_network.png"

        if graph_image.exists():
            st.image(
                str(graph_image),
                caption="Asset Connectivity Network — generated by Task 5 notebook",
                use_container_width=True,
            )
        else:
            st.warning(
                "Connectivity graph image not found. "
                "Re-run the updated Task 5 notebook."
            )

        st.subheader("Explore Connections")

        asset_ids = sorted(
            graph_metadata["asset_id"].astype(str).unique()
        )

        selected_asset = st.selectbox(
            "Select Asset",
            asset_ids,
            key="connectivity_asset",
        )

        t1, t2 = st.tabs(
            [
                "Connected Assets",
                "Downstream Assets",
            ]
        )

        with t1:
            result = pd.DataFrame(
                connected_assets(
                    graph,
                    selected_asset,
                )
            )

            if result.empty:
                st.info("No direct connections found.")
            else:
                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                )

        with t2:
            result = pd.DataFrame(
                downstream_assets(
                    graph,
                    selected_asset,
                )
            )

            if result.empty:
                st.info("No downstream assets found.")
            else:
                st.dataframe(
                    result,
                    use_container_width=True,
                    hide_index=True,
                )

    with hierarchy_tab:

        hierarchy_image = (
            CONNECTIVITY_DIR / "asset_hierarchy.png"
        )

        if hierarchy_image.exists():
            st.image(
                str(hierarchy_image),
                caption="Asset Hierarchy — generated by Task 5 notebook",
                use_container_width=True,
            )
        else:
            st.warning(
                "Hierarchy graph image not found. "
                "Re-run the updated Task 5 notebook."
            )

        sites = sorted(
            graph_metadata["site_id"].astype(str).unique()
        )

        selected_site = st.selectbox(
            "Select Site",
            sites,
            key="hierarchy_site",
        )

        site_result = assets_under_site(
            graph_metadata,
            selected_site,
        )

        st.dataframe(
            site_result,
            use_container_width=True,
            hide_index=True,
        )

    with impact_tab:

        impact_image = (
            CONNECTIVITY_DIR / "failure_propagation.png"
        )

        if impact_image.exists():
            st.image(
                str(impact_image),
                caption="Potential Downstream Impact — generated by Task 5 notebook",
                use_container_width=True,
            )
        else:
            st.warning(
                "Failure propagation graph image not found. "
                "Re-run the updated Task 5 notebook."
            )

        st.info(
            "The graph shows potential downstream impact based on "
            "the directed connectivity relationships. It does not "
            "guarantee physical failure."
        )

    with analysis_tab:

        summary_path = (
            CONNECTIVITY_DIR / "connectivity_summary.csv"
        )
        criticality_path = (
            CONNECTIVITY_DIR / "criticality.csv"
        )
        dq_path = (
            CONNECTIVITY_DIR / "data_quality.csv"
        )

        if summary_path.exists():
            st.subheader("Connectivity Summary")
            st.dataframe(
                pd.read_csv(summary_path),
                use_container_width=True,
                hide_index=True,
            )

        if criticality_path.exists():
            st.subheader("Criticality Indicators")
            st.dataframe(
                pd.read_csv(criticality_path).head(15),
                use_container_width=True,
                hide_index=True,
            )

        if dq_path.exists():
            st.subheader("Data Quality")
            st.dataframe(
                pd.read_csv(dq_path),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Isolated Assets")
        st.dataframe(
            isolated_assets(
                graph,
                graph_metadata,
            ),
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# GRAPHQL BONUS
# ============================================================

elif page == "GraphQL":

    st.subheader("🚀 GraphQL Connectivity API")
    

    backend_url = st.text_input(
        "Backend GraphQL Endpoint",
        value="http://127.0.0.1:5000/graphql",
    )

    # GraphQL uses human-readable asset_name, not asset_id.
    asset_names = sorted(
        metadata["asset_name"].dropna().astype(str).unique()
    )

    site_ids = sorted(
        metadata["site_id"].dropna().astype(str).unique()
    )

    query_type = st.selectbox(
        "Select GraphQL Query",
        [
            "Connected Assets",
            "Downstream Assets",
            "Assets Under Site",
            "Isolated Assets",
        ],
    )

    if query_type == "Connected Assets":

        selected_name = st.selectbox(
            "Asset Name",
            asset_names,
            key="graphql_connected_asset",
        )

        query = """
query($assetName: String!) {
  connectedAssets(assetName: $assetName) {
    asset_id
    direction
    connection_type
    relationship_strength
  }
}
"""

        variables = {
            "assetName": selected_name
        }

    elif query_type == "Downstream Assets":

        selected_name = st.selectbox(
            "Asset Name",
            asset_names,
            key="graphql_downstream_asset",
        )

        query = """
query($assetName: String!) {
  downstreamAssets(assetName: $assetName) {
    asset_id
    depth
    connection_type
    relationship_strength
  }
}
"""

        variables = {
            "assetName": selected_name
        }

    elif query_type == "Assets Under Site":

        selected_site = st.selectbox(
            "Site",
            site_ids,
            key="graphql_site",
        )

        query = """
query($siteId: String!) {
  assetsUnderSite(siteId: $siteId) {
    asset_id
    asset_name
    asset_type
    site_id
    building_id
  }
}
"""

        variables = {
            "siteId": selected_site
        }

    else:

        query = """
query {
  isolatedAssets {
    asset_id
    asset_name
    asset_type
    site_id
    building_id
  }
}
"""

        variables = {}

    with st.expander("View GraphQL Query"):
        st.code(
            query,
            language="graphql",
        )

    if st.button(
        "▶ Run GraphQL Query",
        type="primary",
        use_container_width=True,
    ):

        try:

            response = requests.post(
                backend_url,
                json={
                    "query": query,
                    "variables": variables,
                },
                timeout=15,
            )

            response.raise_for_status()

            payload = response.json()

            if payload.get("errors"):

                st.error("GraphQL returned an error.")
                st.json(payload["errors"])

            else:

                result = payload.get(
                    "data",
                    {},
                )

                st.success(
                    "GraphQL query executed successfully."
                )

                st.json(result)

                if result:

                    result_value = next(
                        iter(result.values())
                    )

                    if isinstance(
                        result_value,
                        list,
                    ):

                        st.subheader("Result")

                        st.dataframe(
                            pd.DataFrame(
                                result_value
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to the Flask backend."
            )

            st.code(
                "python backend/app.py",
                language="powershell",
            )

        except Exception as exc:

            st.error(
                f"GraphQL request failed: {exc}"
            )

