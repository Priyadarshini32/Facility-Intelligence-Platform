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
from src.prediction_service import (
    predict_failure,
    load_model as load_failure_model,
)
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


@st.cache_resource
def load_pm_model():
    return load_failure_model(
        PM_DIR / "predictive_maintenance_model.joblib"
    )


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

    st.subheader("Challenge Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Telemetry Records",
        f"{len(telemetry):,}",
    )

    c2.metric(
        "Assets",
        f"{metadata['asset_id'].nunique():,}",
    )

    c3.metric(
        "Buildings",
        f"{metadata['building_id'].nunique():,}",
    )

    c4.metric(
        "Sites",
        f"{metadata['site_id'].nunique():,}",
    )

    st.divider()

    st.markdown(
        """
        This dashboard connects the five challenge tasks:

        **EDA → Predictive Maintenance → Energy Forecasting →
        Anomaly Detection → Asset Connectivity**

        The notebooks are responsible for training and saving
        models. The dashboard loads those saved artifacts and
        performs inference or displays their results.
        """
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
        "Key EDA outputs generated by the Task 1 notebook. "
        "The dashboard displays the saved notebook artifacts."
    )

    figures_dir = OUTPUTS_DIR / "eda" / "figures"
    tables_dir = OUTPUTS_DIR / "eda" / "tables"

    # Quick dataset summary
    e1, e2, e3, e4 = st.columns(4)

    e1.metric("Telemetry Records", f"{len(telemetry):,}")
    e2.metric("Assets", f"{metadata['asset_id'].nunique():,}")
    e3.metric("Buildings", f"{metadata['building_id'].nunique():,}")
    e4.metric("Sites", f"{metadata['site_id'].nunique():,}")

    st.divider()

    if figures_dir.exists():

        image_files = sorted(
            list(figures_dir.glob("*.png"))
            + list(figures_dir.glob("*.jpg"))
            + list(figures_dir.glob("*.jpeg"))
        )

        if image_files:
            st.subheader("EDA Visualizations")

            # Show figures in a compact 2-column gallery
            for start_idx in range(0, len(image_files), 2):
                cols = st.columns(2)

                for col, image in zip(
                    cols,
                    image_files[start_idx:start_idx + 2]
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
            st.info("No EDA figures found.")

    if tables_dir.exists():

        table_files = sorted(
            list(tables_dir.glob("*.xlsx"))
            + list(tables_dir.glob("*.csv"))
        )

        if table_files:
            st.subheader("EDA Summary Tables")

            for table in table_files:

                with st.expander(
                    table.stem.replace(
                        "_", " "
                    ).title()
                ):

                    try:
                        if table.suffix.lower() == ".xlsx":
                            table_df = pd.read_excel(table)
                        else:
                            table_df = pd.read_csv(table)

                        st.dataframe(
                            table_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                    except Exception as exc:
                        st.error(
                            f"Could not read {table.name}: {exc}"
                        )

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

    model = load_pm_model()
    config = load_pm_config()

    c1, c2 = st.columns(2)

    c1.metric(
        "Model",
        config["model_name"],
    )

    c2.metric(
        "Alert Threshold",
        f"{config['threshold']:.2f}",
    )

    st.caption(
        "Prediction uses the saved preprocessing + model "
        "pipeline from the Task 2 notebook."
    )

    asset_ids = sorted(
        metadata["asset_id"].unique()
    )

    selected_asset = st.selectbox(
        "Select Asset",
        asset_ids,
    )

    asset_history = telemetry[
        telemetry["asset_id"] == selected_asset
    ].sort_values("timestamp")

    if asset_history.empty:
        st.warning("No telemetry found for this asset.")
    else:

        latest = asset_history.iloc[-1].copy()

        current_telemetry = latest.to_dict()

        try:
            result = predict_failure(
                asset_id=selected_asset,
                current_telemetry=current_telemetry,
                telemetry_history=asset_history.iloc[:-1],
                metadata=metadata,
                model=model,
                threshold=float(config["threshold"]),
            )

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Failure Probability",
                f"{result['failure_probability']:.2%}",
            )

            p2.metric(
                "Prediction",
                (
                    "Maintenance Alert"
                    if result["predicted_failure"]
                    else "No Alert"
                ),
            )

            p3.metric(
                "Latest Reading",
                str(latest["timestamp"]),
            )

            st.dataframe(
                pd.DataFrame([result]),
                use_container_width=True,
                hide_index=True,
            )

        except Exception as exc:
            st.error(
                f"Prediction could not be generated: {exc}"
            )

    summary_path = (
        PM_DIR / "asset_alert_summary.csv"
    )

    if summary_path.exists():
        st.subheader(
            "Test-Period Asset Risk Summary"
        )

        summary = pd.read_csv(summary_path)

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

