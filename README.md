# Nectar Intelligent Facilities Analytics

## 3-Line Description

An end-to-end Data Science solution for intelligent facility management using IoT telemetry, asset metadata, and asset connectivity data.  
The project covers EDA, 24-hour predictive maintenance, building energy forecasting, hybrid anomaly detection, and multi-asset connectivity/failure-impact analysis.  
A Streamlit dashboard and Flask/GraphQL backend expose the notebook-generated insights, saved ML models, forecasts, anomaly results, and connectivity analysis.

---

# 1. Project Overview

This project was developed for the Nectar Data Scientist Challenge.

The solution analyzes connected building assets such as:

- Chillers
- AHUs
- Pumps
- Energy Meters
- Environmental/IoT sensors

The project is designed around the following flow:

```text
Raw IoT Data
     |
     v
Jupyter Notebooks
     |
     +--> EDA
     +--> Predictive Maintenance
     +--> Energy Forecasting
     +--> Anomaly Detection
     +--> Connectivity Analysis
     |
     v
Saved Models / Results / Graphs
     |
     v
Flask REST + GraphQL API
     |
     v
Streamlit Dashboard
```

The dashboard does not retrain the models. The notebooks perform analysis/training and save the artifacts used by the application.

---

# 2. Main Tasks

## Task 1 — Exploratory Data Analysis

Analyzes:

- Sensor distributions
- Missing values
- Data quality
- Temporal patterns
- Site-level behavior
- Building-level behavior
- Asset-type behavior
- Operating modes
- Fault patterns
- Energy relationships
- Pre-fault behavior

Outputs are saved under:

```text
outputs/eda/
├── figures/
└── tables/
```

The Streamlit dashboard reads these saved notebook artifacts.

---

## Task 2 — Predictive Maintenance

Objective:

> Predict whether an asset is likely to experience a fault within the next 24 hours.

### Target

The target is engineered from future fault events:

```text
failure_next_24h
```

The current `fault_flag` is not used as the prediction target.

### Features

The model uses:

- Temperature
- Humidity
- Pressure
- Vibration
- Power consumption
- Occupancy
- Operating mode
- Asset type
- Manufacturer
- Capacity
- Asset age
- Parent-asset information
- Time features
- Rolling statistics
- Recent sensor changes
- Missing-value indicators

### Models evaluated

- Random Forest
- XGBoost
- CatBoost

The final predictive-maintenance pipeline uses the saved XGBoost model.

### Saved artifacts

```text
outputs/models/predictive_maintenance/
├── predictive_maintenance_model.joblib
├── model_config.json
├── test_predictions.csv
├── feature_importance.csv
└── asset_alert_summary.csv
```

The dashboard loads the saved pipeline rather than retraining it.

---

## Task 3 — Energy Consumption Forecasting

Objective:

> Forecast building-level energy consumption for the next 24 hours.

The Energy Meter asset is used as the building-level energy target.

### Models evaluated

- Prophet
- SARIMA
- XGBoost
- LSTM

The corrected XGBoost implementation uses hourly data and recursive forecasting.

### XGBoost features

- Previous 1-hour energy
- Previous 6-hour energy
- Previous 24-hour energy
- Previous 7-day energy
- 6-hour rolling mean
- 24-hour rolling mean
- 24-hour rolling standard deviation
- Hour
- Day of week
- Weekend indicator
- Cyclic hour features

### Saved artifacts

```text
outputs/models/energy_forecasting/
├── xgboost_energy_models.joblib
├── model_config.json
├── exact_24h_results.csv
├── exact_24h_predictions.csv
└── model_comparison.csv
```

The dashboard loads the building-specific saved XGBoost model and generates a recursive 24-hour forecast.

---

## Task 4 — Anomaly Detection

The anomaly-detection workflow combines multiple signals rather than relying on one detector.

Methods include:

- Statistical thresholding
- Isolation Forest
- Autoencoder
- DBSCAN
- Rolling-baseline behavior-change detection

The signals are combined into a hybrid anomaly score and categorized as:

```text
Normal
Watch
Anomaly
Critical
```

### Saved artifacts

```text
outputs/models/anomaly_detection/
├── anomaly_detection_models.joblib
└── anomaly_results.csv
```

The dashboard loads the precomputed hybrid anomaly results rather than retraining the detectors for every request.

---

## Task 5 — Multi-Asset Connectivity Analysis

The connectivity analysis uses:

```text
Site
  |
Building
  |
Asset
  |
Parent / Child relationships
```

and directed operational relationships:

```text
source_asset_id
       |
       v
target_asset_id
```

Each connection contains:

- Connection type
- Relationship strength

Connection types include:

- Supplies
- Controls
- Monitors

### Analysis includes

- Connected assets
- Downstream assets
- Failure impact
- Criticality indicators
- Isolated assets
- Parent-child validation
- Duplicate relationships
- Invalid relationships
- Self-connections
- Connectivity quality

### Saved artifacts

```text
outputs/connectivity/
├── asset_connectivity.csv
├── asset_hierarchy.csv
├── connectivity_summary.csv
├── criticality.csv
├── data_quality.csv
├── asset_hierarchy.png
├── asset_connectivity_network.png
└── failure_propagation.png
```

The dashboard displays the graphs generated by the notebook.

---

# 3. GraphQL Bonus

The project exposes the required connectivity operations through GraphQL.

Available operations:

### Connected assets

```graphql
query($assetName: String!) {
  connectedAssets(assetName: $assetName) {
    asset_id
    direction
    connection_type
    relationship_strength
  }
}
```

### Downstream assets

```graphql
query($assetName: String!) {
  downstreamAssets(assetName: $assetName) {
    asset_id
    depth
    connection_type
    relationship_strength
  }
}
```

### Assets under a site

```graphql
query($siteId: String!) {
  assetsUnderSite(siteId: $siteId) {
    asset_id
    site_id
    building_id
    asset_name
    asset_type
  }
}
```

### Isolated assets

```graphql
query {
  isolatedAssets {
    asset_id
    asset_name
    site_id
    building_id
    asset_type
  }
}
```

GraphQL is an API/query layer. A separate database is not required for this implementation because the connectivity graph is constructed using NetworkX from the saved connectivity data.

---

# 4. Dashboard

The Streamlit dashboard contains:

```text
Overview
Task 1 — EDA
Task 2 — Predictive Maintenance
Task 3 — Energy Forecasting
Task 4 — Anomaly Detection
Task 5 — Connectivity
GraphQL — Bonus
```

## Overview

Displays:

- Telemetry records
- Assets
- Buildings
- Sites
- Saved-artifact availability

## EDA

Displays:

- Notebook-generated figures
- EDA summary tables
- Dataset summary

## Predictive Maintenance

Allows the user to:

- Select an asset
- Generate a failure probability
- Apply the saved alert threshold
- View risk level
- View saved test-period risk summaries

## Energy Forecasting

Allows the user to:

- Select a building
- Generate the next 24-hour forecast
- View forecast values
- View saved evaluation results

## Anomaly Detection

Allows filtering by:

- Severity
- Asset type
- Asset ID

## Connectivity

Allows:

- Viewing connected assets
- Viewing downstream assets
- Viewing site assets
- Viewing isolated assets
- Viewing notebook-generated connectivity graphs
- Viewing hierarchy and failure-propagation graphs

## GraphQL

Provides an interactive interface for the four GraphQL challenge queries.

---

# 5. Project Structure

```text
Nectar Task/
│
├── backend/
│   ├── app.py
│   └── graphql_schema.py
│
├── frontend/
│   └── streamlit_app.py
│
├── src/
│   ├── data_loader.py
│   ├── prediction_service.py
│   ├── energy_service.py
│   ├── anomaly_service.py
│   └── graph_service.py
│
├── data/
│   └── raw/
│       ├── sensor_telemetry.csv
│       ├── asset_metadata.csv
│       └── asset_connectivity.csv
│
├── outputs/
│   ├── eda/
│   │   ├── figures/
│   │   └── tables/
│   │
│   ├── models/
│   │   ├── predictive_maintenance/
│   │   ├── energy_forecasting/
│   │   └── anomaly_detection/
│   │
│   └── connectivity/
│
├── jupyter notebook/
│   ├── 01_eda.ipynb
│   ├── 02_predictive_maintenance.ipynb
│   ├── 03_energy_consumption_forecasting.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_multi_asset_connectivity_analysis.ipynb
│
├── requirements.txt
└── README.md
```

---

# 6. Dataset

The project uses three raw datasets.

## Sensor Telemetry

Important fields:

```text
timestamp
site_id
building_id
asset_id
temperature
humidity
pressure
vibration
power_consumption
occupancy_count
operating_mode
fault_flag
```

## Asset Metadata

```text
asset_id
site_id
building_id
asset_name
asset_type
manufacturer
installation_date
capacity
parent_asset_id
```

## Asset Connectivity

```text
source_asset_id
target_asset_id
connection_type
relationship_strength
```

---

# 7. Installation

Create and activate a virtual environment if required.

### Windows

```powershell
python -m venv myvenv
myvenv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

# 8. Run the Notebooks

Open the project root and launch Jupyter:

```powershell
jupyter notebook
```

Run the notebooks in order:

```text
01_eda.ipynb
02_predictive_maintenance.ipynb
03_energy_consumption_forecasting.ipynb
04_anomaly_detection.ipynb
05_multi_asset_connectivity_analysis.ipynb
```

The notebooks generate the required artifacts under `outputs/`.

The notebooks should be run before starting the dashboard if the saved models/results do not already exist.

---

# 9. Run the Flask Backend

From the project root:

```powershell
python backend/app.py
```

The backend runs at:

```text
http://127.0.0.1:5000
```

Health check:

```text
http://127.0.0.1:5000/health
```

Available API areas include:

```text
GET  /health
GET  /api/overview
GET  /api/artifacts
GET  /api/assets

POST /api/predict_failure

GET  /api/energy/forecast/<building_id>

GET  /api/anomalies
GET  /api/anomalies/summary

GET  /api/connectivity/<asset_id>
GET  /api/downstream/<asset_id>
GET  /api/sites/<site_id>/assets
GET  /api/isolated-assets

POST /graphql
```

---

# 10. Run the Streamlit Dashboard

Open another terminal from the project root:

```powershell
streamlit run frontend/streamlit_app.py
```

The dashboard will normally open at:

```text
http://localhost:8501
```

Both the Flask backend and Streamlit application should be running when using the GraphQL functionality.

---

# 11. Requirements

The main dependencies are:

```text
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
catboost
prophet
statsmodels
tensorflow
jupyter
notebook
ipykernel
networkx
joblib
flask
flask-cors
graphql-core
streamlit
openpyxl
```

Install them with:

```powershell
pip install -r requirements.txt
```

---

# 12. Important Design Principle

The project separates **analysis/training** from **application inference**.

```text
Notebook
    |
    | Training / Analysis
    v
Saved Artifact
    |
    | Loading
    v
Service Layer
    |
    v
Flask / GraphQL / Streamlit
```

The application does not retrain the models.

For example:

```text
Task 2:
Saved XGBoost Pipeline
        ↓
prediction_service.py
        ↓
Failure Prediction
```

```text
Task 3:
Saved Building XGBoost Models
        ↓
energy_service.py
        ↓
24-hour Forecast
```

```text
Task 4:
Saved Hybrid Anomaly Results
        ↓
anomaly_service.py
        ↓
Anomaly Dashboard
```

```text
Task 5:
Saved Connectivity Data
        ↓
graph_service.py
        ↓
NetworkX / GraphQL / Dashboard
```

This keeps the dashboard consistent with the notebook results and avoids duplicate training logic.

---

# 13. Important Limitations

This project is a proof-of-concept based on the challenge dataset.

- Predictive-maintenance results should be interpreted as early-warning performance rather than independent failure-event detection for every telemetry row.
- The dataset is synthetic, so real maintenance data would be required for production validation.
- Energy forecasting models use different modeling resolutions in the notebook; the corrected XGBoost inference operates hourly.
- A single exact 24-hour forecast-origin evaluation should not be interpreted as a universal performance guarantee.
- Anomaly thresholds and hybrid scores are dataset-specific.
- Connectivity indicates potential dependencies; a graph relationship does not automatically guarantee physical failure propagation.
- The current challenge data does not contain a dedicated floor identifier, so the hierarchy uses the available Site → Building → Asset/Parent Asset structure.
- Graph structural metrics such as downstream count and betweenness are indicators of connectivity influence, not definitive operational criticality.

---

# 14. End-to-End Run Order

For a fresh setup:

```text
1. Install requirements
        ↓
2. Verify data/raw
        ↓
3. Run Task 1 notebook
        ↓
4. Run Task 2 notebook
        ↓
5. Run Task 3 notebook
        ↓
6. Run Task 4 notebook
        ↓
7. Run Task 5 notebook
        ↓
8. Verify outputs/
        ↓
9. Start Flask
        ↓
10. Start Streamlit
        ↓
11. Open dashboard
```

For subsequent runs, if the saved artifacts already exist, the notebooks do not need to be rerun unless the data/model/results have changed.

---

# 15. Application Architecture

```text
                         ┌─────────────────────┐
                         │   Raw IoT Data       │
                         │   Metadata           │
                         │   Connectivity       │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Jupyter Notebooks   │
                         │                     │
                         │ EDA                 │
                         │ Predictive Maint.   │
                         │ Energy Forecasting  │
                         │ Anomaly Detection   │
                         │ Connectivity        │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Saved Artifacts     │
                         │ Models              │
                         │ Results             │
                         │ Graphs              │
                         └──────────┬──────────┘
                                    │
                 ┌──────────────────┴──────────────────┐
                 ▼                                     ▼
        ┌─────────────────┐                   ┌─────────────────┐
        │ Flask Backend   │                   │ Streamlit       │
        │ REST APIs       │◄──── GraphQL ───►│ Dashboard       │
        └─────────────────┘                   └─────────────────┘
```

---

# 16. Project Objective

The overall objective is to demonstrate how IoT telemetry and connected-asset data can be transformed into actionable facility-management intelligence:

```text
Observe
   ↓
Analyze
   ↓
Predict
   ↓
Detect
   ↓
Understand Dependencies
   ↓
Support Facility Operations
```

The implementation combines data science, machine learning, time-series forecasting, anomaly detection, graph analysis, REST APIs, GraphQL, and interactive visualization in one reproducible project.
