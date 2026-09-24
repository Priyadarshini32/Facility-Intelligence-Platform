"""
NectarIT Data Scientist Challenge
Synthetic IoT Dataset Generator

Generates three CSV files:
    data/raw/sensor_telemetry.csv
    data/raw/asset_metadata.csv
    data/raw/asset_connectivity.csv

The generated data is synthetic and is designed to support:
    1. Exploratory Data Analysis
    2. Predictive Maintenance
    3. Energy Consumption Forecasting
    4. Anomaly Detection
    5. Multi-Asset Connectivity Analysis

Important:
- This is synthetic challenge data, not real NectarIT production data.
- The raw telemetry contains fault_flag, but NOT failure_next_24h.
  The 24-hour prediction target should be created later during
  Task 2 feature engineering to avoid target leakage.
"""

import os
import numpy as np
import pandas as pd


# ============================================================
# 0. CONFIGURATION
# ============================================================

SEED = 42
np.random.seed(SEED)

OUTPUT_DIR = "data/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

START_DATE = "2026-01-01"
END_DATE = "2026-03-31 23:45:00"
FREQUENCY = "15min"

# 15 minutes = 15/60 hour
INTERVAL_HOURS = 0.25


# ============================================================
# 1. CREATE SITE / BUILDING STRUCTURE
# ============================================================

sites = {
    "SITE_01": ["BLDG_01", "BLDG_02"],
    "SITE_02": ["BLDG_03", "BLDG_04"],
    "SITE_03": ["BLDG_05", "BLDG_06"],
}


# ============================================================
# 2. ASSET DEFINITIONS
# ============================================================

# Each building contains:
#   1 Chiller
#   2 AHUs
#   1 Pump
#   1 Energy Meter
#
# Therefore:
#   6 buildings × 5 assets = 30 assets

asset_types = [
    "Chiller",
    "AHU",
    "AHU",
    "Pump",
    "Energy Meter",
]

manufacturers = {
    "Chiller": ["Carrier", "Trane", "Daikin"],
    "AHU": ["Daikin", "Johnson Controls", "Trane"],
    "Pump": ["Grundfos", "Wilo", "KSB"],
    "Energy Meter": ["Siemens", "Schneider", "ABB"],
}


# ============================================================
# 3. CREATE ASSET METADATA
# ============================================================

assets = []
asset_counter = 1

for site_id, buildings in sites.items():

    for building_id in buildings:

        for asset_type in asset_types:

            asset_id = f"ASSET_{asset_counter:03d}"

            if asset_type == "Chiller":

                asset_name = f"Chiller-{building_id[-2:]}"
                capacity = np.random.choice([500, 750, 1000, 1200])
                parent_asset_id = None

            elif asset_type == "AHU":

                existing_ahus = [
                    a
                    for a in assets
                    if a["building_id"] == building_id
                    and a["asset_type"] == "AHU"
                ]

                asset_name = (
                    f"AHU-{building_id[-2:]}"
                    f"-{len(existing_ahus) + 1}"
                )

                capacity = np.random.choice([100, 150, 200, 250])

                # AHU is hierarchically associated with its building's
                # chiller.
                parent_asset_id = next(
                    (
                        a["asset_id"]
                        for a in assets
                        if a["building_id"] == building_id
                        and a["asset_type"] == "Chiller"
                    ),
                    None,
                )

            elif asset_type == "Pump":

                asset_name = f"Pump-{building_id[-2:]}"
                capacity = np.random.choice([50, 75, 100])

                # Pump is hierarchically associated with the chiller.
                parent_asset_id = next(
                    (
                        a["asset_id"]
                        for a in assets
                        if a["building_id"] == building_id
                        and a["asset_type"] == "Chiller"
                    ),
                    None,
                )

            else:

                asset_name = f"EnergyMeter-{building_id[-2:]}"
                capacity = np.random.choice([500, 1000, 1500])

                # The energy meter is treated as a building-level
                # measurement device rather than a child of the chiller.
                parent_asset_id = None

            manufacturer = np.random.choice(
                manufacturers[asset_type]
            )

            # Installation dates range from 2019 to 2025.
            installation_date = (
                pd.Timestamp("2019-01-01")
                + pd.Timedelta(
                    days=int(np.random.randint(0, 2200))
                )
            )

            assets.append(
                {
                    "asset_id": asset_id,
                    "site_id": site_id,
                    "building_id": building_id,
                    "asset_name": asset_name,
                    "asset_type": asset_type,
                    "manufacturer": manufacturer,
                    "installation_date": installation_date,
                    "capacity": capacity,
                    "parent_asset_id": parent_asset_id,
                }
            )

            asset_counter += 1


metadata_df = pd.DataFrame(assets)


# ============================================================
# 4. CREATE ASSET CONNECTIVITY
# ============================================================

connections = []

for _, asset in metadata_df.iterrows():

    asset_type = asset["asset_type"]
    building_id = asset["building_id"]

    # --------------------------------------------------------
    # Chiller -> AHU
    # --------------------------------------------------------

    if asset_type == "Chiller":

        ahus = metadata_df[
            (metadata_df["building_id"] == building_id)
            & (metadata_df["asset_type"] == "AHU")
        ]

        for _, ahu in ahus.iterrows():

            # Chiller supplies AHU
            connections.append(
                {
                    "source_asset_id": asset["asset_id"],
                    "target_asset_id": ahu["asset_id"],
                    "connection_type": "Supplies",
                    "relationship_strength": round(
                        np.random.uniform(0.75, 0.98),
                        2,
                    ),
                }
            )

            # Chiller controls AHU
            connections.append(
                {
                    "source_asset_id": asset["asset_id"],
                    "target_asset_id": ahu["asset_id"],
                    "connection_type": "Controls",
                    "relationship_strength": round(
                        np.random.uniform(0.60, 0.90),
                        2,
                    ),
                }
            )

    # --------------------------------------------------------
    # Pump -> Chiller
    # --------------------------------------------------------

    elif asset_type == "Pump":

        chiller = metadata_df[
            (metadata_df["building_id"] == building_id)
            & (metadata_df["asset_type"] == "Chiller")
        ]

        if not chiller.empty:

            chiller_id = chiller.iloc[0]["asset_id"]

            connections.append(
                {
                    "source_asset_id": asset["asset_id"],
                    "target_asset_id": chiller_id,
                    "connection_type": "Supplies",
                    "relationship_strength": round(
                        np.random.uniform(0.70, 0.95),
                        2,
                    ),
                }
            )

    # --------------------------------------------------------
    # Energy Meter -> Chiller
    # --------------------------------------------------------

    elif asset_type == "Energy Meter":

        chiller = metadata_df[
            (metadata_df["building_id"] == building_id)
            & (metadata_df["asset_type"] == "Chiller")
        ]

        if not chiller.empty:

            chiller_id = chiller.iloc[0]["asset_id"]

            connections.append(
                {
                    "source_asset_id": asset["asset_id"],
                    "target_asset_id": chiller_id,
                    "connection_type": "Monitors",
                    "relationship_strength": round(
                        np.random.uniform(0.80, 1.00),
                        2,
                    ),
                }
            )


connectivity_df = pd.DataFrame(connections)


# ============================================================
# 5. CREATE TIME RANGE
# ============================================================

timestamps = pd.date_range(
    start=START_DATE,
    end=END_DATE,
    freq=FREQUENCY,
)


# ============================================================
# 6. CREATE FAULT EVENTS
# ============================================================

# We deliberately leave 6 assets fault-free during the observation
# period. This creates both failure-prone and healthy assets.
#
# The raw fault_flag represents an observed/current fault state.
# It is NOT the final predictive-maintenance target.

fault_events = {}

for asset_index, asset_id in enumerate(
    metadata_df["asset_id"]
):

    if asset_index >= 24:

        fault_events[asset_id] = []
        continue

    # Each failure-prone asset receives 1–3 possible fault events.
    num_faults = np.random.randint(1, 4)

    selected_times = np.random.choice(
        len(timestamps) - 96,
        size=num_faults,
        replace=False,
    )

    fault_events[asset_id] = [
        timestamps[i]
        for i in selected_times
    ]


# ============================================================
# 7. CREATE BUILDING-LEVEL OCCUPANCY
# ============================================================

# Occupancy is a property of the building, not of an individual
# piece of equipment.
#
# Therefore every asset belonging to the same building receives
# exactly the same occupancy value at a given timestamp.

building_occupancy = {}

for building_id in metadata_df["building_id"].unique():

    building_occupancy[building_id] = {}

    for timestamp in timestamps:

        hour = timestamp.hour
        day_of_week = timestamp.dayofweek

        if day_of_week >= 5:

            # Lower occupancy on weekends.
            occupancy = np.random.normal(25, 8)

        elif 8 <= hour <= 18:

            # Higher occupancy during working hours.
            occupancy = np.random.normal(150, 30)

        else:

            # Low occupancy outside working hours.
            occupancy = np.random.normal(30, 10)

        building_occupancy[building_id][timestamp] = max(
            0,
            round(occupancy),
        )


# ============================================================
# 8. GENERATE TELEMETRY
# ============================================================

telemetry_rows = []

for _, asset in metadata_df.iterrows():

    asset_id = asset["asset_id"]
    asset_type = asset["asset_type"]
    building_id = asset["building_id"]

    asset_fault_times = fault_events[asset_id]

    # Asset-specific environmental baseline.
    base_temperature = np.random.uniform(21, 24)
    base_humidity = np.random.uniform(45, 60)

    asset_age_years = (
        pd.Timestamp(END_DATE)
        - asset["installation_date"]
    ).days / 365.25

    for timestamp in timestamps:

        hour = timestamp.hour
        day_of_week = timestamp.dayofweek

        # ====================================================
        # BUILDING-LEVEL OCCUPANCY
        # ====================================================

        occupancy = building_occupancy[
            building_id
        ][timestamp]

        # ====================================================
        # DAILY TEMPERATURE PATTERN
        # ====================================================

        daily_temperature_effect = (
            3
            * np.sin(
                2
                * np.pi
                * (hour - 6)
                / 24
            )
        )

        temperature = (
            base_temperature
            + daily_temperature_effect
            + np.random.normal(0, 0.5)
        )

        # ====================================================
        # OPERATING MODE
        # ====================================================

        if 8 <= hour <= 18:

            if temperature > 23:

                operating_mode = "Cooling"

            else:

                operating_mode = np.random.choice(
                    [
                        "Cooling",
                        "Heating",
                        "Idle",
                    ],
                    p=[
                        0.72,
                        0.08,
                        0.20,
                    ],
                )

        else:

            operating_mode = np.random.choice(
                [
                    "Idle",
                    "Cooling",
                    "Heating",
                ],
                p=[
                    0.82,
                    0.15,
                    0.03,
                ],
            )

        # ====================================================
        # HUMIDITY
        # ====================================================

        humidity = (
            base_humidity
            - 0.4 * (temperature - 22)
            + np.random.normal(0, 3)
        )

        humidity = np.clip(
            humidity,
            20,
            90,
        )

        # ====================================================
        # PRESSURE
        # ====================================================

        if asset_type == "Pump":

            pressure = (
                4.5
                + np.random.normal(0, 0.2)
            )

        elif asset_type == "Chiller":

            pressure = (
                1.5
                + np.random.normal(0, 0.08)
            )

        else:

            pressure = (
                1.0
                + np.random.normal(0, 0.1)
            )

        # ====================================================
        # VIBRATION
        # ====================================================

        if asset_type == "Pump":

            vibration = (
                0.20
                + 0.01 * asset_age_years
                + np.random.normal(0, 0.04)
            )

        elif asset_type == "Chiller":

            vibration = (
                0.15
                + 0.008 * asset_age_years
                + np.random.normal(0, 0.03)
            )

        else:

            vibration = (
                0.08
                + np.random.normal(0, 0.02)
            )

        # ====================================================
        # ENERGY CONSUMPTION
        # ====================================================

        # base_power is an instantaneous load in kW.
        #
        # The telemetry is sampled every 15 minutes.
        # Therefore:
        #
        #     interval_energy_kwh = power_kw × 0.25
        #
        # The stored power_consumption column represents the
        # energy consumed during that 15-minute interval.

        if asset_type == "Chiller":

            base_power = (
                100
                + 0.7 * occupancy
                + 10 * max(
                    temperature - 22,
                    0,
                )
            )

            if operating_mode == "Cooling":

                base_power *= 1.5

            elif operating_mode == "Heating":

                base_power *= 1.25

            elif operating_mode == "Idle":

                base_power *= 0.3

        elif asset_type == "AHU":

            base_power = (
                40
                + 0.25 * occupancy
            )

            if operating_mode == "Cooling":

                base_power *= 1.3

            elif operating_mode == "Heating":

                base_power *= 1.15

        elif asset_type == "Pump":

            base_power = (
                35
                + 0.08 * occupancy
            )

            if operating_mode == "Idle":

                base_power *= 0.4

        else:

            # Energy Meter represents the building-level measured
            # energy load. It is a measurement device rather than
            # an independent consuming HVAC asset.

            base_power = (
                185
                + 0.90 * occupancy
                + 14 * max(
                    temperature - 22,
                    0,
                )
            )

            if operating_mode == "Cooling":

                base_power *= 1.15

            elif operating_mode == "Heating":

                base_power *= 1.08

            elif operating_mode == "Idle":

                base_power *= 0.65

        power_consumption = (
            base_power
            + np.random.normal(0, 8)
        )

        power_consumption = max(
            1,
            power_consumption,
        )

        # Convert kW to kWh for the 15-minute interval.
        power_consumption = (
            power_consumption
            * INTERVAL_HOURS
        )

        # ====================================================
        # FAULT / DEGRADATION
        # ====================================================

        fault_flag = 0

        for fault_time in asset_fault_times:

            hours_to_fault = (
                fault_time - timestamp
            ).total_seconds() / 3600

            # -----------------------------------------------
            # Gradual degradation during the 12 hours
            # before the fault.
            # -----------------------------------------------

            if 0 <= hours_to_fault <= 12:

                degradation = (
                    1
                    - hours_to_fault / 12
                )

                # Asset-specific health signals.
                if asset_type == "Pump":

                    vibration += (
                        degradation
                        * np.random.uniform(
                            0.45,
                            0.90,
                        )
                    )

                    pressure += (
                        degradation
                        * np.random.uniform(
                            0.20,
                            0.60,
                        )
                    )

                elif asset_type == "Chiller":

                    vibration += (
                        degradation
                        * np.random.uniform(
                            0.30,
                            0.70,
                        )
                    )

                    pressure += (
                        degradation
                        * np.random.uniform(
                            0.08,
                            0.25,
                        )
                    )

                    temperature += (
                        degradation
                        * np.random.uniform(
                            0.8,
                            2.5,
                        )
                    )

                else:

                    vibration += (
                        degradation
                        * np.random.uniform(
                            0.20,
                            0.50,
                        )
                    )

                # Increasing energy consumption during degradation.
                power_consumption *= (
                    1
                    + degradation
                    * np.random.uniform(
                        0.10,
                        0.25,
                    )
                )

                # Environmental deviation for air-handling/cooling
                # assets before failure.
                if asset_type in [
                    "AHU",
                    "Chiller",
                ]:

                    temperature += (
                        degradation
                        * np.random.uniform(
                            0.5,
                            2.0,
                        )
                    )

            # -----------------------------------------------
            # Fault state.
            # -----------------------------------------------

            if 0 <= hours_to_fault <= 1:

                fault_flag = 1

        # ====================================================
        # RANDOM ANOMALIES
        # ====================================================

        # Rare energy spikes.
        if np.random.random() < 0.002:

            power_consumption *= np.random.uniform(
                1.8,
                3.0,
            )

        # Rare vibration spikes.
        if np.random.random() < 0.001:

            vibration *= np.random.uniform(
                3,
                6,
            )

        # ====================================================
        # PHYSICAL VALUE VALIDATION
        # ====================================================

        # Vibration magnitude cannot be negative.
        vibration = max(
            0,
            vibration,
        )

        # ====================================================
        # MISSING SENSOR VALUES
        # ====================================================

        if np.random.random() < 0.01:

            temperature = np.nan

        if np.random.random() < 0.01:

            humidity = np.nan

        if np.random.random() < 0.008:

            pressure = np.nan

        if np.random.random() < 0.008:

            vibration = np.nan

        # ====================================================
        # STORE ROW
        # ====================================================

        telemetry_rows.append(
            {
                "timestamp": timestamp,
                "site_id": asset["site_id"],
                "building_id": building_id,
                "asset_id": asset_id,
                "temperature": round(
                    temperature,
                    3,
                )
                if pd.notna(temperature)
                else np.nan,
                "humidity": round(
                    humidity,
                    3,
                )
                if pd.notna(humidity)
                else np.nan,
                "pressure": round(
                    pressure,
                    3,
                )
                if pd.notna(pressure)
                else np.nan,
                "vibration": round(
                    vibration,
                    3,
                )
                if pd.notna(vibration)
                else np.nan,
                "power_consumption": round(
                    power_consumption,
                    3,
                ),
                "occupancy_count": int(
                    occupancy
                ),
                "operating_mode": operating_mode,
                "fault_flag": fault_flag,
            }
        )


telemetry_df = pd.DataFrame(
    telemetry_rows
)


# ============================================================
# 9. VALIDATION BEFORE SAVING
# ============================================================

# Basic uniqueness check.
assert not telemetry_df.duplicated(
    subset=[
        "timestamp",
        "asset_id",
    ]
).any(), "Duplicate asset/timestamp telemetry found."

# Vibration should never contain negative non-missing values.
assert (
    telemetry_df["vibration"].dropna() >= 0
).all(), "Negative vibration values found."

# Occupancy must be identical for all assets in a building
# at the same timestamp.
occupancy_consistency = (
    telemetry_df
    .groupby(
        [
            "building_id",
            "timestamp",
        ]
    )["occupancy_count"]
    .nunique()
    .max()
)

assert (
    occupancy_consistency == 1
), "Occupancy is not consistent within building/timestamp."

# Connectivity IDs must exist in metadata.
asset_ids = set(
    metadata_df["asset_id"]
)

assert set(
    connectivity_df["source_asset_id"]
).issubset(asset_ids), "Invalid source asset ID."

assert set(
    connectivity_df["target_asset_id"]
).issubset(asset_ids), "Invalid target asset ID."

# No self-connections.
assert not (
    connectivity_df["source_asset_id"]
    == connectivity_df["target_asset_id"]
).any(), "Self-loop found."

# No duplicate relationships.
assert not connectivity_df.duplicated(
    subset=[
        "source_asset_id",
        "target_asset_id",
        "connection_type",
    ]
).any(), "Duplicate connectivity relationship found."


# ============================================================
# 10. SAVE DATA
# ============================================================

metadata_df.to_csv(
    f"{OUTPUT_DIR}/asset_metadata.csv",
    index=False,
)

connectivity_df.to_csv(
    f"{OUTPUT_DIR}/asset_connectivity.csv",
    index=False,
)

telemetry_df.to_csv(
    f"{OUTPUT_DIR}/sensor_telemetry.csv",
    index=False,
)


# ============================================================
# 11. PRINT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("NECTARIT SYNTHETIC DATA GENERATION COMPLETED")
print("=" * 60)

print("\nTelemetry:")
print(telemetry_df.shape)

print("\nMetadata:")
print(metadata_df.shape)

print("\nConnectivity:")
print(connectivity_df.shape)

print("\nFault records:")
print(
    telemetry_df["fault_flag"].value_counts()
)

print("\nAsset types:")
print(
    metadata_df["asset_type"].value_counts()
)

print("\nAssets with fault events:")
print(
    sum(
        len(events) > 0
        for events in fault_events.values()
    )
)

print("\nAssets without fault events:")
print(
    sum(
        len(events) == 0
        for events in fault_events.values()
    )
)

print("\nOperating modes:")
print(
    telemetry_df["operating_mode"].value_counts()
)

print("\nConnectivity types:")
print(
    connectivity_df["connection_type"].value_counts()
)

print("\nValidation checks:")
print(
    "Negative vibration values:",
    int(
        (
            telemetry_df["vibration"] < 0
        ).sum()
    ),
)

print(
    "Max occupancy values per "
    "building/timestamp:",
    int(occupancy_consistency),
)

print(
    "Duplicate telemetry rows:",
    int(
        telemetry_df.duplicated(
            subset=[
                "timestamp",
                "asset_id",
            ]
        ).sum()
    ),
)

print(
    "Connectivity self-loops:",
    int(
        (
            connectivity_df["source_asset_id"]
            == connectivity_df["target_asset_id"]
        ).sum()
    ),
)

print("\nFiles created:")
print(
    "data/raw/sensor_telemetry.csv"
)
print(
    "data/raw/asset_metadata.csv"
)
print(
    "data/raw/asset_connectivity.csv"
)

print("\n" + "=" * 60)
print("DATASET IS READY FOR TASK 1 EDA")
print("=" * 60)
