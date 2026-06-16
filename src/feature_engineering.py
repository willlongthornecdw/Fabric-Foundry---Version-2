from pathlib import Path
import pandas as pd


def add_future_failure_label(
    telemetry_df: pd.DataFrame,
    events_df: pd.DataFrame,
    lookahead_hours: int = 24,
) -> pd.DataFrame:
    """
    Create the future-looking target label.

    For each telemetry row:
    failure_within_24h = 1 if the same machine has a failure event
    within the next 24 hours.
    """

    telemetry_df = telemetry_df.copy()
    telemetry_df["failure_within_24h"] = 0

    failure_events = events_df[events_df["event_type"] == "failure"].copy()
    failure_events["event_timestamp"] = pd.to_datetime(failure_events["event_timestamp"])

    lookahead = pd.Timedelta(hours=lookahead_hours)

    for machine_id, machine_rows in telemetry_df.groupby("machine_id"):
        machine_failure_times = failure_events.loc[
            failure_events["machine_id"] == machine_id,
            "event_timestamp",
        ].sort_values()

        if machine_failure_times.empty:
            continue

        failure_times = machine_failure_times.to_list()
        machine_indices = machine_rows.index

        labels = []
        for current_time in machine_rows["timestamp"]:
            future_failure = any(
                current_time < failure_time <= current_time + lookahead
                for failure_time in failure_times
            )
            labels.append(1 if future_failure else 0)

        telemetry_df.loc[machine_indices, "failure_within_24h"] = labels

    return telemetry_df


def build_features(
    telemetry_path: str,
    events_path: str,
    output_path: str,
) -> pd.DataFrame:
    """
    Convert V2 source data into a gold model-ready feature table.

    V2 improvement:
    - Read telemetry from the sensor source.
    - Read maintenance/failure events from a separate event source.
    - Create a future-looking failure label from the event history.
    - Build rolling-window sensor features for predictive modeling.
    """

    telemetry_df = pd.read_csv(telemetry_path, parse_dates=["timestamp"])
    events_df = pd.read_csv(events_path, parse_dates=["event_timestamp"])

    telemetry_df = telemetry_df.drop_duplicates()
    telemetry_df = telemetry_df.sort_values(["machine_id", "timestamp"])

    numeric_cols = [
        "temperature_c",
        "vibration_mm_s",
        "pressure_psi",
        "rpm",
        "power_kw",
        "oil_quality_pct",
        "error_count",
        "runtime_hours",
    ]

    for col in numeric_cols:
        telemetry_df[col] = pd.to_numeric(telemetry_df[col], errors="coerce")

    telemetry_df[numeric_cols] = (
        telemetry_df.groupby("machine_id")[numeric_cols].ffill().bfill()
    )

    # Add target label from the separate event log.
    telemetry_df = add_future_failure_label(
        telemetry_df=telemetry_df,
        events_df=events_df,
        lookahead_hours=24,
    )

    grouped = telemetry_df.groupby("machine_id", group_keys=False)

    # 15-minute readings x 16 = 4 hours
    telemetry_df["temp_rolling_4h"] = (
        grouped["temperature_c"].rolling(16).mean().reset_index(level=0, drop=True)
    )
    telemetry_df["vibration_rolling_4h"] = (
        grouped["vibration_mm_s"].rolling(16).mean().reset_index(level=0, drop=True)
    )
    telemetry_df["pressure_rolling_4h"] = (
        grouped["pressure_psi"].rolling(16).mean().reset_index(level=0, drop=True)
    )
    telemetry_df["power_rolling_4h"] = (
        grouped["power_kw"].rolling(16).mean().reset_index(level=0, drop=True)
    )
    telemetry_df["errors_rolling_4h"] = (
        grouped["error_count"].rolling(16).sum().reset_index(level=0, drop=True)
    )

    # 15-minute readings x 96 = 24 hours
    telemetry_df["temp_rolling_24h"] = (
        grouped["temperature_c"].rolling(96).mean().reset_index(level=0, drop=True)
    )
    telemetry_df["vibration_rolling_24h"] = (
        grouped["vibration_mm_s"].rolling(96).mean().reset_index(level=0, drop=True)
    )
    telemetry_df["pressure_rolling_24h"] = (
        grouped["pressure_psi"].rolling(96).mean().reset_index(level=0, drop=True)
    )
    telemetry_df["power_rolling_24h"] = (
        grouped["power_kw"].rolling(96).mean().reset_index(level=0, drop=True)
    )
    telemetry_df["errors_rolling_24h"] = (
        grouped["error_count"].rolling(96).sum().reset_index(level=0, drop=True)
    )

    telemetry_df["temp_change_4h"] = grouped["temperature_c"].diff(16)
    telemetry_df["vibration_change_4h"] = grouped["vibration_mm_s"].diff(16)
    telemetry_df["pressure_change_4h"] = grouped["pressure_psi"].diff(16)

    telemetry_df["temp_change_24h"] = grouped["temperature_c"].diff(96)
    telemetry_df["vibration_change_24h"] = grouped["vibration_mm_s"].diff(96)
    telemetry_df["pressure_change_24h"] = grouped["pressure_psi"].diff(96)

    telemetry_df["sensor_stress_index"] = (
        0.20 * telemetry_df["temperature_c"]
        + 10.0 * telemetry_df["vibration_mm_s"]
        - 0.10 * telemetry_df["pressure_psi"]
        + 0.40 * telemetry_df["power_kw"]
        - 0.15 * telemetry_df["oil_quality_pct"]
        + 2.0 * telemetry_df["error_count"]
    )

    # Add event-context features from the event log.
    failure_events = events_df[events_df["event_type"] == "failure"].copy()
    maintenance_events = events_df[events_df["event_type"] == "maintenance"].copy()

    failure_counts = (
        failure_events.groupby("machine_id")
        .size()
        .rename("historical_failure_count")
        .reset_index()
    )

    maintenance_counts = (
        maintenance_events.groupby("machine_id")
        .size()
        .rename("historical_maintenance_count")
        .reset_index()
    )

    telemetry_df = telemetry_df.merge(failure_counts, on="machine_id", how="left")
    telemetry_df = telemetry_df.merge(maintenance_counts, on="machine_id", how="left")

    telemetry_df["historical_failure_count"] = (
        telemetry_df["historical_failure_count"].fillna(0)
    )
    telemetry_df["historical_maintenance_count"] = (
        telemetry_df["historical_maintenance_count"].fillna(0)
    )

    telemetry_df = telemetry_df.dropna()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    telemetry_df.to_csv(output, index=False)
    telemetry_df.to_parquet(str(output).replace(".csv", ".parquet"), index=False)

    return telemetry_df


if __name__ == "__main__":
    features = build_features(
        telemetry_path="data/raw/machine_sensor_readings.csv",
        events_path="data/raw/maintenance_failure_events.csv",
        output_path="data/gold/machine_features_v2.csv",
    )

    print(f"Created V2 feature table with {len(features):,} rows")
    print("Saved CSV to: data/gold/machine_features_v2.csv")
    print("Saved Parquet to: data/gold/machine_features_v2.parquet")
    print()
    print(features.head())
    print()
    print("Future-looking label rate:")
    print(features["failure_within_24h"].value_counts(normalize=True))