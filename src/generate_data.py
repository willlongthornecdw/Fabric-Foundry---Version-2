from pathlib import Path
import numpy as np
import pandas as pd


def generate_v2_source_data(
    n_machines: int = 20,
    days: int = 45,
    freq: str = "15min",
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate V2 synthetic source data.

    V2 intentionally creates two separate source datasets:

    1. machine_sensor_readings.csv
       Represents manufacturing floor telemetry from sensors / historian / IoT systems.

    2. maintenance_failure_events.csv
       Represents operational event history from a CMMS / EAM / work order system.

    The key V2 lesson:
    Predictive maintenance usually requires joining telemetry with actual maintenance
    and failure event history. Telemetry alone is often not enough to define the
    business outcome we want to predict.
    """

    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        end=pd.Timestamp.now().floor("15min"),
        periods=int((24 * 60 / 15) * days),
        freq=freq,
    )

    telemetry_rows = []
    event_rows = []

    for machine_number in range(1, n_machines + 1):
        machine_id = f"M-{machine_number:03d}"
        line_id = f"L-{((machine_number - 1) // 5) + 1}"

        machine_age_years = rng.integers(1, 12)

        base_temp = rng.normal(68, 3)
        base_vibration = rng.normal(2.0, 0.3)
        base_pressure = rng.normal(102, 5)
        base_rpm = rng.normal(1500, 60)
        base_power = rng.normal(11.5, 1.5)

        degradation = rng.uniform(0.0, 0.2)
        degradation_rate = rng.uniform(0.0004, 0.0012)

        for ts in timestamps:
            # Hidden machine health state, not directly visible in raw telemetry.
            degradation += degradation_rate + rng.normal(0.0005, 0.0007)
            degradation = max(0, degradation)

            # Sensor readings reflect degradation, but imperfectly.
            temperature_c = base_temp + degradation * 10 + rng.normal(0, 1.5)
            vibration_mm_s = base_vibration + degradation * 1.6 + rng.normal(0, 0.18)
            pressure_psi = base_pressure - degradation * 7 + rng.normal(0, 2.2)
            rpm = base_rpm + rng.normal(0, 25)
            power_kw = base_power + degradation * 2.5 + rng.normal(0, 0.6)
            oil_quality_pct = max(0, 100 - degradation * 20 + rng.normal(0, 1.5))
            error_count = rng.poisson(max(0.02, degradation * 1.1))

            runtime_hours = (
                machine_age_years * 365 * 24
                + ((ts - timestamps[0]).total_seconds() / 3600)
            )

            telemetry_rows.append(
                {
                    "timestamp": ts,
                    "machine_id": machine_id,
                    "line_id": line_id,
                    "temperature_c": round(temperature_c, 2),
                    "vibration_mm_s": round(vibration_mm_s, 3),
                    "pressure_psi": round(pressure_psi, 2),
                    "rpm": round(rpm, 0),
                    "power_kw": round(power_kw, 2),
                    "oil_quality_pct": round(oil_quality_pct, 2),
                    "error_count": int(error_count),
                    "runtime_hours": round(runtime_hours, 1),
                }
            )

            # Failure event logic.
            # This produces a separate event log, not a direct training label.
            failure_pressure = (
                0.04 * max(temperature_c - 78, 0)
                + 1.10 * max(vibration_mm_s - 3.0, 0)
                + 0.03 * max(95 - pressure_psi, 0)
                + 0.04 * max(power_kw - 14, 0)
                + 0.04 * max(55 - oil_quality_pct, 0)
                + 0.30 * error_count
                + 0.75 * max(degradation - 1.2, 0)
            )

            failure_probability = 1 / (1 + np.exp(-(failure_pressure - 5.5)))
            actual_failure_event = rng.random() < failure_probability

            if actual_failure_event:
                failure_type = rng.choice(
                    [
                        "bearing_wear",
                        "overheating",
                        "hydraulic_pressure",
                        "lubrication",
                    ],
                    p=[0.35, 0.30, 0.20, 0.15],
                )

                event_rows.append(
                    {
                        "event_timestamp": ts,
                        "machine_id": machine_id,
                        "line_id": line_id,
                        "event_type": "failure",
                        "failure_type": failure_type,
                        "maintenance_action": "reactive_repair",
                        "downtime_minutes": int(rng.integers(45, 360)),
                    }
                )

                # Reactive repair partially restores health.
                degradation *= rng.uniform(0.15, 0.40)

            else:
                # Preventive maintenance may occur when degradation is visibly high.
                preventive_probability = min(0.18, max(0, degradation - 0.8) * 0.12)
                preventive_maintenance = rng.random() < preventive_probability

                if preventive_maintenance:
                    event_rows.append(
                        {
                            "event_timestamp": ts,
                            "machine_id": machine_id,
                            "line_id": line_id,
                            "event_type": "maintenance",
                            "failure_type": "none",
                            "maintenance_action": "preventive_service",
                            "downtime_minutes": int(rng.integers(15, 90)),
                        }
                    )

                    degradation *= rng.uniform(0.45, 0.75)

    telemetry_df = pd.DataFrame(telemetry_rows)
    events_df = pd.DataFrame(event_rows)

    return telemetry_df, events_df


if __name__ == "__main__":
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    telemetry_df, events_df = generate_v2_source_data()

    telemetry_csv = output_dir / "machine_sensor_readings.csv"
    telemetry_parquet = output_dir / "machine_sensor_readings.parquet"

    events_csv = output_dir / "maintenance_failure_events.csv"
    events_parquet = output_dir / "maintenance_failure_events.parquet"

    telemetry_df.to_csv(telemetry_csv, index=False)
    telemetry_df.to_parquet(telemetry_parquet, index=False)

    events_df.to_csv(events_csv, index=False)
    events_df.to_parquet(events_parquet, index=False)

    print(f"Generated telemetry rows: {len(telemetry_df):,}")
    print(f"Generated event rows: {len(events_df):,}")
    print()
    print(f"Saved telemetry CSV to: {telemetry_csv}")
    print(f"Saved event CSV to: {events_csv}")
    print()
    print("Telemetry sample:")
    print(telemetry_df.head())
    print()
    print("Event log sample:")
    print(events_df.head())
    print()
    print("Event type counts:")
    print(events_df["event_type"].value_counts())