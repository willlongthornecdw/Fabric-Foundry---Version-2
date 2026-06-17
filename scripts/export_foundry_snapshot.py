import json
import random
from pathlib import Path
from datetime import datetime, timezone

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "foundry_machine_risk_snapshot.json"


def risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "Critical"
    if probability >= 0.50:
        return "High"
    if probability >= 0.25:
        return "Medium"
    return "Low"


def recommended_action(band: str, top_factors: list[str]) -> str:
    if band == "Critical":
        return f"Inspect immediately. Focus on {', '.join(top_factors[:2])}."
    if band == "High":
        return f"Inspect during the next shift. Focus on {', '.join(top_factors[:2])}."
    if band == "Medium":
        return "Review during the next planned maintenance window."
    return "Continue normal operation and monitor."


def generate_mock_snapshot(num_machines: int = 30) -> list[dict]:
    plants = ["Plant-A", "Plant-B", "Plant-C"]
    possible_factors = [
        "vibration_rolling_24h",
        "temperature_rolling_24h",
        "pressure_rolling_24h",
        "power_rolling_24h",
        "error_count_24h",
        "runtime_hours",
    ]

    machines = []

    for i in range(1, num_machines + 1):
        machine_id = f"M-{i:03d}"
        plant_id = random.choice(plants)

        # Create a spread of mostly normal machines with some risky ones.
        failure_probability = round(random.betavariate(2, 5), 3)

        # Force a few high/critical examples so the demo is interesting.
        if i in [4, 11, 17]:
            failure_probability = round(random.uniform(0.76, 0.93), 3)
        elif i in [7, 15, 22]:
            failure_probability = round(random.uniform(0.55, 0.74), 3)

        band = risk_band(failure_probability)
        top_factors = random.sample(possible_factors, 3)

        machines.append(
            {
                "machine_id": machine_id,
                "plant_id": plant_id,
                "snapshot_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "failure_probability": failure_probability,
                "risk_band": band,
                "top_risk_factors": top_factors,
                "sensor_summary": {
                    "vibration_rolling_24h": round(random.uniform(0.1, 1.0), 3),
                    "temperature_rolling_24h": round(random.uniform(60, 105), 1),
                    "pressure_rolling_24h": round(random.uniform(20, 90), 1),
                    "power_rolling_24h": round(random.uniform(2.0, 12.0), 2),
                    "error_count_24h": random.randint(0, 12),
                    "runtime_hours": random.randint(100, 15000),
                },
                "recommended_action": recommended_action(band, top_factors),
                "work_order_priority": {
                    "Critical": "P1",
                    "High": "P2",
                    "Medium": "P3",
                    "Low": "P4",
                }[band],
            }
        )

    return sorted(
        machines,
        key=lambda x: x["failure_probability"],
        reverse=True,
    )


def main():
    snapshot = generate_mock_snapshot()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Saved Foundry machine risk snapshot to: {OUTPUT_FILE}")
    print(f"Machine count: {len(snapshot)}")
    print()
    print("Top 5 risky machines:")

    for machine in snapshot[:5]:
        print(
            f"- {machine['machine_id']} | "
            f"{machine['plant_id']} | "
            f"{machine['risk_band']} | "
            f"{machine['failure_probability']}"
        )


if __name__ == "__main__":
    main()