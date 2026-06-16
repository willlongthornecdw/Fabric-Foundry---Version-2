from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "temperature_c",
    "vibration_mm_s",
    "pressure_psi",
    "rpm",
    "power_kw",
    "oil_quality_pct",
    "error_count",
    "runtime_hours",
    "temp_rolling_4h",
    "vibration_rolling_4h",
    "pressure_rolling_4h",
    "power_rolling_4h",
    "errors_rolling_4h",
    "temp_rolling_24h",
    "vibration_rolling_24h",
    "pressure_rolling_24h",
    "power_rolling_24h",
    "errors_rolling_24h",
    "temp_change_4h",
    "vibration_change_4h",
    "pressure_change_4h",
    "temp_change_24h",
    "vibration_change_24h",
    "pressure_change_24h",
    "sensor_stress_index",
    "historical_failure_count",
    "historical_maintenance_count",
]

TARGET_COLUMN = "failure_within_24h"


def train_model(input_path: str) -> None:
    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "feature_importance": {
            feature: float(importance)
            for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_)
        },
    }

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    model_path = output_dir / "predictive_maintenance_model_v2.joblib"
    metrics_path = output_dir / "model_metrics_v2.json"

    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("V2 model training complete.")
    print(f"Saved model to: {model_path}")
    print(f"Saved metrics to: {metrics_path}")
    print()
    print(f"ROC AUC: {metrics['roc_auc']:.3f}")
    print()
    print("Confusion matrix:")
    print(metrics["confusion_matrix"])
    print()
    print("Top feature importances:")
    for feature, importance in sorted(
        metrics["feature_importance"].items(),
        key=lambda item: item[1],
        reverse=True,
    )[:10]:
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    train_model("data/gold/machine_features_v2.csv")