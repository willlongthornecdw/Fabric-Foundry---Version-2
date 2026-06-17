# Fabric + Foundry Predictive Maintenance Lab — V2

This project is a synthetic predictive maintenance lab for a manufacturing scenario.

It builds on the lesson from V1: **having sensor telemetry does not automatically mean you have a predictive data product**.

In V1, the project generated machine telemetry, created a gold feature table, trained a model, and discovered that the model was not predictive. The model produced a ROC AUC of approximately `0.476`, which is worse than random guessing.

V2 improves the data product by adding a second source system: a maintenance and failure event log. This makes it possible to engineer a proper future-looking prediction label:

> Will this machine fail within the next 24 hours?

The result is a much stronger predictive maintenance workflow.

## Business Scenario

A manufacturing customer wants to get more value from data coming off the factory floor.

The desired business outcome is to predict maintenance needs earlier, reduce unplanned downtime, and maximize uptime across production lines.

In a real enterprise environment, the relevant data would likely come from multiple systems:

* Sensor telemetry from PLCs, SCADA, historians, or IoT systems
* Maintenance records from a CMMS or EAM system
* Failure events from work order history
* Production context from MES
* Downtime and quality data from operational systems

This lab simulates the first two:

1. Machine sensor telemetry
2. Maintenance and failure event history

## Core Lesson

V1 showed this:

```text
Sensor telemetry only
+ naive failure label
= weak predictive signal
```

V2 shows this:

```text
Sensor telemetry
+ maintenance/failure event history
+ future-looking label engineering
= much stronger predictive signal
```

The model did not improve because the AI became magical.

The model improved because the data product became more complete and the prediction target became more meaningful.

## Architecture

Conceptual architecture:

```text
Synthetic machine sensor telemetry
        +
Synthetic maintenance/failure event log
        ↓
Raw data layer
        ↓
Gold feature table
        ↓
Future-looking failure label
        ↓
Predictive maintenance model
        ↓
Model evaluation
```

In a real Microsoft architecture, this could map to:

```text
Manufacturing floor systems
        ↓
Microsoft Fabric ingestion
        ↓
OneLake / Fabric Lakehouse
        ↓
Bronze: raw telemetry and event records
        ↓
Silver: cleaned and standardized machine data
        ↓
Gold: model-ready predictive maintenance features
        ↓
Predictive model using Azure ML / Foundry-adjacent ML workflow
        ↓
Future Foundry agent or assistant layer
```

## Important Clarification: This Version Is Traditional Machine Learning

This repo is currently a classic machine learning / data science workflow.

It does **not** use an LLM yet.

The model is a `RandomForestClassifier` trained on structured numeric features. It predicts whether a machine is likely to fail within the next 24 hours.

The LLM layer would come later.

A future version could use an LLM or Azure AI Foundry agent to:

* explain model predictions
* summarize risk drivers
* recommend maintenance actions
* generate technician-friendly work order notes
* capture human feedback
* create private evals from technician corrections and actual outcomes

The clean division of labor is:

```text
Traditional ML model:
Predict failure risk.

LLM / agent:
Explain the risk, recommend action, and capture feedback.
```

## Medallion Pattern in This Repo

This project uses a simplified medallion architecture.

### Raw Layer

Location:

```text
data/raw/
```

Generated files:

```text
machine_sensor_readings.csv
machine_sensor_readings.parquet
maintenance_failure_events.csv
maintenance_failure_events.parquet
```

The telemetry file represents manufacturing floor sensor readings.

Example fields:

```text
timestamp
machine_id
line_id
temperature_c
vibration_mm_s
pressure_psi
rpm
power_kw
oil_quality_pct
error_count
runtime_hours
```

The event file represents maintenance and failure history.

Example fields:

```text
event_timestamp
machine_id
line_id
event_type
failure_type
maintenance_action
downtime_minutes
```

In a real implementation, these would likely come from different source systems.

For example:

```text
Telemetry source:
PLC / SCADA / historian / IoT platform

Event source:
CMMS / EAM / work order system / downtime log
```

### Gold Layer

Location:

```text
data/gold/
```

Generated files:

```text
machine_features_v2.csv
machine_features_v2.parquet
```

The gold layer is the model-ready feature table.

It combines telemetry with event history and creates features such as:

```text
temp_rolling_4h
vibration_rolling_4h
pressure_rolling_4h
power_rolling_4h
errors_rolling_4h
temp_rolling_24h
vibration_rolling_24h
pressure_rolling_24h
power_rolling_24h
errors_rolling_24h
temp_change_4h
vibration_change_4h
pressure_change_4h
temp_change_24h
vibration_change_24h
pressure_change_24h
sensor_stress_index
historical_failure_count
historical_maintenance_count
failure_within_24h
```

The key label is:

```text
failure_within_24h
```

This is not generated directly by the sensor stream.

It is engineered by looking at the separate maintenance/failure event log and asking:

> From this timestamp, does this machine have a failure event within the next 24 hours?

That is the central improvement in V2.

## Repository Structure

```text
fabric-foundry-toy-v2-events/
  README.md
  requirements.txt
  .gitignore

  src/
    generate_data.py
    feature_engineering.py
    train_model.py

  foundry/
    agent/
      system_prompt.md
      sample_questions.md
    knowledge/
      maintenance_policy.md
      escalation_matrix.md
    tools/
    evals/

  scripts/
    export_foundry_snapshot.py

  data/
    raw/
      .gitkeep
    gold/
      .gitkeep

  outputs/
    .gitkeep
    foundry_machine_risk_snapshot.json

Generated training data and model artifacts are intentionally ignored by Git.

A user should clone the repo, run the scripts, and generate the local data/model outputs themselves.

The foundry/ directory is the beginning of the AI application layer. It does not yet deploy a real Azure AI Foundry agent, but it contains the prompts, grounding documents, and future tool/evaluation structure that will be used to operationalize the predictive model.

## How to Run

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install pandas numpy scikit-learn matplotlib pyarrow joblib notebook ipykernel
```

Optional:

```powershell
pip freeze > requirements.txt
```

### 3. Generate synthetic source data

```powershell
python src\generate_data.py
```

Expected outputs:

```text
data/raw/machine_sensor_readings.csv
data/raw/machine_sensor_readings.parquet
data/raw/maintenance_failure_events.csv
data/raw/maintenance_failure_events.parquet
```

### 4. Build the gold feature table

```powershell
python src\feature_engineering.py
```

Expected outputs:

```text
data/gold/machine_features_v2.csv
data/gold/machine_features_v2.parquet
```

### 5. Train the predictive maintenance model

```powershell
python src\train_model.py
```

Expected outputs:

```text
outputs/predictive_maintenance_model_v2.joblib
outputs/model_metrics_v2.json
```

6. Export a Foundry-facing machine risk snapshot


After training the model, generate a small operational output file that represents what the AI application layer would consume:
```text
python scripts\export_foundry_snapshot.py
```
Expected output:
```text
outputs/foundry_machine_risk_snapshot.json
```
This file represents the handoff from the predictive maintenance model to the future Foundry agent/tool layer.

The snapshot includes fields such as:

machine_id
plant_id
failure_probability
risk_band
top_risk_factors
sensor_summary
recommended_action
work_order_priority

In a real implementation, this type of output could be stored in a Fabric Lakehouse, Warehouse, SQL endpoint, or operational API. In this lab, it is written to JSON so the next step can focus on the Foundry application pattern.

## Model Results

V1 result:

```text
ROC AUC: 0.476
```

V2 result after adding maintenance/failure event history and engineering a future-looking label:

```text
ROC AUC: 0.953
```

V2 confusion matrix:

```text
[[11832, 1164],
 [ 1311, 6813]]
```

Interpreting the confusion matrix:

```text
True negatives:  11,832
False positives:  1,164
False negatives:  1,311
True positives:   6,813
```

The model catches most of the actual `failure_within_24h` cases while still producing some false positives.

In a real predictive maintenance workflow, this would create a business discussion around threshold tuning:

* A lower threshold catches more potential failures but creates more inspections.
* A higher threshold reduces false alarms but may miss more failures.
* The right threshold depends on the cost of downtime versus the cost of preventive maintenance.

## Feature Importance

Top feature importances from the V2 model:

```text
runtime_hours: 0.2402
pressure_rolling_24h: 0.0800
temp_rolling_24h: 0.0782
vibration_rolling_24h: 0.0757
power_rolling_24h: 0.0745
errors_rolling_24h: 0.0691
vibration_rolling_4h: 0.0630
temp_rolling_4h: 0.0547
power_rolling_4h: 0.0396
pressure_rolling_4h: 0.0384
```

The model is learning that older machines with worsening 24-hour operating patterns are more likely to fail.

This is directionally consistent with a predictive maintenance use case.

However, `runtime_hours` is very influential in this synthetic dataset. In a real customer environment, that would need to be validated carefully. Runtime may be truly predictive, or it may be acting as a proxy for missing information such as asset class, maintenance history, duty cycle, operating environment, or production load.

## Why V2 Improved

V2 improved because it changed the data product.

V1 had telemetry and a naive label.

V2 added a separate maintenance/failure event source and created a true future-looking label.

The improvement was not simply “better modeling.” It was better data engineering:

```text
V1:
The model had data, but not a trustworthy business outcome.

V2:
The model had telemetry, event history, and a properly engineered target.
```

This is the core enterprise AI lesson:

> Predictive AI depends on the quality of the data product underneath it.

## What This Demonstrates

This repo demonstrates:

* synthetic manufacturing data generation
* lakehouse-style medallion thinking
* raw telemetry data
* separate maintenance/failure event history
* gold feature engineering
* future-looking label creation
* traditional machine learning for predictive maintenance
* model evaluation
* the business importance of data engineering before AI

## What This Does Not Yet Demonstrate

This repo now includes the beginning of a Foundry operationalization layer, but it does not yet include a live Azure AI Foundry deployment.

It does not yet demonstrate:

* real Microsoft Fabric deployment
* real OneLake tables
* deployed Azure AI Foundry project
* deployed Foundry agent
* live tool calling from an agent
* LLM-based explanation or recommendation against live model outputs
* technician feedback capture
* private evals
* reinforcement learning
* production-grade time-series modeling

The current foundry/ directory is intentionally lightweight. It contains the initial agent prompt, sample questions, grounding documents, and placeholder folders for tools and evals.

The next implementation step is to expose the machine-risk snapshot through a small local API and then use that API as the conceptual tool layer for a Foundry agent.

## Local Machine Risk API

This repo includes a small FastAPI app that exposes the machine-risk snapshot as an operational API.

Run the API:

    uvicorn foundry.tools.machine_risk_api:app --reload

Then open:

    http://127.0.0.1:8000/docs

Available endpoints:

    GET /
    GET /machines
    GET /machines/high-risk
    GET /machines/{machine_id}
    GET /plants/{plant_id}/risk-summary

This API represents the future tool layer for an Azure AI Foundry agent. In a real implementation, the agent would call a secured API, Azure Function, Fabric SQL endpoint, or other operational data service. In this lab, the API reads from:

    outputs/foundry_machine_risk_snapshot.json

## Possible Next Version

A future V3 could add the AI application layer.

Example architecture:

```text
Predictive maintenance model
        ↓
Machine risk score
        ↓
LLM / Foundry assistant
        ↓
Maintenance recommendation
        ↓
Technician approval / edit / rejection
        ↓
Feedback and actual outcome captured
        ↓
Private eval dataset
```

The LLM would not replace the predictive model. It would operationalize the model output by turning risk scores into explainable recommendations and capturing human feedback.

## Strategic Learning Goal

This project is part of a broader applied AI learning path.

The ultimate goal is to understand how a company could build a learning loop where:

```text
manufacturing data
+ lakehouse architecture
+ predictive analytics
+ human technician feedback
+ private evals
= continuously improving AI capability
```

This is the enterprise AI pattern: not just using a model, but building an organizational learning system around the model.

## Disclaimer

This project uses synthetic data and a toy model. It is not production-ready and should not be used for real maintenance decisions.

The purpose is educational: to understand the workflow, architecture, and common failure modes of predictive maintenance projects.
