# Maintenance Policy

## Risk bands

Machines are classified into one of four risk bands:

- Low: failure_probability below 0.25
- Medium: failure_probability from 0.25 to 0.50
- High: failure_probability from 0.50 to 0.75
- Critical: failure_probability above 0.75

## Required action by risk band

Low-risk machines should remain in normal operation.

Medium-risk machines should be reviewed during the next planned maintenance window.

High-risk machines should be inspected within the next shift.

Critical-risk machines should be inspected as soon as possible. If the machine is involved in a production-critical process, maintenance and operations leaders should jointly decide whether to pause production.

## Sensor-based guidance

If vibration is a top risk factor, inspect bearings, alignment, mounting, and rotating components.

If temperature is a top risk factor, inspect cooling systems, lubrication, airflow, and motor load.

If pressure is a top risk factor, inspect valves, seals, pumps, and filters.

If power draw is a top risk factor, inspect motors, electrical load, friction, and abnormal startup behavior.

If error codes are a top risk factor, review recent controller logs and previous maintenance history.

## Escalation

Critical-risk machines should be escalated to the maintenance supervisor.

Critical-risk machines with two or more elevated sensor categories should also be escalated to reliability engineering.

High-risk machines that remain high-risk for more than two scoring runs should be reviewed for preventive replacement or deeper root-cause analysis.