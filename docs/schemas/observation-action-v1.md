# Observation, Action, and Trajectory Schemas v1

Status: active for Phase 2 data generation
Last updated: 2026-09-19

The Rust simulation is authoritative. All positions, velocities, and
accelerations in the training interface use signed fixed-point simulation
units, where 1 rendered pixel equals 1,024 units. Python receives these values
without floating-point conversion.

## Observation v1

`OBSERVATION_SCHEMA_VERSION = 1` contains:

- simulation tick;
- player position, velocity, and one-tick velocity delta (acceleration);
- left and right palm position and one-tick velocity delta;
- each hand's phase, phase tick, cooldown remaining, and active-contact flag;
- synchronized-attack flag;
- round status, elapsed ticks, and remaining ticks.

Hand phases are `0=track`, `1=wind-up`, `2=strike`, `3=impact-hold`, and
`4=recover`. Round states are `0=active`, `1=hit`, and `2=survived`.

Acceleration is computed after boundary clamping, so it includes a player's
forced stop at the edge of the flight area. Palm velocity is the exact change
in palm position during the last simulation tick.

## Action v1

`ACTION_SCHEMA_VERSION = 1` contains an integer target `(x, y)` and a Boolean
`strike`. The target guides both palms during tracking. A strike command is
accepted only while both hands are in track with zero cooldown; it freezes one
shared target for the synchronized slap.

The expert predicts the first active-contact tick from the configured wind-up
and strike timing, then projects the player's current velocity to that tick.
Targets are clamped to the player's legal flight area.

## Trajectory and dataset v1

`TRAJECTORY_SCHEMA_VERSION = 1` supplies one pointer destination per tick.
Implemented sources are stationary, constant-direction, timed cardinal or
diagonal turns, seeded random waypoints, a telegraph-reactive evasive bot, and
recorded human pointer destinations.

`DATASET_SCHEMA_VERSION = 1` flattens the complete pre-step observation and
adds:

- episode ID and trajectory name;
- pointer destination;
- expert target and strike command;
- whether the strike was accepted or violated availability;
- post-step round and attack state;
- current and running-minimum palm-to-player edge gap;
- completed-miss and hit flags.

Parquet files repeat all four version numbers in columns and store observation,
action, and dataset versions in file metadata. Any semantic field change must
increment the corresponding schema version. Adding derived experiment metrics
does not change the observation or action version.

## Commands

From the repository root, after building the Python extension:

```bash
PYTHONPATH=python .venv/bin/python scripts/evaluate-expert.py \
  --metrics runs/phase-2-expert.metrics.json \
  --replay-html runs/phase-2-expert-replay.html

PYTHONPATH=python .venv/bin/python scripts/generate-expert-dataset.py \
  --out data/expert-v1.parquet \
  --metrics runs/expert-v1.metrics.json
```

The generated HTML replay has frame-by-frame scrubbing and displays the player,
both palm hitboxes, hand phases, predicted target, closest gap, and terminal
state.
