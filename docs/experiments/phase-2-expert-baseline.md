# Phase 2 Expert Baseline

Date: 2026-09-19
Status: passed as an initial mechanics and logging baseline

## Question

Can a deterministic expert use the versioned simulation observation to time a
synchronized slap, hit simple unseen motion, miss for an understandable reason
against an evasive controller, and produce replayable training rows without
issuing invalid commands?

## Method

The expert computes the first active-contact tick from the hand wind-up and
strike configuration. It projects current player velocity to that tick, clamps
the result to the flight area, and requests a strike only when both hands are
in track with zero cooldown.

The fixed evaluation suite contains 12 episodes: stationary, four constant
directions, four timed turning paths, two held-out seeded random-waypoint paths,
and one bot that reverses when the hands commit. Every simulation transition is
logged from the Python binding using observation, action, trajectory, and
dataset schema v1.

Command:

```bash
PYTHONPATH=python .venv/bin/python scripts/evaluate-expert.py \
  --metrics docs/experiments/phase-2-expert-baseline.metrics.json \
  --replay-html runs/phase-2-expert-replay.html \
  --replay-episode evasive
```

## Result

- Simple held-out trajectories hit: 9/9.
- All trajectories hit: 12/12.
- Mean hit time: 79.7 ticks.
- Accepted strikes: 14.
- Completed misses: 2: one seeded random-waypoint episode and the
  telegraph-reactive evasive episode.
- Strike availability violations: 0.

The evasive episode survives the first committed target by reversing during
wind-up and is hit by the second attempt at tick 142. The HTML replay exposes
the target, phases, collision geometry, closest gap, and terminal state for
visual inspection.

## Limits

This is a scripted expert, not a trained controller. Twelve episodes establish
pipeline correctness and a first mechanics baseline; they do not establish
general gameplay skill. The current evaluation starts every episode from the
same state, and the first behavior-cloning dataset remains too small and too
strike-imbalanced for Phase 3 training. Human pointer recordings and broader
trajectory/strike timing coverage are the next inputs.
