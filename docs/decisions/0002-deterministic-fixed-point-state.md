# ADR 0002: Use fixed-point state for deterministic gameplay

Date: 2026-09-18

Status: Accepted for the gray-box simulation.

## Context

The same game environment must support browser play, headless training, replay fixtures, and comparisons between policies. Native Rust, WebAssembly, and Python need to agree on state transitions closely enough that a recorded action sequence is a useful scientific and debugging artifact.

Floating-point results can vary subtly across compilation targets and optimization settings. Those differences are avoidable for basic 2D movement, timers, kinematics, and hitboxes.

## Decision

Represent authoritative world positions and per-tick velocities with integers at 1,024 simulation units per rendered pixel. Step the simulation at a fixed 60 Hz. Rendering may interpolate or convert to floating point, but it must not feed those rounded values back into authoritative state.

Use integer algorithms for magnitude clamps and other authoritative calculations where practical. Policy observations can still be normalized to floating point at the model boundary.

## Consequences

- Replays can compare exact state rather than tolerances for the basic mechanics.
- Browser frame rate does not change game speed.
- Units and overflow bounds need deliberate documentation.
- More complex geometry may eventually require carefully specified fixed-point approximations or explicitly tolerance-tested floating-point calculations.
