# Conventional Policy Features v1

Status: active for the Phase 3 baselines and Phase 4 MaleCNS controller
Last updated: 2026-09-20

Feature schema v1 deterministically maps observation v1 to 33 floating-point
values:

- round tick fraction;
- normalized player position, velocity, and acceleration;
- for each hand: normalized position and velocity, five one-hot phases, phase
  progress, cooldown, and active-contact flag;
- synchronized-attack flag and remaining-round fraction.

Positions use the legal flight rectangle. Player velocity uses the configured
five-pixel-per-tick speed cap; palm velocity uses 32 pixels per tick as a fixed
scale. Phase tick and cooldown use their maximum configured v1 durations.

Targets are normalized over the same flight rectangle. The learned policy
outputs target coordinates and a strike logit. The runtime accepts a learned
strike only when both hands are tracking with zero cooldown and no synchronized
attack is already active.

The ordered feature names are stored in every PyTorch checkpoint and the
browser export. The MaleCNS controller freezes a balanced random signed mapping
from these 33 features into the 6,365 neurons annotated as `vnc_sensory`; its
outputs come only from the 708 neurons annotated as `vnc_motor`. A field,
order, normalization, or semantic change requires a new feature schema version.
