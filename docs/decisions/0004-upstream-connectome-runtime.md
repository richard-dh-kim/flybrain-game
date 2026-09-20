# ADR 0004: Pin upstream benchmarks and vendor only the minimal connectome core

Date: 2026-09-18

Status: Accepted for Phase 0 and the first MaleCNS integration.

## Context

`flyhard` provides a tested sparse recurrent MaleCNS implementation and a
bounded full-graph benchmark. Its original code is MIT licensed. The MaleCNS
v1.0 connectivity data is separately licensed under CC BY 4.0 and requires
attribution and provenance to follow derived graphs and checkpoints.

`fly-self-driving` is also MIT licensed and contains a faster CUDA backward
path that caches the transposed sparse graph. Applying that patch silently
would make it harder to compare this machine with the published `flyhard`
pilot.

The game does not need FlyGym, MuJoCo, CARLA, vehicle assets, or the rest of
the upstream experiment stack.

## Decision

- Reproduce the Phase 0 benchmark from `flyhard` commit
  `328906f4a0e62c8f9fc18805cf6edae6989b82a5` without modifying its numerical
  core.
- Treat the `fly-self-driving` CUDA optimization at commit
  `3516a094a6462e2a4ab7cc517286d5c3bc3ff63e` as a separate follow-up benchmark,
  with results labeled independently from the unmodified baseline.
- Do not add either repository as a production Git submodule or runtime
  dependency.
- When Phase 4 begins, adapt only the sparse connectome core and its numerical
  tests into this repository. Retain the upstream MIT copyright and license
  notice with that code and record the source commit.
- Acquire MaleCNS data through a hash-verified script. Keep raw data, derived
  graphs, and checkpoints out of Git. Preserve the MaleCNS source, CC BY 4.0
  license, selection rule, and hashes in every derived artifact manifest.
- Do not import the upstream body, driving, rendering, or asset dependencies
  unless a later decision demonstrates that the game needs them.

## Consequences

- Phase 0 results remain comparable to the published upstream pilot.
- The shipped project avoids a large, unrelated dependency tree.
- The later training runtime can incorporate the proven sparse-gradient
  technique while retaining clear attribution and version history.
- Data and model releases must include the MaleCNS attribution even though the
  surrounding game and runtime code use permissive software licenses.
