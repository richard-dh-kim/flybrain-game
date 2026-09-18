# ADR 0001: Split browser development and ML training environments

Date: 2026-09-18

Status: Provisionally accepted; revisit after the Phase 0 compatibility spike.

## Context

The shipped game must run as a static browser application, while the current upstream connectome reproduction path is Linux-first and includes PyTorch/CUDA and custom GPU work. The host currently has a ready Node installation, no Rust toolchain, CPython 3.14 without PyTorch, and WSL2 without a Linux distribution.

## Decision

Use native Windows for the TypeScript/browser application and portable Rust core. Use WSL2 Ubuntu 24.04 for the Python/PyTorch training environment and upstream reproduction work.

Keep the boundary explicit:

- Rust owns deterministic game stepping, kinematics, collision, and replay formats.
- TypeScript owns browser input, rendering, screens, audio, and UI.
- Python owns training, evaluation, and model export.
- Versioned observations, actions, trajectories, and checkpoints are the interfaces between them.

## Consequences

- The game loop remains easy to run and debug in a normal Windows browser.
- The ML environment stays close to upstream Linux instructions and CUDA assumptions.
- Cross-environment determinism must be tested rather than assumed.
- File-system placement and WSL path performance need a small benchmark before large datasets are downloaded.
- If WSL GPU access or custom kernels fail, the fallback is a dedicated Linux environment rather than forcing the whole browser project into Linux.
