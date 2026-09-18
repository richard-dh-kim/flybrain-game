# ADR 0003: Use WSL2 Ubuntu as the canonical development environment

Date: 2026-09-18

Status: Accepted. Supersedes the environment split in ADR 0001.

## Context

ADR 0001 provisionally split browser development onto native Windows and ML
work into WSL2 Ubuntu. Since then, the native Rust-to-WASM/browser spike passed,
while the remaining Phase 0 risks are Linux-first Python, PyTorch/CUDA, upstream
connectome tooling, and cross-language simulation parity.

Maintaining two authoritative toolchains and building Linux dependencies against
a repository stored under `/mnt/c` would add path, filesystem-performance, and
dependency-drift risks without a clear project benefit.

## Decision

Use WSL2 Ubuntu as the canonical environment for:

- the Git working tree;
- Rust and WebAssembly builds;
- Node, TypeScript, Phaser, and Vite;
- Python, PyTorch, CUDA, training, and evaluation;
- tests, benchmarks, generated artifacts, and development scripts.

Store the working tree in the WSL Linux filesystem, for example
`~/projects/flybrain-game`, rather than under `/mnt/c`.

Use Windows Chrome or Edge to exercise the development server through
`localhost`. Native Windows builds may be used as an occasional compatibility
check, but they are not a second canonical development environment.

## Consequences

- Browser, simulation, and ML development share one filesystem and toolchain
  environment.
- The main development environment more closely matches Linux CI and the
  upstream connectome/PyTorch ecosystem.
- Ubuntu must have its own Git authentication, Node, Rust, wasm-bindgen,
  Python, and project dependencies.
- Generated Windows artifacts and dependency directories must not be copied
  into Ubuntu; they will be rebuilt there.
- WSL2 boot and NVIDIA GPU visibility are prerequisites. The prior
  `HCS_E_HYPERV_NOT_INSTALLED` failure must be resolved before migration is
  considered complete.
- Windows browser behavior still needs explicit testing before release.
