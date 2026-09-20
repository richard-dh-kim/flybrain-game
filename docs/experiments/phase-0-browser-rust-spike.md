# Phase 0 Browser/Rust Compatibility Spike

Date: 2026-09-18

Status: Passed for native Rust, browser compilation, and Python parity.

## Scope

This bounded spike answers only whether a deterministic Rust simulation core can be tested natively, exported through `wasm-bindgen`, and consumed by the TypeScript/Phaser build. It is not a connectome benchmark and makes no inference-performance claim.

## Environment

- Rust/Cargo 1.98.1, `x86_64-pc-windows-msvc`
- `wasm32-unknown-unknown` target
- `wasm-bindgen` and CLI 0.2.128
- Visual Studio C++ Build Tools 17.14.37710.0
- Node.js 22.12.0 and npm 10.9.0
- Phaser 4.2.1, TypeScript 7.0.2, Vite 7.3.6

## Implemented slice

- Integer fixed-point positions and velocities at 1,024 units per pixel.
- Fixed 60 Hz browser stepping with a bounded catch-up accumulator.
- Acceleration, deceleration, vector speed cap, and world-bound collision.
- A `wasm-bindgen` `Simulation` boundary exposing step, tick, and player position.
- A Phaser gray-box scene whose player position is sourced from the Rust/WASM state.

## Checks

| Check | Result |
| --- | --- |
| `cargo fmt --all -- --check` | Passed |
| `cargo clippy --workspace --all-targets -- -D warnings` | Passed |
| `cargo test --workspace` | Passed: 4 tests |
| Optimized `wasm32-unknown-unknown` compile | Passed |
| Integrated `npm run build` | Passed |
| Generated simulation WASM | 23.85 kB raw; 10.11 kB gzip as reported by Vite |
| Total main JavaScript bundle | 1,394.30 kB raw; 374.29 kB gzip as reported by Vite |

The JavaScript bundle is dominated by the current full Phaser import. This is acceptable for the first gray box, but bundle splitting and lazy loading should be revisited before deployment.

## Behavioral tests

- Replaying the same 1,000-input sequence twice produces identical state.
- Velocity stays within the configured vector speed cap.
- The player cannot leave world bounds.
- The player decelerates exactly to rest when input stops.

## Completed follow-up

- Added PyO3/maturin against the Python 3.12 training environment.
- Ran the same 387-tick golden movement fixture through native Rust, WASM, and
  Python with exact integer checkpoint agreement.
- Reproduced the bounded upstream full-graph forward/backward workload and
  recorded timing and memory in `phase-0-connectome-feasibility.md`.
