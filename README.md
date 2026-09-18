# FlyBrain Swatter

FlyBrain Swatter is a browser game in which a tiny winged human dodges the hands of a giant cartoon fruit fly. The long-term controller is a recurrent neural network constrained by the measured connectivity of the adult fruit-fly brain.

The project is deliberately being built in layers:

1. A deterministic, fun gray-box game.
2. A scripted predictive opponent that provides a difficulty baseline and training labels.
3. Conventional learned baselines.
4. A connectome-constrained controller, evaluated honestly against those baselines.
5. Static browser deployment with local inference when feasible.

No biological-performance or speed claim should be treated as a result until it appears with reproducible measurements in this repository.

## Current status

Phase 0 is in progress. The repository, native Rust toolchain, first deterministic simulation step, Rust-to-WASM boundary, and gray-box browser shell are working. The Python boundary and bounded connectome feasibility benchmark remain. The detailed product and implementation plan is in [PROJECT_PLAN.md](PROJECT_PLAN.md). Historical research and earlier decisions are preserved in [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md).

## Run the current prototype

```powershell
cmd.exe /d /c npm install
cmd.exe /d /c npm run dev
```

The production build regenerates the Rust WebAssembly package before Vite bundles the site:

```powershell
cmd.exe /d /c npm run build
```

The current gray box supports WASD and arrow-key movement. Mouse movement, the articulated attacking hand, collision, and the predictive opponent are the next gameplay slice.

## Intended stack

- TypeScript, Vite, and Phaser for the browser presentation.
- Rust for deterministic simulation, collision, inverse kinematics, replay, and portable inference support.
- WebAssembly plus a Web Worker for browser-side execution.
- Python, PyTorch, and CUDA for training and evaluation.
- `wasm-bindgen` and PyO3/maturin so browser and training code share the same simulation contract.

The working title, package names, and final visual identity may change.
