# FlyBrain Swatter

> Continuing development? Start with [CURRENT.md](CURRENT.md) for the active
> milestone, Ubuntu/WSL migration notes, verification steps, and next work.

FlyBrain Swatter is a browser game in which a tiny winged human dodges the hands of a giant cartoon fruit fly. The long-term controller is a recurrent neural network constrained by the measured connectivity of the adult fruit-fly brain.

The project is deliberately being built in layers:

1. A deterministic, fun gray-box game.
2. A scripted predictive opponent that provides a difficulty baseline and training labels.
3. Conventional learned baselines.
4. A connectome-constrained controller, evaluated honestly against those baselines.
5. Static browser deployment with local inference when feasible.

No biological-performance or speed claim should be treated as a result until it appears with reproducible measurements in this repository.

## Current status

Phase 0 and the playable Phase 1 gray box are complete. The full retained
MaleCNS graph passed a bounded forward/backward test on the local RTX 4060 Ti.
Phase 2 now has versioned simulation observations and actions, deterministic
curricula, a timing-aware expert, Parquet logging, and visual replay. The first
Phase 3 MLP and GRU controllers are trained, and the GRU runs locally in the
browser. Phase 4 now has a first full MaleCNS controller: its fixed measured
topology carries 165,122 neuron states and produces valid closed-loop actions,
with 33/33 hits on the varied-start validation suite versus 0/33 for its
untrained initialization. Phase 5 now has a checksummed inference-only package
and a Rust core that agrees with PyTorch, but the exact 196 MiB float32 model
misses the native 60 Hz CPU target. A validated 148 MiB u16 package preserves
33/33 live hits; its CPU and WASM paths also miss 60 Hz, while a checksummed
WebGPU worker passes software-adapter correctness and a real RTX browser run at
5.0 ms median / 6.9 ms p95. Press `5` to load and play against that genuine
165,122-neuron controller locally. This is still a one-seed result. See
[CURRENT.md](CURRENT.md) for the exact handoff.

## Run the current prototype

```bash
npm ci
npm run dev
```

The production build regenerates the Rust WebAssembly package before Vite bundles the site:

```bash
npm run build
```

The current build uses pointer or touch movement. A larger fly fills the
background while two floating hands track from opposite sides and commit to a
coordinated slap. It includes collision, a 30-second round, and immediate
restart. Run `npm run prepare:connectome-web` once when the ignored model assets
are absent. A normal visit starts loading the full MaleCNS controller through
WebGPU; the small GRU controls the hands during loading and automatically
remains active on unsupported devices. A dedicated or integrated WebGPU device
can run the full model, and an RTX card is not required.

The fly watches the player and locks onto the committed slap point. The
wind-up target shrinks before the hands close, while motion streaks, impact
bursts, and hit/miss reactions make the attack timing visible without changing
the deterministic simulation or collision rules.

The always-visible brain below the playfield shows 64 live values. With the
full controller active, those are 16 sensory, 32 whole-graph, and 16 motor
samples from its 165,122-neuron state. The brain shape is anatomical, while the
sample positions and faint links are illustrative. Press `H` to show technical
timing, velocity, targets, collision shapes, phases, and cooldowns. Debug mode
also enables scripted controller keys `1` through `3` and `E` trajectory
export. During development, `4` selects the CPU-only compact GRU and `5`
selects the full MaleCNS controller directly so their behavior and activity
labels can be compared. Automatic CPU fallback covers browsers without a
usable WebGPU adapter as well as model loading, verification, or inference
failures. The green debug marker is the mouse destination and the yellow marker
is the hand target.

Run the automated gameplay gates with:

```bash
npm run test:wasm-parity
npm run test:policies
npm run test:expert
npm run test:gru-parity
npm run test:python
npm run test:browser
npm run export:connectome-packed
npm run test:connectome-packed-parity
npm run test:connectome-rust-parity
npm run evaluate:connectome-quantization
npm run export:connectome-quantized
npm run test:connectome-quantized-rust-parity
npm run benchmark:connectome-wasm
npm run test:connectome-webgpu
```

The first topology controls use the same seed and two-stage training schedule:

```bash
npm run train:connectome-shuffled-pilot
npm run train:connectome-shuffled-refine
npm run train:connectome-random-pilot
npm run train:connectome-random-refine
```

The shuffled control preserves both incoming degrees and the outgoing-degree
multiset. The random-sparse control preserves each incoming degree but draws
new unique sending partners. See
`docs/experiments/phase-4-topology-controls.md` for the fixed comparison rules
and current evidence.

## Intended stack

- TypeScript, Vite, and Phaser for the browser presentation.
- Rust for deterministic simulation, hand paths, collision, replay, and portable inference support.
- WebAssembly plus a Web Worker for browser-side execution.
- Python, PyTorch, and CUDA for training and evaluation.
- `wasm-bindgen` and PyO3/maturin so browser and training code share the same simulation contract.

The working title, package names, and final visual identity may change.
