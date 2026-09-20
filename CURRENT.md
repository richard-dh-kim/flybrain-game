# Current Project Status

Last updated: 2026-09-20

This is the first file a new Codex session should read. Then read
`PROJECT_PLAN.md` and the active architecture decisions in `docs/decisions/`.

## Current milestone

Phase 0 (environment and feasibility) and Phase 1 (playable gray box) are
complete. The user's pointer playtest accepted the larger fly, hand-only
presentation, synchronized slap, collision behavior, and policy switching.
Phase 2 mechanics validation is technically complete, with human-path
collection continuing during playtests. Phase 3 conventional learned baselines
are active, and the first Phase 4 full MaleCNS controller has passed its
single-seed trained-versus-initialized gate. Phase 5 now has a validated packed
u16 model and an experimental full-connectome WebGPU game mode.

Completed:

- Canonical repository and development environment moved to the WSL2 Ubuntu
  filesystem.
- Python 3.12 environment locked to PyTorch 2.8.0+cu128; CUDA sees the RTX 4060
  Ti with 16,380 MiB VRAM.
- Upstream licenses reviewed and the dependency decision recorded in
  `docs/decisions/0004-upstream-connectome-runtime.md`.
- Hash-verified MaleCNS v1.0 graph prepared with 165,122 retained neurons and
  25,563,197 retained edges.
- Bounded full-graph forward/backward benchmark passed on the local GPU. See
  `docs/experiments/phase-0-connectome-feasibility.md`.
- Deterministic fixed-point Rust simulation core and native tests.
- Rust-to-WASM and Rust-to-Python bindings.
- The 387-tick movement replay and a 1,868-tick versioned movement/action
  replay pass at every exact integer checkpoint in native Rust, WebAssembly,
  and Python. The action replay covers a complete 30-second survival, all five
  phases on both hands, restart, active contact, and a collision-ending round.
- Phaser gray-box scene with a larger background-filling fly, table, two
  hand-only attackers, player, and pointer/touch movement.
- Authoritative flight-area bounds, normalized pointer movement, synchronized
  hand tracking, and a coordinated two-hand wind-up/slap/contact/recovery/
  cooldown cycle.
- One-hit 30-second round, immediate restart, and selectable idle,
  current-position chase, and predictive-intercept policies.
- Toggleable `H` debug overlay showing velocity, policy target, collision
  shapes, hand phases, phase ticks, cooldowns, and active contact.
- Automated policy comparison over 36 turning trajectories. Predictive
  interception wins 13, ties 18, and loses 5; mean time to hit is 101.4 ticks
  versus 177.3 for current-position chasing (ratio 0.572).
- Playwright coverage for load, pointer movement, policy and debug switching,
  coordinated collision, and restart passes in headless Chromium.
- Versioned observation and action v1 exposes exact player acceleration, palm
  velocities, hand state, round state, and synchronized attack state from the
  same Rust simulation through Python and WebAssembly.
- Deterministic stationary, constant-direction, turning, seeded random
  waypoint, evasive, and recorded-human trajectory sources.
- A timing-aware interception expert hits 9/9 simple held-out trajectories and
  12/12 total evaluation episodes with zero cooldown violations. The evasive
  and one random-waypoint episode each cause a completed, visually attributable
  miss before the second slap lands. See
  `docs/experiments/phase-2-expert-baseline.md`.
- Versioned Parquet logging includes complete observations, destinations,
  expert actions, accepted/invalid strikes, closest approach, misses, and
  outcomes. A standalone HTML replay supports frame-by-frame inspection.
- Browser rounds record exact pointer destinations; press `E` to export a CSV
  compatible with the recorded-human curriculum loader.
- A draft brain-activity display contract reserves synchronized model/tick
  telemetry without pretending that scripted policy state is neural activity.
- The varied-start behavior-cloning split contains 8,847 training ticks from
  78 episodes and 2,877 held-out ticks from 33 episodes. Training and
  validation use disjoint starting-position grids, turn periods, offsets, and
  random seeds.
- A 6,531-parameter MLP and 19,203-parameter GRU were retrained on those starts.
  Across 204 trajectory/start combinations, the MLP hits 200 and the GRU hits
  188. Their held-out target RMSE values are 13.17 and 22.67 pixels.
- The varied-start GRU hits 188/204 with hidden state carried and 0/204 when
  state is reset each tick. The MLP hits 200/204 with or without reset, so the
  result establishes state dependence for this GRU rather than a general
  recurrent advantage.
- The frozen GRU runs locally in the browser on key `4`. With `H` enabled, a
  labeled panel draws all 64 real hidden-state values as computed model
  activity. It remains the earlier fixed-start browser checkpoint; browser
  automation covers model load, recurrence, and collision.
- The TypeScript GRU matches PyTorch over 140 held-out recurrent ticks, all
  outputs, and all hidden values with maximum absolute error `3.708e-6`.
- The minimal MIT-licensed sparse-gradient core from pinned `flyhard` commit
  `328906f4a0e62c8f9fc18805cf6edae6989b82a5` is adapted locally with numerical,
  finite-difference, state-carry, and immutable-topology tests.
- The Phase 4 controller injects the same 33 game features through a frozen
  signed mapping into 6,365 `vnc_sensory` neurons. One measured graph update is
  carried per game tick, and a trainable readout uses only 708 `vnc_motor`
  neurons. There is no observation-to-action bypass.
- A 25,730,446-parameter single-seed checkpoint completed 1,800 optimizer
  steps, about 3.22 passes over the varied-start training rows. Its topology
  remained unchanged and all parameter groups had finite nonzero gradients.
- Offline carried-state target error improved to 56.76 pixels, compared with
  175.13 pixels when neural state is reset each tick. The full graph uses 3.45
  GB peak allocated GPU memory during batch-four training.
- An offline strike threshold of 0.90 produced zero live strikes. A separate
  12-episode closed-loop sweep selected 0.50 with 10/12 hits; freezing that
  threshold then produced 33/33 hits on the varied-start validation suite,
  versus 0/33 and zero strikes for the calibrated untrained initialization.
- Resetting the trained controller's 165,122-neuron state before every decision
  also produces 0/33 hits and zero strikes at the same frozen 0.50 threshold.
  The carried and reset policies use identical weights, inputs, and readout.
- Full-graph batch-one inference on the selected validation run measured 5.40
  ms median and 6.06 ms p95 on the RTX 4060 Ti. The selected 99 MB training
  checkpoint has SHA-256
  `f795652899f3df47f43eda72bf6de4bf3f43337a28865723762b450e0f91daa7`.
- The result is documented in
  `docs/experiments/phase-4-male-cns-v1.md`. It uses one training seed and has
  no shuffled-topology or random-sparse comparison, so it does not support a
  claim that biological topology is better.
- The Phase 5 inference-only format fuses learned edge gains and leaks, strips
  training state, records all source and array hashes, and passes full-state
  PyTorch-to-NumPy and NumPy-to-Rust parity. Its exact float32 arrays occupy
  196.34 MiB unpacked and 92.84 MiB with measured Brotli quality 5.
- A dependency-free Rust runtime compiles natively and for WASM. On the Ryzen 5
  7600 under WSL2, its first single-thread float32 baseline measured 20.73 ms
  median and 22.44 ms p95, missing the 16.67 ms 60 Hz budget. See
  `docs/experiments/phase-5-packed-inference-baseline.md`.
- Row-wise u16 weights passed all 2,877 held-out rows with no strike-decision
  changes and preserved the 33/33 live-suite aggregate without zeroing any
  measured edge. Packed v2 is 148.21 MiB unpacked and 71.36 MiB with measured
  Brotli quality 5. The 8-bit candidate was rejected because it zeroed 214,917
  edges and changed 13 held-out strike decisions.
- The u16 runtime measured 19.26 ms median natively and 21.40 ms in Node V8
  WebAssembly. Both miss 60 Hz, so the next backend experiment is WebGPU in an
  on-demand worker. See `docs/experiments/phase-5-u16-candidate.md`.
- A WebGPU worker verifies all package hashes, consumes packed u16
  weights directly, carries GPU-resident state, and matches the native 60-tick
  output in headless Chromium. SwiftShader correctness passed at 106.8 ms
  median.
- A user-run 300-tick hardware benchmark identified the `nvidia lovelace`
  adapter and measured 5.0 ms median and 6.9 ms p95, passing the 16.67 ms
  60 Hz budget. Its final outputs agree with the native reference within
  `3.4e-6`; local load, checksum verification, and upload took 621.5 ms. See
  `docs/experiments/phase-5-u16-candidate.md`.
- A normal visit now starts loading the full 165,122-neuron controller locally.
  The compact GRU controls the hands during that load, so the round starts
  immediately, and remains the automatic fallback on unsupported browsers or
  when the model package is missing. The technical view shows verified loading
  progress, backend, adapter, format version, and a short package hash.
- Live full-controller inference no longer pauses mouse movement when a brain
  result misses a render frame. The simulation continues at its fixed rate
  using the latest hand action, with at most one brain update in flight. This
  prevents a slow frame from building a backlog and samples the newest game
  state when the worker becomes available. Switching development controllers
  cancels work from the previous recurrent epoch. The technical status line
  reports brain median/p95 time, end-to-end p95, queue depth, simulation rate,
  and render rate. The `H` overlay labels the green mouse destination separately
  from the yellow hand target.
- An always-visible display below the playfield resembles the optic lobes,
  central brain, and descending motor region of a fruit fly. With the full
  controller active, its lights show 64 real values sampled from the same
  GPU-resident recurrent state that produced the current hand action: 16
  sensory neurons, 32 evenly spaced whole-graph neurons, and 16 motor neurons.
  The anatomy, sample positions, and faint links are explicitly labeled as an
  illustrative layout rather than physical neuron coordinates. During loading
  or fallback, the same display honestly identifies and shows the compact GRU.
- The full MaleCNS controller is the product default and the GRU is its
  automatic fallback. `H` reveals technical statistics, hitboxes, targets, and
  the development-only scripted controller keys `1` through `3`; hiding it
  removes the complete debug panel and its background. For the current
  comparison build, `4` directly selects the compact GRU and `5` returns to the
  full MaleCNS controller even when debug details are hidden.
- The activity display makes the active source explicit. Full mode uses a green
  `FULL MALECNS · LIVE` badge and sensory/network/motor labels. Loading says the
  CPU-only compact GRU is controlling the hands; automatic fallback uses a
  `COMPACT GRU · CPU FALLBACK` badge, while manual mode `4` says
  `COMPACT GRU · CPU ONLY`. Both compact states label all lights as 64 GRU units
  and explain that they are hidden-state values rather than anatomical neuron
  samples.
- The first Phase 6 readability pass animates the fly independently of the
  authoritative simulation: its pupils follow the player, lock onto a committed
  target, and its head and expression anticipate the slap. A shrinking target,
  strike lines, hand motion streaks, impact burst, and short hit/miss labels make
  all five hand phases easier to read. The fly shakes and sweats after a miss.
  Browser automation verifies that wind-up appears before collision and that a
  direction change during commitment can produce visible miss feedback.
- Rust formatting, Clippy, workspace tests, Python replay test, WASM replay
  test, policy comparison, browser automation, TypeScript checking, and the
  production browser build pass in WSL2.

A first connectome-constrained gameplay result and exact portable inference
reference now exist. The genuine packed MaleCNS controller is the browser
default; the 19,203-parameter conventional GRU covers loading and unsupported
devices. This is still a one-seed model, and the ignored 148 MiB local model
assets must be prepared before the full brain can load from a fresh checkout.

## Local environment note

The Ubuntu environment currently lacks the system `build-essential`,
`pkg-config`, `python3.12-venv`, and Chromium runtime packages because
installing them requires a local sudo password. The completed checks used a
project-local ignored Zig 0.16 toolchain, bootstrapped pip into the local
virtual environment, and extracted the three required headless Chromium
libraries under `.tools/`.

For the normal development setup, run once:

```bash
sudo apt update
sudo apt install -y build-essential pkg-config python3.12-venv
PLAYWRIGHT_BROWSERS_PATH=.tools/ms-playwright npx playwright install chromium
sudo npx playwright install-deps chromium
```

The project does not depend on Zig after those packages are installed.

## Verification

After installing the system packages and sourcing Cargo's environment:

```bash
npm ci
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
npm run build
npm run test:wasm-parity
npm run test:policies
npm run test:expert
npm run test:gru-parity
npm run test:python
npm run test:browser
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements/phase-0.txt
.venv/bin/maturin develop
PYTHONPATH=python .venv/bin/python -m unittest discover -s crates/game-python/tests -v
```

The Vite development server runs in Ubuntu and is reachable from a Windows
browser through `localhost`.

## Immediate next work

Continue model validation alongside Phase 6 presentation work:

1. Complete the seed-1701 shuffled-presynaptic and matched random-sparse runs
   defined in `docs/experiments/phase-4-topology-controls.md`, then repeat all
   three topology conditions across additional predeclared seeds before making
   a biological-topology comparison.
2. Export several real rounds with `E` and use human or learned-policy failures
   for DAgger-style data collection without committing personal raw recordings
   by default.
3. Recheck full-brain hardware timing after major presentation changes and test
   an intentionally missing model to confirm the visible CPU fallback. A phone
   browser is the remaining practical device check.
4. Treat the current cute visual style as the intended art direction. Additional
   onboarding, accessibility options, and sound are optional polish rather than
   release gates.
