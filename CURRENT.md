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
single-seed trained-versus-initialized gate. Phase 5 browser export has not
started.

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
- Rust formatting, Clippy, workspace tests, Python replay test, WASM replay
  test, policy comparison, browser automation, TypeScript checking, and the
  production browser build pass in WSL2.

A first connectome-constrained gameplay result now exists, but it runs only in
the Python/CUDA evaluation path. The current browser mode `4` still uses the
19,203-parameter conventional GRU and must not be presented as the fly
connectome. Phase 5 must define and validate a local browser inference format
before the MaleCNS policy can become a playable mode.

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

Continue validation while beginning the Phase 5 export work:

1. Replicate the conventional and MaleCNS runs across additional seeds, then
   add shuffled-topology and matched random-sparse controls before making any
   topology comparison.
2. Export several real rounds with `E` and use human or learned-policy failures
   for DAgger-style data collection without committing personal raw recordings
   by default.
3. Define the checksummed Phase 5 packed inference format, strip training-only
   state, and benchmark a Rust/WASM CPU prototype before deciding whether a
   WebGPU path is needed.
4. Expose synchronized sampled MaleCNS activity to the existing `H` panel only
   after browser numerical parity is established.
