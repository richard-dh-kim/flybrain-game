# FlyBrain Swatter: Current Project Plan

Last updated: 2026-09-18

Status: Phase 0 in progress. No game, trained controller, or performance result described below exists yet.

This document records the current product direction. Where it conflicts with `PROJECT_HANDOFF.md`, this document takes precedence.

## Product decision

Build one polished browser game before considering a collection of minigames.

The player is a tiny winged human. A giant cartoon fruit fly sits behind a table, facing the player, and reaches into the play space with one visible pair of articulated forelegs/hands. A MaleCNS-constrained recurrent controller will eventually decide where and when those hands strike.

The immediate hook is:

> You are a tiny flying human trying to survive while an AI based on the measured wiring of roughly 165,000 fruit-fly neurons controls a giant fly's hands and tries to swat you.

`FlyBrain Swatter` is a working title, not a final product name.

## Scope choice: complex 2D, not 3D

Use layered 2D presentation with a false sense of depth:

- Giant fly centered in the background, viewed from the front.
- Table spanning the bottom of the frame.
- One pair of forelegs anchored near the fly and reaching toward the foreground.
- Tiny winged human moving within a bounded airspace above the table.
- Layered shadows, scale changes, impact frames, screen shake, and squash-and-stretch to suggest depth.
- Analytic two-link inverse kinematics for each foreleg rather than rigid-body simulation.

The visual target is an original hand-drawn cartoon with the immediacy of an old browser/Flash game: bold silhouettes, inked outlines, limited palette, slightly imperfect animation, exaggerated wind-ups, and fast restarts. It may draw from broad rubber-hose-animation and early web-game conventions, but it should not reproduce another game's characters or exact visual identity.

Full 3D, rigid-body hands, finger articulation, and physically simulated cloth or wings are explicitly out of scope for the first release.

## Screen composition

Render the scene in these conceptual layers:

1. Background wall, props, and subtle ambient animation.
2. Giant fly head and torso behind the table.
3. Table surface and the fly's resting limb segments.
4. Active limb segments and hand/palm hit shapes.
5. Player character and wing trail.
6. Impact effects, shadows, telegraphs, and HUD.

The background fly should react visibly to the controller: eyes tracking the player, a short anticipatory lean, arm wind-up, strike, frustration after a miss, and recovery. These animations communicate model intent without pretending that every cosmetic motion comes from the neural network.

## Initial game loop

- A round begins after a short countdown.
- The player moves inside a fixed rectangular flight area.
- The fly tracks the player and periodically commits one foreleg to a predicted interception point.
- A committed strike has a readable wind-up and limited ability to redirect.
- Contact during the active strike ends the round.
- A miss incurs a visible recovery and cooldown.
- The score is survival time.
- Restart is immediate; the default target round length is 30 seconds and will be tuned through playtesting.
- Winning initially means surviving the target duration. Endless and daily-seed modes can follow later.

One-hit rounds keep the premise legible and produce a clean time-to-first-hit metric. Lives, shields, collectibles, obstacles, dashes, and power-ups are deferred until the basic chase is fun.

## Movement and input

Support mouse and keyboard through one shared movement interface.

### Mouse and touch

- Pointer location is a desired destination, not the player's actual position.
- The character accelerates toward it subject to maximum speed and acceleration.
- This preserves intuitive mouse control without allowing teleportation or infinite-speed cursor movement.
- Touch uses the same desired-destination model.

### Keyboard

- WASD and arrow keys produce a normalized desired movement vector.
- Keyboard and pointer control share the same acceleration, drag, and maximum-speed constraints.

Start with mouse as the default because it suits a short browser game. Keep WASD available for accessibility and for players who prefer direct directional control.

The player should be faster than either hand in instantaneous movement. The hand competes through reach, prediction, timing, and coordinated attacks rather than by simply exceeding the player's top speed.

## Hand mechanics

Each foreleg uses a small explicit state machine:

```text
rest -> track -> wind_up -> strike -> impact/hold -> recover -> rest
```

Initial behavior:

- Both forelegs are visible.
- Only one foreleg attacks at a time in the MVP.
- A deterministic selector chooses an available hand based on reach and cooldown.
- The second hand supports the composition and telegraphing but does not independently attack yet.
- The controller chooses a target and strike intent; deterministic mechanics choose the hand and execute the motion.

Candidate tuning ranges, to be validated rather than treated as final:

- Controller decision rate: 5-10 Hz.
- Render and physics rate: fixed 60 Hz.
- Wind-up: 150-300 ms.
- Strike travel: 150-350 ms depending on distance.
- Active contact window: 50-120 ms.
- Recovery/cooldown: 500-1,000 ms.

During wind-up and strike, target correction must be capped. Otherwise the hand becomes a pursuit cursor and predictive interception stops mattering.

## Simulation contract

Rendering, input, controller policy, and deterministic simulation must be separate from the start.

### Observation supplied to policies

The first structured observation should include:

```text
player position and velocity
player acceleration or recent motion summary
left and right hand positions and velocities
foreleg joint angles
attack phase and time remaining
cooldowns
reachable-workspace information
normalized round time
```

### Policy action

The first policy action should stay small:

```text
target_x
target_y
strike_probability
```

The game code, not the neural network, owns hand selection, inverse kinematics, hitboxes, cooldown enforcement, and animation. A later experiment may add hand selection as a fourth output.

### Determinism

- Fixed simulation timestep.
- Explicit seeded random number generator.
- No wall-clock time inside simulation logic.
- Serializable initial state, player inputs, policy actions, and outcomes.
- Replaying the same seed and input/action stream must reproduce the same result.

## Technology choice

Use a web-first stack:

- TypeScript, Vite, and Phaser for scenes, input, audio, asset loading, and 2D rendering.
- Rust for the authoritative deterministic simulation and, later, the sparse inference runtime.
- `wasm-bindgen` to expose the Rust core to the browser.
- PyO3/maturin bindings so Python training and evaluation can call the same Rust environment.
- PyTorch/CUDA for behavior cloning, DAgger, experiments, and the full MaleCNS training path.
- A Web Worker for browser inference so the renderer remains responsive.
- A CPU WebAssembly inference path first; a WebGPU/WGSL sparse path only after profiling demonstrates that it is needed.

Do not use Phaser's general physics engine for authoritative gameplay. The required mechanics are simple enough to implement deterministically in the Rust core, while Phaser remains the presentation layer.

Phaser is intentionally chosen because it is a maintained browser-first 2D framework with TypeScript support. Ruby is not useful for the client-side game or neural inference; a small leaderboard backend can be chosen independently later.

## Repository shape

Target structure:

```text
flybrain-game/
  apps/
    web/                 TypeScript + Phaser browser application
  crates/
    game-core/           deterministic game state, movement, IK, collision
    game-wasm/           wasm-bindgen browser boundary
    game-python/         PyO3/maturin training boundary
    brain-runtime/       future packed sparse inference runtime
  python/
    flybrain_training/   policies, datasets, training and evaluation
  assets/
    source/              editable art/audio sources
    generated/           spritesheets and optimized web assets
  configs/               versioned game/model/experiment configurations
  tests/
    replays/             golden deterministic replay fixtures
  docs/
    decisions/           short architecture decision records
    experiments/         benchmark and evaluation reports
```

Keep the training stack and browser runtime in one repository initially so model formats, environment versions, and replay fixtures cannot silently drift apart.

## Build order: game first, but only to a stable gray-box gate

Do not fully illustrate and polish the game before training. Use this order:

```text
gray-box deterministic game
    -> scripted/expert opponent
    -> playtest and lock mechanics
    -> conventional learned baseline
    -> connectome integration
    -> visual and audio polish
    -> public deployment
```

The model requires a working environment and an expert capable of producing labels. Conversely, expensive art should wait until player speed, reach, timing, hitboxes, and screen composition are stable.

## Implementation phases and gates

### Phase 0: environment and feasibility

Deliverables:

- Initialize the independent repository and basic documentation.
- Record GPU model/VRAM, driver, Python, PyTorch, CUDA, Rust, Node, and browser versions.
- Review upstream licenses and decide how the MaleCNS runtime will relate to `flyhard`.
- Run a bounded upstream full-graph inference/training benchmark on the confirmed RTX 4060 Ti 16 GB.
- Measure forward latency, backward latency, VRAM, host RAM, compilation time, and numerical checks.
- Create a tiny Rust-to-WASM and Rust-to-Python proof of concept.

Gate:

- The full graph completes a bounded forward/backward test without an unsafe dense intermediate.
- A single deterministic Rust step produces identical results through native, WASM, and Python bindings.

Do not begin a long training run during this phase.

### Phase 1: gray-box playable prototype

Use circles, capsules, lines, and placeholder sprites only.

Deliverables:

- Player acceleration, speed cap, bounds, mouse input, and WASD input.
- Front-facing fly/table composition with two articulated placeholder forelegs.
- Two-link IK and reachable-workspace clamping.
- One attacking hand with wind-up, strike, hitbox, recovery, and cooldown.
- One-hit round, timer, restart, fixed seeds, and input replay.
- Scripted policies: idle, chase-current-position, and simple predictive intercept.
- Debug overlays for velocity, predicted target, reach, collision shapes, and state transitions.

Gate:

- A new player can understand the objective without explanation after seeing one round.
- The predictive expert clearly outperforms current-position chasing on turning trajectories.
- The game replays deterministically in native and browser builds.
- Mouse and keyboard players are both constrained by the same motion limits.

### Phase 2: mechanics validation and expert

Deliverables:

- Tune player acceleration, hand speed, commitment, telegraphing, and cooldown.
- Build curriculum trajectories: stationary, constant velocity, turning, random waypoint, evasive bot, and recorded human play.
- Implement an expert that estimates hand travel time and future player position.
- Log full observations, expert actions, closest approach, hits, misses, and cooldown violations.
- Add automated expert evaluation and visual replay inspection.

Gate:

- The expert succeeds reliably on simple held-out trajectories but remains beatable by humans.
- Misses feel attributable to player movement rather than collision bugs.
- Observation and action schemas are versioned and considered stable enough to generate training data.

### Phase 3: conventional learned baseline

Deliverables:

- Behavior-cloning dataset generated from the expert and curriculum.
- Small MLP diagnostic model.
- GRU recurrent baseline with state carried between decisions.
- Training, validation, checkpoint, and evaluation pipelines.
- Reset-state versus carried-state comparison.
- Browser playback of a frozen learned policy, even if inference initially runs through a development service.

Gate:

- A learned baseline reliably reproduces interception on held-out scripted trajectories.
- Losses, recurrent state, replay, and evaluation agree between offline and live execution.
- The baseline is good enough to reveal game/interface bugs before introducing the full connectome.

### Phase 4: MaleCNS controller

Deliverables:

- Integrate the benchmarked sparse MaleCNS core.
- Document sensory injection, recurrent updates, state persistence, and motor readout.
- Train with short TBPTT windows and conservative batch sizes.
- Check finite gradients and compare initial versus trained behavior.
- Run DAgger rounds after behavior cloning works.
- Preserve every environment version, seed set, checkpoint hash, and configuration.

Gate:

- The controller produces valid actions and beats its untrained initialization on held-out trajectories.
- Full-graph inference latency is measured at batch sizes relevant to local training and browser deployment.
- No claim of biological-topology superiority is made without replicated controls.

### Phase 5: browser inference export

Deliverables:

- Strip optimizer and training-only state.
- Fuse inference-time base weights and learned gains where mathematically valid.
- Define a versioned, checksummed packed model format.
- Implement Rust/WASM CPU inference in a Web Worker.
- Measure model download, decompressed memory, latency, and numerical agreement with PyTorch.
- Quantize only after comparing gameplay and held-out metrics against the reference checkpoint.
- Add WebGPU/WGSL only if the CPU path misses the latency target.

Gate:

- The public game performs genuine local inference from the frozen controller.
- The main render loop remains responsive during inference.
- Model origin, version, execution backend, and any reduced/lite mode are visible to the user.

### Phase 6: art, sound, and presentation

Deliverables:

- Original giant-fly character, winged-human character, table, background, and UI.
- Foreleg sprites that preserve the same IK anchors and collision geometry as the prototype.
- Wind-up, strike, hit, miss, frustration, idle, eye-tracking, and restart animations.
- Limited palette, inked outlines, subtle paper/grain treatment, and intentionally snappy timing.
- Original sound effects and short looping music appropriate to an early browser-game feel.
- Accessibility options: reduced shake, mute, high-contrast hitbox/telegraph mode, keyboard remapping where practical.

Gate:

- Art improves readability rather than hiding hitboxes or wind-ups.
- The first round begins quickly and restart friction is minimal.
- The neural visualization is labeled as computed model activity and does not obscure gameplay.

### Phase 7: evaluation and public release

Deliverables:

- Frozen public model version and fixed evaluation suite.
- Conventional GRU, shuffled-topology, random sparse, and reset-state controls as compute permits.
- Multiple training seeds before comparative claims.
- Static web deployment with cached model shards.
- Optional anonymous leaderboard, daily seed, and replay sharing.
- Rate limiting and deterministic score verification for any public leaderboard.
- Public method, limitations, model card, licenses, and attribution.

Gate:

- A cold browser load, cached reload, unsupported-device fallback, and full round are tested on representative desktop and mobile hardware.
- Published numbers are generated by committed configurations and reproducible evaluation commands.
- The public page explains the game in one sentence before presenting technical detail.

## Test strategy

Start tests with the simulation rather than the rendering:

- Player speed and acceleration limits.
- Boundary behavior.
- IK reachability and joint continuity.
- Hand phase transition timing.
- Collision only during the active strike window.
- Cooldown enforcement.
- Fixed-seed reproducibility.
- Native/WASM/Python replay equivalence.
- Observation normalization and action clamping.
- Checkpoint/model-format validation and checksums.
- Numerical agreement between PyTorch and browser inference within declared tolerances.

Use browser automation for loading, starting, playing a scripted input trace, restarting, switching input modes, and exercising inference fallbacks.

## Deferred features

- Two independently attacking hands.
- Direct joint-torque control.
- Pixel observations and optic-neuron mapping.
- Obstacles, food, hiding zones, and power-ups.
- Multiple lives or boss phases.
- Online adaptation to an individual player.
- Full 3D presentation.
- Additional minigames.
- Texas Hold'em, chess, or an arcade hub.

These are expansion paths, not MVP requirements.

## Immediate next work

1. Record the confirmed RTX 4060 Ti 16 GB and complete the development toolchain.
2. Review the upstream licenses and reproduce the bounded full-graph benchmark.
3. Initialize the Rust/TypeScript/Python workspace.
4. Prove one shared Rust movement step through native, WASM, and Python.
5. Build the gray-box screen with a moving player, table, fly silhouette, and one articulated attacking foreleg.
6. Add the simple predictive expert before producing final art or training the connectome.
