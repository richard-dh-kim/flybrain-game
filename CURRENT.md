# Current Project Status

Last updated: 2026-09-18

This is the first file a new Codex session should read. Then read
`PROJECT_PLAN.md` and the active architecture decisions in `docs/decisions/`.

## Current milestone

Phase 0 (environment and feasibility) is in progress.

Completed:

- Repository initialized and published at
  `https://github.com/richard-dh-kim/flybrain-game`.
- Deterministic Rust simulation core with fixed-point player movement.
- Native tests for deterministic stepping, speed limits, bounds, and
  deceleration.
- `wasm-bindgen` boundary and an integrated Rust-to-WASM build.
- Phaser gray-box scene with a placeholder fly, table, hands, player, and
  keyboard movement.
- Native Rust, WASM, TypeScript, and production browser builds passed in the
  original Windows workspace.

Not yet completed:

- Python 3.12/PyTorch/CUDA training environment.
- PyO3/maturin `game-python` binding.
- Exact native/WASM/Python golden-trajectory parity test.
- Upstream `flyhard`/MaleCNS license and dependency decision.
- Bounded full-graph forward/backward feasibility benchmark.
- Phase 1 attacking-hand gameplay.

No trained controller or connectome performance result exists yet.

## Development-environment decision

The canonical development environment is moving from the Windows filesystem to
WSL2 Ubuntu. Source code, Git, Rust, Node, Python, builds, tests, and ML work
should run from the Linux filesystem. Windows remains the host for Chrome/Edge
browser testing and other desktop applications.

Do not continue active Linux development from `/mnt/c`. Clone the repository
into a path such as `~/projects/flybrain-game` after WSL2 is working. See
`docs/decisions/0003-ubuntu-first-development.md`.

## Safe transfer procedure

The current committed state is already on GitHub. From Ubuntu:

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/richard-dh-kim/flybrain-game.git
cd flybrain-game
```

Clone rather than copying generated directories from Windows. Reinstall or
rebuild `node_modules`, `target`, generated WASM, Python environments, and other
ignored artifacts inside Ubuntu.

Keep the Windows checkout temporarily until the Ubuntu checkout has passed all
verification. Do not make commits in both checkouts during the transition.

## First Ubuntu-session checks

Before changing project code, record the results of:

```bash
git status --short --branch
uname -a
nvidia-smi
python3 --version
node --version
npm --version
rustc --version
cargo --version
```

The earlier Windows investigation found an RTX 4060 Ti with 16 GB VRAM. WSL2
previously failed to start with `HCS_E_HYPERV_NOT_INSTALLED`, so WSL boot and GPU
visibility must be confirmed rather than assumed.

After installing the Ubuntu-side toolchain, run:

```bash
npm ci
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
npm run typecheck
npm run build
```

The Vite development server can run in Ubuntu and be opened from a Windows
browser through `localhost`.

## Immediate next work

Finish Phase 0 before expanding the game:

1. Make WSL2 Ubuntu boot reliably and confirm `nvidia-smi` sees the GPU.
2. Create and lock the Python 3.12 environment, then confirm PyTorch can use
   CUDA.
3. Review the upstream `flyhard`/MaleCNS licenses and choose whether to depend
   on, fork, vendor, or minimally reimplement the required runtime pieces.
4. Run only the bounded full-graph forward/backward feasibility benchmark and
   record latency, VRAM, host RAM, startup/compilation time, and numerical
   checks. Do not start a long training run.
5. Add the PyO3/maturin boundary and prove exact deterministic state parity
   across native Rust, WASM, and Python.

After the Phase 0 gate passes, begin the next gray-box gameplay slice: shared
mouse/keyboard movement, two-link foreleg IK, and one attacking hand with
wind-up, strike, collision, recovery, and cooldown.
