# Development Environment Baseline

Recorded: 2026-09-18 (Asia/Seoul)

This is the measured starting point for Phase 0. It is an environment record, not a performance result.

## Confirmed hardware and host

| Component | Observed value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4060 Ti |
| Reported GPU memory | 16,380 MiB |
| NVIDIA driver | 591.86 |
| Host OS build | Microsoft Windows NT 10.0.26200.0 |
| C: storage | 1,862.01 GiB total; 967.15 GiB free at inspection |
| Host RAM | 31.15 GiB reported usable physical memory |

The earlier 5060 Ti assumption in the handoff was incorrect and is superseded by this direct measurement.

## Confirmed tools

| Tool | Observed value | Status |
| --- | --- | --- |
| Git | 2.43.0 | Ready in WSL2 |
| Node.js | 22.13.0 | Ready in WSL2 |
| npm | 10.9.2 | Ready in WSL2 |
| Python | CPython 3.12.3 | Project target |
| PyTorch | 2.8.0+cu128 | CUDA verified |
| CUDA runtime reported by PyTorch | 12.8 | Ready |
| Rust / Cargo | 1.98.1 / 1.98.1 | Ready in WSL2 |
| wasm-bindgen CLI | 0.2.128 | Ready |
| maturin | 1.15.0 | Ready in the project virtual environment |
| WSL2 | Ubuntu 24.04.1, Linux 6.18.33.2-microsoft-standard-WSL2 | Ready |
| Visual Studio C++ Build Tools | 17.14.37710.0 | Ready |
| Microsoft Edge | 153.0.4234.32 | Recorded |
| Google Chrome | 153.0.8010.48 | Recorded |
| Phaser | 4.2.1 | Locked |
| TypeScript | 7.0.2 | Locked |
| Vite | 7.3.6 | Locked |
| Playwright | 1.63.0, Chromium 153.0.8010.12 | Browser automation verified |

PyTorch reports the RTX 4060 Ti as CUDA compute capability 8.9 with
17,175,150,592 bytes of device memory. CUDA access is hidden inside the normal
Codex command sandbox but succeeds for approved commands outside that sandbox
and in the user's ordinary WSL terminal.

## Canonical environment

WSL2 Ubuntu is the canonical environment for source, builds, tests, Python,
CUDA, and generated artifacts. Windows remains the host for Chrome or Edge
browser checks through `localhost`. This supersedes the provisional split in
ADR 0001; see ADR 0003.

## Phase 0 result

The environment, full-graph feasibility benchmark, Python binding, and exact
native/WASM/Python replay checks are complete. See
`docs/experiments/phase-0-connectome-feasibility.md` and
`docs/experiments/phase-0-browser-rust-spike.md`.
