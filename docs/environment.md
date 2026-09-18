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
| Git | 2.53.0.windows.2 | Ready |
| Node.js | 22.12.0 | Ready |
| npm | 10.9.0 | Ready through `cmd.exe`; PowerShell script execution blocks `npm.ps1` |
| Python | CPython 3.14.3 | Installed, but not the target ML interpreter |
| PyTorch | Not installed | Pending |
| Rust / Cargo | 1.98.1 / 1.98.1 | Ready |
| wasm-bindgen CLI | 0.2.128 | Ready |
| WSL2 | Ubuntu registered as a stopped WSL2 distribution; VM start currently returns `HCS_E_HYPERV_NOT_INSTALLED` | Restart/firmware check pending |
| Visual Studio C++ Build Tools | 17.14.37710.0 | Ready |
| Microsoft Edge | 153.0.4234.32 | Recorded |
| Google Chrome | 153.0.8010.48 | Recorded |
| Phaser | 4.2.1 | Locked |
| TypeScript | 7.0.2 | Locked |
| Vite | 7.3.6 | Locked |

Python 3.14 is intentionally not selected for the ML environment. The upstream reproduction documentation exercises Python 3.12 and PyTorch 2.8.0 with CUDA 12.8, so Phase 0 should first reproduce that environment rather than debug an avoidable version mismatch.

## Provisional environment split

Use two cooperating environments unless the compatibility spike disproves the choice:

- **Native Windows:** Node/Vite/Phaser, Rust, `wasm-bindgen`, browser testing, and ordinary game development.
- **WSL2 Ubuntu 24.04:** Python 3.12, PyTorch/CUDA, upstream connectome tooling, training, and bounded benchmarks.

This follows the upstream Linux-first runtime while keeping the browser workflow simple. The simulation contract and serialized trajectories must behave identically across the two environments. Large ML datasets and training runs should live in the Linux filesystem if `/mnt/c` I/O proves materially slower; only source, small fixtures, metrics, and export artifacts need to cross the boundary.

## Phase 0 checks still required

- Restart Windows, then verify that the registered Ubuntu distribution boots and can see the NVIDIA GPU. If the same hypervisor error remains, verify CPU virtualization in firmware.
- Create the Python 3.12 environment and record exact locked dependencies.
- Add the Python binding and compare its first deterministic replay with native Rust and WASM.
- Reproduce a bounded upstream full-graph forward/backward test without beginning a long training run.
- Record startup/compile time, forward/backward latency, peak VRAM, host RAM, and numerical checks.
