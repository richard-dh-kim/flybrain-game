# Phase 5 Packed Inference Baseline

Date: 2026-09-20
Status: reference export and parity passed; single-thread CPU budget missed

## Result

The selected Phase 4 controller now exports to a versioned inference-only
package. The export fuses normalized synapse counts with learned edge gains,
fuses the bounded learned leaks, downcasts CSR indices from 64 to 32 bits, and
removes optimizer and training state. It retains exact float32 inference
values. The generated package is ignored, while its schema, exporter, loader,
tests, hashes, and compact measurements are versioned.

The package contains 165,122 neurons, 25,563,197 measured edges, and ten typed
arrays totaling 205,876,086 bytes (196.34 MiB). Standard transfer compression
reduces those arrays to 111.55 MiB with gzip level 6 or 92.84 MiB with Brotli
quality 5. The latter remains too large for a comfortable public game load,
especially on mobile. Runtime memory remains the full unpacked size plus neural
state, scratch state, and loader overhead.

The package ID is
`1e78b5264d23ebcb080c2299eacf65c72fc76ed51b52c7052b097abc04074787`.
It commits to every array checksum, shape, dtype, model dimension, interface
rule, output rule, source checkpoint, graph hash, and provenance field.

## Numerical agreement

The memory-mapped NumPy reference was compared with the selected PyTorch model
over four carried-state recurrent ticks. Maximum target, strike-logit, and full
neural-state absolute errors were `2.98e-8`, `2.38e-7`, and `1.19e-7`.

The dependency-free Rust runtime was then compared with the NumPy reference
over 60 carried-state ticks. Maximum target, strike-logit, and full-state errors
were `2.09e-7`, `1.20e-7`, and `1.19e-7`. Both comparisons passed the declared
`2e-5` absolute tolerance. The Rust core also compiles for
`wasm32-unknown-unknown`.

## Native CPU baseline

The first Rust runtime uses one thread and exact float32 weights. On an AMD
Ryzen 5 7600 under WSL2, a release build produced:

| Measurement | Result |
|---|---:|
| Unpacked model load | 194.40 ms |
| Inference median | 20.73 ms |
| Inference p95 | 22.44 ms |
| Inference maximum | 23.11 ms |
| 60 Hz frame budget | 16.67 ms |

This path misses the frame budget before adding worker messaging, feature
construction, or browser overhead. It should remain the correctness baseline,
but it is not yet suitable as the public execution path. A Web Worker would
protect rendering responsiveness while still allowing decisions to arrive too
late for one update per 60 Hz game tick.

## Next gate

The follow-up u16 candidate and its CPU/WASM measurements are recorded in
`phase-5-u16-candidate.md`. It passes offline and fixed-suite validation while
reducing transfer size, but still misses 60 Hz. That result advances the next
execution experiment to an on-demand WebGPU worker.

No quantized model should replace the selected controller based on numerical
error alone. It must preserve the 33/33 fixed-suite result, strike behavior,
and target accuracy within declared limits before browser integration.

## Commands

```bash
npm run export:connectome-packed
npm run test:connectome-packed-parity
npm run test:connectome-rust-parity
```

The detailed compact record is stored beside this report in
`phase-5-packed-inference-baseline.metrics.json`.
