# Phase 5 Row-wise u16 Candidate

Date: 2026-09-20
Status: validation passed; hardware WebGPU 60 Hz budget passed

## Selection

Two row-wise unsigned weight encodings were tested against the exact float32
package. Each postsynaptic row uses its own maximum-derived scale.

The 8-bit candidate retained 33/33 live hits, but it rounded 214,917 measured
edges to zero and changed 13 held-out strike decisions at the frozen 0.50
threshold. It is rejected. Keeping it would conflict with the declared fixed
topology and would require a stronger behavioral case than this experiment
provides.

The 16-bit candidate rounded no measured edge to zero and changed none of the
2,877 held-out strike decisions. Target RMSE was 56.7609 pixels, compared with
56.7610 for float32. Its maximum target, strike-logit, and episode-final state
differences were `6.91e-6`, `2.62e-5`, and `1.21e-5`.

On the fixed 33-episode live suite it reproduced the float32 aggregate exactly:
33 hits, mean hit tick 116.45, 64 accepted strikes, and 31 completed misses.
The per-episode hit ticks differ by at most one tick in several borderline
cases, which is expected from small recurrent numerical changes.

## Package

The accepted candidate is packed format v2, package SHA-256
`0f5baf90bf5bed5802931b289d551474524547872ceb3eff657e25d9d8e54a37`.
Its export is gated on the hashed quantization evaluation. It stores one u16
weight per edge and one float32 scale per neuron, while retaining all v1 source
hashes, licenses, interface fields, and array checksums.

| Representation | Unpacked | Gzip 6 | Brotli quality 5 |
|---|---:|---:|---:|
| Exact float32 v1 | 196.34 MiB | 111.55 MiB | 92.84 MiB |
| Row-wise u16 v2 | 148.21 MiB | 89.23 MiB | 71.36 MiB |

This is a useful reduction, but 71 MiB remains appropriate only as an optional
on-demand model download. The GRU or another small controller is still needed
when WebGPU is unavailable or the user declines the larger model.

## Portable runtime

The Rust runtime consumes u16 weights directly and applies one scale per row.
Across 60 carried-state ticks it agrees with an edge-by-edge dequantized NumPy
reference within `1.67e-6` on the strike logit and `1.20e-7` on the complete
neural state.

On the Ryzen 5 7600 under WSL2, 300 native ticks measured 19.26 ms median and
20.99 ms p95. The same core compiled to WebAssembly and executed by Node V8 at
21.40 ms median and 23.35 ms p95, with about 357 MiB process RSS after both JS
and WASM model copies. Both miss the 16.67 ms budget before worker messaging or
game work.

The CPU/WASM path remains a parity reference and possible reduced-rate fallback.

The WebGPU worker prototype now verifies the manifest and every array, uploads
packed u16 weights without expanding them, carries ping-pong neural state, and
runs sparse recurrence plus motor readout on the GPU. A 60-tick headless
Chromium run passed output checks against the native result. That environment
used Google's SwiftShader software adapter, where median latency was 106.8 ms;
this validates shader execution only and is not evidence about the RTX 4060 Ti.
The same 300-tick benchmark was then run manually in a normal Windows browser.
The adapter identified itself as `nvidia lovelace`, confirming that it used the
RTX rather than SwiftShader. It measured 5.0 ms median and 6.9 ms p95, passing
the 16.67 ms 60 Hz budget with substantial room. The final target differed from
the native reference by about `1.1e-6`, and the strike logit by about `3.3e-6`.
One 279 ms maximum outlier occurred; the p95 shows that it was not sustained.
The cold local load, hash verification, and GPU upload took 621.5 ms.

This result clears the performance gate for experimental game integration.
Key `5` now loads the package on demand in a worker, reports verified loading,
backend, adapter, format version, and package hash, and advances the fixed-step
simulation only when the corresponding recurrent decision is ready. If WebGPU
or the package is unavailable, the game reports the error and returns to the
small GRU on key `4`.

## Commands

```bash
npm run evaluate:connectome-quantization
npm run export:connectome-quantized
npm run test:connectome-quantized-rust-parity
npm run benchmark:connectome-wasm
npm run test:connectome-webgpu
```

To repeat the standalone hardware measurement, run `npm run dev`, open
`http://localhost:5173/connectome-benchmark.html?ticks=300` in a normal browser,
and inspect the displayed adapter and timings. Run `npm run
prepare:connectome-web` first if the ignored model assets are absent.
