# Packed Connectome Format v1

Status: active Phase 5 reference format
Last updated: 2026-09-20

This format contains everything needed for inference by the selected MaleCNS
controller and excludes gradients, optimizer state, raw synapse counts, and
unfused trainable gains. The reference export is intentionally float32. It is
the numerical baseline for later compression or quantization work.

## Package layout

A package is one directory containing `manifest.json` and ten raw typed-array
files. All multibyte values use little-endian byte order. Arrays are contiguous
in C order and have no per-file header.

| Array | Type | Shape | Meaning |
|---|---|---|---|
| `row_offsets` | `u32` | neurons + 1 | CSR edge range for every postsynaptic row |
| `column_indices` | `u32` | edges | Presynaptic neuron for each measured edge |
| `edge_weights` | `f32` | edges | Normalized count with learned edge gain fused |
| `leak` | `f32` | neurons | Bounded learned leak with its sigmoid fused |
| `sensory_indices` | `u32` | sensory neurons | Driven neuron indices |
| `sensory_feature_ids` | `u8` | sensory neurons | Feature assigned to each driven neuron |
| `sensory_signs` | `i8` | sensory neurons | Positive or negative feature sign |
| `motor_indices` | `u32` | motor neurons | Neurons read by the output head |
| `readout_weight` | `f32` | 3 x motor neurons | Row-major linear output weights |
| `readout_bias` | `f32` | 3 | Linear output bias |

Every array entry records its file name, dtype, shape, byte length, and SHA-256.
`package_sha256` is the SHA-256 of canonical JSON for the entire manifest after
removing that one field. The canonical JSON uses sorted keys, UTF-8, and compact
separators. Because the remaining manifest contains every array checksum, the
package digest commits to all model bytes as well as their interpretation.

The manifest also fixes the model dimensions, feature schema and order,
sensory assignment, output order and activations, strike threshold, source
checkpoint and graph hashes, dataset license and provenance, and the exact
inference fusions.

## Inference

CSR rows are postsynaptic and columns are presynaptic. One game tick performs
one recurrent update:

```text
signal[row] = sum(edge_weight[e] * state[column[e]]) + drive[row]
state' = (1 - leak) * state + leak * tanh(signal)
raw = readout_weight * state'[motor_indices] + readout_bias
target = sigmoid(raw[0:2])
strike_probability = sigmoid(raw[2])
```

The feature assigned to each sensory neuron is multiplied by its stored sign
and added to that neuron's signal. Neural state starts at zero at the beginning
of an episode and is carried between ticks. A strike is requested when its
probability reaches the manifest threshold and the simulation reports that a
synchronized strike is available.

Consumers must reject an unknown format version, a manifest checksum mismatch,
an array checksum or byte-length mismatch, an invalid shape, a non-monotonic
CSR layout, or an out-of-range index. Format semantics cannot change without a
new `format_version`.

## Reference commands

```bash
npm run export:connectome-packed
npm run test:connectome-packed-parity
```

Generated packages remain ignored under `artifacts/`. Browser delivery will
use immutable hashed assets derived from this reference package after CPU
latency and compression measurements are complete.
