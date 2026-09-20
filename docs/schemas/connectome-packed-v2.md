# Packed Connectome Format v2

Status: validated Phase 5 u16 candidate
Last updated: 2026-09-20

Version 2 retains the v1 manifest, CSR topology, interface, provenance, and
float32 leak/readout arrays. It replaces the per-edge float32 weight array with
row-wise unsigned 16-bit weights and one float32 scale per postsynaptic row.

```text
edge_weight[e] = edge_scales[row] * edge_weights_u16[e]
```

Each row scale is its maximum v1 weight divided by 65,535. Values are rounded
to the nearest integer. Empty rows have a zero scale. The selected model has no
zero-valued quantized measured edges, so this representation preserves every
edge in the declared topology.

The `quantization` manifest object identifies the scheme, bit width, equation,
float32 reference package, validation artifact hash, zero-edge count, and
validation gate. `edge_weights` has dtype `u16`, and `edge_scales` has dtype
`f32` and shape `[neurons]`. All other validation, checksum, byte-order, and
version rejection requirements from v1 still apply.

The export command refuses to create v2 unless the supplied validation result
uses the named v1 package, retains 33/33 fixed-suite hits, removes no measured
edges, and changes no held-out strike decisions.

```bash
npm run evaluate:connectome-quantization
npm run export:connectome-quantized
```

Version 2 does not imply that all u16 implementations are equivalent. Runtime
parity and latency are separate gates because accumulating integers and applying
one row scale can round differently from multiplying every edge first.
