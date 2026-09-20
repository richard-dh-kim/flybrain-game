# Phase 4 MaleCNS Controller v1

Date: 2026-09-20
Status: single-seed trained-versus-initialized gate passed

## Controller contract

The controller uses the hash-verified MaleCNS v1.0 graph with 165,122 traced
neurons and 25,563,197 measured directed edges. CSR rows are postsynaptic and
columns are presynaptic. The topology never changes during optimization.

The game supplies the same 33 normalized feature-schema-v1 values used by the
conventional baselines. A seeded, frozen, balanced mapping assigns those
features with random signs to 6,365 neurons annotated as `vnc_sensory`. The
controller carries all neuron states between 60 Hz game ticks and performs one
measured graph update per tick. A trainable linear head reads only the 708
neurons annotated as `vnc_motor` and emits two normalized target coordinates
plus a strike logit. There is no direct observation-to-action path.

The trainable parameters are 25,563,197 bounded edge gains, 165,122 neuron
leaks, and 2,127 motor-readout values, for 25,730,446 total. Edge transmission
is an unsigned numerical proxy based on normalized synapse counts; this model
does not claim transmitter, receptor, or functional sensory assignments.

The sparse core is adapted from MIT-licensed `flyhard` commit
`328906f4a0e62c8f9fc18805cf6edae6989b82a5`. MaleCNS v1.0 is separately CC BY
4.0. The selected checkpoint stores dataset, license, source page, selection
rule, and graph hash.

## Training

Training used seed 1701, four recurrent streams, four-tick truncated
backpropagation, and one graph update per game tick. The 600-step pilot used a
0.02 learning rate. A separately hashed refinement resumed its parameters,
reset the Adam optimizer, lowered the rate to 0.01, and ran 1,200 more steps.
Together the runs processed 28,480 rows, about 3.22 passes over the 8,847-row
training split, in 827.8 seconds on the RTX 4060 Ti.

The first backward pass produced finite nonzero gradients for the edge gains,
leaks, readout weights, and readout bias. The full topology digest was identical
before and after training. Peak batch-four allocation was 3.45 GB and peak
reserved memory was 3.75 GB.

On all 2,877 offline validation rows, carried-state target RMSE was 56.76
pixels. Resetting all 165,122 states every tick increased it to 175.13 pixels.
The carried strike head reached 0.409 precision and 0.214 recall at the
offline-selected 0.90 threshold. This remains substantially worse offline than
the 13.17-pixel MLP and 22.67-pixel GRU controls.

## Closed-loop calibration and result

The initial offline threshold was not valid under policy-induced observation
shift. At 0.90, both the trained controller and its calibrated untrained
initialization made zero accepted strikes across all 33 live validation
episodes. This negative result is preserved in the run artifacts.

A separate six-value closed-loop sweep on the 12-episode evaluation curriculum,
capped at 600 ticks per episode, selected 0.50 because it produced the most
hits: 10/12. The threshold was then frozen before the varied-start validation
suite was rerun.

| Condition | Threshold | Hits | Mean hit tick | Accepted strikes | Completed misses |
|---|---:|---:|---:|---:|---:|
| Untrained initialization | 0.05, offline calibrated | 0/33 | n/a | 0 | 0 |
| Trained MaleCNS | 0.50, sweep selected | 33/33 | 116.45 | 64 | 31 |
| Trained MaleCNS, state reset every tick | 0.50, frozen | 0/33 | n/a | 0 | 0 |

The trained controller therefore beats its untrained initialization on the
fixed held-out suite and passes the first Phase 4 behavior gate. Batch-one
inference measured 5.40 ms median and 6.06 ms p95 across 3,843 decisions on the
RTX 4060 Ti. The matched live reset-state control confirms that accumulated
neural state is behaviorally necessary for this checkpoint. These are
PyTorch/CUDA measurements, not browser timings.

The selected 99 MB parameter checkpoint is
`checkpoints/connectome-v1-selected.pt`, SHA-256
`f795652899f3df47f43eda72bf6de4bf3f43337a28865723762b450e0f91daa7`.
The graph remains a separate 296 MB compressed artifact with SHA-256
`eff4093bf53c4dd17d7ee4f2f838f6ae5ede70570f9a91317dd8824cca5c771d`.

## Reproduction

After placing the verified graph at `data/graph-traced-v1` or setting
`FLYBRAIN_GRAPH`:

```bash
npm run train:connectome-smoke
npm run train:connectome-pilot
npm run train:connectome-refine
npm run evaluate:connectome
npm run sweep:connectome-thresholds
npm run evaluate:connectome-selected
npm run evaluate:connectome-reset
npm run select:connectome
```

Generated datasets, graph files, checkpoints, and detailed run JSON remain
ignored. A compact result with all relevant hashes is versioned beside this
report as `phase-4-male-cns-v1.metrics.json`.

## Limits and next gate

This is one seed with an engineered random feature interface and trainable
motor readout. It does not show that the measured biological topology is better
than another topology. Shuffled-topology, matched random-sparse, additional
seed controls remain. Human paths and DAgger rounds also remain.

The seed-1701 topology controls were subsequently completed in
`phase-4-topology-controls.md`. The random-sparse condition slightly beat the
measured graph in that seed, so the original biological-topology claim limit
still applies and additional seeds remain necessary.

The controller is not in the browser. Phase 5 must define a packed model,
measure download and decompressed memory, implement Rust/WASM inference in a
worker, and establish numerical and gameplay agreement before this checkpoint
can replace or supplement browser mode `4`.
