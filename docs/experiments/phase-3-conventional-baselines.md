# Phase 3 Conventional Baselines v1

Date: 2026-09-20
Status: initial browser baseline and varied-start follow-up passed

## Data and task

The v1 expert produced 9,145 training transitions across 78 episodes and 2,949
held-out transitions across 33 episodes. Train and validation trajectories use
disjoint turn periods, offsets, and random seeds. The training set contains 131
positive strike labels and 53 completed misses; validation contains 43 strike
labels and 10 completed misses.

Both models receive the same 33 normalized features from observation v1 and
predict a two-dimensional slap target plus a strike logit. The runtime masks
strike commands while either hand is unavailable, preserving the simulation's
hard action constraint. Training used seed 1701 and PyTorch 2.8.0 on CPU; these
small diagnostics do not need the GPU.

## Results

| Model | Parameters | Epochs | Target RMSE | Strike precision / recall | Live held-out hits | Mean hit tick |
|---|---:|---:|---:|---:|---:|---:|
| MLP | 6,531 | 150 | 12.09 px | 0.315 / 0.651 | 12/12 | 89.7 |
| GRU, carried state | 19,203 | 120 | 14.19 px | 1.000 / 1.000 | 12/12 | 79.1 |
| GRU, reset every tick | 19,203 | 120 | 14.19 px | 1.000 / 1.000 | 0/12 | n/a |

The reset-state control uses the same trained GRU weights and observations but
clears hidden state before every decision. Its 0/12 result versus 12/12 with
carried state confirms that the exported recurrent state is behaviorally
necessary for this checkpoint. It does not establish an advantage over all
feed-forward controllers: the MLP also reaches 12/12 on this small suite.

The GRU checkpoint is exported to a development JSON format and runs locally in
the browser on key `4`. Browser automation confirms that the frozen controller
loads, carries state, issues a synchronized slap, and ends a round. Pressing `H`
in learned mode shows the 64 actual GRU hidden values as a labeled **computed
model activity** panel.

The TypeScript recurrent runtime also matches PyTorch across a committed
140-tick held-out feature sequence. The maximum absolute difference across all
three outputs and all 64 hidden values is `3.708e-6`, below the `2e-5` gate.

Exact dataset and checkpoint SHA-256 values, platform, training time, offline
confusion counts, and per-episode live outcomes are in
`phase-3-mlp-v1.metrics.json` and `phase-3-gru-v1.metrics.json`.

## Varied-start follow-up

The follow-up regenerated 8,847 training rows across 78 episodes and 2,877
validation rows across 33 episodes. The splits rotate through disjoint grids of
starting positions as well as disjoint turn periods, offsets, and random seeds.
A robustness suite crosses the 12 evaluation trajectories with all 17 train
and validation starts, producing 204 closed-loop episodes.

| Model | Target RMSE | Strike precision / recall | Robustness hits | Reset-state hits | Mean hit tick |
|---|---:|---:|---:|---:|---:|
| MLP | 13.17 px | 0.457 / 0.500 | 200/204 | 200/204 | 82.3 |
| GRU | 22.67 px | 0.837 / 0.976 | 188/204 | 0/204 | 79.4 |

These results preserve the MLP as the stronger general baseline for this seed.
They also strengthen the narrower GRU state-dependence result: clearing state
eliminates every accepted strike and hit across the 204-episode suite. The
browser still packages the earlier fixed-start GRU because replacing a frozen
browser baseline requires a new parity fixture and explicit checkpoint choice.
Compact hashes and results are in
`phase-3-varied-start-v1.metrics.json`.

## Reproduction

```bash
npm run data:baseline
npm run train:mlp
npm run train:gru
npm run export:gru
npm run test:gru-parity
npm run build
npm run test:browser
```

Generated Parquet data, PyTorch checkpoints, and working metrics live in
ignored `data/`, `checkpoints/`, and `runs/` directories. The compact exported
development model and result metrics are versioned with the browser source.

## Limits

The evaluation is deterministic and synthetic. The varied-start suite is much
broader than the original 12 scenarios but still uses one training seed and no
human pointer paths. The MLP's strike classification is visibly weaker offline
even though hard action masking and repeated valid opportunities let it finish
nearly every live episode. Multiple seeds and human paths remain before
treating either model as a finished baseline. Neither model has MaleCNS
connectivity.
