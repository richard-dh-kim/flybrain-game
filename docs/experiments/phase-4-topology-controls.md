# Phase 4 MaleCNS topology controls

Date: 2026-09-21
Status: initial three-seed comparison complete

## Question

The existing seed-1701 MaleCNS controller can play the game, but that alone does
not show that the measured biological wiring helped. The first controlled
comparison trains two networks with the same neuron count, edge count, input
mapping, output population, data, optimizer schedule, and random seed.

## Conditions

| Condition | What changes | What remains matched |
|---|---|---|
| `measured` | Nothing | Original MaleCNS graph |
| `shuffled-presynaptic` | A seeded one-to-one permutation changes every sending neuron identity | Neurons, edges, every receiving-neuron degree, edge weights, and the sending-degree multiset |
| `random-sparse` | Each receiving row samples unique sending partners without replacement | Neurons, edges, every receiving-neuron degree, and every edge weight |

All generated topologies are deterministic. Their variant, seed, construction,
and SHA-256 topology digest are stored in both checkpoints and metrics. Resume
and evaluation reject a checkpoint if the reconstructed digest differs.

The random-sparse condition does not preserve each sending neuron's degree. It
is a broader sparsity control, while the shuffled-presynaptic condition is the
more tightly degree-matched control. Self-connections are allowed in all three
conditions. No control introduces duplicate sending partners within a receiving
row.

## Fixed first comparison

The first comparison uses interface seed 1701 and topology seed 1701. Each
control follows the measured model's exact schedule:

1. 600 pilot optimizer steps at learning rate 0.02.
2. 1,200 refinement steps at learning rate 0.01, resumed from the pilot.
3. Offline carried-state and reset-state validation.
4. The same six-threshold closed-loop sweep on the 12-episode evaluation suite.
5. Select the threshold by maximizing hits, then minimizing mean successful hit
   time, completed misses, accepted strikes, and finally the numeric threshold.
6. Freeze that value, then run the 33-episode validation suite once.

This seed-matched result can reveal a useful difference, but it still cannot
support a broad claim about biological topology. The full comparison needs the
same three conditions repeated across additional predeclared seeds.

## Seed 1701 results

All three conditions completed the same 1,800 optimizer steps and processed the
same 28,480 training rows. Lower pixel error, hit ticks, and misses are better.

| Condition | Offline target RMSE | Sweep hits | Validation hits | Mean hit tick | Slaps | Misses |
|---|---:|---:|---:|---:|---:|---:|
| Measured MaleCNS | 56.76 px | 10/12 at 0.50 | 33/33 | 116.45 | 64 | 31 |
| Shuffled presynaptic | 75.58 px | 9/12 at 0.20 | 33/33 | 248.21 | 133 | 100 |
| Random sparse | 51.95 px | 11/12 at 0.50 | 33/33 | 108.18 | 63 | 30 |

Measured wiring clearly outperformed the tightly degree-matched presynaptic
shuffle on offline target error and closed-loop efficiency. The broader
random-sparse control, however, slightly outperformed the measured graph on
offline target error, the 12-episode sweep, and all three validation-efficiency
measures. All three eventually hit every validation path.

The honest seed-1701 conclusion is mixed: graph structure matters, but this run
provides no evidence that the measured biological topology is better than
matched random sparsity. Differences between measured and random are small in
closed loop and could reverse with another initialization. At least several
predeclared seeds are required before estimating a topology effect.

The next replication seeds were fixed before those runs as 3407 and 99017.
Each replicate uses the same value for its interface and topology seed in all
three conditions. Together with 1701, this creates an initial three-seed
comparison; more seeds will be required if its variance remains large.

The recurrent state remained important offline. Resetting state every tick
raised target RMSE to 175.13 px for measured, 163.46 px for shuffled, and
174.72 px for random sparse.

## Seed 3407 results

All three conditions again completed the same 1,800 optimizer steps, this time
processing 28,483 rows per condition. The six threshold candidates and their
selection order were fixed before the runs. The selected threshold was then
used once on the separate 33-path validation suite.

| Condition | Offline target RMSE | Sweep hits | Validation hits | Mean hit tick | Slaps | Misses |
|---|---:|---:|---:|---:|---:|---:|
| Measured MaleCNS | 50.45 px | 12/12 at 0.50 | 33/33 | 110.06 | 59 | 26 |
| Shuffled presynaptic | 84.31 px | 8/12 at 0.20 | 29/33 | 131.79 | 107 | 74 |
| Random sparse | 44.54 px | 12/12 at 0.35 | 33/33 | 92.79 | 60 | 27 |

Measured wiring again beat the tightly degree-matched shuffle: it had 40%
lower offline target error, four more validation hits, faster successful hits,
and far fewer failed slaps. Random sparse again had the lowest offline error.
It tied measured at 33 validation hits and caught successful paths faster,
although it used one additional slap and recorded one additional miss.

Across the first two predeclared seeds, measured wiring consistently beat the
tightly matched presynaptic shuffle. Random sparse matched or beat measured,
however, so those two seeds did not show that the measured biological topology
was superior.

Resetting recurrent state at seed 3407 raised target RMSE to 168.18 px for
measured, 166.66 px for shuffled, and 183.69 px for random sparse.

## Seed 99017 results

All three conditions completed the same 1,800 optimizer steps and processed
the same 28,475 training rows. Threshold selection and validation followed the
same predeclared procedure.

| Condition | Offline target RMSE | Sweep hits | Validation hits | Mean hit tick | Slaps | Misses |
|---|---:|---:|---:|---:|---:|---:|
| Measured MaleCNS | 51.24 px | 11/12 at 0.35 | 33/33 | 100.58 | 62 | 29 |
| Shuffled presynaptic | 53.91 px | 12/12 at 0.50 | 33/33 | 88.24 | 53 | 20 |
| Random sparse | 44.14 px | 12/12 at 0.50 | 33/33 | 93.67 | 56 | 23 |

Measured again had lower offline target error than the tightly matched shuffle,
but this seed reversed their live-play ordering: shuffled caught successful
paths faster and used fewer slaps and misses. Random sparse again had the lowest
offline error and landed between shuffled and measured in live efficiency.

Resetting recurrent state raised target RMSE to 172.07 px for measured, 169.70
px for shuffled, and 180.73 px for random sparse.

## Three-seed result

The table below aggregates the initial predeclared comparison. Target RMSE and
mean hit tick are arithmetic means across seeds; hit, slap, and miss counts are
totals. Each topology received 5,400 optimizer steps across the three seeds.

| Condition | Mean offline RMSE | Sweep hits | Validation hits | Mean hit tick | Slaps | Misses |
|---|---:|---:|---:|---:|---:|---:|
| Measured MaleCNS | 52.82 px | 33/36 | 99/99 | 109.03 | 185 | 86 |
| Shuffled presynaptic | 71.27 px | 29/36 | 95/99 | 156.08 | 293 | 194 |
| Random sparse | 46.88 px | 35/36 | 99/99 | 98.21 | 179 | 80 |

Measured wiring beat the tightly degree-matched shuffle in the aggregate and
had lower offline error in all three paired seeds. Its closed-loop advantage
was large at seeds 1701 and 3407, while shuffled was more efficient at seed
99017. This supports the limited conclusion that the graph's neuron-to-neuron
organization affects learning and behavior in this setup.

The broader random-sparse control was best on every aggregate measure. The
experiment therefore does not show that the measured biological topology is
better for this game or training method. Three seeds are enough to finish the
predeclared initial comparison, but too few for a broad biological claim or a
precise effect estimate.

## Smoke evidence

Both 165,122-neuron, 25,563,197-edge controls completed two CUDA optimizer steps
with the real four-tick training window on the RTX 4060 Ti. Every parameter
group received finite, nonzero gradients and each topology digest was unchanged
after optimization.

- Shuffled-presynaptic digest:
  `b46e427da8b60154cd1619c5d0cb9e51b77a6c378d3b030f5a9d8f01653ccb3f`
- Random-sparse digest:
  `618603aa963d6f098a1d9f196381b44a2b3ac5e1228b4fa4c32800c25fa43b77`

The shuffled checkpoint also passed a short closed-loop reconstruction check:
evaluation regenerated the exact stored digest and produced valid decisions.
The two-step smoke outputs are temporary diagnostics, not gameplay results.

The compact, versioned result is
`phase-4-topology-controls.metrics.json`. Generated histories, full metrics,
and roughly 99 MB checkpoints remain ignored; their hashes are recorded in the
compact result.

## Commands

```bash
npm run train:connectome-shuffled-pilot
npm run train:connectome-shuffled-refine
npm run train:connectome-random-pilot
npm run train:connectome-random-refine

PYTHONPATH=python .venv/bin/python scripts/evaluate-connectome.py \
  --checkpoint checkpoints/connectome-control-shuffled-v1-refine.pt \
  --trained-only --suite evaluation --maximum-ticks 600 \
  --thresholds 0.05 0.2 0.35 0.5 0.65 0.8 \
  --out runs/connectome-control-shuffled-v1-refine/threshold-sweep.metrics.json
PYTHONPATH=python .venv/bin/python scripts/evaluate-connectome.py \
  --checkpoint checkpoints/connectome-control-shuffled-v1-refine.pt \
  --trained-only --suite validation --thresholds 0.2 \
  --out runs/connectome-control-shuffled-v1-refine/live-validation-selected.metrics.json

PYTHONPATH=python .venv/bin/python scripts/evaluate-connectome.py \
  --checkpoint checkpoints/connectome-control-random-v1-refine.pt \
  --trained-only --suite evaluation --maximum-ticks 600 \
  --thresholds 0.05 0.2 0.35 0.5 0.65 0.8 \
  --out runs/connectome-control-random-v1-refine/threshold-sweep.metrics.json
PYTHONPATH=python .venv/bin/python scripts/evaluate-connectome.py \
  --checkpoint checkpoints/connectome-control-random-v1-refine.pt \
  --trained-only --suite validation --thresholds 0.5 \
  --out runs/connectome-control-random-v1-refine/live-validation-selected.metrics.json
```
