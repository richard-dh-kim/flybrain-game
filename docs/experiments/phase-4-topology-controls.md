# Phase 4 MaleCNS topology controls

Date: 2026-09-20
Status: full-size smoke checks passed; matched training runs pending

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
5. Freeze the best evaluation threshold, then run the 33-episode validation
   suite once.

This seed-matched result can reveal a useful difference, but it still cannot
support a broad claim about biological topology. The full comparison needs the
same three conditions repeated across additional predeclared seeds.

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

## Commands

```bash
npm run train:connectome-shuffled-pilot
npm run train:connectome-shuffled-refine
npm run train:connectome-random-pilot
npm run train:connectome-random-refine
```
