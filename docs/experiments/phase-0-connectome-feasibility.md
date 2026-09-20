# Phase 0 MaleCNS full-graph feasibility benchmark

Date: 2026-09-18

Status: Passed.

## Question

Can the complete retained MaleCNS graph run a bounded forward/backward training
test on the local RTX 4060 Ti 16 GB without constructing a dense neuron-by-
neuron gradient intermediate?

This is a numerical feasibility test. It is not game training, a biological
claim, or evidence that the measured topology outperforms another model.

## Pinned inputs

- `flyhard`: commit `328906f4a0e62c8f9fc18805cf6edae6989b82a5`
- Baseline implementation: unmodified `flyhard` sparse connectome core
- MaleCNS dataset: v1.0, acquired through the upstream hash-verified script
- Data license: CC BY 4.0
- Python: CPython 3.12
- PyTorch: 2.8.0+cu128
- GPU: NVIDIA GeForce RTX 4060 Ti, 16,380 MiB, driver 591.86

The optional faster CUDA backward from `fly-self-driving` commit
`3516a094a6462e2a4ab7cc517286d5c3bc3ff63e` is reserved for a separately
labeled follow-up. It is not part of the baseline run.

## Bounded run

Use the upstream `scripts/benchmark_core.py` with the complete graph, batch 1,
at most 10 optimizer steps, and at most 180 seconds of training time:

```bash
python scripts/acquire_connectome.py
.venv/bin/python scripts/prepare_graph.py
.venv/bin/python scripts/benchmark_core.py \
  --out runs/flybrain-4060ti-baseline \
  --batch 1 \
  --steps 10 \
  --seconds 180
```

The smaller batch and strict wall-time cap answer the Phase 0 fit and latency
question without starting a long training run.

## Required evidence

- Exact upstream and graph revisions/hashes
- Retained neuron and edge counts
- Setup/compilation time
- Per-step forward and backward time
- Peak allocated and reserved GPU memory
- Peak host memory observed externally
- Finite, nonzero gradients for trainable parameters
- Unchanged sparse topology after optimization
- Initial and final held-out loss
- Numerical status and any error output

## Result

The unmodified baseline completed successfully:

| Measurement | Result |
| --- | ---: |
| Retained neurons | 165,122 |
| Retained edges | 25,563,197 |
| Trainable parameters | 25,728,319 |
| Optimizer steps | 10 |
| Total command wall time | 8.12 s |
| Model setup time | 1.657 s |
| Training-loop wall time | 4.035 s |
| Median steady forward time | 9.18 ms |
| Median steady backward time | 233.19 ms |
| Peak allocated GPU memory | 2.996 GB |
| Peak reserved GPU memory | 3.137 GB |
| Peak host resident memory | 1,657,284 KiB |
| Initial held-out loss | 2.10659 |
| Final held-out loss | 0.17305 |
| Held-out loss reduction | 91.79% |

Every edge-gain gradient and every leak gradient was finite and nonzero. The
post-run topology check passed, and the graph hash remained
`eff4093bf53c4dd17d7ee4f2f838f6ae5ede70570f9a91317dd8824cca5c771d`.
The upstream benchmark reported `status: passed`.

Graph preparation took 27.27 seconds and reached 6,262,548 KiB peak host
resident memory. It retained the declared `status == "Traced"` population and
did not apply an edge threshold.

The upstream-generated `metrics.json` SHA-256 was
`a45098865bae5e9d459858d90f09523bfc03f48608f29e4adc40403cb55d916f`.
The committed, newline-normalized copy is
`phase-0-connectome-feasibility.metrics.json`, with SHA-256
`53c1f87cd543fe1a0962cbdede4e15b1765c6e7f7d6c60ea87703c7fef666876`.
The large checkpoint, neural trace, raw tables, and derived graph remain
untracked generated artifacts.

## Interpretation

The local 16 GB GPU clears the Phase 0 full-graph feasibility requirement with
substantial memory headroom at batch 1. This result supports beginning game and
binding work; it does not predict end-to-end training time, demonstrate a game
controller, or compare the measured topology with a baseline.
