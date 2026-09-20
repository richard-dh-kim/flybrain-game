# flyhard sparse connectome core

The file `python/flybrain_training/connectome.py` adapts
`src/flyhard/connectome.py` from
[flyhard](https://github.com/MarkUnthank/flyhard) commit
`328906f4a0e62c8f9fc18805cf6edae6989b82a5`.

The upstream code is Copyright (c) 2026 Mark Unthank and distributed under the
MIT License reproduced in this directory. The adaptation adds local naming,
typing, formatting, and documentation while preserving the sparse-gradient
algorithm and measured-edge semantics.

MaleCNS v1.0 graph data is a separate CC BY 4.0 dataset. It is not vendored in
Git; generated manifests and checkpoints must preserve its source, selection
rule, license, and graph hash.
