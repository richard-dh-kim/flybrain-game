# Brain Activity Visualization Contract (Draft)

Status: design placeholder; no trained controller emits this yet
Last updated: 2026-09-19

The future side panel will describe **computed model activity**. It will not be
labeled as thoughts, biological consciousness, or a direct recording from a
living fly.

The renderer should receive activity at 5–10 Hz from the inference worker,
independent of the 60 Hz game loop. A frame will identify the model version,
simulation tick, layer or recurrence step, normalization range, and selected
neuron or region activations. Static neuron positions and region metadata load
once with the model.

The default view should draw all neurons as low-cost points, brighten active
ones, color stable biological or model categories, and label sensory and motor
groups. It should omit the full 25.5-million-edge graph during live play.
Optional connections may be limited to a small top-active set. The worker
should send compact typed arrays to a separate WebGL2 canvas or
`OffscreenCanvas`; WebGPU can remain an optional later backend.

The binary layout and normalization are intentionally unversioned until the
first conventional recurrent model exposes real hidden state. Phase 2 datasets
already retain the simulation tick needed to synchronize a future activity
frame with gameplay and replay.
