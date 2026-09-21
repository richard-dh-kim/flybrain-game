# Can't Swat This

[**Play Can't Swat This**](https://richard-dh-kim.github.io/flybrain-game/)

**The tables have turned. You dodge. The fly swats.**

Can't Swat This is a cute browser dodge game. You control a tiny winged
dodger while a giant fruit fly tries to catch you with two coordinated hands.
The fly uses a recurrent neural network constrained by the measured wiring of
an adult fruit-fly nervous system.

Move with a mouse or touch and survive for 30 seconds. Press `R` to restart and
`H` to show technical details.

## The fly brain

The game starts immediately with a small CPU controller while it loads the
full model. On a browser with working WebGPU, it then switches to the full
MaleCNS controller:

- **Full MaleCNS:** 165,122 neuron states and 25,563,197 measured connections,
  evaluated locally with WebGPU.
- **CPU fallback:** a compact 19,203-parameter GRU used during loading and on
  devices without compatible WebGPU.

An RTX card is not required. A compatible integrated GPU can use WebGPU, and
the CPU fallback keeps the game playable on unsupported devices. The full
model is a roughly 148 MiB optional download.

The brain graphic shows 64 live samples from the active controller. Its shape
and faint links are an illustration; it does not display every neuron or claim
to show biological thoughts.

## What the experiment found

The connectome controller learned to play, but that alone does not prove that
biological wiring is best for this task. In a three-seed comparison:

- measured MaleCNS wiring consistently beat a tightly degree-matched shuffle;
- a broader random-sparse network performed best overall.

The result shows that wiring affects this controller. It does **not** establish
that the measured biological topology is superior. The full measurements and
selection rules are in the
[topology-control report](docs/experiments/phase-4-topology-controls.md). The
[browser inference report](docs/experiments/phase-5-u16-candidate.md) records
the quantization, parity, and hardware timing results.

## Run locally

The project uses Node.js 22 and the Rust toolchain pinned in
`rust-toolchain.toml`.

```bash
npm ci
npm run dev
```

A fresh checkout runs with the compact CPU controller. To include the full
model, download the `flybrain-connectome-u16-v2.tar.gz` asset from the
[`connectome-u16-v2` release](https://github.com/richard-dh-kim/flybrain-game/releases/tag/connectome-u16-v2),
extract it into `artifacts/connectome-packed-u16-v2`, and run:

```bash
npm run prepare:connectome-web
npm run dev
```

Build the static site with:

```bash
npm run build
```

## Verify the project

The main development checks are:

```bash
npm run typecheck
npm run test:browser
npm run test:wasm-parity
npm run test:gru-parity
npm run test:python
```

WebGPU hardware validation requires the full model and a compatible browser:

```bash
npm run test:connectome-webgpu
```

Training, evaluation, export, and parity commands remain in `package.json`.
Versioned experiment reports and compact metrics are under
[`docs/experiments`](docs/experiments), while binary interface definitions are
under [`docs/schemas`](docs/schemas).

## How it is built

- **TypeScript, Phaser, and Vite** render the browser game and interface.
- **Rust and WebAssembly** provide the deterministic 60 Hz simulation,
  collision rules, replays, and portable inference support.
- **Python, PyTorch, and CUDA** handle training and scientific evaluation.
- **WebGPU in a worker** runs the full sparse recurrent controller without
  blocking rendering.

The source graph comes from the
[MaleCNS v1.0 dataset](https://male-cns.janelia.org/download/) under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Derived model
packages retain source attribution, provenance, and hashes in their manifests.
Adapted upstream runtime notices are preserved under [`third_party`](third_party).

Deployment details are documented in [docs/deployment.md](docs/deployment.md).

## License

The project's original code is available under the [MIT License](LICENSE).
MaleCNS data and derived model assets retain their CC BY 4.0 terms and
attribution. Files under `third_party` remain subject to their included
licenses and notices.
