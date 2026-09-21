# FlyBrain connectome browser model v2

This release asset contains the quantized MaleCNS controller used by the
FlyBrain Swatter static website. The archive is reproducible from the local
packed model with `npm run package:connectome-release`.

- Asset: `flybrain-connectome-u16-v2.tar.gz`
- Size: 88,315,352 bytes
- Asset SHA-256: `fbbd7193b4eaffb0cd72378bdfe6e037f4ac56f4e63a1d007b7999edaa8461ca`
- Packed-model SHA-256: `0f5baf90bf5bed5802931b289d551474524547872ceb3eff657e25d9d8e54a37`

The model is derived from the **MaleCNS v1.0** dataset from the
[Janelia MaleCNS download page](https://male-cns.janelia.org/download/).
The source dataset is licensed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).
The packaged `manifest.json` records the source graph hash, training checkpoint
hash, quantization data, array sizes, and array hashes.
