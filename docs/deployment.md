# Static deployment

## Selected host

The first public build targets GitHub Pages at:

`https://richard-dh-kim.github.io/flybrain-game/`

The game is a static Vite application and does not require a server. GitHub
Pages is sufficient for the current small-audience release. The published site
is about 149 MiB because the full connectome package is downloaded in the
background while the compact GRU starts play immediately.

The model is deliberately not committed to Git. Its largest array is
102,252,788 bytes, above GitHub's normal 100 MiB Git-file limit. Instead, the
ignored packed model is stored in one versioned GitHub Release archive. The
manual Pages workflow downloads that archive, verifies its SHA-256 hash,
unpacks it, builds the project with the `/flybrain-game/` base path, and deploys
the resulting static files.

## Fixed model release

- Release tag: `connectome-u16-v2`
- Asset: `flybrain-connectome-u16-v2.tar.gz`
- Asset size: 88,315,352 bytes
- Asset SHA-256:
  `fbbd7193b4eaffb0cd72378bdfe6e037f4ac56f4e63a1d007b7999edaa8461ca`
- Packed-model package SHA-256:
  `0f5baf90bf5bed5802931b289d551474524547872ceb3eff657e25d9d8e54a37`

Regenerate the deterministic release archive with:

```bash
npm run package:connectome-release
```

The ignored output is
`artifacts/releases/flybrain-connectome-u16-v2.tar.gz`. Any change to that
archive requires updating the pinned hash in `.github/workflows/pages.yml`
before deployment.

The model is derived from the **MaleCNS v1.0** dataset, available from the
[Janelia MaleCNS download page](https://male-cns.janelia.org/download/), and
retains its source and graph hashes in `manifest.json`. The source dataset is
licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/).

## First publication

1. Create a GitHub release with tag `connectome-u16-v2` and upload
   `artifacts/releases/flybrain-connectome-u16-v2.tar.gz` as its asset.
2. In the repository's **Settings → Pages**, select **GitHub Actions** as the
   build and deployment source.
3. Open **Actions → Deploy GitHub Pages → Run workflow** and run it from
   `main`.
4. Open the deployment URL and confirm that the compact controller starts
   immediately, the full model reaches `FULL MALECNS · LIVE` on a WebGPU
   device, and an unsupported device visibly remains on CPU fallback.

The deployed hardware check is available at
`https://richard-dh-kim.github.io/flybrain-game/connectome-benchmark.html?ticks=300`.

The workflow is manual so pushing ordinary research commits cannot
accidentally publish or replace the public game.

## Other hosting options

Cloudflare Pages, Netlify, or another static CDN can serve the application, but
the large model still needs either a compatible object store or a release-style
download during the build. GitHub Pages keeps the first release inside the
existing public repository. If real traffic approaches GitHub Pages' soft
bandwidth limit, move the model arrays to object storage behind a CDN while
leaving the small game shell on Pages.
