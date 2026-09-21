#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${1:-$project_root/artifacts/connectome-packed-u16-v2}"
output_file="${2:-$project_root/artifacts/releases/flybrain-connectome-u16-v2.tar.gz}"
expected_package_sha256="0f5baf90bf5bed5802931b289d551474524547872ceb3eff657e25d9d8e54a37"

node "$project_root/scripts/verify-connectome-package.mjs" \
  "$source_dir" \
  "$expected_package_sha256"

mkdir -p "$(dirname "$output_file")"
temporary_file="$(mktemp "$(dirname "$output_file")/.connectome-release.XXXXXX.tar.gz")"
trap 'rm -f "$temporary_file"' EXIT

# Stable ordering, ownership, and timestamps make this archive reproducible.
tar \
  --sort=name \
  --mtime='UTC 1970-01-01' \
  --owner=0 \
  --group=0 \
  --numeric-owner \
  -C "$source_dir" \
  -cf - . \
  | gzip -n -9 > "$temporary_file"

mv "$temporary_file" "$output_file"
trap - EXIT

sha256sum "$output_file"
