#!/usr/bin/env bash
# Copies the parts of three.js that the web demo uses into demo/web/vendor/,
# so that the page works offline. Without them, the server redirects the
# page's requests for vendor/three/ to jsDelivr.
#
#   demo/vendor_three.sh

set -euo pipefail

VERSION="0.170.0"
# The npm registry's checksum of three-$VERSION.tgz (dist.shasum).
SHA1="6087f97aab79e9e9312f9c89fcef6808642dfbb7"
FILES=(
  LICENSE
  build/three.module.js
  examples/jsm/controls/OrbitControls.js
  examples/jsm/controls/TransformControls.js
  examples/jsm/loaders/GLTFLoader.js
  examples/jsm/utils/BufferGeometryUtils.js
)

OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/web/vendor/three"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

curl -fsSL "https://registry.npmjs.org/three/-/three-$VERSION.tgz" -o "$WORK/three.tgz"
echo "$SHA1  $WORK/three.tgz" | sha1sum --check --quiet -
tar -xzf "$WORK/three.tgz" -C "$WORK" "${FILES[@]/#/package/}"
for f in "${FILES[@]}"; do
  mkdir -p "$OUT/$(dirname "$f")"
  cp "$WORK/package/$f" "$OUT/$f"
done
echo "three.js $VERSION copied to $OUT"
