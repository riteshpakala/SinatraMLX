#!/bin/bash
# WHAT: Build and run every test, MLX-gated suites included.
# PIN:  The test bundle is code-signed at build time and a metallib copied into it breaks
#       the seal, so it is removed before the build and installed after, then the tests
#       run from the already-built bundle (FRIGATE_MLX_TESTS=1 enables the MLX suites).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
for bundle in .build/out/Products/Debug/*.xctest; do
  [ -d "$bundle" ] || continue
  rm -f "$bundle/Contents/MacOS/mlx.metallib"
  rm -rf "$bundle/Contents/MacOS/Resources"
done
swift build --build-tests
"$ROOT/scripts/metallib.sh" debug > /dev/null
FRIGATE_MLX_TESTS=1 swift test --skip-build "$@"
