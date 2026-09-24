#!/bin/bash
# WHAT: Build Frigate's mlx.metallib next to this package's debug/release
#       binaries and .xctest bundles. `swift build` never compiles the Metal
#       shaders, so without this the first GPU op fails at runtime.
# IN:   [debug|release] (default debug); FRIGATE_DIR to point at a checkout.
# OUT:  .build/<config>/mlx.metallib and copies beside the test bundles.
# PIN:  Delegates to Frigate/scripts/build-metallib.sh --package <this repo>.
set -euo pipefail
CONFIG="${1:-debug}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRIGATE="${FRIGATE_DIR:-}"
if [ -z "$FRIGATE" ]; then
  for candidate in "$ROOT/../../rao/repositories/Frigate" "$ROOT/.build/checkouts/Frigate"; do
    if [ -f "$candidate/scripts/build-metallib.sh" ]; then FRIGATE="$candidate"; break; fi
  done
fi
if [ -z "$FRIGATE" ]; then
  echo "metallib.sh: Frigate checkout not found; set FRIGATE_DIR" >&2; exit 1
fi
exec "$FRIGATE/scripts/build-metallib.sh" "$CONFIG" --package "$ROOT"
