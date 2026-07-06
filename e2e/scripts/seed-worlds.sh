#!/usr/bin/env bash
# seed-worlds.sh: idempotently seed every fixture world under
# e2e/fixtures/worlds/ onto the target lilbattle server via
# `ww worlds ensure`.
#
# Requires:
#   LILBATTLE_SERVER    — target API endpoint (e.g. http://localhost:8090/api)
#   ww                  — on PATH (`make cli` in the lilbattle repo installs to $GOBIN)
#
# `ww worlds ensure` treats existing-and-matching as success and content
# mismatch as a hard error — so this script never overwrites production
# worlds. If a mismatch surfaces, the operator picks the fix path.
set -euo pipefail

if [[ -z "${LILBATTLE_SERVER:-}" ]]; then
    echo "seed-worlds: LILBATTLE_SERVER not set" >&2
    exit 1
fi

E2E_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURE_ROOT="$E2E_ROOT/fixtures/worlds"

if [[ ! -d "$FIXTURE_ROOT" ]]; then
    echo "seed-worlds: fixture root $FIXTURE_ROOT missing" >&2
    exit 1
fi

for fixture_dir in "$FIXTURE_ROOT"/*/; do
    [[ -d "$fixture_dir" ]] || continue
    world_id="$(basename "$fixture_dir")"
    echo "[seed-worlds] ensuring $world_id from $fixture_dir"
    ww worlds ensure "$world_id" --data-dir "$fixture_dir"
done

echo "[seed-worlds] done"
