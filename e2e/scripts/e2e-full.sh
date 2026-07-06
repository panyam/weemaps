#!/usr/bin/env bash
# e2e-full.sh: boot a lilbattle dev server, seed fixture worlds, run the
# replay harness against it, tear the server down cleanly on exit
# (including on failure or Ctrl-C).
#
# Prereqs:
#   - $LILBATTLE_REPO points at a lilbattle checkout (default: ~/projects/lilbattle)
#   - `ww` on PATH (`make cli` in the lilbattle repo)
#
# Config knobs (env overrides):
#   LILBATTLE_REPO                 (default ~/projects/lilbattle) — server source
#   LILBATTLE_E2E_HTTP_PORT        (default 8090)
#   LILBATTLE_E2E_GRPC_PORT        (default 9091)
#   LILBATTLE_E2E_STARTUP_TIMEOUT  (default 30)
#
# Anything after `--` is passed to `go test`, so you can target a single
# replay: `scripts/e2e-full.sh -- -run TestReplayScripts/29146`.
#
# CI: cheapest possible orchestration for the full suite. Local dev: use
# `go test -tags=e2e ./...` against a lilbattle server you already have
# running instead.
set -euo pipefail

LILBATTLE_REPO="${LILBATTLE_REPO:-$HOME/projects/lilbattle}"
HTTP_PORT="${LILBATTLE_E2E_HTTP_PORT:-8090}"
GRPC_PORT="${LILBATTLE_E2E_GRPC_PORT:-9091}"
TIMEOUT="${LILBATTLE_E2E_STARTUP_TIMEOUT:-30}"

E2E_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ ! -f "$LILBATTLE_REPO/main.go" ]]; then
    echo "[e2e-full] LILBATTLE_REPO=$LILBATTLE_REPO does not look like a lilbattle checkout (no main.go)" >&2
    exit 1
fi

echo "[e2e-full] booting server on :$HTTP_PORT (gRPC :$GRPC_PORT) from $LILBATTLE_REPO"
(cd "$LILBATTLE_REPO" && \
    LILBATTLE_WEB_PORT=":$HTTP_PORT" \
    LILBATTLE_GRPC_PORT=":$GRPC_PORT" \
    DISABLE_API_AUTH=true \
    go run main.go -games_service_be=local -worlds_service_be=local >/tmp/e2e-server.log 2>&1) &
SERVER_PID=$!

# Kill the server on any exit — success, failure, or interrupt.
cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[e2e-full] stopping server ($SERVER_PID)"
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# Wait for the server to answer HTTP. `curl -s` doesn't error on non-2xx;
# the game viewer route returns 200 or 404 depending on the game ID,
# either of which proves the listener is bound.
echo "[e2e-full] waiting for server to become ready..."
deadline=$(( SECONDS + TIMEOUT ))
until curl -s -o /dev/null "http://localhost:$HTTP_PORT/games/probe/view"; do
    if (( SECONDS >= deadline )); then
        echo "[e2e-full] server did not become ready within ${TIMEOUT}s" >&2
        echo "[e2e-full] last 20 lines of server log:" >&2
        tail -20 /tmp/e2e-server.log >&2 || true
        exit 1
    fi
    sleep 1
done
echo "[e2e-full] server ready"

# Seed fixture worlds via `ww worlds ensure`. Idempotent — noop on
# re-run. First invocation covers a fresh CI checkout; subsequent runs
# against the same server are cheap probes.
echo "[e2e-full] seeding fixture worlds"
LILBATTLE_SERVER="http://localhost:$HTTP_PORT/api" \
    bash "$E2E_ROOT/scripts/seed-worlds.sh"

# Run the e2e tests. Any extra args after `--` land here so a single
# replay can be targeted from the command line.
extra_args=()
if (( $# > 0 )); then
    while [[ "${1:-}" != "--" && $# -gt 0 ]]; do shift; done
    if [[ "${1:-}" == "--" ]]; then shift; fi
    extra_args=("$@")
fi

echo "[e2e-full] running tests (extra args: ${extra_args[*]:-<none>})"
# `go test -C <dir>` requires -C to be the FIRST flag. Cd into E2E_ROOT
# so tag / -run / -count parsing is unambiguous.
(cd "$E2E_ROOT" && \
    LILBATTLE_E2E_SERVER="http://localhost:$HTTP_PORT/api" \
    go test -tags=e2e -v -count=1 "${extra_args[@]}")
