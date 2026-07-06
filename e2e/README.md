# weemaps/e2e — lilbattle replay harness

Recorded-game integration tests for [lilbattle](https://github.com/turnforge/lilbattle).
Each replay under `replays/*.sh` drives real `ww` through a recorded
game's moves with `ww assert ...` checks after each step.

## Prereqs

- `ww` on `PATH` — build it in the lilbattle repo via `make cli`.
- A running lilbattle server. `scripts/e2e-full.sh` boots one for you,
  or point at one you already have via `LILBATTLE_E2E_SERVER`.

## Run

```bash
# All replays, boot + teardown handled automatically.
./scripts/e2e-full.sh

# One replay by name.
./scripts/e2e-full.sh -- -run TestReplayScripts/29146

# Against a server you already have running.
export LILBATTLE_E2E_SERVER=http://localhost:8090/api
bash scripts/seed-worlds.sh          # once per fresh server
go test -tags=e2e -v -count=1
```

Watch mode auto-opens the game URL in a browser tab per replay:

```bash
LILBATTLE_E2E_WATCH=true go test -tags=e2e -v -count=1
```

Overrides:

- `LILBATTLE_REPO` (for `e2e-full.sh`) — path to a lilbattle checkout.
  Default `~/projects/lilbattle`.
- `LILBATTLE_WW_BIN` — explicit `ww` binary path. Default resolves via `PATH`.
- `LILBATTLE_E2E_{HTTP,GRPC}_PORT` — override server ports if `:8090` /
  `:9091` conflict.

## Layout

```
e2e/
├── go.mod                         # module: github.com/panyam/weemaps/e2e
├── harness.go                     # test helpers (server URL, ww lookup)
├── replay_test.go                 # TestReplayScripts — enumerates replays/
├── doc.go                         # package stub for default-tag builds
├── replays/                       # recorded ww command streams
│   ├── 24280.sh
│   ├── 29146.sh
│   └── 29190.sh
├── fixtures/worlds/               # world data seeded onto the target server
│   ├── 32112070/{metadata,data}.json
│   └── 7e5016a4/{metadata,data}.json
└── scripts/
    ├── e2e-full.sh                # boot server + seed + run + teardown
    └── seed-worlds.sh             # provision fixture worlds via ww worlds ensure
```

## Build tag

The Go tests are gated behind `-tags=e2e` while the recorded scripts
drift from current lilbattle rules (tracked in lilbattle issue 183).
Default `go test ./...` compiles `doc.go` and reports "no tests to run".

## Adding a new replay

1. Generate the `.sh` from a `public.json` game dump via
   `scripts/history.py` in the weemaps root.
2. Drop it into `e2e/replays/`.
3. If the world it references isn't already under `e2e/fixtures/worlds/`,
   add a `<worldID>/metadata.json` + `<worldID>/data.json` pair.

## Seeding a target server manually

For staging / prod / any server you don't want `e2e-full.sh` to touch:

```bash
export LILBATTLE_SERVER=https://staging.example.com/api
ww worlds ensure 7e5016a4 --data-dir fixtures/worlds/7e5016a4/
```

`ww worlds ensure` never overwrites — content mismatch is a hard error
so it's safe to point at any server. See lilbattle's `docs/DEVELOPER_GUIDE.md`
for the full semantics.
