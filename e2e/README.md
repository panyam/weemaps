# weemaps/e2e — replay test data

Recorded game replays + world fixtures consumed by the lilbattle e2e
harness. This directory holds only DATA; the harness itself (Go +
shell orchestration) lives in the lilbattle repo under `tests/e2e/`.

## Layout

```
e2e/
├── replays/*.sh           # recorded ww command streams
└── fixtures/worlds/       # world data seeded onto the target server
    ├── 32112070/{metadata,data}.json
    └── 7e5016a4/{metadata,data}.json
```

## Running

From a lilbattle checkout, point the harness here:

```bash
cd ~/projects/lilbattle
make cli                                                # build ww once
make e2e-full DATA_DIR=~/projects/weemaps/e2e           # all replays
make e2e-full DATA_DIR=~/projects/weemaps/e2e ARGS='-run TestReplayScripts/29146'
```

Or against a server you already have running:

```bash
export LILBATTLE_E2E_SERVER=http://localhost:8090/api
export LILBATTLE_E2E_DATA_DIR=~/projects/weemaps/e2e
cd ~/projects/lilbattle
make e2e
make e2e-watch          # auto-opens the game URL in a browser
```

See `~/projects/lilbattle/docs/DEVELOPER_GUIDE.md` → "Recorded replay
harness" for the full option list, seeding semantics, and drift
diagnosis recipe.

## Adding a new replay

1. Generate the `.sh` from a `public.json` game dump via
   `../scripts/history.py` (weemaps root).
2. Drop it into `replays/`.
3. If the world it references isn't already under `fixtures/worlds/`,
   add a `<worldID>/metadata.json` + `<worldID>/data.json` pair.

## Note on drift

Recorded scripts may fail against the current lilbattle rules engine —
tracked in lilbattle issue 183. That's expected until either the rules
regressions are fixed or the scripts are regenerated from source dumps.
