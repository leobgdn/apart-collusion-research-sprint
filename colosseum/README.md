# colosseum/ — Colosseum feasibility check (Track B)

Lives inside `apart-collusion-research-sprint/` but is self-contained: own Colosseum clone
(`repo/`, pinned `0048f1b`, own `.venv`), own outputs, own `.gitignore` (ignores `repo/` and `outputs/`).
Writes nothing outside this folder. Does not touch the sprint's `collusion/` package, the sibling
`GitHub/colosseum` checkout used by `collusion/envs/colosseum_adapter/`, or `~/collusion_shared/`
(read-only: `endpoint.txt`, `GPU_BUSY`).

- `reports/colosseum_check.md` — the go/no-go report (partial until the real pilot runs)
- `scripts/proxy.py` — mock server / logging proxy (thinking-off, ≤4 concurrent, honours GPU_BUSY)
- `scripts/run_real.sh one|pilot` — real runs via the proxy against the shared endpoint
- `scripts/summarize.py` — per-run metrics + per-condition collusion counts
- `configs/` — mock.yaml, real_one.yaml, pilot.yaml (6 episodes)
- `outputs/` — run artefacts and proxy JSONL logs
