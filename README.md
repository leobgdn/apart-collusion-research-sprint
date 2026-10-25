# Apart collusion research sprint

Which task features make LLM agents collude, and at what cost to the task can
each be restricted? Two tracks, each self-contained in its own folder, with its
own environment, README, tests and reports.

| folder | what | status |
|---|---|---|
| [`pricing/`](pricing/) | **Track A**: a repeated logit-Bertrand pricing game with two Qwen3-8B sellers served locally (vLLM in WSL2, RTX 3090). Factorial sweep over channel x history window x horizon, plus a placebo. Action-level collusion index, LLM judge, activation probes. | main sweep complete (450 episodes); analysis in progress |
| [`colosseum/`](colosseum/) | **Track B**: feasibility check of the Colosseum multi-agent environment against the same local model server. | see `colosseum/reports/` |
| [`archive/`](archive/) | Superseded first version of the project (Plan B Stage 1 and the Fireworks-hosted Plan A adapter), kept for history. Not used by either track. | frozen |

Start with `pricing/README.md`. Key Track A documents:

- `pricing/prereg.md`: binding pre-registration, committed before main-sweep data
- `pricing/DECISIONS.md`: every design decision and why
- `pricing/reports/`: gate reports (gate1, gate2, gate3) and generated data

Shared between the tracks, outside the repo: the model server at
`http://localhost:8000/v1` (`~/collusion_shared/endpoint.txt`) and the
`~/collusion_shared/GPU_BUSY` lock file.
