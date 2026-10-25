# Gate 3 report: main sweep `main_v1` complete

Every number here is computed from `data/episodes/main_v1/` and
`data/logs/main_v1.jsonl`. The pre-registration (commit `48fb015`) was
committed before any of this data existed. **No confirmatory analysis has been
run yet**; that is Stage 4.

## Completion

- **450/450 episodes completed** (9 cells x 50 seeds, 80 rounds each). 0
  failed, 0 discarded at the deadline, 0 not launched. **All 25 balanced batches
  are complete**, and every cell has exactly 50 episodes.
- Wall time: 159.0 min (launched about 21:00, finished 23:39 on 2026-10-24),
  roughly 170 episodes/h at concurrency 32, sharing the server with the
  Colosseum session. Median episode time 654 s (range 394-1695 s).
- Turn log: 72,000 rows (525 MB). GPU during the run: 100% utilisation, about
  346 W of the 350 W cap, KV cache 75-87% full, 0 requests waiting.

## Data quality and stop-the-line checks

| check | result | limit |
|---|---|---|
| parse failures after retries | 657 / 72,000 = **0.91%** | stop at 10% |
| turns that needed a retry | 726 | - |
| reasoning blocks in output | **0** | stop at any |
| logged-prompt vs server token mismatches | **0** / 72,000 | stop at any |
| message leaks in no-channel cells | 0 | - |
| truncated prose | 0 | - |

All 657 final failures are `price_off_grid`: a price not on the
1.00-2.50 menu, after which the previous price is carried forward as
pre-registered. They are concentrated in no-channel cells: 264 in
none/short/long-run (3.3% of that cell's turns), 149 in the placebo cell, 126,
68 and 38 in other no-channel cells, and 12 in one channel-on cell. This is
reported per cell, as the pre-registration requires.

## Categories (not yet a result; the pre-registered exclusions apply)

sub_nash 233, partial 190, near_nash 16, near_monopoly 7, **over_priced 4**,
below_cost 0. The 4 over-priced episodes are excluded from the Delta analyses
as pre-registered: 3 in none/short/long-run (seeds 9, 12 and 48), 1 in the
placebo cell (seed 28). They coincide with the cells that had the most
off-menu replies.

## Next (Stage 4), in order

1. LLM judge on the GPU (about 15-20 min).
2. Prepare the human labelling set (`reports/judge_validation_digest.md`,
   `data/judge_validation.csv`).
3. Stop vLLM; activation extraction (about 30-60 min).
4. Confirmatory effects (H1-H3, placebo), the text-vs-action quadrants, the
   cost map, predict-then-verify and figures. Then Gate 4.
