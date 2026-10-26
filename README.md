# Apart collusion research sprint

Which task features make LLM agents collude, what does it cost consumers, and
which monitoring signal catches it? Everything lives in
[`pricing/`](pricing/): a repeated logit-Bertrand pricing game with two Qwen3-8B
sellers served locally (vLLM in WSL2, RTX 3090). Factorial sweep over channel x
history window x horizon, plus a placebo. Action-level collusion index, LLM
judge, activation probes.

**Status:** complete: 450 episodes, effects, judge, probes. Results for the
paper are in `pricing/reports/results_dossier.md`.

Start with `pricing/README.md`. Key documents:

- `pricing/prereg.md`: binding pre-registration, committed before main-sweep data
- `pricing/DECISIONS.md`: every design decision and why
- `pricing/reports/`: gate reports and the results dossier

**History (in git only):** Colosseum was evaluated first ("Plan A") and not
adopted: its collusion tasks are one-shot, and a local check found no
action-level collusion lever. Its final check report is at commit `78790d2`
(`colosseum/reports/colosseum_check.md`). The superseded first version of the
project, including the original Plan A decision record, is at commit `f09b4dc`
(`archive/`). Both folders were removed from the working tree on 2026-10-25.
