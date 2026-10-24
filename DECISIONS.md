# Decisions log

Every design decision not fully specified by the build prompt, recorded with rationale, per Section 3 rule 10.

## 2026-10-24: Sprint deadline revised to ~36 hours (was ~48)

**Decision:** Compress the schedule as follows, rather than cutting scope uniformly:

1. **Skip Plan A (Colosseum) exploration entirely.** Go straight to Plan B (the
   repeated pricing environment, Section 7). Plan B is fully specified, needs
   no external repo, and runs entirely local on the available GPU with $0 API
   spend — already the direction implied by the local-model / $0-budget
   answers in Section 0. The Section 6 decision rule capped Colosseum
   exploration at ~90 minutes anyway; at 36 hours that budget is not worth
   spending even to confirm the rule triggers.
2. **Narrow the factorial for the confirmatory sweep:**
   - F1 (private channel, H1) and F2 (history window, H2) are the two
     confirmatory factors in the main sweep — cheap to implement, directly
     testable, and the two hypotheses with the clearest predicted direction.
   - F3 (horizon framing, H3) is added as a third factor only if Gate 2
     (pilot) completes with time to spare.
   - H4 (mixed- vs. same-model pairs) is demoted to exploratory/stretch: it
     multiplies the number of model-pairing cells and local inference is
     slower than API calls would have been, so it is not part of the
     confirmatory pre-registration unless time remains after Gate 3.
3. **Probe layer (Stage 5) is a stretch goal, not a committed deliverable.**
   It was already gated on "if PROBE_MODEL is set and time remains"; at 36
   hours that gate will likely land at or after the deadline. Logging
   (Section 3 rule 8: everything needed to reconstruct what each agent saw)
   stays mandatory regardless, so the probe layer remains runnable in a
   follow-up session even if not executed in this sprint.
4. **Gates are not skipped**, only compressed. Gate 1 (environment choice) is
   resolved immediately as "Plan B, no Colosseum trial" rather than spending
   90 minutes confirming it, so more of the remaining time goes to the pilot
   (Gate 2) and main sweep (Gate 3).

**Why:** Preserves the correctness/honesty bar (pre-registration, gates,
resumability) while fitting the reduced window by cutting the exploratory
and stretch portions first, not the confirmatory core.
