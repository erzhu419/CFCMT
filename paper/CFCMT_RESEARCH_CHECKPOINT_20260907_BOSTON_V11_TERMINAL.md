# CFCMT Research Checkpoint: Boston V11 Terminal Outcome (2026-09-07)

This note supersedes only the running-state statements for task `t89321` in
`paper/CFCMT_RESEARCH_CHECKPOINT_20260907_CONTINUATION.md`. It does not alter
the recorded V145, V146, V148 or V149 decisions.

## Terminal state

- `t89321` finished with evaluator exit code 2 after a valid full-admission
  execution; there was no runtime exception.
- Boston package v11 is rejected because one collision occurred at simulation
  time 13,146 s on lane `8647416#3_0`.
- Package, city, backend and full-scope identities matched the declaration:
  manifest `e46d49518dac1596c93fba04839f5a4508fc6296ee2d8bbb557e6baf37f7604e`,
  BOS, libsumo/SUMO 1.22.0, 3,806,510 trips, 4,203 traffic lights and the
  unrestricted 7,201--71,996 s horizon.
- The collision gate stopped execution before complete-day demand accounting;
  zero teleports was the only dynamic safety gate that remained true.
- The exact conflict is recorded in
  `paper/tsc_v118r_boston_v11_full_admission_outcome.md` and the small local
  result JSON named there.
- Automatic deterministic retry `t89359` is running. It does not authorize a
  controller experiment or provide an independent confirmation unit.

## Active boundary

V149 remains unauthorized. The next admissible engineering action is structured
localization of the `joinedS_514` link-0 merge conflict, followed by a newly
declared minimal package lineage and the full sequential admission ladder. No
controller evaluation may start from package v11.
