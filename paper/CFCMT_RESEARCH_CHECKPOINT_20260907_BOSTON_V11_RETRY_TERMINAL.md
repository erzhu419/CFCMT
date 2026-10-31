# CFCMT Research Checkpoint: Boston V11 Retry Terminal (2026-09-07)

This note appends the terminal state of automatic scheduler retry `t89359` to
`paper/tsc_v118r_boston_v11_full_admission_outcome.md` and
`paper/CFCMT_RESEARCH_CHECKPOINT_20260907_BOSTON_V11_TERMINAL.md`.

Task `t89359` completed on `node005` and exactly reproduced the scientific
failure from parent task `t89321`: simulation time 13,146 s, collision vehicles
`77188` and `95307`, lane `8647416#3_0`, position 2.628 m, all vehicle-state
counts, all admission checks and zero teleports were identical. Package,
full-scope and runtime contracts were also identical. Only the host and
wall-clock execution time differed (`4,221.43` s versus `3,759.18` s).

The retry result is stored at
`cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/full_admission_v8/result_t89359.json`.
It confirms deterministic reproduction but is not an independent city-level
scientific unit. Boston package v11 remains rejected, and V149 plus all Boston
controller evaluation remain unauthorized. No successor package is launched by
this checkpoint.
