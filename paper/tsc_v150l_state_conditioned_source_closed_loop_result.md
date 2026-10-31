# V150L State-Conditioned Source Closed-Loop Result

## Decision

V150L is retained as a negative closed-loop development result. Its v3 smoke
matrix passed the pre-existing operational safety contract, but it did not pass
a performance screen and therefore does not authorize the 216-rollout full
matrix.

The later V154 audit identified a feature-column error in its rigid scoring
path. The outcomes below remain factual records of the original implementation;
they do not isolate the efficacy of correctly evaluated rigid/source methods.
The isolated correction is recorded in
`paper/tsc_v154_rigid_feature_alignment_correction.md`.

## Evidence

The complete v3 smoke matrix contains four cities, one previously unused
development seed (`41242` for this amended smoke), and four matched arms. All 16
rollouts completed. Every arm had zero teleports. The selected-source arm was
non-inferior in collision incidents to both rigid CFCMT and PhasePressure in
every paired scenario-seed unit. Cities rejected by the offline selector
returned the rigid policy exactly.

The performance result was unfavorable:

| City | PhasePressure | Rigid | Source | Placebo | Source minus rigid |
|---|---:|---:|---:|---:|---:|
| Atlanta | 2276.759 | 2248.066 | 2248.066 | 2248.066 | 0.000 |
| Cologne | 14.991 | 1484.364 | 2047.510 | 1535.594 | +563.145 |
| New York | 1085.870 | 1331.852 | 1366.917 | 1393.513 | +35.065 |
| RESCO synthetic | 31.166 | 122.289 | 100.585 | 97.282 | -21.703 |

The source arm recorded two direct source-induced changes from its rigid choice
in Cologne, with mean waiting 37.9% above the separate rigid rollout. This count
does not imply that the two resulting trajectories differed at only two action
times. The later feature-column diagnosis affects both the rigid scoring and
the offline utility evidence, so this comparison cannot by itself establish a
failure of a correctly implemented mean-regret selector.

## Protocol Interpretation

The aggregate JSON reports an operational `PASS` because the frozen v3 smoke
gate intentionally contained no performance checks. That status means the
runner, paired collision audit, and exact fallback worked. It is not a
scientific performance pass. Expanding V150L after observing the Cologne
regression would be unjustified.

The 2026-09-09 v4 adjudication preserves that v3 artifact and applies the existing
development performance thresholds to the retained smoke outcomes as an explicit
post-smoke stop screen. It reports `operational_passed=true`, `passed=false`, and
`decision=reject_v150l_full_matrix_after_smoke`. This corrects the v3 machine
decision that incorrectly authorized expansion on operational checks alone;
it does not describe a new rollout or a preregistered smoke performance test.

The local service audit also changes the interpretation of Cologne: rigid
itself admitted only 409 of 2,015 demanded vehicles, with 210 arrivals and 1,606
vehicles still waiting for insertion. Source admitted 363, with 128 arrivals
and 1,652 waiting for insertion. Rigid spent 3,505 of 3,600 seconds in green and
made only 22 phase switches, so excessive clearance time is not the leading
explanation. The retained trace shows persistent phase holding and zero-speed
traffic. V154 subsequently reproduced the trace exactly and observed red-bound
heads blocking the two nominally green shared lanes, with all receiving lanes
empty. A model-input audit then reproduced the bad phase scores exactly from
nine misbound feature columns.
Cologne1 has one controlled TLS; concurrent interventions across several
controlled intersections cannot explain this particular failure.

The historical V150M experiment changed the candidate set before nested utility fitting: a
source or placebo proposal may keep the rigid action or select the current
PhasePressure reference. Any third action is replaced group-wise by the exact
rigid score. This constraint is applied symmetrically to real source and matched
placebo candidates, both offline and online.

## Artifact

Canonical aggregate:
`cf_h2o/results/paper_artifacts/tsc_v150l_state_conditioned_source_closed_loop_smoke_v3_paired_safety.json`.

Corrected machine adjudication:
`cf_h2o/results/paper_artifacts/tsc_v150l_state_conditioned_source_closed_loop_smoke_v4_performance_adjudication.json`.
