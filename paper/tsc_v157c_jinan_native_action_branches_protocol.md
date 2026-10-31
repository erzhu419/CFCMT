# TSC V157C Jinan Native One-Action Branch Protocol

V123 and V157A retained a favourable B100 model-path contrast on the reused
Jinan selector cache, but V157B found that their target and source paths used
different Jinan domain handling. V157B v2 exactly reproduced the historical
arrays for audit, then rechecked the fixed uniform-source arm against a
domain-aligned Jinan target comparator. Both required V157B gates passed, so
V157C was authorized to test whether the corrected comparison survives native
SUMO execution from matched physical states. This is a development validation
of the one-action label, not a full-controller test. The subsequent execution
is recorded separately because one of the three frozen seed units was invalid
at a fixed checkpoint; the protocol is therefore incomplete rather than passed.

## Frozen roster

The corrected V157B selector gate reused `jinan_3x4_real` over 22 seeds, whereas
the B100 training groups span all three Jinan scenarios. Testing the other two
training scenarios with one new seed would change the estimand from the one
that authorized this step. V157C therefore uses `jinan_3x4_real` and the first
three seeds in the already frozen V157A selector order: 80314, 88625 and 27178.
This rule was fixed without reading the new branch outcomes. The evidence is
still development evidence because the seeds occurred in the selector cache.

Each seed uses checkpoints at 300, 480 and 660 seconds. The longest run ends at
1110 seconds, within the frozen network's 3600-second horizon. One uninterrupted
PhasePressure trajectory per seed supplies the three checkpoint states and all
three 450-second PhasePressure continuations.

## One-action contract

Jinan has multiple controlled intersections, but each V123 counterfactual label
changes one focal traffic light for one 10-second interval. Simultaneous model
actions at every intersection would test a different estimand. At each
checkpoint, the V157B v2 domain-aligned B100 target-only model, fixed uniform
mean of the exact V157A-path source refits, and matched source-label placebo are
scored on the shared PhasePressure state. The exact V157A target refit is
audit-only and is not a runtime arm. The focal traffic light is chosen before
observing any branch outcome. A light is action-eligible at a checkpoint only
when its safe executor exposes more than one state through
`feasible_states_now()`. This is the exact set that the frozen V3 evaluator can
pass to an action originator; a light with one feasible state has no action
contrast and continues under PhasePressure without model scoring.

The focal rule is:

1. Consider lights where uniform-source and target-only choose different phase
   states.
2. Select the light with the largest maximum predicted advantage over
   PhasePressure across those two arms; break ties by traffic-light ID.
3. If no light differs, apply the same deterministic rule over all
   action-eligible lights and retain the window as an exact zero
   source-versus-target action contrast.

A fixed checkpoint with no action-eligible light is invalid. Its time is not
shifted and the checkpoint is not silently dropped.

For each learned arm, a fresh native simulation runs PhasePressure from t=0 to
the checkpoint, applies that arm's focal action for 10 seconds, and then runs
PhasePressure for 440 seconds. Every other traffic light uses PhasePressure for
the entire run. No SUMO state save or restore is used.

The three learned arms produce nine branch runs per seed. Together with one
shared PhasePressure trajectory, this is 10 native runs per seed and 30 total.

## Validity and analysis

Before accepting a branch, its checkpoint state must exactly match the shared
PhasePressure trajectory in vehicle IDs, lanes, routes, route indices,
positions, speeds, pending-vehicle IDs and safe-executor state. Its complete
three-arm action roster over the action-eligible lights at the checkpoint must
also reproduce the roster scored on the shared trajectory. The compact result
records this roster's IDs and count. A learned override may occur only at the
focal light and only at the checkpoint.

The cost is the mean per-second SUMO halted-vehicle count across all controlled
lanes, divided by the controlled-lane count, over seconds 1 through 450 after
the checkpoint. The primary paired contrast is fixed uniform-source minus the
domain-aligned target-only comparator. Uniform-source minus matched placebo is
the placebo contrast. Each learned arm is also compared with the shared
PhasePressure continuation. Identical selected actions remain in the analysis
as exact zero contrasts. The seed is the analysis unit after averaging its
three fixed checkpoints.

Collisions and teleports are retained in the diagnostics. Collision counts are
not the primary gate for this traffic-efficiency experiment. Any incomplete
horizon, prefix mismatch, action-roster mismatch, extra learned intervention or
teleport runtime failure makes the affected branch invalid.

V157C cannot establish fresh-seed transfer, unseen-city transfer, continued
model-control performance or safety superiority.

The first execution attempt after the input-interface repairs (`t92559`--
`t92561`) completed the shared PhasePressure trajectory but stopped before any
learned branch because the implementation incorrectly required model decisions
for single-candidate lights. No branch outcome from those tasks was produced or
used. The eligibility clarification above fixes that pre-outcome bookkeeping
error without moving a checkpoint or changing the selected-action rule.

The corrected execution completed all ten runs for seeds 80314 and 88625.
Seed 27178 stopped after its shared PhasePressure trajectory because the fixed
480-second checkpoint had no action-eligible light. Under the frozen rule above,
that checkpoint cannot be shifted, silently dropped, treated as a zero outcome,
or replaced by another seed. Consequently no complete three-seed aggregate or
V157C PASS exists. The two complete seed units remain usable only as descriptive
mechanism evidence; their adjudication is reported in the V157C result.
