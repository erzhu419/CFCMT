# TSC v112 movement-arrival redevelopment protocol

## Scientific status

V112 is disclosed post-confirmation redevelopment after the rejected V111
confirmation. All V83--V111 seeds and outcomes are open development evidence.
No result from V112 may be described as independent confirmation. A later V113
confirmation may be opened only after the complete V112 gate below is passed
and its model, information budget, thresholds, and seed generator are frozen.

## Failure addressed

V111 selected source identity and its deployment guard from one-step B100
counterfactual action regret. The selected guard then executed roughly two
hundred phase overrides per rollout and increased fresh-seed waiting time by
9.110% relative to exact PhasePressure. The previous local-demand feature was
a traffic-signal total: every candidate phase at a signal received the same
future demand summary. It therefore could not identify which candidate phase
served an approaching arrival wave.

V112 changes the predictive information, not the safety or confirmation rules.
It projects each scheduled route through controlled edge-pair movements and
provides each candidate phase with its served and unserved arrivals over 30,
60, and 120 seconds. A route traversal contributes once to a movement even
when SUMO represents that movement with several lane-to-lane connections.

## Information budget

The movement-arrival timeline may consume only:

- target SUMO network topology and signal link indices;
- target route, trip, or flow schedules available before deployment;
- edge lengths and legal speeds for an uncalibrated free-flow arrival time;
- current online signal state and the candidate signal state.

It may not consume target transition outcomes, tripinfo, queue labels,
counterfactual branch outcomes, or a target closed-loop policy result. B100
target counterfactual labels remain available only to fit/select the residual
model after the deterministic arrival features have been constructed.

## Equal-information controls

The same movement-arrival timeline is supplied to both:

1. the CFCMT mechanism model; and
2. an anticipatory-pressure rule that adds a frozen candidate arrival-wave
   score to ordinary phase pressure.

The primary method comparison remains exact PhasePressure. The
anticipatory-pressure control determines whether any gain comes from the new
schedule information alone rather than causal mechanism transfer.

## Ordered development gate

### Gate A: deterministic static audit

Across every admitted Jinan and Los Angeles full network, route projection must
be deterministic, finite, non-negative, and non-empty. Candidate green phases
must receive different arrival vectors for at least one signal-time pair. Each
scheduled route traversal is counted once per traversed controlled movement,
independent of the number of parallel lane connections.

### Gate B: group-blocked offline development

All tuning is nested inside action-group folds. The policy-relevant target is
the longest available waiting-aligned rollout prefix, not one-step local
regret. Source identity, source mass, arrival-score coefficient, and residual
capacity are selected without evaluating a held-out action group. Report exact
PhasePressure, anticipatory pressure, target-only CFCMT, and source-weighted
CFCMT under identical B100 target information.

#### Gate B1 outcome: static movement arrivals

Gate A passed on all audited Jinan and Los Angeles route packages. The
equal-information anticipatory-pressure coefficient selected zero in every
leave-one-seed-out fold. Adding the same arrivals to the matched latent model
reduced ungated error slightly but did not produce a feasible guarded policy:
the best static-arrival safe subset had a larger harmful-action fraction than
the matched no-arrival model. Static movement-arrival features are therefore
rejected from the primary CFCMT path. This is a negative development result,
not a confirmation result.

#### Gate B2: waiting-aligned causal source admission

The seven V93 source-city components and the strict V98 target-only model are
evaluated without refitting on 22 simulator seeds disjoint from their B100 fit
seeds. Model scores use their frozen generalized-pressure contrast, while the
policy estimand and fallback are exact PhasePressure under the 450-second
halted-queue target. Source weights are limited to 0.25, 0.50, 0.75, and 1.00;
deployment guards are the Cartesian product of risk multipliers 0, 0.5, and 1
and minimum context trust 0.25, 0.50, and 0.75.

Selection is nested leave-one-simulator-seed-out. A nonzero source profile must
simultaneously satisfy all of the following on the training seeds:

- family-wise one-sided upper confidence bounds are non-positive both versus
  exact PhasePressure and versus the same guard applied to strict target-only;
- mean normalized improvement is at least 0.0015 versus PhasePressure and
  0.0005 versus target-only;
- worst-seed normalized regression is at most 0.0075;
- the harmful fraction among accepted overrides is at most 0.42; and
- the intervention fraction is between 0.0025 and 0.05.

The source and target-only search families each receive one half of a total
0.05 one-sided family-wise error budget. If no source passes, the selector
falls back first to a separately certified target-only guard and then to exact
PhasePressure. Gate B2 advances only when at least half of nested folds select
a source, the held-out mean improves by at least 0.0015, the seed-bootstrap
95% upper limit is below zero, no held-out seed regresses by more than 0.0075,
and at most one quarter of held-out seeds regress.

#### Gate B2 outcome and Gate B3 multisource consensus

Gate B2 rejected every individual source component. All 22 nested folds fell
back to exact PhasePressure. The result does preserve one real positive effect:
source blending substantially reduced the loss of the strict target-only
model. However, every individual-source profile with a material intervention
rate still increased waiting relative to PhasePressure, while the profiles
closest to PhasePressure accepted essentially no overrides. Thus the older
"source improves target-only" result is valid but does not establish a useful
source policy effect against the classical controller.

Gate B3 tests whether agreement among independently fitted source mechanisms
can identify a smaller invariant action set. The seven source scores are
aggregated by elementwise mean, median, and worst-source score. Each aggregate
uses the same source-weight, risk, and trust grids as Gate B2, with an additional
minimum source-support fraction of 0.6, 0.8, or 1.0 for the proposed action.
All source predictions are label-free and computed before nested selection.
The B2 family-wise admission thresholds and exact fallback remain unchanged.
This is disclosed redevelopment after observing B2 and requires a new V113
confirmation if it passes.

#### Gate B3 outcome

Gate B3 also rejected the one-step source components. All 22 nested folds
selected exact PhasePressure. The best active consensus profile was the
all-source mean with source weight 0.5, risk multiplier 1, and unanimous
source support. Its source contribution relative to the matched target-only
guard was favorable (approximately -0.00650 mean normalized waiting cost),
which independently confirms the earlier positive transfer signal. However,
the same profile remained approximately +0.00089 above exact PhasePressure
and 52.1% of its accepted overrides were harmful. Source averaging therefore
reduces target-only error but does not repair the training/deployment estimand
mismatch.

#### Gate B4: waiting-aligned source and target rebuild

Gate B4 replaces the one-step training label rather than adding another
deployment residual. Every source mechanism is refit on the exact 450-second
halted-queue target used by source admission. The full source collection
attempts all 18 scenarios, three simulator seeds, and 16 collection shards per
scenario-seed. A scenario enters training only after complete cache identity,
replay, symmetric collision censoring, target equivalence, behavior-trace,
and TLS-coverage audits pass.

Ingolstadt21 produced all 48 requested cache files but retained zero action
groups: every 450-second branch group was safety-censored. A separate native
SUMO audit, with no external signal control, found collisions in all three
source seeds (55, 59, and 67 collision-event steps). It is therefore retained
as a complete failed-admission case and excluded from mechanism fitting. The
training bank has 17 scenarios while retaining all seven source city groups;
Ingolstadt1 and Ingolstadt7 remain admitted. Collision monitoring, demand,
and TLS coverage thresholds are not relaxed.

Target adaptation is rebuilt on the same estimand for the complete Los Angeles
and Jinan target manifest: one Los Angeles scenario and all three Jinan demand
scenarios, seeds 5057 and 6067, with 16 shards per scenario-seed. The target
pool is selected by the frozen scenario- and seed-balanced coverage-first B100
rule. Three equal-information models are fit:

1. B0 source mechanisms with target static context and no target transition
   labels;
2. strict target-only B100; and
3. source-augmented CFCMT B100 using exactly the same selected target groups.

All three use exact PhasePressure as their action reference. B0 is the
target-label-free comparison; B100 is target offline adaptation and is never
called zero-shot. Gate B4 advances only after the 17-scenario source audit and
four-scenario target audit pass, followed by the unchanged nested
leave-one-seed source-admission criteria from Gate B2.

#### Gate B4 implementation audit and invalidation

Gate B4 did not advance to model fitting. An implementation audit found that
the cache metadata declared a sum of SUMO halted-vehicle counts, whereas the
runtime cost path called the legacy `_lane_queue` feature and therefore added
`0.35 * vehicle_count + 0.025 * occupancy` to every lane-second. The resulting
label was a composite queue feature, not the declared pure halted-vehicle
integral. At discovery, 816 of 864 source shards and 118 of 128 target shards
had been written. The two remaining jobs were stopped; all written files are
retained as rejected diagnostic evidence and are excluded from every fit,
selector, table, and claim.

Gate B5 restarts collection in empty roots under the new
`pure-halted-queue-rollout-value-v2` contract. The cache identity now embeds the
full cost contract, and a regression test fixes the lane-second cost to
`getLastStepHaltingNumber` even when vehicle count and occupancy are nonzero.
No B0, B100, source-selection, or closed-loop result is licensed until the
replacement 864-file source and 128-file target audits pass.

### Gate C: disclosed closed-loop development

Run paired full-horizon, full-network development seeds on all admitted Jinan
and Los Angeles demand scenarios. A candidate advances only if:

- it is non-inferior to exact PhasePressure in each city at a +1% paired mean
  waiting-time margin;
- it improves the pooled equal-city paired mean over exact PhasePressure;
- it improves over the equal-information anticipatory-pressure control;
- its intervention rate and pressure-rule disagreement are reported; and
- collisions and teleports satisfy the existing policy-independent admission
  rules without post-hoc seed removal.

Passing Gate C licenses protocol freezing, not a publication claim.

## V113 boundary

If all V112 gates pass, V113 uses new PCG64 seeds disjoint from V83--V112 and
one immutable three-arm comparison: exact PhasePressure, frozen anticipatory
pressure, and frozen CFCMT. The primary estimand is paired mean trip waiting
time with equal city/scenario aggregation. Any method or threshold change after
reading V113 outcomes invalidates V113 and requires another confirmation.
