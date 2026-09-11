# GPT6 Diagnosis Remediation Progress

## Status

The diagnosis has been converted into executable stages rather than treated as
a new generic gate.

On 2026-09-09, an exact model-input audit found a positional-column error in the
expanded-contrast rigid scoring path: nine of the Cologne anchor's 29 inputs
were read from the wrong fields. This affects V150K offline/online comparisons
and the V153 path as well as the observed V150L control failure. After correction,
the same original Cologne model and seed improved from 1484.364 to 19.713 seconds
mean waiting, with arrivals increasing from 210 to 1997 and pending insertion
falling from 1606 to zero. That original-network rerun still had eighteen
collision incidents. V153 v3 has now
completed all seven cities with corrected feature binding and strict nested
selection: zero cities admitted source and all exactly recovered corrected
rigid. The intermediate V153 v2 admission is superseded for interpretation.
See `paper/tsc_v154_rigid_feature_alignment_correction.md`.

The subsequent fixed-network V154F comparison is complete on the existing
three-seed roster. Rigid has zero collisions in 3/3 seeds; PhasePressure has
zero in 2/3, with one incident in seed 52282. Rigid remains slower in every
pair: mean waiting is 20.125 versus 13.354 seconds (+50.70%) and system vehicle
hours are 40.524 versus 32.424 (+24.98%). Arrivals differ by only 1.67 vehicles
per seed on average. The waiting-point repairs support these observed rigid
trajectories, while a separate PhasePressure collision and the rigid service
gap remain. See the V154F section below.

| Diagnosis item | Experiment or repair | Current result |
|---|---|---|
| Source benefit repeatability | V150A leave-one-seed-out inventory | Positive headroom repeated: forced source improved 16/21 seed units and all 7 city means, but this remains diagnostic selection using other evaluation seeds. |
| Deployment-observable representation | V150B action-independent city signature | Rejected; enriched representation did not identify source utility. |
| Mechanism-level prior | V150C source prior vs same-architecture target-only | Passed the frozen development gate; 6/7 city means improved, macro effect `-0.00408763`. Source identity versus placebo remained weak. |
| Rigid integration | V150D mechanism correction on rigid CFCMT | Rejected; macro effect `+0.00478435`, with a large Ingolstadt regression. |
| Source-identity admission | V150E source must beat target-only and matched placebo in B25 OOF | Rejected; macro effect `+0.00829202`, with regressions in Atlanta and New York. |
| State vs future randomness | V150F RNG-free fixed-state paired futures | Partial: RNG omission and exact replay passed; 93/105 pairs were valid, all 7 cities had cost-label variation, but three Cologne/Hangzhou checkpoints failed the minimum-valid-futures rule. |
| Complete-selector crossfit | V150G leave-one-B25-fold selection evaluation | Rejected; only Hangzhou admitted a source. Exact fallback prevented regressions, but the frozen gate required at least two independently improving cities. |
| Target-information sensitivity | V150H nested B25/B50/B100 complete-selector curve | Rejected; admissions increased from 1 to 2 cities, but B50 and B100 admitted harmful sources on the common untouched evaluation set. More labels alone did not solve identification. |
| Target-evidence-aware shrinkage | V150I inverse-budget source-prior precision | Partial repair, gate rejected; B100 removed the harmful Cologne admission and became non-degrading in 7/7 cities, but only Atlanta admitted a source, so multicity source generality remains unsupported. |
| Conditional mechanism representation | V150J deterministic local right-of-way and neighboring-execution interactions | Rejected; all seven full-data selectors found apparently useful candidates, but none survived complete five-fold cross-fitting. Exact rigid fallback was used in 7/7 cities. |
| State-conditioned source utility | V150K nested per-state source-mechanism gate | Historical offline PASS: three cities admitted source, with effects `-0.00051819` versus rigid and `-0.00085633` versus placebo. The feature-column defect affects this utility fitting and its comparisons; they require recomputation, and the old utility payload does not authorize use with the corrected rigid scorer. |
| State-conditioned closed loop | V150L four-city paired-safety smoke | Operationally passed, but not scientifically authorized for a full matrix. Cologne degraded from `1484.364` under rigid to `2047.510` under selected source; New York also degraded. The 2026-09-09 v4 adjudication now explicitly rejects full-matrix expansion while retaining the original operational PASS. The 216-rollout matrix was not launched. |
| Pressure-aligned utility | V150M rigid-or-PhasePressure candidate projection | Rejected with source admission `0/7`. Final nested record counts were 0--15, all below the frozen minimum of 48. |
| Pressure-pairwise utility | V150N source preference over PhasePressure versus rigid | Rejected with source admission `0/7`. The final reserve had 3--1,682 raw source proposals per city, but nested B100 record counts remained 0--20 because proposal-conditioned labels were too sparse. |
| Dense pressure-pairwise utility | V150O fixed rigid-versus-PhasePressure labels plus source-blind control | Rejected. Record support increased to 1,512--3,240 per city and Atlanta/New York passed nested admission, but Atlanta lost to both source controls on untouched target groups. New York retained a source-specific gain. |
| Feature-aligned dense utility replay | V156A exact V150O replay with stored-name rigid binding | Rejected globally and city-specifically. Atlanta/New York admissions disappear; Hangzhou alone is admitted in cross-fitting but worsens on the untouched reserve against rigid (`+0.00942188`), placebo (`+0.00061303`) and source-blind (`+0.00092563`). No city qualifies for closed-loop follow-up. |
| Historical Jinan source-value replay | V157A exact feature-name replay of the V123 budget curve | Historical arrays reproduced, but the B100 target and source paths used different Jinan domain labels. The apparent relative source gain is confounded and provides no current branch authorization. Both learned arms remain worse than PP. |
| Domain-aligned Jinan B100 freeze | V157B v2 exact refits and common-`jinan` target comparator | The reused-selector source-minus-target gate passes (`-0.01357348`, 20/22 seeds), while source remains `+0.02533749` worse than PP. This authorized only V157C's fixed one-action development branch. |
| Native Jinan one-action check | V157C three fixed seeds and three fixed checkpoints | **INCOMPLETE**: two seeds completed and seed27178 had no action-eligible light at fixed t=480. The two-seed descriptive source-minus-target mean is `+0.00236368` (1/2 improves), and source is `+0.00361626` worse than PP in 2/2. No seed replacement, three-seed inference or adoption. |
| Cross-city meta utility | V151A leave-target-city-out utility relation | Rejected with source admission `0/7`. Every target recovered rigid CFCMT exactly, but the source-contribution conditions failed. The labels were already action-group-range normalized, so the result does not support a cost-scale explanation. |
| Robust pooled mechanism prior | V152A leave-target-city-out equal-city hierarchical prior | Rejected with source admission `0/7`. The forced prior improved 3/7 cities but regressed four, including `+0.022101` in Hangzhou. All rejected targets recovered rigid exactly. |
| Target-calibrated hierarchical prior | V153A B100 prior selection, final correction v3 | All seven v3 jobs completed using strict nested folds and name-aligned rigid inputs. REJECT, `0/7` admissions, exact corrected-rigid fallback in every city. Forced prior has macro normalized effect `-0.012130`, interval `[-0.038793, +0.002660]`; Atlanta dominates that mean but ties placebo and zero-centred controls. V1/v2 remain historical implementation records. |
| Rigid service failure | V154 exact replay and feature-column correction | Reproduced all 154 original intervention rows exactly, then proved that 9/29 anchor inputs read the wrong columns. Fixing name binding alone reduced Cologne waiting `1484.364 -> 19.713`, increased arrivals `210 -> 1997`, and removed all final pending insertion. Eighteen collision incidents remain, versus 20 under PhasePressure; corrected waiting still exceeds PhasePressure's 14.991 s. Occupancy-unit formulas remain a separate defect. |
| Fixed repaired network comparison | V154F rigid versus PhasePressure, three existing seeds | All six cells valid. Rigid: zero collisions in 3/3 seeds. PhasePressure: one incident in 52282, zero in 2/3. Rigid waiting `20.125` versus `13.354` s and system vehicle hours `40.524` versus `32.424`; both are higher in every pair. Five new simulations, one reused V154E cell. |
| Boston network repair | v18 controlled shared-receiving response repair | Microscopic admission rejected and repair branch closed. Package v18 passed static admission and complete-route admission for all 3,806,510 trips, then failed the 10,800 s trigger at simulation time 17,265 s. Read-only task `t90329` reproduced the same time, lane and participants. The victim changed from lane 0 into the collider's lane at 17,264 s with a `2.288 m` bumper gap (`0.788 m` clearance above the `1.5 m` minimum), then braked at `-7.068 m/s2`; the bumper gap fell to `0.943 m` at 17,265 s and triggered the frozen minimum-gap collision criterion; the watched major foe lane was empty throughout the trace. Static audits found a unique 2-to-2 route mapping, exact downstream priority relation, no lane merge and 605 other explicit 9 m minor-visibility values. The package failed the strict microscopic protocol; the native mesoscopic source is a compatibility concern, while the final braking cause remains unresolved. Absence of internal edges alone does not establish that microscopic use is impossible. No v19 or full-day run is authorized. |

## What Still Holds

Rigid/action-contrast CFCMT remains a positive structural result. V148's exact
source-null fallback beat the dense H2O+-style residual in all seven cities, with
a mean paired difference of `-0.06914109`. V150A also shows that useful source
headroom exists. Neither fact proves that the currently deployable selector can
identify and exploit the correct source.

## Current Boundary

V150K authorized a bounded source-aware closed-loop development test, but V150L
then rejected that deployment route: the Cologne source arm made two direct action overrides and had a large
closed-loop regression, and New York also degraded. Rigid itself showed severe
service loss, so the deterioration cannot be explained solely by source use. No full source-aware
closed-loop matrix is currently authorized.
V150F found real future-seed variation, but its city-macro
within-state variance fraction was only `0.00905`, and only two of 20 analyzable
checkpoints changed utility sign. V150G then cross-fitted candidate selection
itself. It admitted only Hangzhou and returned exact rigid fallback elsewhere,
for a safe but non-general macro effect of `-0.00091399` versus rigid and
`-0.00060177` versus the matched placebo. V150H showed that larger target budgets
do not cure this by themselves: B50 admitted an Ingolstadt regression of
`+0.03881`, while B100 admitted a Cologne regression of `+0.00347` and failed
the source-identity placebo comparison. V150I corrected the fixed-prior defect by
scaling source precision as `0.10*25/B`. This removed B100 negative transfer:
the macro effect was `-0.00523386`, all seven cities were non-degrading and the
matched-placebo effect was `-0.00000833`. However, only Atlanta admitted a
source, so the two-city generality gate still failed. The remaining bounded
hypothesis is representation: the current mechanism design gives each block one
state conditioner and does not express the joint local right-of-way and neighbor
execution conditions identified in the diagnosis.

V150J tested that representation hypothesis at B100 with the corrected `0.025`
source-prior strength. It did not authorize source transfer: source admission
was `0/7`. Although every full-data selector found a candidate satisfying its
mean filters, the complete held-out selector audit exposed unstable gains or
failed source-identity comparisons in every city. This closes the bounded
city-level representation branch. The remaining development hypothesis changes
the supervision unit: a low-complexity utility model may decide, per observable
decision state, whether any source-mechanism proposal should replace rigid
CFCMT. The proposal model, utility labels, utility gate, candidate choice and
matched placebo must all be nested out of fold; otherwise the experiment would
repeat the V142/V140 failure under a new name. V150K implemented that complete
nested pipeline and passed: three cities admitted source-mechanism interventions,
the city-unit confidence intervals versus rigid and placebo excluded zero, and
all rejected cities recovered rigid exactly. The gain remains small and the
seed-unit intervals cross zero, so this authorizes only a source-aware
closed-loop development comparison. A method freeze still requires closed-loop
non-degradation against rigid CFCMT and a matched placebo; publication-level
source benefit then requires untouched cities.

Boston is an independent package-admission task. Passing it cannot rescue a
source-transfer claim, and a Boston failure cannot negate the seven-city model
results.

Boston v18 has now been rejected under that independent gate. The exact
collision reproduction and static audits are recorded in
`paper/tsc_v150_boston_v18_microscopic_admission_result.md`. The observed
lane-change and braking sequence does not expose a narrow erroneous topology
relation. The 2026-09-09 corrected trace summary distinguishes the TraCI leader clearance
from physical bumper gap. The original v18 FAIL is retained. Further
car-following, lane-changing, visibility or collision parameter changes on this
observed event would be post-observation calibration; no-internal-links itself
is a supported SUMO simplification, not proof of inevitable microscopic failure. The repair chain is
therefore closed without v19 or a full-day run.

V150O subsequently removed V150N's proposal-conditioned record bottleneck. It
admitted Atlanta and New York and improved rigid CFCMT by a seven-city mean of
`-0.00122377`, but failed the frozen source-identity gate: Atlanta was worse
than matched placebo by `+0.00024366` and worse than source-blind by
`+0.00058923` on target groups outside B100. This rejects target-local dense
utility as the final source selector. The next bounded method must train the
utility relation across leave-one-city-out pseudo-targets, not alter V150O's
observed thresholds.

V151A performed that leave-target-city-out test and rejected it without a
target regression: all seven targets used exact rigid fallback. Across the six
pseudo-target folds available to each target, most source gates failed the
minimum mean-gain or cross-city nondegradation conditions, and no target passed
all source, placebo and source-blind comparisons. This closes the per-state
dense utility-gate branch. The next bounded method changes the transfer object
to a robust equal-city hierarchical mechanism prior selected entirely on source
cities; it does not reuse V151A with weaker thresholds.

V152A implemented that robust prior and also rejected it with exact fallback in
all seven cities. Its forced arm improved Atlanta, Cologne and RESCO synthetic,
but regressed the remaining targets and had a city-macro effect of `+0.002769`.
The target-label-free source-city selector therefore did not predict where the
pooled prior would help. This closes the source-only selector branch for the
current seven-city representation. The next bounded experiment may use the
already-declared B100 target adaptation labels to select among the same pooled
mechanism priors, provided target-only receives the same labels and all target
evaluation groups remain untouched. That is a few-shot target-adaptation claim,
not zero-shot transfer.

V153A v3 completed the B100 few-shot test with both implementation defects
corrected: strict inner folds exclude the outer held-out groups, and rigid
inputs are resolved by their stored feature names. All seven tasks
`t90757`--`t90763` completed. No target admitted source; all selected effects
are zero because every city exactly recovers corrected rigid. Forced source
has macro normalized-cost effect `-0.012130`, with city-bootstrap interval
`[-0.038793, +0.002660]`. Atlanta contributes `-0.090340`, but its source,
placebo and zero-centred reserve means are identical. That improvement does
not establish a benefit from the source prior centre. Ingolstadt's temporary
v2 admission disappears under correct rigid scoring. The unchanged candidate
grid and admission thresholds still reject the operational source claim.
The v1/v2 artifacts remain preserved, with their implementation limitations
identified. This recomputation uses the same development bank, not independent
confirmation. Detailed results are in
`paper/tsc_v153a_target_calibrated_hierarchical_prior_v3_result.md`.

## 2026-09-09 Correction And Service Diagnostic

Boston trace summary v2 corrects the leader-gap API interpretation while
retaining the raw trace, old summary and package FAIL. V150L adjudication v4
separates its operational PASS from a performance REJECT using the existing
development thresholds; no full matrix was launched. V153A v2 refits the
complete nested selector with unchanged B100 groups, candidate grid, controls
and thresholds. All seven correction jobs completed; Ingolstadt passed and the
seven-city gate remained rejected. Its separate correction launch is recorded under
`tsc_v153_target_calibrated_hierarchical_prior_20260909/nested_cv_launch_v2.json`.
The subsequent feature-column diagnosis affects those v2 source comparisons;
their temporary Ingolstadt admission is not current source evidence. The final
v3 correction completed all seven tasks and returned exact corrected-rigid
fallback in every city; its launch is
`tsc_v153_target_calibrated_hierarchical_prior_20260909/feature_binding_launch_v3.json`.

The retained Cologne1 trace contains one controlled TLS, prolonged holding of
`rrrGGrrrrrrrrGGrrrrr`, and a sustained zero-speed suffix. Rigid used 3,505 s
of green time and only 22 switches, with 1,606 of 2,015 due vehicles still
waiting for insertion at the end. Static TLS inspection found that the held
phase grants left/U-turn movements on two shared through/left lanes while
their through movements remain red. The current lane-level aggregation counts
all queue on a lane with any green movement as green queue. The bounded
read-only replay `t90447` completed and reproduced all 154 retained original
intervention records exactly. At 26700, both shared-lane heads wanted red
straight movements while 10 and 4 vehicles behind them wanted green movements;
all eight directly connected outgoing lanes were empty. The rigid score still
favoured retaining that phase. This identifies an accessible-service mismatch
in this failure, not downstream saturation or excessive switching. Detailed
observations are in `paper/tsc_v154_cologne_rigid_service_diagnostic.md`.
The subsequent model audit identified nine wrongly read columns and reproduced
the erroneous scores exactly. A full original-start 3600-second correction
replay, `t90756`, removed the severe service collapse without changing the fitted
model or traffic parameters. Its zero-incident diagnostic remains FAIL.

The separate saved-state diagnostic `t90749` stopped at its reproduction
prerequisite: initial pending insertion was read as zero instead of 526, although
later samples matched. It did not run the switch branch and supplies no causal
intervention effect. That failure is retained rather than relaxed into a pass.

## 2026-09-09 Four-City Correction, Collision Geometry And Occupancy Units

The original four-city smoke roster has now completed the feature-binding-only
correction. Atlanta, New York and RESCO used new tasks `t90907`--`t90909`;
Cologne reuses `t90756`. All retain their original fitted models, seed 41242,
3600-second runs and controller parameters in frozen snapshot
`589fd20266b7265b6f2f`. No source correction or occupancy-formula change entered
these comparisons.

| City | Waiting, original → corrected (s) | Arrived, original → corrected / due demand | Corrected pending | Corrected collision incidents |
|---|---:|---:|---:|---:|
| Atlanta | 2248.07 → 2194.33 | 140 → 150 / 2171 | 1729 | 0 |
| Cologne | 1484.36 → 19.71 | 210 → 1997 / 2015 | 0 | 18 |
| New York | 1331.85 → 1281.19 | 2790 → 3096 / 15841 | 4244 | 0 |
| RESCO synthetic | 122.29 → 59.02 | 1445 → 1442 / 1473 | 0 | 0 |

Waiting includes all departed vehicles, including unfinished tripinfo records
at the horizon; pending insertion is excluded from that mean. System vehicle
hours, which include pending vehicles after warmup, decrease by 0.67%, 95.37%,
2.04% and 24.72%, respectively. Cologne and RESCO show substantial fixed-scenario
improvement. RESCO's three fewer arrivals equal 0.20% of due demand and are an
acceptable tradeoff against its 51.74% waiting reduction. Atlanta and New York
remain service-limited. Corrected rigid does not consistently outperform the
retained PhasePressure comparators. The complete comparison is in
`paper/tsc_v154b_feature_aligned_rigid_smoke_result.md` and
`cf_h2o/results/paper_artifacts/tsc_v154b_feature_aligned_rigid_smoke_v1.json`.

The Cologne first-collision replay `t90912` completed from the original start
under observation snapshot `85b2b43cd12832c5a1dd`. All 23 action-trace rows and
both collision reports reproduced exactly, with 108 observations across the
simulation/executor stages. A stopped left-turn passenger vehicle intrudes
0.056509 m into the body of the already-entered straight vehicle at 25657.
The intrusion begins during green, before yellow and all-red. The next phase
opens only at 25665. The intended static conflict and left-yielding relations
are present; the specific failure is insufficient clearance at the realized
internal waiting position. Extending the current all-red interval alone cannot
remove this pair's existing geometric conflict. The original network is intact.
See `paper/tsc_v154_cologne_first_collision_diagnostic_result.md` for the
measured polygons, signal sequence and remaining scope.

The occupancy correction is now wired into pressure features, the complete
analytic prior and the spillback-pressure baseline. Fractions use receiving
divisors 1.2 and 1.35, with the occupancy output, speed term and cost term
converted together. The empirical queue-proxy coefficient is unchanged.
Cache versions v21/v22, cache load/merge, source-rule cache identity, V150K
runtime-model v2 and V150L model/input validation carry the same fraction
equation contract. The new path rejects old caches and runtime payloads;
frozen snapshots remain the reproduction path for old results. Diagnostic
fields now say `occupancy_fraction`; old raw JSON is retained.

Fresh pilot `t90917` passed under snapshot `484f9fe12a951597a2bb`: 300 seconds,
120-second warmup, 30-second control and prediction horizon, one Cologne TLS,
seed 41242. It collected six fresh groups of eight candidates (48 rows),
preserved cache arrays and metadata exactly through reload, matched the
fraction service-pressure equation with zero error, and fitted a nonconstant
ranker whose serialized predictions matched exactly. Runtime was 2.143 seconds.
The fresh bank and model remain on the server; only the 6869-byte result JSON
was retrieved. Occupancy-related verification totals 107 passing tests;
the three observation-hook and geometry regression tests also pass.
See `paper/tsc_occupancy_fraction_contract.md`.

The pilot validates collection and model plumbing on a low-occupancy window,
not closed-loop improvement or a complete V150K source-utility deployment.
The four-city evidence above retains the old occupancy equations and cannot
be presented as performance evidence for the new formulas. Source efficacy
remains unestablished. The separate waiting-point geometry package progressed
from the short V154C/V154D diagnostics to the original-duration V154E validation
reported below.

## 2026-09-09 V154C Waiting-Point Geometry Repair

The independent geometry package and paired validation are complete. Task
`t90940` reproduced the original baseline exactly: 23 retained action records,
two collision reports and 108 observations across 466 seconds. Both arms used
the original model and controller snapshot `85b2b43cd12832c5a1dd`, including
its old occupancy equations. The validation driver came from tooling snapshot
`d07d45b3d0700d993910`.

The package changes exactly six XML attributes on the 13→24 internal waiting
split: two lane shapes, two lane lengths and the waiting junction coordinates.
The split moves from 8.76 to 8.48 m, with continuation length increasing from
19.77 to 20.05 m. The complete path, connections, conflict responses, signals
and vehicle parameters are preserved. A 4.3 × 1.8 m passenger body with its
front exactly at the new split has 2.37 mm clearance from the opposing straight
swept strip. Another vehicle approaching rest at the repaired point has
measured clearance of 3.565 cm, close to the pre-run prediction of 3.562 cm
with the original stopping offset. This is direct support for the local
geometry repair; it is a different vehicle from the original collision pair.

**The frozen paired validation remains FAIL.** Both simulations completed
normally with no teleports, but the repaired arm retained one native collision
incident at 25629 on the untouched mirrored 3→20 waiting point against straight
lane 11_1. Its stopped body intruded by 4.96 cm. This movement pair also collided
later in the original full rollout, so the result exposes an existing mirror
defect under changed traffic feedback. The original left focal vehicle passed
the junction, while the original straight focal vehicle had not exited by
the fixed endpoint. Their encounter was not reproduced. Arrivals were 256
versus 247, with one pending vehicle in each arm; this window establishes no
service improvement.

The geometry and validation tests total 13 passes. The paired runner took
3.689 seconds; only the 4,486-byte geometry report and 672,985-byte result JSON
were retrieved. Network, route and model files remained on the server.
The result, measured geometry comparison and retained failure evidence are in
[the V154C report](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154c_cologne_waiting_geometry_repair_result.md).
The mirrored waiting point was subsequently repaired under the same
body-clearance rule in V154D, reported below. V154C's shift, fixed window,
thresholds and FAIL are retained.

## 2026-09-09 V154D Bilateral Waiting Geometry

Task `t90968` completed the independent bilateral repair and paired validation.
The original arm again reproduced all 23 action records, two collision reports
and 108 detailed observations exactly. The bilateral arm completed all 466
seconds and 108 observations with **zero native collision events/incidents and
zero teleports**. The two V154C mirrored focal vehicles both passed the junction:
straight vehicle `121463_406_0` reached its outgoing edge at 25627, and left-turn
vehicle `168358_425_0` at 25635.

The package preserves V154C's 13→24 split exactly and adds six mirrored geometry
attribute changes, for 12 versus the original network. Waiting lane 3 changes
from 8.62 to 8.37 m, and continuation 20 from 19.58 to 19.83 m. The same
front-at-split passenger-body rule gives +1.530 mm static clearance. At 25630,
the actual mirrored focal left vehicle has **+3.637 cm** clearance, versus
−4.963 cm for that same vehicle in V154C. Its speed is 0.000818 m/s, close to
rest but above the frozen stationary threshold. The original repaired side
also retains positive measured clearance, with a minimum of +3.653 cm in its
reconstructible retained observations.

**The overall frozen result remains FAIL solely because the original straight
focal vehicle has not passed by 25666.** The original left turn passed at
25633. At the last detailed observation (25665), the straight vehicle is still
on the incoming edge, 10.771 m before its green signal, travelling at 5.575 m/s.
The absence of its original collision does not recreate that encounter. This
result therefore supports the mirrored repair and zero collisions within the
bounded run while retaining the unmet four-vehicle passage criterion.

Arrivals in the retained original, V154C and V154D windows are 256, 247 and 241,
respectively, with 322 departures and one pending vehicle in each. There is
no service-improvement claim from this window. Controller feedback differs
from the first retained action at 25260 under unchanged parameters. Runtime
source remains `85b2b43cd12832c5a1dd`; tooling is `8b6203057536f22e5707`.
The occupancy correction and source selection remain outside this comparison.

The original full rollout's 18 incidents have only 20 retained detailed
reports, covering 10 incidents. Nine of those involve the two repaired
movement pairs. The remaining retained incident at 26701 involves moving
straight lane 16_0 and left-turn continuation 22_0; it is outside this repair.
The other eight incidents cannot be classified from the capped sample alone.
The subsequent V154E original-duration run, reported below, supplies a complete
incident inventory and focal passage accounting while preserving this
short-window FAIL and geometry.

Three geometry tests and 13 validation tests passed. The paired runner took
3.499 seconds. Only 13,240 bytes of geometry report and 800,684 bytes of result
JSON were retrieved (813,924 bytes total); network, route and model remain on
the server. See [the V154D report](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154d_cologne_bilateral_waiting_geometry_result.md)
and [compact results](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154d_cologne_bilateral_waiting_geometry_v1.json).

## 2026-09-09 V154E Original-Duration Validation: PASS

Task `t90988` completed the prespecified 3600-second comparison on the unchanged
V154D bilateral network. **The full-duration validation passes: zero native
collisions, zero teleports, and all four focal vehicles passed the controlled
junction.** The original-network prerequisite exactly reproduced all 77
metrics, originator diagnostics, 175 intervention records and 20 retained
collision reports from `t90756`. The complete event inventory agrees with
the native event, incident and collision-step totals in both arms.

The new full run also exactly reproduced V154D's first 466 seconds: 23 action
records, 108 detailed observations and all four passage prefixes. The previously
late straight vehicle entered its internal lane at 25667 and reached its
outgoing edge at 25681. This supplies its full passage evidence. V154D remains
FAIL at its original 25666 endpoint; V154E is a separately frozen original-
duration PASS.

| Full 3600-second quantity | Original network | Unchanged bilateral repair |
|---|---:|---:|
| Native collision incidents / events | 18 / 36 | **0 / 0** |
| Teleports | 0 | 0 |
| Due demand | 2015 | 2015 |
| Departed | 2015 | 2014 |
| Arrived | 1997 | 1993 |
| Active at endpoint | 18 | 21 |
| Pending insertion at endpoint | 0 | 1 |
| Mean tripinfo waiting (s) | 19.712655 | 20.962264 |
| System vehicle hours | 41.079722 | 41.150833 |

The observed tradeoff is four fewer arrivals (0.20% of due demand), 1.250 s
more mean waiting (+6.34%), and 0.173% more system vehicle hours. The full-run
collision result improves substantially with a small system-level service
cost in this seed. Waiting retains the departed-vehicle interpretation,
including unfinished tripinfo; system hours include pending insertion after
warmup. The one pending and 21 active vehicles account for the 22 vehicles
not yet arrived at the endpoint.

The uncapped original-arm inventory now classifies all 18 incidents: 15 at
13_0/1_1, two at 3_0/11_1, and one at 16_0/22_0. No participant pose is missing
from the first report of these incidents. The repaired arm has no events or
incidents. The unmodified 16/22 pair also does not collide in this new
trajectory; that absence does not establish that its separate geometry was
directly corrected.

The same original controller snapshot `85b2b43cd12832c5a1dd` and model were used;
tooling is `95293dee78371b53157d`. Sixteen focused tests passed. The paired
runner took 10.844 seconds, and only the 756,788-byte result JSON was retrieved.
Network, demand, model and scratch tripinfo stayed on the server.

V154F subsequently retained this fixed bilateral network for both rigid and
PhasePressure on the existing three-seed roster, as reported below. V154E's
own PASS establishes its observed Cologne scenario and seed. See
[the V154E result](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154e_cologne_bilateral_full_validation_result.md)
and [compact results](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154e_cologne_bilateral_full_validation_v1.json).

## 2026-09-09 V154F Fixed-Network Controller Comparison

All six cells completed valid 3600-second runs on the unchanged bilateral
network, using the original V150L seeds **52282, 41242, 63792**. Tasks
`t91006`–`t91010` supplied five new simulations; rigid/41242 reuses the repair
arm of V154E `t90988`. Every cell preserves the 2015-vehicle population,
complete collision accounting and zero teleports.

| Seed | Rigid / PP incidents | Rigid / PP waiting (s) | Rigid / PP system vehicle hours | Rigid / PP arrivals |
|---|---:|---:|---:|---:|
| 52282 | 0 / 1 | 19.660546 / 12.472953 | 38.963333 / 31.328333 | 1995 / 1997 |
| 41242 (geometry development) | 0 / 0 | 20.962264 / 14.357498 | 41.150833 / 33.409444 | 1993 / 1996 |
| 63792 | 0 / 0 | 19.752854 / 13.231762 | 41.457778 / 32.535556 | 1996 / 1996 |

Rigid passes the zero-collision outcome in all three seeds. PhasePressure
retains a safety FAIL in 52282: one native junction incident, reported at
27481 and 27482. This completed, colliding cell remains in the service means.
The straight collider is on internal lane 16_1; the other vehicle is already
on outgoing lane `32324544#0_1`. Its recorded route and unique static connection
identify a U-turn via 9→23. Neither is a repaired waiting movement. The
recorded speeds are 9.606 and 11.989 m/s during yellow; the rear of the latter
vehicle is still on its preceding lane, so the current body reconstruction is
incomplete. These records identify the next diagnostic movement pair without
establishing its collision cause.

The equal-seed rigid-minus-PP waiting difference is **+6.771150 s (+50.70%)**;
system vehicle hours differ by **+8.099537 (+24.98%)**. Both differences are
positive in every seed. The prespecified additional-seed summary (52282 and
63792) also shows +6.854342 s waiting (+53.33%) and +8.278611 system vehicle
hours (+25.93%). The outcome therefore does not depend on the geometry-
development seed. Mean arrivals differ by only −1.67 vehicles, and departed
and final pending counts match within every pair. The cumulative system-hour
gap includes +5.177870 active and +2.921667 pending vehicle hours: close final
throughput does not remove the observed delay throughout the run.

Twenty-two focused tests passed before submission. Five new runner times sum
to 21.938 seconds, excluding scheduling, transfer and the reused simulation.
Only five result JSON files, **697,623 bytes** total, were retrieved. Controller
source is `85b2b43cd12832c5a1dd`; tooling is `8c7842518a4d53225905`.

### Limitations

This comparison supports zero observed collisions for the fixed rigid model
on these three Cologne trajectories and consistently faster PhasePressure
service. It establishes neither universal rigid safety nor overall superiority
of either controller: PhasePressure still has a retained collision. One city,
one demand file and one fitted model do not test repeated training, other-city
robustness, new occupancy formulas or source-transfer efficacy. V154D's
short-window FAIL, V154E's one-seed PASS and the new PhasePressure FAIL remain
separate results. The remaining collision calls for a bounded replay of the
straight/U-turn merge before another geometry change.

See [the V154F result](/home/erzhu419/mine_code/CFCMT/paper/tsc_v154f_cologne_repaired_controller_comparison_result.md)
and [the complete comparison](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154f_cologne_repaired_controller_comparison_v1.json).

## 2026-09-09 V154G Remaining Collision Replay

`t91112` exactly reproduced the V154F PP/52282 incident and retained 138 observations over 27440–27485. At 27480, changing G/g to y/y with unchanged vehicle positions and speeds removes the straight vehicle from the U-turn's leader query. The U-turn then accelerates, crosses short continuation 23, and collides while its rear remains on lane 9. This matches SUMO 1.22's [both-yellow speed-based response](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSVehicle.cpp#L7539). Static waiting-point clearance is positive; the next phase has not opened. The original safety FAIL remains.

The local trigger is `_clearance_yellow_state`: both G and g become y. Next: a bounded counterfactual using **G→Y, g→y** for this transition, preserving the priority distinction supported by [SUMO's link states](https://github.com/eclipse-sumo/sumo/blob/v1_22_0/src/microsim/MSLink.h), with network, timing and pre-transition trajectory fixed. Collision removal remains to be tested. Eight tests passed; one replay took 3.050 s and returned 324,465 bytes. [Compact evidence](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154g_cologne_uturn_collision_replay_v1.json).

## 2026-09-09 V154H Yellow-Priority Counterfactual

`t91152` changed only the 27480 yellow transition to **G→Y, g→y**, with the same network and executor timing: the retained PP/52282 collision fell from one incident to zero, and the straight vehicle and U-turn passed at 27482 and 27483. The U-turn immediately recovered the straight vehicle as its leader and braked instead of accelerating; all 122 pre-transition observations and the original pre-intervention sample match exactly.

The local verdict is PASS after correcting a tuple/list comparison error; the raw runner FAIL and original safety FAIL remain intact. Seven focused tests passed; one simulation took 2.959 s and only 316,291 bytes of JSON were retrieved. This establishes the cause and remedy for this event; next is the shared executor correction followed by the fixed-network, three-seed, 3600-second rigid/PhasePressure comparison. [Compact evidence](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154h_cologne_yellow_priority_counterfactual_v1.json).

## 2026-09-09 V154I Shared Yellow-Priority Full Comparison

The public executor now retains **G→Y, g→y**. Six new runs (`t91635`–`t91640`) completed 3600 seconds on the fixed bilateral network with seeds 52282, 41242 and 63792; both rigid and PhasePressure have **3/3 zero-collision runs**, zero teleports and complete population accounting. Only this executor change was loaded over the frozen original runtime/model/occupancy equations. Thirty focused tests passed; simulation runtime summed to 33.682 s, with 943,214 bytes of result JSON retrieved.

Rigid versus PhasePressure mean waiting is **21.074 vs 14.037 s (+50.13%)**, system vehicle hours **40.555 vs 33.222 (+22.07%)**, and arrivals **1997.33 vs 1997.67**. Both delay differences remain positive in every seed; the additional-two-seed summary agrees. The original PP collision is removed, while the old results remain intact. This one-city comparison does not resolve rigid service quality: next is fresh candidate-label collection and target-only refitting under the corrected occupancy-fraction contract, with the original training budget/horizon and these fixed-network controls. [Compact results and per-cell before/after changes](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154i_cologne_priority_yellow_comparison_v1.json).

## 2026-09-09 V154J Fraction-Contract Recollection

The original B100 group roster was recollected under the fraction equations and corrected executor (`t91647`–`t91655`): **80/100 groups retained**, with 18 entire groups censored for unsafe branches and two unavailable at their requested opportunity. The original-roster attempt is FAIL and no B80 model was substituted. Cologne1 retained 33/34 groups with zero behavior collisions; Cologne3 and Cologne8 retained 16/33 and 31/33, with 41 and 53 behavior collision incidents respectively. The nine small results total 170,928 bytes; collection runtime sums to 194.160 s. [Failure evidence](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154j_cologne_fraction_refit_v1.json).

The user selected a separate **Cologne1-local B100** continuation. Its frozen pool requests 204 new groups across the original three training seeds and reuses 33 safe local groups; it selects 34/33/33 groups by time coverage, preserving the original model settings and 450-second labels. This changes training coverage from the old Cologne1 groups, all at 3110–3590 seconds. The original mixed-scenario FAIL remains intact; this local comparison will not isolate an occupancy-unit-only effect.

The local continuation is complete: collection `t91688`–`t91690` produced 162 additional retained groups, with nine unsafe groups censored and 33 unavailable opportunities. Together with the 33 reused groups, the pool contains 195 groups; `t91695` fitted the frozen model on exactly **100 groups / 800 rows**. Three 3600-second evaluations (`t91696`–`t91698`) have zero collisions and teleports. Mean waiting is **23.698 s**, versus **21.074 s** for the old rigid model (+12.45%) and **14.037 s** for reused PhasePressure (+68.82%). Waiting worsens in every seed. System vehicle hours are **40.246 / 40.555 / 33.222**, respectively; the slight mean reduction from old rigid is not consistent across seeds. Mean arrivals are **1993.00 / 1997.33 / 1997.67**.

This refit did not improve service. Thirty-seven focused tests passed. All V154J collection, fit and evaluation runner times sum to 513.009 s, including the original failed attempt; 16 result JSON files total 1,261,508 bytes, with banks and models left on the server. Next: use the 95 retained groups excluded from fitting to diagnose action-ranking regret against their existing 450-second labels before changing the model or objective. These same-training-seed groups provide a diagnostic, not independent generalization evidence. The combined unit/training-coverage change and nine censored branch collisions limit attribution and safety claims. [Compact comparison, accounting and limitations](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154j_cologne_local_fraction_refit_v2.json).

## 2026-09-09 V154K Unused-Group Ranking Diagnosis

`t91723` scored the existing 95 unused groups and 100 training groups with the frozen model and deployment tie-break; no refit or simulation was needed. On unused groups, 67 predicted improvements over PhasePressure yield **30 beneficial / 33 harmful / 4 equal-cost** actions, versus **53 / 6 / 1** among 60 training-group overrides. Mean 450-second halted-queue cost improves only **1.17%** on unused groups, versus **9.15%** in training; two of the three unused seed subsets worsen. Rigid selects a label-optimal action in **19/95** unused groups versus **67/100** training groups, and mean regret rises from **0.01585 to 0.08522** halted vehicles per controlled lane. This identifies an offline training-to-unused ranking gap; further closed-loop mismatch remains unmeasured.

Next: keep B100, the model family and labels fixed, use training-seed cross-fitting to test whether predicted advantage can support a useful minimum-advantage threshold over PhasePressure, and reject thresholding if out-of-fold label costs do not improve. The inspected 95 groups remain development diagnostics. Five focused tests passed; runtime was 1.694 s and only 144,015 bytes of result JSON were retrieved. [Results and next-step specification](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154k_cologne_local_rigid_ranking_v1.json).

## 2026-09-09 V154L OOF Threshold Calibration

`t91733` fitted the unchanged model family on 66/67/67 groups, scoring the held-out 34/33/33 groups from the original B100 roster. Only these OOF predictions selected the normalized-advantage threshold **0.3580414874633295**; the threshold was saved before reading the 95 development-group outcomes. Equal-seed queue-label cost changes from **0.772044 to 0.757354**, versus PP **0.767379** (+0.61% to −1.31%). Applying the frozen threshold to the existing full-B100 model's development predictions changes **0.774401 to 0.771224**, versus PP **0.783935** (−1.22% to −1.62%). Development overrides fall from 67 to 34: harmful actions decrease from 33 to 15, but beneficial actions also decrease from 30 to 17.

This modest offline gain warrants a fixed-threshold, same-network three-seed closed-loop comparison; it does not establish closed-loop improvement. OOF minima are calibration results, and the previously inspected 95 groups remain development diagnostics; one development seed is still slightly worse than PP. Keep the exact threshold without retuning for seeds 52282/41242/63792 over 3600 seconds. Three fold fits used 100 unique groups and 200 summed training-group appearances; no new labels or simulations were collected. Six focused tests passed, runner time was 3.316 s, and two JSON files totalled 265,337 retrieved bytes. [Calibration and frozen next step](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154l_cologne_rigid_oof_threshold_v1.json).

## 2026-09-09 V154M Frozen-Threshold Closed Loop

All three new 3600-second runs (`t91746`–`t91748`) completed with zero collisions and teleports using the unchanged B100 model, core source, network and V154L threshold. Mean waiting falls from **23.698 to 15.702 s (−33.74%)**, versus PP **14.037 s (+11.86%)**; system vehicle hours fall from **40.246 to 33.868 (−15.85%)**, versus PP **33.222 (+1.94%)**. Both metrics improve over ungated rigid in every seed. Waiting remains above PP in all three seeds; system vehicle hours beat PP only in 63792. Mean arrivals are **1998.33 / 1993.00 / 1997.67** for thresholded rigid / ungated rigid / PP. All new runs insert all 2015 vehicles with zero final pending insertions.

On the new trajectories, the threshold accepts 207 of 653 rigid proposals and rejects 446. This establishes improvement within this fixed comparison, not superiority over PP or broader safety. Next: retain the threshold and replay seed52282's highest-score accepted action at **t=25800** (advantage 0.789751), comparing its first 10 seconds against the same-state PP reference with a common PP continuation to t=26250. The existing trace identifies the decision but does not establish that it is harmful. Twenty-three focused tests passed; the three runner times sum to 13.157 s and only 393,618 bytes of result JSON were retrieved. [Full comparison and bounded next diagnostic](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154m_cologne_threshold_closed_loop_v1.json).

## 2026-09-09 V154N Accepted-Action Counterfactual

`t91776` replayed the unchanged seed52282 prefix directly to t=25800. With the original 450-second objective and common PP continuation, mean halted vehicles per controlled lane are **0.843611 for rigid versus 1.020833 for PP (−17.36%)**. Both branches are collision/teleport-free. All 49 prefix decisions and physical starting states match the reference; rigid's first ten costs match the original trajectory exactly, and its repeated 460-value outcome vector is identical.

The initial snapshot-based attempt, `t91766`, remains **INVALID**: restoration changes the first-step halted count from 19 to 14 despite matching recorded positions, speeds and executor state. Its diagnostic costs, 1.046944 versus 0.713333, reverse the native ordering. Next: freeze nine original B100 groups (early/middle/late within each training seed), reproduce their original collection prefixes, and compare all eight native action labels against the saved labels before refitting or changing the threshold. Twelve focused tests passed; both runner times total 5.373 s, and only two JSON files totalling **135,293 bytes** were retrieved. [Compact evidence](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154n_cologne_accepted_action_counterfactual_v2.json).

Limitations: this one beneficial action does not explain the remaining closed-loop waiting gap. The specific cause of restoration divergence and its effect on existing B100 labels remain unmeasured.

## 2026-09-09 V154O B100 Training-Label Audit

Nine original B100 groups were frozen by training seed and early/middle/late position. All **72 native branches** match the stored features, contexts and candidate phases, share identical within-group starting states, and complete 450 seconds without collisions or teleports. The unchanged original collector separately reproduces all 72 saved labels with **zero numerical difference**. Nevertheless, native outcomes change the optimal action set in **8/9 groups** and strictly reverse the advantage over PP for **23/63 non-reference actions**. Cost MAE is **0.063403 halted vehicles/lane**, with maximum error 0.381944. Selecting the old label optimum incurs native regret 0.077130 on average; the middle-period mean is 0.161944, versus 0.000185 late.

The initial shortened-history restore controls fail to reproduce six old PP labels and remain INVALID (`t91785`–`t91787`); only their fully validated native branches are reused. Original-collector replays (`t91805`–`t91807`) resolve the label binding. Ten focused tests passed; runner times total 106.429 s, and six small JSON files total **174,017 bytes**. Next: replace restore-based label generation with native-prefix branching, reuse these nine groups, and recollect the remaining 91 original B100 groups before refitting or recalibrating. Keep group IDs fixed and retain any unsafe group as a failure. [Compact evidence and all nine comparisons](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154o_cologne_training_label_restore_audit_v1.json).

Limitations: nine selected groups do not estimate the affected fraction of the whole B100. Two late optimal-action changes have native regret only 0.000278. The specific hidden SUMO state behind the divergence, and its contribution to the closed-loop waiting gap, remain unresolved.

## 2026-09-09 V154P Native B100 Recollection and Refit

The unchanged **100 groups / 800 actions** are complete: `t91825`–`t91827` collected 728 fresh native branches and reused 72 validated V154O branches, all without collisions or teleports. No group was replaced. Native labels change the optimal action set in **59/100 groups** and strictly reverse PP-relative advantage for **264/700** non-reference actions; cost MAE is 0.079422 halted vehicles/lane.

`t91845` fitted the original model family on B100 and the three fixed seed folds. The unchanged OOF selection rule chooses **threshold 0**: equal-seed cost is **0.777294 versus PP 0.802919 (−3.19%)**, with 47 beneficial / 25 harmful / 6 equal-cost overrides. Applying the old 0.358041 threshold to these new OOF predictions gives 0.792273. Seed2027 still worsens. Twelve focused tests passed; new runner times total 899.470 s, and only five JSON files (**57,439 bytes**) were retrieved.

Next: freeze the new model and threshold 0, then run the same three 3600-second seeds against V154M's original thresholded model and V154I's PP controls. Limitations: this is calibration performance; the new model has no closed-loop result yet. [Compact results and fixed next comparison](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154p_cologne_native_b100_v1.json).

## 2026-09-09 V154Q Native-Label Closed Loop

Three fixed 3600-second runs (`t91851`–`t91853`) complete with zero collisions/teleports, but **service deteriorates in every seed**. New / original-thresholded / PP mean waiting is **274.701 / 15.702 / 14.037 s**, system vehicle hours **295.889 / 33.868 / 33.222**, and arrivals **1012 / 1998.33 / 1997.67**. New pending insertions average **802**; system vehicle hours include this unserved population. The additional-two-seed comparison agrees.

Of 888 accepted overrides, **808 keep the current phase despite PP requesting a switch**. Actual phase switches fall from 756 to 208 across three runs. Seed52282 enters a persistent phase at **t=26920**; the retained trace later records 1425 seconds of continuous green. Next: preserve this native prefix/model/threshold and compare three 450-second branches—one model action then PP, PP throughout, and repeated model actions—to separate initial misranking from continuation effects. Twenty-one focused tests passed; only three result JSON files were retrieved. [Results, accounting and selected replay](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154q_cologne_native_label_closed_loop_v1.json).

Limitations: model and threshold changed together; their separate effects remain unresolved. Each intervention trace is capped at256 records; full-run counts above use uncapped audits. Waiting excludes pending insertions, and this remains one-city development evidence.

## 2026-09-09 V154R Entry Action and Continuation Replay

`t91864` reproduced the seed52282 state at t=26920 through three fresh native prefixes. All branches share the same physical state and 156 prefix decisions; repeated model execution exactly matches Q's 157 recorded interventions through t=27370. All three branches complete 450 seconds without collisions or teleports. Halted vehicles/lane cost is **0.703333 for one model action then PP, 0.621111 for PP, and 9.978333 for repeated model actions**. The entry action is already harmful (**+13.24%**); 44 subsequent stay overrides increase cost by another 9.275. Arrivals are **240 / 240 / 27**, with final pending **0 / 0 / 105**.

Next: keep the model and threshold 0, reject only accepted model holds that override a PP switch while the executor is green, and run the same three seeds for 3600 seconds. This tests the prolonged holding mechanism without correcting the entry action's ranking. Five focused tests passed; runtime was 7.715 s and one 9,504-byte JSON was retrieved. Limitation: this selected window does not establish that stays explain all losses over the full run. [Evidence and frozen ablation](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154r_cologne_continuation_counterfactual_v1.json).

## 2026-09-09 V154S Model Stay Ablation

`t91871`–`t91873` changed only the accepted model holds that veto PP switching, retaining the native B100 model and threshold 0. All three 3600-second runs are valid and collision/teleport-free. Executed stay overrides fall from **808 to 0**; 441 proposals are vetoed, 251 remain accepted, and longest continuous green is **20 / 26 / 20 s**. Mean waiting falls from **274.701 to 17.108 s** and arrivals recover from **1012 to 1993.33**, resolving the observed prolonged-green service collapse.

Service still trails PP: waiting **17.108 vs 14.037 s (+21.87%)**, system vehicle hours **37.700 vs 33.222 (+13.48%)**; original V154M achieves 15.702 s and 33.868 hours. Both remaining gaps hold in every seed. Next: preserve the model and stay veto, apply that veto to the existing 100 training-seed OOF predictions, and recalibrate the remaining switch threshold using the original selection rule, including PP-only. No evaluation outcomes enter calibration. Twenty-nine focused tests passed; runtime totals 12.533 s and three JSON files total 453,532 retrieved bytes. [Comparison and fixed next step](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154s_cologne_no_stay_closed_loop_v1.json).

Limitations: this ablation was chosen after observing these development trajectories. It removes a major failure mechanism without correcting the remaining model switch-ranking errors.

## 2026-09-10 V154T No-Stay OOF Recalibration

`t91888` reused the original 100 groups / 800 native labels and cached seed-OOF scores, preserving the model and S stay veto. The original selection rule chooses **0.2079045043** from 53 candidates, including PP-only. Equal-seed cost is **0.799072**, versus **0.799800** at no-stay threshold 0 and **0.802919** for PP: improvements of only **0.09% / 0.48%**. It accepts 44 switch overrides (23 beneficial / 15 harmful / 6 equal); 13 proposals fail the threshold and 21 further proposals are vetoed. All 100 groups remain in the cost denominator.

Next: freeze this threshold and run the same three 3600-second seeds against S and PP. Limitations: this is calibration performance, seed2027 still worsens versus PP, and continuous deployment remains untested. Twenty-eight focused tests passed; no fits, labels or simulations were added, and only 10,076 JSON bytes were retrieved. [Compact result and next comparison](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154t_cologne_no_stay_oof_threshold_v1.json).

## 2026-09-10 V154U Recalibrated No-Stay Closed Loop

`t91893`–`t91895` complete the three fixed 3600-second runs with zero collisions/teleports. New / threshold-zero / PP mean waiting is **16.578 / 17.108 / 14.037 s**, system vehicle hours **36.745 / 37.700 / 33.222**, and arrivals **1991.67 / 1993.33 / 1997.67**. Waiting improves **3.10%** over threshold zero but remains **18.10%** above PP; seed63792 worsens, and all three seeds still trail PP on waiting and vehicle hours. Counts are 209 accepted switches, 261 threshold rejections and 264 stay vetoes; executed model stays remain zero, with maximum green **30 / 35 / 26 s**.

Next: keep the settings fixed and replay seed63792's highest-score accepted switch at **t=27360** using identical native prefixes: one model action then PP versus PP throughout for 450 seconds. Thirty-two focused tests passed; only three JSON files (398,214 bytes) were retrieved. Limitations: the comparison and selected replay are development evidence; the selected action's true advantage is not yet known. [Results and bounded next replay](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154u_cologne_recalibrated_no_stay_v1.json).

## 2026-09-10 V154V Accepted Switch Native Replay

`t91928` reproduces seed63792 at t=27360 with identical physical/executor states, pending vehicles and 173 prefix decisions. Both 450-second branches pass trajectory, population and safety checks. **One model action then PP costs 0.910833, versus 1.274444 for PP throughout (−28.53%)**, with arrivals **206 / 196**. This selected switch is beneficial relative to PP under the native label; it does not explain the full-run service gap. Five tests passed; one 8,955-byte JSON was retrieved.

Next: reuse these branches and add one unchanged-U continuation through t=27810, matching its first 10 costs to the model-once branch, to measure subsequent model interventions. Limitation: this selected development window does not establish how often model actions help or harm. PP denotes the untrained PhasePressure rule baseline; the threshold-zero and new-threshold columns are our learned-controller variants. [Result and fixed continuation comparison](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154v_cologne_no_stay_switch_counterfactual_v1.json).

## 2026-09-10 V154W Continued Model Replay

`t91939` adds one continuous-model branch, reusing both V branches. Native starting state, 173 prefix decisions, the target decision and first 10 costs match exactly; all 37 accepted records match U through the window. Costs for **one model action then PP / PP throughout / continued model** are **0.910833 / 1.274444 / 1.190556**, with arrivals **206 / 196 / 201**. The 11 subsequent model switches jointly erase **76.93%** of the initial gain, although continued control still beats same-state PP by **6.58%**. All branches are collision/teleport-free; the new window has zero model stays and longest recorded green 19 seconds. Four tests passed; one 12,354-byte JSON was retrieved.

Next: preserve the model, threshold and stay veto; test one accepted 10-second model action followed by 440 seconds of PP before another override, using the original 450-second label horizon without tuning and the three fixed 3600-second seeds. Limitation: this selected window identifies a joint continuation loss, not the contribution of each later switch or overall policy superiority. [Three-branch result and fixed next comparison](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154w_cologne_no_stay_continuation_replay_v1.json).

## 2026-09-10 V154X Spaced Model Deployment

`t91944`–`t91946` complete the fixed three 3600-second runs. Each executes eight model switches at least450 seconds apart. On the original scalar queue objective, spaced / continuous-U / PP means are **7.282877 / 8.784430 / 7.233644 stopped vehicles** (divide by the shared eight lanes for per-lane costs): spacing improves17.09% versus U but remains0.68% above PP. Waiting is **13.918 / 16.578 / 14.037 s**; these supporting service measures do not define additional objectives.

One same-lane collision incident (13 reports) remains recorded for seed41242. All runs use `collision_action=warn`, have zero teleports and remain in the comparison; zero collisions are not an efficiency admission criterion. Next: freeze X and compare all **21 accepted interventions with complete450-second windows** against same-native-state PP, using the existing scalar cost. The three incomplete end windows are excluded by one time-boundary rule. Thirty-five tests passed; three JSON files total201,760 retrieved bytes. Limitation: this development roster has not established consistent gains over PP. [Comparison and fixed action-analysis roster](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154x_cologne_spaced_no_stay_v1.json).

## 2026-09-10 V154Y Complete Spaced-Action Replay

`t91997`–`t91999` complete all **21 native-state pairs / 42 branches**, preserving X. All starting states, prefixes and450-second samples match their contracts. **13 actions help and8 harm**; model / same-state PP mean queue cost is **0.889167 / 0.875423 halted vehicles/lane (+1.57%)**. Per-seed cost changes are **−1.64%, +6.00%, −0.24%**. Four harmful actions initially help during the first150 seconds, so early gains do not settle the450-second ranking.

Model scores use group normalization; direct score-versus-raw-cost bias and MAE are invalid and were removed. Next: reuse B100's cached OOF predictions to audit the44 T/S-accepted actions, using their full eight-action normalization and raw-cost harm contributions. Sixteen focused tests passed; three result JSON files total **262,662 bytes**, with97.969 seconds summed runner time. Limitation: these local PP branches share X prefixes and do not represent a new full-run comparison. [All21 pairs and next audit](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154y_cologne_spaced_action_counterfactual_v1.json).

## 2026-09-10 V154Z OOF Target-Scale Audit

`t92016` exactly reproduces T's44 accepted actions: **23 beneficial /15 harmful /6 ties**, retaining all100 policy-cost groups. In the correct full-eight-action normalized units, mean predicted /actual gain is **0.506855 /0.118297**, bias **+0.388558**, MAE **0.589389**, and correlation **−0.03282**. Ranking errors therefore already occur in training-seed OOF selections. The largest raw loss contributes **23.13%** of raw harm but **12.31%** after normalization.

Next: retain B100, features, HGB settings and each training fold's original sample weights; change only the fitted target to raw cost difference (C=1), then apply the original OOF threshold-selection rule. No new labels or simulations are needed. Ten tests passed; one **38,034-byte** result was retrieved. Limitation: selected OOF residuals and harm shares motivate this ablation but do not establish normalization as the cause. [44-action audit and fixed next ablation](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v154z_cologne_oof_normalization_audit_v1.json).

## 2026-09-10 V155A Raw-Target B100 Refit

`t92032` changes only the fitted target to raw cost difference, preserving B100, features, HGB settings and original fold-specific sample weights. Three OOF fits plus one full fit select **threshold0** from50 candidates. New /old-T /PP equal-seed costs are **0.781472 /0.799072 /0.802919**: **−2.20%** versus T and **−2.67%** versus PP. All three training seeds improve versus T;48 accepted actions comprise31 beneficial,15 harmful and2 ties.

Next: freeze the new model and threshold0, retain S stay veto and X450-second spacing, and run the three original3600-second seeds against X and PP. Fifteen focused tests passed; no labels or simulations were added, and two JSON files total **68,590 bytes**. Limitations: these are OOF calibration results; seed2027 still trails PP, and original weights still depend on normalization. [Result and fixed next comparison](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155a_cologne_raw_target_refit_v1.json).

## 2026-09-10 V155B Raw-Target Closed Loop

`t92044`–`t92046` complete the three fixed3600-second runs. New /X /PP mean queues are **7.618846 /7.282877 /7.233644**: the raw-target model costs **4.61% more than X and5.33% more than PP**, trailing PP in every seed. Waiting is **14.988 /13.918 /14.037 s**; arrivals are **1996 /1997.67 /1997.67**. Each seed executes eight effective model switches with at least450-second spacing and no model stays. All runs are valid; the new arm has zero collisions/teleports.

Next: freeze B and compare all21 complete450-second intervention windows against same-native-state PP, using the existing scalar cost and raw-unit predictions. Fourteen focused tests passed; three JSON files total **152,482 bytes**, with11.517 seconds summed simulation runtime. Limitations: OOF improvement did not transfer to this development roster; full-run differences alone do not identify individual action errors. [Results and fixed21-window replay](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155b_cologne_raw_target_closed_loop_v1.json).

## 2026-09-10 V155C Raw-Target Native Action Replay

`t92067`–`t92069` complete all21 pairs/42 branches with exact native starts and prefixes. **Five actions help and16 harm**; model/PP cost is **0.922500/0.863743 (+6.80%)**. In matching raw units, mean predicted/actual gain is **+0.081864/−0.058757**, bias **+0.140621**, MAE **0.161485**, with18 overpredictions. Early/middle/late150-second mean gains are all negative. Two first interventions also harm, so prior model-induced state drift cannot explain every error.

Next: retain raw targets, B100, features, HGB and OOF folds; test base group-balanced sample weights, removing A's inherited normalized-magnitude/sign weighting. Select the threshold from training OOF only. Twenty-three tests passed; three result JSONs total **152,352 bytes**, with86.812 seconds summed runner time. Limitations: these local development replays do not decompose the full-run gap or establish sample weights as its cause. [Results and fixed training-weight ablation](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155c_cologne_raw_target_action_counterfactual_v1.json).

## 2026-09-10 V155D Base-Weight Raw Refit

`t92085` completes three OOF fits and one full fit with equal candidate weights, preserving raw targets and HGB settings. Threshold remains **0**. D/A/PP equal-seed costs are **0.787601/0.781472/0.802919**: D worsens **0.78%** versus A despite remaining1.91% below PP. Its47 accepted actions comprise29 beneficial,16 harmful and2 ties; seed2027 accounts for most of the deterioration.

Next: retain A and inspect its48 accepted OOF actions against the nearest three distinct training groups in the original29-feature space, using only each fold's training rows for scaling and neighbors. Twenty-one tests passed; no labels or simulations were added, and two JSON files total **66,183 bytes**. Limitations: these are calibration results; the next diagnosis concerns original training OOF states and does not establish coverage of C's deployment states. [Result and fixed neighborhood diagnosis](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155d_cologne_group_balanced_raw_refit_v1.json).


## 2026-09-10 V155E Training-Neighborhood Diagnosis

`t92094` audits all48 A-accepted OOF actions against three distinct training-group neighbors. All15 harmful actions remain inside every training-fold marginal feature range;13 have positive neighbor-mean gains, including six with three positive neighbors. Seed2027 contributes11 harms,10 with positive neighbor means. Beneficial/harmful nearest-distance medians are1.727/1.876 with overlapping ranges;35/48 neighborhoods contain mixed gain signs.

Next: retain all48 queries and144 neighbor pairs; inspect omitted phase age and green/red lane-queue maxima and variation, using existing bank columns without reranking neighbors. Fourteen tests passed; zero fits, labels or simulations were added. Only55,647 JSON bytes were retrieved; runner time was1.428 seconds. Limitations: marginal range membership does not prove joint support, and these results neither establish a cause nor diagnose C deployment-state coverage. [Result and fixed next diagnosis](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155e_cologne_oof_neighborhood_v1.json).


## 2026-09-10 V155F Omitted-Feature Pair Audit

`t92103` inspects the fixed48 queries/144 pairs without reranking. Label signs agree in59 pairs, disagree in69, and involve ties in16. The15 harmful queries supply45 pairs:33 disagree and10 agree. All13 omitted-feature standardized-gap medians are smaller for those33 disagreements than for the10 agreements; green-queue dispersion is0.419 versus0.990. Overall and per-seed directions vary, providing no consistent separation by omitted-feature gaps.

Next: test the complete predefined13-column block (29→42 features) with only three OOF fits, preserving B100, raw targets, A weights/HGB, threshold0 and the stay veto; compare all100 groups against A/PP before full fitting. Thirteen tests passed; zero fits, labels or simulations were added. Retrieved JSON totals96,851 bytes; runtime1.462 seconds. Limitations: pairs are dependent, phase age is capped, and this diagnosis establishes neither cause nor deployment coverage. [Result and fixed feature ablation](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155f_cologne_omitted_feature_audit_v1.json).


## 2026-09-10 V155G Fixed Feature-Block OOF Ablation

`t92141` fits only the three original seed folds, expanding29→42 columns with exact A sample weights and threshold0. G/A/PP equal-seed costs are **0.799818/0.781472/0.802919**: G worsens **2.35%** versus A, with all three seeds worse. Its44 accepted actions comprise24 beneficial/17 harmful/3 ties. Of33 changed actions,10 lower cost,20 raise it and3 preserve it.

Retain A; do not fully fit or deploy G. Next: reuse all100 original labels and A decisions to decompose loss against the S-allowed label optimum into unnecessary PP overrides, suboptimal switches and missed beneficial switches. Sixteen tests passed; three OOF fits, zero full fits, labels or simulations were added. One61,666-byte JSON was retrieved; runtime0.981 seconds. Limitations: this rejects the fixed block under these settings; it establishes neither universal feature uselessness nor closed-loop superiority. [Result and fixed loss decomposition](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155g_cologne_feature_block_oof_v1.json).


## 2026-09-10 V155H S-Allowed Policy Loss Decomposition

`t92154` retains all 100 groups. PP/A/S-allowed label-optimum costs are **0.802919/0.781472/0.704819**; A realizes **21.86%** of that label-based improvement potential. Remaining regret comprises 9 unnecessary overrides (**6.59%**), 28 suboptimal switches (**31.50%**) and 37 missed beneficial switches (**61.91%**); 26 groups already achieve the allowed optimum. Of the 37 misses, 23 follow stay vetoes and 14 have PP predicted best; threshold rejection contributes none.

Next: freeze A scores and threshold0; compare filtering forbidden actions before the original argmin against top1-then-veto, using all 100 groups and reporting new harms as well as recovered gains. Seventeen tests passed; zero fits, labels or simulations were added. Two small JSON outputs total **103,380 bytes**. Limitations: the optimum uses future rollout labels, and these training-group bounds do not establish achievable deployment gains. [Result and fixed mask-order comparison](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155h_cologne_policy_loss_decomposition_v1.json).


## 2026-09-10 V155I Pre-Mask Ranking Ablation

`t92167` changes only S mask ordering with cached A scores. I/A/PP equal-seed costs are **0.789812/0.781472/0.802919**: I worsens **1.07%** versus A. The 24 changed actions comprise 10 beneficial and 14 harmful; all 68 non-veto groups remain unchanged. The 23 missed-veto groups recover 0.004172 cost, but six harmful overrides among nine PP-optimal groups lose 0.012511. Seeds 2027/3037 worsen; 4047 improves.

Retain A; do not deploy I. Next: use all 100 cached groups to distinguish switch-versus-PP benefit-sign errors from ordering errors between executable switches, with group/seed weighting and separate tie accounting. Fourteen tests passed; zero fits, labels or simulations were added. One 152,816-byte JSON was retrieved; runtime 1.788 seconds. Limitations: this repeated OOF development comparison does not establish closed-loop performance. [Result and fixed next diagnosis](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155i_cologne_pre_mask_ranking_v1.json).


## 2026-09-10 V155J PP Sign and Switch-Pair Diagnostic

All100 cached groups yield615 switch-versus-PP comparisons and1590 switch pairs. Group/seed-weighted reversal rates are **44.97%/37.12%**; reversed label-gap shares are **42.76%/40.67%**. PP comparisons include146 false benefits and130 missed benefits. Seed2027 primarily overpredicts gains;3037/4047 more often miss gains. These results do not isolate a ranking-only defect.

Next: natively relabel the original95 unused groups (760 new action labels), retain A's29 features/training rule/threshold0/S, and compare three B195 OOF fits against A on the same original100 held-out groups. This tests additional data, with no full fit or closed-loop run planned yet. Fourteen tests passed; J added zero fits, labels, simulations or downloads. Limitations: pair diagnostics are not policy regret; the next experiment increases label budget and adds no new training seeds. [Result and fixed data-size ablation](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155j_cologne_pair_score_diagnostic_v1.json).


## 2026-09-10 V155K Prepared; SSH Staging Blocked

The fixed95-group native collector and B195 three-fold comparison are implemented;23 focused tests passed. All three staging attempts failed, and both configured SSH relays independently timed out before authentication. **No tasks, labels, simulations or model fits were started.**

Next: after connectivity recovers, use the saved submit script with the unchanged protocol; collect760 native labels in six shards, then run the three OOF fits on the common original100 evaluation groups. [Prepared status and executable continuation](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155k_cologne_expanded_data_oof_v1.json).

## 2026-09-10 V155K Expanded-Data OOF Result

Connectivity recovered through the existing cluster entry. Six collection tasks (`t92192`–`t92197`) completed the prespecified95 additional groups /760 native branches; all branches completed and reported zero collisions. `t92203` then fitted three B195 seed-OOF models while excluding the held-out seed from both the original and additional groups and evaluating only the original100 groups. The expanded / original-A / PP equal-seed costs are **0.778955 /0.781472 /0.802919 halted vehicles per controlled lane**: B195 improves **0.32%** over A and **2.98%** over PP. Per-seed changes versus A are **−2.11%, +1.83%, −1.00%** for2027/3037/4047. Accepted overrides fall from48 to26; the expanded model's26 comprise20 beneficial and6 harmful actions, versus31 beneficial,15 harmful and2 ties for A.

The frozen advancement condition is met, but the gain is small and uneven. This is an added-label-budget result (B195 versus B100), the95 groups add time coverage within the same three training seeds, and the original100 comparison has been reused for development. Next: fit exactly one full B195 model, retain the29 features, raw target, A weights, threshold0, post-selection stay veto and450-second spacing, then run the three existing3600-second closed-loop seeds against A and PP without retuning. Collection plus fitting took831.327 seconds in summed runner time; only seven result JSON files totalling133,218 bytes were retrieved. [Completed comparison and accounting](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155k_cologne_expanded_data_oof_v1.json).

## 2026-09-10 V155L B195 Full Fit

`t92229` assembled the frozen original100 plus additional95 native groups and fitted exactly one full model:195 groups,1560 rows and1365 non-reference training rows with the same29 columns, raw action-minus-PP target, A weighting formula and HGB settings. The serialized alpha0 model preserves the shared anchor/correction object and exact scores; all195 PP reference predictions are zero. Threshold0 is carried from K without another calibration. The fit artifact is COMPLETE with all checks true; runtime was0.486 seconds, and only the29,368-byte result plus906-byte threshold JSON were retrieved. [Full-fit result](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/cluster/tsc_v155l_cologne_expanded_data_full_fit_20260910/fit_v1/result.json).

## 2026-09-10 V155M B195 Closed Loop

`t92237`–`t92239` complete the fixed three3600-second runs. B195 / A / PP equal-seed mean queues are **7.402899 /7.618846 /7.233644 stopped vehicles**. B195 improves **2.83%** over A and does so in every seed, so K's added-data benefit transfers to these closed loops. It still costs **2.34%** more than PP; only seed41242 beats PP. Mean waiting is **13.986 /14.988 /14.037 s**, system vehicle hours **33.595 /34.530 /33.222**, and arrivals **1997.33 /1996.00 /1997.67**. The prespecified queue objective therefore remains worse than PP even though descriptive waiting is slightly lower.

All three B195 trajectories are valid and report zero collisions/teleports. They execute24 model switches, exactly8 per seed, with zero model stays and minimum450-second gaps. Thus the remaining queue gap is not caused by a failed stay veto or spacing rule. Thirty-seven combined new/regression tests passed; the three simulation runtimes sum to10.514 seconds and only153,481 bytes of result JSON were retrieved. The scheduler later classified the four jobs as failed because an SSH probe outage hid their exit tokens and their uppercase `COMPLETE` output did not match its configured success markers; the physical result artifacts and scientific validity checks are complete.

Retain B195 as an improvement over A under the larger label budget, but do not claim superiority over PP. The next controlled experiment should use six untouched evaluation seeds, selected before outcomes by adding100000 and200000 to each original evaluation seed, and run fixed A/B195/PP for18 paired3600-second trajectories. This tests whether the data-size benefit and PP gap persist outside the repeatedly inspected development seeds; no parameter is retuned afterward. [Nine-cell result, provenance and accounting](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155m_cologne_expanded_data_closed_loop_v1.json).

## 2026-09-10 V155N Fresh-Seed Fixed-Policy Validation

All18 prespecified cells on seeds152282/141242/163792/252282/241242/263792 are complete. B195 / A / PP equal-seed mean queues are **7.278123 / 7.135931 / 7.205450 stopped vehicles**. B195 is **1.99% worse than A** and **1.01% worse than PP**; it trails A in five of six seeds and splits3/3 against PP. The frozen decisions are therefore both negative: the larger-label data effect does not survive the fresh-seed check, and B195 does not beat PhasePressure. Mean waiting is **13.993 / 13.685 / 13.822 s**, and system vehicle hours are **33.431 / 32.994 / 33.042**; these supporting metrics agree with the primary queue direction.

Every trajectory is valid and reports zero collisions and zero starting/ending teleports. B195 executes46 effective switches and A48, with no model stays, no executor rejection and the required450-second minimum spacing. The result therefore does not point to a failed stay veto, spacing rule or repaired executor. The more plausible limitation is the training roster: the extra95 groups increased time coverage inside the same three training seeds but added no seed diversity, and the apparent benefit on the repeatedly inspected development seeds did not generalize.

PP's first three attempts per seed reached post-simulation validation but wrote no result because the runner read a learned-model guard key from PP's correctly empty audit dictionary. The interface-only correction left the scientific configuration unchanged; the fourth automatic attempts (`t92317`, `t92320`, `t92323`, `t92326`, `t92327`, `t92328`) produced the six effective PP results. The scheduler nevertheless marked all tasks failed because the short-run logs did not match its configured success markers, then exhausted three automatic retries for every submitted signature. Exact remote-log accounting is **18 COMPLETE result writers, 18 post-simulation PP KeyErrors and60 FileExists refusals**, totalling96 scheduler records and36 actual simulator executions. The non-overwrite guard preserved every unique result. The executed entrypoint snapshot is retained, and the reusable local entrypoint now emits the recognized `Eval complete` marker. Forty-three focused/regression tests and the scheduler-marker check pass; only **827,402 bytes** of result JSON were retrieved.

Do not retain B195 as an improvement over A, and keep PP as the current controller benchmark. V155N closes the V155K--N Cologne target-only label-expansion route: it authorizes no further Cologne labels, model fits or closed-loop runs, and supersedes the completed next-step directives in V155J--M. Further method work returns to the source-contribution question under a fixed architecture and target-label budget, comparing source-plus-target directly with target-only before asking whether source information supports a beneficial departure from PP. [Fresh-seed result and task-attempt accounting](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v155n_cologne_expanded_data_fresh_seed_validation_v1.json).

## 2026-09-10 V156A Feature-Aligned Dense Utility Replay

The exact V150O correction replay is complete. Tasks `t92470`--`t92476` used
the original V150O snapshot with only stored-feature-name binding applied;
cached counterfactual costs were reused and utility records recomputed. All
seven input, source-bank and split invariants pass. Atlanta and New York lose
their old admissions. Hangzhou alone passes the nested cross-fit gate, but its
untouched-reserve source policy costs `+0.00942188` versus rigid,
`+0.00061303` versus matched placebo and `+0.00092563` versus source-blind.

Both frozen decisions are negative: the seven-city global gate fails, and no
city passes the separate four-condition follow-up gate. The old New York
source-specific result is superseded; excluding failed cities cannot produce a
valid corrected source-transfer claim. Stop this dense selector family and do
not launch a city-specific closed loop or Bologna confirmation from it. The
first current-snapshot attempt remains an operational pre-fit failure with no
scientific output. Summed successful runner time is 2,505.084 seconds; seven
result JSON files totaling 601,208 bytes were retrieved. Twenty-one V156A
contract tests and two focused binding tests pass. [Result and immutable
provenance](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v156a_feature_aligned_dense_source_utility.json).

## 2026-09-11 V157A Feature-Aligned V123 Source-Value Replay

`t92484` completes the correction-only V123 Jinan budget curve. Stored-feature-name
binding leaves all six B25--B1000 summaries numerically unchanged from V123.
At B100, source-minus-architecture-matched-target is **-0.01309419**, paired
95% interval **[-0.01653226, -0.00964111]**, with **19/22** improving selector
seeds; uniform source weight 1 is selected in all 22 folds. The separately
frozen historical-path B100 gate passed.

Both learned arms nevertheless remain worse than PhasePressure: source / target-only
deltas are **+0.02533749 / +0.03843168**. PP is the untrained PhasePressure
heuristic baseline, so the original absolute V123 gate remains rejected. This
comparison was subsequently found to mix Jinan domain handling: the target-only
fit retained three scenario labels while the source-component fits relabelled
the same target-adaptation rows to common `jinan`. V157A verifies the historical
arrays but does not isolate source-row contribution and no longer authorizes the
branch experiment.

The task ran on `node005` for 1,203.413 seconds, reused cached counterfactuals
and added zero SUMO simulations. The 209,690,586-byte prediction artifact
remains server-side; only the 457,870-byte result JSON was retrieved. [Historical
replay result](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v157a_feature_aligned_target_budget_source_value_curve.json).

## 2026-09-11 V157B v2 Domain-Aligned B100 Runtime Freeze

V157B v2 completes the frozen 16-fit roster. Across all 158,808 selector rows,
the audit-only target, seven source components, uniform component mean and
runtime wrapper exactly reproduce the V157A B100 score, uncertainty and
context-trust arrays under `numpy.array_equal`. The exact V157A identity gate
passes. The runtime target then relabels all selected B100 target rows to the
common `jinan` domain, removing the V123/V157A comparator confound.

On the reused 22-seed Jinan selector, uniform-source minus domain-aligned
target-only has mean **-0.01357348**, paired 95% interval
**[-0.01783523, -0.00934509]**, and improves on **20/22** seeds. The corrected
B100 relative gate passes. The source arm remains **+0.02533749** worse than PP,
so the absolute PP gate remains rejected.

V157B freezes the domain-aligned target, uniform-source, matched source-label
placebo and PP reference only for the V157C single-focal-TLS, one-action,
450-second Jinan branch experiment. This is no placebo result, native branch
result, fresh-seed result, closed-loop result, cross-city result or PP
superiority result. [Corrected result and bounded V157C authorization](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v157b_feature_aligned_b100_runtime_freeze_result_v2.json).

## 2026-09-11 V157C Jinan Native One-Action Branch Adjudication

The bounded V157C authorization was exercised on the frozen Jinan scenario,
seeds 80314/88625/27178 and checkpoints 300/480/660. Seeds 80314 and 88625 each
completed the shared PhasePressure trajectory plus all nine learned branches;
all physical-start, 450-second-horizon, action-roster and intervention-location
checks pass. These are 20 valid native simulations. All reported collision,
teleport and emergency-stop counts are zero.

The third prespecified seed is not a performance observation. Task `t92564`
completed its shared PhasePressure trajectory and then found no action-eligible
traffic light at the fixed 480-second checkpoint. It stopped with
`V157C checkpoint 480 has no scorable TLS` before any learned branch. Automatic
retry `t92565` was refused by the non-overwrite guard and `t92566` was cancelled.
The frozen protocol forbids moving the checkpoint, treating the missing window
as zero, dropping the seed, or substituting another seed. V157C is therefore
**INCOMPLETE**, and no formal three-seed aggregate or PASS is reported.

For the two complete seed units only, the descriptive uniform-source-minus-
target mean is **+0.00236368 halted vehicles per controlled lane**, with one
seed improving and one worsening. Uniform source minus matched placebo is
`-0.00231739`, but five of the six windows choose the same source and placebo
action and the entire difference comes from one window. Uniform source minus
PhasePressure is **+0.00361626**, worse in both valid seeds; target-only minus
PhasePressure is `+0.00125257`. The six source-target window contrasts contain
two improvements, two regressions and two exact same-action zeros. These native
directions do not confirm the V157B offline relative benefit and do not support
source-model or full-controller adoption.

The direct calibration diagnostic is also negative. Uniform source made five
effective overrides with positive stored predicted advantage over PP, but only
two reduced native 450-second cost; target-only and placebo each improved only
one of five. Across the four windows where source and target chose different
actions, their predicted relative ordering matches the native ordering in only
one window. This points to a native-horizon action-ranking mismatch shared by
the fitted arms, rather than an eligibility bookkeeping problem that could be
fixed by replacing the missing seed.

The earlier interface and roster attempts remain operational provenance only.
V157C v5 failed before evaluator entry because its parent evaluator lacked the
observer hook. V157C v6 completed only the PP baselines and failed before branch
outcomes because singleton feasible-action lights were incorrectly included in
the scored roster. The corrected v7 snapshot has SHA-256
`102b77f2df56f9400f85702f548fe849e0524ca0b7063f93ebc648c1e9f1be4c`.
No additional run is authorized to rescue this observed result. The V157 route
is closed; any later eligibility sensitivity study must be a separately frozen
mechanism diagnostic and cannot be called a V157C completion. [Incomplete
artifact and task accounting](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v157c_jinan_native_action_branches_incomplete_v1.json).

## 2026-09-11 V158 Native-Prefix Ranking Calibration

The final frozen ranking-calibration attempt completed its full development
stage. Tasks `t92692`--`t92701` produced all ten prescribed seed banks: 100
groups, 800 candidate actions, 700 fresh non-reference native-prefix branches,
30 PhasePressure trajectories and zero state restores. Every bank passed its
prefix, roster, horizon, population and teleport checks. Four collision events
occurred in two non-reference branches and none in the PP baselines; these were
diagnostic only under the fixed efficiency protocol.

Fit task `t92703` returned **OOF_GATE_FAIL**. Source minus target was
`-0.00112577`, paired 95% `[-0.00239585,+0.00019170]`, with 6/10 seeds
improving. Source minus matched placebo was `-0.00072284`, paired 95%
`[-0.00218503,+0.00090648]`, with 7/10 improving. Source minus PhasePressure
was `+0.00013549`, paired 95% `[-0.00238959,+0.00260611]`, with 5/10
improving. The first two pass their mean thresholds but not their bootstrap
requirements; source versus target also misses the seed-win requirement. The
PP comparison misses all three requirements. Consequently
`reserve_authorized=false`, and no reserve simulation was submitted.

The native bank contains real headroom: an outcome-aware oracle improves 80 of
100 groups and has mean advantage `0.01653210` over PP. The learned source arm
does not rank it reliably. Its 48 PP overrides split 21 improvements and 27
regressions, with predicted-versus-realised advantage correlation `-0.0974`.
It improves over PP in the `_2000` demand but regresses in the base and `_2500`
demands. This locates the remaining failure in seed- and demand-stable action
ranking, rather than snapshot restoration, action availability or executor
bookkeeping.

V158 closes the incremental Jinan native-prefix calibration route. The
server-side full-data calibrator is audit-only and is not authorized for
reserve use, controller adoption or deployment. No post-result threshold,
feature, seed, fold or model change is permitted under this protocol. This
negative case must remain visible if later claims are narrowed to separately
confirmed cities. [Final OOF result and provenance](/home/erzhu419/mine_code/CFCMT/cf_h2o/results/paper_artifacts/tsc_v158_native_prefix_ranking_calibration_v1.json).
