# CFCMT Research Checkpoint (2026-09-01)

## Current objective

Refactor CFCMT into a causal source-selection and weighting framework that can
detect and suppress negative transfer. Development must pass source-value,
absolute-safety, placebo and held-out gates before any untouched-city
confirmation is launched.

## Resume update (2026-09-01 11:52 CST)

- V127 task `t87922` completed all fitting and fold evaluation, then failed
  before publishing a result because strict JSON serialization rejected the
  mathematical `+infinity` sentinel of a disabled intervention profile. It is
  not an efficacy result. Identical automatic retry `t88263` was cancelled
  before launch.
- The JSON-safe correction represents that disabled threshold as `null` and
  explicitly accepts no interventions when the profile is disabled. The model,
  folds, calibration rules and pass gate are unchanged. Combined V127 and
  Boston package tests pass (`16 passed`).
- Corrected V127 task `t88306` completed from immutable
  snapshot `08997193db8cfaff6a53` with signature
  `CFCMT/v127/causal-pairwise-source-prior-b500-v2-json-safe`. Its initial hard
  `node003` placement was changed before launch to the audited non-`node004`
  CPU-node allowlist; the frozen command and scientific protocol were
  unchanged. It ran on `node005` for 5,824.03 s and was rejected by the frozen
  gate. Result SHA-256:
  `126526c8adc900aa63a219a47288c9c1c37712fcfd67ea9d22784de8b9cde782`.
- Boston's full-run failure was traced to 551 connection attributes that
  remained `state="Z"` after package v5 changed only 138 junction types.
- Package v6 jointly changes the 138 declared junction types and their 551
  declared connection states (`Z->m`). Static compatibility admission passed
  while preserving all other network XML attributes, 3,806,510 trips and 4,203
  traffic-light programs.
- Package v6 manifest:
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/package_v6/package_manifest.json`
  (SHA-256
  `7dae903cf5f3b77263140ac4509d4d6bc9c22c1a2394def35b9a6e21b729636a`).
- Full route admission passed for all 3,806,510 trips:
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/route_admission_v4/result.json`
  (SHA-256
  `01cb52cb21595f8eaed9511057671906d1cee3c5b3425b9d00250e9c29053807`).
- The 3,000 s package-v6 trigger window passed from 7,201 through 10,201,
  crossing the former 9,518 s failure with zero collisions, zero teleports and
  all 4,203 controllable traffic lights. Result SHA-256:
  `3b585570eba671969b6a41341327b787b4cc3f8b636db343096b4d4199a95fd1`.
- Complete Boston admission scheduler task `t88337`, signature
  `CFCMT/v118r/eth-boston-full-admission-v4-junction-link-state`. It requests
  16 cores because the frozen SUMO configuration uses 15 synchronized routing
  threads, and uses a new v6-complete RAM history family so the earlier
  9,518-s failed run cannot understate full-day memory. It failed the frozen
  zero-collision gate at simulation time 12,279 s on lane `-426602284_0`.
  Deterministic automatic retry `t88390` was force-cancelled.

## Continuation update (2026-09-01 14:08 CST)

- V127 completed and was rejected. Its three calibrated held-out policies were
  identical; only 1/22 folds enabled and the paired source contributions were
  exactly zero. The source-as-an-extra-feature family is closed.
- V128 implements a frozen B500 target/source offset plus a causal
  reference-contrast target residual. A ten-seed paired calibration gate may
  select the aligned source offset or an explicit target-only source null. The
  placebo receives the same selection capacity.
- V128 local and V127 regression tests pass (`15 passed`). Immutable snapshot:
  `e4c93d6ecc6687c4c839` (SHA-256
  `e4c93d6ecc6687c4c83901055c5d2314693910565faee182a41b6704f8c144bf`).
- V128 scheduler task `t88451` was submitted only to load-admitted `node003`
  with signature
  `CFCMT/v128/anchored-source-null-reference-residual-b500-v1`.
- Boston full admission `t88337` stopped after one collision between vehicles
  `56953` and `32599` at time 12,279 s, lane `-426602284_0`. It had 28,398
  departed and 25,266 arrived vehicles, zero teleports and all 4,203 traffic
  lights. Result SHA-256:
  `de4f558dd3ad8a6a9b762e6cafe6e60bc0d3def7184b68e09df35b67806d7d32`.
- The collision edge receives three movements into one lane at junction
  `61351006`; one straight movement is uncontrolled while two turning
  movements belong to TLS `joinedS_726`. The duplicate retry was stopped while
  that reachable merge topology is audited.

## Continuation update (2026-09-01 14:30 CST)

- The Boston v6 collision was localized to two priority-green turn movements
  entering the same 15.85 m single receiving lane as an uncontrolled major
  straight movement. Package v7 changes only `joinedS_726` phase 10/link 12
  and phase 12/link 14 from `G` to yielding `g`.
- The structured static audit passed: exactly two declared phase characters
  changed, while the other eight package files reuse v6 exactly. Package v7
  manifest SHA-256:
  `b39d2676ae72f57f7494d908640ac697c96397b09c587dafa96e4c038e0ef2b9`.
- Complete-demand route admission task `t88602` was submitted from immutable
  snapshot `b04cf81c9fefa8311c29`. It must pass before the 5,400 s targeted
  trigger window is launched.
- V128 task `t88451` completed on `node003` in 1,334.66 s and was rejected.
  Adaptive source-null selected the aligned source in 0/22 folds and was
  exactly identical to target-only. Its mean normalized delta was harmful at
  `+0.0002073`; result SHA-256:
  `f7fd449d2ac62338815ccb62ad2e0bab662fbdec40afcd9467d9548070cb4155`.
- V128's per-arm target residual refits can cancel each arm's prior. V129 is
  authorized to fit one target-only residual per outer fold and freeze that
  residual across target, aligned-source and source-placebo priors. All
  source-null, placebo, pressure and held-out gates remain unchanged.
- Boston v7 complete-demand route admission passed all 3,806,510 trips in
  515.61 s on task `t88602`; result SHA-256:
  `b82d5f81aeaf1fad89c8937b333aef81776822be36939afc5375bf8d71a211f2`.
  The 5,400 s libsumo trigger-window task is `t88664` on `node002`.
- V129 local regression tests pass (`18 passed`). Immutable snapshot
  `884f285e8112a87d183e` was staged and task `t88665` submitted to
  load-admitted `node003` with signature
  `CFCMT/v129/shared-target-residual-source-null-b500-v1`.
- V129 task `t88665` completed all 22 folds but failed before publishing
  because its final call passed an unsupported keyword to the already-strict
  atomic JSON writer. No efficacy artifact was produced. Automatic identical
  retry `t88697` was force-cancelled. The execution-only correction removes
  that keyword and makes a scientific `REJECT` a successful scheduler task;
  the method, data, folds and gates are unchanged.
- Corrected V129 task `t88699` completed on `node003` in 565.65 s and was
  rejected. The aligned source was selected in only 1/22 folds; that held-out
  fold was harmful. Adaptive source-null minus target-only was `+0.0000346`.
  Result SHA-256:
  `d0724b1e211944713bc424bfe701885acd2a67478836c2a1e45e492794bfb237`.
- V129 proves that shared residuals prevent prior cancellation but do not solve
  seed-held-out intervention identification. V130 must screen target-side
  causal improvement classification and sign-balanced group-normalized
  advantage regression without source inputs. Source fusion resumes only if
  an absolute intervention model passes.

## Claim boundary already established

- The earlier positive result is real. V98 improved over its matched
  lower-capacity target-only/reference policy in all 56 paired seeds, with mean
  waiting approximately 4.53% lower.
- V123 also found positive source value relative to its architecture-matched
  target-only arm at budgets 25, 50, 100, 500 and 1000.
- These results establish relative transfer value. They do not establish that
  the learned controller beats PhasePressure in absolute closed-loop control.
- V123's stronger deployment gate therefore correctly rejected the source
  policy even though the relative source effect was positive.

## Continuation update (2026-09-01 15:37 CST)

- V130 screened target-side intervention identification before reintroducing
  any source prediction. Its 22 outer folds used 11 selector seeds for fit, 10
  disjoint seeds for retention calibration and one held-out seed for scoring.
  The held-out labels were not available to fit or calibration.
- Immutable snapshot `d64a24724e1eb467a3b5` (SHA-256
  `d64a24724e1eb467a3b584bf67e6a07e36ba0f0f8c337330fb567e61a7cc6ce1`)
  ran as task `t88708` on `node005` in 727.10 s. The 20-worker execution used
  no source or target-prior prediction artifact.
- The causal improvement classifier enabled 2/22 held-out folds and selected
  18 interventions. Its mean normalized waiting delta was `-0.0001545`, with
  95% interval `[-0.0004198, 0]`. Both enabled folds improved, but the frozen
  requirement of mean at most `-0.0005` and a strictly negative upper bound
  was not met.
- The sign-balanced group-normalized advantage regressor enabled 5/22 folds
  and selected 133 interventions. Its mean was `-0.0000184`, with 95% interval
  `[-0.0008052, +0.0008617]`; two enabled held-out folds were harmful.
- V130 is therefore rejected. Result SHA-256:
  `f7f15c35ab48018f5caf2e384c434c0b5a013cdfa109a5844c564426204dce4e`.
  Source-veto/placebo integration is not authorized for either screened
  family. The next development step must change the intervention
  representation or supervision, not tune source weights against these folds.

## Continuation update (2026-09-01 16:21 CST)

- Boston v7 trigger task `t88664` crossed the former 9,518 s failure but was
  rejected at 11,504 s after one collision on receiving lane
  `185856197#0_0`. Result SHA-256:
  `0a266c4e5d90a0bfe4d8eaddbd9273ccbe33e2c667c7eb808dbbf600eaebad0f`.
- The collision was localized to junction `71937638`, where equal-priority
  single-lane approaches `89717098#1` and `9429279#2` were both declared
  uncontrolled major movements into the same one-lane edge
  `185856197#0`. Package v8 changes only the lateral approach connection
  `9429279#2 -> 185856197#0`, lane 0 to lane 0, from major `M` to minor yield
  `m`.
- The v8 structured audit passed with exactly one declared connection
  attribute change; the other eight package files reuse v7 exactly. Package
  manifest SHA-256:
  `dec12562c325c63246e92a73e66b1b76f8e7e145c4cba4e899d723709969ef31`.
  Complete-demand route admission task `t88711` is running on `node002`.
- V131 replaces the failed 117-feature target action models with the existing
  31-member generalized-pressure mechanism family. Eleven target seeds select
  one low-dimensional mechanism and retention level, ten disjoint seeds apply
  the frozen authorization gate, and one seed is held out. No source artifact
  is read. Task `t88712` used immutable snapshot `01377cf6ec2bc9f853ef`
  on `node005`.

## Continuation update (2026-09-01 16:50 CST)

- Boston package v8 complete-demand route admission task `t88711` passed all
  3,806,510 trips on `node002`. Result SHA-256:
  `bc347e3d42ba0d925887a922067b719ccc8705b820c73a48fe0adddb9de52cf5`.
- The package-v8 5,400 s trigger-window task is `t88714` on `node002`. It must
  cross all three prior failure times, 9,518, 11,504 and 12,279 s, with zero
  collisions and zero teleports before complete admission can launch.
- V131 task `t88712` completed in 101.27 s and was rejected. No outer fold
  passed its disjoint ten-seed authorization gate; all 22 held-out folds fell
  back exactly to PhasePressure. Result SHA-256:
  `f379ade41bad5caba0d532808b826630cfb8b0574f9eacab3f9541dd2a680b42`.
- V131's selected profiles frequently changed sign between the 11 training and
  10 calibration seeds. This is genuine instability, but the split also uses
  only half of the available development seeds at each stage.
- V132 is frozen to distinguish those explanations without changing the action
  family. Each outer fold retains one untouched seed and uses all remaining 21
  seeds to evaluate the same 186 predeclared rule-retention profiles. A
  Gaussian-multiplier one-sided max-t simultaneous bound controls selection
  over the correlated family. The absolute effect, negative simultaneous
  bound, intervention-count and 80% seed-consistency gates are unchanged in
  substance. No source input is used.

## Continuation update (2026-09-01 17:02 CST)

- V132 immutable snapshot `72ae8a3c6cad3b50c805` passed 29 related tests and
  ran as task `t88956` on `node006` for 177.47 s. Result SHA-256:
  `f4f57f4aac0f31a7705652cf4c7f20b7b61f49fd1fde4697595d400d162d5007`.
- V132 was rejected with zero authorized outer folds. Its correlation-aware
  max-t critical value was approximately 2.91; this is not a Bonferroni-driven
  rejection. The best non-empty profiles usually contained only 2--7
  interventions across 21 development seeds, averaged approximately
  `-0.000090`, and improved no more than 29% of seeds. They were far from the
  frozen `-0.000500`, 80-intervention and 80%-consistency requirements.
- V131 and V132 therefore close the 31-rule generalized-pressure intervention
  family. Adding source priors or tuning the gate against these folds is not
  authorized.
- V133 is the next bottom-up diagnostic: align the same target scenario by
  control interval, TLS and exact candidate state across 21 historical seeds,
  form a conservative action consensus, and score only on the 22nd seed. It
  tests cross-realization identifiability of the 450-second counterfactual
  target before another learned causal model is attempted.

## Continuation update (2026-09-01 17:13 CST)

- V133 task `t88958` ran from immutable snapshot
  `33843e6ae577f04961dc` on `node005` for 168.05 s. Result SHA-256:
  `ea6227aae879599e9ce14a3914797dbdae4115db1bccdb7797585e1bcfd2595d`.
- V133 was rejected. Only 4,718/19,851 held-out groups had any exact
  scenario/interval/TLS/candidate-state history match, and only 91 had
  pressure-safe support in at least 17/21 development seeds.
- The liberal historical mean lookup made 29 interventions across 18 folds and
  was harmful on held-out data: mean `+0.0002466`, 95% interval
  `[-0.0000460, +0.0005576]`. The conservative 80%-agreement arm made no
  intervention.
- Direct prediction or memorization of the 450-second action label is therefore
  closed. V134 must test one-control-interval local mechanism changes as
  surrogate objectives. Only a surrogate whose held-out oracle improves the
  450-second waiting estimand under a familywise gate may license another
  CFCMT/MC-WM model.

## Continuation update (2026-09-01 17:25 CST)

- V134 task `t88961` ran from immutable snapshot `35c367707a86ee60774e`
  on `node005` for 163.77 s. Result SHA-256:
  `5613ccdf2a6727769109b348cfd14bef1b766dc68bea79ddd0fa76a8cccfda0e`.
- V134 was rejected under its preregistered gate: 0/48 one-step mechanism
  profiles passed all absolute, familywise and cross-seed requirements.
- The balanced physical proxy retained real aggregate signal. At 5% retention
  it made 173 interventions with mean effect `-0.0007069` and simultaneous
  upper 95% bound `-0.0000349`; at 10% retention it made 334 interventions
  with mean effect `-0.0013154` and simultaneous upper bound `-0.0001973`.
  They failed only the frozen 80% improving-seed gate, reaching 77.3% and
  68.2%, respectively. The gate is not relaxed post hoc.
- Do not train a global one-step surrogate ranker. V135 may test only a frozen
  outer-fold context selector that uses pre-action observables to abstain in
  heterogeneous regimes. If that target-only selector fails the same absolute
  gate, close the mechanism-control branch and retain CFCMT as a transfer layer.
- Detailed result note:
  `paper/tsc_v134_local_mechanism_surrogate_oracle_result.md`.

## Continuation update (2026-09-01 17:51 CST)

- V135 task `t88967` ran from immutable snapshot `4b22e42cd9d0c99210ce`
  on `node003` for 302.26 s. Result SHA-256:
  `05359a3283d0bae821f6e691050a53cbe451fb7fd882510104a4e31524b6e372`.
- V135 was rejected. Its nested pre-action context gate enabled 9/22 folds,
  made 76 interventions and achieved mean normalized waiting effect
  `-0.0002592`, bootstrap 95% interval
  `[-0.0005943, +0.0000338]`. Only 27.3% of held-out seeds improved.
- Every selected profile used risk multiplier zero; no OOF-error-penalized
  profile survived inner authorization. Harm remained in three of the nine
  active held-out folds despite negative calibration results.
- Close the direct one-step mechanism-control branch. Do not fit V136 merely
  to approximate an oracle that already failed context identifiability.
  Preserve CFCMT as a calibration-light causal residual/transfer layer and
  narrow the TSC claim relative to PhasePressure.
- Detailed result note:
  `paper/tsc_v135_context_gated_mechanism_oracle_result.md`.

## Completed development evidence

### V123 target-budget source-value curve

- Result: `cf_h2o/results/cluster/tsc_v123_target_budget_source_value_curve_20260901/development_v2_vectorized/result.json`
- Result SHA-256: `cc59ab48768901a0e73d91cde878b0a764b6a8e16f8a929df4a2406efd80d944`
- Source minus matched target-only waiting effects:
  - B25: -0.0317155, 95% CI [-0.03607, -0.02768]
  - B50: -0.0114794, 95% CI [-0.01574, -0.00721]
  - B100: -0.0130942, 95% CI [-0.01653, -0.00964]
  - B250: inconclusive
  - B500: -0.0085338, 95% CI [-0.01160, -0.00550]
  - B1000: -0.0149057, 95% CI [-0.02041, -0.01004]
- Final absolute selection: reject.

### V124 conservative intervention gate

- Result: `cf_h2o/results/cluster/tsc_v124_conservative_source_intervention_gate_20260901/result_v2.json`
- Result SHA-256: `314992433c0d6619e0602d730dcbddefb016e5b74beb7618016ca29dce4cc418`
- Decision: reject; all deployment arms fall back to PhasePressure.
- Interpretation: the simple held-out conservative gate did not turn the
  relative transfer gain into safe absolute deployment.

### V125 intervention feasibility oracle

- Result: `cf_h2o/results/cluster/tsc_v125_source_intervention_feasibility_20260901/result_v1.json`
- Result SHA-256: `8749c85ffb864c32f96ab29d786f85eac25f7db14843bfc493b005b234ef22c6`
- Pressure-nondegrading oracle effect: -0.0770925, 95% CI
  [-0.08067, -0.07349], with 3359 interventions and zero harmful selected
  interventions.
- The B500 learned held-out guard was slightly harmful: +0.0001315, 95% CI
  [+0.0000224, +0.0002657].
- Interpretation: useful safe action headroom exists; the bottleneck is action
  ranking and intervention identification, not a hard environment ceiling.

### V126 linear source ranker

- Result: `cf_h2o/results/cluster/tsc_v126_source_ranker_feasibility_20260901/development_v1/result.json`
- Result SHA-256: `368801bf6207d0bc77081e6894921f6051ffeb7c8a374d0acacba26b66d028cf`
- Decision: reject. Neither target-only nor source-aligned linear rankers beat
  PhasePressure, and the source-placebo comparison did not identify a reliable
  source contribution.
- Interpretation: the 175-dimensional linear conversion is insufficient even
  under abundant target counterfactual supervision.

### V130 target-side intervention model screen

- Result:
  `cf_h2o/results/cluster/tsc_v130_intervention_model_screen_20260901/development_v1/result.json`
- Result SHA-256:
  `f7f15c35ab48018f5caf2e384c434c0b5a013cdfa109a5844c564426204dce4e`
- Decision: reject both the causal binary improvement classifier and the
  candidate-only, sign-balanced, group-normalized advantage regressor.
- Interpretation: changing the target objective improves selected folds but
  does not make intervention identification stable across selector seeds.
  Source selection cannot repair an action model that fails its target-only
  absolute gate.

## Completed V127 development task

### V127 causal pairwise source-prior feasibility

- First scheduler task: `t87922` (failed only during final JSON serialization;
  no result artifact was published)
- Cancelled identical retry: `t88263`
- Corrected scheduler task: `t88306` (done)
- Execution node: `node005` after the pre-launch placement-only amendment
- Signature:
  `CFCMT/v127/causal-pairwise-source-prior-b500-v2-json-safe`
- Snapshot root: `08997193db8cfaff6a53`
- Runtime: 5,824.03 s with a 20-core allocation on shared `node005`.
- Local result:
  `cf_h2o/results/cluster/tsc_v127_causal_pairwise_source_prior_feasibility_20260901/development_v2_json_safe/result.json`
- Remote output:
  `/home/zhengliang01/scheduleurm_work/CFCMT_RESULTS/tsc_v127_causal_pairwise_source_prior_feasibility_20260901/development_v2_json_safe/result.json`
- Result SHA-256:
  `126526c8adc900aa63a219a47288c9c1c37712fcfd67ea9d22784de8b9cde782`.
- The approximately 210 MB prediction PKL remained on the server; only the
  367,922-byte result JSON was retrieved.
- Protocol: nonlinear pairwise action ranking; target-only, source-aligned and
  whole-action-group source-placebo arms; 16 selector seeds, 5 disjoint
  calibration seeds and 1 held-out test seed per outer fold; PhasePressure is
  always available; only pressure-nondegrading actions are eligible.
- This is an abundant-target feasibility experiment, not yet a few-shot or
  untouched-city confirmation experiment.
- The exact target-information accounting is recorded in
  `paper/tsc_v127_target_information_budget_audit.md`. Per outer fold, the
  pairwise ranker consumes 14,195--14,654 target selector groups and the gate
  consumes another disjoint 4,368--4,744 groups, in addition to the B500
  upstream target prior. A successor curve must cap the union of target groups
  used by the prior, ranker, source weighting, and calibration; changing only
  the upstream prior budget is not a few-shot experiment.
- Decision: reject. Target-only, source-aligned and source-placebo all had mean
  normalized delta `+0.0000998` with 95% CI `[0, +0.0002995]`; only one of 22
  folds enabled, and the paired source contributions were exactly zero.
- Result note:
  `paper/tsc_v127_causal_pairwise_source_prior_feasibility_result.md`.
- Consequence: close the source-as-an-extra-feature family. The next abundant
  feasibility method must use a source-pretrained pairwise prior, a target OOF
  residual or specialist and an explicit source-null option before a strict
  target-budget curve is authorized.

## Boston complete-network admission

- Static package v5 and route admission passed for all 3,806,510 trips and all
  4,203 traffic lights.
- The 600 s operational preflight passed, but it was correctly not counted as
  scientific admission.
- Full task `t87794` failed deterministically at simulation time 9518 s:
  `FatalTraCIError: Zipper junctions with more than two conflicting lanes are not supported (at junction '61394761')`.
- Failed result preserved at:
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/full_admission_v3_failed_t87794/result.json`.
- Automatic identical retry `t87923` was force-cancelled because it used the
  same package and could only reproduce the known failure.
- Direct inspection confirmed that junction `61394761` was `priority` in v5
  while all four associated connections remained `state="Z"`. SUMO 1.22 uses
  link state to enter its zipper speed path, so the type-only repair was
  incomplete.
- Package v6, full route admission and the 3,000-s trigger-window gate pass.
  Complete microscopic admission task `t88337` nevertheless failed at 12,279 s
  because one collision occurred on lane `-426602284_0`; result:
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/full_admission_v4/result.json`.
- Automatic retry `t88390` used the same deterministic package and was
  force-cancelled. No Boston controller evaluation is authorized until a
  topology repair passes static, route, trigger-window and complete admission
  again.

## Resume order

1. V127 is adjudicated and rejected. Preserve its result and close the
   source-as-an-extra-feature pairwise family.
2. V128 is adjudicated and rejected. Preserve the result and do not interpret
   its small unadjusted source-aligned mean as deployment evidence.
3. V129 is adjudicated and rejected. Preserve its result and do not tune source
   weights against the failed folds.
4. V130 is adjudicated and rejected. Preserve its result and do not integrate
   source-veto/placebo with either failed intervention family.
5. Boston v8 static repair and complete route admission passed. Complete trigger
   task `t88714` across 9,518, 11,504 and 12,279 s, followed by complete
   admission. Do not use Boston for controller evaluation before all pass.
6. V131--V135 are adjudicated and rejected. Preserve them; close generalized
   pressure, direct 450-second action prediction, unconditional one-step
   surrogates and context-gated one-step surrogates. Do not train V136.
7. Do not run an absolute-superiority source budget curve without a target-only
   method that passes the absolute PhasePressure gate. Source selection and
   weighting may proceed only under a separately frozen transfer-layer
   estimand that keeps absolute PhasePressure performance explicit. Any later
   untouched-city confirmation remains tuning-free.
8. Update manuscript claims, dataset table and fixed figure scripts only from
   accepted artifacts; preserve relative and absolute comparisons as separate
   estimands.

## Cluster constraints

- Use `node001`-`node006`, at most 20 CFCMT workers per node; do not place new
  work on blocklisted `node004`.
- Use immutable staged snapshots and `libsumo`, not TraCI.
- Inspect completed, queued and running signatures before dispatch to avoid
  duplicates.
- All scientific city experiments use complete city/day/routes. Short runs are
  operational diagnostics only.
- Pull logs before JSON/CSV and never pull large CSV, PKL or checkpoint artifacts
  to the local disk.

## Continuation update (2026-09-01 18:35 CST)

- Boston v8 trigger task `t88714` was rejected after one collision at simulation
  time 11,304 s on lane `618556460#8_0`, 2.30 m downstream of junction
  `cluster_73110325_73135282`. It had 19,800 departures, 17,321 arrivals,
  2,479 active vehicles, five pending vehicles and zero teleports at exit.
- Server-side structured XML inspection found two uncontrolled mainline lanes
  with `state="M"` and a `joinedS_777` ramp movement with protected green `G`
  entering the same receiving lane. Package v9 changes only phase 2, link 8 of
  `joinedS_777` from `G` to yield green `g`.
- Package v9 passed static exact-change admission. Its manifest is
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/package_v9/package_manifest.json`,
  SHA-256 `d2b3918188a14ad68552d7f10204076d9f7cf1a3cd27c444ea745c2eade90a3f`.
- Route task `t89030` did not reach duarouter because the admission dispatcher
  recognized only package protocols through v8. The protocol wiring was added
  and covered by 16 focused tests; corrected route task `t89035` uses immutable
  snapshot `090e7509d43984eb9880` and a new signature.
- V140 freezes a corrected total target-information budget. The five smallest
  selector seeds are adaptation-only; the remaining 17 are evaluation-only.
  Target-only and seven one-source causal models are cross-fitted by adaptation
  seed, source weights are nonnegative with an explicit source-null mass, and a
  whole-action-group source permutation is the matched placebo.
- V140 local validation comprises 29 related model regressions plus 14 direct
  protocol/aggregate tests, all passing. Six budgets were submitted as tasks
  `t89036`--`t89041` on node001/2/3/5/6; node004 remains excluded. B25 is the
  sole primary relative-transfer test. Other budgets receive shared-seed max-t
  simultaneous intervals and cannot be selected after evaluation.
- Before any V140 outcome, V141 fixes Boston as the sole new-target-city
  confirmation. It runs only if V140 B25 and Boston complete admission both
  pass. It inherits B25, all seven sources, ridge 0.05, source-null, placebo and
  efficacy thresholds without retuning. Failure does not permit substituting a
  different city after outcomes are known. Protocol:
  `paper/tsc_v141_boston_total_budget_confirmation_protocol.md`.

## Continuation update (2026-09-01 19:05 CST)

- Corrected Boston package-v9 route task `t89035` passed all 3,806,510 trips in
  373.84 s with SUMO 1.22, TAZ routing and no ignored routing errors. Result:
  `cf_h2o/results/cluster/tsc_v118r_eth_boston_schema_remediation_20260901/route_admission_v7b/result.json`.
- The package-v9 5,400 s trigger-window admission is task `t89043`. It must
  cross all previous 9,518, 11,304, 11,504 and 12,279 s failure points before
  complete admission can be submitted.
- V140 tasks `t89036`--`t89041` completed and the six-budget aggregate passed
  all seed, input-identity and nested-budget integrity checks. Every budget was
  rejected. B25, the sole preregistered primary budget, assigned source-null
  mass 1.0 after its five adaptation held-out effects were harmful versus
  target-only (mean +0.08200; 2/5 improving) and placebo (mean +0.01432).
  V141 is therefore not authorized and no alternative budget may be selected.
- V140's optimizer itself did not collapse: its B25 raw pre-gate source weights
  were non-zero and summed to one. The evaluation-only uniform-source
  descriptive arm had mean normalized policy delta 0.04750 versus 0.05975 for
  target-only. This is post-outcome diagnostic evidence that continuous
  row-MSE weight fitting plus a five-seed gate may discard useful source signal;
  it is not a passing V140 arm.
- V142 freezes one isolated correction: direct downstream-policy selection
  over exact source-null, seven single-source arms and a uniform-source arm at
  masses 0.25, 0.50, 0.75 and 1.00. The total budget remains B25, with an
  alternating 11-adaptation/11-evaluation seed split and the matched
  whole-action-group placebo. A pass can authorize multicity development only,
  never Boston confirmation directly. Protocol:
  `paper/tsc_v142_total_budget_policy_aligned_selection_protocol.md`.
- V140/V142 core and launcher regressions pass (`17 passed`). V142 uses
  immutable snapshot `02d390ebf1d9592b9c7b`; task `t89045` was submitted to
  the non-node004 CPU-node allowlist.

## Continuation update (2026-09-01 20:01 CST)

- V142 completed and was rejected. Its direct policy selector chose a non-null
  RESCO source arm in all 11 adaptation folds, but only 3/11 target and 5/11
  placebo held-out folds improved. Both one-sided upper bounds remained
  positive, so the frozen gate returned exact source-null. Lowering the gate is
  not authorized.
- V143 froze a target-label-free uniform source rule over seven complete-city
  holdouts at B25. The first task set `t89048`--`t89054` exposed an engineering
  error: ragged 2/3-action groups were incorrectly passed through
  `np.vstack`. Four tasks failed and three were cancelled; no result from that
  set is scientific evidence. The corrected implementation retains ragged row
  indices, explicitly selects the PhasePressure reference row and passes 30
  focused regressions.
- Corrected V143 tasks `t89067`--`t89073` completed from immutable snapshot
  `ef21656bd95c229ba81f`. The seven-city aggregate is
  `cf_h2o/results/cluster/tsc_v143_multicity_uniform_source_ensemble_20260901/development_v2/aggregate.json`.
  Uniform CFCMT improved pooled H2O+ in all seven cities (mean effect
  `-0.07887`, one-sided 95% upper bound `-0.03395`). It nevertheless failed the
  frozen development gate: versus target-only it improved 4/7 cities, had mean
  `-0.00973`, one-sided upper bound `+0.00520` and maximum city regression
  `+0.01592`; versus matched placebo it improved 5/7 but the one-sided upper
  bound was `+0.00823`. The fixed uniform ensemble is closed. This is positive
  transfer evidence against dense H2O+, not a universal safety claim.
- V144 now separates headroom measurement from deployment. It records all 42
  ordered one-source-to-target effects, source-specific whole-group placebos
  and scenario-equal city signatures using only current features and static
  context. Oracle source identities are descriptive only. A source-only
  leave-one-city gate may be developed only if at least five targets have a
  source improving target-only by `0.0005`. Seven inventory tasks
  `t89080`--`t89086` use immutable snapshot `62a153db872884395f3a`; protocol:
  `paper/tsc_v144_multicity_source_compatibility_inventory_protocol.md`.
- Boston package v9 trigger task `t89043` failed at simulation time 9,735 s on
  lane `846300601#2_0` (vehicles `3610` and `4793`). Server-side two-trip
  duarouter evidence places them on mutually exclusive link 3 and link 7
  phases, while the junction has only a one-second yellow clearance and no
  internal lanes. No phase-duration repair is authorized from static inference
  alone. Diagnostic replay `t89087` records participant route neighbors and
  the live TLS phase/state at collision; it is not admission evidence.

## Revised resume order

1. Aggregate V144 only after all seven result contracts pass. If headroom is
   admitted, train a source-only compatibility gate with each evaluated target
   city absent from all meta-gate labels, including rows where it acts as a
   source. Do not use V144 oracle identities as a deployed method.
2. Read `t89087` logs first, then its small result JSON. Repair Boston only when
   the live collision phase and participant predecessor edges identify one
   declared mechanism. Every repair restarts static, route, trigger and full
   admission; controller evaluation remains prohibited until all pass.
3. An untouched-city confirmation is authorized only by a passing frozen
   source-only gate and a separately admitted target network. V141 remains
   blocked by V140 and cannot be revived by later development results.
