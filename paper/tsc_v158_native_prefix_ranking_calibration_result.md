# TSC V158 Native-Prefix Ranking Calibration Result

**Status: OOF_GATE_FAIL.** The frozen development bank and five-fold fit both
completed, but the source-native arm failed the preregistered whole-seed OOF
gate. `reserve_authorized` is false, so no reserve trajectory was submitted and
the V158 native-prefix calibration route is closed.

## Complete development bank

Tasks `t92692`--`t92701` produced all ten fixed seed banks from execution
snapshot `423dc81960505f69f08e35808afd1690146bd49aba6a3a4c1b09d0679ddc3599`.
The merged bank has exactly 100 action groups, 800 candidate rows, 700 fresh
non-reference branches, 30 uninterrupted PhasePressure trajectories, and zero
SUMO state restores. Every seed passed the physical-prefix, candidate-roster,
450-second-horizon, active-population and teleport checks.

Four collision events occurred in two non-reference branches from two seeds;
the PhasePressure baselines had none. They are retained as diagnostics and do
not enter the efficiency gate, as fixed by the protocol.

## Preregistered OOF result

Lower cost is better. The four equal-seed OOF means are:

| Arm | Mean 450-second cost |
|---|---:|
| target native | 1.02511667 |
| source native | 1.02399090 |
| matched placebo native | 1.02471373 |
| PhasePressure | 1.02385540 |

The three comparisons all had to pass their mean, bootstrap-upper-bound and
seed-win conditions. None did.

| Comparison | Mean difference | Paired bootstrap 95% | Improving seeds | Failed conditions |
|---|---:|---:|---:|---|
| source - target | -0.00112577 | [-0.00239585, +0.00019170] | 6/10 | upper bound, seed wins |
| source - placebo | -0.00072284 | [-0.00218503, +0.00090648] | 7/10 | upper bound |
| source - PhasePressure | +0.00013549 | [-0.00238959, +0.00260611] | 5/10 | mean, upper bound, seed wins |

Thus the source predictions retain a small average advantage over the other
learned arms, but it is not stable across unseen development seeds and it does
not beat the PhasePressure heuristic.

## Failure diagnosis

The failure is not caused by a lack of beneficial actions. Selecting the best
native action after observing all eight outcomes would improve 80 of 100 groups
and reduce mean cost by 0.01653210 relative to PhasePressure. This is oracle
headroom and is not deployable.

The learned ranker cannot identify that headroom reliably. The source-native
controller overrode PhasePressure in 48 groups: 21 overrides improved cost and
27 worsened it. Its mean predicted advantage over those overrides was
+0.00775157, while the mean realised advantage was -0.00028228; their Pearson
correlation was -0.0974. The target and placebo arms show the same sign failure.

Calibration did reduce damage: the uncalibrated frozen source action overrode
PP in 82 groups and cost +0.00247963 relative to PP, whereas V158 reduced this
gap by 94.5% to +0.00013549 with 48 overrides. It therefore learned useful
abstention, but captured -0.82% of the available oracle headroom and did not
turn the reduction in harm into a stable benefit.

The source prediction does affect ordering, but only sparsely. Source and
target selected different actions in 18 of 100 groups; source was better in 12
and worse in six, producing the favourable source-minus-target mean. That
benefit was concentrated enough to yield only 6/10 improving seeds. Source and
placebo differed in 20 groups, split 12 improvements and eight regressions.

The absolute result also changes with demand: source minus PhasePressure was
-0.00304784 in `jinan_3x4_real_2000`, but +0.00114815 in
`jinan_3x4_real` and +0.00196862 in `jinan_3x4_real_2500`. The fixed causal
features, residual model and B100 native budget therefore do not learn a
ranking that transfers consistently across seed and demand variation.

## Decision and evidence

Fit task `t92703` ended normally in 4.29 seconds. Its `OOF_GATE_FAIL` is a
scientific result rather than an operational failure. The full-data calibrator
is retained server-side only for audit and is not authorized for reserve use or
deployment. No threshold, fold, seed, feature, model or cooldown was changed
after seeing the outcome.

The compact authoritative artifact is
`cf_h2o/results/paper_artifacts/tsc_v158_native_prefix_ranking_calibration_v1.json`
(SHA-256 `41d534a6126d640ea8669baa79d7fe65ff0f28cb17041e095eb5e29445a390a9`).
The exact fit result and post-hoc diagnostic are retained under
`cf_h2o/results/cluster/tsc_v158_native_prefix_ranking_calibration_20260911/`;
the 236,663-byte merged bank, 708,468-byte calibrator and full OOF records
remain on the server.

V158 rejects this Jinan native-prefix source-calibration controller. It does
not turn earlier protocol-specific controller-pair results into general source
transfer evidence, and it provides no basis for deleting Jinan after observing
the failure. A future paper may narrow a claim to independently confirmed
cities, but this failed prespecified city must remain visible as a negative
result rather than being pooled into or removed from a general transfer claim.
