# TSC V159 Native Endpoint Ablation Result

**Status: OOF_GATE_FAIL.** The target-only endpoint representation lowered the
mean development cost, but the improvement was not stable across seeds or the
three fixed demand settings. The direct-endpoint hypothesis is closed under the
frozen V159 design, and no source or reserve stage is authorized.

## Frozen comparison

V159 reused the exact V158 native B100 bank: 100 action groups, 800 rows, ten
development seeds and three Jinan demand settings. It added no SUMO trajectory.
Both arms used the same whole-seed five-fold split, HGB parameters, raw
450-second label, zero decision threshold and no-stay veto. The only model
difference was the direct addition of 40 physical midpoint features
`0.5 * (candidate + PhasePressure reference)` to the existing 23 state and 46
action-difference features.

The delta arm reproduced the V158 target-native cost for every seed with zero
numeric difference. This confirms that the input bank, fold assignment and
baseline execution remained unchanged.

## OOF result

Lower cost is better.

| Arm | Mean 450-second cost |
|---|---:|
| delta target | 1.02511667 |
| endpoint target | 1.02451019 |
| PhasePressure | 1.02385540 |

Endpoint minus delta was `-0.00060648`, with paired bootstrap 95% interval
`[-0.00223628, +0.00081223]` and 5/10 improving seeds. It passed the frozen mean
improvement requirement of `-0.0005`, but failed the negative bootstrap upper
bound and 7/10 seed-win requirements.

The demand-level endpoint-minus-delta differences were:

- base: `-0.00201119`;
- `_2000`: `+0.00157150`;
- `_2500`: `-0.00091152`.

The `_2000` regression fails the requirement that none of the three fixed
demands worsen. Endpoint target also remained `+0.00065478` above
PhasePressure, with only 3/10 seeds improving and a bootstrap interval of
`[-0.00152407, +0.00254359]`; this comparison was diagnostic and was not part
of the endpoint-versus-delta gate.

## Interpretation and decision

Some descriptive ranking diagnostics moved in the favorable direction.
Candidate benefit-sign accuracy rose from 0.4857 to 0.5071, pairwise ordering
accuracy rose from 0.5113 to 0.5288, predicted/realized candidate-advantage
correlation moved from -0.0569 to +0.0266, and total harmful-override cost fell
from 0.42289 to 0.33094. Other diagnostics did not improve: total realized gain
fell from 0.29676 to 0.26546, mean missed-benefit regret rose from 0.01356 to
0.01388, and oracle top-1 selection remained 16/100.

The descriptive improvement was not stable enough to support the next stage.
Harmful overrides remained 26, the seed split was 5/5, one demand worsened,
and endpoint target was worse than PhasePressure in seven seeds. These results
reject direct physical endpoints as a sufficient repair for the V158
action-ranking failure under the fixed B100 and HGB design. No threshold,
feature set, fold, seed or demand may be changed to rescue V159.
History-conditioned inputs or repeated-future labels would require a new
research protocol rather than another V159 adjustment.

The compact authoritative artifact is
`cf_h2o/results/paper_artifacts/tsc_v159_native_endpoint_ablation_v1.json`.
