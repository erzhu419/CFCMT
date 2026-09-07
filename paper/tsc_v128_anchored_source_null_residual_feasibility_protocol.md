# V128 Anchored Source-Null Residual Feasibility Protocol

## Motivation

V127 rejected the source-as-an-extra-feature construction. Target-only,
source-aligned and source-placebo produced the same calibrated held-out policy,
and only one of 22 folds enabled any intervention. The source channel was
therefore neither identifiable nor operationally useful even with abundant
target counterfactual labels.

V128 changes the role of transfer information rather than adding another dense
module. The frozen V123 B500 target prediction is the target-only offset. The
frozen `Atlanta, weight 0.75` B500 prediction is the source-aligned offset. A
causal reference-contrast HGB learns only the target residual around each
offset; the offset is not supplied as a feature and therefore cannot be
silently ignored. The placebo permutes complete incremental source action
vectors within each seed before reference centering.

## Frozen split and source-null rule

Each of 22 outer folds holds one selector seed out. Ten other seeds calibrate
both intervention retention and the source-versus-target choice; the remaining
11 seeds fit the residual models. The held-out seed is used for neither role.

Each model may choose only an action whose instantaneous service pressure does
not decrease relative to exact PhasePressure. Calibration tests retention
fractions from 0.5% through 20%. A profile is enabled only with at least 40
interventions, mean normalized gain of at least 0.0005, a negative one-sided
95% upper bound across ten seeds and improvement in at least eight seeds.

The aligned source offset is deployed only if its calibrated policy also beats
the independently fitted target-only profile by the same mean, upper-bound and
replication requirements. Otherwise the fold uses the target-only profile; if
that profile is disabled, execution is exact PhasePressure. The placebo receives
the identical source-null selection procedure so selection capacity is matched.

## Development gate

The adaptive source-null policy must beat PhasePressure, target-only and the
adaptive placebo-null policy by at least 0.0005 mean normalized cost, with all
paired bootstrap upper 95% bounds below zero. Thresholds and splits are frozen
before V128 is run.

## Information and claim boundary

V128 reuses a B500 upstream target prior and consumes 11 complete target
selector seed banks for residual fitting plus ten disjoint banks for source-null
calibration in each fold. It remains an abundant-information development
diagnostic. Passing can authorize a successor that enforces the total target
group budget across prior fitting, residual fitting, source selection and
safety calibration. V128 itself is neither few-shot adaptation nor an
untouched-city confirmation.
