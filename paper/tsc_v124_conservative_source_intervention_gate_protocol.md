# V124 Conservative Source-Intervention Gate Protocol

## Motivation

V123 confirmed that source-city mechanisms improve the architecture-matched
target model at five of six positive target budgets, but every raw
source-augmented action selector remained worse than PhasePressure. V124 tests
the resulting mechanistic diagnosis: source information is useful, while the
unrestricted residual argmin converts prediction error into harmful control.

## Frozen development design

V124 consumes the immutable 209,690,586-byte V123 prediction artifact with
SHA-256
`3f199276f08ab67e855d9598900d4023ec1f7c4c3529b4361e92b0e851bf983a`.
It does not refit a world model or regenerate labels. Evaluation uses the same
22-seed Jinan selector bank and the same B0--B1000 target-budget curve.

For every held-out selector seed, the other 21 seeds choose the source city,
source weight and guard profile. The held-out seed cannot influence any of
those choices. Target-only and source-augmented predictions use the same guard
family. A guard may intervene only on actions with sufficiently large robust
predicted advantage, sufficient trust, bounded pressure loss, and optionally
agreement with the target model.

## Negative-transfer control

The source arm is authorized within a fold only when its training-seed
one-sided 95% upper bound is below PhasePressure and its paired improvement over
the architecture-matched target guard is at least 0.0005 with a negative
one-sided upper bound. Otherwise deployment falls back to the target guard; if
that guard is not admissible, it falls back to exact PhasePressure.

## Decision rule

The development gate passes only if one positive target budget has both:

1. source-policy mean at most -0.0005 versus PhasePressure with a negative 95%
   upper bound; and
2. source-minus-target mean at most -0.0005 with a negative 95% upper bound.

Budget selection is development-only. Passing authorizes exactly one frozen
fresh-city confirmation. Failing retains PhasePressure or the target-only guard
and does not support a deployable source-transfer claim.
