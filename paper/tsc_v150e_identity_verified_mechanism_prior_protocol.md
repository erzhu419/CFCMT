# TSC V150E Identity-Verified Mechanism-Prior Protocol

## Motivation

V150D was rejected because its B25 selector admitted source priors whose apparent
gain was reproduced by the same source/mechanism permutation placebo. The most
damaging case was Ingolstadt: source and placebo had identical OOF cost, yet the
source arm regressed rigid CFCMT on untouched evaluation groups.

## Frozen redevelopment

V150E retains the V150D model, features, target budget, five-fold split, source
priors, prior strength, rigid score, and seven development cities. It changes
only source admission. A source/mechanism candidate is eligible when both hold:

1. its B25 out-of-fold policy cost improves rigid target-only by at least
   `0.0005`;
2. its B25 out-of-fold policy cost improves the same-source, same-mechanism,
   same-strength permutation placebo by at least `0.0005`.

If no candidate is eligible, the deployed score is bitwise-identical to rigid
CFCMT. The matched-placebo arm follows the same admission decision.

## Development gate

The city is the inferential unit. This stage passes only if at least two cities
admit an identity-verified source, no city regresses rigid CFCMT, every admitted
city improves rigid and its matched placebo, every rejected city exactly falls
back, and the seven-city mean improves both comparators.

This protocol was written after inspecting the rejected V150D aggregate. The
evaluation groups remain label-isolated, but all seven cities are development
data. A pass can authorize closed-loop redevelopment only; it cannot establish
confirmation or external validity.
