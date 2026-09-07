# TSC V133 Historical Action Identifiability Result

## Decision

`REJECT`. Exact schedule/TLS/action history does not identify the 450-second
counterfactual effect across stochastic simulator realizations. Another direct
450-second action regressor is not justified on this evidence.

## Evidence

- Scheduler task: `t88958` on `node005`
- Runtime: 168.05 s
- Immutable snapshot: `33843e6ae577f04961dc`
- Result:
  `cf_h2o/results/cluster/tsc_v133_historical_action_identifiability_20260901/development_v1/result.json`
- Result SHA-256:
  `ea6227aae879599e9ce14a3914797dbdae4115db1bccdb7797585e1bcfd2595d`

Only 4,718 of 19,851 outer held-out groups had any exact historical
scenario/interval/TLS/candidate-state match. Just 91 groups had pressure-safe
support in at least 17 of the 21 development seeds. The liberal historical
mean lookup intervened 29 times across 18 folds and was harmful on held-out
data: mean normalized delta `+0.0002466`, 95% interval
`[-0.0000460, +0.0005576]`. The conservative 80%-agreement arm authorized no
intervention.

## Consequence

The V125 oracle headroom cannot be converted into a policy by memorizing the
same schedule position and action across target simulations. The next screen
must move below the noisy long-horizon cost: V134 will test whether observed
one-control-interval queue, served-movement, red-accumulation, spillback or
mobility changes are valid surrogate objectives for the 450-second waiting
effect. A causal mechanism model is authorized only for a surrogate whose own
held-out oracle passes a familywise absolute gate.
