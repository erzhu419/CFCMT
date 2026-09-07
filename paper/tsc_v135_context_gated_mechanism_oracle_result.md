# TSC V135 Context-Gated Mechanism Oracle Result

## Decision

`REJECT`. The nested pre-action context gate did not make the V134 balanced
one-step oracle stable across selector seeds. Per the frozen protocol, the
one-step mechanism-control branch is closed and V136 is not authorized.

## Evidence

- Scheduler task: `t88967` on `node003`
- Runtime: 302.26 s
- Immutable snapshot: `4b22e42cd9d0c99210ce`
- Result:
  `cf_h2o/results/cluster/tsc_v135_context_gated_mechanism_oracle_20260901/development_v1/result.json`
- Result size: 162,602 bytes
- Result SHA-256:
  `05359a3283d0bae821f6e691050a53cbe451fb7fd882510104a4e31524b6e372`

The gate authorized intervention in 9/22 held-out folds and made 76 total
interventions. Its mean normalized 450-second waiting effect was `-0.0002592`,
with bootstrap 95% interval `[-0.0005943, +0.0000338]`. Only 6/22 held-out
seeds improved, an improving-seed fraction of 27.3%, versus the frozen 80%
requirement.

All nine selected inner profiles used risk multiplier zero. Six selected the
10% base oracle and three selected the 5% base oracle. The OOF-error-penalized
profiles were never authorized, so the apparent effect was not robust to the
predeclared context-model uncertainty. Among the nine active held-out folds,
three were harmful despite strongly negative calibration means.

## Interpretation

V134 established that the balanced realized one-step proxy contains an
aggregate long-horizon signal. V135 now shows that its cross-realization
failures are not reliably separable by the available pre-action local state,
deterministic graph and candidate-action parents. A learned one-step mechanism
model would replace the V135 oracle proposal with a noisier estimate while
retaining this unresolved context problem. Fitting it is therefore not a
scientifically justified next step.

This result does not invalidate the broader CFCMT transfer layer. It narrows
the defensible TSC claim: CFCMT can be evaluated as a calibration-light causal
residual/transfer framework relative to dense residual, simulator-only and
H2O+-style transfer, but the current evidence does not support a claim of
universal superiority over PhasePressure.
