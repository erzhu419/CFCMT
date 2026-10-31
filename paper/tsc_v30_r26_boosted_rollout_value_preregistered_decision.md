# TSC v30/r26 Parent-Restricted Nonlinear Rollout Value: Preregistered Decision

Frozen before any v30 target result. Salt Lake remains sealed.

## v29 Diagnosis

The zero-prior linear terminal mechanism improved macro regret by 2.24% and
three cities, confirming that the fixed-policy terminal estimand contains
transferable signal. Its invariant Ridge head nevertheless leaves a large gap
to the observed terminal-value ranking ceiling and fails the 10% gate. This
round tests model capacity without relaxing the causal parent restriction.

## Frozen Candidate

Evaluate exactly one family:

`cfcmt_boosted_direct_rollout_value_rigid_residual`

It is identical to v29 except that the single terminal-clearance mechanism uses
the existing causal `HistGradientBoostingRegressor` backend. The backend sees
only a source-selected registered parent variant; it never receives all local
features, city/network ID, or target context as dense inputs. Parent selection
uses source-city OOF terminal action-ranking error. Hyperparameters are the
existing frozen `BoostedFitConfig` defaults: learning rate 0.06, 40 iterations,
15 maximum leaves, 20 minimum samples per leaf, L2 5.0, winsor quantile 0.01,
and random seed 20260803.

The terminal target remains domain-normalized with a zero simulator prior. The
outer stack remains `rigid_anchored_mechanism_residual_v1`, with source-only
nested city cross-fitting, five source domains, and ten pair-excluded rigid and
mechanism fits. No target adapter, expert gate, local mechanism combination,
or direct `interval_cost` mechanism is permitted.

## Frozen Gate

Against the immutable v26 rigid reference on the same six development cities,
promote only with at least 10% macro regret improvement, at least four cities
improved, maximum city regression at most 0.05, and complete fail-closed
provenance/holdout/zero-target-label audit.

If this fails, the next scientific decision is not a larger dense network. It
is to quantify the target-value predictability ceiling under source-only
features and decide whether a multiscale local-plus-value composition is
justified. Salt Lake remains untouched until a development family passes.

## Claim Boundary

Passing would support nonlinear invariant policy-value transfer under a fixed
continuation policy. It would not establish a general transition world model
or an uncalibrated-simulator benefit.
