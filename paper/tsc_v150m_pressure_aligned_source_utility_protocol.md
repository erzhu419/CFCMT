# V150M Pressure-Aligned Source Utility Protocol

## Question

Can source-mechanism information provide incremental value over the same rigid
CFCMT architecture when source corrections are restricted to a
deployment-observable, pressure-aligned action set?

## Method

For every action group, let `a_R` be the rigid target-only action and `a_P` be
the PhasePressure reference. Each source-mechanism candidate is projected before
utility-record construction:

1. retain the candidate when it selects `a_R`;
2. retain the candidate when it changes `a_R` to `a_P`;
3. replace its complete group score by the exact rigid score otherwise.

The source and same-capacity placebo pipelines use the identical projection,
five-fold nested utility fitting, target budget B100, source-prior strength
0.025, and admission thresholds inherited from V150K. The held-out evaluation
reserve remains outside candidate fitting, utility fitting, and admission.

## Gates

Offline development passes only if at least two cities admit source, no city
regresses after fallback, and every admitted city improves both rigid CFCMT and
its separately nested matched placebo. Rejected cities must reproduce rigid
CFCMT exactly.

Only an offline pass may produce runtime models. Closed-loop development then
uses fresh SUMO seeds and four matched arms: PhasePressure, rigid target-only,
pressure-aligned source, and pressure-aligned placebo. Operational admission
requires complete rollouts, zero teleports, paired collision-incident
non-inferiority, and exact fallback. Performance is aggregated as paired
relative effects within scenario-seed and then equally across cities; raw
seconds are reported but do not define the city-macro effect.

## Claim Boundary

This remains seven-city method development. A pass can freeze V150M for a
separate untouched-city confirmation; it cannot itself establish fresh-city or
real-world intervention efficacy.
