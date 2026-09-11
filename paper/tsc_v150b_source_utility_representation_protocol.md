# V150B Source-Utility Representation Diagnostic

## Question

V150A establishes that forced source transfer has repeatable average benefit,
while a deployable source choice remains unstable. V150B asks whether missing,
decision-time observable traffic-signal structure explains source-utility
ranking better than the existing city covariates.

## Frozen comparison

The outcome is the frozen V144 source-arm cost minus the same-architecture
target-only cost for all 42 directed source-target pairs. Lower is better.
No V144 evaluation outcome is used to construct the new representation.

The existing representation uses a fixed subset of demand, signal-layout,
route, local phase, and graph covariates already present in V144. The enriched
representation adds:

- protected versus permissive candidate service;
- shared receiving-lane and uncontrolled-major merge exposure;
- same-time observed neighbor phase type, elapsed green, switching, and
  clearance state;
- explicit neighbor-state observation coverage.

Signature extraction reads only candidate features, network topology, and row
metadata from the frozen V114 admitted cache. It does not read prior or outcome
arrays. A scenario declared by the broader source manifest but absent from the
admitted cache is reported and excluded; it is not silently regenerated.

## Exclusion and placebo

For each outer fold, the held-out city is absent from training both as a target
and as a source. Ridge strength is selected by nested city exclusion among the
remaining six cities. Six non-identity cyclic permutations of the new city
signature are matched placebos.

## Promotion rule

The enriched representation advances only if all preregistered checks pass:

1. pair MAE is lower than with existing covariates;
2. mean held-out source-rank Spearman improves by at least 0.10;
3. that rank score exceeds the cyclic-placebo median by at least 0.10;
4. null-aware selected transfer improves mean cost by at least 0.0005;
5. null-aware selection improves at least five of seven cities.

Failure closes this representation candidate. Passing authorizes development
of a mechanism-specific source prior; it does not establish closed-loop or
fresh-city efficacy.
