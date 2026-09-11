# V152A Hierarchical Mechanism Prior Protocol

## Question

V152A asks whether source cities provide a transferable coefficient prior for
one causal mechanism block after the V151A state-utility gate admitted no
source contribution. The immutable target-only rigid CFCMT model remains the
non-degradation baseline. This experiment does not reinterpret V151A or tune a
weaker threshold after observing its result.

## Data Roles

For each evaluated target, the other six cities are pseudo-target domains.
Each pseudo-target uses the fixed B100 adaptation subset and evaluation reserve
already defined for the seven-city bank. The evaluated target is excluded from
all pseudo-target labels, candidate source pools, source coefficient estimates,
and feature scaling statistics. Consequently, target evaluation labels cannot
select a mechanism block or prior strength.

The evaluated target contributes B100 labels to its rigid model and target-only
mechanism fit. V152A is therefore target offline adaptation, not zero-shot
control.

## Prior And Integration

Each source city fits the same reference-centred causal mechanism design with
equal action-group weighting. Source cities then receive equal weight: the
prior centre is the coordinatewise median of their coefficients. A
block-normalized inverse-MAD reliability determines coordinatewise prior
precision, clipped to `[0.25, 2.0]`.

The fixed candidate set is the Cartesian product of six mechanism blocks and
prior strengths `[0.005, 0.01, 0.025, 0.05, 0.1]`. For one candidate, only its
mechanism block is shrunk towards the source prior. Its score is

`rigid score + prior-assisted mechanism score - target-only mechanism score`.

This preserves the rigid architecture and injects only the source-induced
coefficient change.

## Controls And Selection

The matched placebo repeats the full pipeline with action-group targets
permuted using a seed fixed by source-city identity. The zero-centred control
uses the same source-derived precision but a zero prior centre. A source
candidate must improve all three comparators by at least `0.0005` on average,
be non-degrading in at least five of six pseudo-target cities for every
comparison, and alter at least six action groups. The eligible candidate with
the largest worst comparator mean gain is selected. With no eligible candidate,
the method returns rigid CFCMT exactly.

## Development Gate

The seven-target aggregate requires at least two admissions, no target-city
regression, improvement over rigid and both controls for every admitted target,
and exact fallback for every rejected target. Passing authorizes only a
closed-loop development experiment. It does not establish untouched-city or
real-world causal efficacy.
