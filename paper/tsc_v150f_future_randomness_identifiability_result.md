# TSC V150F Fixed-State Future-Randomness Result

## Decision

V150F completed as a partial development diagnostic. RNG state was absent from
all 21 snapshots, and every same-seed replay was exact (`0.0` maximum absolute
difference). Across 105 requested fixed-state/future-seed action pairs, 93 were
valid. The integrity gate did not pass because two Cologne checkpoints retained
only two valid futures and one Hangzhou checkpoint retained none; the prespecified
minimum was three per checkpoint.

The invalid branches cannot be attributed only to a deliberately poor alternative.
Cologne had three reference and three alternative failures; Hangzhou had five
reference and one alternative failure. All were rejected SUMO collision rollouts.

## Variance Diagnosis

Among valid branches, changing only the future seed changed the paired action-cost
label in all seven cities. However, future randomness was not the dominant source
of heterogeneity in this pilot: the city-macro mean within-state variance fraction
was `0.00905`, and only two of 20 analyzable checkpoints changed the sign of the
action utility across future seeds. Both sign-unstable checkpoints were in New
York and had near-zero mean utility. Seventeen of 21 checkpoints retained all
five future pairs.

The result therefore narrows the diagnosis:

1. historical exact replay does not provide independent future replicates;
2. independent future noise is real and should be averaged near the decision
   boundary;
3. most source-selection instability is more consistent with state heterogeneity,
   B25 sampling and selection among many candidates than with future randomness
   alone;
4. Cologne and Hangzhou also require explicit handling of future-dependent
   baseline collision risk before their fixed-state labels can be called complete.

## Consequence

Another selector on the same 25 pooled OOF labels is not justified. The next
low-complexity test must cross-fit the complete candidate-selection procedure,
so each held-out B25 fold evaluates a source/mechanism candidate chosen without
that fold. Future averaging should be added for near-zero or safety-sensitive
labels, not assumed to solve the much larger between-state variation.

## Evidence

- Aggregate: `cf_h2o/results/paper_artifacts/tsc_v150f_future_randomness_identifiability_v3.json`
- Per-city results: `cf_h2o/results/cluster/tsc_v150_source_identifiability_20260908/future_randomness_identifiability_v2/`
- Frozen protocol: `paper/tsc_v150f_future_randomness_identifiability_protocol.md`
