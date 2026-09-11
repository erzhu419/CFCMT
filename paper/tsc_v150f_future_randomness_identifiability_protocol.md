# TSC V150F Fixed-State Future-Randomness Protocol

## Question

V150E showed that positive B25 OOF source and placebo margins do not reliably
predict untouched action cost. V150F asks whether the label itself varies when
the observed decision state and candidate action are fixed but the subsequent
SUMO random stream changes.

## Design

One prespecified scenario is used for each of the seven development-city groups.
An uninterrupted PhasePressure trace supplies three checkpoints after a 300 s
warm-up, spaced by 180 s. At each checkpoint, the tested pair is PhasePressure
versus a deterministic feasible non-reference phase with the nearest green-link
count. Both actions are evaluated for 90 s under each of five future seeds.

The state is saved with `save-state.rng=false`. Each branch performs a full
network reload using the same physical state and a newly specified future seed.
Actions within a pair share that future seed. The first reference branch at every
checkpoint is repeated under the same seed and must reproduce exactly. The saved
XML is audited to ensure that it contains no `rngState` element. Every
checkpoint must retain at least three valid future-seed action pairs; otherwise
the seven-city integrity result is partial rather than passed.

This protocol uses `libsumo`; it does not use TraCI. It is separate from the
historical counterfactual caches, whose `save-state.rng=true` setting correctly
provides exact matched replay but supplies no independent future replicas for a
fixed state.

## Outputs

For every fixed state, V150F reports the distribution of

`reference cumulative cost - alternative cumulative cost`,

its sign agreement, outcome and cost ranges across future seeds, and a
within-fixed-state versus between-state variance decomposition. The integrity
gate concerns only RNG omission, same-seed replay and valid paired outcomes.

V150F is a development diagnostic. It does not select a source, establish source
benefit, or authorize closed-loop source transfer.
