# V125 Source-Intervention Feasibility Diagnostic

## Purpose

V124 correctly returned exact PhasePressure fallback, but its result alone does
not distinguish a selector with no control headroom from a conservative gate
that failed to identify existing headroom. V125 is a diagnostic, not a new
efficacy test.

## Frozen inputs

V125 reads the same immutable V123 prediction artifact and 22-seed Jinan
450-second halted-queue selector used by V124. It refits no model and changes no
V124 decision threshold.

## Questions

1. What is the per-group oracle improvement over PhasePressure when all eight
   candidate actions are available?
2. How much oracle headroom remains if an action may not reduce instantaneous
   service pressure relative to PhasePressure?
3. What held-out value would the V124 source guard have produced before its
   paired source-over-target authorization rule forced fallback?
4. Is the B500 lead robust to the paired one-sided significance requirement
   when the frozen `0.0005` minimum-effect requirement is removed only as a
   labelled post-hoc diagnostic?

## Boundary

The oracle uses unavailable counterfactual outcomes and is an upper bound, not
a policy. The significance-only analysis is explicitly post hoc and cannot
rescue V124 or authorize confirmation. Its only purpose is to decide whether a
new source-weighting architecture is technically justified.
