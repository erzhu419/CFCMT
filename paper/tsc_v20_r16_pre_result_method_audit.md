# TSC v20/r16 pre-result method audit

Recorded at 2026-08-08T19:26:53+08:00, before any result payload from the
`tsc_v20r16c_subset_safety_dev600_20260808` development matrix existed.

## Finding

The frozen fused CFCMT policy is compared with a target-only specialist, but
that comparison does not isolate the mechanism stack. The fused policy contains
three distinguishable components:

1. a rigid source action-advantage core;
2. a nested cross-fitted source mechanism stack;
3. a group-cross-fitted target specialist.

A gain over the target-only specialist can therefore be caused by the rigid
source core even when every source mechanism-stack weight is zero. The current
r16 acceptance test is valid for the fused controller as a whole, but it is not
sufficient evidence for a mechanism-fusion claim.

## Required exact ablation

Before a submission claim is accepted, add `cfcmt_fused_rigid`, which must use
the same source data, rigid core, target specialist, target group splits,
specialist weight grid, uncertainty calibration, pressure prior, guard,
coordination graph, and evaluation seeds as `cfcmt_fused`. Its only change is
that the source mechanism-stack weight is fixed to zero.

The mechanism claim requires all of the following on a fresh frozen matrix:

- the experiment-integrity audit passes;
- fused CFCMT passes the broad city-level efficacy gates against the selected
  source pressure prior;
- fused CFCMT improves over both `cfcmt_fused_rigid` and target-only control by
  a predeclared material margin on at least one intermediate or high budget;
- at least one qualifying target uses both a positive source mechanism-stack
  weight and a non-degenerate target-specialist weight;
- the primary policy remains collision-incident non-inferior to its selected
  source pressure prior and has zero teleports.

The running r16 matrix remains useful for diagnosing whether the source
mechanism stack and target specialist are ever selected. It is not sufficient
for the final causal-mechanism contribution claim, regardless of aggregate
performance.
