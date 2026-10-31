# CFCMT Research Checkpoint Continuation (2026-09-07)

This document continues `paper/CFCMT_RESEARCH_CHECKPOINT_20260901.md` without
rewriting its historical record.

## Completed method decisions

- V144 passed its descriptive source-headroom criterion. Across all 42 ordered
  source-target pairs, every target had at least one useful source; the pairwise
  improving fraction was `0.595238` and the mean one-source effect versus
  target-only was `-0.00037944`. Source identities from this oracle inventory
  remain nondeployable.
- V145 was rejected. It passed pooled H2O+ and matched-placebo comparisons but
  failed target-only safety: mean `-0.00282806`, one-sided 95% upper bound
  `+0.00929308`, 4/7 improving cities and maximum city regression `+0.02810353`.
- V146 was rejected. It passed pooled H2O+ but failed target-only and matched-
  placebo gates. Versus target-only its mean was `-0.00161194`, one-sided upper
  bound `+0.00950967`, 3/7 improving cities and maximum regression
  `+0.02622154`. V147 is therefore closed and was not launched.
- V143--V146 use the rigid causal action rankers in their deployed paths. They
  do not constitute evidence for a full multi-mechanism MC-WM controller.
- V148 evaluated fixed-parent, rank-0 physical mechanism compatibility using
  five B20-to-B5 labeled adaptation folds. Evaluation groups remained excluded,
  and the mechanism diagnostics only shrank source residual mass; the V143
  rigid ranker still selected actions. Corrected tasks `t89229` and
  `t89263`--`t89268` all completed. The earlier Atlanta engineering failure
  `t89226` produced no scientific result.
- V148 was rejected. The robust leave-one-target-and-source history gate
  admitted no source in any city, so all seven targets returned exact target-
  only. Effects versus target-only and matched placebo were exactly zero. The
  target-only fallback still improved pooled H2O+ in 7/7 cities (mean
  `-0.06914109`, one-sided upper bound `-0.02983742`), but this is not source-
  transfer or full MC-WM evidence. Canonical result:
  `cf_h2o/results/cluster/tsc_v148_multicity_mechanism_compatibility_gate_20260901/development_v2_corrected/aggregate.json`.
- V149 is not authorized and was not launched. It cannot be revived by a later
  Boston admission result or by substituting another city.

## Boston v11 admission

- Boston package v10 crossed the earlier trigger failures but collided at time
  11,871 s. Live topology evidence identified a protected TLS movement sharing
  a receiving lane with an uncontrolled major movement. Package v11 changes
  only phase 12, link 10 of `joinedS_1100` from `G` to yielding green `g`.
- Boston v11 passed all 21 static gates and complete route admission for
  3,806,510 trips. Trigger task `t89216` reached 12,601 s with zero collisions
  and zero teleports, crossing all failures at 9,518, 9,735, 11,304, 11,504,
  11,871 and 12,279 s.
- Full-day admission task `t89321` is running from immutable snapshot
  `4187c3ab21a61280e249`. A submitted or partial task does not authorize
  controller evaluation. A passing result will close package integrity only;
  it will not authorize V149.

## Current decision boundary

1. Wait for `t89321`; read its log first and then only the small result JSON.
2. Close V145, V146 and V148 without threshold retuning. Their combined evidence
   supports rigid target adaptation over pooled dense H2O+-style residuals, but
   not a reliable positive contribution from source-city labels.
3. Keep the TSC manuscript framed around structured matched-action adaptation
   and network-dependent negative transfer. Any future source-selection method
   requires a new precommitted family and genuinely new city units; V148 cities
   can be development evidence but not confirmation evidence for that method.
