# Round-2 Reviewer Report: Experimental Readiness After Revision

Reviewer stance: I review the revised manuscript as a primary target for *Transportation Research Part C* and a secondary target for IEEE ITS / IEEE T-ITS. This round focuses on whether the new experiments sufficiently address the first-round experimental concerns.

## Overall Recommendation

**Transportation Research Part C: major revision, but now credible if framed as an open-data cross-city transfer benchmark.**

The revision is substantially stronger. The new calibration-budget sweep, per-mechanism decomposition, mechanism/capacity ablations, source-diversity analysis, route-bootstrap intervals, target-construction audit, and expanded seeded rollout directly address many predictable reviewer objections. The paper is no longer just a single cross-city result; it now has a defensible experimental story.

However, I would still not recommend acceptance at Part C if the paper is framed as a field-ready bus holding control method. The central evidence remains static-data-derived and pseudo-operational. The closed-loop rollout reaches parity with the dense policy, not a clear operational advantage. The strongest publishable claim is therefore:

> Under reproducible open static-data-derived cross-city validation, CFCMT improves calibration-light multi-source residual transfer relative to a dense H2O+-style residual baseline.

**IEEE ITS / IEEE T-ITS: borderline major revision to weak accept, depending on framing and baseline precision.**

For an IEEE ITS audience, the revised experimental package is closer to sufficient because the algorithmic comparison, ablation coverage, and reproducibility are much stronger. The remaining risk is whether reviewers accept the H2O+-style baseline as a fair H2O+ comparison and whether the paper avoids overclaiming policy performance.

## What The Revision Fixed Well

1. **Calibration-budget evidence is now meaningful.** The route-level target budget sweep is much better than the earlier synthetic calibration-strength argument. The result that uncalibrated weighted CFCMT stays below the calibrated dense baseline through 25% target-route budgets on three of four targets is a strong point.

2. **Mechanism-level evidence is now visible.** The per-mechanism table shows the gain is not only a total-MSE artifact. Weighted CFCMT improves demand/stop/reward, headway/gap, and speed on average, with the largest improvement in headway/gap.

3. **Ablations now prevent the easiest causal-factorization objection.** Random mechanism grouping and action/time-only variants fail badly, which supports the claim that the chosen parent sets matter. The dense matched-sparse result is also useful because it forces the paper to make a narrower and more honest claim.

4. **Source diversity is now a result, not just a caveat.** The source-size and subset analyses show that multi-source transfer is the real regime where CFCMT works best. Single-source transfer is mixed, and the Halifax-MBTA pair is worse than H2O+. This is exactly the kind of boundary condition reviewers trust.

5. **The sampled rollout was handled correctly.** The revision did not hide the weak rollout evidence. It fixed the evaluation alignment issue, reseeded the stochastic components fairly, and then correctly described rollout as auxiliary because the margin is tiny and headway error remains slightly worse.

6. **The target-construction audit improves credibility.** Separating observed, apportioned, schedule-proxy, and deterministic components makes the benchmark much more transparent.

## Major Remaining Concerns

### 1. The paper still lacks true operational ground truth

This remains the largest Part C weakness. Singapore has the strongest observed evidence, but Austin, Halifax, and MBTA still rely heavily on schedule-derived or apportioned quantities. The paper can be accepted as a static-derived benchmark contribution only if it keeps this limitation central.

Required before a strong Part C submission:

- Add at least one true AVL/APC validation slice if any public or partner data can be obtained.
- If true trajectories are impossible, explicitly reposition the paper as an open-data benchmark and remove any wording that implies real-world policy superiority.
- In the target-construction section, add a short table column for "observed", "apportioned", "schedule proxy", and "deterministic simulator rule" per output group, not only per city.

### 2. The H2O+ comparison is still vulnerable

The manuscript calls the baseline "dense H2O+-style residual." That is safer than calling it H2O+, but the paper's motivating claim still compares against H2O+. A reviewer familiar with H2O+ may ask whether the baseline reproduces the original offline-to-online objective, discriminator, calibration procedure, and policy learning loop.

Required revision:

- Keep "H2O+-style" everywhere unless the exact H2O+ training stack is reproduced.
- Add a baseline-implementation paragraph listing what is inherited from H2O+ and what is not.
- Add a small table: "baseline component" vs "implemented in this paper" vs "difference from original H2O+".
- Avoid a headline claim like "CFCMT outperforms H2O+"; use "outperforms a dense H2O+-style residual transfer baseline."

### 3. The policy/control evidence is still secondary

The dynamics evidence is strong, but Part C reviewers will care about passenger-facing control outcomes. The one-step policy proxy improves several metrics, but the executable rollout is essentially parity: weighted CFCMT reward is only slightly better than H2O+-style, while headway error is slightly worse.

Required revision:

- Keep the primary contribution as world-model transfer, not closed-loop bus holding control.
- Report rollout confidence intervals across route seeds, not only mean values.
- Add passenger-facing rollout metrics if available: waiting time, excess waiting time, in-vehicle delay from holding, headway CV, bunching frequency, and hold-time distribution.
- If possible, add at least threshold holding and schedule/headway hybrid baselines, not only no-hold/fixed/dense-policy.

### 4. Four target cities limit the generalization claim

The strict city-level protocol is correct, but four held-out cities are still few. Route-level bootstrap helps, but it does not create new independent city-level evidence. This limitation is manageable, but the language must remain scoped.

Required revision:

- Avoid "generalizes across cities" without qualifiers.
- Use "across four open-data city bundles" or "in the evaluated open-data city set."
- Report city-level counts as the primary unit: 4/4 strict targets, 3/4 for several robustness cases.
- Add more cities if the data pipeline can support them; even two additional static-derived cities would significantly improve credibility.

### 5. The causal mechanism claim should be narrowed

The ablations support mechanism-wise sparsity, but they do not prove causal identification. Dense matched-sparse features are close to full CFCMT, which means some of the advantage may come from feature pruning and regularization rather than causal invariance.

Required revision:

- Prefer "mechanism-factored" or "mechanism-wise sparse" in experimental sections.
- Use "causal" mainly as motivation and design hypothesis unless stronger invariance or intervention evidence is added.
- Add one sentence in Results explicitly stating that matched-capacity dense sparsification remains a competitive explanation.

### 6. The calibration-budget result needs per-target reporting in the main text

The mean calibration-budget curve is useful, but the result drops from 4/4 wins at 0% to 3/4 wins once target calibration is introduced. Reviewers will ask which city fails and why.

Required revision:

- Add a compact main-text table or sentence identifying the failing target at each calibration budget.
- Explain whether the failure is Singapore, and whether it is caused by weighting, observed traffic mismatch, or target-route heterogeneity.
- Report both unweighted and weighted CFCMT on the calibration-budget curve if space permits.

## Recommended Experimental Additions

Priority order for the next revision:

1. **One real trajectory slice.** Even a small AVL/APC validation set for one city would change the Part C credibility level more than any additional pseudo-static experiment.

2. **Rollout uncertainty and passenger metrics.** Repeat executable rollout over more route seeds and include passenger-facing metrics. The current rollout is useful but too weak to carry a control paper.

3. **Faithful H2O+ baseline note or rename.** Either implement the full original H2O+ training/evaluation path or explicitly demote the baseline name everywhere.

4. **Per-target calibration-budget analysis.** Identify which target loses at 1%, 5%, 10%, and 25% target calibration.

5. **More cities if cheap.** Add two to four additional public-data city bundles, even if they are schedule-derived, and label evidence levels honestly.

6. **Normalized error reporting.** Total MSE can be scale-dominated. Add normalized RMSE or per-output standardized error in the appendix.

## Claim Guidance

Acceptable:

> CFCMT improves static-data-derived cross-city residual transfer over a dense H2O+-style baseline on all four strict held-out city targets, with the strongest evidence in multi-source transfer.

Acceptable:

> The calibration-budget sweep suggests that mechanism-factored transfer can remain competitive with dense residual correction even when the dense baseline receives a small amount of target-route calibration.

Not acceptable yet:

> CFCMT is better than H2O+ for real-world bus holding control.

Not acceptable yet:

> CFCMT achieves field-ready zero-calibration transfer.

Not acceptable yet:

> The experiments prove causal mechanism invariance.

## Likely Reviewer Questions

1. Are the target labels independent enough from the deterministic generator to avoid evaluating generator artifacts?
2. Is the H2O+-style baseline faithful enough to support comparisons against H2O+?
3. Why does source weighting weaken Singapore relative to unweighted CFCMT?
4. Which target fails in the calibration-budget sweep once target data is added to the dense baseline?
5. Why should a Part C reader care about one-step policy proxy if closed-loop rollout only reaches parity?
6. Are total-MSE improvements dominated by headway/gap outputs?
7. Would the result survive on true AVL/APC trajectories?
8. How many independent city-level test units are enough to support the generalization language?

## Bottom Line

The revised paper is much closer to a credible submission. The experimental package now supports a solid benchmark-style claim: **CFCMT is a stronger calibration-light multi-source cross-city residual transfer method than a dense H2O+-style baseline under the constructed open-data validation protocol.**

For Part C, the paper still needs either true operational validation or very careful repositioning as a reproducible open-data benchmark. For IEEE ITS, the revised version is close, provided the H2O+ baseline naming is disciplined and the policy/control claims stay secondary.
