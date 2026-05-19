# Pre-Submission Reviewer Report: Experimental Readiness for Transportation Research Part C / IEEE ITS

Reviewer stance: I review this as a primary target for *Transportation Research Part C* and a secondary target for IEEE ITS venues. The comments focus on experimental validity, claims, and evidence sufficiency rather than writing polish.

## Overall Recommendation

**Recommendation if submitted today to Transportation Research Part C: major revision / likely reject.**

The paper has a promising and unusually reproducible cross-city experimental protocol. The current evidence is strong enough to support a scoped claim about **static-data-derived cross-city model transfer**, but it is not yet strong enough for a high-confidence transportation systems claim about real bus operations or deployment without calibration.

For Part C, the main weakness is that the target labels are deterministic pseudo-observations built from heterogeneous public static data, not vehicle-level AVL/APC trajectories or calibrated operational simulation. For IEEE ITS, the paper is closer because algorithmic transfer and reproducible benchmarking matter more, but the method still needs stronger ablations, calibration-budget curves, and closed-loop policy evidence.

My suggested positioning is:

> CFCMT improves cross-city residual model transfer under open static-data-derived validation, and the advantage is strongest in multi-source transfer with diverse source evidence. It is a benchmarkable step toward calibration-light bus holding transfer, not yet a proof of field-ready zero-calibration control.

## Strengths

1. **Clear cross-city unit of evaluation.** The strict leave-one-city-out protocol correctly treats city as the evaluation unit rather than inflating evidence with many routes from the same city.

2. **Transparent claim boundaries.** The manuscript already states that the validation is static-derived and not real AVL/APC operational validation. This honesty is important and should remain.

3. **Useful robustness diagnostics.** Source-weighting sensitivity, generator perturbations, source-subset robustness, and policy-domain metrics are stronger than the typical single-table comparison.

4. **Promising main result.** Weighted CFCMT beats the dense H2O+-style baseline on all four strict held-out city targets, with mean total-MSE ratio 0.578. This is a meaningful signal.

5. **Negative evidence is reported.** The source-subset failure cases are not hidden: excluding Singapore makes the result nearly tied with H2O+ and the Halifax-MBTA pair is worse than H2O+. This makes the paper more credible.

## Major Concerns

### 1. The target labels are not real operational ground truth

The biggest Part C issue is that the paper evaluates against static-data-derived pseudo-observations. Singapore has observed passenger and traffic data, but Austin traffic/demand are schedule-derived, Halifax traffic is schedule-derived with route-level APC apportionment, and MBTA traffic is schedule-derived with stop-level ridership apportionment.

This is acceptable for a benchmark paper only if the manuscript makes the target construction central and auditable. It is not enough to say "offline data" or "open data"; the reviewer will ask whether CFCMT is learning artifacts of the generator.

Required revision:

- Add a dedicated "Target Construction and Validity" subsection.
- For each city, separate observed quantities, apportioned quantities, schedule proxies, and deterministic generator rules.
- Add a leakage check explaining why features used by CFCMT and H2O+ do not trivially reconstruct the target generator.
- Add route-level residual plots or summary statistics showing that target residuals are nontrivial and heterogeneous across cities.
- If possible, add at least one true AVL/APC validation slice, even for a small subset of routes. Without this, the claim must remain static-derived.

### 2. The baseline is H2O+-style, not necessarily H2O+

The current baseline is described as a dense H2O+-style residual model. A tough reviewer will challenge whether this is a fair representation of H2O+ rather than a convenient baseline inspired by it.

This matters because the expected headline is "CFCMT vs H2O+ cross-city validation." If the implementation does not reproduce the actual H2O+ training objective, offline buffer use, simulator interaction, and calibration behavior, the comparison should be renamed and framed more carefully.

Required revision:

- State exactly which parts of H2O+ are implemented and which are not.
- Rename the baseline to "dense residual transfer baseline inspired by H2O+" unless the implementation is faithful.
- Add a stronger baseline family:
  - target-city uncalibrated simulator,
  - pooled dense residual,
  - city-conditioned dense residual,
  - matched-capacity sparse residual,
  - oracle / in-city calibrated upper bound if any target labels are allowed.
- Add calibration-budget curves: 0%, 1%, 5%, 10%, 25%, and 100% target-city data. This directly tests the no-calibration claim.

### 3. Four city targets are not enough for broad generalization claims

The strict protocol has only four independent target-city evaluations. The full transition counts are large, but they are not independent city-level evidence. A Part C reviewer will not accept transition-level sample size as proof of cross-city generalization.

Required revision:

- Keep the primary claim at the city level: 4/4 strict targets, not millions of transitions.
- Add route-level or line-cluster bootstrap intervals within each held-out city.
- Report per-output ratios, not only total MSE. A total MSE win can hide failures in reward or passenger-related mechanisms.
- Add a pairwise transfer matrix if computationally feasible: each single source city to each target city, plus multi-source combinations.
- Avoid wording like "generalizes across cities" without qualifiers. Prefer "generalizes across the four evaluated open-data city bundles under static-derived targets."

### 4. The source-diversity result is both important and under-analyzed

The source-subset robustness table is probably the most scientifically interesting result. With all four cities, weighted CFCMT has mean ratio 0.578. Excluding Singapore makes the result 0.998, and the Halifax-MBTA pair is 1.042. This means the method is not simply "cross-city robust"; it depends on source diversity and evidence quality.

This should be elevated from limitation to analysis.

Required revision:

- Add a source-diversity section in Results or Discussion.
- Show which target cities fail when Singapore is excluded.
- Report the learned source weights for each held-out target.
- Analyze whether Singapore helps because of observed speed bands, larger route diversity, different headway distribution, or demand evidence quality.
- Add a practical recommendation: CFCMT should be used as multi-source transfer, not arbitrary pairwise transfer.

### 5. Policy validation is too weak for a bus holding control paper

The one-step policy validation is useful, but it is not enough for Part C if the paper is presented as bus holding control. The policy metrics are very small in absolute terms for several cities: bunching and large-gap rates are near zero, and reward differences are tiny for Singapore and Austin. The sampled rollout has only two episodes, so it should not carry much weight.

Required revision:

- Move one-step policy validation to "policy proxy" language unless closed-loop rollout is expanded.
- Add route-level closed-loop simulation for a representative set of lines in each city, even if not full SUMO for every route.
- Include standard transit metrics:
  - passenger waiting time,
  - excess waiting time,
  - headway coefficient of variation,
  - bunching frequency,
  - in-vehicle delay caused by holding,
  - total passenger generalized cost,
  - hold-time distribution and maximum hold frequency.
- Compare against operationally meaningful policies:
  - no holding,
  - fixed holding,
  - threshold headway holding,
  - schedule/headway hybrid,
  - dense residual policy.
- Add confidence intervals over routes, time periods, or random seeds.

### 6. The "causal-factored" mechanism claim needs sharper evidence

The mechanism factorization is plausible, but the current experiments do not prove that the gains come from causal mechanism separation rather than reduced capacity, feature pruning, or regularization.

Required revision:

- Add ablations:
  - CFCMT without source weighting,
  - CFCMT with random output groupings,
  - CFCMT with dense features but same ridge penalty,
  - dense baseline with matched feature count,
  - factored model with all parents available,
  - factored model with one mechanism removed.
- Report per-mechanism errors: headway, demand/waiting, dwell, speed, reward.
- Avoid implying causal identification unless interventions or invariance tests support it. Use "mechanism-factored" more often than "causal" in the experimental sections.

## Part C-Specific Bar

For *Transportation Research Part C*, I would expect at least one of the following before recommending acceptance:

1. A true vehicle-level validation slice from AVL/APC data.
2. A substantially expanded closed-loop policy evaluation showing passenger and reliability benefits.
3. A reframing as a reproducible open-data benchmark paper, with very explicit limits and a stronger target-construction audit.

The paper currently satisfies part of item 3, but not enough of items 1 or 2.

Part C reviewers will care less about total MSE alone and more about whether the model changes operational decisions in a realistic way. The manuscript should therefore shift some emphasis from dynamics MSE to passenger-facing and reliability-facing outcomes.

## IEEE ITS / IEEE T-ITS Bar

For IEEE ITS venues, the algorithmic contribution is closer to publishable, but the following would still be expected:

1. Stronger baseline taxonomy and faithful naming of the H2O+ comparison.
2. Calibration-budget experiments.
3. Runtime and reproducibility details.
4. Ablations proving the contribution of mechanism factorization rather than only regularization.
5. Closed-loop or multi-step validation beyond one-step policy selection.

IEEE reviewers may be more accepting of simulation-derived validation than Part C reviewers, but they will be stricter about algorithmic novelty, ablation completeness, and benchmark reproducibility.

## Required Experiments Before Submission

Priority order:

1. **Calibration-budget curve.** This is the most direct test of the paper's intended claim. Compare H2O+-style dense residual, unweighted CFCMT, weighted CFCMT, and in-city calibration as target data increases.

2. **Per-mechanism error table.** Report MSE or normalized error separately for headway, demand/waiting, dwell, speed, gap, and reward.

3. **Pairwise and subset transfer matrix.** Convert the current source-subset limitation into a structured source-diversity analysis.

4. **Mechanism ablation.** Random grouping and matched-capacity baselines are necessary to defend the causal/factored interpretation.

5. **Expanded policy evaluation.** At minimum, add more route-level rollouts and passenger/reliability metrics. If full closed-loop simulation is too expensive, present it as limited but statistically repeated route-level evaluation.

6. **Target-construction audit.** Document exactly how each static source becomes target transitions and add leakage/nontriviality checks.

## Recommended Claim Revisions

Avoid:

> CFCMT can achieve H2O+ performance without calibration.

Use:

> Under four open-data city bundles and static-derived cross-city validation, CFCMT outperforms a dense H2O+-style residual transfer baseline without target-city calibration, with the strongest gains in multi-source settings.

Avoid:

> CFCMT is better for real bus systems.

Use:

> The results support CFCMT as a calibration-light cross-city transfer mechanism that should next be validated on vehicle-level AVL/APC trajectories and closed-loop operations.

Avoid:

> Causal factorization explains the improvement.

Use:

> Mechanism-wise factorization is consistent with the observed transfer gains; targeted ablations are required to isolate it from capacity and regularization effects.

## Minor Comments

1. Clarify whether "full-route city bundles" means all bus routes available in the source feed after filtering, and list filtering rules.
2. Report the date/version of every public data source.
3. Add units to all table columns where applicable.
4. Explain whether total MSE is normalized across output dimensions; if not, output scales may dominate.
5. Add statistical uncertainty to figures where possible.
6. Keep "H2O+-style" consistently unless the implementation exactly matches H2O+.
7. Define the static descriptor vector for source weighting in an appendix table.
8. Include code/data release instructions or a reproducibility manifest.

## Bottom Line

This is a promising paper, but not yet a Part C-ready paper if positioned as bus holding control validation. It can become Part C-ready if the authors either add real trajectory / closed-loop evidence or explicitly reposition the work as a reproducible open-data cross-city transfer benchmark.

For IEEE ITS, the paper is closer, but still needs calibration-budget experiments, mechanism ablations, and stronger baseline naming. The best near-term path is to make the experimental story sharper: **CFCMT wins under static-derived multi-source cross-city transfer, source diversity matters, and zero-calibration operational claims remain future work.**

## Revision Status After Reviewer-Driven Experiments

The follow-up revision added the six requested experiment families: target-construction audit, target-route calibration-budget sweep, per-mechanism errors, mechanism and capacity ablations, source-size/pairwise transfer analysis, and expanded seeded BusSimEnv rollout where executable line environments are available.

The strongest remaining limitation is unchanged: the paper still does not include true vehicle-level AVL/APC trajectory validation. The revised manuscript should therefore keep the primary claim on static-data-derived cross-city dynamics transfer and treat closed-loop policy evidence as secondary.
