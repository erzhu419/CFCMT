# Formal Reviewer Report for Transportation Research Part C / IEEE ITS / IEEE T-ITS

## Manuscript Summary

The manuscript studies calibration-light cross-city transfer for bus holding control. It proposes a mechanism-factored residual transfer approach, CFCMT, and evaluates it against dense H2O+-style residual baselines on four open-transit city bundles. The current revision adds a guarded source-city ensemble, a source-only feature-set selector, same-information dense source-ensemble baselines, calibration-budget analysis, per-mechanism diagnostics, source-subset robustness, generator robustness, route-bootstrap uncertainty, and an external passive Austin APC/AVL validation slice.

The paper is best understood as a reproducible open-data benchmark and method study for target-label-free cross-city residual world-model transfer, not as a field deployment or fully validated operational control paper.

## Overall Recommendation

**Transportation Research Part C: Major Revision.**

The paper is now substantially more credible than earlier versions. The experimental design is transparent, the revised source-selection protocol avoids the most serious post-hoc selection concern, and the same-information dense baseline makes the comparison more defensible. However, I would still require major revision for Part C because the primary evidence remains static-data-derived, the policy/control validation is secondary, and the manuscript must be extremely disciplined about not claiming real-world bus holding superiority.

**IEEE Transactions on Intelligent Transportation Systems / IEEE ITS: Borderline Accept to Major Revision.**

For IEEE ITS-oriented venues, the algorithmic contribution and ablation package are close to sufficient, provided the paper frames the contribution as target-label-free world-model transfer rather than operational control validation. The remaining decision would depend on how strictly the venue expects faithful H2O+ reproduction and closed-loop control evidence.

## Main Strengths

1. **Clear and conservative evaluation unit.** The strict leave-one-city-out protocol correctly treats each city as the independent target unit. The paper no longer inflates claims by treating repeated configured splits as independent city samples.

2. **Improved target-label-free selection protocol.** The source-only feature-set selector addresses a major validity concern. The selected target-summary feature set is chosen by leave-one-source validation before target evaluation, rather than by target MSE.

3. **Same-information baseline added.** The dense H2O+-style source ensemble receives the same unlabeled target-summary source weights as the guarded CFCMT ensemble. This is an important fairness improvement. The result is informative: the dense source ensemble improves substantially, but the guarded mechanism-factored ensemble remains better.

4. **Strong diagnostic coverage.** The current appendix contains useful selector ablations, threshold sensitivity, target-label usage audit, oracle-gap diagnostics, source-subset robustness, calibration-budget sweeps, per-mechanism errors, and generator perturbations.

5. **Claim boundaries are more honest.** The manuscript now uses target-label-free / low target-label calibration language and avoids treating the method as pure source-only zero-shot transfer.

6. **Negative and boundary evidence is visible.** The local-only and uniform source weighting variants regress toward the pooled selector, and the Austin passive APC/AVL validation shows that source-only transfer can overshoot a state-informed passive baseline. These results make the paper more credible.

## Major Concerns

### 1. Static-derived targets still limit the Part C claim

The central Part C concern remains that most target labels are generated or reconstructed from open static data and deterministic rules rather than observed counterfactual bus operations. Singapore has stronger observed passenger and traffic evidence, but Austin, Halifax, and MBTA still include schedule-derived or apportioned components.

This does not invalidate the paper, but it limits what the paper can claim. The strongest claim is about reproducible static-data-derived cross-city residual transfer, not field-ready bus holding control.

**Required revision:**

- Keep the target-construction audit prominent in the main paper, not only the appendix.
- State explicitly which output groups are observed, apportioned, schedule-derived, or simulator-rule-derived for each city.
- Avoid any wording that implies the method has been validated on real counterfactual holding outcomes.
- Consider adding normalized per-output errors, because total MSE may be scale-dominated by headway/gap outputs.

### 2. Policy/control evidence is not yet strong enough for operational conclusions

The one-step policy validation and sampled rollout are useful, but they do not establish closed-loop bus holding superiority. The real APC/AVL validation is passive action-0 validation and does not contain counterfactual hold actions. This is a significant limitation for a transportation-control journal.

**Required revision:**

- Treat policy evaluation as secondary evidence.
- Use "one-step policy proxy" or "policy-facing diagnostic" rather than "policy validation" where appropriate.
- If possible, expand closed-loop route-level rollout with repeated seeds and passenger-facing metrics: waiting time, excess waiting time, in-vehicle delay, headway coefficient of variation, bunching frequency, hold-time distribution, and passenger generalized cost.
- Compare with simple operational policies such as no holding, fixed holding, threshold headway holding, and schedule/headway hybrid rules.

### 3. The H2O+ comparison remains vulnerable

The manuscript is safer now because it uses "H2O+-style" language and adds a same-information dense source ensemble. Still, a reviewer familiar with H2O+ may challenge whether the dense residual baseline faithfully reproduces the original H2O+ training objective, offline-to-online adaptation procedure, and policy-learning loop.

**Required revision:**

- Keep "H2O+-style" consistently unless the original H2O+ algorithm is fully reproduced.
- Add a short baseline taxonomy table:
  - original H2O+ component,
  - implemented component in this manuscript,
  - deviation from original H2O+,
  - reason for the deviation.
- Avoid headline language such as "CFCMT outperforms H2O+" unless the implementation is faithful.

### 4. Four independent city targets remain a small sample

The four-city strict leave-one-out setup is correct, but it is still only four independent target units. The six configured selector-stress splits are useful diagnostics, but repeated target cities should not be used as independent generalization evidence.

**Required revision:**

- Continue reporting strict 4-city LOO as the primary city-level result.
- Make it explicit whenever the six configured splits are selector diagnostics rather than independent target-city evidence.
- If feasible, add two or more additional static-derived city bundles. Even lower-evidence cities would help if clearly labeled.
- Avoid broad statements such as "generalizes across cities"; use "in the evaluated four-city open-data setting."

### 5. The causal interpretation should remain restrained

The mechanism-wise factorization is plausible and well motivated, and the ablations support the chosen parent sets. However, the experiments do not prove causal invariance. Some gains may still come from sparsity, regularization, target-summary source selection, or reduced capacity.

**Required revision:**

- Prefer "mechanism-factored" or "mechanism-wise sparse" in experimental sections.
- Use "causal" as a design motivation unless stronger invariance or interventional evidence is added.
- Include a sentence in Results or Discussion stating that the evidence supports mechanism-factored transfer, not formal causal identification.

### 6. Source-feature selection needs a concise algorithmic description

The new source-only feature-set selector is a major improvement, but it must be described as an algorithm, not only as an appendix table. A reviewer should be able to understand exactly when and how the feature set is selected.

**Required revision:**

- Add pseudocode or a numbered procedure:
  1. Define candidate target-summary feature families.
  2. Run leave-one-source folds.
  3. Score each feature family by source-fold selected-candidate performance.
  4. Select the feature family before target evaluation.
  5. Compute target source weights and apply the dominance guard.
- Clarify the tie-breaking rule.
- Clarify that target transition labels are used only for final reporting.

## Specific Comments on Current Experimental Results

### Guarded ensemble and same-information baseline

The revised same-information comparison is useful:

- Dense H2O+-style pooled residual: mean MSE 884.853.
- Dense H2O+-style source-weighted ensemble with the same unlabeled target summary: mean MSE 846.466.
- Rigid CFCMT: mean MSE 851.893.
- Guarded CFCMT source ensemble: mean MSE 836.115.

This result strengthens the paper because it shows that the target-summary signal helps both dense and factored models, but mechanism-factored source selection still adds value beyond giving H2O+ the same unlabeled target information.

### Feature-set selector

The source-only feature-set selector is a substantial improvement over post-hoc reporting of the best target feature set. The result that simulator-output summaries are selected for most splits and observation-only summaries for the duplicated Singapore split is plausible.

However, the main paper should avoid overinterpreting "simulator-output-only is best" as a universal result. It is selected under this dataset and source-fold scoring protocol.

### Strict leave-one-city-out subset

The strict four-target guarded subset remains the most important evidence. The manuscript should emphasize the four unique targets rather than the repeated six-split diagnostic. The six-split diagnostic is acceptable as a selector stress test.

### Oracle-gap diagnostics

The oracle-gap table is useful and should remain clearly diagnostic. It demonstrates that the selected method is not secretly using target labels, and that target-oracle candidate selection would still be somewhat better. This supports the claim that there is no target-label leakage, but also shows room for source selector improvement.

### External APC/AVL validation

The Austin APC/AVL slice is valuable but limited. It validates passive observed transition prediction, not counterfactual holding. This should be framed as a reality check for dynamics transfer under observed operations, not as evidence of control effectiveness.

## Required Revisions Before Submission

1. **Add algorithmic pseudocode for guarded source selection and source-only feature-set selection.**

2. **Add a same-information baseline paragraph in the main text.**
   The table is useful, but the paper should explicitly state that dense H2O+ also improves when given the same target-summary weights and still remains weaker than guarded CFCMT.

3. **Move target-construction evidence closer to the main argument.**
   Part C reviewers will care strongly about whether the target is observed or generated.

4. **Add a baseline-faithfulness table for H2O+.**
   This is necessary to avoid reviewer objections about comparing against a partial H2O+ implementation.

5. **Add a limitation sentence on causal identification.**
   The method is mechanism-factored; the current experiments do not prove causal invariance.

6. **Reduce control-policy claims.**
   Use dynamics/world-model transfer as the primary claim; keep policy evidence as secondary.

## Minor Comments

1. Define all target-summary feature families in a compact appendix table.
2. State whether total MSE is averaged over raw output scales or normalized outputs.
3. Add units to all MSE-related tables where possible.
4. Check that all configured-split tables explicitly say repeated target cities are included.
5. Clarify whether route-bootstrap intervals are route-level uncertainty only, not city-level uncertainty.
6. Consider shortening the appendix table captions; there are now many guarded-ensemble diagnostics.
7. Use consistent terminology: "target-label-free", "unlabeled target summary", "source-fold selector", and "H2O+-style dense residual baseline."

## Suggested Decision Letter

The manuscript addresses an important and timely problem: transferring bus holding dynamics models across cities without target transition-label calibration. The revised experimental package is unusually transparent and includes several valuable diagnostics. In particular, the source-only feature-set selector and same-information dense source-ensemble baseline address important selection and fairness concerns.

Nevertheless, I recommend major revision. The evidence currently supports a static-data-derived cross-city world-model transfer claim, but not yet a strong operational bus holding control claim. The manuscript should further clarify target construction, baseline fidelity, policy-evaluation limits, and the distinction between mechanism-factored design and formal causal identification.

If the authors make these framing and documentation changes, I would view the paper as potentially suitable for IEEE ITS / IEEE T-ITS and as a credible Part C submission if positioned as a reproducible open-data cross-city transfer benchmark rather than a field-validated control method.

## Bottom Line

The paper has crossed the threshold from "interesting but experimentally fragile" to "credible with careful framing." The remaining risk is not the guarded ensemble result itself; it is overclaiming beyond the available validation. With disciplined language, a clear source-selection algorithm, and stronger transparency around H2O+-style baselines and static-derived targets, the manuscript can be a strong submission.
