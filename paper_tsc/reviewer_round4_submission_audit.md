# Round 4 pre-submission reviewer assessment

## Review setup

- **Input scope:** Complete local TSC-first manuscript, canonical figures and tables, V43 external confirmation, V89 execution-layer confirmation, V90 post-freeze target-only ablation, V91 preregistered 64-seed target-only confirmation, source-data builders and regression-test evidence.
- **Assessment boundary:** This review can assess the submitted simulation evidence, internal protocol integrity, claim alignment and manuscript presentation. It cannot assess an uncreated public archive, independent reproduction, field deployment or unpublished comparison implementations.
- **Shared manuscript claim summary:** Anchored CFCMT replaces absolute dense residual prediction with parent-restricted, matched action contrasts and an antisymmetric correction. It improves a same-information dense residual on two external city-derived SUMO benchmarks, but does not beat pressure control and does not establish a general benefit from source-labelled data over target-label-only fitting.
- **Visible evidence base:** Eighteen development networks in seven city groups; two external target geometries; seed-blocked target adaptation; an untouched offline seed; 56 original 3,600-s rollouts; 512 fresh source-contribution rollouts; simulator accounting, incident audits, frozen provenance and reported negative confirmation outcomes.
- **Missing materials affecting confidence:** A persistent public repository and DOI, same-protocol implementations of the nearest transfer-learning baselines, more independent target cities, and field or calibrated-digital-twin validation.

## Reviewer 1

### Overall assessment

The manuscript is unusually strong in protocol traceability and unusually candid about negative results. The revised target-label-only experiment is decisive, however: it rejects the proposition that the source bank provides a general closed-loop benefit under the current fitting rule. The paper now reports this correctly, but the central scientific case for cross-city mechanism transfer is consequently incomplete.

### Who would be interested in the results, and why

Researchers in traffic-signal control, offline reinforcement learning, simulator-assisted transfer and causal representation learning would value the matched counterfactual protocol, the separation of structural-model gain from source-data gain, and the demonstration that favourable offline ranking can coexist with closed-loop negative transfer.

### Major strengths

- The method, target information budget and causal boundary are defined explicitly.
- The dense residual receives matched contrast data and the same target refit budget, making that comparison informative.
- V91 uses 64 new seeds, exact matrix completeness and no failed-seed exclusion.
- The manuscript retains the large Los Angeles failure and reports opposite city effects instead of hiding them in a macro average.
- Metric-population, completion, teleport and collision accounting are stronger than typical simulation studies.

### Major concerns

- Source contribution is the central transfer question, and V91 rejects it: CFCMT is worse in the equal-city mean, worse in Los Angeles and has more collision incidents than target-only.
- The original favourable closed-loop comparison is against a dense residual, simulator-only and rigid variants, not against a target-only controller. It supports a structured model, not source-city transfer.
- Only two independent external-city units are available. Three Jinan demands and 64 seeds do not increase the city-level sample size.
- All counterfactual labels and closed-loop outcomes come from one simulator physics family. Network shift is tested; sim-to-real or field causal transfer is not.
- The declared parent sets and action-common nuisance assumption are design commitments, not empirically identified invariances.

### Technical failings that need to be addressed before the case is established

1. Either reformulate the main contribution as structured target adaptation with a documented negative-transfer result, or develop a source-inclusion/source-weight rule without using V91 outcomes and confirm it on new cities.
2. Test whether the proposed invariance assumptions predict the observed Los Angeles versus Jinan sign reversal. At present, the theory explains contrast cancellation under assumptions but does not diagnose when source mechanisms become harmful.
3. Add independent target-city units. Seed-level precision cannot support a population-level cross-city claim when the city count is two.
4. Add same-protocol nearest transfer baselines or narrow the comparative claim further. Native RESCO and LibSignal runs do not fill this role.
5. Release the exact code, converted inputs, model artifacts and rollout evidence in a persistent archive.

### Assessment against Nature-style criteria

- **Originality:** The matched action-contrast formulation and auditable source-contribution falsification are original enough to merit attention, but the individual modelling ingredients are not shown to constitute a broadly new control paradigm.
- **Scientific importance:** The negative-transfer result is scientifically useful; a generally effective cross-city controller is not established.
- **Interdisciplinary readership:** The lesson that source data can hurt despite favourable offline validation may interest transfer-learning researchers beyond transportation, but the current empirical scope is narrow.
- **Technical soundness:** The reported experiments are internally well controlled. The unsupported step is from structured-model advantage to source-transfer benefit.
- **Readability for nonspecialists:** The causal boundary and information budget are now clear. Protocol-version density still imposes a high reading cost.

### Recommendation posture

Technically careful and potentially publishable after major reframing, but the stronger cross-city source-transfer case is not established from the current evidence.

## Reviewer 2

### Overall assessment

The strongest contribution is no longer the controller's absolute performance. It is the separation of three often-confounded claims: contrast structure can outperform a dense residual, source labels may or may not help, and neither result implies superiority over classical pressure control. This is a valuable scientific correction, but its significance depends on whether the paper embraces that correction as the main discovery rather than presenting it as a limitation around a transfer method.

### Who would be interested in the results, and why

The paper should interest readers studying negative transfer, domain adaptation under limited target data, counterfactual world models and benchmark methodology. Transportation readers will also value the evidence that a simple local rule remains a demanding ceiling.

### Major strengths

- The manuscript distinguishes zero-shot, target offline adaptation and field validation.
- The action-contrast theory states exact algebraic properties rather than claiming an identified field causal graph.
- The sequential record exposes failed development tracks and adaptive-to-confirmatory shrinkage.
- Target-only is a particularly useful comparator because it holds the model family, feature semantics, target labels and executor fixed.
- The paper reports that offline source benefit in Los Angeles did not translate to fresh closed-loop benefit, which is a nontrivial result.

### Major concerns

- The title and CFCMT name still foreground cross-city transfer even though the additional value of source-city labels is unconfirmed.
- The theory predicts cancellation of action-common bias, but the main empirical surprise is network-dependent negative transfer. The manuscript does not yet offer a mechanism-level explanation or a testable criterion for that failure.
- Pressure policies outperform CFCMT, so practical significance cannot be based on controller replacement.
- The closest cross-city methods are discussed but not reproduced under the same information and executor protocol.
- The broad claim remains simulator-local and limited to two external geometries.

### Technical failings that need to be addressed before the case is established

1. Make the source-contribution failure the organizing scientific question, or provide a newly frozen method that can choose target-only when source data are harmful.
2. Add a pre-outcome hypothesis for the city heterogeneity, then test it on additional cities. Post-hoc explanation of Los Angeles alone would not establish generality.
3. Provide an adaptation-budget curve that includes target-only and source-plus-target models at identical target-label budgets. The current full-budget comparison does not establish where source data become useful.
4. Reproduce at least one nearest transfer method under the same state, action, target-label and rollout contract.
5. Resolve data and code availability before submission.

### Assessment against Nature-style criteria

- **Originality:** Strongest as an experimental dissection of negative transfer; less strong as another traffic-control architecture.
- **Scientific importance:** Potentially important if the work yields a general criterion for safe source use. The current two-city sign reversal is evidence of the problem, not yet its solution.
- **Interdisciplinary readership:** Negative transfer under counterfactual simulation has cross-domain relevance, but the manuscript needs a more general abstraction of the failure.
- **Technical soundness:** The evidence chain is rigorous. The causal-transfer interpretation must remain bounded to simulator interventions and declared parents.
- **Readability for nonspecialists:** The revised abstract is clear, but MC-WM lineage, anchored CFCMT and the separate v86-v89 successor create competing narratives.

### Recommendation posture

Promising as a rigorous negative-transfer and target-adaptation study; the broad significance case remains underdeveloped unless the source-selection problem is solved or made the explicit main result.

## Reviewer 3

### Overall assessment

The manuscript is substantially clearer and more trustworthy after integrating V91. A nonspecialist can now see what information is available, what is transferred and what failed. The remaining problem is narrative hierarchy: the paper contains development history, a final v43 controller, a target-only source test and a separate execution-layer successor. All are relevant to transparency, but the main scientific message competes with its own audit trail.

### Who would be interested in the results, and why

Traffic-control and reinforcement-learning readers will use the concrete benchmark results. A broader methods audience may care about the mismatch between offline action ranking and closed-loop value, and about protocol designs that preserve negative evidence.

### Major strengths

- The abstract now reports the failed source-contribution confirmation rather than only favourable baselines.
- Figures establish the data budget, transferred object and network shift without claiming calibrated city-wide twins.
- Tables report pressure policies, target-only results, completion and incidents in a common direction.
- The Discussion names the exact target where transfer fails and avoids presenting a simulator result as field causality.
- Build scripts and source tables connect manuscript numbers to frozen audits.

### Major concerns

- The manuscript remains long and version-heavy for readers outside the immediate project. Labels such as v43, v89, v90 and v91 are useful for audit but should not carry the explanatory burden in the main text.
- Two distinct 64-seed experiments can be confused: one tests an execution-layer successor against phase pressure; the other tests source-plus-target CFCMT against target-only.
- The main title does not signal that source contribution is conditional and that the strongest result is a failure boundary.
- The Data and Code Availability section still describes a future deposit rather than an available archive.
- The practical use case is uncertain because pressure control is stronger and no field or calibrated-twin result is shown.

### Technical failings that need to be addressed before the case is established

1. Reduce main-text development chronology and move protocol identifiers to a compact evidence map in the supplement.
2. Distinguish the two 64-seed experiments with stable descriptive names everywhere, not only version numbers.
3. Align the title, abstract and conclusion around one supported claim: structured action contrast with network-dependent source contribution.
4. Provide a public evidence package and a concise reproduction path.
5. Add at least one further independent target-city replication before making a general cross-city statement.

### Assessment against Nature-style criteria

- **Originality:** The transparent negative-transfer audit is distinctive; the broad novelty of the controller alone is less evident.
- **Scientific importance:** Important within transfer-oriented traffic control, but immediate and far-reaching implications are not yet demonstrated.
- **Interdisciplinary readership:** The offline-to-closed-loop mismatch is the best bridge to a wider audience.
- **Technical soundness:** Internally credible within SUMO; external validity and source-transfer generality remain open.
- **Readability for nonspecialists:** Improved, but the version chronology and multiple successor tracks remain barriers.

### Recommendation posture

The manuscript is credible and readable enough for specialist review, but a stronger narrative and broader independent validation are needed for a high-impact interdisciplinary case.

## Cross-review synthesis

### Consensus strengths

- The manuscript now aligns its claims with the V91 negative result.
- Frozen role separation, exact matrix accounting and retention of failed seeds make the evidence auditable.
- The target-label-only comparator meaningfully isolates deployed source-label contribution.
- The same-information dense residual and strong pressure policies are valuable comparators.
- The causal and target-information boundaries are stated more honestly than in the earlier draft.

### Consensus technical risks

- The current method does not establish a general closed-loop benefit from source-city labels.
- Two external cities cannot support population-level cross-city inference.
- Simulator-only counterfactuals do not establish field causal or sim-to-real effectiveness.
- The theory does not yet predict or prevent the observed network-dependent negative transfer.
- Same-protocol nearest transfer methods and a persistent public archive are absent.

### Where emphasis differs across reviewers

- Reviewer 1 places greatest weight on the failed source-contribution estimand and the missing independent-city evidence.
- Reviewer 2 places greatest weight on whether negative transfer can become the paper's central scientific advance or motivate a newly confirmed source selector.
- Reviewer 3 places greatest weight on narrative hierarchy, protocol-version burden and accessibility beyond the immediate project.

### Broad-interest / significance readout

The potentially broad result is not that CFCMT wins a traffic-control leaderboard. It is that matched offline action ranking can favour source-plus-target fitting while fresh closed-loop control favours target-only fitting in one network. This is relevant to simulator-assisted transfer beyond traffic control, but the present manuscript documents the phenomenon in two geometries rather than establishing a general theory or remedy.

### Most important issues to resolve before a strong Nature-style case is established

1. Decide the paper's scientific identity: an honest structured-adaptation study with a negative-transfer finding, or a new source-selection method confirmed on untouched cities.
2. Add independent-city evidence and at least one same-protocol nearest transfer baseline.
3. Explain or predict the Los Angeles versus Jinan source-effect reversal without reusing V91 as development data.
4. Deposit a complete public evidence package with a persistent identifier.
5. Simplify the main narrative so that the two separate 64-seed experiments cannot be conflated.

## Risk / unsupported claims

- **General source benefit:** Not supported; V91 rejects it.
- **Universal cross-city transfer:** Not supported with two city units and opposite city effects.
- **Controller superiority:** Not supported; MaxPressure and phase pressure remain stronger.
- **Field causal transfer or sim-to-real effectiveness:** Not assessed by the supplied evidence.
- **Population-level confidence interval over cities:** Not available; current intervals are seed- or matrix-descriptive.
- **Nearest-method superiority:** Not assessable without same-protocol CrossLight, X-Light, MetaLight or GESA implementations.
- **Independent reproducibility:** Not assessable until the public archive and DOI exist.
