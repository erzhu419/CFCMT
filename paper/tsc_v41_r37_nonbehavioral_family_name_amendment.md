# TSC v41/r37 Non-Behavioral Family-Name Amendment

Before reading any 12-network formal outcome, an engineering preflight exposed
that the frozen protocol JSON used the stale descriptive string
`causal_pairwise_preference_advantage`. The executable constant, runner import,
fit log, and result payload all used
`causal_antisymmetric_pairwise_advantage`.

This amendment replaces only that stale JSON string and adds an executable
equality check between the JSON family list and `BASE_FAMILIES`. It changes no
data partition, target, feature, model implementation, candidate, selector,
threshold, seed, or efficacy criterion.

The snapshot `ceb35f971a53cf36d664eb0a33e495f9a79e959db7a8ef730b596df3108d1170`
and its 12-network launch are engineering-only and excluded from v41 efficacy
auditing. A new immutable snapshot and a clean results root are required.
