# TSC v42/r38 pre-evaluation baseline-freeze amendment

Frozen at `2026-08-09T05:04:07Z`, before collection of external evaluation seed
`7079` and before inspection of any LA/Jinan OOF regret values.

## Reason

The external confirmation must compare the frozen v41 selector against more
than its rigid anchor.  Every baseline below therefore receives exactly the
same source18 cache, source-rule evidence, target-city B100 adaptation groups,
five folds, and label-free city context.  Baseline models are serialized before
the held-out evaluation cache is allowed to exist.

## Frozen families

| Executable family | Paper role | Claim boundary |
|---|---|---|
| `simulator` | simulator-only | source-selected rule prior; no learned residual |
| `dense` | H2O+-style dense residual | architecture-style comparator, not an exact H2O+ reimplementation |
| `dense_advantage` | dense same-estimand control | same action-ranking estimand without causal parent restrictions |
| `causal_rigid_advantage` | previous rigid CFCMT | pre-v41 rigid approximation |
| `cfcmt_mechanism` | previous full CFCMT | pre-v41 full causal-mechanism route |
| `causal_target_only` | target-only adaptation | only the 80 in-fold target groups; no source transfer benefit |

The frozen v41 CFCMT method remains the two-family anchored selector
`scope_all_tau_0_rho_0p1_lambda_0`.  None of the baseline OOF results may tune
that selector or alter its candidate grid.

## Leakage gate

1. LA and Jinan are fitted in separate processes and never source each other.
2. Adaptation uses seeds `5057` and `6067` only.
3. Evaluation seed `7079` is collected only after both method and baseline
   artifacts pass round-trip and SHA-256 audits.
4. Jinan is one city with three equal-weight demand scenarios, not three
   independent transfer targets.
