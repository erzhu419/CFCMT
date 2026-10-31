# Multihorizon Estimand-Aligned Closed-Loop Development Audit

- Status: `PASS`
- Decision: `retain_phase_pressure_and_reject_estimand_alignment`
- Matrix: `198` paired rollouts
- Primary feasible: `0`
- Diagnostic feasible: `0`

| Policy | Role | Global cooldown | Min priority | Mean relative delta | 95% CI | Improved seeds | Active seeds | Overrides | Feasible |
|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| mh_compact_g0_p005 | preregistered_primary | 0 | 0.05 | 0.1774% | [-0.3140%, 0.6505%] | 40.9% | 100.0% | 181 | no |
| mh_compact_g2_p005 | preregistered_primary | 2 | 0.05 | 0.0289% | [-0.5652%, 0.6549%] | 59.1% | 100.0% | 118 | no |
| mh_compact_g5_p000 | preregistered_primary | 5 | 0.00 | 0.8921% | [0.3107%, 1.4620%] | 22.7% | 100.0% | 233 | no |
| mh_compact_g5_p002 | preregistered_primary | 5 | 0.02 | 0.5681% | [0.0928%, 1.0453%] | 36.4% | 100.0% | 163 | no |
| mh_compact_g5_p005 | preregistered_primary | 5 | 0.05 | 0.0754% | [-0.5630%, 0.6748%] | 45.5% | 100.0% | 113 | no |
| mh_compact_g5_p010 | preregistered_primary | 5 | 0.10 | -0.1888% | [-0.6213%, 0.2526%] | 59.1% | 100.0% | 51 | no |
| mh_compact_g5_p015 | preregistered_primary | 5 | 0.15 | -0.2427% | [-0.7324%, 0.2469%] | 50.0% | 100.0% | 34 | no |
| mh_state_action_g5_p005 | predeclared_stability_diagnostic | 5 | 0.05 | 0.0036% | [-0.5588%, 0.5111%] | 36.4% | 100.0% | 121 | no |

## Claim Boundary

This 198-rollout matrix is adaptive successor development after the v80 rejection. The global-cooldown and priority grids were frozen from v80 traces before any v81 outcome. Only the seven compact-h60 candidates may authorize an untouched v84 experiment. The state-action-h60 candidate is diagnostic only. No efficacy or generalization claim is authorized by v81.
