# TSC v22/r18 Joint Mechanism-Gate Outcome

Date: 2026-08-08

## Frozen protocol

- Run ID: `tsc_v22r18_joint_gate_dev600_20260808`
- Snapshot manifest: `cf_h2o/results/cluster/tsc_v22r18_joint_gate_snapshot_20260808.json`
- Snapshot SHA-256: `2b229dd9eca6239cacff8ee4777320bd127a9bfb43e587bc9759a5a77f6f9c3f`
- Source-tree SHA-256: `d97c1f6eec3b7fc51858672905a895123d96b3c93c137e68327139a8422ca51d`
- SUMO/libsumo: `1.22.0`
- Budgets: `0, 8, 16, 32, 60, 120` matched target-simulator groups
- Development horizon: 600 s; 16 scenarios, 17 policies, 3 evaluation seeds

The final `budget_120/result.json` was synchronized only after the remote
process exited with `rc=0`. Its local and remote SHA-256 both equal
`247ff48c50dfc41b8beb2fb3255dfc87bbf1250164d4cffd76ae40ce2a83fe6d`.
It contains 816 unique scenario-policy-seed rows, all with finite required
metrics, `ok=true`, zero starting teleports, and zero ending teleports.

## Preregistered audit

The authoritative audit is
`cf_h2o/results/cluster/tsc_v22r18_joint_gate_dev600_20260808/development/budget_matrix_audit.json`.
It reports:

- overall audit: **FAIL**
- overall method gate: **FAIL**
- efficacy gate: **FAIL**
- mechanism-contribution gate: **FAIL**
- guarded collision noninferiority failure at budget 60: `41 > 40`
- guarded collision noninferiority failure at budget 120: `43 > 40`

| Budget | Primary mean vs prior | Median vs prior | Winning cities | Worst city | Top-gain share | Broad |
|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | no |
| 8 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | no |
| 16 | 0.00% | 0.00% | 0/6 | 0.00% | 0.0% | no |
| 32 | -1.68% | 0.00% | 1/6 | 0.00% | 100.0% | no |
| 60 | -1.45% | -0.08% | 3/6 | +0.56% | 96.0% | yes |
| 120 | -2.25% | 0.00% | 2/6 | +0.52% | 90.3% | no |

At budget 120, the primary guarded full model is 1.72% better than the
strongest paired rule in the pooled mean, but this does not rescue the method:
the city-level median is zero, almost all gain is concentrated in Atlanta, the
worst city regresses, and paired collision noninferiority fails. The simpler
target-only MPC is the best paired policy at budgets 32, 60, and 120; at budget
120 it is 3.52% better than the selected source prior.

## Decision

Do not promote v22/r18 to a 3600 s confirmatory matrix and do not cite it as a
successful full-CFCMT result. Preserve it as a negative exact ablation showing
that the legacy mechanism stack and target gate do not produce broad,
monotone, safety-noninferior cross-city gains.

The next development stage must repair the estimand rather than tune the same
gate. In particular, v22's legacy mechanism protocol zeroes the simulator
prior and therefore is not the analytic-prior residual model described in the
paper. The additive v23 branch tests a unit-audited local physical simulator
residual first, followed by latent adaptation only if the rank-0 residual
passes source-only and held-out-city screening.
