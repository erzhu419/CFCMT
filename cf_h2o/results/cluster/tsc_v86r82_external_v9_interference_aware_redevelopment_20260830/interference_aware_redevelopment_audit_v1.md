# V86 Interference-Aware Redevelopment Audit

- Status: `PASS`
- Decision: `retain_phase_pressure_and_reject_v86_redevelopment`
- Matrix: `270` adaptive-development rollouts
- Selected candidate: `None`
- Deployment fallback: `phase_pressure`

## Candidate Results

| Candidate | Mean delta | 95% CI | Improved | Active | Collisions | Gate |
|---|---:|---:|---:|---:|---:|:---:|
| `uh_global_h120_m1` | -0.4289% | [-0.8176%, -0.0326%] | 68.5% | 100.0% | 1 | FAIL |
| `uh_local_h120_m1` | -0.0806% | [-0.4680%, 0.3319%] | 59.3% | 100.0% | 0 | FAIL |
| `uh_local_h120_m2` | -0.1164% | [-0.5168%, 0.3025%] | 59.3% | 100.0% | 0 | FAIL |
| `uh_local_h120_graph` | -0.1164% | [-0.5168%, 0.3025%] | 59.3% | 100.0% | 0 | FAIL |

## Claim Boundary

V86 is adaptive redevelopment after the failed v84 confirmation. It compares the frozen h120 originator under network-global and conflict-graph-local horizon alignment. The 54 seeds combine prior development and failed-confirmation outcomes and can never support confirmation. V85 prospective evidence remains sealed.
