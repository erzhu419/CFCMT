# Untouched V84 Uncertainty-Horizon Confirmation

- Status: `PASS`
- Decision: `retain_phase_pressure_and_reject_v84_confirmation`
- Matrix: `64` rollouts
- Mean relative delta: `-0.3836%`
- Paired bootstrap 95% CI: `[-0.9411%, 0.1941%]`
- Improved seeds: `59.4%`
- Active seeds: `100.0%`
- Executed overrides: `82`
- Minimum intervention gap: `120.0` s

## Claim Boundary

This 64-rollout experiment is the first untouched confirmation of the single v83-selected uh_state_action_h120 policy. It compares against paired PhasePressure under the same full-day SUMO protocol. No v84 outcome was available when the policy and gate were frozen; the eight-seed v85 prospective root remains sealed.
