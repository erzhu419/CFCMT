# TSC v38/r34 Global Pairwise Salt Lake Outcome

This document reports the frozen v38/r34 outcome without target-specific family switching.

- integrity: `PASS`
- preregistered confirmation: `FAIL`
- combined cache SHA-256: `14c6dfad87999e55fa86fa942ac0ca2334129f4d1662b3c31a6e4fc7aa74c242`
- source tree SHA-256: `0445e6e00700e12ba6f5032cc227d97d817117c611955d9ae8978b2c2f5c3e43`

| Salt Lake network | B | Evaluation groups | Group-normalized fallback | Pairwise candidate | Absolute improvement |
|---|---:|---:|---:|---:|---:|
| saltlake_400s_200w_q1_weekday_peak | 60 | 126 | 0.093151 | 0.097671 | -0.004520 |
| saltlake_state_university_q1_weekday_peak | 60 | 110 | 0.108182 | 0.128578 | -0.020395 |

## Frozen gate

- fallback macro regret: `0.100666717`
- pairwise macro regret: `0.113124602`
- relative improvement: `-12.3754%`
- improved networks: `0/2`
- maximum absolute regression: `0.020395359`
- decision: `reject_global_pairwise_salt_lake_confirmation`

## Claim boundary

This is target-simulator adaptation with 60 complete matched action groups per network. It is not zero-shot, passive AVL/APC few-shot learning, or real-world intervention evidence.
