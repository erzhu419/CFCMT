# TSC v38/r34 Global Pairwise Salt Lake Confirmation

Date frozen: 2026-08-09. Written after all six-city development results through v37/r33 were finalized and before generating or reading any Salt Lake CFCMT counterfactual-cache or model outcome. Existing Salt Lake demand construction and rule-controller safety admission are not blinded; the final CFCMT confirmation is.

## Development decision frozen before confirmation

The development stage compared model families on six city groups. Three target-specific deployment selectors (v35 calibration holdout, v36 five-fold cross-fit, and v37 leave-one-group-out) failed their own preregistered breadth gates and remain negative ablations. They will not be retuned after observing their outcomes.

The globally selected method is therefore fixed without a target-specific gate:

- target information budget: exactly 60 complete simulator counterfactual action groups;
- candidate: `causal_antisymmetric_pairwise_advantage`;
- same-information fallback: `causal_group_normalized_rigid_advantage`;
- both methods receive the identical source bank and identical 60 target groups;
- the family choice is made once from the six-city development stage and is not changed per Salt Lake network.

The v37 audit was frozen at SHA-256 `15a83a08b64bec3614e72df0ab4bebbb6bfbd118a02d6daa9ca42f16b1624b8f`. Its final all-60-group comparison, not its failed selector, motivated the global architecture choice. This is a development-set model-selection decision, not evidence about Salt Lake.

## Confirmatory domain and claim boundary

Salt Lake City is absent from every source fit. The two target networks are:

1. `saltlake_400s_200w_q1_weekday_peak`;
2. `saltlake_state_university_q1_weekday_peak`.

Both share `city_group=salt_lake_city`; when either is evaluated, **both** Salt Lake networks are excluded from the source bank. Each network is adapted independently using its own 60 complete target-simulator action groups.

This experiment is **target-simulator adaptation**, not zero-shot and not passive-history few-shot learning. Each target group contains matched outcomes for every admissible focal action. The paper must not describe these groups as AVL/APC logs, real intervention records, or calibration-free target data.

## Frozen data admission

Demand was fixed before method confirmation using the Q1 2023 weekday detector protocol. The route-file hashes are:

- 400 S / 200 W: `f60ab1f5d63b3f292abbf549a78ee48bb65388bf6841a3435cfd6ca8e6313db0`;
- State / University: `094ee431eba138fc1fd4036d5cd5b00b4647c77787e0cbacab58d748386838ca`.

The final rule-controller admission artifact is `saltlake_full_admission_local_20260808/admission.json`, SHA-256 `7da84de7268858a54ce2671c6efc037dae7b38ee32e18b46d121459572c5d14b`. It reports PASS under full-junction occupancy clearance with no collision or teleport across two networks, three pressure rules, and four seeds. Earlier collision-discovery artifacts remain part of the audit trail and are not treated as passing evidence.

## Frozen counterfactual collection

Salt Lake counterfactual data use exactly the v25 source-cache protocol:

- SUMO/libsumo `1.22.0`;
- seeds `2027`, `3037`, and `4047`;
- 600 s episode, 60 s warm-up, 10 s control interval;
- up to four focal traffic lights per interval;
- six control intervals per counterfactual rollout;
- 16 deterministic collection shards per seed;
- `phase_pressure` behavior policy;
- full controllable-TLS coverage required;
- no teleport, junction collision monitoring, and symmetric censoring of the whole matched action group when any branch has a safety event.

Collection first produces an immutable Salt-only cache. A new combined cache is then assembled from byte-identical hard links to the frozen 16-network source cache and the admitted Salt-only cache. The source cache is never modified. Exact file counts, per-file hashes, aggregate hash, identity fields, replay audit, coverage, and complete-action-group counts must pass before model fitting.

## Frozen adaptation and evaluation

For each Salt Lake target independently:

1. deterministically order safe complete action groups using `target-adaptation-split-v3`;
2. select exactly the first 60 groups for target adaptation;
3. fit candidate and fallback on all six source city groups plus those exact same 60 target groups;
4. evaluate both only on safe complete target groups outside the selected 60;
5. compute normalized action regret with the existing v34-v37 implementation.

No Salt Lake group used for fitting may enter evaluation. No target-specific selector, threshold adjustment, family switch, or hyperparameter change is permitted after any Salt Lake CFCMT outcome is read.

## Primary confirmation gate

The frozen pairwise candidate passes the independent offline confirmation only if all conditions hold:

1. Salt-Lake-network macro normalized action regret improves by at least 10% over the equally informed fallback;
2. both of the two Salt Lake networks improve strictly;
3. maximum network-level absolute regression is at most `0.05`.

Integrity failure is not a scientific failure and requires rerunning only the invalid collection or evaluation artifact under the same protocol. A valid outcome that fails any efficacy condition is reported as FAIL; it cannot be repaired by Salt-informed method changes.

## Closed-loop progression

Closed-loop Salt Lake control is run only after the offline confirmation passes. The closed-loop policy protocol and acceptance rule must be frozen in a separate document before the first final CFCMT rollout. If offline confirmation fails, Salt Lake closed-loop efficacy is not used to rescue the method.

The old v25/r21 Salt Lake stage described a zero-shot confirmation for an earlier method. That record remains immutable. This v38/r34 protocol governs only the newly frozen 60-group target-simulator-adapted pairwise method and does not retroactively change the v25 claim.
