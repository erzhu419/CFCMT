# TSC v39b/r35 Capacity-Admission Amendment

Date frozen: 2026-08-09. This amendment was written after v39a exposed a deterministic target-data capacity error and before any v39 candidate regret, selected action, candidate ranking, or city-level outcome was read. It modifies target eligibility only; it does not modify the anchored-pairwise method or its success gate.

## Why v39a is invalid

The original preregistration, SHA-256 `18f9bb2bf0718cdfd59cb1c8bf5d346f7256f08607c0667bd4e9ccc33546e976`, requires exactly 60 safe complete counterfactual action groups for every target network. Submission record `0089d43408a40f8269fc4240fed0a99c14ea3043457efc0027cd28bfa783eeb0` attempted all 18 networks.

Four Hangzhou single-intersection networks failed before model evaluation because their entire frozen banks contain fewer than 60 safe complete groups:

| Network | Available safe complete groups | Required groups |
|---|---:|---:|
| `hangzhou_bc_tyc` | 25 | 60 |
| `hangzhou_kn_hz` | 42 | 60 |
| `hangzhou_qc_yn` | 33 | 60 |
| `hangzhou_sb_sx` | 36 | 60 |

The capacity audit is independent of candidate predictions and costs. Its SHA-256 is `2d22a844fd2c0efecd1fc621b494ffd6bde92eeed45a518564daa543b5e2b407`. The incomplete v39a output is archived under `cf_h2o/results/cluster/tsc_v39r35_anchored_pairwise_20260809/anchored_development_invalid_capacity`; no completed v39a target result may be reused in v39b.

## Frozen correction

Target eligibility now requires at least 60 safe complete counterfactual groups before any method is fitted. The four insufficient Hangzhou networks remain available as source data for non-Hangzhou targets but are source-only and cannot contribute target metrics. The two Hangzhou 4x4 networks remain eligible, so all seven development city groups remain represented.

All 14 eligible targets are rerun from a new immutable source snapshot and a new result root. Each target still receives exactly 60 deterministic complete action groups. All remaining safe complete groups are evaluation-only.

## Unchanged method and gate

The following v39 clauses are unchanged:

1. group-normalized causal anchor and antisymmetric pairwise correction;
2. all 11 constant shrinkage candidates and 20 local shrinkage candidates;
3. identical B=60 data for both base models;
4. same-city source holdout using the full 18-network manifest;
5. network averaging within city followed by equal city weighting;
6. all LOCO and global 2%, breadth, and maximum-regression thresholds;
7. one globally frozen candidate if and only if both development gates pass.

Los Angeles and Nanchang remain sealed external holdouts. No CFCMT result from either city may be generated or inspected before v39b passes and a separate external-confirmation protocol is frozen.
