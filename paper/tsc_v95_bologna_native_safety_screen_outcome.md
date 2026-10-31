# Bologna native SUMO pre-efficacy safety-screen outcome

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This outcome is a simulator-safety screen only. No CFCMT, H2O+, source
selection, target adaptation, policy reward, queue, delay, or throughput result
from Bologna was computed or inspected.

## Frozen source and package

- Source: the complete `bologna/` directory from DLR-TS `sumo-scenarios` at
  commit `00f6eb479a9dc0fbaeb731495c11d48d7a9661d3`.
- Coverage: all four published subscenarios (`acosta`,
  `acosta_persontrips`, `joined`, and `pasubio`) and all 39 source files.
- Package protocol: `cfcmt-dlr-bologna-native-sumo-package-v1`.
- The source networks, traffic-light programs, and demand files were preserved
  byte for byte. Passive detector/output definitions were removed from the
  canonical run configuration; traffic semantics were neither rebuilt nor
  calibrated.
- Package manifest SHA-256:
  `76e10f5dc9b3b568289e56e243a3fe4e7d0c4424a928823977ce3638eb35275b`.
- Package tree SHA-256:
  `531dfddf465a84edae136e2de797c73bd14ed0b7b5e162d214672140e4b4659b`.

## Fixed screening protocol

Each scenario was run for the full native 3,600 s horizon with libsumo under
the current SUMO 1.22 execution environment and the first development seed,
`5057`. Admission required the full horizon, exact deterministic demand
loading, all declared controllable traffic signals, zero teleports, and zero
unique collision incidents. A failure at this topology-only screen terminates
the candidate before any controller efficacy experiment or additional seeds.

## Observed result

| Scenario | Road vehicles | Persons | TLS | Unique collisions | Teleports | Result |
|---|---:|---:|---:|---:|---:|---|
| `bologna_acosta` | 8,779 | 0 | 7 | 24 | 0 | Reject |
| `bologna_acosta_persontrips` | 8,770 | 7,200 | 7 | 162 | 0 | Reject |
| `bologna_joined` | 11,176 | 0 | 13 | 132 | 0 | Reject |
| `bologna_pasubio` | 8,776 | 0 | 8 | 30 | 0 | Reject |

All four simulations reached the full horizon, loaded the expected demand,
exposed the expected traffic signals, and reported no teleports. Every scenario
failed only the predeclared zero-collision requirement. The first observed
collision occurred at 455 s for `acosta`, 618 s for `acosta_persontrips`, and
62 s for both `joined` and `pasubio`.

## Decision

Bologna is rejected as the new-city confirmation target. No multi-seed safety
matrix and no efficacy evaluation will be run. The zero-collision threshold is
not relaxed, and the native traffic programs will not be repaired after seeing
this outcome. The failed package and four admission JSON files remain on the
shared server as provenance evidence.
