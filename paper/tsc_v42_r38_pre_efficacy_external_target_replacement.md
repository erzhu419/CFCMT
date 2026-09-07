# TSC v42/r38 pre-efficacy external-target replacement

Frozen: 2026-08-09 03:14:09 UTC
TLS-safe amendment: 2026-08-09 03:50:37 UTC

## Decision

The formal external confirmation targets are changed from Los Angeles plus
Nanchang to Los Angeles plus Jinan. This amendment precedes every external
CFCMT efficacy fit, selector decision, held-out evaluation, and closed-loop
comparison. No target cost, reward, regret, or controller-performance metric
was inspected during target replacement.

The replacement criterion is simulator integrity only: the unmodified full
network must load its complete demand and expose controllable signals, with
zero unique physical collision incidents and zero teleports under the strict
SUMO junction-collision protocol. The collision detector was not disabled and
the network geometry was not relaxed to make any candidate pass.

## Rejected candidates

### Nanchang

Both 3,600 s SUMO 1.22 admissions rejected the converted CBEngine network:

| Seed | Unique collision incidents | Teleports | Artifact SHA-256 |
|---:|---:|---:|---|
| 5057 | 886 | 0 | `88ce4d8b5c8f687e0d4ddb50df133eecf370c29b5e576a133fac675beeb7ec7d` |
| 6067 | 870 | 0 | `5bece33ab7ba83597cda70907a2e6063870b25222e47401ffa9aadf526b42442` |

A separately frozen 600 s diagnostic found 23 unique incidents, beginning at
89 s. Lane-link and demand-order corrections did not remove the conflicts.
Inspection showed that CBEngine's symbolic approach ordering can disagree with
the physical geometry required by SUMO, so the conversion is not an admissible
counterfactual environment.

### Monaco MoST and Bologna small

These maintained native SUMO candidates were screened locally with SUMO 1.27.1
and the same strict junction-collision settings before any efficacy analysis.
MoST produced 13 unique incidents in a 600 s peak-period probe. Bologna
`acosta`, `joined`, and `pasubio` produced 1, 14, and 6 incidents,
respectively. None produced a teleport. They are valid published mobility
scenarios under their own execution settings, but they do not satisfy this
paper's collision-free counterfactual branching contract.

## Replacement target

Jinan 3x4 is taken from the public CoLight repository at full commit
`04440b722e7f36d7043a3f2ad8e40c3df301fad3`. The city has 12 controlled
intersections. A 600 s local preflight of the primary real flow loaded 1,136
vehicles and produced zero collisions and zero teleports. Formal admission is
still required under the frozen SUMO 1.22 runtime and two independent seeds.

All three published Jinan real-flow variants are retained:

| Scenario | Explicit vehicles | Source SHA-256 |
|---|---:|---|
| `jinan_3x4_real` | 6,295 | `233739633ef0b637125cb304dfffff9488503bac6e6861ca39243cc8ffdcebd5` |
| `jinan_3x4_real_2000` | 4,365 | `d0931c1b759479f9e748d69c16414020bfba03d555dcc6cdf7223a0c18cb9e69` |
| `jinan_3x4_real_2500` | 5,494 | `4245107cc7ce91b9699519f2c1258be4397e83291080a43cb0b93479557739cd` |

The city, rather than an individual flow file, is the adaptation and reporting
unit. Sampling is coverage-first across all three variants and evaluation uses
equal scenario weight within Jinan. Los Angeles and Jinan never enter each
other's source pool.

## Frozen continuation

1. Convert LA and all three Jinan variants with netconvert 1.22 and preserve
   exact source hashes and relocatable paths.
2. Admit every scenario under seeds 5057 and 6067 for the full 3,600 s.
3. Open only the two adaptation-seed banks and freeze the five-fold B100 model
   ensemble plus the preselected global selector.
4. Collect seed 7079 only after the model artifact is immutable, then run the
   offline city-macro gate.
5. Run seeds 8081 and 9091 closed-loop only if the offline gate passes.

The governing machine-readable protocol is
`cf_h2o/config/traffic_signal_tsc_v26_external_la_jinan_confirmation.json`.

## TLS-safe pre-efficacy amendment

The first SUMO 1.22 conversion, scheduler task `t77233`, is rejected before
efficacy analysis. `netconvert` reordered controlled connection `linkIndex`
values while the supplied CityFlow phase strings retained their provisional
ordering. Consequently, all 14 LA phases and all 96 Jinan phases disagreed
with their source road-link movement sets. The defect was exposed by the
strict admission: Jinan real flow at seed 6067 produced one junction collision
at 815 s. Its rejected admission SHA-256 is
`2eed99d2ebd7d9dfe2dff4109aa15617f6fd8aa3d1a905f93fe2abb6c3aca14b`.

No delay, queue, throughput, reward, prediction-error, or controller metric
from the external targets had been inspected. Therefore this is an input
semantics correction, not a result-conditioned model change. The v7 converter
now reconstructs each phase after `netconvert` using the final lane-connection
identity, represents phase-0 right turns as permissive `g`, and inserts a
physical yellow intergreen. Every v7 network must report zero post-remap phase
link difference before the two full-horizon collision/teleport admissions.
No v6 artifact may enter offline fitting, held-out evaluation, or closed-loop
control.

The first remapped v7 conversion was also rejected before efficacy analysis.
Seven of eight full-horizon admissions passed, while LA seed 6067 exposed two
same-movement merge collisions. The CityFlow lane-link graph permits several
incoming lanes to target the same outgoing lane; assigning all such links
protected green suppresses the merge priority that `netconvert` encoded as
one `O` and multiple `o` connections. The v8 amendment retains at most one
`G` per target lane and encodes the remaining merge links as yielding `g`.
Its local 3,600 s LA/6067 preflight produced zero collisions, teleports, and
SUMO safety warnings. The rejected v7 conversion hashes are
`f8438f14135dff710f197516cf934a2f07dc48842229137f5340113e6f1997a8`
(manifest) and
`1990ea83c813fb869dfab5ea211e2d7a013e9f92356b17f05cc5807c713ccfdb`
(tree). No v7 result may enter efficacy fitting or evaluation.
