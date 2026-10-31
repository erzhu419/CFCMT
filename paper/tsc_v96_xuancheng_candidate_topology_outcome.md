# Xuancheng pre-simulation route-topology outcome

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This outcome was reached before a Xuancheng SUMO package or controller rollout
was produced. No CFCMT, H2O+, source-selection, target-adaptation, policy,
reward, queue, delay, collision, teleport, or throughput outcome from Xuancheng
was computed or inspected.

## Evidence

The fixed 3 April candidate contains 377,322 source flow records and 9,418
unique consecutive anchor pairs. All anchor IDs occur in both the published
CityFlow roadnet and native SUMO network; the two files contain exactly the same
1,744 external road IDs.

- 2,040 unique anchor pairs are direct native SUMO connections.
- 7,376 additional pairs can be completed by deterministic length routing on
  the native SUMO graph while preserving every anchor.
- Two pairs are unreachable in the native directed graph. They occur in 259
  distinct source records (0.0686% of the day).
- Both failures start at edge `34180203073`, whose published native SUMO end is
  junction `J44` and which has no outgoing connection. The two following
  anchors are `34180201954` (258 records) and `34180200709` (one record).
- The endpoints lie in the same weakly connected network, but neither pair is
  direction-repairable using a published reverse edge. Preserving the sequence
  would require adding a network connector, teleporting, splitting trips, or
  dropping records.

## Decision

Xuancheng is rejected as the unseen closed-loop SUMO confirmation target. The
259 records will not be removed and the published target network will not be
repaired after this audit. Therefore the candidate does not proceed to safety
simulation, the remaining 29 large daily files are not downloaded, and no
controller efficacy result is generated.

The dataset remains suitable as an external real-AVI passive/offline source,
where the original observed anchor sequences can be retained without asserting
closed-loop route feasibility. Any such use must be reported separately from
SUMO counterfactual evaluation.
