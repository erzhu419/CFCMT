# Xuancheng anchor-route and unseen-target safety protocol

Date frozen: 2026-08-31 (Asia/Shanghai)

## Claim boundary

This amendment precedes every Xuancheng controller efficacy result. It governs
source-data interpretation, deterministic route construction, and simulator
safety admission only. No CFCMT, H2O+, source-selection, target-adaptation,
reward, queue, delay, or throughput result from Xuancheng has been computed or
inspected.

## Frozen source

- Figshare article `29925824`, version 5, DOI
  `10.6084/m9.figshare.29925824`, CC BY 4.0.
- Published native SUMO network: `xuancheng.net.xml`, source MD5
  `1cb2ca8504b64267170c82c3d4633230`.
- Published CityFlow roadnet: `roadnet_xuancheng250319.json`, source MD5
  `c2a12f2005f5f53c8182213614b0eda4`.
- Candidate day: `data_2023_04_03_type_filtered.json`, source MD5
  `68d0515673c964f9152d5e50f7b101da`. This day was fixed because the authors'
  published Xuancheng configuration names 3 April, before any SUMO safety or
  controller outcome was available.
- If the candidate passes, the full experiment must include every published
  day from 1--30 April 2023. No route, time interval, or low-demand day may be
  omitted.

## Anchor-route correction

The first topology audit treated each CityFlow `route` array as an already
expanded SUMO path. That interpretation was rejected before simulation because
7,378 unique consecutive anchor pairs were not direct native SUMO connections.
The road IDs themselves matched exactly: all 1,744 CityFlow roads were present
in the native SUMO network.

CityFlow's published flow specification defines `route` as source, destination,
and optional anchor roads; its router connects consecutive anchors with shortest
paths. A first source-compatible implementation used only CityFlow `roadLinks`.
It stopped before producing a package because one observed anchor pair was not
reachable in that reduced graph. The same pair is reachable in the native SUMO
network published with Figshare version 5. Because the target simulator is SUMO,
the released SUMO topology is authoritative for conversion. The frozen protocol
is therefore `native-sumo-length-dijkstra-cityflow-anchor-completion-v2`:

1. Use the released SUMO edges, connections, XML order, and average native lane
   length with strict Dijkstra relaxation.
2. Complete every consecutive anchor pair and retain every source anchor in its
   original order.
3. Materialize the result as an explicit SUMO route before simulation.
4. Reject the entire day if any anchor is absent, any anchor pair is
   unreachable, or any completed pair is absent from the native SUMO
   connections.
5. Keep all source flow records and inclusive CityFlow departures. The three
   records at 86,400 s are retained, so the half-open SUMO horizon is 86,401 s.

This is a source-format and target-topology instantiation, not demand
calibration. The published SUMO network and its 132 traffic-light programs
remain byte-for-byte unchanged. No route is selected using a controller outcome,
queue, delay, collision, or reward.

## Candidate safety gate

The complete 3 April demand must be simulated for 86,401 s with libsumo and
development seed `5057`. Admission requires:

- all source flow records expanded and loaded;
- all 132 native SUMO traffic-light programs exposed;
- the full horizon reached;
- zero unique collision incidents; and
- zero teleports.

Failure terminates Xuancheng before downloading the remaining 29 large daily
files or running any controller. The network, TLS programs, vehicle parameters,
collision threshold, and teleport threshold will not be altered after this
screen.

## Full-month gate after candidate admission

If and only if the candidate passes, acquire and verify all 30 daily files.
Every day must pass the same full-horizon, zero-collision, zero-teleport gate
under the eight development seeds `5057, 6067, 7103, 8171, 9277, 12001, 13007,
14009`. The unchanged package is then checked with disjoint seeds `15101, 16001,
17011, 18013, 19001, 20011, 21013, 22003`. All 30 days must remain admitted;
otherwise Xuancheng is excluded from controller efficacy analysis.
