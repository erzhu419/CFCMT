# TSC v18 Salt Lake Detector-Demand Construction Protocol

Date fixed: 2026-08-08, after auditing the downloaded RESCO Salt Lake files and before generating or evaluating any Salt Lake controller rollout.

## Purpose

The current controlled benchmark contains 16 SUMO networks grouped into five observed cities and one synthetic domain. RESCO also distributes two Salt Lake City two-intersection networks and headerless five-minute movement-count files. Their static route templates contain no vehicles, while the upstream conversion utility introduces stochastic rounding, random departure times, and UUID identifiers. Those choices prevent byte-level reconstruction and can silently discard demand assigned to an unavailable downstream turn.

This protocol freezes a deterministic, count-conserving construction before Salt Lake is admitted to the cross-network benchmark. Controller outcomes play no role in selecting dates, time windows, movements, routes, or seeds.

## Frozen primary demand profile

- Source maps: `saltlake2_400sX200w` and `saltlake2_stateXuniversity`.
- Detector signals: 7241/7242 and 7243/7142, respectively.
- Source period: 2023-01-01 through 2023-03-31, inclusive.
- Eligible dates: Monday through Friday only; a date is used only when both signals have 288 valid five-minute rows with exact timestamps and non-negative integer counts.
- Robust profile: coordinate-wise median count for every signal, approach, movement, and five-minute time slot across all complete eligible dates.
- Primary window: the earliest aligned one-hour window maximizing pooled external-origin arrivals across both robust map profiles.
- Pre-generation audit result: all 65 eligible weekdays are complete and the frozen pooled peak is 17:00-18:00. Each map independently has its largest robust external-arrival total in the same hour.
- Simulation clock: source 17:00 is shifted to simulation time 0; the primary episode is 3,600 seconds.

The 08:00-09:00 fixed window will be retained as a preregistered temporal sensitivity context after the primary two-network integration passes. It will not be used to choose or tune the method.

## Frozen route construction

1. The omitted CSV headers are reconstructed from the signal-specific movement order distributed in RESCO's `csv_to_flo.py`.
2. `TR` detector columns are split between through and right movements by deterministic Hamilton apportionment.
3. External movements with direct routes retain their exact integer profile count.
4. Movements crossing the adjacent signal are split using the concurrent internal detector's downstream turn proportions.
5. Downstream proportions are normalized only over routes that exist in the supplied SUMO template. A zero-count interval falls back to the selected-hour aggregate proportions; a fully unsupported origin is an error.
6. Every five-minute interval must conserve the rounded external-origin count exactly.
7. Departures are evenly distributed within each interval using a stable hash phase. Vehicle identifiers, tie breaks, and output ordering are deterministic under seed 20260808.
8. The original network, route template, license, every included CSV, generator source, generated route file, generated configuration, and movement profile receive SHA-256 provenance records.

Construction protocol identifier: `q1-weekday-coordinate-median-pooled-peak-deterministic-apportionment-v1`.

## Admission gates

Both maps must pass all gates before a candidate benchmark manifest is created:

- exactly 65 included profile dates and no excluded eligible date;
- exact timestamp, column-count, integer, and non-negativity validation for every included CSV row;
- exact interval-level and episode-level external-demand conservation;
- every generated vehicle references an existing, connected route;
- two independent builds have identical route, configuration, movement-profile, and provenance hashes;
- SUMO 1.22.0 loads the network and all generated demand without XML, route, collision, or teleport errors under the frozen no-teleport protocol;
- the candidate manifest contains 18 physical networks in seven leave-one-city groups, with both Salt Lake networks assigned to one `salt_lake_city` group.

Salt Lake source counterfactuals and source-policy rules will be built only after these data gates pass. The existing 16-network cache is immutable and will not be relabeled as an 18-network result.
