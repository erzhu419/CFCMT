# TSC v104: Nagoya NUMo independent unseen-city input outcome

## Frozen source

- Repository: `ToyotaInfoTech/numo` at commit
  `5409c4696ad9b4d81e0665d1de51e327beda0505`.
- Fixed-commit archive: 236,759,921 bytes, SHA-256
  `9c2f57df9d15d120ca874ad17365fc414650f9a22ef60f2c8d152415b483e442`.
- Extracted source: 55 files and 1,513,457,275 file bytes, including the
  complete network, WAUT signal schedule, native configuration, and all 48
  named half-hour route files.

## Count gate

The frozen protocol required exactly 1,627,151 unique route vehicles, matching
the repository's published full-day departed-vehicle statistic. A complete
count of `<vehicle>` opening elements across the 48 route files found
1,627,148, three fewer than required. The final file extends to vehicle ID
`1627154`, so the discrepancy is not a simple truncated final half hour.

No uniqueness assumption can recover the gate: 1,627,148 total vehicle
elements is an upper bound on the number of unique vehicles. The discrepancy
may reflect a difference between the route files at the frozen commit and the
simulation run used for the README statistic, but resolving that provenance
question would require changing the pre-registered input contract.

## Decision

**REJECT before simulation.** The exact published-count gate failed. The
unique-ID scan, route-edge scan, SUMO 1.22 strict package, no-teleport safety
run, source selection, CFCMT evaluation, and baseline evaluation were not run
after this irreversible failure. No count tolerance or corrected v104 target
is introduced post hoc.
