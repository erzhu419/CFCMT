# TSC v102: ETH Lisbon unseen-city input outcome

## Frozen source

- Dataset: *Traffic Simulations for Boston, Lisbon, Los Angeles, Rio de
  Janeiro, San Francisco*, DOI `10.3929/ethz-b-000584669`.
- Complete archive: 30 entries and 658,810,696 bytes, acquired directly to
  shared cluster storage.
- Observed archive MD5:
  `802321f1d2c61de527ed579ce9c54842`, equal to the repository ETag/MD5.
- License: CC BY-NC 4.0.
- Code snapshot: `4532d84b6fb7058494f8`, full snapshot SHA-256
  `4532d84b6fb7058494f89657d075972dd441b1a64ab2df54d94e4d4f83fb1ecd`.

## Lisbon input audit

The complete `LIS/` target unit contains the original network, route demand,
additional file, TAZ file, and mesoscopic configuration. Static streaming
validation found:

- 769,834 unique trips and no duplicate identifiers;
- no missing required trip attributes;
- no trip origin/destination edge outside the published network;
- no missing trip TAZ reference;
- 100,909 edges, 44,235 junctions, and 1,199 traffic lights with 6,265 phases;
- monotonically ordered departures from 7,214.42 to 50,393.47 seconds.

The topology, demand references, and signal inventory are therefore correct.
The rejected condition is temporal scope: the departure span is only
43,179.05 seconds (approximately 12 hours), not the full-day interval required
by the frozen unseen-city protocol. The authors' supplied configuration is
also explicitly mesoscopic and runs only 14,400--49,800 seconds with
`ignore-route-errors=true`, `collision.action=none`, and a 360-second teleport
threshold. It cannot establish microscopic controller safety.

A streaming screen of the other four route files found the same scope:
BOS 7,201.78--50,395.74, LAX 7,208.84--50,393.08, RIO
7,225.11--50,391.75, and SFO 7,213.23--50,399.69 seconds. None of the five
archive cities can replace Lisbon under the frozen full-day gate.

## Decision

**REJECT before simulation.** No Lisbon microscopic collision/teleport run,
CFCMT evaluation, baseline evaluation, source selection, or target adaptation
is authorized. The data remain suitable for an explicitly labeled
approximately 12-hour mesoscopic or auxiliary stress test, but not for the
independent full-day microscopic confirmation.

This rejection does not alter the v98 result: Jinan's B100 offline selector
chose the Hangzhou causal component and improved all 56 fresh confirmation
seeds, while Los Angeles fell back to strict target-only. It only leaves the
independent unseen full-day city gate unresolved.
