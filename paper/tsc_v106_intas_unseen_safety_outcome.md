# TSC v106: InTAS independent unseen-city safety outcome

Date: 2026-08-31 (Asia/Shanghai)

## Decision

**REJECT InTAS before efficacy evaluation.** The first and only frozen
microscopic admission encountered a junction collision. Under the v106
sequential protocol, InTAS cannot enter source selection, zero-shot, B100, or
closed-loop controller comparisons, and the network, demand, seed, simulator,
or safety gate will not be amended and rerun.

## Input admission

The complete fixed InTAS Git tree and static package passed input admission:

- repository commit:
  `0f7951ba01dda8483f0a852f2c3e4ff0d8a1c0ee`;
- package manifest SHA-256:
  `4f79a34946addde7efd0ba88b4a3318716de907f8a8b5d30b50ef096771fc282`;
- 22 road-demand files, the complete bus flow file, and the complete pedestrian
  file;
- 185,923 road vehicles, 1,581 expanded bus vehicles, and 254 persons;
- 23,648 edges, 33,204 lanes, 4,687 junctions, and 98 traffic lights; and
- no static identity, reference, or full-day coverage failure.

Thus the rejection is not a missing-file, sampled-demand, or conversion
failure.

## Frozen execution

The run used immutable source snapshot
`6263d96157e4899322097e6022d7a46c09c74305d09052620bdf1c2bb913da50`,
source-tree identity
`9d32447ee59f9887ee8f4ddd97dff74d9b0f34463a4bfb908f0be61a03131594`,
libsumo/SUMO 1.22.0, Python 3.10.20, seed 5107, the native 0.1-second step,
demand scale one, all 24 demand sources, all 98 traffic lights,
`max-depart-delay=-1`, and time-based teleportation disabled. Collision and
teleport events were inspected after every simulation step.

At SUMO event time 184.80 seconds, the simulator reported:

- collider: `carIn105456:1`;
- victim: `carIn101117:1`;
- type: junction collision;
- lane: `:2420195934_0_0`; and
- reported gap: -1.00 m.

The libsumo collision object was collected at simulation time 184.9 seconds at
lane position 8.4519670722 m. There were no starting or ending teleports, no
runtime exception, and the 98-signal inventory matched exactly. The admission
terminated in 8.27 wall-clock seconds because the collision gate had already
failed; this duration is not a full-day performance measurement.

## Consequence

The failed completion and demand-conservation checks are consequences of the
mandatory early stop, not additional evidence of source loss. The decisive
gate is `zero_collisions=false`. No CFCMT or baseline efficacy result exists for
InTAS under v106, and none may be claimed from this run.
