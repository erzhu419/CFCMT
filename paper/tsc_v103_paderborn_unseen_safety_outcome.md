# TSC v103: Paderborn independent unseen-city safety outcome

## Frozen inputs

- Dataset: *Paderborn Traffic Scenario* v0.1, Zenodo record `4522059`.
- Frozen package manifest SHA-256:
  `80359be590a12a120004e02082c28ef348b9f5865e5b0182ba1a42f13c9b1955`.
- Runtime source snapshot: `c96c9f09a4dc581d5d24`, source-tree SHA-256
  `85ccfea748eb8a1ca431086af729833be70deee9ac8d03ffac652f2852072c9e`.
- Runtime: SUMO/libsumo 1.22.0, seed 5057, complete network and all
  203,387 precomputed full-day routes.

The static package passed every frozen input gate: all route and stop
references exist, trip and route identifiers match exactly, departures span
0--86,400 seconds, and 129 traffic lights are available at runtime. No route,
vehicle, departure interval, traffic-light program, edge, or vehicle type was
removed or rewritten.

## First irreversible safety event

The strict run stopped at the first junction collision. SUMO reported the
collision at simulation time 163 seconds; libsumo exposed it after the step at
time 164 seconds:

- collider: `randUni16597:1`;
- victim: `h36194c1:1`;
- internal lane: `:25323933_6_0`;
- collision type: `junction`;
- gap: -1.00 m in the SUMO warning.

At termination, 547 vehicles had departed, 2 had arrived, 545 were active,
9 were pending insertion, and 573 remained expected. There were zero starting
or ending teleports and no runtime exception. These partial counts are not
completion failures discovered at the fixed cap; they follow directly from
the pre-registered fail-fast collision rule.

## Decision

**REJECT before CFCMT evaluation.** The zero-collision gate failed. Paderborn
does not enter source selection, target adaptation, controller rollout, or
baseline comparison. The network, vehicle parameters, junction connections,
collision checks, simulator version, and threshold are not revised after this
result.

This rejection does not overturn the v93/v98 positive Jinan result. It shows
that Paderborn cannot serve as the independent unseen-city confirmation under
the manuscript's strict microscopic safety contract.
