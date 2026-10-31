# Chicago v107 route-time quantization amendment

## Trigger

The complete `route_package_v2` CH run finished routing all seven frozen demand
files, but post-routing admission stopped on vehicle
`d0_tnp_g0001316_r00000`. Its input departure was `964.285714` s and SUMO
1.22.0 wrote `964.29` s. This was a representation change, not a missing or
reordered vehicle.

## Full-output audit

Before changing the acceptance rule, the rejected combined route output was
streamed against all 1,422,405 frozen input trips on `node001`.

- matched vehicle IDs: 1,422,405 / 1,422,405;
- unmatched input IDs: 0;
- maximum absolute departure difference: 0.005455000035 s;
- records above 0.0050001 s: 592;
- worst record: `d3_tnp_g0008225_r00008`, input `276095.454545` s, output
  `276095.46` s.

SUMO emits routed departures at centisecond precision after parsing the
six-decimal trip times. The frozen v3 route protocol therefore admits an
absolute departure representation difference of at most 0.006 s. Vehicle
identity, day, source cohort, ordering, route endpoints, and vehicle counts
remain exact admission requirements. The maximum observed difference is
recorded per day in the route-package manifest.

## Consequence

`route_package_v2` remains rejected. A fresh, non-overwriting v3 package must
be generated under the amended protocol before safety simulation starts.
