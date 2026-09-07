# TSC v118r: ETH Boston route-execution amendment

Amended: 2026-09-01 (Asia/Shanghai), before any complete-demand route was
computed and before any Boston controller or efficacy outcome was observed.

The first complete-demand route-admission command terminated in 0.06 s under
SUMO 1.22 with:

```text
Error: Routing algorithm 'CH' does not support bulk routing.
```

The rejected compact result is retained as
`route_admission_v1.json`. Because duarouter rejected the option combination
before reading the 3,806,510 trips, this result says nothing about Boston demand
routability.

The corrected protocol is
`eth-five-city-complete-demand-duarouter-admission-v2`. It removes only the
unsupported `--bulk-routing true` option. CH routing, 20 routing threads,
TAZ-based routing, the complete canonicalized demand, zero route repair, and
`ignore-errors=false` remain unchanged. The package manifest, trip identities,
departures, endpoint anchors, and all scientific outcome gates are unchanged.

This is an execution-compatibility correction matching the already validated
Chicago CH-routing amendment. A fresh, non-overwriting v2 admission result is
required before any microscopic Boston preflight is authorized.
