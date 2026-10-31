# TSC v42/r38 Pre-efficacy Nanchang Demand-order Correction

## Discovery

The v4 full-horizon admission reached 3,600 s without collisions or teleports,
but its TraCI counters reported only 1,692 loaded Nanchang vehicles. The frozen
CBEngine input contains 9,786 flow definitions representing 126,669 vehicles.
Inspection showed that source flow definitions are not ordered by `begin`,
whereas SUMO incrementally reads route files and requires vehicles and flows to
be ordered by departure or begin time. The v4 result therefore proves runtime
stability only for a partially loaded demand file and is not admissible.

This was detected before any external target cache, CFCMT fit, selector
decision, action-regret result, or closed-loop efficacy result was generated.

## Frozen correction

The v5 converter now expands each CBEngine flow interval using the source's
inclusive start-to-end semantics. It writes all 9,786 named routes first and
then writes 126,669 explicit vehicles globally sorted by departure time. This
removes ambiguity in SUMO flow-end semantics and makes the exact vehicle count
part of the conversion manifest.

The v2 admission gate no longer treats a completed rollout or a positive TraCI
counter as proof of full demand loading. It requests SUMO summary output and
requires the final cumulative `loaded` count to equal the manifest's exact
`vehicle_count`. Summary collisions and teleports are checked in addition to
the step-level TraCI incident audit.

## Provenance and exclusion

- v4 config SHA-256:
  `badbac619c6c1268c7c0ef0042e25fa369db86e3c464021f611ac2ccd6466cd0`
- v4 converter SHA-256:
  `4d35b9d775ea5c75586382abba1bbf5917063fbdf12092f4363f989fd4a13db8`
- v5 converter SHA-256:
  `c3f2c35365696b35ee1f920ec2fe1a1a54315aee05406485d0b19143b30bd95a`
- v2 admission implementation SHA-256:
  `9a437dbe0a36963a04603955d579e4e13b266ce9d61a8436b5cdab30c576ea0d`
- excluded v4 conversion root:
  `cf_h2o/results/cluster/tsc_v42r38_external_la_nanchang_20260809/conversion/full_networks_v4`
- canonical v5 candidate root:
  `cf_h2o/results/cluster/tsc_v42r38_external_la_nanchang_20260809/conversion/full_networks_v5`

All earlier Nanchang admission artifacts remain engineering diagnostics only.
They are prohibited from cache construction and efficacy evaluation.
