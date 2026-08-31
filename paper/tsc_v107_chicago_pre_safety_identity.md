# TSC v107: Chicago pre-safety identity freeze

Date frozen: 2026-08-31 (Asia/Shanghai), after full-week routing passed and
before any Chicago microscopic safety result was observed.

This record instantiates the sequential gate in
`tsc_v107_chicago_for_hire_unseen_protocol.md`. It does not change the public
data window, network, demand, route, simulator, completion, collision, or
teleport requirements. The rejected v1 and v2 route packages remain rejected.

## Admitted route package

- Route protocol: `cfcmt-chicago-full-week-duarouter-ch-package-v3`.
- Server root:
  `/home/zhengliang01/scheduleurm_work/CFCMT_DATA/chicago_v107_d5774eb/route_package_v3`.
- Route manifest SHA-256:
  `c6380a9f826f93ead3383d5bffccc5c83cd0846f2e06099d0a3feaa18f86744c`.
- Network SHA-256:
  `45250772dcff2cd79bd4472e0657d09c20eb32be280a1099d5ba93b831ac42ae`.
- Demand manifest SHA-256:
  `835fa16ccc50395505c95fa2c894321b80069f2829348952e9fb3e4b0e87cff1`.
- SUMO version: `1.22.0`; router: `duarouter`, CH, eight threads, seed `5307`.
- Routing completed with return code zero, no ignored errors, no route repair,
  and no dropped included trip.

The package contains exactly 1,422,405 routed vehicles across all seven frozen
days. Daily counts from 2023-08-21 through 2023-08-27 are 158,635; 168,988;
194,876; 216,712; 231,326; 258,294; and 193,574. These sum exactly to the
included demand count. Vehicle identity, source, day, endpoint edges, order,
and counts passed exact post-routing checks. SUMO's centisecond route-output
format produced a maximum departure representation difference of
`0.005455000035 s`, below the independently amended `0.006 s` tolerance.

## Frozen microscopic gate

The admission implementation is
`cf_h2o/eval/traffic_signal_chicago_full_week_admission.py` with protocol
`cfcmt-chicago-full-week-strict-libsumo-safety-admission-v1`. It runs all seven
days using libsumo/SUMO 1.22.0 and seeds `5201` through `5207`, with at most one
libsumo process on each of `node001` through `node006`.

Each day uses the complete routed demand, all 4,960 traffic lights, one-second
steps, demand scale one, fatal route errors, `max-depart-delay=-1`, disabled
time-based teleportation, junction collision checking, and a 108,000-second
completion cap. A collision or teleport stops and rejects that day
immediately. Otherwise, the day passes only if all expected vehicles are
loaded, depart, and arrive exactly, with no active, pending, or expected
vehicle remaining. Chicago is admitted only if every one of the seven days
passes. No Chicago safety or efficacy result is asserted by this freeze.
