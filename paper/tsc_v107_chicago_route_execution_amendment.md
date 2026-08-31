# TSC v107: Chicago route execution amendment

Amended: 2026-08-31 (Asia/Shanghai), before any route was produced and before
any safety or efficacy outcome was available.

The frozen v1 command terminated immediately under SUMO 1.22 with:

```text
Error: Routing algorithm 'CH' does not support bulk routing.
```

The rejected run is retained as
`route_package_v1.rejected/route_failure.json`. It produced no combined route
file. The corrected protocol is
`cfcmt-chicago-full-week-duarouter-ch-package-v2`.

The amendment removes only `--bulk-routing`. CH routing, eight routing threads,
seed `5307`, the complete seven-day demand, exact ID/departure/endpoint audits,
and every strict safety setting remain unchanged. No repair or ignored-error
option is introduced. This is an execution compatibility correction and does
not change data admission or any outcome threshold.
