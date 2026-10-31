# RESCO SUMO Benchmark Probe

Source: https://github.com/Pi-Star-Lab/RESCO.git

This probe downloads selected RESCO scenarios, unpacks referenced route files, and verifies SUMO/libsumo startup. It does not run CFCMT policy evaluation yet.

| scenario | ok | tls_count | trips | vehicles | flows | min_expected_start | departed_probe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| grid4x4 | True | 16 | 0 | 1473 | 0 | 2 | 1 |
| arterial4x4 | True | 16 | 0 | 2484 | 0 | 5 | 4 |
| ingolstadt21 | True | 21 | 4283 | 0 | 0 | 2 | 2 |
| saltlake2_400sX200w | True | 2 | 0 | 0 | 0 | 0 | 0 |
| saltlake2_stateXuniversity | True | 2 | 0 | 0 | 0 | 0 | 0 |

## Next Integration Step

Use the probed TLS phase programs directly instead of the synthetic NS/EW phase assumption. For arbitrary RESCO intersections, actions should select among existing green phases from the SUMO program logic.
