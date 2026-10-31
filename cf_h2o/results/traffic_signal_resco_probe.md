# RESCO SUMO Benchmark Probe

Source: https://github.com/Pi-Star-Lab/RESCO.git

This probe downloads selected RESCO scenarios, unpacks referenced route files, and verifies SUMO/libsumo startup. It does not run CFCMT policy evaluation yet.

| scenario | ok | tls_count | trips | vehicles | flows | min_expected_start | departed_probe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cologne1 | True | 1 | 2015 | 0 | 0 | 1 | 8 |
| cologne3 | True | 3 | 0 | 4494 | 0 | 2 | 25 |
| cologne8 | True | 8 | 2046 | 0 | 0 | 3 | 26 |
| ingolstadt1 | True | 1 | 1716 | 0 | 0 | 1 | 14 |
| ingolstadt7 | True | 7 | 3031 | 0 | 0 | 1 | 30 |

## Next Integration Step

Use the probed TLS phase programs directly instead of the synthetic NS/EW phase assumption. For arbitrary RESCO intersections, actions should select among existing green phases from the SUMO program logic.
