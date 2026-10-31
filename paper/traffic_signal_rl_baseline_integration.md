# Traffic-Signal RL Baseline Integration

This note records the current state of RESCO/LibSignal RL baseline integration
for the traffic-signal-control branch.

## Official RESCO Adapter

Implemented:

- Script: `cf_h2o/eval/traffic_signal_resco_official_rl_baseline.py`
- Test: `cf_h2o/tests/test_traffic_signal_resco_official_rl_baseline.py`
- Source checkout: `H2Oplus/downloads/traffic_signal_resco/repo`
- Supported official RESCO algorithms in the adapter: any algorithm profile
  accepted by official RESCO, with pilot runs for `IDQN` and `MPLight`.
- Parallel execution: `--workers N` launches independent official RESCO
  subprocesses.
- Resume support: each run writes a per-run summary under `--cache-dir`; by
  default, successful cached runs are reused and failed cached runs are retried.
- SUMO route compatibility: `grid4x4` and `arterial4x4` use explicit
  `.rou.xml` route overrides because the official profiles refer to extensionless
  route names in this checkout.  The adapter now creates those override files
  automatically from the official scenario zip archives when needed.

Important protocol boundary:

- Official RESCO agents use RESCO's own state, reward, action, and logging
  definitions.
- These rows are leaderboard-style cross-checks, not same-protocol rows for the
  CFCMT phase-MPC table.

Pilot run:

```bash
python3 cf_h2o/eval/traffic_signal_resco_official_rl_baseline.py \
  --scenarios cologne1,cologne3,ingolstadt1 \
  --algorithms IDQN,MPLight \
  --episodes 1 \
  --testing 1 \
  --timeout-sec 180 \
  --log-dir cf_h2o/results/resco_official_rl_pilot \
  --out cf_h2o/results/traffic_signal_resco_official_rl_pilot.json \
  --md-out cf_h2o/results/traffic_signal_resco_official_rl_pilot.md
```

Pilot result file:

- `cf_h2o/results/traffic_signal_resco_official_rl_pilot.md`

The pilot confirms end-to-end execution and parsing on three RESCO scenarios,
but one training episode is not a trained RL baseline.

Short-budget extended check:

```bash
python3 cf_h2o/eval/traffic_signal_resco_official_rl_baseline.py \
  --scenario-set extended \
  --algorithms IDQN,MPLight \
  --episodes 5 \
  --testing 2 \
  --timeout-sec 1800 \
  --workers 8 \
  --log-dir cf_h2o/results/resco_official_rl_short_budget \
  --out cf_h2o/results/traffic_signal_resco_official_rl_short_budget.json \
  --md-out cf_h2o/results/traffic_signal_resco_official_rl_short_budget.md
```

The route-fix rerun for `grid4x4` and `arterial4x4` completed with zero
failures:

- `cf_h2o/results/traffic_signal_resco_official_rl_short_budget_grid_fix.md`

Current short-budget status:

- `IDQN` runs successfully on all eight extended RESCO scenarios after the route
  override.
- `MPLight` runs successfully on `grid4x4`, `arterial4x4`, `cologne1`,
  `cologne3`, `ingolstadt1`, and `ingolstadt7`.
- `MPLight` still fails on `cologne8` and `ingolstadt21` with an upstream PFRL
  shared-DQN indexing error.  Treat this as an official implementation boundary
  unless a separate RESCO patch is introduced and documented.

Mid-budget diagnostic:

```bash
python3 cf_h2o/eval/traffic_signal_resco_official_rl_baseline.py \
  --scenario-set extended \
  --algorithms IDQN,MPLight \
  --episodes 20 \
  --testing 5 \
  --seeds 131 \
  --timeout-sec 3600 \
  --workers 8 \
  --cache-dir cf_h2o/results/resco_official_rl_mid_budget_cache \
  --log-dir cf_h2o/results/resco_official_rl_mid_budget \
  --out cf_h2o/results/traffic_signal_resco_official_rl_mid_budget.json \
  --md-out cf_h2o/results/traffic_signal_resco_official_rl_mid_budget.md
```

Mid-budget result file:

- `cf_h2o/results/traffic_signal_resco_official_rl_mid_budget.md`
- Supported-only table with known upstream MPLight failures skipped:
  `cf_h2o/results/traffic_signal_resco_official_rl_mid_budget_supported.md`

Status:

- `IDQN` completed all eight extended RESCO scenarios.
- `MPLight` completed `grid4x4`, `arterial4x4`, `cologne1`, `cologne3`,
  `ingolstadt1`, and `ingolstadt7`.
- `MPLight` again failed on `cologne8` and `ingolstadt21` with the same upstream
  `pfrl_dqn.py` shared-DQN indexing error.  This confirms the short-budget
  failure is not a route-file issue.
- Resume/cache was exercised in practice: the interrupted run was restarted and
  successful cached rows were reused while missing/failed rows were retried.
- `--skip-runs` was added for known upstream failures.  The supported-only
  mid-budget table skips `MPLight:cologne8` and `MPLight:ingolstadt21`, producing
  14 successful runs and zero adapter-level failures.

Full-budget command:

```bash
python3 cf_h2o/eval/traffic_signal_resco_official_rl_baseline.py \
  --scenario-set extended \
  --algorithms IDQN,MPLight \
  --episodes 100 \
  --testing 5 \
  --seeds 131,337,911 \
  --workers 8 \
  --timeout-sec 21600 \
  --cache-dir cf_h2o/results/resco_official_rl_full_cache \
  --log-dir cf_h2o/results/resco_official_rl_full \
  --skip-runs MPLight:cologne8,MPLight:ingolstadt21 \
  --out cf_h2o/results/traffic_signal_resco_official_rl_full.json \
  --md-out cf_h2o/results/traffic_signal_resco_official_rl_full.md
```

This is computationally much larger than the CFCMT stress-test run because it
trains RL policies online under the official RESCO loop.

Full-budget result files:

- `cf_h2o/results/traffic_signal_resco_official_rl_full.json`
- `cf_h2o/results/traffic_signal_resco_official_rl_full.md`

Full-budget status:

- 42 supported runs completed with zero failures.
- Coverage is `IDQN` on all eight extended RESCO scenarios with three seeds
  each, plus `MPLight` on six supported scenarios with three seeds each.
- `MPLight:cologne8` and `MPLight:ingolstadt21` remain skipped because the
  official upstream implementation fails on those large irregular networks.
- Runtime is substantial: median per-run elapsed time is about 7,989 seconds,
  and the slowest run is about 17,417 seconds.

Full-budget aggregate:

| algorithm | scenario | runs | test_timeLoss | test_duration | test_waitingTime | elapsed_sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| IDQN | arterial4x4 | 3 | 1505.5358 | 501.3054 | 436.9201 | 17179.2053 |
| MPLight | arterial4x4 | 3 | 1056.7967 | 357.4536 | 238.7453 | 13576.0240 |
| IDQN | cologne1 | 3 | 612.7367 | 164.9049 | 138.9499 | 812.4412 |
| MPLight | cologne1 | 3 | 189.3935 | 122.7513 | 68.8432 | 6985.7837 |
| IDQN | cologne3 | 3 | 86.6096 | 92.3130 | 43.2678 | 2351.5003 |
| MPLight | cologne3 | 3 | 1011.2390 | 543.5108 | 515.3289 | 10132.7531 |
| IDQN | cologne8 | 3 | 24.7600 | 89.3794 | 8.3538 | 5946.6425 |
| IDQN | grid4x4 | 3 | 47.7521 | 158.2155 | 25.2053 | 11120.6226 |
| MPLight | grid4x4 | 3 | 44.3930 | 156.1122 | 16.5979 | 10392.9849 |
| IDQN | ingolstadt1 | 3 | 18.0972 | 37.1910 | 5.4896 | 911.3322 |
| MPLight | ingolstadt1 | 3 | 401.8947 | 85.9055 | 63.1920 | 7888.1431 |
| IDQN | ingolstadt21 | 3 | 256.5877 | 353.1114 | 176.4054 | 13141.2565 |
| IDQN | ingolstadt7 | 3 | 51.8692 | 78.8268 | 17.3567 | 5313.6457 |
| MPLight | ingolstadt7 | 3 | 284.3911 | 111.5600 | 57.4631 | 8755.1389 |

## LibSignal Adapter

Source checkout:

- `H2Oplus/downloads/traffic_signal_libsignal/repo`
- Upstream: `https://github.com/DaRL-LibSignal/LibSignal`

Implemented:

- Script: `cf_h2o/eval/traffic_signal_libsignal_baseline.py`
- Test: `cf_h2o/tests/test_traffic_signal_libsignal_baseline.py`
- Parallel execution: `--workers N` launches independent LibSignal subprocesses.
- Resume support: each run writes a per-run summary under `--cache-dir`; by
  default, successful cached runs are reused and failed cached runs are retried.
- Metric parsing: the adapter parses LibSignal's final stdout metric line
  (`travel time`, `reward`, `queue`, `delay`, `throughput`).

Local compatibility patches in the LibSignal checkout:

- `agent/__init__.py`: skips optional CoLight/MADDPG/MAGD imports when
  `torch_scatter`, `torch_geometric`, or `tensorflow` are unavailable.
- `world/__init__.py` and `generator/lane_vehicle.py`: allow SUMO-only runs when
  `cityflow` is unavailable.
- `run.py`: adds `--episodes`, `--steps`, `--test_steps`,
  `--test_when_train`, `--save_model`, `--save_rate`, and `--torch_threads`
  overrides for smoke, pilot, and full-budget runs.
- `trainer/tsc_trainer.py`: respects `save_model=false` and avoids checkpoint
  writes when they are not part of the evaluation protocol.
- `world/world_sumo.py`: avoids the 500-step phase-discovery rollout by reading
  SUMO program logic directly, skips per-step vehicle-trajectory reconstruction
  unless the subscribed metrics need it, and caches static lane speeds/lengths.
- `generator/lane_vehicle.py`: removes repeated small-array `np.append` and
  `np.mean` allocations in the hot observation/reward path.
- `sumo4x4` asset repair: LibSignal's checkout ignores/misses
  `data/raw_data/grid4x4/grid4x4.net.xml` and `grid4x4.rou.xml`; the adapter
  copies them from the RESCO checkout when needed.

Validated smoke command:

```bash
python3 cf_h2o/eval/traffic_signal_libsignal_baseline.py \
  --agents dqn,presslight,frap,mplight \
  --networks sumo1x1 \
  --episodes 1 \
  --steps 60 \
  --test-steps 60 \
  --timeout-sec 180 \
  --workers 4 \
  --out cf_h2o/results/traffic_signal_libsignal_rl_smoke.json \
  --md-out cf_h2o/results/traffic_signal_libsignal_rl_smoke.md
```

Smoke result file:

- `cf_h2o/results/traffic_signal_libsignal_rl_smoke.md`

All four LibSignal RL agents in that smoke (`DQN`, `PressLight`, `FRAP`,
`MPLight`) completed on `sumo1x1`.  These are startup/integration checks only;
they are not trained baselines.

Short-budget LibSignal pilot:

```bash
python3 cf_h2o/eval/traffic_signal_libsignal_baseline.py \
  --agents dqn,presslight,frap,mplight \
  --networks sumo1x1 \
  --episodes 5 \
  --steps 600 \
  --test-steps 600 \
  --timeout-sec 600 \
  --workers 4 \
  --out cf_h2o/results/traffic_signal_libsignal_short_budget.json \
  --md-out cf_h2o/results/traffic_signal_libsignal_short_budget.md
```

Short-budget result file:

- `cf_h2o/results/traffic_signal_libsignal_short_budget.md`

This pilot also completes without failures.  The metrics are still LibSignal
protocol metrics and are not directly comparable to the CFCMT phase-MPC table.
They should be treated as a runner validation until full-budget, multi-seed
LibSignal experiments are executed.

Multi-network LibSignal diagnostic:

```bash
python3 cf_h2o/eval/traffic_signal_libsignal_baseline.py \
  --agents dqn,presslight,frap,mplight \
  --networks sumo1x1,sumo1x3,sumo4x4 \
  --episodes 5 \
  --steps 600 \
  --test-steps 600 \
  --timeout-sec 900 \
  --workers 8 \
  --cache-dir cf_h2o/results/libsignal_multinet_mid_budget_cache \
  --out cf_h2o/results/traffic_signal_libsignal_multinet_mid_budget.json \
  --md-out cf_h2o/results/traffic_signal_libsignal_multinet_mid_budget.md
```

Multi-network result file:

- `cf_h2o/results/traffic_signal_libsignal_multinet_mid_budget.md`

Status:

- `DQN`, `PressLight`, `FRAP`, and `MPLight` completed on `sumo1x1`,
  `sumo1x3`, and `sumo4x4`.
- The initial `sumo4x4` failures were SUMO startup failures caused by missing
  XML assets, not model failures.  After asset repair, the 12-run table completes
  with zero failures.

Profiling and optimized full-budget protocol:

- Hardware check: the machine has 32 logical CPUs and an RTX 4060 Laptop GPU.
- cProfile showed the dominant cost is CPU-side SUMO/libsumo rollout,
  observation generation, and repeated train-time evaluation, not GPU tensor
  throughput.
- The adapter therefore uses CPU multi-process parallelism: `--workers 8`,
  `--native-threads 1`, and `--torch-threads 1`.
- Per-episode train-time tests are disabled with `--disable-train-test`; the
  final LibSignal test run is still executed and parsed.
- Checkpoint writes are disabled with `--disable-save-model`.

Validated optimized short profile:

- `mplight/sumo4x4`, one 1200-step training episode plus 1200-step final test,
  fell from about 16.0 seconds before optimization to about 3.3 seconds under
  the optimized protocol.

Full-budget LibSignal command:

```bash
python3 cf_h2o/eval/traffic_signal_libsignal_baseline.py \
  --agents dqn,presslight,frap,mplight \
  --networks sumo1x1,sumo1x3,sumo4x4 \
  --episodes 100 \
  --steps 3600 \
  --test-steps 3600 \
  --seeds 131,337,911 \
  --timeout-sec 10800 \
  --workers 8 \
  --cache-dir cf_h2o/results/libsignal_full_optimized_cache \
  --work-root cf_h2o/results/libsignal_full_optimized_workdirs \
  --disable-train-test \
  --disable-save-model \
  --torch-threads 1 \
  --native-threads 1 \
  --out cf_h2o/results/traffic_signal_libsignal_full_optimized.json \
  --md-out cf_h2o/results/traffic_signal_libsignal_full_optimized.md
```

Full-budget result files:

- `cf_h2o/results/traffic_signal_libsignal_full_optimized.json`
- `cf_h2o/results/traffic_signal_libsignal_full_optimized.md`

Full-budget status:

- 36 runs completed with zero failures.
- Coverage is `DQN`, `PressLight`, `FRAP`, and `MPLight` on `sumo1x1`,
  `sumo1x3`, and `sumo4x4`, with seeds `131,337,911`.
- Runtime remains large for FRAP on `sumo4x4`: the three seeds took about
  7,145-7,250 seconds each.  This is a model/environment cost, not a thread
  oversubscription bug.

Full-budget aggregate:

| agent | network | runs | travel_time | reward | queue | delay | throughput | elapsed_sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dqn | sumo1x1 | 3 | 38.9448 | -3.9320 | 3.2398 | 0.2831 | 1998.6667 | 175.9667 |
| dqn | sumo1x3 | 3 | 97.0904 | -28.8363 | 6.5312 | 0.2443 | 2487.3333 | 425.3838 |
| dqn | sumo4x4 | 3 | 140.6097 | -8.2498 | 0.6027 | 0.0409 | 1465.0000 | 2140.7727 |
| frap | sumo1x1 | 3 | 41.9234 | -2.9908 | 24.4611 | 0.4596 | 1664.6667 | 439.6200 |
| frap | sumo1x3 | 3 | 118.7116 | -5.5377 | 11.6722 | 0.3977 | 2505.3333 | 1583.0533 |
| frap | sumo4x4 | 3 | 140.4773 | -0.6911 | 0.6123 | 0.0422 | 1464.0000 | 7188.1846 |
| mplight | sumo1x1 | 3 | 40.5462 | -0.4764 | 4.3731 | 0.3190 | 1995.0000 | 371.5690 |
| mplight | sumo1x3 | 3 | 66.5017 | -0.7088 | 1.8759 | 0.2431 | 2814.3333 | 647.6640 |
| mplight | sumo4x4 | 3 | 140.4690 | -0.6799 | 0.6074 | 0.0423 | 1464.3333 | 1017.2967 |
| presslight | sumo1x1 | 3 | 39.7481 | -9.5055 | 3.4417 | 0.2850 | 1998.6667 | 175.8352 |
| presslight | sumo1x3 | 3 | 63.1881 | -10.1500 | 1.1636 | 0.1882 | 2816.0000 | 393.4139 |
| presslight | sumo4x4 | 3 | 144.2793 | -8.1274 | 0.7107 | 0.0481 | 1463.3333 | 2080.0427 |

Remaining LibSignal caveat:

- Upstream LibSignal pins an older environment (`python 3.9`, `gym==0.21`,
  `numpy==1.21.5`).  The current environment runs full-budget SUMO baselines
  after compatibility and performance patches.  The paper should explicitly
  cite the patched checkout and keep these rows separate from same-protocol
  CFCMT phase-MPC rows.

## Paper Use

Do not cite the pilot numbers as final trained baselines.  The optimized
full-budget LibSignal table can be cited as a patched-checkout, LibSignal-
protocol leaderboard cross-check.

Acceptable current statement:

> We integrated the official RESCO and LibSignal runners and executed
> full-budget leaderboard-style RL baselines under each runner's native
> state/reward/action definitions.  These baselines remain separate from the
> same-protocol CFCMT phase-MPC stress test.
