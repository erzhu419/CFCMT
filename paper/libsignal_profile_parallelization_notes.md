# LibSignal Profiling and Parallelization Notes

## Diagnosis

The slow LibSignal full-budget run was not primarily a GPU bottleneck.

Hardware:

- CPU: AMD Ryzen 9 7945HX, 16 physical cores / 32 logical threads.
- GPU: NVIDIA RTX 4060 Laptop GPU, CUDA available.

cProfile and process monitoring showed three main costs:

1. SUMO/libsumo rollout and Python-side observation generation.
2. LibSignal's default `test_when_train=True`, which ran a full test rollout
   after every training episode.
3. Thread oversubscription: multiple subprocesses each allowed PyTorch/BLAS to
   use many CPU threads.

The GPU was not selected for the baseline run because the workload is dominated
by simulator stepping, per-lane vehicle queries, and small per-intersection
networks.  Sending many small models to one laptop GPU would not address the
main bottleneck and would introduce contention across subprocesses.

## Optimizations Applied

- Added runtime overrides in LibSignal `run.py`:
  `--test_when_train`, `--save_model`, `--save_rate`, and `--torch_threads`.
- Disabled per-episode train-time evaluation for full-budget baselines while
  preserving the final test rollout.
- Disabled checkpoint writes for evaluation-only full-budget runs.
- Forced one native/PyTorch thread per subprocess:
  `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`,
  `NUMEXPR_NUM_THREADS=1`, and `--torch_threads 1`.
- Avoided unnecessary per-step vehicle-trajectory reconstruction in
  `world_sumo.py`.
- Read SUMO traffic-light program phases directly instead of stepping 500
  seconds during phase discovery.
- Cached static lane speeds/lengths.
- Replaced repeated small `np.append`/`np.mean` generator allocations with list
  accumulation and scalar averages.

## Validation

Representative short profile:

- `mplight/sumo4x4`, 1 training episode, 1200 training steps, 1200 final test
  steps.
- Original comparable run: about 16.0 seconds wall time.
- Optimized code with same train-time test protocol: about 7.2 seconds.
- Optimized full-budget protocol (`test_when_train=false`, `save_model=false`):
  about 3.3 seconds.

Parallel probe:

- 9 runs: 3 agents x 3 networks x 1 seed, 1 episode, 1200 steps.
- `workers=8`, one thread per subprocess.
- 9/9 completed, zero failures.
- Each subprocess stayed near one CPU core, avoiding oversubscription.

Full-budget result:

- Command output:
  `cf_h2o/results/traffic_signal_libsignal_full_optimized.json`
- Markdown table:
  `cf_h2o/results/traffic_signal_libsignal_full_optimized.md`
- Coverage: `DQN`, `PressLight`, `FRAP`, `MPLight` x `sumo1x1`, `sumo1x3`,
  `sumo4x4` x seeds `131,337,911`.
- Runs: 36/36 completed, zero failures.
- Slowest cases: `sumo4x4/frap`, about 7,145-7,250 seconds per seed.

## Parallelization Policy

Use CPU process parallelism first:

```bash
--workers 8 --native-threads 1 --torch-threads 1
```

This is the validated stable setting.  The machine has enough cores to test
`workers=12` or `workers=14` in a future speed run, but that should be a
separate sensitivity check rather than mixed with the completed full-budget
table.

Do not use GPU for the current LibSignal full-budget baselines unless a new
profile shows model forward/backward dominates simulator rollout.  Current
profiles do not support that.
