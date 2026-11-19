#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-daemon_wait}
WORK=${2:-/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/phase3_tits}
STAGE=${3:-all}
WORKERS=${4:-12}

mkdir -p "$WORK"

if [[ "$MODE" == "daemon_wait" ]]; then
    LOG="$WORK/run_p3_${STAGE}.log"
    rm -f "$LOG"
    nohup bash "$0" inner "$WORK" "$STAGE" "$WORKERS" > "$LOG" 2>&1 < /dev/null &
    PID=$!
    echo "STARTED_PID=$PID LOG=$LOG"
    sleep 1
    tail --pid="$PID" -n +1 -f "$LOG"
    exit 0
fi

export PYTHONPATH=/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/code:/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/runtime
export LD_LIBRARY_PATH=/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/python/lib
export SUMO_HOME=/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/runtime/sumo
export PATH=/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/runtime/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CFCMT_SUMO_SCRATCH_DIR=/dev/shm
export TMPDIR=/dev/shm

PY=/home/erzhu419/scheduleurm_work/CFCMT_V93_exact_reference_20261031/python/bin/python3.10
P2=/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/phase2_frozen_prereg
STAGED=/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/phase0/staged_scenarios
SOURCES=/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/source_scenarios
EXT=/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/ext_targets

"$PY" -m experiments.tsc_partial_cv_pilot_20261118.phase3_tits_suite \
    --stage "$STAGE" \
    --work-dir "$WORK" \
    --phase2-dir "$P2" \
    --staged-scenarios-dir "$STAGED" \
    --source-scenarios-dir "$SOURCES" \
    --ext-targets-dir "$EXT" \
    --workers "$WORKERS"
