#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-daemon_wait}
WORK=${2:-/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/phase1_v2}
STAGE=${3:-both}
WORKERS=${4:-12}

mkdir -p "$WORK"

if [[ "$MODE" == "daemon_wait" ]]; then
    LOG="$WORK/run_${STAGE}.log"
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
STAGED=/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/phase0/staged_scenarios
SOURCES=/home/erzhu419/scheduleurm_work/CFCMT_partial_cv_pilot_20261118/source_scenarios

if [[ "$STAGE" == "both" || "$STAGE" == "collect_and_fit" || "$STAGE" == "fit_and_eval_fresh" ]]; then
    "$PY" -m experiments.tsc_partial_cv_pilot_20261118.phase1_transfer_pilot \
        --stage collect_and_fit \
        --work-dir "$WORK" \
        --staged-scenarios-dir "$STAGED" \
        --source-scenarios-dir "$SOURCES" \
        --workers "$WORKERS"
fi

if [[ "$STAGE" == "both" || "$STAGE" == "dev_check" ]]; then
    "$PY" -m experiments.tsc_partial_cv_pilot_20261118.phase1_transfer_pilot \
        --stage dev_check \
        --work-dir "$WORK" \
        --staged-scenarios-dir "$STAGED" \
        --source-scenarios-dir "$SOURCES" \
        --workers "$WORKERS"
fi

if [[ "$STAGE" == "eval_fresh" || "$STAGE" == "fit_and_eval_fresh" ]]; then
    "$PY" -m experiments.tsc_partial_cv_pilot_20261118.phase1_transfer_pilot \
        --stage eval_fresh \
        --work-dir "$WORK" \
        --staged-scenarios-dir "$STAGED" \
        --source-scenarios-dir "$SOURCES" \
        --workers "$WORKERS"
fi
