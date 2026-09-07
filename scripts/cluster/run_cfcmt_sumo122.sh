#!/usr/bin/env bash
set -euo pipefail

: "${CFCMT_SOURCE_ROOT:?CFCMT_SOURCE_ROOT must point to an immutable source snapshot}"

CFCMT_RUNTIME_ROOT="${CFCMT_RUNTIME_ROOT:-/home/zhengliang01/scheduleurm_work/runtimes/cfcmt-sumo122-py310-v1}"
CFCMT_PYTHON="${CFCMT_PYTHON:-/home/zhengliang01/scheduleurm_work/conda_envs/freqduet-cpu-py310/bin/python3.10}"
CFCMT_SUMO_HOME="${CFCMT_SUMO_HOME:-${CFCMT_RUNTIME_ROOT}/sumo}"

test -x "${CFCMT_PYTHON}"
test -d "${CFCMT_RUNTIME_ROOT}"
test -d "${CFCMT_SUMO_HOME}/tools"
test -d "${CFCMT_SUMO_HOME}/data/typemap"
test -d "${CFCMT_SOURCE_ROOT}/cf_h2o"

# ``multiprocessing`` spawn re-enters the parent's current directory.  Cluster
# commands can inherit /root even though the worker runs as an unprivileged user.
cd "${CFCMT_SOURCE_ROOT}"

export PYTHONPATH="${CFCMT_RUNTIME_ROOT}:${CFCMT_SOURCE_ROOT}"
export PATH="${CFCMT_RUNTIME_ROOT}/bin:/usr/bin:/bin"
export SUMO_HOME="${CFCMT_SUMO_HOME}"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED=0
export LC_ALL=C
export LANG=C
export CFCMT_EXPECTED_SUMO_VERSION="${CFCMT_EXPECTED_SUMO_VERSION:-1.22.0}"
export CFCMT_SUMO_SCRATCH_DIR="${CFCMT_SUMO_SCRATCH_DIR:-/dev/shm}"
export TMPDIR="${CFCMT_TMPDIR:-/dev/shm}"
export TMP="${CFCMT_TMPDIR:-/dev/shm}"
export TEMP="${CFCMT_TMPDIR:-/dev/shm}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec "${CFCMT_PYTHON}" "$@"
