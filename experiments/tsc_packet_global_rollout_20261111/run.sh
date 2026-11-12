#!/usr/bin/env bash
set -euo pipefail
base=/home/erzhu419/scheduleurm_work/CFCMT_corridor_residual_20261109
root="$base/target_packet_global_rollout"
program="$root/experiments/tsc_packet_global_rollout_20261111"
trap 'printf "%s\n" "$?" > "$root/exit_code"' EXIT
runner=(bash "$program/python_on_server.sh")
if [[ ${1:-} == resume ]]; then
    calibration_args=(--resume --workers 12)
else
    "${runner[@]}" -m experiments.tsc_packet_global_rollout_20261111.calibration --base "$base" --output "$root/run" collect
    calibration_args=(--workers 4)
fi
"${runner[@]}" -m experiments.tsc_packet_global_rollout_20261111.calibration --base "$base" --output "$root/run" "${calibration_args[@]}" calibrate
"${runner[@]}" -m experiments.tsc_packet_global_rollout_20261111.action --base "$base" --output "$root/run/action" --calibration-result "$root/run/calibration/result.json"
