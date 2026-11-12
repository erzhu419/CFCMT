set -o pipefail
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -m experiments.tsc_packet_conditional_particles_20261110.verify --output experiments/tsc_packet_conditional_particles_20261110/output --workers 4 2>&1 | tee experiments/tsc_packet_conditional_particles_20261110/verification.log
