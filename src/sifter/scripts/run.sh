#!/bin/bash
#SBATCH --job-name=sifter-run
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=workq
#
# Note: sifter passes partition/cpu/memory/GPU requests as sbatch
# command-line flags (see SIFTER_SLURM_* in the README), which override
# the directives above. They remain here as documented fallbacks for
# anyone submitting these scripts by hand.
#
# Sifter run script.
#
# Required environment variables:
#   RUN_COMMAND     - Full singularity/apptainer command to execute
# Optional:
#   CONTAINER_PATH  - Path to the container being run

set -euo pipefail

RUN_COMMAND="${RUN_COMMAND:?RUN_COMMAND is required}"
CONTAINER_PATH="${CONTAINER_PATH:-unknown}"

echo "=== Sifter Run Job ==="
echo "Container: ${CONTAINER_PATH}"
echo "Start: $(date)"
echo ""
echo "Command: ${RUN_COMMAND}"
echo ""

bash -lc "${RUN_COMMAND}"

echo ""
echo "=== Run Complete ==="
echo "End: $(date)"
