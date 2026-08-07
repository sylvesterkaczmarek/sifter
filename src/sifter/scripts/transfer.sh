#!/bin/bash
#SBATCH --job-name=sifter-transfer
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=workq
#
# Note: sifter passes partition/cpu/memory/GPU requests as sbatch
# command-line flags (see SIFTER_SLURM_* in the README), which override
# the directives above. They remain here as documented fallbacks for
# anyone submitting these scripts by hand.
#
# Sifter transfer script: runs a backend-supplied push/pull command.
#
# Required environment variables:
#   TRANSFER_COMMAND   - Full command to execute (the backend builds it)
# Optional:
#   TRANSFER_FILENAME  - Filename being transferred
#   TRANSFER_DIRECTION - "push" or "pull"

set -euo pipefail

TRANSFER_COMMAND="${TRANSFER_COMMAND:?TRANSFER_COMMAND is required}"
TRANSFER_FILENAME="${TRANSFER_FILENAME:-unknown}"
TRANSFER_DIRECTION="${TRANSFER_DIRECTION:-transfer}"

echo "=== Transfer Job ==="
echo "Direction: ${TRANSFER_DIRECTION}"
echo "File: ${TRANSFER_FILENAME}"
echo "Start: $(date)"
echo "Command: ${TRANSFER_COMMAND}"
echo ""

bash -lc "${TRANSFER_COMMAND}"

echo ""
echo "=== Transfer Complete ==="
echo "End: $(date)"
