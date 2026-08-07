#!/bin/bash
#SBATCH --job-name=sifter-build
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=144
#SBATCH --mem=0
#SBATCH --partition=workq
#
# Note: sifter passes partition/cpu/memory/GPU requests as sbatch
# command-line flags (see SIFTER_SLURM_* in the README), which override
# the directives above. They remain here as documented fallbacks for
# anyone submitting these scripts by hand.
#
# Generic Sifter Build Script
#
# This script is submitted by sifter to build containers via SLURM.
# It handles:
# - Staging definition files to tmpfs to avoid Lustre fakeroot issues
# - Substituting {{ BASE_IMAGE }} placeholder with cache path
# - Reading build arguments from env file
# - Caching output by content hash
# - Pushing to remote cache (non-blocking)
# - Copying to registry for final tagged output
#
# Arguments (via environment variables):
#   STAGING_DIR              - Directory with staged definition files
#   DEFINITION_PATH          - Relative path to .def file within staging
#   OUTPUT_HASH              - Content hash for output filename
#   CACHE_DIR                - Local cache directory
#   BASE_IMAGE_HASH          - Hash of base image (for chained builds)
#   TAG_NAME                 - Final tag name (if this is the final step)
#   REGISTRY_DIR             - Local registry directory
#   REMOTE_PUSH_CACHE_CMD    - Shell command to push to remote cache (optional)

set -euo pipefail

# --- Configuration ---
STAGING_DIR="${STAGING_DIR:?STAGING_DIR is required}"
DEFINITION_PATH="${DEFINITION_PATH:?DEFINITION_PATH is required}"
OUTPUT_HASH="${OUTPUT_HASH:?OUTPUT_HASH is required}"
CACHE_DIR="${CACHE_DIR:?CACHE_DIR is required}"

# Optional
BASE_IMAGE_HASH="${BASE_IMAGE_HASH:-}"
TAG_NAME="${TAG_NAME:-}"
REGISTRY_DIR="${REGISTRY_DIR:-}"
REMOTE_PUSH_CACHE_CMD="${REMOTE_PUSH_CACHE_CMD:-}"

# --- Setup ---

echo "=== Sifter Build Script ==="
echo "Definition: ${DEFINITION_PATH}"
echo "Output Hash: ${OUTPUT_HASH}"
echo "Cache Dir: ${CACHE_DIR}"
echo "Base Image Hash: ${BASE_IMAGE_HASH:-none}"
echo "Tag Name: ${TAG_NAME:-none}"
echo ""

# Create cache directory if it doesn't exist
mkdir -p "${CACHE_DIR}"

# Build in node-local scratch to avoid --fakeroot issues on networked
# filesystems (e.g. Lustre). Precedence: SIFTER_BUILD_TMP (explicit override) >
# $LOCALDIR (set by SLURM on some clusters) > /tmp.
BUILD_TMP="${SIFTER_BUILD_TMP:-${LOCALDIR:-/tmp}}/sifter-build-$$"
mkdir -p "${BUILD_TMP}"
trap 'rm -rf "${BUILD_TMP}"' EXIT

echo "Build tmpdir: ${BUILD_TMP}"

# Copy staging directory to tmpfs
cp -r "${STAGING_DIR}"/* "${BUILD_TMP}/"

# Full path to definition file
DEF_FILE="${BUILD_TMP}/${DEFINITION_PATH}"
if [[ ! -f "${DEF_FILE}" ]]; then
    echo "ERROR: Definition file not found: ${DEF_FILE}"
    exit 1
fi

# --- Base Image Substitution ---

if [[ -n "${BASE_IMAGE_HASH}" ]]; then
    # Determine base image path
    # If BASE_IMAGE_HASH is already an absolute path, use it directly
    # Otherwise, look for it in the cache directory
    if [[ "${BASE_IMAGE_HASH}" = /* ]]; then
        # Absolute path - use directly (may be from registry or cache)
        if [[ "${BASE_IMAGE_HASH}" == *.sif ]]; then
            BASE_IMAGE_PATH="${BASE_IMAGE_HASH}"
        else
            BASE_IMAGE_PATH="${BASE_IMAGE_HASH}.sif"
        fi
        BASE_IMAGE_FILENAME=$(basename "${BASE_IMAGE_PATH}")
    else
        # Relative path/hash - look in cache
        BASE_IMAGE_PATH="${CACHE_DIR}/${BASE_IMAGE_HASH}.sif"
        BASE_IMAGE_FILENAME="${BASE_IMAGE_HASH}.sif"
    fi

    if [[ ! -f "${BASE_IMAGE_PATH}" ]]; then
        echo "ERROR: Base image not found: ${BASE_IMAGE_PATH}"
        exit 1
    fi

    echo "Substituting {{ BASE_IMAGE }} with ${BASE_IMAGE_PATH}"

    # Copy base image to build tmpdir for faster access
    cp "${BASE_IMAGE_PATH}" "${BUILD_TMP}/${BASE_IMAGE_FILENAME}"

    # Replace placeholder in definition file
    sed -i "s|{{ BASE_IMAGE }}|${BUILD_TMP}/${BASE_IMAGE_FILENAME}|g" "${DEF_FILE}"
fi

# --- Build Arguments ---

BUILD_ARGS=()
BUILD_ARGS_FILE="${BUILD_TMP}/build_args.env"

if [[ -f "${BUILD_ARGS_FILE}" ]]; then
    echo "Reading build arguments from ${BUILD_ARGS_FILE}"
    while IFS='=' read -r key value; do
        # Skip empty lines and comments
        [[ -z "${key}" || "${key}" =~ ^# ]] && continue
        # Build --build-arg flags (use array to preserve values with spaces)
        BUILD_ARGS+=(--build-arg "${key}=${value}")
        echo "  ${key}=${value}"
    done < "${BUILD_ARGS_FILE}"
fi

# --- Build Container ---

OUTPUT_FILE="${BUILD_TMP}/output.sif"

echo ""
echo "=== Building Container ==="
echo "Definition: ${DEF_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo ""

# Set tmpdir for Singularity/Apptainer and general tools (config.guess, etc.)
# This ensures all temp files go to our build tmpdir, avoiding permission issues
# with shared directories like /scratch/$USER/tmp
export SINGULARITY_TMPDIR="${BUILD_TMP}"
export APPTAINER_TMPDIR="${BUILD_TMP}"
export TMPDIR="${BUILD_TMP}"
export TMP="${BUILD_TMP}"
export TEMP="${BUILD_TMP}"

# Build the container
# cd to definition directory so %files relative paths resolve correctly
cd "$(dirname "${DEF_FILE}")"
singularity build --fakeroot "${BUILD_ARGS[@]}" "${OUTPUT_FILE}" "${DEF_FILE}"

if [[ ! -f "${OUTPUT_FILE}" ]]; then
    echo "ERROR: Build produced no output"
    exit 1
fi

echo ""
echo "=== Build Complete ==="
echo ""

# --- Cache Output ---

CACHE_FILE="${CACHE_DIR}/${OUTPUT_HASH}.sif"
echo "Caching to: ${CACHE_FILE}"
cp "${OUTPUT_FILE}" "${CACHE_FILE}"

# --- Push to Remote Cache ---

if [[ -n "${REMOTE_PUSH_CACHE_CMD}" ]]; then
    echo "Pushing to remote cache..."
    if ! bash -c "${REMOTE_PUSH_CACHE_CMD}"; then
        echo "WARNING: Remote cache push failed (continuing)"
    fi
fi

# --- Copy to Registry ---

if [[ -n "${TAG_NAME}" && -n "${REGISTRY_DIR}" ]]; then
    mkdir -p "${REGISTRY_DIR}"
    REGISTRY_FILE="${REGISTRY_DIR}/${TAG_NAME}.sif"
    echo "Copying to local registry: ${REGISTRY_FILE}"
    cp "${CACHE_FILE}" "${REGISTRY_FILE}"
    # No remote publish here — `sifter push` publishes to the registry explicitly.
fi

echo ""
echo "=== Done ==="
echo "Cached: ${CACHE_FILE}"
if [[ -n "${TAG_NAME}" ]]; then
    echo "Registry: ${REGISTRY_FILE}"
fi
