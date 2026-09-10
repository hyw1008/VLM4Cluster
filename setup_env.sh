#!/usr/bin/env bash
# Create or update the vlm4cluster conda environment, then install CUDA 12
# FAISS from PyPI without dependency resolution so numpy stays pinned below 2.
#
# Usage:
#   bash setup_env.sh            # create a new environment
#   bash setup_env.sh --update   # update an existing environment

set -euo pipefail

ENV_NAME="${VLM4CLUSTER_ENV_NAME:-vlm4cluster}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/environment.yaml"
FAISS_PACKAGE="${VLM4CLUSTER_FAISS_PACKAGE:-faiss-gpu-cu12}"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[setup] Missing environment file: ${ENV_FILE}" >&2
    exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
    echo "[setup] conda was not found on PATH. Load conda/miniconda first." >&2
    exit 1
fi

case "${1:-}" in
    "")
        echo "[setup] Creating conda environment '${ENV_NAME}' ..."
        conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
        ;;
    "--update")
        echo "[setup] Updating conda environment '${ENV_NAME}' ..."
        conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
        ;;
    *)
        echo "Usage: bash setup_env.sh [--update]" >&2
        exit 2
        ;;
esac

ENV_PREFIX="$(conda run -n "${ENV_NAME}" python -c "import sys; print(sys.prefix)")"
echo "[setup] Environment prefix: ${ENV_PREFIX}"

if [[ "${FAISS_PACKAGE}" == "none" ]]; then
    echo "[setup] Skipping FAISS wheel install because VLM4CLUSTER_FAISS_PACKAGE=none."
else
    echo "[setup] Installing ${FAISS_PACKAGE} with --no-deps ..."
    conda run -n "${ENV_NAME}" python -m pip install "${FAISS_PACKAGE}" --no-deps
fi

echo ""
echo "Done. Activate the environment with:"
echo "    conda activate ${ENV_NAME}"
