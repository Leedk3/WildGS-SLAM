#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

VENV_DIR="${VENV_DIR:-.venv-wildgs}"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

check_cuda_dev_headers() {
  local missing=()
  local header
  for header in cuda_runtime_api.h cublas_v2.h cusolverDn.h cusparse.h; do
    if [[ ! -f "${CUDA_HOME}/include/${header}" ]]; then
      missing+=("${header}")
    fi
  done

  if (( ${#missing[@]} > 0 )); then
    echo "ERROR: CUDA development headers are missing from ${CUDA_HOME}/include:" >&2
    printf '  - %s\n' "${missing[@]}" >&2
    echo "" >&2
    echo "In the AAC x86 Docker container, install them from the host with:" >&2
    echo "  docker exec -u root aac-2026-container-all apt-get update" >&2
    echo "  docker exec -u root aac-2026-container-all apt-get install -y --no-install-recommends cuda-libraries-dev-12-8" >&2
    exit 1
  fi
}

echo "========================================"
echo " WildGS-SLAM x86 Docker venv setup"
echo "========================================"
echo "  venv:                 ${VENV_DIR}"
echo "  python:               $(${PYTHON_BIN} --version)"
echo "  CUDA_HOME:            ${CUDA_HOME}"
echo "  TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "  MAX_JOBS:             ${MAX_JOBS}"
echo "========================================"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. The x86 container must expose CUDA toolkit / nvcc." >&2
  exit 1
fi
check_cuda_dev_headers

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel
python -m pip install numpy==1.26.3 setuptools==78.1.1 ninja packaging

python -m pip install \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install xformers==0.0.35

python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"capability: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
PY

python -m pip install --no-build-isolation -e thirdparty/lietorch/
python -m pip install --no-build-isolation -e thirdparty/diff-gaussian-rasterization-w-pose/
python -m pip install --no-build-isolation -e thirdparty/simple-knn/

CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
printf 'numpy==1.26.3\n' > "${CONSTRAINTS_FILE}"

grep -vE "^(torch[-_]scatter|torch_scatter|mmcv|opencv-python)" requirements.txt \
  | python -m pip install -c "${CONSTRAINTS_FILE}" -r /dev/stdin
python -m pip install numpy==1.26.3 opencv-python==4.8.1.78
python -m pip install --no-build-isolation mmcv==1.7.2

python -m pip install --no-build-isolation -e .

harnesses/check_sm120_patch.py

python - <<'PY'
import torch
import xformers
import lietorch
import simple_knn
import diff_gaussian_rasterization
import droid_backends

print("extension imports OK")
print(f"xformers: {xformers.__version__}")
print(f"torch cuda available: {torch.cuda.is_available()}")
PY

echo ""
echo "Done. Activate with:"
echo "  source ${SCRIPT_DIR}/${VENV_DIR}/bin/activate"
