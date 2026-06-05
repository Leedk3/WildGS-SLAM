#!/bin/bash
# WildGS-SLAM conda 환경 설정 (RTX 5080 / Blackwell sm_120)
# 변경 사항:
#   - torch_scatter → PyTorch native (ba.py, droid_net.py 패치됨)
#   - mmcv 제거 (직접 import 없음)
#   - nvcc: apt cuda-toolkit-12-8 사용 (conda보다 안정적)
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
ENV_NAME="wildgs-slam"

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
    echo "Install them in the x86 Docker container with:" >&2
    echo "  docker exec -u root aac-2026-container-all apt-get update" >&2
    echo "  docker exec -u root aac-2026-container-all apt-get install -y --no-install-recommends cuda-libraries-dev-12-8" >&2
    exit 1
  fi
}

echo "========================================"
echo " WildGS-SLAM 환경 설정 (sm_120 지원)"
echo "========================================"

# ── 1. nvcc 설치 확인 ────────────────────────────────────────────────
if ! command -v nvcc &>/dev/null; then
  echo "[1/7] Installing CUDA 12.8 nvcc via apt..."
  sudo apt-get update -qq
  sudo apt-get install -y --no-install-recommends \
    cuda-nvcc-12-8 cuda-cudart-dev-12-8 libcuda-12-8-dev 2>/dev/null || \
  sudo apt-get install -y --no-install-recommends \
    nvidia-cuda-toolkit 2>/dev/null || \
    { echo "ERROR: nvcc 설치 실패. 수동으로 CUDA toolkit을 설치하세요."; exit 1; }
  # PATH 추가
  export PATH=/usr/local/cuda-12.8/bin:/usr/local/cuda/bin:${PATH}
else
  echo "[1/7] nvcc already available: $(nvcc --version | grep release)"
fi

# CUDA_HOME 설정
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
echo "  CUDA_HOME=${CUDA_HOME}"
check_cuda_dev_headers

# ── 2. conda 환경 ────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda가 없습니다. Miniconda/Anaconda를 설치한 뒤 다시 실행하세요." >&2
  exit 1
fi

CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME} "; then
  echo "[2/7] Conda env '${ENV_NAME}' already exists — reusing"
else
  echo "[2/7] Creating conda env: ${ENV_NAME} (python=3.10)"
  conda create -y --name "${ENV_NAME}" python=3.10
fi
conda activate "${ENV_NAME}"

# ── 3. 핵심 패키지 ──────────────────────────────────────────────────
echo "[3/7] Installing numpy, setuptools (pinned)"
pip install -q numpy==1.26.3 setuptools==78.1.1

# ── 4. PyTorch + cu128 (sm_120 Blackwell 지원) ─────────────────────
echo "[4/7] Installing PyTorch 2.11.0+cu128"
pip install -q \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -q xformers==0.0.35

# GPU 확인
python3 -c "
import torch
import xformers
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
print(f'xFormers {xformers.__version__}')
if torch.cuda.is_available():
    cap = torch.cuda.get_device_capability()
    print(f'GPU: {torch.cuda.get_device_name(0)} (sm_{cap[0]}{cap[1]})')
"

# ── 5. CUDA extension 빌드 ──────────────────────────────────────────
# sm_120 포함 (RTX 5080 Blackwell)
export TORCH_CUDA_ARCH_LIST="12.0"
echo "[5/7] Building CUDA extensions (arch: ${TORCH_CUDA_ARCH_LIST})"
pip install --no-build-isolation -e thirdparty/lietorch/
pip install --no-build-isolation -e thirdparty/diff-gaussian-rasterization-w-pose/
pip install --no-build-isolation -e thirdparty/simple-knn/

# ── 6. requirements (torch_scatter / mmcv 제외) ─────────────────────
echo "[6/7] Installing requirements (torch_scatter / mmcv excluded)"
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "${CONSTRAINTS_FILE}"' EXIT
printf 'numpy==1.26.3\n' > "${CONSTRAINTS_FILE}"

grep -vE "^(torch[-_]scatter|torch_scatter|mmcv|opencv-python)" requirements.txt \
  | pip install -c "${CONSTRAINTS_FILE}" -r /dev/stdin
pip install numpy==1.26.3 opencv-python==4.8.1.78  # 별도 설치
pip install --no-build-isolation mmcv==1.7.2

# WildGS-SLAM 자체
pip install --no-build-isolation -e .

# ── 7. 검증 ─────────────────────────────────────────────────────────
echo "[7/7] Verification"
python3 -c "
import torch, xformers, lietorch, simple_knn, diff_gaussian_rasterization
print('=== 설치 확인 ===')
print(f'torch:    {torch.__version__}')
print(f'xformers: {xformers.__version__}')
print(f'CUDA:     {torch.cuda.is_available()}')
print('lietorch: OK')
print('simple_knn: OK')
print('diff_gaussian_rasterization: OK')

# torch_scatter 패치 확인
import sys; sys.path.insert(0, 'src')
from geom.ba import scatter_sum
from modules.droid_net.droid_net import scatter_mean
x = torch.randn(3,4).cuda()
idx = torch.tensor([0,1,0]).cuda()
scatter_sum(x, idx, dim=0, dim_size=2)
scatter_mean(x, idx, dim=0, dim_size=2)
print('scatter ops (native): OK')
"

echo ""
echo "========================================"
echo " 완료! 다음 단계:"
echo ""
echo " 1. pretrained 모델 다운로드:"
echo "    conda activate ${ENV_NAME}"
echo "    pip install gdown"
echo "    gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O pretrained/droid.pth"
echo ""
echo " 2. 데모 데이터:"
echo "    bash scripts_downloading/download_demo_data.sh"
echo ""
echo " 3. 실행:"
echo "    python run.py configs/Dynamic/Wild_SLAM_Mocap/crowd_demo.yaml"
echo "========================================"
