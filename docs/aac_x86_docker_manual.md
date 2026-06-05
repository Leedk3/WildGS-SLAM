# AAC x86 Docker WildGS-SLAM Manual

이 문서는 AAC 2026 x86 Docker 컨테이너에서 WildGS-SLAM을 먼저 검증하고, 성공한 절차만 Docker 이미지에 반영하기 위한 로컬 매뉴얼이다.

현재 검증 기준은 RTX 50-series (`sm_120`), CUDA 12.8 `nvcc`, PyTorch `2.11.0+cu128`, Python 3.10 venv이다. x86 Dockerfile의 base image도 CUDA 12.8 devel 계열로 맞춘다. 컨테이너를 새로 만들면 컨테이너 안에서 설치한 apt 패키지와 venv는 사라질 수 있으므로, Dockerfile에 반영하기 전에는 이 절차로 먼저 재현성을 확인한다.

기본 Docker 이미지의 global Python도 `torch==2.11.0+cu128`로 맞춘다. WildGS-SLAM은 `.venv-wildgs`를 사용하지만, venv 활성화를 깜빡했을 때 RTX 5080에서 `torch==2.1.0+cu121`이 잡히는 문제를 피하기 위해서다.

## 1. 전제

호스트 작업 위치:

```bash
cd /home/leedk/ros_ws
```

컨테이너 이름:

```bash
aac-2026-container-all
```

컨테이너 내부 WildGS-SLAM 위치:

```bash
/ros_ws/src/aac_2026/navigation/WildGS-SLAM
```

이 문서의 명령은 두 종류로 나뉜다.

- `host$`: 호스트 터미널에서 실행
- `container$`: Docker 컨테이너 내부 shell에서 실행

## 2. 컨테이너 기본 상태 확인

호스트에서 컨테이너가 살아 있는지 확인한다.

```bash
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Image}}'
```

GPU가 컨테이너에 보이는지 확인한다.

```bash
docker exec aac-2026-container-all nvidia-smi
```

팀 PC에 RTX 5080과 RTX 5060이 섞여 있어도 둘 다 Blackwell `sm_120` 계열로 취급한다. 단, 5060은 VRAM 여유가 작을 수 있으므로 실제 GPU 이름, compute capability, VRAM을 먼저 확인한다.

```bash
docker exec aac-2026-container-all \
  nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader
```

기대값은 `compute_cap`이 `12.0`인 것이다. CUDA 12.8 GA 기준 toolkit driver는 Linux `570.26` 이상이므로, 팀 PC는 R570/R575/R580 계열처럼 충분히 최신인 host NVIDIA driver를 사용하는 것을 권장한다.

Python, `nvcc`, CUDA header 상태를 확인한다.

```bash
docker exec aac-2026-container-all bash -lc '
python3.10 --version
nvcc --version
for f in cuda_runtime_api.h cublas_v2.h cusolverDn.h cusparse.h; do
  printf "%s: " "$f"
  test -f /usr/local/cuda/include/$f && echo yes || echo no
done
'
```

`cusparse.h`, `cublas_v2.h`, `cusolverDn.h` 중 하나라도 `no`이면 다음 단계에서 CUDA dev library를 설치한다.

## 3. CUDA dev library 설치

새 x86 Dockerfile은 `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`를 사용하므로 이 단계가 필요 없어야 한다. 이미 떠 있는 옛 컨테이너에서만 header가 빠져 있으면 호스트에서 `root` 사용자로 실행한다.

```bash
docker exec -u root aac-2026-container-all apt-get update
docker exec -u root aac-2026-container-all apt-get install -y --no-install-recommends cuda-libraries-dev-12-8
```

설치 후 header가 생겼는지 다시 확인한다.

```bash
docker exec aac-2026-container-all bash -lc '
for f in cuda_runtime_api.h cublas_v2.h cusolverDn.h cusparse.h; do
  printf "%s: " "$f"
  test -f /usr/local/cuda/include/$f && echo yes || echo no
done
'
```

모두 `yes`가 되어야 `droid_backends` 같은 PyTorch CUDA extension 빌드가 통과한다.

## 4. WildGS-SLAM venv 설치

컨테이너 안으로 들어간다.

```bash
docker exec -it aac-2026-container-all bash
```

컨테이너 내부에서 설치 스크립트를 실행한다.

```bash
cd /ros_ws/src/aac_2026/navigation/WildGS-SLAM
bash setup_venv_x86.sh
source .venv-wildgs/bin/activate
```

이 스크립트가 하는 일:

- `.venv-wildgs` 생성
- `torch==2.11.0`, `torchvision==0.26.0`, `torchaudio==2.11.0`을 `cu128` wheel로 설치
- `xformers==0.0.35` 설치
- `numpy==1.26.3` 고정
- `torch_scatter`, upstream `opencv-python` 설치 경로 회피
- Metric3D용 `mmcv==1.7.2`를 `--no-build-isolation`으로 별도 설치
- `lietorch`, `diff-gaussian-rasterization-w-pose`, `simple-knn`, `droid_backends`를 `sm_120`으로 빌드
- `harnesses/check_sm120_patch.py` 실행

## 5. 설치 검증

venv가 활성화된 상태에서 실행한다.

```bash
cd /ros_ws/src/aac_2026/navigation/WildGS-SLAM
source .venv-wildgs/bin/activate
harnesses/check_sm120_patch.py
```

기대 출력:

```text
WildGS-SLAM sm_120 patch harness OK
```

CUDA extension import를 직접 확인한다.

```bash
python - <<'PY'
import numpy, torch
import xformers
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("xformers", xformers.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))

import lietorch
from simple_knn._C import distCUDA2
import diff_gaussian_rasterization
import droid_backends

print("extensions OK")
PY
```

기대 상태:

- `numpy 1.26.3`
- `torch 2.11.0+cu128`
- `xformers 0.0.35`
- GPU capability `(12, 0)`
- `extensions OK`

## 6. Demo 실행 준비

demo 실행에는 extension 빌드 외에 데이터와 pretrained checkpoint가 필요하다.

데모 데이터를 받는다.

```bash
cd /ros_ws/src/aac_2026/navigation/WildGS-SLAM
source .venv-wildgs/bin/activate
bash scripts_downloading/download_demo_data.sh
```

`pretrained/droid.pth`를 준비한다.

```bash
mkdir -p pretrained
pip install gdown
gdown 1PpqVt1H4maBa_GbPJp4NwxRsd9jk-elh -O pretrained/droid.pth
```

Google Drive 제한 때문에 `gdown`이 실패하면 브라우저로 upstream README의 `droid.pth` 링크에서 직접 받은 뒤, 같은 경로에 둔다.

첫 실행 때 Metric3D 모델을 `torch.hub`로 받을 수 있다. cache 위치를 repo 안의 ignored path로 고정하려면 다음 변수를 사용한다.

```bash
export TORCH_HOME=/ros_ws/src/aac_2026/navigation/WildGS-SLAM/pretrained/torch_hub
```

Docker 비대화형 실행을 위해 Metric3D와 FiT3D/DINOv2 `torch.hub.load` 호출은 `trust_repo=True`로 패치되어 있다. 그렇지 않으면 첫 실행 때 저장소 신뢰 여부를 묻는 프롬프트에서 EOF로 중단된다.

`xFormers not available` 경고가 보이면 venv 안에서 다음을 확인한다. 정상 설치되어 있으면 Depth-Anything/DINOv2 attention 경고가 사라지고 일부 attention 연산이 더 가볍게 실행된다.

```bash
source .venv-wildgs/bin/activate
python - <<'PY'
import torch, xformers
from xformers.ops import memory_efficient_attention
q = torch.randn(1, 8, 2, 16, device="cuda", dtype=torch.float16)
out = memory_efficient_attention(q, q, q)
torch.cuda.synchronize()
print("xformers", xformers.__version__, out.shape)
PY
```

Metric3D의 torch.hub 코드에는 최신 xFormers에서 제거된 `xformers.components` API를 확인하는 레거시 코드가 남아 있다. 실제 실행은 fallback으로 정상 동작하므로, 이 오래된 probe에서 나오는 `xFormers not available` 경고는 local wrapper에서 `dinov2` logger를 `ERROR`로 낮춰 숨긴다.

## 7. Docker headless smoke test

Docker 환경에서는 GUI/OpenGL 문제를 줄이기 위해 headless config부터 실행한다.

```bash
cd /ros_ws/src/aac_2026/navigation/WildGS-SLAM
source .venv-wildgs/bin/activate
export TORCH_HOME=/ros_ws/src/aac_2026/navigation/WildGS-SLAM/pretrained/torch_hub
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_headless.yaml
```

출력 위치:

```bash
output/Wild_SLAM_Mocap_demo_headless/crowd_demo
```

이 smoke test가 통과하면 다음을 확인한다.

```bash
find output/Wild_SLAM_Mocap_demo_headless/crowd_demo -maxdepth 3 -type f | head
```

RTX 5060처럼 VRAM이 작은 PC에서는 low-VRAM config부터 실행한다.

```bash
cd /ros_ws/src/aac_2026/navigation/WildGS-SLAM
source .venv-wildgs/bin/activate
export TORCH_HOME=/ros_ws/src/aac_2026/navigation/WildGS-SLAM/pretrained/torch_hub
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_lowvram.yaml
```

이 config는 smoke test 용도로 `max_frames: 50`, `H_out: 176`, `W_out: 320`, `metric3d_vit_small`을 사용한다. DROID feature map 크기 mismatch를 피하려면 `H_out`, `W_out`은 8의 배수로 둔다. 결과 품질 비교나 최종 실험에는 5080에서 더 높은 해상도 config를 다시 사용한다.

## 8. ROS2 image topic으로 실행

ROS2 camera topic을 WildGS-SLAM에 바로 연결할 때는 `scripts_run/ros2_image_topic_wrapper.py`를 사용한다. 이 wrapper는 `sensor_msgs/Image`를 받아 `datasets/ROS2/.../rgb/frame00000.png` 형태로 저장하면서, WildGS-SLAM은 새 `ros2_image_topic` dataset으로 해당 폴더를 읽는다.

기본 모드는 `--frames`개를 처리한 뒤 정상 종료하는 finite run이다. 짧은 smoke test에는 `--frames 50`도 충분하지만, ROS2 live 실험에서는 너무 빨리 끝나므로 `--frames 1000` 이상을 권장한다. 끝나지 않고 계속 topic을 받으려면 `--stream`을 사용한다. 이 경우 기존 tracker의 `len(stream)` 구조 때문에 내부적으로는 `--stream-max-frames`만큼의 큰 frame budget을 사용한다.

WildGS-SLAM은 monocular SLAM이므로 카메라가 충분히 움직여야 tracker warmup과 keyframe 생성이 진행된다. Gazebo에서 차량이 정지한 상태면 topic capture는 성공해도 SLAM map 초기화가 안 될 수 있으니, 전체 실행 테스트 때는 joystick/autonomous command로 차량을 천천히 움직인다.

Gazebo front camera finite run 예시:

```bash
cd /ros_ws/src/aac_2026/navigation/WildGS-SLAM
source /opt/ros/$ROS_DISTRO/setup.bash
[ -f /ros_ws/install/setup.bash ] && source /ros_ws/install/setup.bash
source .venv-wildgs/bin/activate
export TORCH_HOME=/ros_ws/src/aac_2026/navigation/WildGS-SLAM/pretrained/torch_hub

python scripts_run/ros2_image_topic_wrapper.py \
  --config ./configs/ROS2/aac_gazebo_front_lowvram.yaml \
  --image-topic /camera_front/image_raw \
  --camera-info-topic /camera_info \
  --kill-stale-workers \
  --frames 1000
```

계속 스트리밍을 받고 싶으면:

```bash
python scripts_run/ros2_image_topic_wrapper.py \
  --config ./configs/ROS2/aac_gazebo_front_lowvram.yaml \
  --image-topic /camera_front/image_raw \
  --camera-info-topic /camera_info \
  --stream
```

최종 `video.npz`와 map export까지 깔끔하게 남기는 실험은 finite run(`--frames 1000`처럼 충분히 큰 값)을 권장한다. `--stream`은 live 동작 확인용이며, 종료는 `Ctrl+C`로 한다.

`configs/ROS2/aac_gazebo_front_lowvram.yaml`은 ROS2 live 동작 확인을 우선한 가벼운 config다. 오래 걸리는 final BA, loop closure, uncertainty feature training을 끄고 mapping iteration을 줄여 `Mapping Frame N ...` 이후 긴 무응답 시간을 줄인다. 최종 품질 비교가 필요하면 별도 config에서 해당 옵션을 다시 켠다.

만약 `Mapping Frame 99 ...` 같은 출력 뒤에 멈춘 것처럼 보이면 먼저 이전 WildGS worker가 남아 있는지 확인한다.

```bash
ps -eo pid,ppid,pgid,sid,stat,etime,pcpu,pmem,cmd | \
  grep -E "spawn_main|resource_tracker|run.py|ros2_image_topic_wrapper" | \
  grep -v grep
```

부모 PID가 `1`인 WildGS multiprocessing worker가 남아 있으면 다음 실행 때 `--kill-stale-workers`를 붙이거나, 현재 orphan worker를 직접 종료한다.

SLAM을 띄우기 전에 topic capture만 빠르게 확인하려면:

```bash
python scripts_run/ros2_image_topic_wrapper.py \
  --config ./configs/ROS2/aac_gazebo_front_lowvram.yaml \
  --image-topic /camera_front/image_raw \
  --camera-info-topic /camera_info \
  --frames 3 \
  --capture-only
```

USB camera node를 사용할 때는 AAC README 기준 topic에 맞춘다.

```bash
python scripts_run/ros2_image_topic_wrapper.py \
  --config ./configs/ROS2/aac_gazebo_front_lowvram.yaml \
  --image-topic /camera1/image_raw \
  --camera-info-topic /camera1/camera_info \
  --frames 1000
```

wrapper 동작 순서:

- ROS2 image topic을 구독한다.
- `--start-after-frames`만큼 이미지가 저장되면 runtime config를 생성한다.
- CameraInfo가 있으면 `fx`, `fy`, `cx`, `cy`, distortion을 runtime config에 반영한다.
- CameraInfo가 없으면 base config의 camera intrinsics 또는 `--fx`, `--fy`, `--cx`, `--cy` 값을 사용한다.
- WildGS-SLAM `run.py`를 같은 Python으로 subprocess 실행한다.

생성되는 주요 파일:

```bash
datasets/ROS2/aac_gazebo_front/rgb/frame00000.png
datasets/ROS2/aac_gazebo_front/timestamps.txt
configs/ROS2/_runtime_ros2_image_topic.yaml
output/ROS2_aac_gazebo_front/<scene>/
```

CameraInfo topic을 사용할 수 없는 경우:

```bash
python scripts_run/ros2_image_topic_wrapper.py \
  --image-topic /camera_front/image_raw \
  --camera-info-topic '' \
  --frames 1000 \
  --fx 914.0443685627914 \
  --fy 914.0443685627914 \
  --cx 640.0 \
  --cy 360.0
```

Docker venv에서 `rclpy`나 `cv_bridge`가 안 보이면 먼저 ROS 환경을 source한다.

```bash
source /opt/ros/$ROS_DISTRO/setup.bash
source /ros_ws/install/setup.bash
source .venv-wildgs/bin/activate
```

`cv_bridge` 자체가 설치되어 있지 않으면 컨테이너 이미지에 `ros-$ROS_DISTRO-cv-bridge`를 추가해야 한다.

## 9. Docker 이미지로 반영할 때

컨테이너 실험이 통과한 뒤 Dockerfile에는 CUDA 12.8 base image와 global PyTorch 버전을 반영한다.

x86 Dockerfile의 base image는 CUDA 12.8 devel 계열로 둔다. `devel` image를 쓰면 `nvcc`와 CUDA 개발 header가 base image에 포함되어 WildGS-SLAM CUDA extension 빌드가 단순해진다.

```dockerfile
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV TORCH_CUDA_ARCH_LIST=12.0
```

global Python의 PyTorch는 RTX 5080 `sm_120`을 지원하는 `cu128` wheel로 고정한다.

```dockerfile
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

OpenCV CUDA build도 RTX 50-series 기준으로 `CUDA_ARCH_BIN=12.0`을 사용한다. 5080과 5060 모두 같은 `sm_120` 대상이므로 별도 arch를 추가할 필요는 없다. 필요한 경우 Docker build 시 `CUDA_ARCH_BIN` 환경변수로 override할 수 있다.

venv까지 이미지에 bake할지는 별도로 결정한다. `bash setup_venv_x86.sh`를 Docker build 중에 실행하면 이미지 빌드 시간이 길고 용량이 커진다. 개발 중에는 시스템 의존성만 이미지에 넣고, WildGS-SLAM venv는 컨테이너 시작 후 workspace에서 재생성하는 방식이 더 다루기 쉽다.

완전히 고정된 실험 이미지가 필요하면 다음 조건을 만족한 뒤 venv bake를 고려한다.

- `setup_venv_x86.sh`가 깨끗한 컨테이너에서 통과
- `harnesses/check_sm120_patch.py` 통과
- `crowd_demo_headless.yaml` smoke test 통과
- pretrained/data/cache 경로를 이미지에 넣을지 volume으로 둘지 결정

## 9. 문제 해결

`cusparse.h: No such file or directory`

CUDA runtime은 있지만 dev header가 없는 상태다. 호스트에서 다음을 실행한다.

```bash
docker exec -u root aac-2026-container-all apt-get install -y --no-install-recommends cuda-libraries-dev-12-8
```

`NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible`

system Python의 오래된 torch를 사용 중일 가능성이 높다. venv를 활성화했는지 확인한다.

```bash
which python
python -c 'import torch; print(torch.__version__, torch.cuda.get_arch_list())'
```

`numpy 2.x` 관련 경고 또는 extension import 오류

WildGS-SLAM은 NumPy 1.x 기준으로 맞춰둔다.

```bash
source .venv-wildgs/bin/activate
pip install numpy==1.26.3
harnesses/check_sm120_patch.py
```

`ModuleNotFoundError: No module named simple_knn`

`thirdparty/simple-knn/simple_knn/__init__.py`가 있는지 확인하고 설치 스크립트를 다시 실행한다.

```bash
bash setup_venv_x86.sh
```

GUI 또는 OpenGL 관련 오류

Docker 검증에서는 기본 `crowd_demo.yaml` 대신 다음 config를 사용한다.

```bash
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_headless.yaml
```

CUDA out-of-memory

headless config의 해상도 또는 frame 수를 줄인다.

```yaml
max_frames: 50
cam:
  H_out: 176
  W_out: 320
```

## 10. 최소 성공 기준

Dockerfile로 옮기기 전에 최소한 아래가 모두 통과해야 한다.

```bash
cd /ros_ws/src/aac_2026/navigation/WildGS-SLAM
source .venv-wildgs/bin/activate
harnesses/check_sm120_patch.py
python - <<'PY'
import torch
import lietorch
from simple_knn._C import distCUDA2
import diff_gaussian_rasterization
import droid_backends
print(torch.__version__, torch.cuda.get_device_capability(0))
print("OK")
PY
```

그리고 가능하면 다음 smoke test까지 통과시킨다.

```bash
python run.py ./configs/Dynamic/Wild_SLAM_Mocap/crowd_demo_headless.yaml
```
