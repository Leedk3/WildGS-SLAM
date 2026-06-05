#!/usr/bin/env python3
"""Lightweight checks for the local RTX 50-series WildGS-SLAM patch."""

import ast
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def load_function(path: Path, function_name: str):
    module = ast.parse(path.read_text())
    matches = [
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    if not matches:
        raise AssertionError(f"{function_name} not found in {path}")

    namespace = {"torch": torch}
    ast.fix_missing_locations(matches[0])
    exec(compile(ast.Module(body=[matches[0]], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[function_name]


def reference_sum(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int):
    shape = list(src.shape)
    shape[dim] = dim_size
    result = torch.zeros(shape, dtype=src.dtype, device=src.device)

    for src_pos in range(src.shape[dim]):
        dst_pos = int(index[src_pos])
        result.select(dim, dst_pos).add_(src.select(dim, src_pos))

    return result


def reference_mean(src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int):
    result = reference_sum(src, index, dim, dim_size)
    counts = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    counts.scatter_add_(0, index, torch.ones_like(index, dtype=src.dtype))

    shape = [1] * src.dim()
    shape[dim] = dim_size
    return result / counts.clamp(min=1).view(shape)


def assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor):
    if not torch.allclose(actual, expected):
        raise AssertionError(
            f"{name} mismatch\nactual={actual}\nexpected={expected}"
        )


def check_scatter_ops():
    scatter_sum = load_function(ROOT / "src/geom/ba.py", "scatter_sum")
    scatter_mean = load_function(ROOT / "src/modules/droid_net/droid_net.py", "scatter_mean")

    src0 = torch.arange(12, dtype=torch.float32).view(3, 4)
    index0 = torch.tensor([0, 1, 0])
    assert_close(
        "scatter_sum_dim0",
        scatter_sum(src0, index0, dim=0, dim_size=2),
        reference_sum(src0, index0, dim=0, dim_size=2),
    )
    assert_close(
        "scatter_mean_dim0",
        scatter_mean(src0, index0, dim=0, dim_size=2),
        reference_mean(src0, index0, dim=0, dim_size=2),
    )

    src1 = torch.arange(2 * 5 * 3, dtype=torch.float32).view(2, 5, 3)
    index1 = torch.tensor([0, 2, 0, 1, 2])
    assert_close(
        "scatter_sum_dim1",
        scatter_sum(src1, index1, dim=1, dim_size=3),
        reference_sum(src1, index1, dim=1, dim_size=3),
    )
    assert_close(
        "scatter_mean_dim1",
        scatter_mean(src1, index1, dim=1, dim_size=3),
        reference_mean(src1, index1, dim=1, dim_size=3),
    )


def check_build_files():
    setup_py = (ROOT / "setup.py").read_text()
    lietorch_setup_py = (ROOT / "thirdparty/lietorch/setup.py").read_text()
    setup_conda = (ROOT / "setup_conda.sh").read_text()
    setup_venv = (ROOT / "setup_venv_x86.sh").read_text()
    metric_depth_estimators = (
        ROOT / "src/utils/mono_priors/metric_depth_estimators.py"
    ).read_text()
    img_feature_extractors = (
        ROOT / "src/utils/mono_priors/img_feature_extractors.py"
    ).read_text()
    slam_py = (ROOT / "src/slam.py").read_text()

    required = [
        "compute_120,code=sm_120",
        "thirdparty/lietorch",
        'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"',
        "torch==2.11.0",
        "cu128",
        "xformers==0.0.35",
        "torch[-_]scatter|torch_scatter",
        "opencv-python==4.8.1.78",
        "mmcv==1.7.2",
        "cuda-libraries-dev-12-8",
    ]
    combined = setup_py + "\n" + lietorch_setup_py + "\n" + setup_conda + "\n" + setup_venv
    missing = [item for item in required if item not in combined]
    if missing:
        raise AssertionError(f"Missing expected sm_120 setup entries: {missing}")

    if "compute_120,code=sm_120" not in setup_py:
        raise AssertionError("Main droid_backends setup.py is missing sm_120")
    if "compute_120,code=sm_120" not in lietorch_setup_py:
        raise AssertionError("thirdparty/lietorch setup.py is missing sm_120")
    if "trust_repo=True" not in metric_depth_estimators:
        raise AssertionError("Metric3D torch.hub load must be non-interactive")
    if "trust_repo=True" not in img_feature_extractors:
        raise AssertionError("Feature extractor torch.hub load must be non-interactive")
    if 'getLogger("dinov2").setLevel(logging.ERROR)' not in metric_depth_estimators:
        raise AssertionError("Metric3D stale xFormers warning should be suppressed")
    if 'getLogger("dinov2").setLevel(logging.ERROR)' not in img_feature_extractors:
        raise AssertionError("Feature extractor stale xFormers warning should be suppressed")
    if "mapper_ready" not in slam_py:
        raise AssertionError("SLAM termination should guard short smoke tests")


def check_ros2_mapper_patches():
    mapper_path = ROOT / "src/mapper.py"
    mapper_py = mapper_path.read_text()
    mapper_ast = ast.parse(mapper_py)

    bad_calls = []
    for node in ast.walk(mapper_ast):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "get_loss_mapping":
            if len(node.args) > 4:
                bad_calls.append(node.lineno)

    if bad_calls:
        raise AssertionError(
            "get_loss_mapping calls must not pass opacity positionally; "
            f"bad calls at lines {bad_calls}"
        )

    if "torch.ones_like(rendered_depth[0]).detach().cpu()" not in mapper_py:
        raise AssertionError(
            "Non-uncertainty plotting should create CPU 2D placeholder maps"
        )

    ros2_wrapper = (ROOT / "scripts_run/ros2_image_topic_wrapper.py").read_text()
    ros2_config = (ROOT / "configs/ROS2/aac_gazebo_front_lowvram.yaml").read_text()

    for required in [
        "--kill-stale-workers",
        "stale_worker_pids",
        "kill_stale_workers",
        "stream_max_frames",
    ]:
        if required not in ros2_wrapper:
            raise AssertionError(f"ROS2 wrapper missing {required}")

    for required in [
        "dataset: 'ros2_image_topic'",
        "max_frames: 1000",
        "final_ba: False",
        "enable_loop: False",
        "activate: False",
    ]:
        if required not in ros2_config:
            raise AssertionError(f"ROS2 low-VRAM config missing {required}")


def check_runtime_versions():
    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        raise AssertionError(f"NumPy 2.x is installed ({np.__version__}); use numpy==1.26.3")

    if torch.cuda.is_available() and "sm_120" not in torch.cuda.get_arch_list():
        raise AssertionError(f"PyTorch CUDA arch list does not include sm_120: {torch.cuda.get_arch_list()}")

    try:
        import xformers
        from xformers.ops import memory_efficient_attention
    except ImportError as exc:
        raise AssertionError("xFormers is not installed") from exc

    if torch.cuda.is_available():
        q = torch.randn(1, 8, 2, 16, device="cuda", dtype=torch.float16)
        out = memory_efficient_attention(q, q, q)
        torch.cuda.synchronize()
        if out.shape != q.shape or not torch.isfinite(out).all():
            raise AssertionError(f"xFormers attention smoke test failed: {out.shape}")


def main():
    check_scatter_ops()
    check_build_files()
    check_ros2_mapper_patches()
    check_runtime_versions()
    print("WildGS-SLAM sm_120 patch harness OK")


if __name__ == "__main__":
    main()
