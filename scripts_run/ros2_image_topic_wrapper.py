#!/usr/bin/env python3
"""Capture a ROS2 Image topic and run WildGS-SLAM on the captured stream."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
WILDGS_WORKER_MARKER = f"{REPO_ROOT}/.venv-wildgs/bin/python -c from multiprocessing"


def add_ros_python_paths() -> None:
    """Make ROS2 Python modules visible from the WildGS virtualenv."""

    ros_distro = os.environ.get("ROS_DISTRO", "humble")
    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(f"/opt/ros/{ros_distro}/lib/{py_ver}/site-packages"),
        Path(f"/opt/ros/{ros_distro}/local/lib/{py_ver}/dist-packages"),
        Path(f"/opt/ros/{ros_distro}/lib/{py_ver}/dist-packages"),
        Path("/usr/lib/python3/dist-packages"),
    ]

    for path in candidates:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.append(path_str)


def import_ros2_modules():
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo, Image
    except ModuleNotFoundError:
        add_ros_python_paths()
        try:
            import rclpy
            from cv_bridge import CvBridge
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ROS2 Python modules are not visible. Source ROS first, e.g. "
                "`source /opt/ros/$ROS_DISTRO/setup.bash`, and make sure "
                "`ros-$ROS_DISTRO-cv-bridge` is installed."
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                "ROS2 Python modules were found, but ROS shared libraries were not. "
                "Start a fresh shell and source ROS before activating the WildGS venv: "
                "`source /opt/ros/$ROS_DISTRO/setup.bash && source .venv-wildgs/bin/activate`."
            ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 Python modules were found, but ROS shared libraries were not. "
            "Start a fresh shell and source ROS before activating the WildGS venv: "
            "`source /opt/ros/$ROS_DISTRO/setup.bash && source .venv-wildgs/bin/activate`."
        ) from exc

    return rclpy, CvBridge, SingleThreadedExecutor, qos_profile_sensor_data, CameraInfo, Image


def stale_worker_pids() -> list[int]:
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid,ppid,cmd"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.SubprocessError:
        return []

    pids = []
    for line in proc.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid_str, ppid_str, cmd = parts
        if WILDGS_WORKER_MARKER in cmd and ppid_str == "1":
            pids.append(int(pid_str))
    return pids


def kill_stale_workers(timeout_s: float = 3.0) -> list[int]:
    pids = stale_worker_pids()
    if not pids:
        return []

    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not stale_worker_pids():
            return pids
        time.sleep(0.1)

    for pid in stale_worker_pids():
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    return pids


class ImageTopicRecorder:
    def __init__(
        self,
        image_topic: str,
        camera_info_topic: str | None,
        output_dir: Path,
        max_frames: int | None,
        image_encoding: str,
        frame_stride: int,
    ) -> None:
        (
            self.rclpy,
            bridge_cls,
            self.executor_cls,
            qos_profile_sensor_data,
            camera_info_cls,
            image_cls,
        ) = import_ros2_modules()

        self.output_dir = output_dir
        self.rgb_dir = output_dir / "rgb"
        self.max_frames = max_frames
        self.image_encoding = image_encoding
        self.frame_stride = frame_stride

        self.bridge = bridge_cls()
        self.first_frame_event = threading.Event()
        self.done_event = threading.Event()
        self.camera_info_event = threading.Event()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.received_count = 0
        self.saved_count = 0
        self.first_shape: tuple[int, int] | None = None
        self.camera_info: dict[str, Any] | None = None
        self.timestamps: list[tuple[int, int, int]] = []

        self.rclpy.init(args=None)
        self.node = self.rclpy.create_node("wildgs_ros2_image_topic_recorder")
        self.node.create_subscription(
            image_cls,
            image_topic,
            self._on_image,
            qos_profile_sensor_data,
        )

        if camera_info_topic:
            self.node.create_subscription(
                camera_info_cls,
                camera_info_topic,
                self._on_camera_info,
                qos_profile_sensor_data,
            )

        self.executor = self.executor_cls()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)

    def start(self) -> None:
        self.spin_thread.start()

    def shutdown(self) -> None:
        self.stop_event.set()
        self.write_metadata()
        try:
            self.executor.shutdown()
        finally:
            self.node.destroy_node()
            self.rclpy.shutdown()
        self.spin_thread.join(timeout=2.0)

    def wait_for_first_frame(self, timeout_s: float) -> bool:
        return self.first_frame_event.wait(timeout_s)

    def wait_for_camera_info(self, timeout_s: float) -> bool:
        return self.camera_info_event.wait(timeout_s)

    def wait_until_buffered(self, frame_count: int, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self.lock:
                if self.saved_count >= frame_count:
                    return True
            if self.done_event.is_set():
                return self.saved_count >= frame_count
            time.sleep(0.05)
        return False

    def wait_done(self) -> None:
        self.done_event.wait()

    def write_metadata(self) -> None:
        timestamps_path = self.output_dir / "timestamps.txt"
        with timestamps_path.open("w", encoding="utf-8") as fp:
            for frame_idx, sec, nanosec in self.timestamps:
                fp.write(f"{frame_idx:05d} {sec}.{nanosec:09d}\n")

    def _on_camera_info(self, msg) -> None:
        if self.camera_info is not None:
            return

        self.camera_info = {
            "height": int(msg.height),
            "width": int(msg.width),
            "k": [float(x) for x in msg.k],
            "d": [float(x) for x in msg.d],
        }
        self.camera_info_event.set()

    def _on_image(self, msg) -> None:
        if self.stop_event.is_set() or self.done_event.is_set():
            return

        with self.lock:
            received_idx = self.received_count
            self.received_count += 1

            if received_idx % self.frame_stride != 0:
                return

            frame_idx = self.saved_count
            if self.max_frames is not None and frame_idx >= self.max_frames:
                self.done_event.set()
                return

        try:
            import cv2

            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
        except ModuleNotFoundError:
            self.node.get_logger().error(
                "OpenCV Python module is missing. Activate .venv-wildgs or install opencv-python."
            )
            self.done_event.set()
            return
        except Exception as exc:
            self.node.get_logger().error(f"Failed to convert image: {exc}")
            return

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if self.first_shape is None:
            self.first_shape = (int(image.shape[0]), int(image.shape[1]))
            self.first_frame_event.set()

        path = self.rgb_dir / f"frame{frame_idx:05d}.png"
        tmp_path = self.rgb_dir / f".frame{frame_idx:05d}.tmp.png"
        if not cv2.imwrite(str(tmp_path), image):
            self.node.get_logger().error(f"Failed to write image: {tmp_path}")
            return
        os.replace(tmp_path, path)

        stamp = msg.header.stamp
        self.timestamps.append((frame_idx, int(stamp.sec), int(stamp.nanosec)))

        with self.lock:
            self.saved_count += 1
            if self.max_frames is not None and self.saved_count >= self.max_frames:
                self.write_metadata()
                (self.output_dir / ".capture_done").write_text("done\n", encoding="utf-8")
                self.done_event.set()


def prepare_capture_dir(path: Path, clear_input: bool) -> None:
    if clear_input and path.exists():
        rgb_dir = path / "rgb"
        if rgb_dir.exists():
            shutil.rmtree(rgb_dir)
        for stale_file in [path / ".capture_done", path / "timestamps.txt"]:
            if stale_file.exists():
                stale_file.unlink()

    (path / "rgb").mkdir(parents=True, exist_ok=True)


def camera_overrides(args, recorder: ImageTopicRecorder) -> dict[str, Any]:
    if recorder.first_shape is None:
        raise RuntimeError("No image shape is available yet.")

    height, width = recorder.first_shape
    cam = {
        "H": height,
        "W": width,
        "distortion": None,
    }

    if recorder.camera_info is not None:
        info = recorder.camera_info
        k = info["k"]
        cam.update(
            {
                "H": info["height"] or height,
                "W": info["width"] or width,
                "fx": k[0],
                "fy": k[4],
                "cx": k[2],
                "cy": k[5],
            }
        )
        if info["d"]:
            cam["distortion"] = info["d"]
    elif all(value is not None for value in [args.fx, args.fy, args.cx, args.cy]):
        cam.update(
            {
                "fx": args.fx,
                "fy": args.fy,
                "cx": args.cx,
                "cy": args.cy,
            }
        )

    if args.h_out is not None:
        cam["H_out"] = args.h_out
    if args.w_out is not None:
        cam["W_out"] = args.w_out

    return cam


def write_runtime_config(args, recorder: ImageTopicRecorder) -> Path:
    config_path = Path(args.runtime_config).resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    cfg: dict[str, Any] = {
        "inherit_from": str(Path(args.config).resolve()),
        "dataset": "ros2_image_topic",
        "scene": args.scene,
        "max_frames": -1 if args.stream else args.frames,
        "data": {
            "input_folder": str(Path(args.input_folder).resolve()),
        },
        "cam": camera_overrides(args, recorder),
        "ros2": {
            "file_timeout_s": args.file_timeout,
            "poll_interval_s": args.poll_interval,
            "stream_max_frames": args.stream_max_frames,
        },
    }

    if args.output:
        cfg["data"]["output"] = args.output
    if args.headless:
        cfg["gui"] = False
        cfg["mapping"] = {"online_plotting": False}

    with config_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False)

    return config_path


def launch_wildgs(runtime_config: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(REPO_ROOT / "pretrained" / "torch_hub"))
    env.setdefault("PYTHONUNBUFFERED", "1")
    cmd = [sys.executable, "run.py", str(runtime_config)]
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        start_new_session=True,
    )


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=10)
        return
    except KeyboardInterrupt:
        print("[ros2-wrapper] Escalating WildGS shutdown after repeated interrupt.", flush=True)
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=5)
        return
    except KeyboardInterrupt:
        print("[ros2-wrapper] Force killing WildGS process group.", flush=True)
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait(timeout=5)


def wait_for_capture_or_process_exit(
    recorder: ImageTopicRecorder, proc: subprocess.Popen
) -> int | None:
    while not recorder.done_event.wait(timeout=0.5):
        if proc.poll() is not None:
            return proc.returncode
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WildGS-SLAM from a ROS2 sensor_msgs/Image topic."
    )
    parser.add_argument(
        "--config",
        default="./configs/ROS2/aac_gazebo_front_lowvram.yaml",
        help="Base WildGS-SLAM config to inherit from.",
    )
    parser.add_argument(
        "--image-topic",
        default="/camera_front/image_raw",
        help="ROS2 sensor_msgs/Image topic.",
    )
    parser.add_argument(
        "--camera-info-topic",
        default="/camera_info",
        help="Optional ROS2 sensor_msgs/CameraInfo topic. Use '' to disable.",
    )
    parser.add_argument("--frames", type=int, default=1000, help="Number of frames to run.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Keep recording images until interrupted. WildGS still receives a large "
            "virtual frame budget from --stream-max-frames."
        ),
    )
    parser.add_argument(
        "--stream-max-frames",
        type=int,
        default=100000,
        help="Virtual WildGS frame budget used with --stream.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Save every Nth received image.",
    )
    parser.add_argument(
        "--input-folder",
        default="./datasets/ROS2/aac_gazebo_front",
        help="Folder used as the live RGB_NoPose dataset.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override data.output in the runtime config.",
    )
    parser.add_argument("--scene", default="ros2_image_topic")
    parser.add_argument(
        "--runtime-config",
        default="./configs/ROS2/_runtime_ros2_image_topic.yaml",
        help="Generated config path.",
    )
    parser.add_argument(
        "--encoding",
        default="bgr8",
        help="cv_bridge desired encoding for the image topic.",
    )
    parser.add_argument(
        "--start-after-frames",
        type=int,
        default=2,
        help="Start WildGS after this many frames are saved.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for initial image frames.",
    )
    parser.add_argument(
        "--camera-info-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for CameraInfo when a topic is provided.",
    )
    parser.add_argument(
        "--file-timeout",
        type=float,
        default=120.0,
        help="Seconds WildGS dataset waits for each captured file.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Seconds between file existence checks in the WildGS dataset.",
    )
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--cx", type=float, default=None)
    parser.add_argument("--cy", type=float, default=None)
    parser.add_argument("--h-out", type=int, default=None)
    parser.add_argument("--w-out", type=int, default=None)
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Disable GUI and online plotting in the generated config.",
    )
    parser.add_argument(
        "--no-clear-input",
        action="store_true",
        help="Keep existing captured images in the input folder.",
    )
    parser.add_argument(
        "--capture-only",
        action="store_true",
        help="Capture frames and write the runtime config without launching WildGS-SLAM.",
    )
    parser.add_argument(
        "--kill-stale-workers",
        action="store_true",
        help="Terminate orphaned WildGS multiprocessing workers before starting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.stream_max_frames <= 0:
        raise ValueError("--stream-max-frames must be positive")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.start_after_frames <= 0:
        raise ValueError("--start-after-frames must be positive")

    if args.kill_stale_workers:
        killed = kill_stale_workers()
        if killed:
            print(f"[ros2-wrapper] Killed stale WildGS workers: {killed}", flush=True)
    else:
        stale_pids = stale_worker_pids()
        if stale_pids:
            print(
                "[ros2-wrapper] Warning: stale WildGS worker processes are still running: "
                f"{stale_pids}. Stop them or rerun with --kill-stale-workers.",
                flush=True,
            )

    input_folder = Path(args.input_folder).resolve()
    prepare_capture_dir(input_folder, clear_input=not args.no_clear_input)

    capture_limit = None if args.stream else args.frames
    camera_info_topic = args.camera_info_topic or None
    recorder: ImageTopicRecorder | None = None
    proc: subprocess.Popen | None = None
    try:
        recorder = ImageTopicRecorder(
            image_topic=args.image_topic,
            camera_info_topic=camera_info_topic,
            output_dir=input_folder,
            max_frames=capture_limit,
            image_encoding=args.encoding,
            frame_stride=args.frame_stride,
        )
        recorder.start()
        print(f"[ros2-wrapper] Waiting for image topic: {args.image_topic}", flush=True)
        if not recorder.wait_for_first_frame(args.startup_timeout):
            raise TimeoutError(f"No image received within {args.startup_timeout:.1f}s")

        if camera_info_topic and not recorder.wait_for_camera_info(args.camera_info_timeout):
            print(
                "[ros2-wrapper] CameraInfo was not received in time; "
                "using base config intrinsics or --fx/--fy/--cx/--cy overrides.",
                flush=True,
            )

        start_after = min(args.start_after_frames, args.frames)
        if not recorder.wait_until_buffered(start_after, args.startup_timeout):
            raise TimeoutError(
                f"Only buffered {recorder.saved_count}/{start_after} frames "
                f"within {args.startup_timeout:.1f}s"
            )

        runtime_config = write_runtime_config(args, recorder)
        print(f"[ros2-wrapper] Runtime config: {runtime_config}", flush=True)
        print(f"[ros2-wrapper] Capturing to: {input_folder}", flush=True)
        if args.stream:
            print(
                "[ros2-wrapper] Streaming mode: recording until interrupted "
                f"(WildGS frame budget: {args.stream_max_frames}).",
                flush=True,
            )
        else:
            print(
                f"[ros2-wrapper] Finite mode: WildGS will finish after {args.frames} frames.",
                flush=True,
            )

        if args.capture_only:
            if args.stream:
                print("[ros2-wrapper] Capture-only streaming; press Ctrl+C to stop.", flush=True)
                while True:
                    time.sleep(1.0)
            else:
                recorder.wait_done()
                print(f"[ros2-wrapper] Captured {recorder.saved_count} frames.", flush=True)
            return 0

        proc = launch_wildgs(runtime_config)
        if args.stream:
            return proc.wait()

        early_return_code = wait_for_capture_or_process_exit(recorder, proc)
        if early_return_code is not None:
            print(
                f"[ros2-wrapper] WildGS exited before capture finished: {early_return_code}",
                flush=True,
            )
            return early_return_code
        print(f"[ros2-wrapper] Captured {recorder.saved_count} frames.", flush=True)
        return proc.wait()
    except KeyboardInterrupt:
        print("[ros2-wrapper] Interrupted.", flush=True)
        return 130
    except (RuntimeError, TimeoutError) as exc:
        print(f"[ros2-wrapper] {exc}", flush=True)
        return 1
    finally:
        if proc is not None:
            stop_process(proc)
        if recorder is not None:
            recorder.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
