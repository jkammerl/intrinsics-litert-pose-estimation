# litert_pose_estimation — 6D pose estimation in ROS 2 with LiteRT

A ROS 2 node that finds the OMTS raw stock workpiece in an RGB-D image and
estimates its 6D pose, running every neural network with
**[LiteRT](https://github.com/google-ai-edge/LiteRT)**, Google's on-device
inference runtime (the successor of TensorFlow Lite). It implements the same
pipeline, with the same parameters, as Intrinsic's IOC pose estimator service
in the [Open Machine Tending Solution](https://github.com/intrinsic-ai/intrinsic-omts),
and is verified against that service's recorded inputs and outputs.

```
RGB + depth + camera info ──► RF-DETR segmentation ──► FoundationPose ──► vision_msgs/Detection3DArray
                               (LiteRT)                 (LiteRT, 280 pose
                                                        candidates refined
                                                        and scored)
```

## Why LiteRT

LiteRT runs `.tflite` models on almost any device, from phones and
microcontrollers to Linux PCs and robots, with one API and no Python, CUDA or
framework runtime:

* **One model, every accelerator.** LiteRT's `CompiledModel` API compiles the
  same `.tflite` file for the CPU, the GPU or an NPU at load time. On the GPU,
  LiteRT's **ML Drift** accelerator runs on **WebGPU** (Vulkan on Linux and
  Android, Metal on Apple devices) or **OpenCL**, so it
  works on NVIDIA, AMD, Intel, Arm Mali, Qualcomm Adreno and Apple GPUs, not
  only CUDA devices. NPUs (e.g. Qualcomm, MediaTek, Google Tensor) are
  supported through vendor dispatch libraries.
* **Graceful fallback, op by op.** Ops the GPU accelerator doesn't support
  run on the CPU in the same model, and if the GPU can't be used at all, this
  node recompiles for the CPU (see below). The same binary therefore runs on
  a laptop without a GPU, on a Jetson, or on a robot's embedded board.
* **Fast CPU inference.** On the CPU, LiteRT uses **XNNPACK**, highly
  optimized kernels for Arm NEON/SVE and x86 AVX, multi-threaded.
* **Small and self-contained.** The runtime is a single ~6 MB shared library
  (`libLiteRt.so`) plus an optional GPU accelerator library, with a stable C
  API: easy to ship in a container or on an embedded system, and native on
  arm64.
* **Precision control.** Models run in float32 for bit-for-bit comparable
  results, or float16 on the GPU (`gpu_fp16`) for speed; the repository also
  contains float16-weight models (`*_fp16.tflite`) at half the size.
* **Zero-copy I/O.** LiteRT tensor buffers wrap host memory directly (and
  GPU buffers such as WebGPU buffers), so camera frames don't need to be
  copied into the runtime.

## How the node uses LiteRT

Each network (RF-DETR, the FoundationPose refiner, and a scorer per batch
size) is a `perception::Model` (`cpp/litert_model.h`) that wraps LiteRT's
`CompiledModel`:

| `accelerator` | Behavior |
| :--- | :--- |
| `auto` (default) | If there is a hardware GPU: compile for it (unsupported ops on the CPU), run once on pseudo-random inputs and compare with the CPU; if anything fails or differs, or there is no hardware GPU, use the CPU and log why. |
| `gpu` | The same, but also on a software GPU (e.g. Mesa's llvmpipe), for testing LiteRT's GPU path anywhere. A network that can't run on the GPU (e.g. a tensor larger than its maximum buffer size, or results that differ from the CPU's) still runs on the CPU, and the startup log says why. |
| `cpu` | XNNPACK on all CPU cores. |

At startup the node logs where every network runs, e.g.:

```
[litert_pose_estimation]: FoundationPose refiner runs on the GPU (unsupported ops on the CPU).
```

The GPU accelerator library is installed next to the node
(`lib/litert_pose_estimation/litert/libLiteRtWebGpuAccelerator.so`) and
loaded by LiteRT at runtime, so no system-wide installation is needed. On
Linux it needs a Vulkan driver (for example Mesa, NVIDIA or AMD drivers).

## Pipeline (identical to the IOC pose estimator service)

1. **Segmentation** — RF-DETR (`rfdetr_seg.tflite`). Detections with a
   score or visibility at or below `confidence_threshold` /
   `visibility_threshold` (0.6) are dropped.
2. **Pose estimation** — NVIDIA's FoundationPose, per detection: the object's
   center is guessed from the mask and the depth; 280 pose candidates (40
   views × 7 in-plane rotations) are refined `refinement_iterations` (6) times
   in chunks of `batch_size` (128) by the refiner network
   (`foundationpose_refine.tflite`), comparing a rendering of the CAD model
   with the observed RGB-D crop; then the scorer network
   (`foundationpose_score_b128.tflite`, `_b24`) ranks each chunk's candidates.
   The best candidate, shifted to the CAD model's bounding-box center, is the
   result.

The C++ port (`cpp/foundationpose.cc`) reproduces the service's Python
implementation operation by operation, including its float32 arithmetic and
its CPU rasterizer.

## Interfaces

| Name | Type | Description |
| :--- | :--- | :--- |
| `rgb/image_raw` | `sensor_msgs/Image` (sub) | Color image: `rgb8`, `bgr8`, `rgba8`, `bgra8` or `mono8`. |
| `depth/image_raw` | `sensor_msgs/Image` (sub) | Depth registered to the color image: `32FC1` (m) or `16UC1` (mm). |
| `rgb/camera_info` | `sensor_msgs/CameraInfo` (sub) | Color camera intrinsics (`k`). |
| `~/estimate` | `std_srvs/Trigger` (srv) | Estimates the poses in the latest synchronized frame. |
| `~/detections` | `vision_msgs/Detection3DArray` (pub) | Per object: class (`object_id`), FoundationPose score, pose in the camera frame, bounding box. |
| `~/poses` | `geometry_msgs/PoseArray` (pub) | The poses alone. |
| `~/segmentation` | `sensor_msgs/Image` (pub) | `mono8` instance mask: 0 background, i + 1 for object i. |
| `~/markers` | `visualization_msgs/MarkerArray` (pub) | The objects' bounding boxes, for RViz. |

Parameters (defaults in `config/omts_raw_stock.yaml`): `models_dir`,
`cad_obj` (OBJ in meters), `object_id`, `accelerator`, `gpu_fp16`,
`confidence_threshold`, `visibility_threshold`, `refinement_iterations`,
`batch_size`, `continuous` (estimate every frame instead of on `~/estimate`),
`sync_slop`, `queue_size`, `litert_library_dir`.

## Build and run

```bash
sudo apt install ros-lyrical-vision-msgs   # plus a ROS 2 desktop installation
git clone <this repository> ~/intrinsics-litert-pose-estimation   # with git-lfs
mkdir -p ~/litert_ws && cd ~/litert_ws
source /opt/ros/lyrical/setup.bash
colcon build --base-paths ~/intrinsics-litert-pose-estimation/ros \
    ~/intrinsics-litert-pose-estimation/ros_demo \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# With a camera driver:
ros2 launch litert_pose_estimation pose_estimation.launch.py \
    perception_root:=$HOME/intrinsics-litert-pose-estimation \
    rgb:=/camera/color/image_raw depth:=/camera/depth/image_raw \
    camera_info:=/camera/color/camera_info
ros2 service call /litert_pose_estimation/estimate std_srvs/srv/Trigger

# Or the demo with a recorded frame (ros_demo/):
ros2 launch litert_pose_estimation_demo demo.launch.py \
    repo_root:=$HOME/intrinsics-litert-pose-estimation
```

CMake downloads LiteRT 2.2.0 (the runtime and its WebGPU accelerator, from the
`ai-edge-litert` wheel for the host architecture) and installs them with the
node. The node is also a composable component
(`litert_pose_estimation::PoseEstimationNode`) for zero-copy intra-process
pipelines with a camera driver.

## Tests

`colcon test --packages-select litert_pose_estimation` runs the end-to-end
golden test: it publishes the RGB-D frame that the IOC pose estimator service
received in OMTS's simulated Lab BB-01 cell (`testdata/service_golden`),
calls `~/estimate`, and checks the published detection against the service's
result (translation within 1 mm, rotation within 0.5° modulo the box's
half-turn symmetries, score within 0.05). Set `PERCEPTION_ACCELERATOR=gpu` to
run it on the GPU. The C++ tests in `cpp/` check every stage against the
service (`docs/service_golden.md`).

Results on an arm64 VM (6 cores, no GPU): the node reproduces the service's
pose exactly (0.000 mm, 0.000° modulo symmetry, identical score); RF-DETR
takes 0.5 s and FoundationPose 164 s on the CPU, similar to the service's
ONNX Runtime CPU path. This machine has no GPU: LiteRT's WebGPU accelerator
finds Mesa's software Vulkan device (llvmpipe), compiles the models for it
and fails when running the kernels, so `auto` falls back to the CPU as
designed; GPU timings remain to be measured on GPU hardware.
