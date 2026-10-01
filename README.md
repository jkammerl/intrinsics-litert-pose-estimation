# Intrinsic LiteRT pose estimation

Segmentation and 6D pose estimation of the raw stock workpiece from the
[Open Machine Tending Solution (OMTS)](https://github.com/intrinsic-ai/intrinsic-omts),
running entirely on [LiteRT](https://github.com/google-ai-edge/LiteRT), Google's
on-device inference runtime: on the GPU through LiteRT's WebGPU accelerator,
or with XNNPACK on the CPU. No Triton, ONNX Runtime,
CUDA or Python is needed at runtime. The pipeline follows Intrinsic's IOC pose
estimator service step by step and reproduces its results on the service's
recorded inputs (`docs/service_golden.md`).

| Path | Contents |
| :--- | :--- |
| [`models/`](models/README.md) | The TFLite models, each documenting its input and output tensors in its metadata. |
| `assets/raw_stock_2x3x5/` | The workpiece's CAD model (OBJ and GLB, in meters). |
| `cpp/` | The C++ pipeline: LiteRT model wrapper with GPU/CPU selection (`litert_model.h`), RF-DETR (`rfdetr.h`), the FoundationPose port (`foundationpose.h`), the full pipeline (`pose_estimator.h`), and tests. |
| [`ros/litert_pose_estimation/`](ros/litert_pose_estimation/README.md) | ROS 2 node (`vision_msgs/Detection3DArray`, `~/estimate` service), launch file, golden end-to-end test. |
| `ros_demo/` | ROS 2 demo: publishes a recorded RGB-D frame and triggers the node. |
| [`demo/`](demo/README.md) | Web demo: move the raw stock in a 3D scene, estimate its pose from the rendered RGB-D frame with LiteRT, compare with the truth. |
| `python/` | Python bindings (`litert_pose_estimation` module). |
| [`intrinsic/`](intrinsic/README.md) | Intrinsic integration: LiteRT backend for the IOC pose estimator service, drop-in service image, tests. |
| `testdata/` | Rendered RGB-D scenes with the ONNX models' outputs, and the IOC service's recorded inputs and outputs (`service_golden/`). |
| `tools/` | Python: conversion, test data generation, reference implementations. |
| `docs/` | How the golden data was recorded. |

## Pipeline

1. **Segmentation** (`rfdetr_seg.tflite`): RF-DETR finds the workpiece in an
   RGB image and returns a box, a score, a visibility and a mask per
   detection. Pre-processing (scale to 504 pixels, normalize, pad) and
   post-processing (thresholds, mask upsampling) run outside the model:
   `tools/rfdetr.py` in Python, `cpp/rfdetr.cc` in C++.
2. **Pose estimation** (`foundationpose_refine.tflite`,
   `foundationpose_score.tflite`): NVIDIA's FoundationPose estimates the
   workpiece's pose from the RGB-D image, the mask, the camera intrinsics and
   the CAD model. It samples 280 pose candidates, refines them 5 times by
   rendering the CAD model at each candidate and comparing it with the
   observed crop, then scores them against each other.
   `tools/foundationpose.py` is the Python reference, with the renderer and
   helpers in `tools/foundationpose_numpy.py`. `cpp/foundationpose.cc` is
   the C++ port of the service's FoundationPose model, including its CPU
   rasterizer, matching its float32 arithmetic.

The tensor specs are in [models/README.md](models/README.md).

### Differences from OMTS

* OMTS's `segmentation.onnx` includes pre- and post-processing, with dynamic
  shapes that TFLite can't express. `rfdetr_seg.tflite` is only the network
  between them; together with `rfdetr.py`, the results are the same.
* OMTS's FoundationPose model processes the 280 candidates in chunks of 240
  and 40. Here the refiner runs on batches of 40 (candidates are independent,
  so the results are the same), and the scorer on all 280 at once, since the
  TFLite models need fixed batch sizes. The scores differ slightly from OMTS's,
  because the scorer compares the candidates in a batch against each other.

The C++ pipeline (`cpp/pose_estimator.h`) also follows the service in the
FoundationPose chunking: it refines and scores the candidates in chunks of
the service's `batch_size` (128 in OMTS) with scorers converted for those
batch sizes (`foundationpose_score_b128.tflite`, `_b24`; and `_b240`, `_b40`
for the service's default of 240), so that its scores match the service's.
Other batch sizes need their scorers (`tools/convert.py --score_batches`).

## C++ tests

The tests run the TFLite models on the test data and compare them with the
original ONNX models' outputs, and check each model's documented tensor spec.
They need CMake 3.24 or later and a C++17 compiler, and download pinned
versions of LiteRT (headers from the v2.2.0 release and `libLiteRt.so` from
the `ai-edge-litert` 2.2.0 wheel), GoogleTest, nlohmann/json and stb_image:

```bash
cmake -S cpp -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Linux x86-64 and arm64 are supported. On 6 arm64 cores, the suite takes
about 2.5 minutes and peaks at about 5 GB of memory, mostly for the scorer's
batch of 280 candidates.

The tests check, against the ONNX models' outputs:

* every model's `io_spec` metadata against its actual tensors;
* each network on the saved inputs (float32 and float16 models);
* the full pipeline against the IOC pose estimator service
  (`cpp/service_golden_test.cc`, `docs/service_golden.md`): pose candidates,
  network inputs, refiner, segmentation and the final pose. Set
  `PERCEPTION_ACCELERATOR=auto|gpu|cpu` to choose LiteRT's hardware;
* RF-DETR end to end, from `rgb.png` through `cpp/rfdetr.cc` and the TFLite
  model to boxes, scores and masks, against the original `segmentation.onnx`
  (masks agree to IoU > 0.99; so far they are identical);
* LiteRT's GPU accelerator (`cpp/gpu_test.cc`): RF-DETR and the
  FoundationPose refiner on the GPU through WebGPU (Vulkan) against the
  service's outputs and the CPU's; the scorers, which some GPUs can't hold
  (157 MB tensors at batch 128) or compute wrongly (llvmpipe at batch 24):
  an error without `cpu_fallback`, the CPU's scores with it; and `auto` using
  the CPU without a hardware GPU. A software Vulkan device (Mesa's llvmpipe)
  is enough to run them; without any Vulkan device they are skipped.

## GPU or CPU

`perception::ModelOptions` (`cpp/litert_model.h`) selects the hardware:

| `accelerator` | |
| :--- | :--- |
| `auto` | The GPU if there is a hardware GPU, else the CPU. |
| `gpu` | The GPU, also a software one (e.g. Mesa's llvmpipe), for testing. |
| `cpu` | XNNPACK on all CPU cores. |

On the GPU, ops that LiteRT's GPU accelerator doesn't support run on the CPU
within the same model. Before a network runs on the GPU, it is run once on
pseudo-random inputs on the GPU and on the CPU, and the outputs are compared
(`validate_gpu`): GPU drivers can compile and run a model and still compute
it wrongly. A network that fails to compile or run on the GPU, or computes
other results than the CPU, is an error by default. With `cpu_fallback` it
runs on the CPU instead, and the reason is reported (`fallback_reason()`).
The ROS node (`cpu_fallback` parameter), the Python module (`cpu_fallback=`),
the Intrinsic service (`LITERT_CPU_FALLBACK=1`) and the web demo expose the
switch.

For Android, build the same sources with the NDK and link `libLiteRt.so` from
LiteRT's Maven package (`com.google.ai.edge.litert`).

## Test data

`testdata/scenes/<scene>/` holds one 640x480 RGB-D scene each:

| File | Contents |
| :--- | :--- |
| `rgb.png` | The image. |
| `depth.npy` | float32 depth in meters. |
| `mask.png` | Ground-truth mask of the workpiece. |
| `scene.json` | Camera intrinsics `K`, and the workpiece's ground-truth `pose` (4x4, CAD model frame in camera frame; `null` if absent). |
| `segmentation/input.npy` | Pre-processed input of `rfdetr_seg.tflite`. |
| `segmentation/{logits,visibility_logits,boxes,mask_logits}.npy` | The network's outputs, from ONNX Runtime. |
| `segmentation/result.json`, `mask_<i>.png` | Detections of the original `segmentation.onnx` (with pre- and post-processing), at OMTS's thresholds. |
| `foundationpose/result.json` | Estimated pose and score from `tools/foundationpose.py` with the ONNX models, and its error to the ground truth (ADD-S). |
| `foundationpose/{initial,refined}_poses.npy` | The 280 candidates before refinement and after the last refinement. |

`testdata/networks/` holds the FoundationPose networks' inputs and outputs
from the first scene's first refinement (16 candidates) and scoring (all 280).
Inputs are float16 to save space; the outputs were computed from the rounded
inputs, so they match.

The scenes are rendered: the CAD model with simple shading on a textured,
noisy table plane, with slightly noisy depth. The scenes are:
`raw_stock_top`, `raw_stock_tilted`, `raw_stock_side`, and `empty` (no
workpiece; no detections expected).

## Regenerating

```bash
python3.12 -m venv venv && venv/bin/pip install -r tools/requirements.txt
cd tools
../venv/bin/python convert.py --segmentation_onnx=segmentation.onnx \
    --refine_onnx=foundationpose_refine.onnx \
    --score_onnx=foundationpose_score.onnx --out=../models
../venv/bin/python add_metadata.py --models=../models
../venv/bin/python make_testdata.py --segmentation_onnx=segmentation.onnx \
    --rfdetr_core_onnx=../models/rfdetr_core.onnx \
    --refine_onnx=foundationpose_refine.onnx \
    --score_onnx=foundationpose_score.onnx --out=../testdata
```

`segmentation.onnx` is in `segmentation.tar.gz` of the
[intrinsic-omts 20260922.0 release](https://github.com/intrinsic-ai/intrinsic-omts/releases/tag/20260922.0).
The FoundationPose models are `refine_model.onnx` and `score_model.onnx` from
NVIDIA's [FoundationPose model on NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/isaac/models/foundationpose),
version 1.0.0_onnx.

## Licenses

The code, the segmentation model and the CAD model are under the Apache
License 2.0 ([LICENSE](LICENSE)); `tools/foundationpose_numpy.py` is a port of
NVIDIA's Apache-2.0 FoundationPose code. The FoundationPose models are under
the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
