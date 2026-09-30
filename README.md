# Raw stock perception with LiteRT

Segmentation and 6D pose estimation of the raw stock workpiece from the
[Open Machine Tending Solution (OMTS)](https://github.com/intrinsic-ai/intrinsic-omts),
as TFLite models that run with [LiteRT](https://ai.google.dev/edge/litert), and
nothing else: no Intrinsic platform, Triton or ONNX Runtime. It contains
everything needed to run both steps, for example in an Android app:

| Path | Contents |
| :--- | :--- |
| [`models/`](models/README.md) | The TFLite models, each documenting its input and output tensors in its metadata. |
| `assets/raw_stock_2x3x5/` | The workpiece's CAD model (OBJ and GLB, in meters). |
| `testdata/` | RGB-D test scenes, with the original models' outputs at every step. |
| `cpp/` | C++ code that runs the models with LiteRT's C API, and tests against the test data. |
| `tools/` | Python: conversion, test data generation, and reference implementations. |

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
   helpers in `tools/foundationpose_numpy.py`; there is no C++ port yet.

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
* RF-DETR end to end, from `rgb.png` through `cpp/rfdetr.cc` and the TFLite
  model to boxes, scores and masks, against the original `segmentation.onnx`
  (masks agree to IoU > 0.99; so far they are identical). For Android, build the same sources
with the NDK and link `libLiteRt.so` from LiteRT's Maven package
(`com.google.ai.edge.litert`).

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
