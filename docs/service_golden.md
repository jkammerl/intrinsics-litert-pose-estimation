# Golden data from the IOC pose estimator service

`testdata/service_golden/` holds the inputs and outputs of Intrinsic's IOC
pose estimator service (`rs-pose-estimator-service`) for one capture of the
simulated Orbbec Gemini 335Le in OMTS's Lab BB-01 cell. Every test of the
LiteRT pipeline compares against it: the C++ stage tests
(`cpp/service_golden_test.cc`), the ROS 2 node's end-to-end test
(`ros/litert_pose_estimation/test/golden_test.cpp`) and the Intrinsic backend
test (`intrinsic/litert_backend_test.py`).

## How it was recorded

1. OMTS ran in simulation (`--operation_mode=sim`) on Intrinsic Core
   20260922.0, with the raw stock pose estimator registered as OMTS's
   tutorial does (`refinement_iters=6`, confidence and visibility thresholds
   0.6) and the service configured with `batch_size: 128` (OMTS's
   `configs/common/pose_estimator_config.textproto`).
2. `tools/pose_estimation:run_pose_estimation` captured an image from the
   robot's view pose and returned the service's result.
3. Inside the service's pod, a script instantiated `IocPoseEstimatorService`
   with the pod's own configuration and Python environment (through a copy of
   the service's Bazel bootstrap), wrapped its two model clients to record
   their inputs and outputs, and called `RunPoseEstimation` on the same
   capture. It returned the same pose and score as step 2, digit for digit.
4. `tools/make_service_golden.py` re-ran OMTS's FoundationPose model
   (`third_party/foundationpose/model.py`, the Python backend the service
   calls) on the recorded inputs with ONNX Runtime on the CPU, checked that it
   reproduces the service's result (to 1e-6), recorded its intermediate
   results, and wrote the test case.

The ONNX models were byte-identical to the deployed ones (SHA-256 checked).

## Contents

| File | Contents |
| :--- | :--- |
| `case.json` | Camera matrix, image size, the service's configuration, segmentation results (box, score, visibility, mask file) and poses (rotation, translation, score). |
| `rgb.png` | The color image the service received (1280x800). |
| `depth.npy` | The depth image, float32 meters; non-finite where the simulated camera hits nothing. |
| `raw_stock_2x3x5.obj` | The CAD model as the service passed it to FoundationPose (OBJ, meters). |
| `mask_0.png` | The service's segmentation mask. |
| `foundationpose/candidate_poses.npy` | The 280 initial pose candidates. |
| `foundationpose/first_poses.npy`, `first_in1.npy`, `first_in2.npy` | The first 4 candidates of the first chunk and their refiner inputs in the first iteration (rendering and observed crop). |
| `foundationpose/refine_{translation,rotation}_<i>.npy` | The refiner's outputs for each of the 18 refiner calls (3 chunks x 6 iterations). |
| `foundationpose/chunk_scores_<i>.npy` | The scorer's outputs for the 3 chunks (128, 128, 24 candidates). |

## What the tests check

| Stage | Tolerance |
| :--- | :--- |
| Pose candidates | 1e-5 |
| Observed crops (OpenCV warps) | 1e-5 |
| Rendering: covered pixels / points | identical / 1e-4 |
| Refiner outputs (LiteRT vs ONNX) | 1e-3 |
| Segmentation: box / score, visibility / mask IoU | 0.5 px / 2e-3 / > 0.995 |
| End to end: translation / rotation / score | 1 mm / 0.5° / 0.05 |

Measured end to end on the CPU: 0.000 mm, 0.000°, ADD-S 0.000 mm and the
same score (-21.3072).

The rotation is compared modulo the raw stock's symmetries: a 2 x 3 x 5 inch
box looks the same after a half turn about any of its axes, so FoundationPose
candidates that differ by such a turn get the same score up to rounding, and
which one wins the argmax depends on the last bits of the arithmetic. Both
are correct; the LiteRT pipeline and the service may return either.
