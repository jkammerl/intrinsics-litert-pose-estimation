# Using the LiteRT pipeline in Intrinsic

The LiteRT pipeline reproduces Intrinsic's IOC pose estimator service (the
`PoseEstimationService` that OMTS's `estimate_pose_multi_view` skill calls),
checked against the service's recorded inputs and outputs
(`docs/service_golden.md`). There are three ways to use it in an Intrinsic
solution.

## 1. Drop-in service image (recommended)

`Dockerfile` layers the pipeline onto the service's own image
(`ioc_pose_estimator_image`): the service's code, configuration, KV-store
capture loading, asset (CAD model, thresholds, iterations) loading and gRPC
API stay exactly as Intrinsic ships them. Only its two calls to the ML
inference service (`segmentation_model.run_inference`,
`pose_estimator_model.run_inference`) are replaced, in-process, by LiteRT
(`litert_backend.py`, started by `litert_service_main.py`).

```bash
# The base image, e.g. from an intrinsic-core or OMTS build:
podman load -i bazel-bin/external/intrinsic-core+/intrinsic_perception/intrinsic/perception/service/ioc_pose_estimator/ioc_pose_estimator_image.tar
podman build -f intrinsic/Dockerfile \
    --build-arg BASE_IMAGE=localhost/intrinsic_perception/intrinsic/perception/service/ioc_pose_estimator/ioc_pose_estimator_image:latest \
    -t ioc-pose-estimator-litert .
```

Then use this image for the pose estimator service asset (the
`intrinsic_service` rule's `images`, or replace the image of an existing
solution's service). Consequences for the solution:

* The ML inference service and its model server are no longer needed for
  pose estimation. No GPU is required; with one, LiteRT uses it through its
  WebGPU (Vulkan) accelerator without CUDA, also on non-NVIDIA GPUs.
* Environment: `LITERT_ACCELERATOR` (`auto`, `gpu`, `cpu`),
  `LITERT_GPU_FP16`, `LITERT_MODELS_DIR`.
* The `ENTRYPOINT` names the service's bundled Python for aarch64
  (`rules_python++python+python_3_11_aarch64-unknown-linux-gnu`); on x86-64,
  replace `aarch64` with `x86_64` (and build on an x86-64 host).

The pose estimator asset is still registered with
`register_using_train_service` as before; nothing else in OMTS changes.

## 2. ROS 2 node through the Flowstate ROS bridge

`ros/litert_pose_estimation` is a ROS 2 node with the same pipeline
(`vision_msgs/Detection3DArray` out, `~/estimate` service). In a solution
with the Flowstate ROS bridge (as OMTS), camera images reach ROS through the
camera driver's topics; the node's detections can be fed back into the world
through the bridge. See `ros/litert_pose_estimation/README.md`.

## 3. Python module in any Intrinsic Python service or skill

`python/bindings.cc` builds the module `litert_pose_estimation`
(CMake option `PERCEPTION_BUILD_PYTHON=ON`):

```python
import litert_pose_estimation as lpe

segmenter = lpe.Segmenter("/litert/models", accelerator="auto")
boxes, scores, masks, visibility = segmenter.segment(rgb, 0.6, 0.6)

foundationpose = lpe.FoundationPose("/litert/models", batch_size=128)
rotations, translations, confidences = foundationpose.estimate(
    rgb, depth_m, masks.astype("uint8"), camera_matrix, cad_obj_text,
    iterations=6)
print(segmenter.info(), foundationpose.info())  # where each network runs
```

The signatures and outputs match the service's model calls, so code written
against the ML inference service can switch with a few lines
(`litert_backend.install()` does exactly that for the IOC service).

## Tests

```bash
# Build the module for the host Python, then:
PYTHONPATH=<build dir>:. python3 intrinsic/litert_backend_test.py
```

The backend test installs the backend on a stand-in service object and checks
both calls against the service's recorded inputs and outputs (segmentation:
box within 0.5 px, mask IoU > 0.995; pose: 1 mm, 0.5° modulo the box's
symmetries, score within 0.05).
