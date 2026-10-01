"""Runs Intrinsic's IOC pose estimator service with LiteRT.

The IOC pose estimator service (intrinsic-core's
intrinsic_perception/intrinsic/perception/service/ioc_pose_estimator) handles
the PoseEstimationService API: it loads the capture from the KV store and the
CAD model and parameters from the registered pose estimator asset, and calls
two models through the ML inference service: RF-DETR segmentation and
FoundationPose. `install()` replaces those two calls with the LiteRT pipeline
(the `litert_pose_estimation` Python module), with the same signatures and
outputs, so that everything else in the service stays unchanged:

  from intrinsic_perception...service import ioc_pose_estimator_service
  import litert_backend

  servicer = ioc_pose_estimator_service.IocPoseEstimatorService(config=config)
  litert_backend.install(servicer, models_dir="/models", accelerator="auto")

The service then no longer needs the ML inference service (nor a GPU-only
Triton or ONNX Runtime server): LiteRT runs the networks in-process, on the
GPU through its WebGPU/OpenCL accelerator or on the CPU.
"""

import os
import time
from typing import Any

import numpy as np

import litert_pose_estimation


class LiteRtSegmentationModel:
  """Drop-in for the service's SegmentationModel.run_inference."""

  def __init__(self, segmenter, original=None):
    self._segmenter = segmenter
    self._original = original  # For its _visualize_predictions.
    self.last_duration = 0.0

  def wait_for_model_ready(self, *args, **kwargs):
    del args, kwargs  # The model is compiled when the backend is installed.

  def run_inference(self, image, confidence_threshold=0.6,
                    visibility_threshold=0.6, return_vis=False):
    """Like SegmentationModel.run_inference: image is (1, 3, H, W) uint8."""
    rgb = np.asarray(image)
    if rgb.ndim == 4:
      rgb = rgb[0]
    if rgb.shape[0] == 3 and rgb.shape[-1] != 3:
      rgb = np.transpose(rgb, (1, 2, 0))
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    start = time.perf_counter()
    boxes, scores, masks, visibility = self._segmenter.segment(
        rgb, confidence_threshold, visibility_threshold)
    self.last_duration = time.perf_counter() - start
    if return_vis:
      vis = (self._original._visualize_predictions(  # pylint: disable=protected-access
          rgb, boxes, scores, masks, visibility)
             if self._original is not None else rgb.copy())
      return boxes, scores, masks, visibility, vis
    return boxes, scores, masks, visibility


class LiteRtPoseEstimationModel:
  """Drop-in for the service's PoseEstimationModel.run_inference."""

  def __init__(self, foundationpose, original=None):
    self._foundationpose = foundationpose
    self._original = original  # For its _visualize_predictions.
    self.last_duration = 0.0

  def wait_for_model_ready(self, *args, **kwargs):
    del args, kwargs

  def run_inference(self, rgb, depth, mask, intrinsic_matrix, cad_bytes,
                    num_iterations=3, batch_size=None, return_vis=False):
    """Like PoseEstimationModel.run_inference (FoundationPose's model.py)."""
    del batch_size  # Fixed when the backend is installed (scorer models).
    masks = np.asarray(mask)
    if masks.ndim == 2:
      masks = masks[None]
    masks = np.ascontiguousarray((masks > 0).astype(np.uint8))
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    start = time.perf_counter()
    rotations, translations, confidences = self._foundationpose.estimate(
        rgb, depth, masks, np.asarray(intrinsic_matrix, np.float64),
        bytes(cad_bytes).decode("utf-8", errors="ignore"), int(num_iterations))
    self.last_duration = time.perf_counter() - start
    if return_vis:
      vis = rgb.copy()
      if self._original is not None and hasattr(
          self._original, "_visualize_predictions"):
        try:
          vis = self._original._visualize_predictions(  # pylint: disable=protected-access
              rgb, rotations, translations, confidences, intrinsic_matrix,
              cad_bytes)
        except Exception:  # pylint: disable=broad-except
          vis = rgb.copy()
      return rotations, translations, confidences, vis
    return rotations, translations, confidences


def install(servicer: Any, models_dir: str, accelerator: str = "auto",
            batch_size: int | None = None, gpu_fp16: bool = False):
  """Replaces the servicer's two inference calls with LiteRT.

  Args:
    servicer: An IocPoseEstimatorService.
    models_dir: Directory with rfdetr_seg.tflite and the FoundationPose models
      (foundationpose_refine.tflite, foundationpose_score_b<N>.tflite).
    accelerator: "auto" (GPU, else CPU), "gpu" or "cpu".
    batch_size: FoundationPose chunk size; defaults to the service's
      configured batch size, so that the scores match the service's.
    gpu_fp16: Run on the GPU in float16 (faster, less accurate).

  Returns:
    A dict describing where each network runs.
  """
  if batch_size is None:
    batch_size = getattr(servicer, "_batch_size", 128)
  # LiteRT's GPU accelerator library is installed next to the module (see
  # Dockerfile), or found in the build tree.
  library_dir = os.path.dirname(os.path.abspath(litert_pose_estimation.__file__))
  if not os.path.exists(
      os.path.join(library_dir, "libLiteRtWebGpuAccelerator.so")):
    library_dir = litert_pose_estimation.DEFAULT_LITERT_LIBRARY_DIR
  segmenter = litert_pose_estimation.Segmenter(
      models_dir, accelerator=accelerator, gpu_fp16=gpu_fp16,
      litert_library_dir=library_dir)
  foundationpose = litert_pose_estimation.FoundationPose(
      models_dir, batch_size=batch_size, accelerator=accelerator,
      gpu_fp16=gpu_fp16, litert_library_dir=library_dir)
  servicer.segmentation_model = LiteRtSegmentationModel(
      segmenter, getattr(servicer, "segmentation_model", None))
  servicer.pose_estimator_model = LiteRtPoseEstimationModel(
      foundationpose, getattr(servicer, "pose_estimator_model", None))
  return {"segmentation": segmenter.info(),
          "foundationpose": foundationpose.info()}
