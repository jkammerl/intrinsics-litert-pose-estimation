"""Runs segmentation and pose estimation with the TFLite models in LiteRT.

For every test scene: segments the image with rfdetr_seg.tflite, estimates
the pose from the first mask with the FoundationPose TFLite models, and
compares the results with testdata/ (from the ONNX models) and the ground
truth.

Usage: python verify_litert.py [--models=../models] [--testdata=../testdata]
    [--fp16]
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
from ai_edge_litert.interpreter import Interpreter

import foundationpose
import rfdetr


class TfLite:
  """Runs a TFLite model with float32 inputs and outputs."""

  def __init__(self, path):
    self._interpreter = Interpreter(model_path=path, num_threads=4)
    self._interpreter.allocate_tensors()
    self._inputs = {d["name"]: d for d in self._interpreter.get_input_details()}
    self._outputs = {
        d["name"]: d for d in self._interpreter.get_output_details()
    }

  def batch(self):
    return int(next(iter(self._inputs.values()))["shape"][0])

  def __call__(self, **inputs):
    for name, value in inputs.items():
      d = self._inputs[name]
      self._interpreter.set_tensor(d["index"], value.astype(d["dtype"]))
    self._interpreter.invoke()
    return {
        name: self._interpreter.get_tensor(d["index"]).astype(np.float32)
        for name, d in self._outputs.items()
    }


def _refine_fn(model):
  """Runs the refiner in batches of the model's size, padding the last."""
  batch = model.batch()

  def refine(input1, input2):
    translations, rotations = [], []
    for start in range(0, len(input1), batch):
      a, b = input1[start:start + batch], input2[start:start + batch]
      pad = batch - len(a)
      if pad:
        a = np.concatenate([a, np.zeros((pad,) + a.shape[1:], a.dtype)])
        b = np.concatenate([b, np.zeros((pad,) + b.shape[1:], b.dtype)])
      out = model(input1=a, input2=b)
      translations.append(out["output1"][:batch - pad])
      rotations.append(out["output2"][:batch - pad])
    return np.concatenate(translations), np.concatenate(rotations)

  return refine


def _rotation_error_deg(a, b):
  cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
  return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def main():
  here = os.path.dirname(__file__)
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--models", default=os.path.join(here, "..", "models"))
  parser.add_argument("--testdata",
                      default=os.path.join(here, "..", "testdata"))
  parser.add_argument("--fp16", action="store_true",
                      help="Use the *_fp16.tflite models.")
  args = parser.parse_args()
  suffix = "_fp16.tflite" if args.fp16 else ".tflite"
  mesh = foundationpose.load_obj(
      os.path.join(here, "..", "assets", "raw_stock_2x3x5",
                   "raw_stock_2x3x5.obj"))

  scenes_dir = os.path.join(args.testdata, "scenes")
  ok = True
  # Segment all scenes first, then estimate poses, so that only one phase's
  # models are in memory at a time.
  seg = TfLite(os.path.join(args.models, "rfdetr_seg" + suffix))
  segmented = {}
  for name in sorted(os.listdir(scenes_dir)):
    d = os.path.join(scenes_dir, name)
    scene = json.load(open(os.path.join(d, "scene.json")))
    want = json.load(open(os.path.join(d, "segmentation", "result.json")))
    rgb = cv2.imread(os.path.join(d, "rgb.png"))[:, :, ::-1].copy()
    depth = np.load(os.path.join(d, "depth.npy"))
    K = np.array(scene["K"], np.float32)

    start = time.perf_counter()
    pre = rfdetr.preprocess(rgb)
    out = seg(input=pre.input.transpose(0, 2, 3, 1))
    _, scores, masks, _ = rfdetr.postprocess(
        pre, out["logits"], out["visibility_logits"], out["boxes"],
        out["mask_logits"].transpose(0, 3, 1, 2),
        want["confidence_threshold"], want["visibility_threshold"])
    seg_time = time.perf_counter() - start
    segmented[name] = (rgb, depth, K, scene, masks)
    line = f"{name}: {len(scores)} detections ({seg_time:.2f} s)"
    if len(scores) != len(want["detections"]):
      ok = False
      line += f", expected {len(want['detections'])}"
    for i, det in enumerate(want["detections"]):
      if i < len(masks):
        ref = cv2.imread(os.path.join(d, "segmentation", det["mask"]), 0) > 127
        iou = (masks[i] & ref).sum() / (masks[i] | ref).sum()
        line += f", mask IoU {iou:.4f}"
        ok &= iou > 0.99
    print(line)
  del seg

  refine = TfLite(os.path.join(args.models, "foundationpose_refine" + suffix))
  score = TfLite(os.path.join(args.models, "foundationpose_score" + suffix))
  estimator = foundationpose.FoundationPose(
      refine=_refine_fn(refine),
      score=lambda a, b: score(input1=a, input2=b)["output1"],
  )
  for name, (rgb, depth, K, scene, masks) in segmented.items():
    d = os.path.join(scenes_dir, name)
    line = f"{name}:"
    if len(masks) and scene["pose"] is not None:
      start = time.perf_counter()
      pose, best = estimator.estimate(rgb, depth, masks[0], K, mesh)
      fp_time = time.perf_counter() - start
      onnx = json.load(open(os.path.join(d, "foundationpose", "result.json")))
      onnx_pose = np.array(onnx["pose"], np.float32)
      truth = np.array(scene["pose"], np.float32)
      translation = 1000 * np.linalg.norm(pose[:3, 3] - onnx_pose[:3, 3])
      rotation = _rotation_error_deg(pose, onnx_pose)
      a = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
      b = mesh.vertices @ truth[:3, :3].T + truth[:3, 3]
      add_s = 1000 * np.linalg.norm(a[:, None] - b[None], axis=-1).min(1).mean()
      line += (
          f" pose ({fp_time:.1f} s) differs from ONNX by {translation:.2f} mm,"
          f" {rotation:.2f} deg; ADD-S to ground truth {add_s:.2f} mm"
          f" (ONNX: {1000 * onnx['add_s_to_ground_truth_m']:.2f} mm)"
      )
      # Correct by the BOP criterion: ADD-S below 10% of the diameter.
      ok &= add_s < 100 * mesh.diameter
    print(line)
  print("OK" if ok else "FAILED")
  raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
  main()
