"""Generates the test data in testdata/ with the original ONNX models.

Usage:
  python make_testdata.py --segmentation_onnx=segmentation.onnx \
      --rfdetr_core_onnx=rfdetr_core.onnx \
      --refine_onnx=foundationpose_refine.onnx \
      --score_onnx=foundationpose_score.onnx --out=../testdata

Reference outputs come from ONNX Runtime running the original models, so the
TFLite models (and ports of the pre- and post-processing) can be tested
against them. See the README for the file layout.
"""

import argparse
import json
import os

import cv2
import numpy as np
import onnxruntime as ort

import foundationpose
import render
import rfdetr

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
_MESH = os.path.join(_ASSETS, "raw_stock_2x3x5", "raw_stock_2x3x5.obj")

HEIGHT, WIDTH = 480, 640
K = np.array([[610.0, 0, 320.0], [0, 610.0, 240.0], [0, 0, 1]], np.float32)
THRESHOLDS = (0.6, 0.6)  # Confidence and visibility, as used by OMTS.
ITERATIONS = 5
# Rows of the refiner's batch saved as network test data. The refiner treats
# samples independently, so a subset (padded to the batch) is enough.
REFINE_ROWS = 16

# name: (rotation axis, angle in degrees, translation in meters) of the
# object in the camera frame, or None for a scene without the object.
SCENES = {
    "raw_stock_top": ([1, 0, 0], 140, [0.02, 0.01, 0.6]),
    "raw_stock_tilted": ([1, 0.3, 0.2], 125, [-0.05, 0.04, 0.55]),
    "raw_stock_side": ([0.2, 1, 0.1], 70, [0.06, -0.03, 0.5]),
    "empty": None,
}


def _session(path):
  options = ort.SessionOptions()
  options.enable_cpu_mem_arena = False  # Halves the peak memory at batch 280.
  return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])


def _refine_in_chunks(session, input1, input2, chunk=40):
  """Runs the refiner in chunks to save memory; samples are independent."""
  outputs = [
      session.run(None, {"input1": input1[i:i + chunk],
                         "input2": input2[i:i + chunk]})
      for i in range(0, len(input1), chunk)
  ]
  return tuple(np.concatenate(o) for o in zip(*outputs))


def _save(directory, name, array):
  os.makedirs(directory, exist_ok=True)
  np.save(os.path.join(directory, name + ".npy"), np.ascontiguousarray(array))


def _add_s(mesh, pose_a, pose_b):
  """Symmetric average distance between two poses of the mesh's vertices."""
  a = mesh.vertices @ pose_a[:3, :3].T + pose_a[:3, 3]
  b = mesh.vertices @ pose_b[:3, :3].T + pose_b[:3, 3]
  return float(np.linalg.norm(a[:, None] - b[None], axis=-1).min(1).mean())


def _segment(scene_dir, rgb, full, core):
  """Saves the segmentation network's I/O and the full model's results."""
  out = os.path.join(scene_dir, "segmentation")
  pre = rfdetr.preprocess(rgb)
  names = [o.name for o in core.get_outputs()]
  raw = dict(zip(names, core.run(None, {"input": pre.input})))
  # The TFLite model is NHWC: [1, 504, 504, 3] in, [1, 126, 126, 200] masks.
  _save(out, "input", pre.input.transpose(0, 2, 3, 1))
  _save(out, "logits", raw["logits"])
  _save(out, "visibility_logits", raw["visibility_logits"])
  _save(out, "boxes", raw["boxes"])
  _save(out, "mask_logits", raw["mask_logits"].transpose(0, 2, 3, 1))

  boxes, scores, masks, visibility = full.run(
      None,
      {
          "input": np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]),
          "thresholds": np.array([THRESHOLDS], np.float32),
      },
  )
  for i, mask in enumerate(masks):
    cv2.imwrite(os.path.join(out, f"mask_{i}.png"), mask.astype(np.uint8) * 255)
  result = {
      "confidence_threshold": THRESHOLDS[0],
      "visibility_threshold": THRESHOLDS[1],
      "scale": pre.scale,
      "resized_hw": pre.resized_hw,
      "detections": [
          {"box_xyxy": b.tolist(), "score": float(s), "visibility": float(v),
           "mask": f"mask_{i}.png"}
          for i, (b, s, v) in enumerate(zip(boxes, scores, visibility))
      ],
  }
  with open(os.path.join(out, "result.json"), "w") as f:
    json.dump(result, f, indent=2)
  return masks


def _save_network_data(out_dir, trace, refine, score):
  """Saves the first refinement's and the scoring's network I/O."""
  # Inputs are stored as float16 to keep the files small. The reference
  # outputs are computed from the rounded inputs, so they match exactly.
  r1 = trace["refine_input1"][0][:REFINE_ROWS].astype(np.float16)
  r2 = trace["refine_input2"][0][:REFINE_ROWS].astype(np.float16)
  t, r = refine.run(None, {"input1": r1.astype(np.float32),
                           "input2": r2.astype(np.float32)})
  _save(out_dir, "refine_input1", r1)
  _save(out_dir, "refine_input2", r2)
  _save(out_dir, "refine_translation", t)
  _save(out_dir, "refine_rotation", r)
  s1 = trace["score_input1"][0].astype(np.float16)
  s2 = trace["score_input2"][0].astype(np.float16)
  (s,) = score.run(None, {"input1": s1.astype(np.float32),
                          "input2": s2.astype(np.float32)})
  _save(out_dir, "score_input1", s1)
  _save(out_dir, "score_input2", s2)
  _save(out_dir, "scores", s)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for flag in ("segmentation_onnx", "rfdetr_core_onnx", "refine_onnx",
               "score_onnx", "out"):
    parser.add_argument("--" + flag, required=True)
  args = parser.parse_args()

  full, core = _session(args.segmentation_onnx), _session(args.rfdetr_core_onnx)
  refine, score = _session(args.refine_onnx), _session(args.score_onnx)
  estimator = foundationpose.FoundationPose(
      refine=lambda a, b: _refine_in_chunks(refine, a, b),
      score=lambda a, b: score.run(None, {"input1": a, "input2": b})[0],
  )
  mesh = foundationpose.load_obj(_MESH)

  for index, (name, placement) in enumerate(SCENES.items()):
    print("Scene", name)
    scene_dir = os.path.join(args.out, "scenes", name)
    os.makedirs(scene_dir, exist_ok=True)
    pose = None
    if placement is None:
      rng = np.random.default_rng(index)
      ys, xs = np.indices((HEIGHT, WIDTH))
      rgb = (90 + 12 * np.sin(xs / 7.0) * np.sin(ys / 11.0))[..., None]
      rgb = np.clip(rgb + rng.normal(0, 2.0, (HEIGHT, WIDTH, 3)), 0, 255)
      rgb = rgb.astype(np.uint8)
      depth = np.full((HEIGHT, WIDTH), 0.6, np.float32)
      mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    else:
      axis, angle, translation = placement
      pose = np.eye(4, dtype=np.float32)
      pose[:3, :3] = render.rotation(axis, angle)
      pose[:3, 3] = translation
      rgb, depth, mask = render.render_scene(
          mesh, pose, K, HEIGHT, WIDTH, seed=index
      )
    cv2.imwrite(os.path.join(scene_dir, "rgb.png"), rgb[:, :, ::-1])
    _save(scene_dir, "depth", depth)
    cv2.imwrite(os.path.join(scene_dir, "mask.png"), mask * 255)

    masks = _segment(scene_dir, rgb, full, core)
    scene = {
        "K": K.tolist(),
        "height": HEIGHT,
        "width": WIDTH,
        "depth_unit": "meters",
        "mesh": "assets/raw_stock_2x3x5/raw_stock_2x3x5.obj",
        # Object in camera frame; the mesh's origin, as in the OBJ.
        "pose": None if pose is None else pose.tolist(),
    }
    with open(os.path.join(scene_dir, "scene.json"), "w") as f:
      json.dump(scene, f, indent=2)
    if not len(masks):
      continue

    # Pose estimation from the first detection's mask, as OMTS does.
    trace = {}
    estimate, best_score = estimator.estimate(
        rgb, depth, masks[0], K, mesh, iterations=ITERATIONS, trace=trace
    )
    result = {
        "iterations": ITERATIONS,
        "mask": "segmentation/mask_0.png",
        "pose": estimate.tolist(),
        "score": best_score,
        "add_s_to_ground_truth_m": _add_s(mesh, estimate, pose),
        "mesh_diameter_m": mesh.diameter,
    }
    print("  ADD-S %.2f mm" % (1000 * result["add_s_to_ground_truth_m"]))
    fp_dir = os.path.join(scene_dir, "foundationpose")
    os.makedirs(fp_dir, exist_ok=True)
    with open(os.path.join(fp_dir, "result.json"), "w") as f:
      json.dump(result, f, indent=2)
    _save(fp_dir, "initial_poses", trace["initial_poses"][0])
    _save(fp_dir, "refined_poses", trace["refined_poses"][0])
    if index == 0:
      _save_network_data(os.path.join(args.out, "networks"), trace, refine,
                         score)


if __name__ == "__main__":
  main()
