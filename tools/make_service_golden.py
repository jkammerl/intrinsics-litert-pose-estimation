"""Builds the golden test case from a capture of the IOC pose estimator service.

The capture (`golden.npz`) is recorded inside the deployed
`rs-pose-estimator-service` pod by running the service's own code on a camera
capture, with the inputs and outputs of its two model calls recorded (see
`docs/service_golden.md`). This script

1. re-runs OMTS's FoundationPose model (`model.py`, the Triton Python backend
   that the service calls) on the recorded inputs, on the CPU with ONNX
   Runtime, and checks that it reproduces the service's pose and score;
2. records that run's intermediate results, so that the C++ port can be
   checked stage by stage;
3. writes the test case to --out.

Usage:
  python make_service_golden.py --capture=golden.npz \
      --omts_foundationpose_dir=~/intrinsic-omts/third_party/foundationpose \
      --onnx_dir=<dir with foundationpose_refine.onnx, foundationpose_score.onnx> \
      --out=../testdata/service_golden
"""

import argparse
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import types

import cv2
import numpy as np


def _fake_pb_utils():
  """A minimal triton_python_backend_utils module for running model.py."""
  pb_utils = types.ModuleType("triton_python_backend_utils")

  class Tensor:

    def __init__(self, name, array):
      self._name, self._array = name, np.asarray(array)

    def name(self):
      return self._name

    def as_numpy(self):
      return self._array

  class InferenceResponse:

    def __init__(self, output_tensors=(), error=None):
      self.output_tensors = {t.name(): t.as_numpy() for t in output_tensors}
      self.error = error

  pb_utils.Tensor = Tensor
  pb_utils.InferenceResponse = InferenceResponse
  pb_utils.TritonError = Exception
  pb_utils.get_input_tensor_by_name = (
      lambda request, name: Tensor(name, request[name])
      if name in request else None)
  return pb_utils


def _load_model(model_dir):
  sys.modules["triton_python_backend_utils"] = _fake_pb_utils()
  sys.path.insert(0, model_dir)
  os.environ["INTRINSIC_INFERENCE_DEVICE"] = "cpu"
  spec = importlib.util.spec_from_file_location(
      "foundationpose_model", os.path.join(model_dir, "model.py"))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  model = module.TritonPythonModel()
  model.initialize({"model_config": json.dumps({})})
  return model


def _record(model, rec):
  """Wraps the model's steps so that their results are recorded in `rec`."""
  sample = model._sample_initial_poses
  def sample_wrap(*args, **kwargs):
    poses = sample(*args, **kwargs)
    rec["candidate_poses"] = np.array(poses, np.float32)
    return poses
  model._sample_initial_poses = sample_wrap

  prepare = model._prepare_model_inputs
  def prepare_wrap(chunk_poses, *args, **kwargs):
    in1, in2 = prepare(chunk_poses, *args, **kwargs)
    if "first_in1" not in rec:  # First chunk, first iteration.
      rec["first_poses"] = np.array(chunk_poses[:4], np.float32)
      rec["first_in1"] = np.array(in1[:4], np.float32)
      rec["first_in2"] = np.array(in2[:4], np.float32)
    return in1, in2
  model._prepare_model_inputs = prepare_wrap

  refine_run = model.refine_session.run
  class RefineSession:
    def run(self, names, feeds):
      outs = refine_run(names, feeds)
      rec.setdefault("refine_translation", []).append(np.array(outs[0]))
      rec.setdefault("refine_rotation", []).append(np.array(outs[1]))
      return outs
    def get_providers(self):
      return ["CPUExecutionProvider"]
  model.refine_session = RefineSession()

  score_run = model.score_session.run
  class ScoreSession:
    def run(self, names, feeds):
      outs = score_run(names, feeds)
      rec.setdefault("chunk_scores", []).append(np.array(outs[0]).reshape(-1))
      return outs
  model.score_session = ScoreSession()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  for flag in ("capture", "omts_foundationpose_dir", "onnx_dir", "out"):
    parser.add_argument("--" + flag, required=True)
  args = parser.parse_args()
  cap = np.load(os.path.expanduser(args.capture))

  with tempfile.TemporaryDirectory() as model_dir:
    src = os.path.expanduser(args.omts_foundationpose_dir)
    for name in ("model.py", "foundationpose_numpy.py"):
      shutil.copy(os.path.join(src, name), model_dir)
    for name in ("foundationpose_refine.onnx", "foundationpose_score.onnx"):
      os.symlink(os.path.join(os.path.expanduser(args.onnx_dir), name),
                 os.path.join(model_dir, name))
    model = _load_model(model_dir)
    rec = {}
    _record(model, rec)
    request = {
        "RGB": cap["fp_in_rgb"],
        "DEPTH": cap["fp_in_depth"],
        "MASK": cap["fp_in_mask"],
        "CAM_K": cap["fp_in_intrinsic_matrix"].astype(np.float32),
        "CAD_MODEL_BYTES": cap["cad_obj_bytes"],
        "NUM_ITERATIONS": np.array([int(cap["fp_in_num_iterations"])],
                                   np.int32),
        "BATCH_SIZE": np.array([int(cap["fp_in_batch_size"])], np.int32),
    }
    (response,) = model.execute([request])
    if response.error is not None:
      raise RuntimeError(response.error)
    out = response.output_tensors

  for name, key in (("rotation", "ROTATION"), ("translation", "TRANSLATION"),
                    ("confidence", "CONFIDENCE")):
    np.testing.assert_allclose(out[key], cap["fp_" + name], rtol=0, atol=1e-6,
                               err_msg=f"{name} differs from the service's")
  print("OMTS's FoundationPose model reproduces the service's result.")

  os.makedirs(args.out, exist_ok=True)
  rgb = cap["fp_in_rgb"]
  cv2.imwrite(os.path.join(args.out, "rgb.png"),
              cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
  np.save(os.path.join(args.out, "depth.npy"),
          cap["fp_in_depth"].astype(np.float32))
  with open(os.path.join(args.out, "raw_stock_2x3x5.obj"), "wb") as f:
    f.write(cap["cad_obj_bytes"].tobytes())
  for i, mask in enumerate(cap["seg_masks"]):
    cv2.imwrite(os.path.join(args.out, f"mask_{i}.png"),
                mask.astype(np.uint8) * 255)
  inter = os.path.join(args.out, "foundationpose")
  os.makedirs(inter, exist_ok=True)
  for key in ("candidate_poses", "first_poses", "first_in1", "first_in2"):
    np.save(os.path.join(inter, key + ".npy"), rec[key])
  for key in ("refine_translation", "refine_rotation", "chunk_scores"):
    for i, v in enumerate(rec[key]):
      np.save(os.path.join(inter, f"{key}_{i}.npy"), v.astype(np.float32))

  k = cap["fp_in_intrinsic_matrix"]
  case = {
      "description": (
          "Inputs and outputs of the IOC pose estimator service "
          "(rs-pose-estimator-service) for one capture of the simulated "
          "Orbbec Gemini 335Le in OMTS's Lab BB-01 cell."),
      "image": {"rgb": "rgb.png", "depth": "depth.npy",
                "depth_units": "meters; non-finite where nothing is hit",
                "height": int(rgb.shape[0]), "width": int(rgb.shape[1])},
      "camera_matrix": [[float(v) for v in row] for row in k],
      "cad_obj": "raw_stock_2x3x5.obj",
      "config": {
          "confidence_threshold": float(cap["seg_thresholds"][0]),
          "visibility_threshold": float(cap["seg_thresholds"][1]),
          "refinement_iterations": int(cap["fp_in_num_iterations"]),
          "batch_size": int(cap["fp_in_batch_size"]),
      },
      "segmentation": [
          {"box_xyxy": [float(v) for v in cap["seg_boxes"][i]],
           "score": float(cap["seg_scores"][i]),
           "visibility": float(cap["seg_visibility"][i]),
           "mask": f"mask_{i}.png",
           "mask_pixels": int(cap["seg_masks"][i].sum())}
          for i in range(len(cap["seg_boxes"]))
      ],
      "poses": [
          {"rotation": [[float(v) for v in row] for row in cap["fp_rotation"][i]],
           "translation": [float(v) for v in cap["fp_translation"][i]],
           "score": float(cap["fp_confidence"][i][0])}
          for i in range(len(cap["fp_rotation"]))
      ],
      "foundationpose_intermediates": {
          "num_candidates": int(len(rec["candidate_poses"])),
          "refine_calls": len(rec["refine_translation"]),
          "score_calls": len(rec["chunk_scores"]),
      },
  }
  with open(os.path.join(args.out, "case.json"), "w") as f:
    json.dump(case, f, indent=2)
  print("Wrote", args.out)


if __name__ == "__main__":
  main()
