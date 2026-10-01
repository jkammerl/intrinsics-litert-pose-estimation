"""Converts the segmentation and pose estimation models to TFLite.

Usage:
  python convert.py --segmentation_onnx=segmentation.onnx \
      --refine_onnx=foundationpose_refine.onnx \
      --score_onnx=foundationpose_score.onnx --out=../models

Writes to --out:
  rfdetr_core.onnx               The fixed-shape RF-DETR network (reference).
  rfdetr_seg.tflite              [1, 504, 504, 3] -> logits [1, 200, 2],
                                 visibility_logits [1, 200, 1],
                                 boxes [1, 200, 4],
                                 mask_logits [1, 126, 126, 200]
  foundationpose_refine.tflite   input1, input2 [40, 160, 160, 6] ->
                                 output1 (translation), output2 (rotation)
                                 [40, 3]
  foundationpose_score.tflite    input1, input2 [280, 160, 160, 6] ->
                                 output1 [1, 280]
  foundationpose_score_b<N>.tflite  The scorer for batches of N candidates,
                                 for each N in --score_batches other than
                                 280 (e.g. 128 and 24 to score in the chunks
                                 the IOC pose estimator service uses).
and float16 variants (*_fp16.tflite) with float16 weights, e.g. for GPUs.
The FoundationPose models get gpu_compat.py's rewrites of ops that GPUs
compute wrongly.

The FoundationPose networks need a fixed batch size (onnx2tf can't convert
them with a dynamic batch). The refiner treats candidates independently, so
it runs on chunks of 40 of the 280 pose candidates, which needs far less
memory. The scorer compares all candidates against each other, so it takes
all 280.

Requires onnx2tf (see requirements.txt). It rewrites its input model, so the
inputs are copied first.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import extract_rfdetr_core
import gpu_compat


def _onnx2tf(onnx_path, out_dir, extra_args):
  subprocess.run(
      ["onnx2tf", "-i", onnx_path, "-o", out_dir, *extra_args], check=True
  )
  name = os.path.splitext(os.path.basename(onnx_path))[0]
  return (
      os.path.join(out_dir, f"{name}_float32.tflite"),
      os.path.join(out_dir, f"{name}_float16.tflite"),
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--segmentation_onnx",
                      help="Required unless --only_foundationpose.")
  for flag in ("refine_onnx", "score_onnx", "out"):
    parser.add_argument("--" + flag, required=True)
  parser.add_argument(
      "--score_batches", default="280",
      help="Comma-separated scorer batch sizes to convert.")
  parser.add_argument(
      "--only_scorer", action="store_true",
      help="Only convert the scorer (for adding batch sizes).")
  parser.add_argument(
      "--only_foundationpose", action="store_true",
      help="Only convert the FoundationPose refiner and scorers (see "
      "fetch_foundationpose.sh).")
  args = parser.parse_args()
  if not (args.only_scorer or args.only_foundationpose or
          args.segmentation_onnx):
    parser.error("--segmentation_onnx is required")
  os.makedirs(args.out, exist_ok=True)
  # onnx2tf runs onnxsim from PATH.
  os.environ["PATH"] = (
      os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
  )

  with tempfile.TemporaryDirectory() as tmp:
    core = os.path.join(args.out, "rfdetr_core.onnx")
    convert_segmentation = not (args.only_scorer or args.only_foundationpose)
    if convert_segmentation:
      extract_rfdetr_core.main(args.segmentation_onnx, core)
    def foundationpose_args(batch):
      # Keep the NHWC inputs as they are, with a fixed batch.
      return ["-kat", "input1", "input2", "-ois",
              f"input1:{batch},160,160,6", f"input2:{batch},160,160,6"]

    jobs = [
        (core, "rfdetr_seg", []),
        (args.refine_onnx, "foundationpose_refine", foundationpose_args(40)),
    ]
    if args.only_scorer:
      jobs = []
    elif not convert_segmentation:
      jobs = jobs[1:]
    for batch in (int(b) for b in args.score_batches.split(",")):
      name = "foundationpose_score" + ("" if batch == 280 else f"_b{batch}")
      jobs.append((args.score_onnx, name, foundationpose_args(batch)))
    for src, name, extra in jobs:
      copy = os.path.join(tmp, name + ".onnx")
      shutil.copyfile(src, copy)
      f32, f16 = _onnx2tf(copy, os.path.join(tmp, name), extra)
      for tflite, suffix in ((f32, ""), (f16, "_fp16")):
        out = os.path.join(args.out, name + suffix + ".tflite")
        shutil.copyfile(tflite, out)
        if name.startswith("foundationpose"):
          gpu_compat.rewrite(out)  # Ops that GPUs compute wrongly.


if __name__ == "__main__":
  main()
