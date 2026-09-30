"""Extracts the fixed-shape network from the RF-DETR segmentation model.

The released segmentation.onnx takes a uint8 image of any size and does its
own pre-processing (resize, normalization, padding) and post-processing
(thresholds, box scaling, mask resizing), which needs dynamic shapes. This
cuts out the network between those steps:

  input:  [1, 3, 504, 504] float32, normalized and padded (see rfdetr.py)
  logits: [1, 200, 2] class logits per query
  visibility_logits: [1, 200, 1]
  boxes:  [1, 200, 4] normalized (cx, cy, w, h) in the padded image
  mask_logits: [1, 200, 126, 126]

Usage: python extract_rfdetr_core.py segmentation.onnx rfdetr_core.onnx
"""

import sys

import onnx
from onnx import utils

_INPUT = "/Pad_output_0"
_OUTPUTS = {
    "/model/class_embed/Add_output_0": "logits",
    "/model/visibility_embed/Add_output_0": "visibility_logits",
    "/model/Concat_output_0": "boxes",
    "/model/segmentation_head/Add_output_0": "mask_logits",
}


def main(src, dst):
  model = onnx.shape_inference.infer_shapes(onnx.load(src))
  extractor = utils.Extractor(model)
  core = extractor.extract_model([_INPUT], list(_OUTPUTS))
  renames = {_INPUT: "input", **_OUTPUTS}
  for node in core.graph.node:
    node.input[:] = [renames.get(i, i) for i in node.input]
    node.output[:] = [renames.get(o, o) for o in node.output]
  for value in list(core.graph.input) + list(core.graph.output):
    value.name = renames.get(value.name, value.name)
  # Fix the input shape so that the converter sees static shapes.
  dims = core.graph.input[0].type.tensor_type.shape.dim
  for dim, size in zip(dims, (1, 3, 504, 504)):
    dim.ClearField("dim_param")
    dim.dim_value = size
  core = onnx.shape_inference.infer_shapes(core)
  onnx.checker.check_model(core)
  onnx.save(core, dst)
  print("ops:", sorted({n.op_type for n in core.graph.node}))


if __name__ == "__main__":
  main(*sys.argv[1:])
