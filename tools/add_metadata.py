"""Documents the TFLite models' input and output tensors.

Embeds each model's spec in the .tflite file itself:
  * the model's `description` field: a one-paragraph summary, and
  * a metadata entry named "io_spec": JSON with every input and output
    tensor's name, shape, dtype, layout and content.
and writes the same information to models/README.md.

Usage: python add_metadata.py --models=../models
"""

import argparse
import json
import os

import flatbuffers
from tensorflow.lite.python import schema_py_generated as schema

METADATA_NAME = "io_spec"

_FP_CROP = (
    "160x160 crop around the candidate: the square window (in image"
    " pixels) that contains the object's bounding sphere, enlarged by 1.2"
    " (compute_crop_window_tf in tools/foundationpose_numpy.py)."
)
_FP_POINTS = (
    "3D points in the camera frame, in meters, minus the candidate pose's"
    " translation, divided by the mesh radius (diameter / 2; the diameter"
    " is the length of the mesh's bounding box diagonal)."
)

def _fp_inputs(batch):
  return [
    {
        "name": "input1",
        "shape": [batch, 160, 160, 6],
        "dtype": "float32",
        "layout": "NHWC, one row per pose candidate",
        "content": (
            "Rendering of the mesh at each candidate pose, in the "
            + _FP_CROP
            + " Channels 0-2: 0.5 where the object is rendered, else 0."
            " Channels 3-5: the rendered surface's " + _FP_POINTS
            + " 0 where nothing is rendered. Rows are top to bottom."
        ),
    },
    {
        "name": "input2",
        "shape": [batch, 160, 160, 6],
        "dtype": "float32",
        "layout": "NHWC, one row per pose candidate",
        "content": (
            "The observed RGB-D image, warped into the same crop. Channels"
            " 0-2: RGB in [0, 1], bilinear. Channels 3-5: the observed "
            + _FP_POINTS
            + " Nearest-neighbor; 0 where depth is 0.05 m or less."
            " The point map is back-projected from depth with the camera"
            " intrinsics: x = (u - cx) * z / fx, y = (v - cy) * z / fy."
        ),
    },
  ]

SPECS = {
    "rfdetr_seg.tflite": {
        "summary": (
            "RF-DETR instance segmentation of the raw stock workpiece (the"
            " network of OMTS's segmentation.onnx, without its pre- and"
            " post-processing). 200 object queries; each has a class score,"
            " a visibility score, a box and a mask."
        ),
        "source": "segmentation.onnx from the intrinsic-omts 20260922.0 release",
        "inputs": [{
            "name": "input",
            "shape": [1, 504, 504, 3],
            "dtype": "float32",
            "layout": "NHWC, RGB",
            "content": (
                "The image, scaled so that its longer side is 504 pixels"
                " (scale = 504 / max(H, W); the scaled size is truncated to"
                " integers), bilinear with half-pixel centers and no"
                " antialiasing, as values / 255, normalized per channel with"
                " (x - mean) / std, mean = [0.485, 0.456, 0.406], std ="
                " [0.229, 0.224, 0.225]. Placed in the top-left corner; the"
                " rest is 0."
            ),
        }],
        "outputs": [
            {
                "name": "logits",
                "shape": [1, 200, 2],
                "dtype": "float32",
                "layout": "[batch, query, class]",
                "content": (
                    "Class logits per query. The detection score is"
                    " sigmoid(logits[0, q, 0]); class 1 isn't used."
                ),
            },
            {
                "name": "visibility_logits",
                "shape": [1, 200, 1],
                "dtype": "float32",
                "layout": "[batch, query, 1]",
                "content": (
                    "Visibility logit per query: sigmoid() is the estimated"
                    " visible (unoccluded) fraction of the object."
                ),
            },
            {
                "name": "boxes",
                "shape": [1, 200, 4],
                "dtype": "float32",
                "layout": "[batch, query, (cx, cy, w, h)]",
                "content": (
                    "Box center and size relative to the 504x504 input, in"
                    " [0, 1]. In image pixels: x1 = (cx - w / 2) * 504 /"
                    " scale, y1 = (cy - h / 2) * 504 / scale, x2 = (cx + w /"
                    " 2) * 504 / scale, y2 = (cy + h / 2) * 504 / scale."
                ),
            },
            {
                "name": "mask_logits",
                "shape": [1, 126, 126, 200],
                "dtype": "float32",
                "layout": "NHWC: [batch, y, x, query]",
                "content": (
                    "Mask logits per query over the 504x504 input, at 1/4"
                    " resolution. To get a mask in the image: resize to"
                    " 504x504 (bilinear, half-pixel centers), crop the scaled"
                    " image's region [0, round_down(H * scale)) x [0,"
                    " round_down(W * scale)), resize to H x W (bilinear), and"
                    " threshold at > 0."
                ),
            },
        ],
        "postprocessing": (
            "OMTS keeps the queries with score > 0.6 and visibility > 0.6, in"
            " query order. See tools/rfdetr.py."
        ),
    },
    "foundationpose_refine.tflite": {
        "summary": (
            "FoundationPose pose refiner. Predicts a pose update for each of"
            " 40 pose candidates by comparing a rendering of the object's"
            " mesh at the candidate pose with the observed RGB-D crop."
            " Candidates are independent of each other: the 280 candidates"
            " run in 7 batches of 40, and a partial batch can be padded."
        ),
        "source": (
            "NVIDIA FoundationPose refine_model.onnx 1.0.0 (NGC"
            " nvidia/isaac/foundationpose), NVIDIA Deep Learning Models"
            " License Agreement (not redistributable stand-alone)"
        ),
        "inputs": _fp_inputs(40),
        "outputs": [
            {
                "name": "output1",
                "shape": [40, 3],
                "dtype": "float32",
                "layout": "[candidate, xyz]",
                "content": (
                    "Translation update in the camera frame, in units of the"
                    " mesh radius: t_new = t + output1 * diameter / 2."
                ),
            },
            {
                "name": "output2",
                "shape": [40, 3],
                "dtype": "float32",
                "layout": "[candidate, xyz]",
                "content": (
                    "Rotation update: the rotation vector w = tanh(output2) *"
                    " 0.34906585 (at most 20 degrees per axis), applied as"
                    " R_new = Rodrigues(w)^T * R."
                ),
            },
        ],
        "postprocessing": (
            "Apply the updates, re-render and repeat (5 iterations by"
            " default). See update_refined_poses in"
            " tools/foundationpose_numpy.py."
        ),
    },
    "foundationpose_score.tflite": {
        "summary": (
            "FoundationPose pose scorer. Ranks 280 refined pose candidates"
            " against each other: the scores depend on the whole batch, so"
            " all 280 candidates must be scored together."
        ),
        "source": (
            "NVIDIA FoundationPose score_model.onnx 1.0.0 (NGC"
            " nvidia/isaac/foundationpose), NVIDIA Deep Learning Models"
            " License Agreement (not redistributable stand-alone)"
        ),
        "inputs": _fp_inputs(280),
        "outputs": [{
            "name": "output1",
            "shape": [1, 280],
            "dtype": "float32",
            "layout": "[1, candidate]",
            "content": (
                "Unnormalized score per candidate; higher is better. Only"
                " comparable within one batch."
            ),
        }],
        "postprocessing": (
            "The best candidate's pose, multiplied on the right by a"
            " translation to the mesh's bounding box center"
            " (apply_mesh_center_offset), is the estimate."
        ),
    },
}


def _spec_for(filename):
  """The spec for `filename`; *_fp16.tflite models compute in float16."""
  base = filename.replace("_fp16.tflite", ".tflite")
  spec = json.loads(json.dumps(SPECS[base]))  # Deep copy.
  spec["model"] = base
  fp16 = filename.endswith("_fp16.tflite")
  spec["weights"] = "float16" if fp16 else "float32"
  if fp16:
    for tensor in spec["inputs"] + spec["outputs"]:
      tensor["dtype"] = "float16"
  return spec


def embed(path, spec):
  """Sets the model's description and "io_spec" metadata, replacing old ones."""
  with open(path, "rb") as f:
    model = schema.ModelT.InitFromObj(
        schema.Model.GetRootAsModel(bytearray(f.read()), 0)
    )
  model.description = spec["summary"].encode()
  data = json.dumps(spec, indent=2).encode()
  model.metadata = [
      m for m in (model.metadata or []) if m.name.decode() != METADATA_NAME
  ]
  buffer = schema.BufferT()
  buffer.data = list(data)
  model.buffers.append(buffer)
  metadata = schema.MetadataT()
  metadata.name = METADATA_NAME.encode()
  metadata.buffer = len(model.buffers) - 1
  model.metadata.append(metadata)
  builder = flatbuffers.Builder(1024)
  builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
  with open(path, "wb") as f:
    f.write(builder.Output())


def read(path):
  """Returns a model's "io_spec" metadata as a dict."""
  with open(path, "rb") as f:
    model = schema.Model.GetRootAsModel(bytearray(f.read()), 0)
  for i in range(model.MetadataLength()):
    metadata = model.Metadata(i)
    if metadata.Name().decode() == METADATA_NAME:
      data = model.Buffers(metadata.Buffer()).DataAsNumpy().tobytes()
      return json.loads(data)
  return None


def _markdown():
  lines = [
      "# Models",
      "",
      "Each .tflite file documents its tensors itself: the model's",
      "`description` field holds a summary, and the metadata entry",
      f"`{METADATA_NAME}` holds this spec as JSON. `*_fp16.tflite` files",
      "compute in float16: their weights, inputs and outputs are float16,",
      "with the same shapes and contents. They are half the size and suit",
      "GPUs; on CPUs, the float32 models are more accurate.",
      "",
      "`rfdetr_core.onnx` is the ONNX network that `rfdetr_seg.tflite` was",
      "converted from, for reference.",
      "",
      "The FoundationPose models (`foundationpose_*.tflite`) aren't in the",
      "repository: NVIDIA's license doesn't allow distributing them",
      "stand-alone. `tools/fetch_foundationpose.sh --accept-nvidia-license`",
      "downloads them from NGC and converts them.",
      "",
      "_Generated by `tools/add_metadata.py`; edit the specs there._",
  ]
  for name, spec in SPECS.items():
    lines += ["", f"## {name}", "", spec["summary"], "",
              f"Source: {spec['source']}.", "",
              "| Tensor | Shape | Type | Layout | Content |",
              "| :--- | :--- | :--- | :--- | :--- |"]
    for kind in ("inputs", "outputs"):
      for t in spec[kind]:
        lines.append(
            f"| {kind[:-1]} `{t['name']}` | `{t['shape']}` | {t['dtype']} |"
            f" {t['layout']} | {t['content']} |"
        )
    lines += ["", f"Post-processing: {spec['postprocessing']}"]
  return "\n".join(lines) + "\n"


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--models", required=True)
  args = parser.parse_args()
  for filename in sorted(os.listdir(args.models)):
    if filename.endswith(".tflite"):
      if filename.replace("_fp16.tflite", ".tflite") not in SPECS:
        # E.g. the scorers for other batch sizes (foundationpose_score_b<N>).
        print("No spec for", filename)
        continue
      path = os.path.join(args.models, filename)
      spec = _spec_for(filename)
      embed(path, spec)
      assert read(path) == spec
      print("Documented", filename)
  with open(os.path.join(args.models, "README.md"), "w") as f:
    f.write(_markdown())


if __name__ == "__main__":
  main()
