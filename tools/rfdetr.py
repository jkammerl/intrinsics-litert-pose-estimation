"""Pre- and post-processing for the RF-DETR segmentation network.

Reproduces what the released segmentation.onnx does around its network (see
extract_rfdetr_core.py), so that the fixed-shape network can run in LiteRT.
cpp/rfdetr.cc implements the same steps in C++.
"""

import dataclasses

import cv2
import numpy as np

SIZE = 504  # Side of the network's square input.
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@dataclasses.dataclass
class Preprocessed:
  input: np.ndarray  # [1, 3, SIZE, SIZE] float32
  scale: float  # SIZE / max(height, width)
  resized_hw: tuple[int, int]  # Image size after scaling, before padding.
  image_hw: tuple[int, int]


def preprocess(rgb: np.ndarray) -> Preprocessed:
  """Scales an RGB uint8 image [H, W, 3] to fit SIZE, normalizes and pads it."""
  h, w = rgb.shape[:2]
  scale = np.float32(SIZE) / np.float32(max(h, w))
  # Truncation, as the model's float -> int64 cast.
  rh, rw = int(np.float32(h) * scale), int(np.float32(w) * scale)
  image = rgb.astype(np.float32) / np.float32(255.0)
  # Bilinear with half-pixel centers and no antialiasing, like ONNX Resize.
  resized = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
  normalized = (resized - MEAN) / STD
  padded = np.zeros((SIZE, SIZE, 3), np.float32)
  padded[:rh, :rw] = normalized
  return Preprocessed(
      input=np.ascontiguousarray(padded.transpose(2, 0, 1)[None]),
      scale=float(scale),
      resized_hw=(rh, rw),
      image_hw=(h, w),
  )


def _sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x))


def postprocess(
    pre: Preprocessed,
    logits: np.ndarray,
    visibility_logits: np.ndarray,
    boxes: np.ndarray,
    mask_logits: np.ndarray,
    confidence_threshold: float,
    visibility_threshold: float,
):
  """Returns (boxes [N, 4] xyxy pixels, scores [N], masks [N, H, W] bool,
  visibility [N]) for the queries above both thresholds, in query order."""
  scores = _sigmoid(logits[0, :, 0]).astype(np.float32)
  visibility = _sigmoid(visibility_logits[0, :, 0]).astype(np.float32)
  keep = (scores > confidence_threshold) & (visibility > visibility_threshold)

  cx, cy, bw, bh = boxes[0].T
  xyxy = np.stack(
      [cx - 0.5 * bw, cy - 0.5 * bh, cx + 0.5 * bw, cy + 0.5 * bh], -1
  )
  xyxy = (xyxy * np.float32(SIZE) / np.float32(pre.scale)).astype(np.float32)

  (rh, rw), (h, w) = pre.resized_hw, pre.image_hw
  masks = []
  for logit in mask_logits[0][keep]:
    full = cv2.resize(logit, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
    image = cv2.resize(
        np.ascontiguousarray(full[:rh, :rw]), (w, h),
        interpolation=cv2.INTER_LINEAR,
    )
    masks.append(image > 0)
  masks = np.array(masks, bool).reshape(-1, h, w)
  return xyxy[keep], scores[keep], masks, visibility[keep]
