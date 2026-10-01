"""Web demo server: 6D pose estimation of the raw stock with LiteRT.

Serves the browser app in web/ and runs the LiteRT pipeline (RF-DETR
segmentation and FoundationPose, the same as the IOC pose estimator service)
on the RGB-D frames the browser renders:

  PYTHONPATH=<dir with litert_pose_estimation*.so> python3 demo/server.py
  # then open http://localhost:8765

POST /api/estimate with JSON:
  {"width": W, "height": H,
   "rgb": base64 of W*H*3 bytes (RGB, row-major, top row first),
   "depth": base64 of W*H float32 (meters; 0 where nothing is hit),
   "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
   "accelerator": "auto" | "gpu" | "cpu",
   "cpu_fallback": run networks the GPU can't run on the CPU (default false;
                   else such a network is an error),
   "iterations": FoundationPose refinement iterations (default 6)}
returns detections (box, score, visibility, pose, pose score), timings in
milliseconds and where each network ran; or {"error": ...} with status 400
for a bad request and 500 if the pipeline fails (e.g. a network that the GPU
can't run, without cpu_fallback).
"""

import argparse
import base64
import functools
import http.server
import json
import os
import threading
import time

import numpy as np

import litert_pose_estimation as lpe

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
# The CAD model as the IOC service receives it from the pose estimator asset
# (trimesh's OBJ export), so that the estimates match the service's.
CAD_OBJ = os.path.join(ROOT, "testdata", "service_golden", "raw_stock_2x3x5.obj")


class Pipeline:
  """The LiteRT networks, compiled once per accelerator setting.

  Compilation errors (e.g. a network that the GPU can't run, without
  cpu_fallback) are raised by estimate() and not cached, so each request
  reports them.
  """

  def __init__(self, models_dir, batch_size=128):
    self._models_dir = models_dir
    self._batch_size = batch_size
    self._models = {}
    self.report = None  # The last summary the page reported (?autorun=1).
    self.save_dir = None  # Where to save the last request's images, if set.
    self._lock = threading.Lock()
    with open(CAD_OBJ) as f:
      self.cad_obj = f.read()

  def models(self, accelerator, cpu_fallback):
    """The networks for these settings, and the milliseconds spent compiling
    them in this call (0 if they were compiled before)."""
    if accelerator not in ("auto", "gpu", "cpu"):
      raise ValueError(f"unknown accelerator {accelerator!r}")
    key = (accelerator, cpu_fallback)
    if key in self._models:
      return (*self._models[key], 0.0)
    start = time.perf_counter()
    segmenter = lpe.Segmenter(self._models_dir, accelerator=accelerator,
                              cpu_fallback=cpu_fallback)
    foundationpose = lpe.FoundationPose(
        self._models_dir, batch_size=self._batch_size,
        accelerator=accelerator, cpu_fallback=cpu_fallback)
    self._models[key] = (segmenter, foundationpose)
    return segmenter, foundationpose, (time.perf_counter() - start) * 1000

  def estimate(self, rgb, depth, camera_matrix, accelerator="auto",
               cpu_fallback=False, iterations=6, confidence_threshold=0.6,
               visibility_threshold=0.6):
    """Runs the service's pipeline; returns a JSON-serializable dict."""
    with self._lock:  # One estimation at a time.
      segmenter, foundationpose, compile_ms = self.models(
          accelerator, cpu_fallback)
      t0 = time.perf_counter()
      boxes, scores, masks, visibility = segmenter.segment(
          rgb, confidence_threshold, visibility_threshold)
      t1 = time.perf_counter()
      if len(masks):
        rotations, translations, confidences = foundationpose.estimate(
            rgb, depth, masks.astype(np.uint8), camera_matrix, self.cad_obj,
            iterations)
      else:
        rotations = translations = confidences = []
      t2 = time.perf_counter()
    detections = []
    for i in range(len(boxes)):
      pose = np.eye(4)
      pose[:3, :3] = rotations[i]
      pose[:3, 3] = translations[i]
      detections.append({
          "box_xyxy": [float(v) for v in boxes[i]],
          "score": float(scores[i]),
          "visibility": float(visibility[i]),
          "mask_pixels": int(masks[i].sum()),
          # Row-major, 8 pixels per byte (numpy.packbits), base64.
          "mask_bits": base64.b64encode(np.packbits(masks[i])).decode(),
          "pose": pose.tolist(),  # The object in the camera (OpenCV) frame.
          "pose_score": float(confidences[i][0]),
      })
    return {
        "detections": detections,
        "timings_ms": {
            "segmentation": (t1 - t0) * 1000,
            "pose_estimation": (t2 - t1) * 1000,
            "total": (t2 - t0) * 1000,
            "model_compilation": compile_ms,
        },
        "accelerators": {
            "segmentation": segmenter.info(),
            "foundationpose": foundationpose.info(),
        },
        "config": {"accelerator": accelerator, "cpu_fallback": cpu_fallback,
                   "iterations": iterations, "batch_size": self._batch_size},
    }


def decode_request(body):
  """Decodes an /api/estimate request into numpy arrays."""
  w, h = int(body["width"]), int(body["height"])
  rgb = np.frombuffer(base64.b64decode(body["rgb"]), np.uint8)
  depth = np.frombuffer(base64.b64decode(body["depth"]), np.float32)
  if rgb.size != w * h * 3 or depth.size != w * h:
    raise ValueError("rgb must have W*H*3 bytes and depth W*H floats")
  return (rgb.reshape(h, w, 3).copy(), depth.reshape(h, w).copy(),
          np.array(body["camera_matrix"], np.float64))


class Handler(http.server.SimpleHTTPRequestHandler):

  def __init__(self, *args, pipeline=None, **kwargs):
    self.pipeline = pipeline
    super().__init__(*args, directory=WEB, **kwargs)

  def translate_path(self, path):
    # The CAD model for the 3D view, from the repository.
    if path.split("?")[0] == "/assets/raw_stock_2x3x5.glb":
      return os.path.join(ROOT, "assets", "raw_stock_2x3x5",
                          "raw_stock_2x3x5.glb")
    return super().translate_path(path)

  def do_GET(self):
    if self.path == "/api/report":
      self._send_json(self.pipeline.report, 200)
      return
    super().do_GET()

  def do_POST(self):
    if self.path == "/api/report":
      # The page's summary of an ?autorun=1 run, for the browser test.
      self.pipeline.report = json.loads(
          self.rfile.read(int(self.headers["Content-Length"])))
      self._send_json({}, 200)
      return
    if self.path != "/api/estimate":
      self.send_error(404)
      return
    try:
      t0 = time.perf_counter()
      body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
      rgb, depth, k = decode_request(body)
      decode_ms = (time.perf_counter() - t0) * 1000
      if self.pipeline.save_dir:
        os.makedirs(self.pipeline.save_dir, exist_ok=True)
        np.save(os.path.join(self.pipeline.save_dir, "rgb.npy"), rgb)
        np.save(os.path.join(self.pipeline.save_dir, "depth.npy"), depth)
      result = self.pipeline.estimate(
          rgb, depth, k, accelerator=body.get("accelerator", "auto"),
          cpu_fallback=bool(body.get("cpu_fallback", False)),
          iterations=int(body.get("iterations", 6)))
      result["timings_ms"]["decode"] = decode_ms
      self._send_json(result, 200)
    except (ValueError, KeyError, TypeError) as e:  # Includes JSON errors.
      self._send_json({"error": str(e)}, 400)
    except Exception as e:  # pylint: disable=broad-except
      self._send_json({"error": str(e)}, 500)

  def _send_json(self, value, status):
    payload = json.dumps(value).encode()
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(payload)))
    self.end_headers()
    self.wfile.write(payload)

  def log_message(self, fmt, *args):
    if "/api/" in str(args[0] if args else ""):
      super().log_message(fmt, *args)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="127.0.0.1",
                      help="address to listen on (0.0.0.0 for all)")
  parser.add_argument("--port", type=int, default=8765)
  parser.add_argument("--models_dir", default=os.path.join(ROOT, "models"))
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--save_dir", default=None,
                      help="saves each request's rgb.npy and depth.npy here")
  args = parser.parse_args()
  pipeline = Pipeline(args.models_dir, args.batch_size)
  pipeline.save_dir = args.save_dir
  server = http.server.ThreadingHTTPServer(
      (args.host, args.port), functools.partial(Handler, pipeline=pipeline))
  print(f"Serving http://{args.host}:{args.port}")
  server.serve_forever()


if __name__ == "__main__":
  main()
