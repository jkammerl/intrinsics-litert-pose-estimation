"""Tests of the web demo, its server and LiteRT's WebGPU inference behind it.

  PYTHONPATH=<dir with litert_pose_estimation*.so> python3 demo/demo_test.py

ServerTest runs the server in-process and sends it requests like the page
does: the service's recorded capture (testdata/service_golden) must give the
service's pose on the CPU; on LiteRT's GPU accelerator (WebGPU, on Vulkan),
every network must run on the GPU or the request fail, naming the network
(without cpu_fallback), and with cpu_fallback the results must be the CPU's.
BrowserTest opens the page in headless Chrome (WebGL through SwiftShader):
pose.js's unit tests, and the whole demo from the rendered RGB-D frame to the
estimate, on the CPU and on the GPU, and the page's error report.

GPU tests run if LiteRT can use a Vulkan device (also a software one such as
llvmpipe, slowly), browser tests if Chrome is installed (or $CHROME is set).
"""

import base64
import functools
import http.server
import json
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

import cv2
import numpy as np

import litert_pose_estimation as lpe
import server

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GOLDEN = os.path.join(ROOT, "testdata", "service_golden")


def _box_symmetric_angle_deg(a, b):
  """Rotation angle between a and b modulo the box's half-turn symmetries."""
  best = 180.0
  for s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
    trace = np.trace(np.asarray(a).T @ np.asarray(b) @ np.diag(s))
    best = min(best, np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1))))
  return best


@functools.cache
def _gpu_unavailable_reason():
  """Why LiteRT's GPU accelerator can't run here, or None if it can."""
  reason = lpe.hardware_gpu_unavailable_reason()
  if not reason or "software Vulkan" in reason:
    return None
  return reason


def _chrome():
  return os.environ.get("CHROME") or next(
      (shutil.which(c) for c in ("google-chrome", "chromium", "chromium-browser")
       if shutil.which(c)), None)


class _Server:
  """The demo server on a free port, in this process."""

  def __init__(self):
    self.pipeline = server.Pipeline(os.path.join(ROOT, "models"))
    self.httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        functools.partial(server.Handler, pipeline=self.pipeline))
    self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
    threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

  def get(self, path):
    with urllib.request.urlopen(self.url + path) as response:
      return response.status, response.read()

  def post(self, path, body):
    request = urllib.request.Request(
        self.url + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
      with urllib.request.urlopen(request) as response:
        return response.status, json.loads(response.read())
    except urllib.error.HTTPError as e:
      with e:
        return e.code, json.loads(e.read())


_server = None


def setUpModule():
  global _server
  _server = _Server()


def tearDownModule():
  _server.httpd.shutdown()


def _golden_request(**kwargs):
  with open(os.path.join(GOLDEN, "case.json")) as f:
    case = json.load(f)
  rgb = cv2.cvtColor(cv2.imread(os.path.join(GOLDEN, "rgb.png")),
                     cv2.COLOR_BGR2RGB)
  depth = np.load(os.path.join(GOLDEN, "depth.npy")).astype(np.float32)
  body = {
      "width": rgb.shape[1], "height": rgb.shape[0],
      "rgb": base64.b64encode(rgb.tobytes()).decode(),
      "depth": base64.b64encode(depth.tobytes()).decode(),
      "camera_matrix": case["camera_matrix"],
  }
  body.update(kwargs)
  return case, body


def _placements(accelerators):
  """(network, info) for each network in a response's "accelerators"."""
  yield "segmentation", accelerators["segmentation"]
  yield from accelerators["foundationpose"].items()


class ServerTest(unittest.TestCase):

  def test_serves_the_page_and_the_cad_model(self):
    for path in ("/", "/app.js", "/pose.js", "/style.css",
                 "/assets/raw_stock_2x3x5.glb", "/test/pose_test.html"):
      status, content = _server.get(path)
      self.assertEqual(status, 200, path)
      self.assertTrue(content, path)
    _, glb = _server.get("/assets/raw_stock_2x3x5.glb")
    self.assertEqual(glb[:4], b"glTF")

  def test_decodes_requests(self):
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    depth = np.linspace(0.2, 0.7, 6, dtype=np.float32).reshape(2, 3)
    k = [[500, 0, 1.5], [0, 500, 1], [0, 0, 1]]
    got_rgb, got_depth, got_k = server.decode_request({
        "width": 3, "height": 2, "camera_matrix": k,
        "rgb": base64.b64encode(rgb.tobytes()).decode(),
        "depth": base64.b64encode(depth.tobytes()).decode()})
    np.testing.assert_array_equal(got_rgb, rgb)
    np.testing.assert_array_equal(got_depth, depth)
    np.testing.assert_array_equal(got_k, k)

  def test_rejects_bad_requests(self):
    _, body = _golden_request()
    status, result = _server.post("/api/estimate", dict(body, width=100))
    self.assertEqual(status, 400)
    self.assertIn("W*H", result["error"])
    status, result = _server.post("/api/estimate",
                                  dict(body, accelerator="tpu"))
    self.assertEqual(status, 400)
    self.assertIn("accelerator", result["error"])

  def test_returns_the_services_pose_on_the_cpu(self):
    case, body = _golden_request(accelerator="cpu", iterations=6)
    status, result = _server.post("/api/estimate", body)
    self.assertEqual(status, 200, result)
    self.assertEqual(len(result["detections"]), 1)
    det, want_seg, want = (result["detections"][0], case["segmentation"][0],
                           case["poses"][0])
    np.testing.assert_allclose(det["box_xyxy"], want_seg["box_xyxy"], atol=0.5)
    self.assertAlmostEqual(det["score"], want_seg["score"], delta=1e-3)
    mask = np.unpackbits(np.frombuffer(base64.b64decode(det["mask_bits"]),
                                       np.uint8))[:1280 * 800].reshape(800, 1280)
    self.assertEqual(int(mask.sum()), det["mask_pixels"])
    pose = np.array(det["pose"])
    self.assertLess(np.linalg.norm(pose[:3, 3] - want["translation"]) * 1000,
                    1.0)
    self.assertLess(_box_symmetric_angle_deg(pose[:3, :3], want["rotation"]),
                    0.5)
    self.assertAlmostEqual(det["pose_score"], want["score"], delta=0.05)
    for key in ("segmentation", "pose_estimation", "total", "decode",
                "model_compilation"):
      self.assertGreaterEqual(result["timings_ms"][key], 0, key)
    self.assertEqual(result["accelerators"]["segmentation"]["accelerator"],
                     "cpu")

  def test_webgpu_accelerator_without_cpu_fallback(self):
    # Every network on the GPU, or an error naming the network that the GPU
    # can't run (on llvmpipe, the scorers).
    if _gpu_unavailable_reason():
      self.skipTest(_gpu_unavailable_reason())
    _, body = _golden_request(iterations=1, accelerator="gpu")
    status, result = _server.post("/api/estimate", body)
    if status == 200:
      for name, info in _placements(result["accelerators"]):
        self.assertEqual(info["accelerator"], "gpu", name)
    else:
      self.assertEqual(status, 500)
      self.assertIn("can't run on this GPU", result["error"])
      self.assertIn("cpu_fallback", result["error"])
      print("WebGPU without CPU fallback:", result["error"][:300])

  def test_webgpu_accelerator_matches_the_cpu(self):
    if _gpu_unavailable_reason():
      self.skipTest(_gpu_unavailable_reason())
    # One refinement iteration: on a software Vulkan device the GPU is slow.
    # With cpu_fallback, networks that this GPU can't run use the CPU.
    _, body = _golden_request(iterations=1)
    status, cpu = _server.post("/api/estimate", dict(body, accelerator="cpu"))
    self.assertEqual(status, 200, cpu)
    status, gpu = _server.post(
        "/api/estimate", dict(body, accelerator="gpu", cpu_fallback=True))
    self.assertEqual(status, 200, gpu)
    for name, info in _placements(gpu["accelerators"]):
      if info["accelerator"] == "cpu":
        self.assertIn("can't run on this GPU", info["fallback_reason"], name)
    accelerators = gpu["accelerators"]
    self.assertEqual(accelerators["segmentation"]["accelerator"], "gpu")
    self.assertEqual(accelerators["foundationpose"]["refiner"]["accelerator"],
                     "gpu")
    print("WebGPU:", json.dumps(accelerators), json.dumps(gpu["timings_ms"]))
    self.assertEqual(len(gpu["detections"]), len(cpu["detections"]))
    for g, c in zip(gpu["detections"], cpu["detections"]):
      np.testing.assert_allclose(g["box_xyxy"], c["box_xyxy"], atol=1.0)
      self.assertAlmostEqual(g["score"], c["score"], delta=5e-3)
      g_pose, c_pose = np.array(g["pose"]), np.array(c["pose"])
      self.assertLess(np.linalg.norm(g_pose[:3, 3] - c_pose[:3, 3]) * 1000, 1.0)
      self.assertLess(_box_symmetric_angle_deg(g_pose[:3, :3], c_pose[:3, :3]),
                      0.5)
      self.assertAlmostEqual(g["pose_score"], c["pose_score"], delta=0.1)


class BrowserTest(unittest.TestCase):

  def setUp(self):
    if not _chrome():
      self.skipTest("no Chrome")

  def _run_page(self, path, timeout_s):
    """Opens the page in headless Chrome; returns what it reported."""
    _server.pipeline.report = None
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:
      chrome = subprocess.Popen(
          [_chrome(), "--headless=new", "--no-first-run",
           f"--user-data-dir={profile}", "--use-angle=swiftshader",
           "--enable-unsafe-swiftshader", "--window-size=1500,1100",
           "--remote-debugging-port=0", _server.url + path],
          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
          start_new_session=True)
      try:
        deadline = time.time() + timeout_s
        while _server.pipeline.report is None and time.time() < deadline:
          time.sleep(0.5)
      finally:
        os.killpg(chrome.pid, signal.SIGKILL)  # Chrome and its children.
        chrome.wait()
    self.assertIsNotNone(_server.pipeline.report, f"{path}: no report")
    return _server.pipeline.report

  def test_pose_math(self):
    report = self._run_page("/test/pose_test.html", 60)
    for test in report["tests"]:
      self.assertTrue(test["ok"], test)
    self.assertGreaterEqual(len(report["tests"]), 8)

  def _check_estimate(self, report, translation_mm, rotation_deg):
    self.assertNotIn("error", report)
    self.assertEqual(report["detections"], 1)
    self.assertGreater(report["segmentation_score"], 0.9)
    print(f"browser: {report['translation_error_mm']:.2f} mm, "
          f"{report['rotation_error_deg']:.2f} deg, "
          f"ADD-S {report['add_s_mm']:.2f} mm, timings "
          f"{json.dumps(report['timings_ms'])}")
    self.assertLess(report["translation_error_mm"], translation_mm)
    self.assertLess(report["rotation_error_deg"], rotation_deg)
    self.assertGreater(report["client_ms"]["capture"], 0)

  def test_estimates_the_pose_of_the_rendered_box(self):
    # The rendered RGB-D frame of the box at its initial pose goes through the
    # whole pipeline; the estimate must match the box's pose in the scene.
    report = self._run_page("/?autorun=1&accelerator=cpu&iterations=3", 900)
    self._check_estimate(report, translation_mm=5, rotation_deg=5)
    self.assertEqual(report["accelerators"]["segmentation"]["accelerator"],
                     "cpu")

  def test_estimates_the_pose_of_a_moved_box(self):
    # Another pose: standing on its end, turned, off the image's center.
    report = self._run_page(
        "/?autorun=1&accelerator=cpu&iterations=3&rest=-0.05,-0.02,110,y", 900)
    self._check_estimate(report, translation_mm=5, rotation_deg=5)

  def test_reports_networks_the_gpu_cant_run_without_cpu_fallback(self):
    if _gpu_unavailable_reason():
      self.skipTest(_gpu_unavailable_reason())
    report = self._run_page("/?autorun=1&accelerator=gpu&iterations=1", 1800)
    if "error" in report:  # This GPU can't run every network.
      self.assertIn("can't run on this GPU", report["error"])
      self.assertIn("cpu_fallback", report["error"])
    else:
      self._check_estimate(report, translation_mm=5, rotation_deg=10)

  def test_estimates_with_the_webgpu_accelerator(self):
    if _gpu_unavailable_reason():
      self.skipTest(_gpu_unavailable_reason())
    report = self._run_page(
        "/?autorun=1&accelerator=gpu&cpu_fallback=1&iterations=1", 1800)
    self._check_estimate(report, translation_mm=5, rotation_deg=10)
    self.assertEqual(report["accelerators"]["segmentation"]["accelerator"],
                     "gpu")
    self.assertEqual(
        report["accelerators"]["foundationpose"]["refiner"]["accelerator"],
        "gpu")


if __name__ == "__main__":
  unittest.main()
