"""Checks the LiteRT backend against the IOC pose estimator service.

The service's inputs and outputs for one capture are in
testdata/service_golden. The backend's two calls, made with the inputs that
the service passed to its models, must return the service's outputs.

  PYTHONPATH=<dir with litert_pose_estimation*.so>:. python3 litert_backend_test.py
"""

import json
import os
import types
import unittest

import cv2
import numpy as np

import litert_backend

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GOLDEN = os.path.join(ROOT, "testdata", "service_golden")
ACCELERATOR = os.environ.get("PERCEPTION_ACCELERATOR", "cpu")


def _box_symmetric_angle_deg(a, b):
  """Rotation angle between a and b modulo the box's half-turn symmetries."""
  best = 180.0
  for s in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
    trace = np.trace(a.T @ b @ np.diag(s))
    best = min(best, np.degrees(np.arccos(np.clip((trace - 1) / 2, -1, 1))))
  return best


class LiteRtBackendTest(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    with open(os.path.join(GOLDEN, "case.json")) as f:
      cls.case = json.load(f)
    cls.rgb = cv2.cvtColor(cv2.imread(os.path.join(GOLDEN, "rgb.png")),
                           cv2.COLOR_BGR2RGB)
    cls.depth = np.load(os.path.join(GOLDEN, "depth.npy"))
    cls.k = np.array(cls.case["camera_matrix"])
    with open(os.path.join(GOLDEN, cls.case["cad_obj"]), "rb") as f:
      cls.cad = f.read()
    cls.servicer = types.SimpleNamespace(
        _batch_size=cls.case["config"]["batch_size"],
        segmentation_model=None, pose_estimator_model=None)
    cls.info = litert_backend.install(
        cls.servicer, os.path.join(ROOT, "models"), accelerator=ACCELERATOR)

  def test_reports_accelerators(self):
    self.assertIn(self.info["segmentation"]["accelerator"], ("cpu", "gpu"))
    self.assertIn("refiner", self.info["foundationpose"])

  def test_segmentation_matches_the_service(self):
    config = self.case["config"]
    # The service passes the image as (1, 3, H, W).
    image = np.transpose(self.rgb, (2, 0, 1))[None]
    boxes, scores, masks, visibility, vis = (
        self.servicer.segmentation_model.run_inference(
            image, confidence_threshold=config["confidence_threshold"],
            visibility_threshold=config["visibility_threshold"],
            return_vis=True))
    want = self.case["segmentation"]
    self.assertEqual(len(boxes), len(want))
    self.assertEqual(vis.shape, self.rgb.shape)
    for i, w in enumerate(want):
      np.testing.assert_allclose(boxes[i], w["box_xyxy"], atol=0.5)
      self.assertAlmostEqual(float(scores[i]), w["score"], delta=2e-3)
      self.assertAlmostEqual(float(visibility[i]), w["visibility"], delta=2e-3)
      want_mask = cv2.imread(os.path.join(GOLDEN, w["mask"]), 0) > 0
      iou = (masks[i] & want_mask).sum() / (masks[i] | want_mask).sum()
      self.assertGreater(iou, 0.995)

  def test_pose_estimation_matches_the_service(self):
    masks = np.stack([cv2.imread(os.path.join(GOLDEN, s["mask"]), 0)
                      for s in self.case["segmentation"]])
    rotations, translations, confidences = (
        self.servicer.pose_estimator_model.run_inference(
            rgb=self.rgb, depth=self.depth, mask=masks,
            intrinsic_matrix=self.k, cad_bytes=self.cad,
            num_iterations=self.case["config"]["refinement_iterations"],
            batch_size=self.case["config"]["batch_size"]))
    print(f"FoundationPose with LiteRT: "
          f"{self.servicer.pose_estimator_model.last_duration:.1f} s")
    for i, want in enumerate(self.case["poses"]):
      self.assertLess(np.linalg.norm(translations[i] - want["translation"]),
                      1e-3)
      self.assertLess(_box_symmetric_angle_deg(
          np.array(want["rotation"]), rotations[i]), 0.5)
      self.assertAlmostEqual(float(confidences[i][0]), want["score"],
                             delta=0.05)


if __name__ == "__main__":
  unittest.main()
