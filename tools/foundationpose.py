"""FoundationPose 6D pose estimation, without Triton.

The same pipeline as FoundationPose's Triton model (intrinsic-omts
third_party/foundationpose/model.py), with the refiner and scorer networks
passed in as functions, so that they can run in ONNX Runtime or LiteRT:

  1. Sample 280 candidate poses around the object's position (40 views x 7
     in-plane rotations, 0 to 360 degrees in 60 degree steps).
  2. Refine all candidates `iterations` times: render each candidate, crop the
     observed RGB and point map around it, and let the refiner predict a pose
     update.
  3. Score the refined candidates against each other and return the best.

Unlike model.py, which processes the candidates in chunks of 240 (240 + 40),
all 280 candidates are refined and scored in one batch. Refinement results are
the same; scores differ slightly, since the scorer compares the candidates in
a batch against each other.

Network inputs are [280, 160, 160, 6] float32: `input1` is the rendered
candidate (0.5 * mask, then the normalized point map) and `input2` the
observed crop (RGB in [0, 1], then the normalized point map).
"""

from collections.abc import Callable
import dataclasses

import cv2
import numpy as np

import foundationpose_numpy as fp

NUM_VIEWS = 40
INPLANE_STEP_DEG = 60.0
BATCH = 280  # NUM_VIEWS * (360 / INPLANE_STEP_DEG + 1) candidates.
RES = (160, 160)
CROP_RATIO = 1.2
MAX_ROTATION_RAD = 0.34906585  # 20 degrees.

# (input1, input2) -> (translation deltas [B, 3], rotation deltas [B, 3]).
RefineFn = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]
# (input1, input2) -> scores [1, B].
ScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclasses.dataclass
class Mesh:
  vertices: np.ndarray  # [V, 3] float32, meters.
  faces: np.ndarray  # [F, 3] int32.

  @property
  def diameter(self) -> float:
    extent = self.vertices.max(axis=0) - self.vertices.min(axis=0)
    return max(float(np.linalg.norm(extent)), 0.05)

  @property
  def center(self) -> np.ndarray:
    return (
        (self.vertices.min(axis=0) + self.vertices.max(axis=0)) / 2.0
    ).astype(np.float32)


def load_obj(path: str) -> Mesh:
  """Loads a triangle (or quad) mesh from an OBJ file."""
  vertices, faces = [], []
  with open(path) as f:
    for line in f:
      parts = line.split()
      if not parts:
        continue
      if parts[0] == "v":
        vertices.append([float(x) for x in parts[1:4]])
      elif parts[0] == "f":
        ids = [int(p.split("/")[0]) - 1 for p in parts[1:]]
        for i in range(1, len(ids) - 1):
          faces.append([ids[0], ids[i], ids[i + 1]])
  return Mesh(np.array(vertices, np.float32), np.array(faces, np.int32))


class FoundationPose:
  """Estimates the pose of a mesh in an RGB-D image."""

  def __init__(self, refine: RefineFn, score: ScoreFn):
    self._refine = refine
    self._score = score
    self._rasterizer = fp.RasterizeCpuContext(RES[0], RES[1], 64)

  def estimate(
      self,
      rgb: np.ndarray,
      depth: np.ndarray,
      mask: np.ndarray,
      K: np.ndarray,
      mesh: Mesh,
      iterations: int = 5,
      trace: dict[str, list[np.ndarray]] | None = None,
  ) -> tuple[np.ndarray, float]:
    """Returns the object's pose in the camera frame (4x4) and its score.

    Args:
      rgb: [H, W, 3] uint8 image.
      depth: [H, W] float32 depth in meters, 0 where invalid.
      mask: [H, W] segmentation mask of the object.
      K: 3x3 camera intrinsics.
      mesh: The object's mesh, in meters.
      iterations: Number of refinement iterations.
      trace: If given, the network inputs and outputs are appended to it.
    """
    K = np.asarray(K, np.float32)
    depth = np.asarray(depth, np.float32)
    rgb_float = rgb.astype(np.float32) / 255.0
    h, w = depth.shape
    ys, xs = np.indices((h, w), dtype=np.float32)
    xyz_map = np.stack(
        [(xs - K[0, 2]) * depth / K[0, 0], (ys - K[1, 2]) * depth / K[1, 1],
         depth],
        axis=-1,
    ).astype(np.float32)

    poses = fp.sample_initial_poses(
        np.ascontiguousarray(mask > 0, np.uint8), depth, K, NUM_VIEWS,
        INPLANE_STEP_DEG,
    )
    assert len(poses) == BATCH, len(poses)
    if trace is not None:
      trace.setdefault("initial_poses", []).append(poses.copy())

    for iteration in range(iterations):
      in1, in2 = self._inputs(poses, K, mesh, xyz_map, rgb_float)
      translation, rotation = self._refine(in1, in2)
      if trace is not None:
        if iteration == 0:  # The inputs are large; keep only the first.
          trace["refine_input1"] = [in1]
          trace["refine_input2"] = [in2]
        trace.setdefault("refine_translation", []).append(translation)
        trace.setdefault("refine_rotation", []).append(rotation)
      poses = fp.update_refined_poses(
          poses, translation.astype(np.float32), rotation.astype(np.float32),
          mesh.diameter, MAX_ROTATION_RAD,
      )

    in1, in2 = self._inputs(poses, K, mesh, xyz_map, rgb_float)
    scores = self._score(in1, in2).reshape(-1)
    if trace is not None:
      for k, v in (("score_input1", in1), ("score_input2", in2),
                   ("scores", scores), ("refined_poses", poses)):
        trace.setdefault(k, []).append(v)
    best = int(np.argmax(scores))
    pose = fp.apply_mesh_center_offset(poses[best], mesh.center)
    return np.asarray(pose, np.float32), float(scores[best])

  def _inputs(self, poses, K, mesh, xyz_map, rgb_float):
    tfs = np.asarray(
        fp.compute_crop_window_tf(
            np.ascontiguousarray(poses, np.float32), K, RES[0], RES[1],
            CROP_RATIO, mesh.diameter,
        ),
        np.float32,
    )
    return (
        self._render(poses, K, mesh, tfs),
        self._crop(poses, tfs, mesh.diameter, xyz_map, rgb_float),
    )

  def _crop(self, poses, tfs, diameter, xyz_map, rgb_float):
    """Crops the observed RGB and point map around each candidate."""
    radius = diameter / 2.0
    out = np.zeros((len(poses), RES[0], RES[1], 6), np.float32)
    for i, (pose, tf) in enumerate(zip(poses, tfs)):
      rgb = cv2.warpPerspective(rgb_float, tf, RES, flags=cv2.INTER_LINEAR)
      xyz = cv2.warpPerspective(xyz_map, tf, RES, flags=cv2.INTER_NEAREST)
      valid = (xyz[:, :, 2] > 0.05).astype(np.float32)[:, :, None]
      out[i, :, :, :3] = rgb
      out[i, :, :, 3:] = (xyz - pose[:3, 3]) / radius * valid
    return out

  def _render(self, poses, K, mesh, tfs):
    """Renders each candidate's mask and normalized point map in its crop."""
    rotations, translations = poses[:, :3, :3], poses[:, :3, 3]
    v_cam = (
        mesh.vertices[None] @ rotations.transpose(0, 2, 1)
        + translations[:, None]
    )
    px = v_cam[..., 0] * K[0, 0] / v_cam[..., 2] + K[0, 2]
    py = v_cam[..., 1] * K[1, 1] / v_cam[..., 2] + K[1, 2]
    px = tfs[:, 0, 0, None] * px + tfs[:, 0, 2, None]
    py = tfs[:, 1, 1, None] * py + tfs[:, 1, 2, None]
    v_clip = np.stack(
        [2.0 * px / RES[1] - 1.0, 1.0 - 2.0 * py / RES[0],
         (v_cam[..., 2] - 0.1) / (100.0 - 0.1), np.ones_like(px)],
        axis=-1,
    ).astype(np.float32)
    faces = np.ascontiguousarray(mesh.faces, np.int32)
    rast = self._rasterizer.rasterize(v_clip, faces, RES[0], RES[1])
    xyz = self._rasterizer.interpolate(
        np.ascontiguousarray(v_cam, np.float32), rast, faces, RES[0], RES[1]
    )
    mask = (rast[..., 3:4] > 0).astype(np.float32)
    out = np.zeros((len(poses), RES[0], RES[1], 6), np.float32)
    out[..., :3] = 0.5 * mask
    out[..., 3:] = (xyz - translations[:, None, None]) / (
        mesh.diameter / 2.0
    ) * mask
    return np.ascontiguousarray(np.flip(out, axis=1))
