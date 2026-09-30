"""Renders synthetic RGB-D test scenes of a mesh at a known pose."""

import numpy as np

import foundationpose
import foundationpose_numpy as fp


def rotation(axis, angle_deg) -> np.ndarray:
  axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
  return fp._axis_angle_to_matrix(axis, np.deg2rad(angle_deg))  # pylint: disable=protected-access


def render_scene(
    mesh: foundationpose.Mesh,
    pose: np.ndarray,
    K: np.ndarray,
    height: int,
    width: int,
    table_offset: float = 0.05,
    seed: int = 0,
):
  """Renders `mesh` at `pose` (object in camera) in front of a table plane.

  Returns (rgb uint8 [H, W, 3], depth float32 [H, W] in meters, mask uint8).
  The object is shaded with a directional light; the table is 5 cm behind the
  object's center, with mild texture and sensor noise so that images aren't
  perfectly flat.
  """
  rng = np.random.default_rng(seed)
  v_cam = mesh.vertices @ pose[:3, :3].T + pose[:3, 3]
  z = v_cam[:, 2]
  px = v_cam[:, 0] * K[0, 0] / z + K[0, 2]
  py = v_cam[:, 1] * K[1, 1] / z + K[1, 2]
  ndc = np.stack(
      [2 * px / width - 1, 2 * py / height - 1, (z - 0.1) / (100.0 - 0.1)], -1
  )
  # Homogeneous clip coordinates with w = z for perspective-correct depth.
  v_clip = np.concatenate([ndc * z[:, None], z[:, None]], -1)[None]
  ctx = fp.RasterizeCpuContext()
  rast = ctx.rasterize(v_clip.astype(np.float32), mesh.faces, height, width)
  xyz = ctx.interpolate(
      v_cam[None].astype(np.float32), rast, mesh.faces, height, width
  )
  mask = rast[0, :, :, 3] > 0

  depth = np.full((height, width), pose[2, 3] + table_offset, np.float32)
  depth[mask] = xyz[0, mask, 2]
  depth += rng.normal(0, 0.0005, depth.shape).astype(np.float32)

  a, b, c = mesh.vertices[mesh.faces].transpose(1, 0, 2)
  normals = np.cross(b - a, c - a)
  normals /= np.linalg.norm(normals, axis=1, keepdims=True)
  normals = normals @ pose[:3, :3].T
  light = np.array([0.3, -0.5, -1.0]) / np.linalg.norm([0.3, -0.5, -1.0])
  shade = np.clip(np.abs(normals @ light), 0.15, 1.0)

  ys, xs = np.indices((height, width))
  table = 90 + 12 * np.sin(xs / 7.0) * np.sin(ys / 11.0)
  rgb = np.repeat(table[:, :, None], 3, axis=2)
  face_ids = rast[0, :, :, 3].astype(np.int64) - 1
  rgb[mask] = shade[face_ids[mask]][:, None] * [200, 205, 210]
  rgb += rng.normal(0, 2.0, rgb.shape)
  rgb = np.clip(rgb, 0, 255).astype(np.uint8)
  return rgb, depth, mask.astype(np.uint8)
