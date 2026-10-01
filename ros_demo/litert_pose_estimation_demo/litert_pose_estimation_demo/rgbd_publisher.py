"""Publishes an RGB-D frame like a camera driver and triggers pose estimation.

The frame is read from a directory in the format of testdata/service_golden:

  case.json   "camera_matrix" (3x3 intrinsics), and optionally
              "image": {"rgb": ..., "depth": ...} file names
  rgb.png     the color image
  depth.npy   the depth registered to the color image, float32 in meters
              (or depth.png: uint16 in millimeters)

The node publishes rgb/image_raw (rgb8), depth/image_raw (32FC1) and
rgb/camera_info with the same timestamp at `rate` Hz (reliable, so that the
large images aren't dropped; best-effort subscribers receive them too), calls
/litert_pose_estimation/estimate after `estimate_after` seconds (and every
`estimate_every` seconds if > 0), and logs the estimated poses.
"""

import json
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray


def _load_frame(frame_dir):
  with open(os.path.join(frame_dir, "case.json")) as f:
    case = json.load(f)
  files = case.get("image", {})
  rgb = cv2.imread(os.path.join(frame_dir, files.get("rgb", "rgb.png")),
                   cv2.IMREAD_COLOR)
  if rgb is None:
    raise FileNotFoundError(f"No color image in {frame_dir}")
  rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
  depth_file = os.path.join(frame_dir, files.get("depth", "depth.npy"))
  if depth_file.endswith(".npy"):
    depth = np.load(depth_file).astype(np.float32)
  else:
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
    depth /= 1000.0
  return rgb, depth, np.array(case["camera_matrix"], dtype=np.float64)


class RgbdPublisher(Node):

  def __init__(self):
    super().__init__("rgbd_publisher")
    frame_dir = self.declare_parameter("frame_dir", "").value
    self._frame_id = self.declare_parameter(
        "frame_id", "camera_color_optical_frame").value
    rate = self.declare_parameter("rate", 2.0).value
    estimate_after = self.declare_parameter("estimate_after", 3.0).value
    self._estimate_every = self.declare_parameter("estimate_every", 0.0).value
    self._rgb, self._depth, self._k = _load_frame(frame_dir)
    h, w = self._rgb.shape[:2]
    self.get_logger().info(f"Publishing {w}x{h} RGB-D frame from {frame_dir}")

    self._rgb_pub = self.create_publisher(
        Image, "rgb/image_raw", 2)
    self._depth_pub = self.create_publisher(
        Image, "depth/image_raw", 2)
    self._info_pub = self.create_publisher(
        CameraInfo, "rgb/camera_info", 2)
    self.create_subscription(Detection3DArray,
                             "/litert_pose_estimation/detections",
                             self._on_detections, 10)
    self._estimate = self.create_client(Trigger,
                                        "/litert_pose_estimation/estimate")
    self._pending = None
    self.create_timer(1.0 / rate, self._publish)
    if estimate_after > 0:
      self._first = self.create_timer(estimate_after, self._trigger_once)

  def _publish(self):
    stamp = self.get_clock().now().to_msg()
    h, w = self._rgb.shape[:2]
    rgb = Image(height=h, width=w, encoding="rgb8", step=w * 3,
                data=self._rgb.tobytes())
    depth = Image(height=h, width=w, encoding="32FC1", step=w * 4,
                  data=self._depth.tobytes())
    info = CameraInfo(height=h, width=w, distortion_model="plumb_bob")
    info.k = self._k.flatten().tolist()
    info.p = [self._k[0, 0], 0.0, self._k[0, 2], 0.0,
              0.0, self._k[1, 1], self._k[1, 2], 0.0, 0.0, 0.0, 1.0, 0.0]
    for msg in (rgb, depth, info):
      msg.header.stamp = stamp
      msg.header.frame_id = self._frame_id
    self._rgb_pub.publish(rgb)
    self._depth_pub.publish(depth)
    self._info_pub.publish(info)

  def _trigger_once(self):
    # Retried by the timer until the node's service is up.
    if not self._estimate.service_is_ready():
      self.get_logger().info("Waiting for /litert_pose_estimation/estimate ...")
      return
    self._first.cancel()
    self._trigger()
    if self._estimate_every > 0:
      self.create_timer(self._estimate_every, self._trigger)

  def _trigger(self):
    if self._pending is not None and not self._pending.done():
      return  # The previous estimation is still running.
    if not self._estimate.service_is_ready():
      self.get_logger().warn("/litert_pose_estimation/estimate is not up yet")
      return
    self.get_logger().info("Triggering pose estimation ...")
    self._pending = self._estimate.call_async(Trigger.Request())
    self._pending.add_done_callback(self._on_estimated)

  def _on_estimated(self, future):
    result = future.result()
    self.get_logger().info(
        f"Pose estimation: {'OK' if result.success else 'FAILED'}: "
        f"{result.message}")
    if not result.success and "received yet" in result.message:
      # The node hasn't got a frame yet: try again shortly.
      self._retry = self.create_timer(2.0, self._retry_once)

  def _retry_once(self):
    self._retry.cancel()
    self._trigger()

  def _on_detections(self, msg):
    for d in msg.detections:
      r = d.results[0]
      p, q = r.pose.pose.position, r.pose.pose.orientation
      self.get_logger().info(
          f"{r.hypothesis.class_id} #{d.id}: position [{p.x:.4f}, {p.y:.4f}, "
          f"{p.z:.4f}] m, orientation (xyzw) [{q.x:.4f}, {q.y:.4f}, "
          f"{q.z:.4f}, {q.w:.4f}], score {r.hypothesis.score:.3f}")


def main():
  rclpy.init()
  node = RgbdPublisher()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
  main()
