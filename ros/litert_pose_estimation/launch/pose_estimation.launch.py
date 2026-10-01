"""Launches the LiteRT pose estimation node for the OMTS raw stock.

Example (Orbbec Gemini driver topics):
  ros2 launch litert_pose_estimation pose_estimation.launch.py \
      perception_root:=$HOME/perception-litert \
      rgb:=/camera/color/image_raw depth:=/camera/depth/image_raw \
      camera_info:=/camera/color/camera_info
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
  share = get_package_share_directory("litert_pose_estimation")
  root = LaunchConfiguration("perception_root")
  return LaunchDescription([
      DeclareLaunchArgument(
          "perception_root",
          description="Checkout of intrinsics-litert-pose-estimation "
          "(models/, assets/)."),
      DeclareLaunchArgument("accelerator", default_value="auto"),
      DeclareLaunchArgument("cpu_fallback", default_value="false"),
      DeclareLaunchArgument("rgb", default_value="rgb/image_raw"),
      DeclareLaunchArgument("depth", default_value="depth/image_raw"),
      DeclareLaunchArgument("camera_info", default_value="rgb/camera_info"),
      Node(
          package="litert_pose_estimation",
          executable="pose_estimation_node",
          name="litert_pose_estimation",
          output="screen",
          parameters=[
              os.path.join(share, "config", "omts_raw_stock.yaml"),
              {
                  "models_dir": PathJoinSubstitution([root, "models"]),
                  "cad_obj": PathJoinSubstitution(
                      [root, "assets", "raw_stock_2x3x5",
                       "raw_stock_2x3x5.obj"]),
                  "accelerator": LaunchConfiguration("accelerator"),
                  "cpu_fallback": ParameterValue(
                      LaunchConfiguration("cpu_fallback"), value_type=bool),
              },
          ],
          remappings=[
              ("rgb/image_raw", LaunchConfiguration("rgb")),
              ("depth/image_raw", LaunchConfiguration("depth")),
              ("rgb/camera_info", LaunchConfiguration("camera_info")),
          ],
      ),
  ])
