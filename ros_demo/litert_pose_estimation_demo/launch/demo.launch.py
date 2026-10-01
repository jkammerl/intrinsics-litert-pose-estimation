"""Runs the LiteRT pose estimation node on a recorded RGB-D frame.

  ros2 launch litert_pose_estimation_demo demo.launch.py \
      repo_root:=$HOME/intrinsics-litert-pose-estimation

Publishes testdata/service_golden (a frame of the simulated Orbbec camera in
OMTS's Lab BB-01 cell) and triggers the estimation after the node is up.
Use frame_dir:=... for another frame, accelerator:=cpu|gpu|auto to choose
LiteRT's hardware, and estimate_every:=N to estimate repeatedly.
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
  repo_root = LaunchConfiguration("repo_root")
  estimation = IncludeLaunchDescription(
      PythonLaunchDescriptionSource(PathJoinSubstitution([
          get_package_share_directory("litert_pose_estimation"), "launch",
          "pose_estimation.launch.py"])),
      launch_arguments={
          "perception_root": repo_root,
          "accelerator": LaunchConfiguration("accelerator"),
      }.items(),
  )
  publisher = Node(
      package="litert_pose_estimation_demo",
      executable="rgbd_publisher",
      output="screen",
      parameters=[{
          "frame_dir": LaunchConfiguration("frame_dir"),
          "estimate_after": LaunchConfiguration("estimate_after"),
          "estimate_every": LaunchConfiguration("estimate_every"),
      }],
  )
  return LaunchDescription([
      DeclareLaunchArgument(
          "repo_root",
          description="Checkout of intrinsics-litert-pose-estimation."),
      DeclareLaunchArgument(
          "frame_dir",
          default_value=PathJoinSubstitution(
              [repo_root, "testdata", "service_golden"])),
      DeclareLaunchArgument("accelerator", default_value="auto"),
      # Compiling the models takes a while; trigger once the node is up.
      DeclareLaunchArgument("estimate_after", default_value="20.0"),
      DeclareLaunchArgument("estimate_every", default_value="0.0"),
      estimation,
      publisher,
  ])
