#include "litert_pose_estimation/pose_estimation_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<litert_pose_estimation::PoseEstimationNode>();
  // Two threads: the ~/estimate service runs while frames keep arriving.
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
