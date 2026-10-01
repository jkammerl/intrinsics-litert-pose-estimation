#include <exception>
#include <memory>

#include "litert_pose_estimation/pose_estimation_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  std::shared_ptr<litert_pose_estimation::PoseEstimationNode> node;
  try {
    node = std::make_shared<litert_pose_estimation::PoseEstimationNode>();
  } catch (const std::exception& e) {
    // E.g. a network that the GPU can't run without cpu_fallback.
    RCLCPP_FATAL(rclcpp::get_logger("litert_pose_estimation"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  // Two threads: the ~/estimate service runs while frames keep arriving.
  rclcpp::executors::MultiThreadedExecutor executor(
      rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
