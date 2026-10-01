// End-to-end golden test of the ROS 2 node: publishes the RGB-D frame that
// the IOC pose estimator service received (testdata/service_golden), calls
// ~/estimate, and compares the published detections with the service's
// result.

#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <limits>
#include <memory>
#include <thread>

#include "gtest/gtest.h"
#include "litert_pose_estimation/pose_estimation_node.hpp"
#include "test_util.h"

namespace {

using namespace std::chrono_literals;
namespace pt = perception::testing;

std::string Golden(const std::string& name) {
  return std::string(PERCEPTION_SOURCE_ROOT) + "/testdata/service_golden/" +
         name;
}

class GoldenTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() { rclcpp::init(0, nullptr); }
  static void TearDownTestSuite() { rclcpp::shutdown(); }
};

TEST_F(GoldenTest, NodeReproducesTheIocPoseEstimatorService) {
  const nlohmann::json golden = pt::LoadJson(Golden("case.json"));
  const char* accelerator = std::getenv("PERCEPTION_ACCELERATOR");
  const char* cpu_fallback = std::getenv("PERCEPTION_CPU_FALLBACK");
  rclcpp::NodeOptions options;
  options.parameter_overrides({
      {"models_dir", std::string(PERCEPTION_SOURCE_ROOT) + "/models"},
      {"cad_obj", Golden(golden["cad_obj"])},
      {"accelerator", std::string(accelerator ? accelerator : "auto")},
      {"cpu_fallback", cpu_fallback != nullptr &&
                           std::string(cpu_fallback) == "1"},
      {"litert_library_dir", std::string(LITERT_LIBRARY_DIR)},
      {"confidence_threshold",
       golden["config"]["confidence_threshold"].get<double>()},
      {"visibility_threshold",
       golden["config"]["visibility_threshold"].get<double>()},
      {"refinement_iterations",
       golden["config"]["refinement_iterations"].get<int64_t>()},
      {"batch_size", golden["config"]["batch_size"].get<int64_t>()},
  });
  auto node =
      std::make_shared<litert_pose_estimation::PoseEstimationNode>(options);
  auto client = std::make_shared<rclcpp::Node>("golden_test_client");

  // The frame, as a camera driver would publish it.
  const pt::Image rgb = pt::LoadPng(Golden(golden["image"]["rgb"]), 3);
  const perception::Tensor depth = pt::LoadNpy(Golden(golden["image"]["depth"]));
  std_msgs::msg::Header header;
  header.frame_id = "orbbec_camera/sensor_link";
  header.stamp = client->now();
  sensor_msgs::msg::Image rgb_msg;
  rgb_msg.header = header;
  rgb_msg.height = rgb.height;
  rgb_msg.width = rgb.width;
  rgb_msg.encoding = "rgb8";
  rgb_msg.step = rgb.width * 3;
  rgb_msg.data = rgb.pixels;
  sensor_msgs::msg::Image depth_msg;
  depth_msg.header = header;
  depth_msg.height = rgb.height;
  depth_msg.width = rgb.width;
  depth_msg.encoding = "32FC1";
  depth_msg.step = rgb.width * 4;
  depth_msg.data.resize(depth.size() * 4);
  std::memcpy(depth_msg.data.data(), depth.data(), depth_msg.data.size());
  sensor_msgs::msg::CameraInfo info;
  info.header = header;
  info.height = rgb.height;
  info.width = rgb.width;
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      info.k[r * 3 + c] = golden["camera_matrix"][r][c].get<double>();
    }
  }

  // Reliable, so that the large images aren't dropped (the node's best-effort
  // subscriptions accept reliable publishers).
  const auto qos = rclcpp::QoS(2).reliable();
  auto rgb_pub = client->create_publisher<sensor_msgs::msg::Image>(
      "rgb/image_raw", qos);
  auto depth_pub = client->create_publisher<sensor_msgs::msg::Image>(
      "depth/image_raw", qos);
  auto info_pub = client->create_publisher<sensor_msgs::msg::CameraInfo>(
      "rgb/camera_info", qos);
  std::promise<vision_msgs::msg::Detection3DArray> received;
  auto sub = client->create_subscription<vision_msgs::msg::Detection3DArray>(
      "litert_pose_estimation/detections", 10,
      [&](const vision_msgs::msg::Detection3DArray& msg) {
        received.set_value(msg);
      });
  auto estimate = client->create_client<std_srvs::srv::Trigger>(
      "litert_pose_estimation/estimate");

  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),
                                                    3);
  executor.add_node(node);
  executor.add_node(client);
  std::thread spinner([&] { executor.spin(); });

  // Publish the frame until the node has it, then estimate through the
  // ~/estimate service, as a client application would.
  ASSERT_TRUE(estimate->wait_for_service(30s));
  std::shared_ptr<std_srvs::srv::Trigger::Response> result;
  for (int attempt = 0; attempt < 50 && !(result && result->success);
       ++attempt) {
    rgb_pub->publish(rgb_msg);
    depth_pub->publish(depth_msg);
    info_pub->publish(info);
    std::this_thread::sleep_for(200ms);
    auto response = estimate->async_send_request(
        std::make_shared<std_srvs::srv::Trigger::Request>());
    ASSERT_EQ(response.wait_for(1800s), std::future_status::ready);
    result = response.get();
  }
  ASSERT_TRUE(result && result->success)
      << (result ? result->message : "no response");
  std::printf("%s\n", result->message.c_str());
  auto future = received.get_future();
  ASSERT_EQ(future.wait_for(30s), std::future_status::ready);
  const vision_msgs::msg::Detection3DArray detections = future.get();

  executor.cancel();
  spinner.join();

  ASSERT_EQ(detections.detections.size(), golden["poses"].size());
  EXPECT_EQ(detections.header.frame_id, header.frame_id);
  for (size_t i = 0; i < detections.detections.size(); ++i) {
    const auto& result = detections.detections[i].results.at(0);
    EXPECT_EQ(result.hypothesis.class_id, "raw_stock_2x3x5");
    const auto& want = golden["poses"][i];
    const auto& p = result.pose.pose.position;
    const double dt = std::hypot(p.x - want["translation"][0].get<double>(),
                                 p.y - want["translation"][1].get<double>(),
                                 p.z - want["translation"][2].get<double>());
    EXPECT_LT(dt, 1e-3) << "translation differs by " << dt * 1000 << " mm";
    EXPECT_NEAR(result.hypothesis.score, want["score"].get<double>(), 0.05);
    // Rotation, modulo the box's half-turn symmetries: the rotated box axes
    // must be parallel (up to sign) to the service's.
    const auto& q = result.pose.pose.orientation;
    const double r[3][3] = {
        {1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w),
         2 * (q.x * q.z + q.y * q.w)},
        {2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z),
         2 * (q.y * q.z - q.x * q.w)},
        {2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w),
         1 - 2 * (q.x * q.x + q.y * q.y)}};
    for (int axis = 0; axis < 3; ++axis) {
      double dot = 0;
      for (int k = 0; k < 3; ++k) {
        dot += r[k][axis] * want["rotation"][k][axis].get<double>();
      }
      EXPECT_GT(std::fabs(dot), std::cos(0.5 * M_PI / 180))
          << "object axis " << axis << " differs";
    }
  }
}

}  // namespace
