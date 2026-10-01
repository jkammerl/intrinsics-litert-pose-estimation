// ROS 2 node for 6D pose estimation of the OMTS raw stock workpiece with
// LiteRT. It runs the same pipeline as Intrinsic's IOC pose estimator service
// (RF-DETR segmentation, then FoundationPose), on the GPU through LiteRT's
// GPU accelerator, falling back to the CPU automatically.

#ifndef LITERT_POSE_ESTIMATION__POSE_ESTIMATION_NODE_HPP_
#define LITERT_POSE_ESTIMATION__POSE_ESTIMATION_NODE_HPP_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "message_filters/subscriber.hpp"
#include "message_filters/sync_policies/approximate_time.hpp"
#include "message_filters/synchronizer.hpp"
#include "pose_estimator.h"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "vision_msgs/msg/detection3_d_array.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

namespace litert_pose_estimation {

class PoseEstimationNode : public rclcpp::Node {
 public:
  explicit PoseEstimationNode(
      const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
  ~PoseEstimationNode() override;

  // Estimates the poses in the latest synchronized RGB-D frame (blocking).
  // Returns false and sets `message` if there is no frame yet.
  bool EstimateLatest(std::string& message);

 private:
  using Image = sensor_msgs::msg::Image;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  using SyncPolicy =
      message_filters::sync_policies::ApproximateTime<Image, Image, CameraInfo>;

  struct Frame {
    Image::ConstSharedPtr rgb, depth;
    CameraInfo::ConstSharedPtr info;
  };

  void OnFrame(const Image::ConstSharedPtr& rgb,
               const Image::ConstSharedPtr& depth,
               const CameraInfo::ConstSharedPtr& info);
  void OnEstimate(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                  std::shared_ptr<std_srvs::srv::Trigger::Response> response);
  void Worker();
  std::string Process(const Frame& frame);
  void LogAccelerators();

  // Parameters.
  std::string object_id_;
  bool continuous_ = false;

  std::unique_ptr<perception::PoseEstimator> estimator_;
  std::array<float, 3> mesh_size_{};  // Bounding box size, for markers.
  std::array<float, 3> mesh_center_{};

  message_filters::Subscriber<Image> rgb_sub_, depth_sub_;
  message_filters::Subscriber<CameraInfo> info_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::CallbackGroup::SharedPtr service_group_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr estimate_srv_;
  rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr
      detections_pub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr poses_pub_;
  rclcpp::Publisher<Image>::SharedPtr mask_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr
      markers_pub_;

  // The latest frame, and the continuous-mode worker.
  std::mutex mutex_;
  std::condition_variable new_frame_;
  Frame latest_;
  bool has_new_frame_ = false;
  std::mutex estimate_mutex_;  // One estimation at a time.
  std::atomic<bool> stop_{false};
  std::thread worker_;
};

}  // namespace litert_pose_estimation

#endif  // LITERT_POSE_ESTIMATION__POSE_ESTIMATION_NODE_HPP_
