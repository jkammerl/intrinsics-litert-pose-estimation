#include "litert_pose_estimation/pose_estimation_node.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

#include "rclcpp_components/register_node_macro.hpp"

namespace litert_pose_estimation {
namespace {

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("Can't read " + path);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

// Converts a sensor_msgs/Image to packed RGB, like the IOC pose estimator
// service converts intensity images (gray is repeated, alpha dropped).
std::vector<uint8_t> ToRgb(const sensor_msgs::msg::Image& msg) {
  int channels = 0;
  bool bgr = false;
  if (msg.encoding == "rgb8") {
    channels = 3;
  } else if (msg.encoding == "bgr8") {
    channels = 3;
    bgr = true;
  } else if (msg.encoding == "rgba8") {
    channels = 4;
  } else if (msg.encoding == "bgra8") {
    channels = 4;
    bgr = true;
  } else if (msg.encoding == "mono8") {
    channels = 1;
  } else {
    throw std::runtime_error("Unsupported color encoding " + msg.encoding);
  }
  std::vector<uint8_t> packed(size_t(msg.height) * msg.width * channels);
  for (uint32_t y = 0; y < msg.height; ++y) {
    std::memcpy(&packed[size_t(y) * msg.width * channels],
                &msg.data[size_t(y) * msg.step], size_t(msg.width) * channels);
  }
  std::vector<uint8_t> rgb =
      perception::ToRgb(packed.data(), msg.height, msg.width, channels);
  if (bgr) {
    for (size_t i = 0; i < rgb.size(); i += 3) std::swap(rgb[i], rgb[i + 2]);
  }
  return rgb;
}

// Converts a depth image (32FC1 in meters, or 16UC1 in millimeters) to
// meters; invalid pixels are 0.
std::vector<float> ToMeters(const sensor_msgs::msg::Image& msg) {
  std::vector<float> depth(size_t(msg.height) * msg.width);
  if (msg.is_bigendian) throw std::runtime_error("Big-endian depth image");
  for (uint32_t y = 0; y < msg.height; ++y) {
    const uint8_t* row = &msg.data[size_t(y) * msg.step];
    for (uint32_t x = 0; x < msg.width; ++x) {
      float d;
      if (msg.encoding == "32FC1") {
        std::memcpy(&d, row + x * 4, 4);
      } else if (msg.encoding == "16UC1") {
        uint16_t mm;
        std::memcpy(&mm, row + x * 2, 2);
        d = mm * 0.001f;
      } else {
        throw std::runtime_error("Unsupported depth encoding " + msg.encoding);
      }
      depth[size_t(y) * msg.width + x] = d;
    }
  }
  return depth;
}

geometry_msgs::msg::Pose ToPose(const perception::foundationpose::Mat4& m) {
  geometry_msgs::msg::Pose pose;
  pose.position.x = m(0, 3);
  pose.position.y = m(1, 3);
  pose.position.z = m(2, 3);
  // Rotation matrix to quaternion (Shepperd's method).
  const double r00 = m(0, 0), r11 = m(1, 1), r22 = m(2, 2);
  const double trace = r00 + r11 + r22;
  double w, x, y, z;
  if (trace > 0) {
    const double s = std::sqrt(trace + 1.0) * 2;
    w = 0.25 * s;
    x = (m(2, 1) - m(1, 2)) / s;
    y = (m(0, 2) - m(2, 0)) / s;
    z = (m(1, 0) - m(0, 1)) / s;
  } else if (r00 > r11 && r00 > r22) {
    const double s = std::sqrt(1.0 + r00 - r11 - r22) * 2;
    w = (m(2, 1) - m(1, 2)) / s;
    x = 0.25 * s;
    y = (m(0, 1) + m(1, 0)) / s;
    z = (m(0, 2) + m(2, 0)) / s;
  } else if (r11 > r22) {
    const double s = std::sqrt(1.0 + r11 - r00 - r22) * 2;
    w = (m(0, 2) - m(2, 0)) / s;
    x = (m(0, 1) + m(1, 0)) / s;
    y = 0.25 * s;
    z = (m(1, 2) + m(2, 1)) / s;
  } else {
    const double s = std::sqrt(1.0 + r22 - r00 - r11) * 2;
    w = (m(1, 0) - m(0, 1)) / s;
    x = (m(0, 2) + m(2, 0)) / s;
    y = (m(1, 2) + m(2, 1)) / s;
    z = 0.25 * s;
  }
  const double n = std::sqrt(w * w + x * x + y * y + z * z);
  pose.orientation.w = w / n;
  pose.orientation.x = x / n;
  pose.orientation.y = y / n;
  pose.orientation.z = z / n;
  return pose;
}

const char* Name(perception::Accelerator a) {
  return a == perception::Accelerator::kGpu ? "GPU" : "CPU";
}

}  // namespace

PoseEstimationNode::PoseEstimationNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("litert_pose_estimation", options) {
  const std::string models_dir = declare_parameter<std::string>("models_dir");
  const std::string cad_obj = declare_parameter<std::string>("cad_obj");
  object_id_ =
      declare_parameter<std::string>("object_id", "raw_stock_2x3x5");
  continuous_ = declare_parameter<bool>("continuous", false);

  perception::PoseEstimatorConfig config;
  config.confidence_threshold = static_cast<float>(
      declare_parameter<double>("confidence_threshold", 0.6));
  config.visibility_threshold = static_cast<float>(
      declare_parameter<double>("visibility_threshold", 0.6));
  config.refinement_iterations = static_cast<int>(
      declare_parameter<int64_t>("refinement_iterations", 6));
  config.batch_size =
      static_cast<int>(declare_parameter<int64_t>("batch_size", 128));

  perception::ModelOptions model_options;
  model_options.accelerator = perception::ParseAccelerator(
      declare_parameter<std::string>("accelerator", "auto"));
  model_options.gpu_fp16 = declare_parameter<bool>("gpu_fp16", false);
  model_options.cpu_fallback = declare_parameter<bool>("cpu_fallback", false);
  model_options.runtime_library_dir = declare_parameter<std::string>(
      "litert_library_dir", PERCEPTION_LITERT_LIBRARY_DIR);

  const std::string obj = ReadFile(cad_obj);
  RCLCPP_INFO(get_logger(),
              "Compiling the LiteRT models in %s (accelerator: %s) ...",
              models_dir.c_str(),
              get_parameter("accelerator").as_string().c_str());
  estimator_ = std::make_unique<perception::PoseEstimator>(
      models_dir, obj, config, model_options);
  LogAccelerators();

  const perception::foundationpose::Mesh mesh =
      perception::foundationpose::ParseObj(obj);
  std::array<float, 3> lo = mesh.vertices[0], hi = mesh.vertices[0];
  for (const auto& v : mesh.vertices) {
    for (int i = 0; i < 3; ++i) {
      lo[i] = std::min(lo[i], v[i]);
      hi[i] = std::max(hi[i], v[i]);
    }
  }
  for (int i = 0; i < 3; ++i) mesh_size_[i] = hi[i] - lo[i];
  mesh_center_ = perception::foundationpose::MeshCenter(mesh);

  const rclcpp::QoS qos = rclcpp::SensorDataQoS();
  rgb_sub_.subscribe(this, "rgb/image_raw", qos);
  depth_sub_.subscribe(this, "depth/image_raw", qos);
  info_sub_.subscribe(this, "rgb/camera_info", qos);
  sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(static_cast<uint32_t>(
          declare_parameter<int64_t>("queue_size", 5))),
      rgb_sub_, depth_sub_, info_sub_);
  sync_->setMaxIntervalDuration(
      rclcpp::Duration::from_seconds(declare_parameter<double>("sync_slop", 0.1)));
  sync_->registerCallback(&PoseEstimationNode::OnFrame, this);
  // Log the first message of each input, to diagnose topics that never
  // synchronize (e.g. remapping or timestamp problems).
  rgb_sub_.registerCallback([this](const Image::ConstSharedPtr& m) {
    RCLCPP_INFO_ONCE(get_logger(), "Receiving color images (%s, %ux%u)",
                     m->encoding.c_str(), m->width, m->height);
  });
  depth_sub_.registerCallback([this](const Image::ConstSharedPtr& m) {
    RCLCPP_INFO_ONCE(get_logger(), "Receiving depth images (%s, %ux%u)",
                     m->encoding.c_str(), m->width, m->height);
  });
  info_sub_.registerCallback([this](const CameraInfo::ConstSharedPtr&) {
    RCLCPP_INFO_ONCE(get_logger(), "Receiving camera info");
  });

  detections_pub_ =
      create_publisher<vision_msgs::msg::Detection3DArray>("~/detections", 10);
  poses_pub_ = create_publisher<geometry_msgs::msg::PoseArray>("~/poses", 10);
  mask_pub_ = create_publisher<Image>("~/segmentation", 10);
  markers_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("~/markers", 10);
  // Its own callback group, so that frames keep arriving while it runs.
  service_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
  estimate_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/estimate",
      std::bind(&PoseEstimationNode::OnEstimate, this, std::placeholders::_1,
                std::placeholders::_2),
      rclcpp::ServicesQoS(), service_group_);
  worker_ = std::thread(&PoseEstimationNode::Worker, this);
  RCLCPP_INFO(get_logger(), "Ready (%s).",
              continuous_ ? "estimating every frame"
                          : "call ~/estimate to estimate the latest frame");
}

PoseEstimationNode::~PoseEstimationNode() {
  stop_ = true;
  new_frame_.notify_all();
  if (worker_.joinable()) worker_.join();
}

void PoseEstimationNode::LogAccelerators() {
  auto log = [&](const char* what, const perception::Model& model) {
    if (!model.fallback_reason().empty()) {
      RCLCPP_WARN(get_logger(),
                  "%s: LiteRT GPU accelerator unavailable, running on the CPU "
                  "(%s)",
                  what, model.fallback_reason().c_str());
    }
    RCLCPP_INFO(get_logger(), "%s runs on the %s%s.", what,
                Name(model.accelerator()),
                model.accelerator() == perception::Accelerator::kGpu &&
                        !model.fully_accelerated()
                    ? " (unsupported ops on the CPU)"
                    : "");
  };
  log("RF-DETR segmentation", estimator_->segmentation_model());
  log("FoundationPose refiner", estimator_->foundationpose().refiner());
  for (const auto& [batch, scorer] : estimator_->foundationpose().scorers()) {
    const std::string what =
        "FoundationPose scorer (batch " + std::to_string(batch) + ")";
    log(what.c_str(), *scorer);
  }
}

void PoseEstimationNode::OnFrame(const Image::ConstSharedPtr& rgb,
                                 const Image::ConstSharedPtr& depth,
                                 const CameraInfo::ConstSharedPtr& info) {
  RCLCPP_INFO_ONCE(get_logger(), "Receiving synchronized RGB-D frames");
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_ = {rgb, depth, info};
    has_new_frame_ = true;
  }
  if (continuous_) new_frame_.notify_one();
}

void PoseEstimationNode::Worker() {
  while (!stop_) {
    Frame frame;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      new_frame_.wait(lock, [&] { return stop_ || (continuous_ && has_new_frame_); });
      if (stop_) return;
      frame = latest_;
      has_new_frame_ = false;
    }
    try {
      RCLCPP_INFO(get_logger(), "%s", Process(frame).c_str());
    } catch (const std::exception& e) {
      RCLCPP_ERROR(get_logger(), "Pose estimation failed: %s", e.what());
    }
  }
}

bool PoseEstimationNode::EstimateLatest(std::string& message) {
  Frame frame;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    frame = latest_;
  }
  if (!frame.rgb) {
    message = "No synchronized RGB, depth and camera info received yet.";
    return false;
  }
  message = Process(frame);
  return true;
}

void PoseEstimationNode::OnEstimate(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
  try {
    response->success = EstimateLatest(response->message);
  } catch (const std::exception& e) {
    response->success = false;
    response->message = e.what();
  }
}

std::string PoseEstimationNode::Process(const Frame& frame) {
  std::lock_guard<std::mutex> lock(estimate_mutex_);
  const Image& rgb_msg = *frame.rgb;
  const Image& depth_msg = *frame.depth;
  if (rgb_msg.width != depth_msg.width || rgb_msg.height != depth_msg.height) {
    throw std::runtime_error(
        "The depth image must be registered to the color image (same size).");
  }
  const std::vector<uint8_t> rgb = ToRgb(rgb_msg);
  const std::vector<float> depth = ToMeters(depth_msg);
  std::array<double, 9> k;
  std::copy(frame.info->k.begin(), frame.info->k.end(), k.begin());

  const std::vector<perception::ObjectPose> objects = estimator_->Estimate(
      rgb.data(), depth.data(), rgb_msg.height, rgb_msg.width, k);

  vision_msgs::msg::Detection3DArray detections;
  detections.header = rgb_msg.header;
  geometry_msgs::msg::PoseArray poses;
  poses.header = rgb_msg.header;
  visualization_msgs::msg::MarkerArray markers;
  visualization_msgs::msg::Marker clear;
  clear.header = rgb_msg.header;
  clear.action = visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(clear);
  Image mask;
  mask.header = rgb_msg.header;
  mask.height = rgb_msg.height;
  mask.width = rgb_msg.width;
  mask.encoding = "mono8";
  mask.step = mask.width;
  mask.data.assign(size_t(mask.height) * mask.width, 0);

  std::ostringstream summary;
  summary << objects.size() << " object(s) in "
          << estimator_->timing().segmentation +
                 estimator_->timing().pose_estimation
          << " s (segmentation " << estimator_->timing().segmentation
          << " s, pose " << estimator_->timing().pose_estimation << " s)";
  for (size_t i = 0; i < objects.size(); ++i) {
    const perception::ObjectPose& object = objects[i];
    const geometry_msgs::msg::Pose pose = ToPose(object.pose);
    vision_msgs::msg::Detection3D detection;
    detection.header = rgb_msg.header;
    detection.id = std::to_string(i);
    vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
    hypothesis.hypothesis.class_id = object_id_;
    hypothesis.hypothesis.score = object.pose_score;
    hypothesis.pose.pose = pose;
    detection.results.push_back(hypothesis);
    // FoundationPose's pose is at the center of the mesh's bounding box.
    detection.bbox.center = pose;
    detection.bbox.size.x = mesh_size_[0];
    detection.bbox.size.y = mesh_size_[1];
    detection.bbox.size.z = mesh_size_[2];
    detections.detections.push_back(detection);
    poses.poses.push_back(pose);

    visualization_msgs::msg::Marker box;
    box.header = rgb_msg.header;
    box.ns = "litert_pose_estimation";
    box.id = static_cast<int>(i);
    box.type = visualization_msgs::msg::Marker::CUBE;
    box.pose = pose;
    box.scale = detection.bbox.size;
    box.color.r = 0.1f;
    box.color.g = 0.8f;
    box.color.b = 0.2f;
    box.color.a = 0.6f;
    markers.markers.push_back(box);

    for (size_t p = 0; p < mask.data.size(); ++p) {
      if (object.detection.mask[p]) mask.data[p] = static_cast<uint8_t>(i + 1);
    }
    summary << "; " << object_id_ << " #" << i << ": position ["
            << pose.position.x << ", " << pose.position.y << ", "
            << pose.position.z << "] m, score " << object.pose_score
            << " (segmentation score " << object.detection.score << ")";
  }
  detections_pub_->publish(detections);
  poses_pub_->publish(poses);
  markers_pub_->publish(markers);
  mask_pub_->publish(mask);
  return summary.str();
}

}  // namespace litert_pose_estimation

RCLCPP_COMPONENTS_REGISTER_NODE(litert_pose_estimation::PoseEstimationNode)
