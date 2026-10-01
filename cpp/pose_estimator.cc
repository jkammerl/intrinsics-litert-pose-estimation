#include "pose_estimator.h"

#include <chrono>
#include <stdexcept>

namespace perception {
namespace {

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                       start)
      .count();
}

}  // namespace

PoseEstimator::PoseEstimator(const std::string& models_dir,
                             const std::string& cad_obj,
                             const PoseEstimatorConfig& config,
                             const ModelOptions& model_options)
    : config_(config), mesh_(foundationpose::ParseObj(cad_obj)) {
  if (mesh_.vertices.empty()) {
    throw std::invalid_argument("The CAD model has no vertices.");
  }
  segmentation_ = std::make_unique<Model>(models_dir + "/rfdetr_seg.tflite",
                                          model_options);
  foundationpose_ = std::make_unique<foundationpose::Estimator>(
      models_dir, config.batch_size, model_options);
}

std::vector<ObjectPose> PoseEstimator::Estimate(
    const uint8_t* rgb, const float* depth, int height, int width,
    const std::array<double, 9>& k) {
  auto start = std::chrono::steady_clock::now();
  const rfdetr::Preprocessed pre = rfdetr::Preprocess(rgb, height, width);
  std::vector<rfdetr::Detection> detections = rfdetr::Postprocess(
      pre, segmentation_->Run({{"input", &pre.input}}),
      config_.confidence_threshold, config_.visibility_threshold);
  timing_.segmentation = SecondsSince(start);

  // The service passes the intrinsics to FoundationPose as float32.
  const foundationpose::Intrinsics intrinsics{
      static_cast<float>(k[0]), static_cast<float>(k[4]),
      static_cast<float>(k[2]), static_cast<float>(k[5])};
  const foundationpose::RgbdImage image{rgb, depth, height, width};
  start = std::chrono::steady_clock::now();
  std::vector<ObjectPose> poses;
  for (rfdetr::Detection& detection : detections) {
    const foundationpose::Result result = foundationpose_->Estimate(
        image, detection.mask.data(), intrinsics, mesh_,
        config_.refinement_iterations);
    poses.push_back({std::move(detection), result.pose, result.score});
  }
  timing_.pose_estimation = SecondsSince(start);
  return poses;
}

std::vector<uint8_t> ToRgb(const uint8_t* pixels, int height, int width,
                           int channels) {
  const size_t n = static_cast<size_t>(height) * width;
  std::vector<uint8_t> rgb(n * 3);
  for (size_t i = 0; i < n; ++i) {
    for (int c = 0; c < 3; ++c) {
      rgb[i * 3 + c] = pixels[i * channels + (channels == 1 ? 0 : c)];
    }
  }
  return rgb;
}

}  // namespace perception
