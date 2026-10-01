// The raw stock pose estimation pipeline of the IOC pose estimator service
// (intrinsic-core's intrinsic_perception/.../ioc_pose_estimator, as configured
// by OMTS), running every network with LiteRT:
//
// 1. Segmentation: RF-DETR finds the workpiece instances in the RGB image;
//    detections with score <= confidence_threshold or visibility <=
//    visibility_threshold are dropped (the service's segmentation model).
// 2. Pose estimation: FoundationPose estimates each instance's pose from the
//    RGB-D image, its mask, the camera intrinsics and the CAD model, refining
//    280 candidates `refinement_iterations` times in chunks of `batch_size`
//    (the service's FoundationPose model).
//
// Like the service, the result is one pose per detected instance, in the
// camera frame, with FoundationPose's score.

#ifndef PERCEPTION_LITERT_POSE_ESTIMATOR_H_
#define PERCEPTION_LITERT_POSE_ESTIMATOR_H_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "foundationpose.h"
#include "litert_model.h"
#include "rfdetr.h"

namespace perception {

// Defaults are OMTS's: the raw stock pose estimator is registered with
// confidence and visibility thresholds of 0.6 and 6 refinement iterations,
// and the service is configured with a batch size of 128.
struct PoseEstimatorConfig {
  float confidence_threshold = 0.6f;
  float visibility_threshold = 0.6f;
  int refinement_iterations = 6;
  int batch_size = 128;
};

struct ObjectPose {
  rfdetr::Detection detection;   // Box, score, visibility and mask.
  foundationpose::Mat4 pose;     // camera_t_target; translation in meters.
  float pose_score = 0;          // FoundationPose's score (higher is better).
};

// Timing of the last Estimate() call, in seconds.
struct PoseEstimatorTiming {
  double segmentation = 0;
  double pose_estimation = 0;
};

class PoseEstimator {
 public:
  // `models_dir` contains rfdetr_seg.tflite and the FoundationPose models
  // (see foundationpose::Estimator). `cad_obj` is the workpiece's mesh as
  // Wavefront OBJ text, in meters.
  PoseEstimator(const std::string& models_dir, const std::string& cad_obj,
                const PoseEstimatorConfig& config,
                const ModelOptions& model_options);

  // Estimates the poses of all workpiece instances. `rgb` is row-major RGB,
  // 3 bytes per pixel; `depth` is in meters (non-finite or <= 0 where
  // invalid), with the same size; `k` is the RGB camera's 3x3 intrinsic
  // matrix, row-major. The depth must be registered to the RGB image.
  std::vector<ObjectPose> Estimate(const uint8_t* rgb, const float* depth,
                                   int height, int width,
                                   const std::array<double, 9>& k);

  const PoseEstimatorTiming& timing() const { return timing_; }
  const Model& segmentation_model() const { return *segmentation_; }
  const foundationpose::Estimator& foundationpose() const {
    return *foundationpose_;
  }

 private:
  PoseEstimatorConfig config_;
  foundationpose::Mesh mesh_;
  std::unique_ptr<Model> segmentation_;
  std::unique_ptr<foundationpose::Estimator> foundationpose_;
  PoseEstimatorTiming timing_;
};

// Converts an image with 1 (gray), 3 (RGB) or 4 (RGBA) channels to RGB, like
// the service's _extract_rgb_depth_and_intrinsics.
std::vector<uint8_t> ToRgb(const uint8_t* pixels, int height, int width,
                           int channels);

}  // namespace perception

#endif  // PERCEPTION_LITERT_POSE_ESTIMATOR_H_
