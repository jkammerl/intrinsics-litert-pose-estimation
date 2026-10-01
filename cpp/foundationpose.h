// FoundationPose 6D pose estimation with LiteRT: a C++ port of OMTS's
// FoundationPose Triton model (third_party/foundationpose/model.py and
// foundationpose_numpy.py in intrinsic-omts), which the IOC pose estimator
// service calls. Every step follows the Python code, including its float32
// arithmetic, so that the results match the service's:
//
// 1. Guess the object's center from the mask and the depth, and sample 280
//    pose candidates around it (40 views x 7 in-plane rotations).
// 2. In chunks of `batch_size` candidates: refine them `iterations` times. For
//    each candidate, crop a 160x160 window around it from the RGB-D image,
//    render the mesh at the candidate pose into the same window, and let the
//    refiner network predict a pose update. Then score the chunk's candidates
//    against each other with the scorer network.
// 3. Return the best-scoring candidate, shifted to the mesh's bounding box
//    center.

#ifndef PERCEPTION_LITERT_FOUNDATIONPOSE_H_
#define PERCEPTION_LITERT_FOUNDATIONPOSE_H_

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "litert_model.h"

namespace perception::foundationpose {

inline constexpr int kCropSize = 160;  // Side of the networks' input crops.
inline constexpr int kNumViews = 40;   // Views of the rotation grid.
inline constexpr double kInplaneStepDeg = 60.0;
inline constexpr float kCropRatio = 1.2f;
inline constexpr float kRotationNormalizer = 0.34906585f;  // 20 degrees.

// A 4x4 float32 transform, row-major.
struct Mat4 {
  std::array<float, 16> m{};
  static Mat4 Identity();
  float& operator()(int r, int c) { return m[r * 4 + c]; }
  float operator()(int r, int c) const { return m[r * 4 + c]; }
};

struct Mesh {
  std::vector<std::array<float, 3>> vertices;
  std::vector<std::array<int32_t, 3>> faces;
};

// Parses a Wavefront OBJ like model.py's _load_obj_mesh: "v" lines, and "f"
// lines with 3 or 4 vertices (quads are split into two triangles). Returns an
// empty mesh if there are no vertices.
Mesh ParseObj(const std::string& text);

// The length of the mesh's bounding box diagonal, at least 0.05 m.
float MeshDiameter(const Mesh& mesh);
// The center of the mesh's bounding box.
std::array<float, 3> MeshCenter(const Mesh& mesh);

// Camera intrinsics, as float32 (model.py receives CAM_K as float32).
struct Intrinsics {
  float fx = 0, fy = 0, cx = 0, cy = 0;
};

// An RGB-D image: `rgb` is row-major, 3 bytes per pixel; `depth` in meters,
// one float per pixel (non-finite values are treated as invalid, like 0).
struct RgbdImage {
  const uint8_t* rgb = nullptr;
  const float* depth = nullptr;
  int height = 0, width = 0;
};

// The 280 pose candidates (object in camera) around the center guessed from
// the mask (height * width, nonzero inside) and the depth.
std::vector<Mat4> SampleInitialPoses(const uint8_t* mask, const float* depth,
                                     int height, int width,
                                     const Intrinsics& k);

// Per candidate, the 3x3 transform (row-major) from image pixels to the
// candidate's crop window.
std::vector<std::array<float, 9>> ComputeCropWindowTf(
    const std::vector<Mat4>& poses, const Intrinsics& k, float mesh_diameter);

// Applies the refiner's outputs to the poses (update_refined_poses).
void UpdateRefinedPoses(std::vector<Mat4>& poses, const float* translation,
                        const float* rotation, float mesh_diameter);

// pose * translation(mesh_center) (apply_mesh_center_offset).
Mat4 ApplyMeshCenterOffset(const Mat4& pose, const std::array<float, 3>& c);

// The refiner and scorer inputs for `poses`: input1 renders the mesh at each
// pose, input2 is the observed RGB-D image, both in each pose's crop window
// ([N, 160, 160, 6] each, written at candidate offset `first` of `in1` and
// `in2`, which have room for at least first + poses.size() candidates).
class InputBuilder {
 public:
  InputBuilder(const RgbdImage& image, const Intrinsics& k, const Mesh& mesh);
  void Build(const std::vector<Mat4>& poses, Tensor& in1, Tensor& in2,
             int first = 0) const;
  float mesh_diameter() const { return mesh_diameter_; }

 private:
  void Render(const Mat4& pose, const std::array<float, 9>& tf,
              float* out) const;
  Intrinsics k_;
  const Mesh& mesh_;
  float mesh_diameter_;
  int height_, width_;
  std::vector<float> rgb_float_;  // [H, W, 3] in [0, 1].
  std::vector<float> xyz_map_;    // [H, W, 3] in meters.
};

struct Result {
  Mat4 pose;  // Object in camera; translation in meters.
  float score = 0;
};

// The FoundationPose networks, compiled once for reuse.
class Estimator {
 public:
  // `models_dir` contains foundationpose_refine.tflite (batch 40) and the
  // scorers for the chunk sizes that `batch_size` produces from 280
  // candidates: foundationpose_score.tflite (280) or
  // foundationpose_score_b<N>.tflite.
  Estimator(const std::string& models_dir, int batch_size,
            const ModelOptions& options);

  // Estimates the pose of the object in `mask` (height * width, nonzero
  // inside), like one mask of model.py's execute().
  Result Estimate(const RgbdImage& image, const uint8_t* mask,
                  const Intrinsics& k, const Mesh& mesh, int iterations);

  const Model& refiner() const { return *refiner_; }
  const std::map<int, std::unique_ptr<Model>>& scorers() const {
    return scorers_;
  }

 private:
  int batch_size_;
  std::unique_ptr<Model> refiner_;
  std::map<int, std::unique_ptr<Model>> scorers_;  // By batch size.
};

}  // namespace perception::foundationpose

#endif  // PERCEPTION_LITERT_FOUNDATIONPOSE_H_
