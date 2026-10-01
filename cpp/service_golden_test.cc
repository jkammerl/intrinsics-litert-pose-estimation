// Golden tests against the IOC pose estimator service: the inputs and outputs
// in testdata/service_golden were recorded from the deployed service (see
// docs/service_golden.md). The C++ pipeline must reproduce the service's
// segmentation and pose, and each FoundationPose stage must match the
// service's FoundationPose model (OMTS's model.py), whose intermediates are in
// testdata/service_golden/foundationpose.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "foundationpose.h"
#include "gtest/gtest.h"
#include "litert_model.h"
#include "pose_estimator.h"
#include "rfdetr.h"
#include "test_util.h"

namespace perception {
namespace {

using foundationpose::Mat4;
using testing::LoadJson;
using testing::LoadNpy;
using testing::LoadPng;
using testing::Path;

std::string Golden(const std::string& name) {
  return Path("testdata/service_golden/" + name);
}

std::string ReadFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

struct GoldenCase {
  nlohmann::json json = LoadJson(Golden("case.json"));
  testing::Image rgb = LoadPng(Golden(json["image"]["rgb"]), 3);
  Tensor depth = LoadNpy(Golden(json["image"]["depth"]));
  std::string obj = ReadFile(Golden(json["cad_obj"]));
  int height = json["image"]["height"];
  int width = json["image"]["width"];

  std::array<double, 9> K() const {
    std::array<double, 9> k;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) k[r * 3 + c] = json["camera_matrix"][r][c];
    }
    return k;
  }
  foundationpose::Intrinsics Intrinsics() const {
    const auto k = K();
    return {float(k[0]), float(k[4]), float(k[2]), float(k[5])};
  }
  std::vector<uint8_t> Mask(int i) const {
    testing::Image m =
        LoadPng(Golden(json["segmentation"][i]["mask"]), 1);
    return m.pixels;
  }
};

const GoldenCase& Case() {
  static const GoldenCase* golden = new GoldenCase();
  return *golden;
}

ModelOptions TestModelOptions() {
  ModelOptions options;
  const char* accelerator = std::getenv("PERCEPTION_ACCELERATOR");
  options.accelerator =
      ParseAccelerator(accelerator != nullptr ? accelerator : "cpu");
  options.runtime_library_dir = LITERT_LIBRARY_DIR;
  return options;
}

// The rotation angle between two rotations, in degrees, modulo the raw
// stock's symmetries: a 2 x 3 x 5 inch box looks the same after a half turn
// about any of its axes, so FoundationPose's candidates that differ by such a
// turn get the same score up to rounding, and either one is a correct result.
double SymmetricAngleDeg(const Mat4& a, const Mat4& b) {
  double best = 180.0;
  // Identity and the half turns about the object's x, y and z axes.
  const int signs[4][3] = {{1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1}};
  for (const auto& s : signs) {
    // trace(a^T b diag(s)).
    double trace = 0;
    for (int i = 0; i < 3; ++i) {
      for (int k = 0; k < 3; ++k) trace += double(a(k, i)) * b(k, i) * s[i];
    }
    best = std::min(best, std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0)) *
                              180.0 / M_PI);
  }
  return best;
}

// The average distance from each point of `points` transformed by `a` to the
// closest point transformed by `b` (ADD-S), in meters.
double AddS(const Mat4& a, const Mat4& b,
            const std::vector<std::array<float, 3>>& points) {
  auto apply = [](const Mat4& m, const std::array<float, 3>& p) {
    std::array<double, 3> q;
    for (int r = 0; r < 3; ++r) {
      q[r] = double(m(r, 0)) * p[0] + double(m(r, 1)) * p[1] +
             double(m(r, 2)) * p[2] + m(r, 3);
    }
    return q;
  };
  double sum = 0;
  for (const auto& p : points) {
    const auto pa = apply(a, p);
    double nearest = 1e9;
    for (const auto& o : points) {
      const auto pb = apply(b, o);
      nearest = std::min(nearest, std::hypot(pa[0] - pb[0], pa[1] - pb[1],
                                             pa[2] - pb[2]));
    }
    sum += nearest;
  }
  return sum / points.size();
}

Mat4 GoldenPose(int i) {
  const nlohmann::json& p = Case().json["poses"][i];
  Mat4 m = Mat4::Identity();
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) m(r, c) = p["rotation"][r][c];
    m(r, 3) = p["translation"][r];
  }
  return m;
}

TEST(ServiceGolden, ParsesTheServicesMesh) {
  const foundationpose::Mesh mesh = foundationpose::ParseObj(Case().obj);
  EXPECT_FALSE(mesh.vertices.empty());
  EXPECT_FALSE(mesh.faces.empty());
  // The raw stock is 2 x 3 x 5 inches: its diagonal is 0.1566 m.
  EXPECT_NEAR(foundationpose::MeshDiameter(mesh), 0.15660, 1e-4);
}

TEST(ServiceGolden, SamplesTheServicesPoseCandidates) {
  const GoldenCase& g = Case();
  const std::vector<uint8_t> mask = g.Mask(0);
  std::vector<float> depth(g.depth.data(), g.depth.data() + g.depth.size());
  for (float& d : depth) {
    if (!std::isfinite(d)) d = 0;
  }
  const std::vector<Mat4> poses = foundationpose::SampleInitialPoses(
      mask.data(), depth.data(), g.height, g.width, g.Intrinsics());
  const Tensor want = LoadNpy(Golden("foundationpose/candidate_poses.npy"));
  ASSERT_EQ(poses.size() * 16, want.size());
  float max_diff = 0;
  for (size_t i = 0; i < poses.size(); ++i) {
    for (int j = 0; j < 16; ++j) {
      max_diff = std::max(max_diff, std::abs(poses[i].m[j] - want[i * 16 + j]));
    }
  }
  EXPECT_LT(max_diff, 1e-5) << "candidate poses differ by " << max_diff;
}

TEST(ServiceGolden, BuildsTheServicesNetworkInputs) {
  const GoldenCase& g = Case();
  const foundationpose::Mesh mesh = foundationpose::ParseObj(g.obj);
  const Tensor poses_npy = LoadNpy(Golden("foundationpose/first_poses.npy"));
  std::vector<Mat4> poses(poses_npy.size() / 16);
  for (size_t i = 0; i < poses.size(); ++i) {
    std::copy(poses_npy.data() + i * 16, poses_npy.data() + (i + 1) * 16,
              poses[i].m.begin());
  }
  const foundationpose::RgbdImage image{g.rgb.pixels.data(), g.depth.data(),
                                        g.height, g.width};
  const foundationpose::InputBuilder builder(image, g.Intrinsics(), mesh);
  const int n = static_cast<int>(poses.size());
  Tensor in1({n, 160, 160, 6}), in2({n, 160, 160, 6});
  builder.Build(poses, in1, in2);
  const Tensor want1 = LoadNpy(Golden("foundationpose/first_in1.npy"));
  const Tensor want2 = LoadNpy(Golden("foundationpose/first_in2.npy"));
  ASSERT_EQ(in1.size(), want1.size());

  // The observed crop uses the same OpenCV warps as the service.
  float max_diff2 = 0;
  for (size_t i = 0; i < in2.size(); ++i) {
    max_diff2 = std::max(max_diff2, std::abs(in2[i] - want2[i]));
  }
  EXPECT_LT(max_diff2, 1e-5) << "observed crops differ by " << max_diff2;

  // The rendering: same coverage, and the same points where both cover.
  size_t coverage_diff = 0, covered = 0;
  float max_diff1 = 0;
  for (size_t p = 0; p < in1.size() / 6; ++p) {
    const bool a = in1[p * 6] > 0, b = want1[p * 6] > 0;
    covered += b;
    if (a != b) {
      ++coverage_diff;
      continue;
    }
    for (int c = 0; c < 6; ++c) {
      max_diff1 = std::max(max_diff1, std::abs(in1[p * 6 + c] - want1[p * 6 + c]));
    }
  }
  EXPECT_GT(covered, 0u);
  EXPECT_EQ(coverage_diff, 0u) << "rendered coverage differs in "
                               << coverage_diff << " of " << covered
                               << " pixels";
  EXPECT_LT(max_diff1, 1e-4) << "rendered points differ by " << max_diff1;
}

TEST(ServiceGolden, RefinerMatchesTheServicesRefiner) {
  // The service's refiner outputs for the first 4 candidates of the first
  // chunk's first iteration, from the inputs recorded above.
  Model refiner(Path("models/foundationpose_refine.tflite"),
                TestModelOptions());
  const Tensor in1_npy = LoadNpy(Golden("foundationpose/first_in1.npy"));
  const Tensor in2_npy = LoadNpy(Golden("foundationpose/first_in2.npy"));
  const int batch = refiner.shape("input1")[0];
  Tensor in1({batch, 160, 160, 6}), in2({batch, 160, 160, 6});
  std::copy(in1_npy.data(), in1_npy.data() + in1_npy.size(), in1.data());
  std::copy(in2_npy.data(), in2_npy.data() + in2_npy.size(), in2.data());
  std::map<std::string, Tensor> out =
      refiner.Run({{"input1", &in1}, {"input2", &in2}});
  const Tensor want_t = LoadNpy(Golden("foundationpose/refine_translation_0.npy"));
  const Tensor want_r = LoadNpy(Golden("foundationpose/refine_rotation_0.npy"));
  const int n = in1_npy.shape()[0];
  float max_diff = 0;
  for (int i = 0; i < n * 3; ++i) {
    max_diff = std::max(max_diff, std::abs(out.at("output1")[i] - want_t[i]));
    max_diff = std::max(max_diff, std::abs(out.at("output2")[i] - want_r[i]));
  }
  EXPECT_LT(max_diff, 1e-3) << "refiner outputs differ by " << max_diff;
}

TEST(ServiceGolden, SegmentationMatchesTheService) {
  const GoldenCase& g = Case();
  Model model(Path("models/rfdetr_seg.tflite"), TestModelOptions());
  const rfdetr::Preprocessed pre =
      rfdetr::Preprocess(g.rgb.pixels.data(), g.height, g.width);
  const auto& config = g.json["config"];
  const std::vector<rfdetr::Detection> detections = rfdetr::Postprocess(
      pre, model.Run({{"input", &pre.input}}),
      config["confidence_threshold"], config["visibility_threshold"]);
  const auto& want = g.json["segmentation"];
  ASSERT_EQ(detections.size(), want.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NEAR(detections[i].box_xyxy[j], want[i]["box_xyxy"][j].get<float>(),
                  0.5f);
    }
    EXPECT_NEAR(detections[i].score, want[i]["score"].get<float>(), 2e-3);
    EXPECT_NEAR(detections[i].visibility, want[i]["visibility"].get<float>(),
                2e-3);
    const std::vector<uint8_t> want_mask = g.Mask(static_cast<int>(i));
    size_t both = 0, either = 0;
    for (size_t p = 0; p < want_mask.size(); ++p) {
      const bool a = detections[i].mask[p] != 0, b = want_mask[p] != 0;
      both += a && b;
      either += a || b;
    }
    EXPECT_GT(double(both) / either, 0.995) << "mask IoU";
  }
}

TEST(ServiceGolden, EndToEndMatchesTheService) {
  const GoldenCase& g = Case();
  PoseEstimatorConfig config;
  config.confidence_threshold = g.json["config"]["confidence_threshold"];
  config.visibility_threshold = g.json["config"]["visibility_threshold"];
  config.refinement_iterations = g.json["config"]["refinement_iterations"];
  config.batch_size = g.json["config"]["batch_size"];
  PoseEstimator estimator(Path("models"), g.obj, config, TestModelOptions());
  const std::vector<ObjectPose> poses = estimator.Estimate(
      g.rgb.pixels.data(), g.depth.data(), g.height, g.width, g.K());
  std::printf("segmentation %.2f s, pose estimation %.2f s (%s)\n",
              estimator.timing().segmentation,
              estimator.timing().pose_estimation,
              estimator.foundationpose().refiner().accelerator() ==
                      Accelerator::kGpu
                  ? "GPU"
                  : "CPU");
  ASSERT_EQ(poses.size(), g.json["poses"].size());
  for (size_t i = 0; i < poses.size(); ++i) {
    const Mat4 want = GoldenPose(static_cast<int>(i));
    double dt = 0;
    for (int r = 0; r < 3; ++r) {
      dt += std::pow(double(poses[i].pose(r, 3)) - want(r, 3), 2);
    }
    dt = std::sqrt(dt);
    const double angle = SymmetricAngleDeg(poses[i].pose, want);
    const double add_s =
        AddS(poses[i].pose, want, foundationpose::ParseObj(g.obj).vertices);
    const float want_score = g.json["poses"][i]["score"];
    std::printf("object %zu: translation %.3f mm, rotation %.3f deg (modulo "
                "the box's symmetries), ADD-S %.3f mm, score %.4f (service "
                "%.4f)\n",
                i, dt * 1000, angle, add_s * 1000, poses[i].pose_score,
                want_score);
    EXPECT_LT(dt, 1e-3) << "translation differs by " << dt * 1000 << " mm";
    EXPECT_LT(angle, 0.5) << "rotation differs by " << angle << " deg";
    EXPECT_LT(add_s, 1e-3) << "ADD-S " << add_s * 1000 << " mm";
    EXPECT_NEAR(poses[i].pose_score, want_score, 0.05f);
  }
}

}  // namespace
}  // namespace perception
