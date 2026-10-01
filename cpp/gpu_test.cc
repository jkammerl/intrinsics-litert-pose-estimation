// Tests of LiteRT's GPU accelerator (ML Drift on WebGPU; Vulkan on Linux).
//
// The networks are run with Accelerator::kGpu, which also uses software
// Vulkan devices such as Mesa's llvmpipe, so these tests run on machines
// without a hardware GPU too (slowly). They are skipped if there is no Vulkan
// device at all. The tests compare the GPU's results with the IOC pose
// estimator service's (testdata/service_golden) and with the CPU's, and check
// when a network runs on the CPU instead (ModelOptions::cpu_fallback).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "gtest/gtest.h"
#include "litert_model.h"
#include "rfdetr.h"
#include "test_util.h"

namespace perception {
namespace {

using testing::LoadJson;
using testing::LoadNpy;
using testing::LoadPng;
using testing::Path;

std::string Golden(const std::string& name) {
  return Path("testdata/service_golden/" + name);
}

ModelOptions Gpu(bool cpu_fallback = false) {
  ModelOptions options;
  options.accelerator = Accelerator::kGpu;
  options.gpu_fp16 = false;  // float32, comparable with the CPU.
  options.runtime_library_dir = LITERT_LIBRARY_DIR;
  options.cpu_fallback = cpu_fallback;
  return options;
}

// Whether LiteRT's GPU accelerator has a Vulkan device here (a hardware one,
// or a software one such as llvmpipe).
bool HasVulkanDevice() {
  const std::string reason = HardwareGpuUnavailableReason();
  return reason.empty() || reason.find("software Vulkan") != std::string::npos;
}

// Compiles `model` for the GPU (see Gpu()), or returns nullptr if there is no
// Vulkan device.
std::unique_ptr<Model> GpuModel(const std::string& model,
                                bool cpu_fallback = false) {
  if (!HasVulkanDevice()) return nullptr;
  return std::make_unique<Model>(Path(model), Gpu(cpu_fallback));
}

void PrintPlacement(const char* name, const Model& model) {
  std::printf("%s runs on the %s%s%s\n", name,
              model.accelerator() == Accelerator::kGpu ? "GPU" : "CPU",
              model.fallback_reason().empty() ? "" : ": ",
              model.fallback_reason().c_str());
}

TEST(GpuAccelerator, RefinerOnTheGpuMatchesTheService) {
  std::unique_ptr<Model> refiner =
      GpuModel("models/foundationpose_refine.tflite");
  if (refiner == nullptr) GTEST_SKIP() << "no Vulkan device";
  ASSERT_EQ(refiner->accelerator(), Accelerator::kGpu);
  std::printf("fully accelerated on the GPU: %s\n",
              refiner->fully_accelerated() ? "yes" : "no");

  const Tensor in1_npy = LoadNpy(Golden("foundationpose/first_in1.npy"));
  const Tensor in2_npy = LoadNpy(Golden("foundationpose/first_in2.npy"));
  const int batch = refiner->shape("input1")[0];
  Tensor in1({batch, 160, 160, 6}), in2({batch, 160, 160, 6});
  std::copy(in1_npy.data(), in1_npy.data() + in1_npy.size(), in1.data());
  std::copy(in2_npy.data(), in2_npy.data() + in2_npy.size(), in2.data());
  std::map<std::string, Tensor> gpu =
      refiner->Run({{"input1", &in1}, {"input2", &in2}});

  Model cpu_refiner(Path("models/foundationpose_refine.tflite"));
  std::map<std::string, Tensor> cpu =
      cpu_refiner.Run({{"input1", &in1}, {"input2", &in2}});

  const Tensor want_t = LoadNpy(Golden("foundationpose/refine_translation_0.npy"));
  const Tensor want_r = LoadNpy(Golden("foundationpose/refine_rotation_0.npy"));
  const int n = in1_npy.shape()[0];
  float vs_service = 0, vs_cpu = 0;
  for (int i = 0; i < n * 3; ++i) {
    vs_service = std::max(vs_service, std::abs(gpu.at("output1")[i] - want_t[i]));
    vs_service = std::max(vs_service, std::abs(gpu.at("output2")[i] - want_r[i]));
    vs_cpu = std::max(vs_cpu, std::abs(gpu.at("output1")[i] - cpu.at("output1")[i]));
    vs_cpu = std::max(vs_cpu, std::abs(gpu.at("output2")[i] - cpu.at("output2")[i]));
  }
  std::printf("GPU vs service %.2e, GPU vs CPU %.2e\n", vs_service, vs_cpu);
  EXPECT_LT(vs_service, 1e-3);
  EXPECT_LT(vs_cpu, 1e-3);
}

TEST(GpuAccelerator, SegmentationOnTheGpuMatchesTheService) {
  std::unique_ptr<Model> model = GpuModel("models/rfdetr_seg.tflite");
  if (model == nullptr) GTEST_SKIP() << "no Vulkan device";
  ASSERT_EQ(model->accelerator(), Accelerator::kGpu);
  const nlohmann::json golden = LoadJson(Golden("case.json"));
  const testing::Image rgb = LoadPng(Golden(golden["image"]["rgb"]), 3);
  const rfdetr::Preprocessed pre =
      rfdetr::Preprocess(rgb.pixels.data(), rgb.height, rgb.width);
  const std::vector<rfdetr::Detection> detections = rfdetr::Postprocess(
      pre, model->Run({{"input", &pre.input}}),
      golden["config"]["confidence_threshold"],
      golden["config"]["visibility_threshold"]);
  const auto& want = golden["segmentation"];
  ASSERT_EQ(detections.size(), want.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    for (int j = 0; j < 4; ++j) {
      EXPECT_NEAR(detections[i].box_xyxy[j], want[i]["box_xyxy"][j].get<float>(),
                  1.0f);
    }
    EXPECT_NEAR(detections[i].score, want[i]["score"].get<float>(), 5e-3);
    const testing::Image want_mask = LoadPng(Golden(want[i]["mask"]), 1);
    size_t both = 0, either = 0;
    for (size_t p = 0; p < want_mask.pixels.size(); ++p) {
      const bool a = detections[i].mask[p] != 0, b = want_mask.pixels[p] != 0;
      both += a && b;
      either += a || b;
    }
    EXPECT_GT(double(both) / either, 0.99) << "mask IoU";
  }
}

TEST(GpuAccelerator, ScorerGivesTheCpusScores) {
  // The scorer for the last chunk of the service's batches (24 of the 280
  // candidates), on the first 24 candidates of testdata/networks. Some GPUs
  // compute it wrongly (Mesa's llvmpipe does); the GPU check at compilation
  // (ModelOptions::validate_gpu) must catch that, so that a model compiled
  // for the GPU never gives other results than the CPU: with cpu_fallback,
  // it then runs on the CPU.
  std::unique_ptr<Model> scorer = GpuModel(
      "models/foundationpose_score_b24.tflite", /*cpu_fallback=*/true);
  if (scorer == nullptr) GTEST_SKIP() << "no Vulkan device";
  PrintPlacement("scorer b24", *scorer);
  const Tensor in1_npy = LoadNpy(Path("testdata/networks/score_input1.npy"));
  const Tensor in2_npy = LoadNpy(Path("testdata/networks/score_input2.npy"));
  const int n = 24;
  const size_t size = size_t(n) * 160 * 160 * 6;
  Tensor in1({n, 160, 160, 6}), in2({n, 160, 160, 6});
  std::copy(in1_npy.data(), in1_npy.data() + size, in1.data());
  std::copy(in2_npy.data(), in2_npy.data() + size, in2.data());
  std::map<std::string, Tensor> out =
      scorer->Run({{"input1", &in1}, {"input2", &in2}});
  const Tensor& gpu = out.at("output1");
  Model cpu_scorer(Path("models/foundationpose_score_b24.tflite"));
  std::map<std::string, Tensor> cpu_out =
      cpu_scorer.Run({{"input1", &in1}, {"input2", &in2}});
  const Tensor& cpu = cpu_out.at("output1");
  float max_diff = 0;
  int gpu_best = 0, cpu_best = 0;
  for (int i = 0; i < n; ++i) {
    max_diff = std::max(max_diff, std::abs(gpu[i] - cpu[i]));
    if (gpu[i] > gpu[gpu_best]) gpu_best = i;
    if (cpu[i] > cpu[cpu_best]) cpu_best = i;
  }
  std::printf("scorer b24: vs CPU %.2e\n", max_diff);
  EXPECT_LT(max_diff, 1e-2);
  // The GPU's best candidate is as good as the CPU's. (Several candidates can
  // tie: the block is symmetric, so poses 180 degrees apart are equivalent.)
  EXPECT_NEAR(cpu[gpu_best], cpu[cpu_best], 1e-4);
}

// The scorer for batches of 128 candidates has a 157 MB input tensor
// (2 x 128 x 6 x 160 x 160 floats), more than many GPUs' (and WebGPU's
// default) maximum buffer size of 128 MB.
constexpr char kLargeScorer[] = "models/foundationpose_score_b128.tflite";

TEST(GpuAccelerator, AModelTheGpuCantRunIsAnErrorWithoutCpuFallback) {
  if (!HasVulkanDevice()) GTEST_SKIP() << "no Vulkan device";
  try {
    Model scorer(Path(kLargeScorer), Gpu());
    PrintPlacement("scorer b128", scorer);  // This GPU can run it.
    EXPECT_EQ(scorer.accelerator(), Accelerator::kGpu);
  } catch (const std::runtime_error& e) {
    std::printf("scorer b128: %s\n", e.what());
    EXPECT_NE(std::string(e.what()).find("can't run on this GPU"),
              std::string::npos);
    EXPECT_NE(std::string(e.what()).find("cpu_fallback"), std::string::npos);
  }
}

TEST(GpuAccelerator, AModelTheGpuCantRunRunsOnTheCpuWithCpuFallback) {
  std::unique_ptr<Model> scorer = GpuModel(kLargeScorer, /*cpu_fallback=*/true);
  if (scorer == nullptr) GTEST_SKIP() << "no Vulkan device";
  PrintPlacement("scorer b128", *scorer);
  if (scorer->accelerator() == Accelerator::kCpu) {
    EXPECT_NE(scorer->fallback_reason().find("can't run on this GPU"),
              std::string::npos);
  } else {
    EXPECT_TRUE(scorer->fallback_reason().empty());
  }
}

TEST(GpuAccelerator, AutoUsesTheCpuWithoutAHardwareGpu) {
  ModelOptions options = Gpu();
  options.accelerator = Accelerator::kAuto;
  Model model(Path("models/foundationpose_refine.tflite"), options);
  if (model.accelerator() == Accelerator::kGpu) {
    GTEST_SKIP() << "this machine has a hardware GPU";
  }
  EXPECT_FALSE(model.fallback_reason().empty());
  std::printf("auto uses the CPU: %s\n", model.fallback_reason().c_str());
}

TEST(GpuAccelerator, ParsesAcceleratorNames) {
  EXPECT_EQ(ParseAccelerator("auto"), Accelerator::kAuto);
  EXPECT_EQ(ParseAccelerator("gpu"), Accelerator::kGpu);
  EXPECT_EQ(ParseAccelerator("cpu"), Accelerator::kCpu);
  EXPECT_THROW(ParseAccelerator("tpu"), std::invalid_argument);
}

}  // namespace
}  // namespace perception
