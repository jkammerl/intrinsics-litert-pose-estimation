// Tests the TFLite models against the ONNX models' outputs in testdata/.

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "litert_model.h"
#include "test_util.h"

namespace perception {
namespace {

using ::perception::testing::LoadJson;
using ::perception::testing::LoadNpy;
using ::perception::testing::Path;

const char* const kScenes[] = {"raw_stock_top", "raw_stock_tilted",
                               "raw_stock_side", "empty"};

// Largest absolute difference, and the largest absolute reference value.
struct Diff {
  float max_abs_diff = 0;
  float max_abs_ref = 0;
};
Diff Compare(const Tensor& got, const Tensor& want, size_t count) {
  Diff d;
  for (size_t i = 0; i < count; ++i) {
    d.max_abs_diff = std::max(d.max_abs_diff, std::abs(got[i] - want[i]));
    d.max_abs_ref = std::max(d.max_abs_ref, std::abs(want[i]));
  }
  return d;
}

void ExpectClose(const Tensor& got, const Tensor& want, float tolerance,
                 const std::string& what, size_t count = 0) {
  if (count == 0) {
    ASSERT_EQ(got.shape(), want.shape()) << what;
    count = want.size();
  }
  Diff d = Compare(got, want, count);
  EXPECT_LE(d.max_abs_diff, tolerance)
      << what << ": max |TFLite - ONNX| = " << d.max_abs_diff
      << " (max |ONNX| = " << d.max_abs_ref << ")";
}

class ModelTest : public ::testing::TestWithParam<std::string> {};

// Each model documents its tensors in its "io_spec" metadata.
TEST_P(ModelTest, IoSpecMatchesModel) {
  Model model(Path("models/" + GetParam()));
  std::string metadata = model.Metadata("io_spec");
  ASSERT_FALSE(metadata.empty());
  nlohmann::json spec = nlohmann::json::parse(metadata);
  EXPECT_FALSE(spec["summary"].get<std::string>().empty());
  EXPECT_EQ(spec["weights"], GetParam().find("_fp16") == std::string::npos
                                 ? "float32"
                                 : "float16");
  for (const char* kind : {"inputs", "outputs"}) {
    const auto& names = std::string(kind) == "inputs" ? model.input_names()
                                                      : model.output_names();
    ASSERT_EQ(spec[kind].size(), names.size()) << kind;
    for (const auto& tensor : spec[kind]) {
      const std::string name = tensor["name"];
      EXPECT_NE(std::find(names.begin(), names.end(), name), names.end())
          << name;
      EXPECT_EQ(tensor["shape"].get<std::vector<int32_t>>(), model.shape(name))
          << name;
      EXPECT_EQ(tensor["dtype"],
                model.is_float16(name) ? "float16" : "float32")
          << name;
      EXPECT_FALSE(tensor["content"].get<std::string>().empty()) << name;
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Models, ModelTest,
    ::testing::Values("rfdetr_seg.tflite", "rfdetr_seg_fp16.tflite",
                      "foundationpose_refine.tflite",
                      "foundationpose_refine_fp16.tflite",
                      "foundationpose_score.tflite",
                      "foundationpose_score_fp16.tflite"));

TEST(SegmentationModelTest, MatchesOnnx) {
  Model model(Path("models/rfdetr_seg.tflite"));
  for (const char* scene : kScenes) {
    SCOPED_TRACE(scene);
    const std::string dir =
        Path(std::string("testdata/scenes/") + scene + "/segmentation/");
    Tensor input = LoadNpy(dir + "input.npy");
    auto outputs = model.Run({{"input", &input}});
    ExpectClose(outputs["logits"], LoadNpy(dir + "logits.npy"), 2e-3,
                "logits");
    ExpectClose(outputs["visibility_logits"],
                LoadNpy(dir + "visibility_logits.npy"), 2e-3,
                "visibility_logits");
    ExpectClose(outputs["boxes"], LoadNpy(dir + "boxes.npy"), 1e-4, "boxes");
    ExpectClose(outputs["mask_logits"], LoadNpy(dir + "mask_logits.npy"), 2e-2,
                "mask_logits");
  }
}

// The refiner treats candidates independently: the saved candidates are
// padded with zeros to the model's batch.
void TestRefine(const std::string& model_file, float tolerance) {
  Model model(Path("models/" + model_file));
  const std::string dir = Path("testdata/networks/");
  Tensor saved1 = LoadNpy(dir + "refine_input1.npy");
  Tensor saved2 = LoadNpy(dir + "refine_input2.npy");
  Tensor input1(model.shape("input1")), input2(model.shape("input2"));
  std::copy_n(saved1.data(), saved1.size(), input1.data());
  std::copy_n(saved2.data(), saved2.size(), input2.data());
  auto outputs = model.Run({{"input1", &input1}, {"input2", &input2}});
  Tensor translation = LoadNpy(dir + "refine_translation.npy");
  Tensor rotation = LoadNpy(dir + "refine_rotation.npy");
  ExpectClose(outputs["output1"], translation, tolerance, "translation",
              translation.size());
  ExpectClose(outputs["output2"], rotation, tolerance, "rotation",
              rotation.size());
}

TEST(RefineModelTest, MatchesOnnx) {
  TestRefine("foundationpose_refine.tflite", 1e-3);
}

TEST(RefineModelTest, Float16WeightsMatchOnnx) {
  TestRefine("foundationpose_refine_fp16.tflite", 2e-2);
}

void TestScore(const std::string& model_file, float tolerance) {
  Model model(Path("models/" + model_file));
  const std::string dir = Path("testdata/networks/");
  Tensor input1 = LoadNpy(dir + "score_input1.npy");
  Tensor input2 = LoadNpy(dir + "score_input2.npy");
  auto outputs = model.Run({{"input1", &input1}, {"input2", &input2}});
  Tensor scores = LoadNpy(dir + "scores.npy");
  ExpectClose(outputs["output1"], scores, tolerance, "scores");
  // The best candidate is as good as ONNX's best. (Several candidates can
  // tie: the block is symmetric, so poses 180 degrees apart are equivalent.)
  const float* got = outputs["output1"].data();
  const size_t best = std::max_element(got, got + scores.size()) - got;
  const float best_onnx =
      *std::max_element(scores.data(), scores.data() + scores.size());
  EXPECT_NEAR(scores[best], best_onnx, tolerance);
}

TEST(ScoreModelTest, MatchesOnnx) {
  TestScore("foundationpose_score.tflite", 1e-2);
}

TEST(ScoreModelTest, Float16WeightsMatchOnnx) {
  TestScore("foundationpose_score_fp16.tflite", 0.5);
}

}  // namespace
}  // namespace perception
