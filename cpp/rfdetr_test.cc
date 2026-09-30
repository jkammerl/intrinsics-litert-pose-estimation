// Tests RF-DETR segmentation end to end, from the image to masks, against
// the released segmentation.onnx's results in testdata/.

#include "rfdetr.h"

#include <algorithm>
#include <cmath>
#include <string>

#include "gtest/gtest.h"
#include "test_util.h"

namespace perception::rfdetr {
namespace {

using ::perception::testing::LoadJson;
using ::perception::testing::LoadNpy;
using ::perception::testing::LoadPng;
using ::perception::testing::Path;

class RfDetrTest : public ::testing::TestWithParam<std::string> {
 protected:
  std::string Dir() const { return Path("testdata/scenes/" + GetParam()); }
};

TEST_P(RfDetrTest, PreprocessMatchesReference) {
  auto image = LoadPng(Dir() + "/rgb.png", 3);
  Preprocessed pre = Preprocess(image.pixels.data(), image.height, image.width);
  Tensor want = LoadNpy(Dir() + "/segmentation/input.npy");
  ASSERT_EQ(pre.input.shape(), want.shape());
  float max_diff = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(pre.input[i] - want[i]));
  }
  EXPECT_LT(max_diff, 1e-4);
  auto result = LoadJson(Dir() + "/segmentation/result.json");
  EXPECT_FLOAT_EQ(pre.scale, result["scale"].get<float>());
  EXPECT_EQ(pre.resized_height, result["resized_hw"][0]);
  EXPECT_EQ(pre.resized_width, result["resized_hw"][1]);
}

TEST_P(RfDetrTest, SegmentationMatchesOnnxModel) {
  static Model* model = new Model(Path("models/rfdetr_seg.tflite"));
  auto image = LoadPng(Dir() + "/rgb.png", 3);
  auto result = LoadJson(Dir() + "/segmentation/result.json");
  Preprocessed pre = Preprocess(image.pixels.data(), image.height, image.width);
  std::vector<Detection> detections =
      Postprocess(pre, model->Run({{"input", &pre.input}}),
                  result["confidence_threshold"].get<float>(),
                  result["visibility_threshold"].get<float>());

  const auto& want = result["detections"];
  ASSERT_EQ(detections.size(), want.size());
  for (size_t i = 0; i < want.size(); ++i) {
    SCOPED_TRACE(i);
    const Detection& got = detections[i];
    EXPECT_NEAR(got.score, want[i]["score"].get<float>(), 1e-3);
    EXPECT_NEAR(got.visibility, want[i]["visibility"].get<float>(), 1e-3);
    for (int k = 0; k < 4; ++k) {
      EXPECT_NEAR(got.box_xyxy[k], want[i]["box_xyxy"][k].get<float>(), 0.5)
          << "box coordinate " << k;
    }
    auto mask = LoadPng(Dir() + "/segmentation/" +
                            want[i]["mask"].get<std::string>(),
                        1);
    ASSERT_EQ(got.mask.size(), mask.pixels.size());
    size_t intersection = 0, union_ = 0;
    for (size_t p = 0; p < mask.pixels.size(); ++p) {
      const bool a = got.mask[p], b = mask.pixels[p] > 127;
      intersection += a && b;
      union_ += a || b;
    }
    EXPECT_GT(static_cast<double>(intersection) / union_, 0.99);
  }
}

INSTANTIATE_TEST_SUITE_P(Scenes, RfDetrTest,
                         ::testing::Values("raw_stock_top", "raw_stock_tilted",
                                           "raw_stock_side", "empty"));

TEST(ResizeBilinearTest, UpsamplesWithHalfPixelCenters) {
  const float src[] = {0, 1};
  float dst[4];
  ResizeBilinear(src, 1, 2, 1, 1, dst, 1, 4);
  // Destination pixel centers map to source x = -0.25, 0.25, 0.75, 1.25.
  EXPECT_FLOAT_EQ(dst[0], 0);
  EXPECT_FLOAT_EQ(dst[1], 0.25);
  EXPECT_FLOAT_EQ(dst[2], 0.75);
  EXPECT_FLOAT_EQ(dst[3], 1);
}

}  // namespace
}  // namespace perception::rfdetr
