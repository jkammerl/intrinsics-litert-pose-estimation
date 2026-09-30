#include "rfdetr.h"

#include <algorithm>
#include <cmath>

namespace perception::rfdetr {
namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3] = {0.229f, 0.224f, 0.225f};
constexpr int kQueries = 200;
constexpr int kMaskSize = 126;

float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

// Source index and weight of the second sample for each destination index,
// as OpenCV's INTER_LINEAR computes them.
void LinearCoefficients(int src_size, int dst_size, std::vector<int>* index,
                        std::vector<float>* weight) {
  const double scale = static_cast<double>(src_size) / dst_size;
  index->resize(dst_size);
  weight->resize(dst_size);
  for (int d = 0; d < dst_size; ++d) {
    float f = static_cast<float>((d + 0.5) * scale - 0.5);
    int s = static_cast<int>(std::floor(f));
    f -= s;
    if (s < 0) {
      s = 0;
      f = 0;
    }
    if (s >= src_size - 1) {
      s = src_size - 1;
      f = 0;
    }
    (*index)[d] = s;
    (*weight)[d] = f;
  }
}

}  // namespace

void ResizeBilinear(const float* src, int src_height, int src_width,
                    int src_stride, int channels, float* dst, int dst_height,
                    int dst_width) {
  std::vector<int> xs, ys;
  std::vector<float> wx, wy;
  LinearCoefficients(src_width, dst_width, &xs, &wx);
  LinearCoefficients(src_height, dst_height, &ys, &wy);
  for (int y = 0; y < dst_height; ++y) {
    const int y0 = ys[y], y1 = std::min(y0 + 1, src_height - 1);
    const float* row0 = src + static_cast<size_t>(y0) * src_width * src_stride;
    const float* row1 = src + static_cast<size_t>(y1) * src_width * src_stride;
    for (int x = 0; x < dst_width; ++x) {
      const int x0 = xs[x], x1 = std::min(x0 + 1, src_width - 1);
      float* out = dst + (static_cast<size_t>(y) * dst_width + x) * channels;
      for (int c = 0; c < channels; ++c) {
        // Horizontal, then vertical, like OpenCV.
        float top = row0[x0 * src_stride + c] * (1 - wx[x]) +
                    row0[x1 * src_stride + c] * wx[x];
        float bottom = row1[x0 * src_stride + c] * (1 - wx[x]) +
                       row1[x1 * src_stride + c] * wx[x];
        out[c] = top * (1 - wy[y]) + bottom * wy[y];
      }
    }
  }
}

Preprocessed Preprocess(const uint8_t* rgb, int height, int width) {
  Preprocessed pre;
  pre.height = height;
  pre.width = width;
  pre.scale = static_cast<float>(kSize) /
              static_cast<float>(std::max(height, width));
  pre.resized_height = static_cast<int>(static_cast<float>(height) * pre.scale);
  pre.resized_width = static_cast<int>(static_cast<float>(width) * pre.scale);

  std::vector<float> image(static_cast<size_t>(height) * width * 3);
  for (size_t i = 0; i < image.size(); ++i) image[i] = rgb[i] / 255.0f;
  std::vector<float> resized(static_cast<size_t>(pre.resized_height) *
                             pre.resized_width * 3);
  ResizeBilinear(image.data(), height, width, 3, 3, resized.data(),
                 pre.resized_height, pre.resized_width);

  pre.input = Tensor({1, kSize, kSize, 3});  // Zero-initialized padding.
  for (int y = 0; y < pre.resized_height; ++y) {
    for (int x = 0; x < pre.resized_width; ++x) {
      const float* in =
          &resized[(static_cast<size_t>(y) * pre.resized_width + x) * 3];
      float* out = &pre.input[(static_cast<size_t>(y) * kSize + x) * 3];
      for (int c = 0; c < 3; ++c) out[c] = (in[c] - kMean[c]) / kStd[c];
    }
  }
  return pre;
}

std::vector<Detection> Postprocess(const Preprocessed& pre,
                                   const std::map<std::string, Tensor>& outputs,
                                   float confidence_threshold,
                                   float visibility_threshold) {
  const Tensor& logits = outputs.at("logits");                 // [1, Q, 2]
  const Tensor& visibility = outputs.at("visibility_logits");  // [1, Q, 1]
  const Tensor& boxes = outputs.at("boxes");                   // [1, Q, 4]
  const Tensor& masks = outputs.at("mask_logits");  // [1, 126, 126, Q]

  std::vector<Detection> detections;
  std::vector<float> full(kSize * kSize);
  std::vector<float> cropped(static_cast<size_t>(pre.resized_height) *
                             pre.resized_width);
  std::vector<float> image(static_cast<size_t>(pre.height) * pre.width);
  for (int q = 0; q < kQueries; ++q) {
    Detection d;
    d.score = Sigmoid(logits[q * 2]);
    d.visibility = Sigmoid(visibility[q]);
    if (!(d.score > confidence_threshold &&
          d.visibility > visibility_threshold)) {
      continue;
    }
    const float cx = boxes[q * 4], cy = boxes[q * 4 + 1];
    const float w = boxes[q * 4 + 2], h = boxes[q * 4 + 3];
    const float to_pixels = static_cast<float>(kSize) / pre.scale;
    d.box_xyxy = {(cx - 0.5f * w) * to_pixels, (cy - 0.5f * h) * to_pixels,
                  (cx + 0.5f * w) * to_pixels, (cy + 0.5f * h) * to_pixels};

    // Query q's mask logits are channel q of the NHWC tensor.
    ResizeBilinear(masks.data() + q, kMaskSize, kMaskSize, kQueries, 1,
                   full.data(), kSize, kSize);
    for (int y = 0; y < pre.resized_height; ++y) {
      std::copy_n(&full[static_cast<size_t>(y) * kSize], pre.resized_width,
                  &cropped[static_cast<size_t>(y) * pre.resized_width]);
    }
    ResizeBilinear(cropped.data(), pre.resized_height, pre.resized_width, 1, 1,
                   image.data(), pre.height, pre.width);
    d.mask.resize(image.size());
    for (size_t i = 0; i < image.size(); ++i) d.mask[i] = image[i] > 0;
    detections.push_back(std::move(d));
  }
  return detections;
}

}  // namespace perception::rfdetr
