// RF-DETR segmentation: pre- and post-processing around rfdetr_seg.tflite.
//
// A C++ port of tools/rfdetr.py without dependencies, so that it also builds
// for Android. See models/README.md for the tensor specs.

#ifndef PERCEPTION_LITERT_RFDETR_H_
#define PERCEPTION_LITERT_RFDETR_H_

#include <array>
#include <cstdint>
#include <vector>

#include "litert_model.h"

namespace perception::rfdetr {

inline constexpr int kSize = 504;  // Side of the network's square input.

struct Preprocessed {
  Tensor input;  // [1, kSize, kSize, 3]
  float scale = 0;  // kSize / max(height, width)
  int resized_height = 0, resized_width = 0;  // Before padding.
  int height = 0, width = 0;  // Of the image.
};

struct Detection {
  std::array<float, 4> box_xyxy;  // In image pixels.
  float score = 0;
  float visibility = 0;
  std::vector<uint8_t> mask;  // height * width, 1 inside the object.
};

// Scales, normalizes and pads an RGB image (row-major, 3 bytes per pixel).
Preprocessed Preprocess(const uint8_t* rgb, int height, int width);

// Returns the detections with score > confidence_threshold and visibility >
// visibility_threshold, in query order. `outputs` are the model's outputs.
std::vector<Detection> Postprocess(const Preprocessed& pre,
                                   const std::map<std::string, Tensor>& outputs,
                                   float confidence_threshold,
                                   float visibility_threshold);

// Bilinear resize with half-pixel centers and clamped borders, like OpenCV's
// INTER_LINEAR (and ONNX Resize with linear/half_pixel) for float images.
// Pixel (y, x), channel c of `src` is at src[(y * src_width + x) * src_stride
// + c]; `dst` is packed with `channels` values per pixel.
void ResizeBilinear(const float* src, int src_height, int src_width,
                    int src_stride, int channels, float* dst, int dst_height,
                    int dst_width);

}  // namespace perception::rfdetr

#endif  // PERCEPTION_LITERT_RFDETR_H_
