// Loads the test data in testdata/ (see the README).

#ifndef PERCEPTION_LITERT_TEST_UTIL_H_
#define PERCEPTION_LITERT_TEST_UTIL_H_

#include <cstdint>
#include <string>
#include <vector>

#include "litert_model.h"
#include "nlohmann/json.hpp"

namespace perception::testing {

// Path of `relative` in the repository.
std::string Path(const std::string& relative);

// Loads a little-endian float32 or float16 .npy file as a float32 tensor.
Tensor LoadNpy(const std::string& path);

struct Image {
  int height = 0, width = 0, channels = 0;
  std::vector<uint8_t> pixels;  // Row-major, `channels` bytes per pixel.
};
// Loads a PNG, as RGB if channels is 3 or grayscale if 1.
Image LoadPng(const std::string& path, int channels);

nlohmann::json LoadJson(const std::string& path);

}  // namespace perception::testing

#endif  // PERCEPTION_LITERT_TEST_UTIL_H_
