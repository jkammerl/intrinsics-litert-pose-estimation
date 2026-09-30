#include "test_util.h"

#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace perception::testing {
namespace {

float HalfToFloat(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exponent = (h >> 10) & 0x1f;
  uint32_t mantissa = h & 0x3ff;
  uint32_t bits;
  if (exponent == 0) {
    if (mantissa == 0) {
      bits = sign;
    } else {  // Subnormal: normalize.
      exponent = 127 - 15 + 1;
      while (!(mantissa & 0x400)) {
        mantissa <<= 1;
        --exponent;
      }
      bits = sign | (exponent << 23) | ((mantissa & 0x3ff) << 13);
    }
  } else if (exponent == 31) {
    bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
  }
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace

std::string Path(const std::string& relative) {
  return std::string(PERCEPTION_ROOT) + "/" + relative;
}

Tensor LoadNpy(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("Can't open " + path);
  char magic[8];
  file.read(magic, 8);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0) {
    throw std::runtime_error("Not a .npy file: " + path);
  }
  uint32_t header_size = 0;
  if (magic[6] == 1) {
    uint16_t size;
    file.read(reinterpret_cast<char*>(&size), 2);
    header_size = size;
  } else {
    file.read(reinterpret_cast<char*>(&header_size), 4);
  }
  std::string header(header_size, ' ');
  file.read(header.data(), header_size);
  const bool f32 = header.find("'<f4'") != std::string::npos;
  const bool f16 = header.find("'<f2'") != std::string::npos;
  if ((!f32 && !f16) ||
      header.find("'fortran_order': False") == std::string::npos) {
    throw std::runtime_error("Unsupported .npy format: " + header);
  }
  std::vector<int32_t> shape;
  const size_t open = header.find('(', header.find("'shape'"));
  std::stringstream dims(header.substr(open + 1, header.find(')', open) - open - 1));
  for (std::string dim; std::getline(dims, dim, ',');) {
    if (dim.find_first_not_of(' ') != std::string::npos) {
      shape.push_back(std::stoi(dim));
    }
  }
  Tensor tensor(shape);
  if (f32) {
    file.read(reinterpret_cast<char*>(tensor.data()),
              tensor.size() * sizeof(float));
  } else {
    std::vector<uint16_t> half(tensor.size());
    file.read(reinterpret_cast<char*>(half.data()), half.size() * 2);
    for (size_t i = 0; i < half.size(); ++i) tensor[i] = HalfToFloat(half[i]);
  }
  if (!file) throw std::runtime_error("Truncated .npy file: " + path);
  return tensor;
}

Image LoadPng(const std::string& path, int channels) {
  Image image;
  int file_channels;
  uint8_t* data = stbi_load(path.c_str(), &image.width, &image.height,
                            &file_channels, channels);
  if (data == nullptr) throw std::runtime_error("Can't load " + path);
  image.channels = channels;
  image.pixels.assign(data, data + static_cast<size_t>(image.width) *
                                       image.height * channels);
  stbi_image_free(data);
  return image;
}

nlohmann::json LoadJson(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Can't open " + path);
  return nlohmann::json::parse(file);
}

}  // namespace perception::testing
