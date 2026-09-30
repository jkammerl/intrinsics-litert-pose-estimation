// Runs a TFLite model on the CPU with LiteRT's C API.

#ifndef PERCEPTION_LITERT_LITERT_MODEL_H_
#define PERCEPTION_LITERT_LITERT_MODEL_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <vector>

#include "litert/c/litert_common.h"

namespace perception {

// A float32 tensor in host memory, aligned as LiteRT requires.
class Tensor {
 public:
  Tensor() = default;
  explicit Tensor(std::vector<int32_t> shape);

  const std::vector<int32_t>& shape() const { return shape_; }
  size_t size() const { return size_; }  // Number of elements.
  float* data() { return data_.get(); }
  const float* data() const { return data_.get(); }
  float& operator[](size_t i) { return data_.get()[i]; }
  float operator[](size_t i) const { return data_.get()[i]; }

 private:
  struct Free {
    void operator()(float* p) const;
  };
  std::vector<int32_t> shape_;
  size_t size_ = 0;
  std::unique_ptr<float, Free> data_;
};

// A model with a single signature and float32 or float16 inputs and outputs.
// Tensors are always float32 on this side; float16 ones are converted.
class Model {
 public:
  // Loads and compiles the model for the CPU; aborts with a message on
  // failure.
  explicit Model(const std::string& path);
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const std::vector<std::string>& input_names() const { return input_names_; }
  const std::vector<std::string>& output_names() const {
    return output_names_;
  }
  const std::vector<int32_t>& shape(const std::string& tensor_name) const;
  // Whether the tensor is float16 in the model (Run converts it).
  bool is_float16(const std::string& tensor_name) const {
    return float16_.count(tensor_name) > 0;
  }

  // The model's metadata entry `key` (e.g. "io_spec"), or "" if missing.
  std::string Metadata(const std::string& key) const;

  // Runs the model. `inputs` must contain every input, with its shape.
  std::map<std::string, Tensor> Run(
      const std::map<std::string, const Tensor*>& inputs);

 private:
  LiteRtEnvironment env_ = nullptr;
  ::LiteRtModel model_ = nullptr;
  LiteRtCompiledModel compiled_ = nullptr;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, std::vector<int32_t>> shapes_;
  std::set<std::string> float16_;  // Tensors that are float16 in the model.
};

}  // namespace perception

#endif  // PERCEPTION_LITERT_LITERT_MODEL_H_
