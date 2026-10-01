// Runs a TFLite model with LiteRT's C API: on the GPU (LiteRT's ML Drift GPU
// accelerator, through WebGPU/Vulkan, OpenCL or Metal depending on the
// platform) or on the CPU (XNNPACK); see Accelerator and ModelOptions.

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

// Hardware that LiteRT runs a model on.
enum class Accelerator {
  // The GPU if there is a hardware GPU, else the CPU (fallback_reason() says
  // why). On the GPU, as kGpu.
  kAuto,
  // The GPU, also a software one (e.g. Mesa's llvmpipe Vulkan device, useful
  // for testing). Ops the GPU accelerator doesn't support run on the CPU. A
  // model that can't run on this GPU at all (it fails to compile or run, e.g.
  // a tensor exceeds the maximum buffer size, or computes other results than
  // the CPU) is an error, unless ModelOptions::cpu_fallback is set.
  kGpu,
  // The CPU only (XNNPACK).
  kCpu,
};

// Parses "auto", "gpu" or "cpu"; throws std::invalid_argument otherwise.
Accelerator ParseAccelerator(const std::string& name);

// Returns "" if a hardware GPU is available to LiteRT's GPU accelerator, else
// why not (probed once). On Linux, the accelerator runs on WebGPU over Vulkan;
// a software Vulkan device (e.g. Mesa's llvmpipe) emulates the GPU on the CPU
// and is slower than LiteRT's XNNPACK CPU path, so kAuto doesn't use it (the
// reason then names it as "only software Vulkan devices").
const std::string& HardwareGpuUnavailableReason();

struct ModelOptions {
  Accelerator accelerator = Accelerator::kCpu;
  // GPU precision: float32 (as the CPU) or float16 (faster on most GPUs, less
  // accurate).
  bool gpu_fp16 = false;
  // CPU (XNNPACK) threads; 0 for the number of hardware threads.
  int cpu_threads = 0;
  // Directory with LiteRT's GPU accelerator libraries (e.g.
  // libLiteRtWebGpuAccelerator.so); empty to search the library path.
  std::string runtime_library_dir;
  // Before using the GPU, check that it computes what the CPU computes: run
  // the model on both with the same pseudo-random inputs, and use the CPU if
  // the outputs differ (relative tolerance 1e-3, 5e-2 with gpu_fp16). Some
  // GPU drivers compile and run models that they then compute wrongly.
  bool validate_gpu = true;
  // If the GPU can't run the model (see Accelerator::kGpu), run it on the CPU
  // instead of failing; fallback_reason() says why.
  bool cpu_fallback = false;
};

// A model with a single signature and float32 or float16 inputs and outputs.
// Tensors are always float32 on this side; float16 ones are converted.
class Model {
 public:
  // Loads and compiles the model for the CPU; throws std::runtime_error on
  // failure.
  explicit Model(const std::string& path) : Model(path, ModelOptions()) {}
  // Loads and compiles the model for `options.accelerator`. On the GPU, the
  // model is compiled and run once (see validate_gpu); if that fails, it is
  // compiled for the CPU with options.cpu_fallback (fallback_reason() says
  // why), else std::runtime_error is thrown, as when the model can't be
  // compiled at all.
  Model(const std::string& path, const ModelOptions& options);
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

  // Runs the model. `inputs` must contain every input, with its shape;
  // throws std::runtime_error if LiteRT fails.
  std::map<std::string, Tensor> Run(
      const std::map<std::string, const Tensor*>& inputs);

  // The hardware the model runs on: Accelerator::kGpu or Accelerator::kCpu.
  Accelerator accelerator() const { return accelerator_; }
  // Whether every op runs on the selected accelerator (on the GPU, ops it
  // doesn't support run on the CPU).
  bool fully_accelerated() const { return fully_accelerated_; }
  // Why the model runs on the CPU although kAuto or kGpu was requested, or
  // "" if it doesn't.
  const std::string& fallback_reason() const { return fallback_reason_; }

 private:
  void Init(const std::string& path, const ModelOptions& options);
  // Destroys the LiteRT objects.
  void Release();
  // Compiles the model for `accelerator`; returns an error message, or "".
  std::string Compile(Accelerator accelerator, const ModelOptions& options);
  // Runs the model; returns an error message, or "".
  std::string TryRun(const std::map<std::string, const Tensor*>& inputs,
                     std::map<std::string, Tensor>* outputs);
  // Runs the GPU-compiled model once, and with options.validate_gpu also the
  // model at `path` on the CPU, on the same inputs; returns "" if the GPU
  // ran and agrees with the CPU, else why not.
  std::string CheckGpu(const std::string& path, const ModelOptions& options);

  LiteRtEnvironment env_ = nullptr;
  ::LiteRtModel model_ = nullptr;
  LiteRtCompiledModel compiled_ = nullptr;
  Accelerator accelerator_ = Accelerator::kCpu;
  bool fully_accelerated_ = false;
  std::string fallback_reason_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, std::vector<int32_t>> shapes_;
  std::set<std::string> float16_;  // Tensors that are float16 in the model.
};

}  // namespace perception

#endif  // PERCEPTION_LITERT_LITERT_MODEL_H_
