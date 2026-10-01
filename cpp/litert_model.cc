#include "litert_model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>
#include <thread>

#if defined(__linux__)
#include <dlfcn.h>
#include <vulkan/vulkan_core.h>
#endif

#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
#include "litert/c/litert_environment_options.h"
#include "litert/c/litert_opaque_options.h"
#include "litert/c/litert_options.h"
#include "litert/c/litert_tensor_buffer.h"

namespace perception {
namespace {

void Check(LiteRtStatus status, const char* what) {
  if (status != kLiteRtStatusOk) {
    // LiteRtGetStatusString isn't exported by the prebuilt libLiteRt.so.
    std::fprintf(stderr, "LiteRT: %s failed with status %d\n", what,
                 static_cast<int>(status));
    std::abort();
  }
}

#define CHECK_LITERT(expr) Check((expr), #expr)

// Returns "" if `status` is OK, else a message naming `what`.
std::string Error(LiteRtStatus status, const char* what) {
  if (status == kLiteRtStatusOk) return "";
  return std::string(what) + " failed with LiteRT status " +
         std::to_string(static_cast<int>(status));
}

#define RETURN_IF_LITERT_ERROR(expr)                       \
  do {                                                     \
    std::string error = Error((expr), #expr);              \
    if (!error.empty()) return error;                      \
  } while (false)

size_t NumElements(const std::vector<int32_t>& shape) {
  return std::accumulate(shape.begin(), shape.end(), size_t{1},
                         std::multiplies<size_t>());
}

LiteRtRankedTensorType TensorType(const std::vector<int32_t>& shape,
                                  bool float16) {
  LiteRtRankedTensorType type{};
  type.element_type =
      float16 ? kLiteRtElementTypeFloat16 : kLiteRtElementTypeFloat32;
  type.layout.rank = shape.size();
  for (size_t i = 0; i < shape.size(); ++i) {
    type.layout.dimensions[i] = shape[i];
  }
  return type;
}

// Returns the tensor's shape, and whether it is float16 (else float32).
std::vector<int32_t> ShapeOf(LiteRtTensor tensor, bool* float16) {
  LiteRtRankedTensorType type;
  CHECK_LITERT(LiteRtGetRankedTensorType(tensor, &type));
  *float16 = type.element_type == kLiteRtElementTypeFloat16;
  if (!*float16 && type.element_type != kLiteRtElementTypeFloat32) {
    std::fprintf(stderr, "Only float32 and float16 tensors are supported.\n");
    std::abort();
  }
  return std::vector<int32_t>(type.layout.dimensions,
                              type.layout.dimensions + type.layout.rank);
}

uint16_t FloatToHalf(float value) {
  uint32_t f;
  std::memcpy(&f, &value, sizeof(f));
  const uint32_t sign = (f >> 16) & 0x8000u;
  const uint32_t abs = f & 0x7fffffffu;
  if (abs >= 0x7f800000u) {  // Inf or NaN.
    return sign | 0x7c00u | (abs > 0x7f800000u ? 0x200u : 0);
  }
  if (abs >= 0x477ff000u) return sign | 0x7c00u;  // Overflows to inf.
  if (abs < 0x38800000u) {  // Subnormal or zero in half precision.
    float magnitude;
    std::memcpy(&magnitude, &abs, sizeof(magnitude));
    // 2^-24 is the smallest subnormal; rint rounds half to even.
    return sign | static_cast<uint32_t>(std::nearbyint(magnitude * 16777216.0f));
  }
  // Normal: rebias the exponent and round the mantissa to nearest even.
  uint32_t half = ((abs - 0x38000000u) >> 13);
  const uint32_t rest = abs & 0x1fffu;
  if (rest > 0x1000u || (rest == 0x1000u && (half & 1u))) ++half;
  return sign | half;
}

float HalfToFloat(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  const uint32_t exponent = (h >> 10) & 0x1fu;
  const uint32_t mantissa = h & 0x3ffu;
  float magnitude;
  if (exponent == 0) {
    magnitude = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31) {
    magnitude = mantissa ? std::nanf("") : INFINITY;
  } else {
    magnitude = std::ldexp(static_cast<float>(mantissa | 0x400u),
                           static_cast<int>(exponent) - 25);
  }
  uint32_t bits;
  std::memcpy(&bits, &magnitude, sizeof(bits));
  bits |= sign;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// 64-byte aligned host memory for a LiteRT tensor buffer.
struct AlignedBuffer {
  explicit AlignedBuffer(size_t bytes)
      : size((bytes + 63) / 64 * 64), data(std::aligned_alloc(64, size)) {}
  ~AlignedBuffer() { std::free(data); }
  AlignedBuffer(const AlignedBuffer&) = delete;
  AlignedBuffer& operator=(const AlignedBuffer&) = delete;
  size_t size;
  void* data;
};

}  // namespace

void Tensor::Free::operator()(float* p) const { std::free(p); }

Tensor::Tensor(std::vector<int32_t> shape)
    : shape_(std::move(shape)), size_(NumElements(shape_)) {
  // LiteRT needs LITERT_HOST_MEMORY_BUFFER_ALIGNMENT (64) byte alignment, and
  // aligned_alloc a size that is a multiple of the alignment.
  size_t bytes = (size_ * sizeof(float) + 63) / 64 * 64;
  data_.reset(static_cast<float*>(std::aligned_alloc(64, bytes)));
  std::memset(data_.get(), 0, bytes);
}

// Returns "" if a hardware GPU is available to LiteRT's GPU accelerator,
// else why not. On Linux, the accelerator runs on WebGPU over Vulkan; a
// software Vulkan device (e.g. Mesa's llvmpipe) emulates the GPU on the CPU
// and is slower than LiteRT's XNNPACK CPU path, so `auto` doesn't use it.
static std::string HardwareGpuUnavailableReason() {
#if defined(__linux__)
  void* lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) return "no Vulkan loader (libvulkan.so.1)";
  auto get = reinterpret_cast<PFN_vkGetInstanceProcAddr>(
      dlsym(lib, "vkGetInstanceProcAddr"));
  auto create = get == nullptr
                    ? nullptr
                    : reinterpret_cast<PFN_vkCreateInstance>(
                          get(nullptr, "vkCreateInstance"));
  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo info{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  info.pApplicationInfo = &app;
  VkInstance instance;
  if (create == nullptr || create(&info, nullptr, &instance) != VK_SUCCESS) {
    dlclose(lib);
    return "can't create a Vulkan instance";
  }
  auto enumerate = reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
      get(instance, "vkEnumeratePhysicalDevices"));
  auto properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
      get(instance, "vkGetPhysicalDeviceProperties"));
  auto destroy = reinterpret_cast<PFN_vkDestroyInstance>(
      get(instance, "vkDestroyInstance"));
  uint32_t count = 0;
  enumerate(instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  enumerate(instance, &count, devices.data());
  bool hardware = false;
  std::string names;
  for (VkPhysicalDevice device : devices) {
    VkPhysicalDeviceProperties p;
    properties(device, &p);
    if (p.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) hardware = true;
    names += (names.empty() ? "" : ", ") + std::string(p.deviceName);
  }
  destroy(instance, nullptr);
  dlclose(lib);
  if (hardware) return "";
  if (count == 0) return "no Vulkan device";
  return "only software Vulkan devices (" + names +
         "), which are slower than the CPU path";
#else
  return "";
#endif
}

static const std::string& CachedHardwareGpuUnavailableReason() {
  static const std::string* reason =
      new std::string(HardwareGpuUnavailableReason());
  return *reason;
}

Accelerator ParseAccelerator(const std::string& name) {
  if (name == "auto") return Accelerator::kAuto;
  if (name == "gpu") return Accelerator::kGpu;
  if (name == "cpu") return Accelerator::kCpu;
  std::fprintf(stderr, "Unknown accelerator \"%s\" (auto, gpu or cpu).\n",
               name.c_str());
  std::abort();
}

Model::Model(const std::string& path, const ModelOptions& options) {
  std::vector<LiteRtEnvOption> env_options;
  if (!options.runtime_library_dir.empty()) {
    LiteRtEnvOption option;
    option.tag = kLiteRtEnvOptionTagRuntimeLibraryDir;
    option.value.type = kLiteRtAnyTypeString;
    option.value.str_value = options.runtime_library_dir.c_str();
    env_options.push_back(option);
  }
  CHECK_LITERT(LiteRtCreateEnvironment(env_options.size(), env_options.data(),
                                       &env_));
  CHECK_LITERT(LiteRtCreateModelFromFile(env_, path.c_str(), &model_));

  LiteRtSignature signature;
  CHECK_LITERT(LiteRtGetModelSignature(model_, 0, &signature));
  LiteRtParamIndex num_inputs, num_outputs;
  CHECK_LITERT(LiteRtGetNumSignatureInputs(signature, &num_inputs));
  CHECK_LITERT(LiteRtGetNumSignatureOutputs(signature, &num_outputs));
  for (LiteRtParamIndex i = 0; i < num_inputs; ++i) {
    const char* name;
    LiteRtTensor tensor;
    CHECK_LITERT(LiteRtGetSignatureInputName(signature, i, &name));
    CHECK_LITERT(LiteRtGetSignatureInputTensor(signature, name, &tensor));
    input_names_.push_back(name);
    bool float16;
    shapes_[name] = ShapeOf(tensor, &float16);
    if (float16) float16_.insert(name);
  }
  for (LiteRtParamIndex i = 0; i < num_outputs; ++i) {
    const char* name;
    LiteRtTensor tensor;
    CHECK_LITERT(LiteRtGetSignatureOutputName(signature, i, &name));
    CHECK_LITERT(LiteRtGetSignatureOutputTensor(signature, name, &tensor));
    output_names_.push_back(name);
    bool float16;
    shapes_[name] = ShapeOf(tensor, &float16);
    if (float16) float16_.insert(name);
  }

  if (options.accelerator == Accelerator::kCpu) {
    std::string error = Compile(Accelerator::kCpu, options);
    if (!error.empty()) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), error.c_str());
      std::abort();
    }
    return;
  }

  // In auto mode, use the GPU only if there is a hardware GPU.
  if (options.accelerator == Accelerator::kAuto &&
      !CachedHardwareGpuUnavailableReason().empty()) {
    fallback_reason_ = CachedHardwareGpuUnavailableReason();
    const std::string cpu_error = Compile(Accelerator::kCpu, options);
    if (!cpu_error.empty()) {
      std::fprintf(stderr, "%s: %s\n", path.c_str(), cpu_error.c_str());
      std::abort();
    }
    return;
  }

  // Compile for the GPU, then run once: some GPU drivers only fail when the
  // kernels are first executed.
  std::string error = Compile(Accelerator::kGpu, options);
  if (error.empty()) {
    std::map<std::string, Tensor> zeros;
    std::map<std::string, const Tensor*> inputs;
    for (const std::string& name : input_names_) {
      zeros[name] = Tensor(shapes_.at(name));
      inputs[name] = &zeros[name];
    }
    std::map<std::string, Tensor> outputs;
    error = TryRun(inputs, &outputs);
  }
  if (error.empty()) return;
  if (options.accelerator == Accelerator::kGpu) {
    std::fprintf(stderr, "%s: can't run on the GPU: %s\n", path.c_str(),
                 error.c_str());
    std::abort();
  }
  fallback_reason_ = error;
  error = Compile(Accelerator::kCpu, options);
  if (!error.empty()) {
    std::fprintf(stderr, "%s: %s\n", path.c_str(), error.c_str());
    std::abort();
  }
}

std::string Model::Compile(Accelerator accelerator,
                           const ModelOptions& options) {
  if (compiled_ != nullptr) {
    LiteRtDestroyCompiledModel(compiled_);
    compiled_ = nullptr;
  }
  LiteRtOptions compile_options;
  RETURN_IF_LITERT_ERROR(LiteRtCreateOptions(&compile_options));
  std::unique_ptr<LiteRtOptionsT, void (*)(LiteRtOptions)> destroy(
      compile_options, LiteRtDestroyOptions);
  if (accelerator == Accelerator::kGpu) {
    // The GPU accelerator runs what it supports; the rest runs on the CPU.
    RETURN_IF_LITERT_ERROR(LiteRtSetOptionsHardwareAccelerators(
        compile_options, kLiteRtHwAcceleratorGpu | kLiteRtHwAcceleratorCpu));
    // GPU accelerator options, as the TOML payload that LiteRT's
    // LrtGetOpaqueGpuOptionsData produces (that helper isn't exported by the
    // prebuilt libLiteRt.so).
    const std::string toml =
        "precision = " +
        std::to_string(static_cast<int>(options.gpu_fp16
                                            ? kLiteRtDelegatePrecisionFp16
                                            : kLiteRtDelegatePrecisionFp32)) +
        "\n";
    LiteRtOpaqueOptions opaque;
    RETURN_IF_LITERT_ERROR(LiteRtCreateOpaqueOptions(
        "gpu_options", strdup(toml.c_str()), std::free, &opaque));
    RETURN_IF_LITERT_ERROR(LiteRtAddOpaqueOptions(compile_options, opaque));
  } else {
    RETURN_IF_LITERT_ERROR(LiteRtSetOptionsHardwareAccelerators(
        compile_options, kLiteRtHwAcceleratorCpu));
  }
  // CPU (XNNPACK) options, as the TOML payload of LrtGetOpaqueCpuOptionsData:
  // LiteRT uses a single thread unless told otherwise.
  const int threads = options.cpu_threads > 0
                          ? options.cpu_threads
                          : static_cast<int>(std::thread::hardware_concurrency());
  const std::string cpu_toml = "num_threads = " + std::to_string(threads) + "\n";
  LiteRtOpaqueOptions cpu_opaque;
  RETURN_IF_LITERT_ERROR(LiteRtCreateOpaqueOptions(
      "xnnpack", strdup(cpu_toml.c_str()), std::free, &cpu_opaque));
  RETURN_IF_LITERT_ERROR(LiteRtAddOpaqueOptions(compile_options, cpu_opaque));
  RETURN_IF_LITERT_ERROR(
      LiteRtCreateCompiledModel(env_, model_, compile_options, &compiled_));
  accelerator_ = accelerator;
  fully_accelerated_ = false;
  LiteRtCompiledModelIsFullyAccelerated(compiled_, &fully_accelerated_);
  return "";
}

Model::~Model() {
  LiteRtDestroyCompiledModel(compiled_);
  LiteRtDestroyModel(model_);
  LiteRtDestroyEnvironment(env_);
}

const std::vector<int32_t>& Model::shape(const std::string& name) const {
  return shapes_.at(name);
}

std::string Model::Metadata(const std::string& key) const {
  const void* data;
  size_t size;
  if (LiteRtGetModelMetadata(model_, key.c_str(), &data, &size) !=
      kLiteRtStatusOk) {
    return "";
  }
  return std::string(static_cast<const char*>(data), size);
}

std::map<std::string, Tensor> Model::Run(
    const std::map<std::string, const Tensor*>& inputs) {
  std::map<std::string, Tensor> outputs;
  std::string error = TryRun(inputs, &outputs);
  if (!error.empty()) {
    std::fprintf(stderr, "LiteRT: %s\n", error.c_str());
    std::abort();
  }
  return outputs;
}

std::string Model::TryRun(const std::map<std::string, const Tensor*>& inputs,
                          std::map<std::string, Tensor>* outputs_out) {
  // Float16 tensors are converted to and from float32 in separate buffers.
  std::vector<std::unique_ptr<AlignedBuffer>> half_buffers;
  auto make_buffer = [&](const std::string& name, float* data, size_t size) {
    const bool float16 = float16_.count(name) > 0;
    LiteRtRankedTensorType type = TensorType(shapes_.at(name), float16);
    void* memory = data;
    size_t bytes = size * sizeof(float);
    if (float16) {
      half_buffers.push_back(
          std::make_unique<AlignedBuffer>(size * sizeof(uint16_t)));
      memory = half_buffers.back()->data;
      bytes = size * sizeof(uint16_t);
    }
    LiteRtTensorBuffer buffer;
    CHECK_LITERT(LiteRtCreateTensorBufferFromHostMemory(
        &type, memory, bytes, /*deallocator=*/nullptr, &buffer));
    return std::make_pair(buffer, static_cast<uint16_t*>(float16 ? memory
                                                                 : nullptr));
  };

  std::vector<LiteRtTensorBuffer> input_buffers, output_buffers;
  for (const std::string& name : input_names_) {
    const Tensor* tensor = inputs.at(name);
    if (tensor->shape() != shapes_.at(name)) {
      std::fprintf(stderr, "Wrong shape for input %s.\n", name.c_str());
      std::abort();
    }
    auto [buffer, half] =
        make_buffer(name, const_cast<float*>(tensor->data()), tensor->size());
    if (half != nullptr) {
      for (size_t i = 0; i < tensor->size(); ++i) {
        half[i] = FloatToHalf((*tensor)[i]);
      }
    }
    input_buffers.push_back(buffer);
  }
  std::map<std::string, Tensor> outputs;
  std::vector<std::pair<Tensor*, const uint16_t*>> half_outputs;
  for (const std::string& name : output_names_) {
    Tensor& tensor = outputs[name] = Tensor(shapes_.at(name));
    auto [buffer, half] = make_buffer(name, tensor.data(), tensor.size());
    if (half != nullptr) half_outputs.emplace_back(&tensor, half);
    output_buffers.push_back(buffer);
  }
  const std::string error = Error(
      LiteRtRunCompiledModel(compiled_, /*signature_index=*/0,
                             input_buffers.size(), input_buffers.data(),
                             output_buffers.size(), output_buffers.data()),
      "LiteRtRunCompiledModel");
  for (auto [tensor, half] : half_outputs) {
    for (size_t i = 0; i < tensor->size(); ++i) {
      (*tensor)[i] = HalfToFloat(half[i]);
    }
  }
  for (LiteRtTensorBuffer buffer : input_buffers) {
    LiteRtDestroyTensorBuffer(buffer);
  }
  for (LiteRtTensorBuffer buffer : output_buffers) {
    LiteRtDestroyTensorBuffer(buffer);
  }
  if (error.empty()) *outputs_out = std::move(outputs);
  return error;
}

}  // namespace perception
