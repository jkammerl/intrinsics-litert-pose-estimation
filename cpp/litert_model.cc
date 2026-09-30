#include "litert_model.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <numeric>

#include "litert/c/litert_compiled_model.h"
#include "litert/c/litert_environment.h"
#include "litert/c/litert_model.h"
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

Model::Model(const std::string& path) {
  CHECK_LITERT(LiteRtCreateEnvironment(0, nullptr, &env_));
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

  LiteRtOptions options;
  CHECK_LITERT(LiteRtCreateOptions(&options));
  CHECK_LITERT(
      LiteRtSetOptionsHardwareAccelerators(options, kLiteRtHwAcceleratorCpu));
  CHECK_LITERT(LiteRtCreateCompiledModel(env_, model_, options, &compiled_));
  LiteRtDestroyOptions(options);
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
  CHECK_LITERT(LiteRtRunCompiledModel(
      compiled_, /*signature_index=*/0, input_buffers.size(),
      input_buffers.data(), output_buffers.size(), output_buffers.data()));
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
  return outputs;
}

}  // namespace perception
