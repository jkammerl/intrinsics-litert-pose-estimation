// Python bindings for the LiteRT pose estimation pipeline.
//
// The functions mirror the two model calls of Intrinsic's IOC pose estimator
// service, so that the service (or any Python code) can run them with LiteRT:
//
//   segmenter = litert_pose_estimation.Segmenter(models_dir, accelerator="auto")
//   boxes, scores, masks, visibility = segmenter.segment(rgb, 0.6, 0.6)
//
//   pose_model = litert_pose_estimation.FoundationPose(models_dir, batch_size=128)
//   rotations, translations, confidences = pose_model.estimate(
//       rgb, depth, masks, camera_matrix, cad_obj_text, iterations=6)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <string>
#include <vector>

#include "foundationpose.h"
#include "litert_model.h"
#include "pose_estimator.h"
#include "rfdetr.h"

namespace py = pybind11;
using perception::Accelerator;
using perception::Model;
using perception::ModelOptions;

namespace {

ModelOptions Options(const std::string& accelerator, bool gpu_fp16,
                     const std::string& litert_library_dir, int cpu_threads) {
  ModelOptions options;
  options.accelerator = perception::ParseAccelerator(accelerator);
  options.gpu_fp16 = gpu_fp16;
  options.runtime_library_dir = litert_library_dir;
  options.cpu_threads = cpu_threads;
  return options;
}

py::dict Describe(const Model& model) {
  py::dict d;
  d["accelerator"] = model.accelerator() == Accelerator::kGpu ? "gpu" : "cpu";
  d["fully_accelerated"] = model.fully_accelerated();
  d["fallback_reason"] = model.fallback_reason();
  return d;
}

using U8Image = py::array_t<uint8_t, py::array::c_style | py::array::forcecast>;
using F32Image = py::array_t<float, py::array::c_style | py::array::forcecast>;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

void CheckRgb(const U8Image& rgb) {
  if (rgb.ndim() != 3 || rgb.shape(2) != 3) {
    throw std::invalid_argument("rgb must be a (H, W, 3) uint8 array");
  }
}

std::array<double, 9> CameraMatrix(const F64Array& k) {
  if (k.size() != 9) throw std::invalid_argument("camera matrix must be 3x3");
  std::array<double, 9> out;
  std::copy(k.data(), k.data() + 9, out.begin());
  return out;
}

class Segmenter {
 public:
  Segmenter(const std::string& models_dir, const std::string& accelerator,
            bool gpu_fp16, const std::string& litert_library_dir,
            int cpu_threads)
      : model_(models_dir + "/rfdetr_seg.tflite",
               Options(accelerator, gpu_fp16, litert_library_dir,
                       cpu_threads)) {}

  // Returns (boxes [N, 4] xyxy, scores [N], masks [N, H, W] bool,
  // visibility [N]) like the service's segmentation model.
  py::tuple Segment(const U8Image& rgb, float confidence_threshold,
                    float visibility_threshold) {
    CheckRgb(rgb);
    const int h = rgb.shape(0), w = rgb.shape(1);
    std::vector<perception::rfdetr::Detection> detections;
    {
      py::gil_scoped_release release;
      const perception::rfdetr::Preprocessed pre =
          perception::rfdetr::Preprocess(rgb.data(), h, w);
      detections = perception::rfdetr::Postprocess(
          pre, model_.Run({{"input", &pre.input}}), confidence_threshold,
          visibility_threshold);
    }
    const py::ssize_t n = detections.size();
    py::array_t<float> boxes({n, py::ssize_t(4)});
    py::array_t<float> scores(n), visibility(n);
    py::array_t<bool> masks({n, py::ssize_t(h), py::ssize_t(w)});
    for (py::ssize_t i = 0; i < n; ++i) {
      for (int j = 0; j < 4; ++j) {
        boxes.mutable_at(i, j) = detections[i].box_xyxy[j];
      }
      scores.mutable_at(i) = detections[i].score;
      visibility.mutable_at(i) = detections[i].visibility;
      bool* m = masks.mutable_data(i);
      for (size_t p = 0; p < detections[i].mask.size(); ++p) {
        m[p] = detections[i].mask[p] != 0;
      }
    }
    return py::make_tuple(boxes, scores, masks, visibility);
  }

  py::dict Info() const { return Describe(model_); }

 private:
  Model model_;
};

class FoundationPose {
 public:
  FoundationPose(const std::string& models_dir, int batch_size,
                 const std::string& accelerator, bool gpu_fp16,
                 const std::string& litert_library_dir, int cpu_threads)
      : estimator_(models_dir, batch_size,
                   Options(accelerator, gpu_fp16, litert_library_dir,
                           cpu_threads)) {}

  // Returns (rotations [N, 3, 3], translations [N, 3], confidences [N, 1])
  // like the service's FoundationPose model, for masks [N, H, W] (nonzero
  // inside). `depth` is in meters.
  py::tuple Estimate(const U8Image& rgb, const F32Image& depth,
                     const U8Image& masks, const F64Array& camera_matrix,
                     const std::string& cad_obj, int iterations) {
    CheckRgb(rgb);
    const int h = rgb.shape(0), w = rgb.shape(1);
    if (depth.ndim() != 2 || depth.shape(0) != h || depth.shape(1) != w) {
      throw std::invalid_argument("depth must be (H, W) like rgb");
    }
    const py::ssize_t n = masks.ndim() == 2 ? 1 : masks.shape(0);
    const std::array<double, 9> k = CameraMatrix(camera_matrix);
    const perception::foundationpose::Mesh mesh =
        perception::foundationpose::ParseObj(cad_obj);
    if (mesh.vertices.empty()) throw std::invalid_argument("empty CAD model");
    const perception::foundationpose::Intrinsics intrinsics{
        float(k[0]), float(k[4]), float(k[2]), float(k[5])};
    const perception::foundationpose::RgbdImage image{rgb.data(), depth.data(),
                                                      h, w};
    py::array_t<float> rotations({n, py::ssize_t(3), py::ssize_t(3)});
    py::array_t<float> translations({n, py::ssize_t(3)});
    py::array_t<float> confidences({n, py::ssize_t(1)});
    std::vector<perception::foundationpose::Result> results(n);
    {
      py::gil_scoped_release release;
      for (py::ssize_t i = 0; i < n; ++i) {
        results[i] = estimator_.Estimate(
            image, masks.data() + i * size_t(h) * w, intrinsics, mesh,
            iterations);
      }
    }
    for (py::ssize_t i = 0; i < n; ++i) {
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          rotations.mutable_at(i, r, c) = results[i].pose(r, c);
        }
        translations.mutable_at(i, r) = results[i].pose(r, 3);
      }
      confidences.mutable_at(i, 0) = results[i].score;
    }
    return py::make_tuple(rotations, translations, confidences);
  }

  py::dict Info() const {
    py::dict d;
    d["refiner"] = Describe(estimator_.refiner());
    for (const auto& [batch, scorer] : estimator_.scorers()) {
      d[py::str("scorer_b" + std::to_string(batch))] = Describe(*scorer);
    }
    return d;
  }

 private:
  perception::foundationpose::Estimator estimator_;
};

}  // namespace

PYBIND11_MODULE(litert_pose_estimation, m) {
  m.doc() =
      "Raw stock segmentation (RF-DETR) and 6D pose estimation "
      "(FoundationPose) with LiteRT, matching Intrinsic's IOC pose estimator "
      "service.";
  m.attr("DEFAULT_LITERT_LIBRARY_DIR") = PERCEPTION_DEFAULT_LITERT_LIBRARY_DIR;

  py::class_<Segmenter>(m, "Segmenter")
      .def(py::init<const std::string&, const std::string&, bool,
                    const std::string&, int>(),
           py::arg("models_dir"), py::arg("accelerator") = "auto",
           py::arg("gpu_fp16") = false,
           py::arg("litert_library_dir") =
               std::string(PERCEPTION_DEFAULT_LITERT_LIBRARY_DIR),
           py::arg("cpu_threads") = 0)
      .def("segment", &Segmenter::Segment, py::arg("rgb"),
           py::arg("confidence_threshold") = 0.6f,
           py::arg("visibility_threshold") = 0.6f)
      .def("info", &Segmenter::Info);

  py::class_<FoundationPose>(m, "FoundationPose")
      .def(py::init<const std::string&, int, const std::string&, bool,
                    const std::string&, int>(),
           py::arg("models_dir"), py::arg("batch_size") = 128,
           py::arg("accelerator") = "auto", py::arg("gpu_fp16") = false,
           py::arg("litert_library_dir") =
               std::string(PERCEPTION_DEFAULT_LITERT_LIBRARY_DIR),
           py::arg("cpu_threads") = 0)
      .def("estimate", &FoundationPose::Estimate, py::arg("rgb"),
           py::arg("depth"), py::arg("masks"), py::arg("camera_matrix"),
           py::arg("cad_obj"), py::arg("iterations") = 6)
      .def("info", &FoundationPose::Info);
}
