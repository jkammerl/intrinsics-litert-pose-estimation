#include "foundationpose.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// The arithmetic below mirrors model.py and foundationpose_numpy.py
// operation by operation: float32 where NumPy computes in float32, double
// where it promotes to float64 (e.g. when a float32 array meets an int64 or
// float64 array). Build with -ffp-contract=off so that no multiply-adds are
// fused, as NumPy doesn't fuse them either.

namespace perception::foundationpose {
namespace {

using Vec3 = std::array<float, 3>;

Vec3 Normalized(const Vec3& v) {
  const float n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  return {v[0] / n, v[1] / n, v[2] / n};
}

Vec3 Cross(const Vec3& a, const Vec3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}

// _generate_icosphere: the unit icosphere's vertices in subdivision order.
std::vector<Vec3> Icosphere(int recursion_level) {
  const float t = static_cast<float>((1.0 + std::sqrt(5.0)) / 2.0);
  std::vector<Vec3> vertices;
  for (const Vec3& p : std::vector<Vec3>{
           {-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0},
           {0, -1, t}, {0, 1, t}, {0, -1, -t}, {0, 1, -t},
           {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}}) {
    vertices.push_back(Normalized(p));
  }
  std::vector<std::array<int, 3>> faces = {
      {0, 11, 5}, {0, 5, 1},  {0, 1, 7},   {0, 7, 10}, {0, 10, 11},
      {1, 5, 9},  {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
      {3, 9, 4},  {3, 4, 2},  {3, 2, 6},   {3, 6, 8},  {3, 8, 9},
      {4, 9, 5},  {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
  std::map<std::pair<int, int>, int> cache;
  auto middle_point = [&](int p1, int p2) {
    const std::pair<int, int> key(std::min(p1, p2), std::max(p1, p2));
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    const Vec3& a = vertices[p1];
    const Vec3& b = vertices[p2];
    vertices.push_back(Normalized(
        {(a[0] + b[0]) / 2.0f, (a[1] + b[1]) / 2.0f, (a[2] + b[2]) / 2.0f}));
    return cache[key] = static_cast<int>(vertices.size()) - 1;
  };
  for (int level = 0; level < recursion_level; ++level) {
    std::vector<std::array<int, 3>> new_faces;
    for (const auto& [v1, v2, v3] : faces) {
      const int a = middle_point(v1, v2);
      const int b = middle_point(v2, v3);
      const int c = middle_point(v3, v1);
      new_faces.push_back({v1, a, c});
      new_faces.push_back({v2, b, a});
      new_faces.push_back({v3, c, b});
      new_faces.push_back({a, b, c});
    }
    faces = std::move(new_faces);
  }
  return vertices;
}

// The median like np.median of a float32 array.
float Median(std::vector<float> values) {
  const size_t n = values.size();
  std::nth_element(values.begin(), values.begin() + n / 2, values.end());
  const float upper = values[n / 2];
  if (n % 2 == 1) return upper;
  const float lower =
      *std::max_element(values.begin(), values.begin() + n / 2);
  return (lower + upper) / 2.0f;
}

// np.sign(x) * np.floor(np.abs(x) + 0.5) in float32.
float RoundHalfAway(float x) {
  const float r = std::floor(std::fabs(x) + 0.5f);
  return x > 0 ? r : (x < 0 ? -r : 0.0f);
}

// _axis_angle_to_matrix, row-major.
std::array<float, 9> AxisAngleToMatrix(const Vec3& axis, float angle) {
  const float x = axis[0], y = axis[1], z = axis[2];
  const float c = std::cos(angle), s = std::sin(angle);
  const float C = 1.0f - c;
  return {c + x * x * C,     x * y * C - z * s, x * z * C + y * s,
          y * x * C + z * s, c + y * y * C,     y * z * C - x * s,
          z * x * C - y * s, z * y * C + x * s, c + z * z * C};
}

}  // namespace

Mat4 Mat4::Identity() {
  Mat4 m;
  m(0, 0) = m(1, 1) = m(2, 2) = m(3, 3) = 1.0f;
  return m;
}

Mesh ParseObj(const std::string& text) {
  Mesh mesh;
  std::istringstream lines(text);
  std::string line;
  while (std::getline(lines, line)) {
    std::istringstream words(line);
    std::vector<std::string> parts;
    for (std::string w; words >> w;) parts.push_back(w);
    if (parts.empty() || parts[0][0] == '#') continue;
    if (parts[0] == "v" && parts.size() >= 4) {
      mesh.vertices.push_back(
          {static_cast<float>(std::strtod(parts[1].c_str(), nullptr)),
           static_cast<float>(std::strtod(parts[2].c_str(), nullptr)),
           static_cast<float>(std::strtod(parts[3].c_str(), nullptr))});
    } else if (parts[0] == "f" && parts.size() >= 4) {
      std::vector<int32_t> ids;
      for (size_t i = 1; i < parts.size(); ++i) {
        ids.push_back(std::atoi(parts[i].substr(0, parts[i].find('/')).c_str()) -
                      1);
      }
      if (ids.size() == 3) {
        mesh.faces.push_back({ids[0], ids[1], ids[2]});
      } else if (ids.size() == 4) {
        mesh.faces.push_back({ids[0], ids[1], ids[2]});
        mesh.faces.push_back({ids[0], ids[2], ids[3]});
      }
    }
  }
  return mesh;
}

namespace {
void BoundingBox(const Mesh& mesh, Vec3& lo, Vec3& hi) {
  lo = hi = mesh.vertices.at(0);
  for (const Vec3& v : mesh.vertices) {
    for (int i = 0; i < 3; ++i) {
      lo[i] = std::min(lo[i], v[i]);
      hi[i] = std::max(hi[i], v[i]);
    }
  }
}
}  // namespace

float MeshDiameter(const Mesh& mesh) {
  if (mesh.vertices.empty()) return 0.05f;
  Vec3 lo, hi;
  BoundingBox(mesh, lo, hi);
  const Vec3 d = {hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]};
  return std::max(std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), 0.05f);
}

Vec3 MeshCenter(const Mesh& mesh) {
  Vec3 lo, hi;
  BoundingBox(mesh, lo, hi);
  return {(lo[0] + hi[0]) / 2.0f, (lo[1] + hi[1]) / 2.0f,
          (lo[2] + hi[2]) / 2.0f};
}

std::vector<Mat4> SampleInitialPoses(const uint8_t* mask, const float* depth,
                                     int height, int width,
                                     const Intrinsics& k) {
  // _guess_translation.
  std::array<double, 3> center = {0.0, 0.0, 0.8};
  int u_min = width, u_max = -1, v_min = height, v_max = -1;
  std::vector<float> valid_depths;
  for (int v = 0; v < height; ++v) {
    for (int u = 0; u < width; ++u) {
      if (mask[v * width + u] == 0) continue;
      u_min = std::min(u_min, u);
      u_max = std::max(u_max, u);
      v_min = std::min(v_min, v);
      v_max = std::max(v_max, v);
      const float d = depth[v * width + u];
      if (d >= 0.1f) valid_depths.push_back(d);
    }
  }
  if (u_max >= 0 && !valid_depths.empty()) {
    const double uc = (u_min + u_max) / 2.0;
    const double vc = (v_min + v_max) / 2.0;
    const double zc = Median(std::move(valid_depths));
    // np.linalg.inv(K) in float32, applied in float64.
    const float i00 = 1.0f / k.fx, i02 = -k.cx / k.fx;
    const float i11 = 1.0f / k.fy, i12 = -k.cy / k.fy;
    center = {(double(i00) * uc + double(i02)) * zc,
              (double(i11) * vc + double(i12)) * zc, zc};
  }

  // make_rotation_grid.
  const std::vector<Vec3> sphere = Icosphere(2);
  const double inplane_step = kInplaneStepDeg / 180.0 * M_PI;
  std::vector<double> inplane_rots;
  for (double rot = 0.0; rot < 2.0 * M_PI; rot += inplane_step) {
    inplane_rots.push_back(rot);
  }
  std::vector<Mat4> poses;
  const Vec3 up = {0.0f, 0.0f, 1.0f};
  for (int view = 0; view < kNumViews; ++view) {
    const Vec3& position = sphere[view];
    const Vec3 n = Normalized(position);
    const Vec3 z_axis = {-n[0], -n[1], -n[2]};
    Vec3 x_axis = Cross(up, z_axis);
    if (x_axis[0] == 0 && x_axis[1] == 0 && x_axis[2] == 0) {
      x_axis = {1.0f, 0.0f, 0.0f};
    }
    x_axis = Normalized(x_axis);
    const Vec3 y_axis = Normalized(Cross(z_axis, x_axis));
    for (double rot : inplane_rots) {
      const float c = static_cast<float>(std::cos(rot));
      const float s = static_cast<float>(std::sin(rot));
      // cam_in_ob @ r_inplane rotates the first two columns; the pose is its
      // inverse: the transposed rotation (the translation is replaced by the
      // guessed center below).
      const Vec3 col0 = {x_axis[0] * c + y_axis[0] * s,
                         x_axis[1] * c + y_axis[1] * s,
                         x_axis[2] * c + y_axis[2] * s};
      const Vec3 col1 = {x_axis[0] * -s + y_axis[0] * c,
                         x_axis[1] * -s + y_axis[1] * c,
                         x_axis[2] * -s + y_axis[2] * c};
      Mat4 pose = Mat4::Identity();
      for (int i = 0; i < 3; ++i) {
        pose(0, i) = col0[i];
        pose(1, i) = col1[i];
        pose(2, i) = z_axis[i];
      }
      pose(0, 3) = static_cast<float>(center[0]);
      pose(1, 3) = static_cast<float>(center[1]);
      pose(2, 3) = static_cast<float>(center[2]);
      poses.push_back(pose);
    }
  }
  return poses;
}

std::vector<std::array<float, 9>> ComputeCropWindowTf(
    const std::vector<Mat4>& poses, const Intrinsics& k, float mesh_diameter) {
  const float r = static_cast<float>(double(mesh_diameter) * 1.2 / 2.0);
  const float offsets[5][3] = {
      {0, 0, 0}, {r, 0, 0}, {-r, 0, 0}, {0, r, 0}, {0, -r, 0}};
  const float out = static_cast<float>(kCropSize);
  std::vector<std::array<float, 9>> tfs;
  for (const Mat4& pose : poses) {
    float us[5], vs[5];
    for (int i = 0; i < 5; ++i) {
      const float x = pose(0, 3) + offsets[i][0];
      const float y = pose(1, 3) + offsets[i][1];
      const float z = pose(2, 3) + offsets[i][2];
      // pts @ K.T
      const float pu = x * k.fx + y * 0.0f + z * k.cx;
      const float pv = x * 0.0f + y * k.fy + z * k.cy;
      const float pw = x * 0.0f + y * 0.0f + z * 1.0f;
      us[i] = pu / pw;
      vs[i] = pv / pw;
    }
    float radius = 0.0f;
    for (int i = 0; i < 5; ++i) radius = std::max(radius, std::fabs(vs[i] - vs[0]));
    const float left = RoundHalfAway(us[0] - radius);
    const float right = RoundHalfAway(us[0] + radius);
    const float top = RoundHalfAway(vs[0] - radius);
    const float bottom = RoundHalfAway(vs[0] + radius);
    tfs.push_back({out / (right - left), 0.0f, -left * out / (right - left),
                   0.0f, out / (bottom - top), -top * out / (bottom - top),
                   0.0f, 0.0f, 1.0f});
  }
  return tfs;
}

void UpdateRefinedPoses(std::vector<Mat4>& poses, const float* translation,
                        const float* rotation, float mesh_diameter) {
  const float half_diameter = static_cast<float>(double(mesh_diameter) / 2.0);
  for (size_t i = 0; i < poses.size(); ++i) {
    Mat4& pose = poses[i];
    for (int j = 0; j < 3; ++j) {
      pose(j, 3) += translation[i * 3 + j] * half_diameter;
    }
    Vec3 rot_vec;
    for (int j = 0; j < 3; ++j) {
      rot_vec[j] = std::tanh(rotation[i * 3 + j]) * kRotationNormalizer;
    }
    const float norm = std::sqrt(rot_vec[0] * rot_vec[0] +
                                 rot_vec[1] * rot_vec[1] +
                                 rot_vec[2] * rot_vec[2]);
    if (!(norm > 1e-6f)) continue;
    const std::array<float, 9> m = AxisAngleToMatrix(
        {rot_vec[0] / norm, rot_vec[1] / norm, rot_vec[2] / norm}, norm);
    // rot_mat_delta = m.T; R = rot_mat_delta @ R.
    float r[3][3];
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) {
        r[a][b] = m[0 * 3 + a] * pose(0, b) + m[1 * 3 + a] * pose(1, b) +
                  m[2 * 3 + a] * pose(2, b);
      }
    }
    for (int a = 0; a < 3; ++a) {
      for (int b = 0; b < 3; ++b) pose(a, b) = r[a][b];
    }
  }
}

Mat4 ApplyMeshCenterOffset(const Mat4& pose, const Vec3& c) {
  Mat4 result = pose;
  for (int j = 0; j < 3; ++j) {
    result(j, 3) = pose(j, 0) * c[0] + pose(j, 1) * c[1] + pose(j, 2) * c[2] +
                   pose(j, 3) * 1.0f;
  }
  return result;
}

InputBuilder::InputBuilder(const RgbdImage& image, const Intrinsics& k,
                           const Mesh& mesh)
    : k_(k),
      mesh_(mesh),
      mesh_diameter_(MeshDiameter(mesh)),
      height_(image.height),
      width_(image.width) {
  const size_t n = static_cast<size_t>(height_) * width_;
  rgb_float_.resize(n * 3);
  xyz_map_.resize(n * 3);
  for (size_t i = 0; i < n * 3; ++i) rgb_float_[i] = image.rgb[i] / 255.0f;
  for (int v = 0; v < height_; ++v) {
    for (int u = 0; u < width_; ++u) {
      const size_t i = static_cast<size_t>(v) * width_ + u;
      float d = image.depth[i];
      if (!std::isfinite(d)) d = 0.0f;
      xyz_map_[i * 3 + 0] = (static_cast<float>(u) - k.cx) * d / k.fx;
      xyz_map_[i * 3 + 1] = (static_cast<float>(v) - k.cy) * d / k.fy;
      xyz_map_[i * 3 + 2] = d;
    }
  }
}

void InputBuilder::Build(const std::vector<Mat4>& poses, Tensor& in1,
                         Tensor& in2, int first) const {
  const std::vector<std::array<float, 9>> tfs =
      ComputeCropWindowTf(poses, k_, mesh_diameter_);
  const float radius = static_cast<float>(double(mesh_diameter_) / 2.0);
  const size_t crop_floats = kCropSize * kCropSize * 6;
  cv::Mat rgb(height_, width_, CV_32FC3, const_cast<float*>(rgb_float_.data()));
  cv::Mat xyz(height_, width_, CV_32FC3, const_cast<float*>(xyz_map_.data()));
  for (size_t i = 0; i < poses.size(); ++i) {
    // _prepare_real_crop_6ch.
    cv::Mat m(3, 3, CV_32F, const_cast<float*>(tfs[i].data()));
    cv::Mat crop_rgb, crop_xyz;
    cv::warpPerspective(rgb, crop_rgb, m, cv::Size(kCropSize, kCropSize),
                        cv::INTER_LINEAR);
    cv::warpPerspective(xyz, crop_xyz, m, cv::Size(kCropSize, kCropSize),
                        cv::INTER_NEAREST);
    float* out = in2.data() + (first + i) * crop_floats;
    const float* t = &poses[i].m[0];
    for (int p = 0; p < kCropSize * kCropSize; ++p) {
      const float* c = crop_rgb.ptr<float>() + p * 3;
      const float* x = crop_xyz.ptr<float>() + p * 3;
      const float valid = x[2] > 0.05f ? 1.0f : 0.0f;
      out[p * 6 + 0] = c[0];
      out[p * 6 + 1] = c[1];
      out[p * 6 + 2] = c[2];
      out[p * 6 + 3] = (x[0] - t[3]) / radius * valid;
      out[p * 6 + 4] = (x[1] - t[7]) / radius * valid;
      out[p * 6 + 5] = (x[2] - t[11]) / radius * valid;
    }
    Render(poses[i], tfs[i], in1.data() + (first + i) * crop_floats);
  }
}

// _render_views_nvdiffrast_6ch with RasterizeCpuContext for one candidate.
void InputBuilder::Render(const Mat4& pose, const std::array<float, 9>& tf,
                          float* out) const {
  constexpr int H = kCropSize, W = kCropSize;
  const size_t nv = mesh_.vertices.size();
  std::vector<Vec3> v_cam(nv);
  std::vector<float> px(nv), py(nv), pz(nv);
  for (size_t j = 0; j < nv; ++j) {
    const Vec3& v = mesh_.vertices[j];
    for (int r = 0; r < 3; ++r) {
      v_cam[j][r] =
          v[0] * pose(r, 0) + v[1] * pose(r, 1) + v[2] * pose(r, 2) + pose(r, 3);
    }
    const float u = (v_cam[j][0] * k_.fx / v_cam[j][2]) + k_.cx;
    const float w = (v_cam[j][1] * k_.fy / v_cam[j][2]) + k_.cy;
    const float u_crop = tf[0] * u + tf[2];
    const float w_crop = tf[4] * w + tf[5];
    const float x_clip = 2.0f * u_crop / static_cast<float>(W) - 1.0f;
    const float y_clip = 1.0f - 2.0f * w_crop / static_cast<float>(H);
    const float z_clip = (v_cam[j][2] - 0.1f) / static_cast<float>(100.0 - 0.1);
    // Pixel coordinates in which pixel centers are at integer + 0.5.
    px[j] = (x_clip + 1.0f) * (W / 2.0f);
    py[j] = (y_clip + 1.0f) * (H / 2.0f);
    pz[j] = z_clip;
  }

  // Rasterize: per pixel, the closest fragment (ties: lowest triangle index).
  std::vector<double> best_z(H * W, std::numeric_limits<double>::infinity());
  std::vector<int> best_face(H * W, -1);
  std::vector<float> bary_u(H * W), bary_v(H * W);
  for (size_t f = 0; f < mesh_.faces.size(); ++f) {
    const auto& face = mesh_.faces[f];
    const float ax = px[face[0]], bx = px[face[1]], cx = px[face[2]];
    const float ay = py[face[0]], by = py[face[1]], cy = py[face[2]];
    const float area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay);
    if (!(std::fabs(area) > 1e-12f)) continue;
    const auto clip = [](float x, int hi) {
      return static_cast<int>(std::min(std::max(x, 0.0f), float(hi)));
    };
    const int x0 = clip(std::ceil(std::min({ax, bx, cx}) - 0.5f), W);
    const int x1 = clip(std::floor(std::max({ax, bx, cx}) - 0.5f) + 1, W);
    const int y0 = clip(std::ceil(std::min({ay, by, cy}) - 0.5f), H);
    const int y1 = clip(std::floor(std::max({ay, by, cy}) - 0.5f) + 1, H);
    const double a = area;
    for (int y = y0; y < y1; ++y) {
      const double sy = y + 0.5;
      for (int x = x0; x < x1; ++x) {
        const double sx = x + 0.5;
        const double b0 =
            ((double(bx) - sx) * (double(cy) - sy) -
             (double(cx) - sx) * (double(by) - sy)) / a;
        const double b1 =
            ((double(cx) - sx) * (double(ay) - sy) -
             (double(ax) - sx) * (double(cy) - sy)) / a;
        const double b2 = 1.0 - b0 - b1;
        if (!(b0 >= 0 && b1 >= 0 && b2 >= 0)) continue;
        const double z = b0 * pz[face[0]] + b1 * pz[face[1]] + b2 * pz[face[2]];
        if (!(z >= -1.0 && z <= 1.0)) continue;
        const int p = y * W + x;
        if (!(z < best_z[p])) continue;
        best_z[p] = z;
        best_face[p] = static_cast<int>(f);
        // Perspective-correct barycentrics; w is 1 for every vertex.
        const double p0 = b0 / 1.0, p1 = b1 / 1.0, p2 = b2 / 1.0;
        const double total = p0 + p1 + p2;
        bary_u[p] = static_cast<float>(p0 / total);
        bary_v[p] = static_cast<float>(p1 / total);
      }
    }
  }

  // Interpolate the camera-frame points and write the 6 channels, with the
  // rows flipped (the rasterizer's row 0 is the bottom of the image).
  const float radius = static_cast<float>(double(mesh_diameter_) / 2.0);
  const float t[3] = {pose(0, 3), pose(1, 3), pose(2, 3)};
  for (int y = 0; y < H; ++y) {
    float* row = out + static_cast<size_t>(H - 1 - y) * W * 6;
    for (int x = 0; x < W; ++x) {
      const int p = y * W + x;
      float* o = row + x * 6;
      if (best_face[p] < 0) {
        std::fill(o, o + 6, 0.0f);
        continue;
      }
      const auto& face = mesh_.faces[best_face[p]];
      const float u = bary_u[p], v = bary_v[p], w = 1.0f - u - v;
      o[0] = o[1] = o[2] = 0.5f * 1.0f;
      for (int c = 0; c < 3; ++c) {
        const float xyz = u * v_cam[face[0]][c] + v * v_cam[face[1]][c] +
                          w * v_cam[face[2]][c];
        o[3 + c] = (xyz - t[c]) / radius * 1.0f;
      }
    }
  }
}

Estimator::Estimator(const std::string& models_dir, int batch_size,
                     const ModelOptions& options)
    : batch_size_(batch_size) {
  refiner_ = std::make_unique<Model>(
      models_dir + "/foundationpose_refine.tflite", options);
  const int total = kNumViews * 7;
  const int batch =
      (batch_size <= 0 || batch_size >= total) ? total : batch_size;
  for (int size : {batch, total % batch}) {
    if (size == 0 || scorers_.count(size)) continue;
    const std::string name =
        size == 280 ? "foundationpose_score.tflite"
                    : "foundationpose_score_b" + std::to_string(size) +
                          ".tflite";
    scorers_[size] = std::make_unique<Model>(models_dir + "/" + name, options);
  }
}

Result Estimator::Estimate(const RgbdImage& image, const uint8_t* mask,
                           const Intrinsics& k, const Mesh& mesh,
                           int iterations) {
  const InputBuilder builder(image, k, mesh);
  const float diameter = builder.mesh_diameter();
  std::vector<float> depth(image.depth,
                           image.depth + size_t(image.height) * image.width);
  for (float& d : depth) {
    if (!std::isfinite(d)) d = 0.0f;
  }
  const std::vector<Mat4> candidates =
      SampleInitialPoses(mask, depth.data(), image.height, image.width, k);
  const int total = static_cast<int>(candidates.size());
  const int batch =
      (batch_size_ <= 0 || batch_size_ >= total) ? total : batch_size_;

  const int refine_batch = refiner_->shape("input1")[0];
  Tensor refine_in1({refine_batch, kCropSize, kCropSize, 6});
  Tensor refine_in2({refine_batch, kCropSize, kCropSize, 6});
  float best_score = -std::numeric_limits<float>::infinity();
  Mat4 best_pose = Mat4::Identity();
  bool found = false;
  for (int start = 0; start < total; start += batch) {
    const int end = std::min(start + batch, total);
    std::vector<Mat4> chunk(candidates.begin() + start,
                            candidates.begin() + end);
    for (int iteration = 0; iteration < iterations; ++iteration) {
      std::vector<float> translation(chunk.size() * 3), rotation(chunk.size() * 3);
      // The refiner treats candidates independently: run the chunk in batches
      // of the refiner's size (the last one padded with zeros).
      for (size_t s = 0; s < chunk.size(); s += refine_batch) {
        const size_t n = std::min<size_t>(refine_batch, chunk.size() - s);
        std::fill(refine_in1.data(), refine_in1.data() + refine_in1.size(), 0.0f);
        std::fill(refine_in2.data(), refine_in2.data() + refine_in2.size(), 0.0f);
        builder.Build({chunk.begin() + s, chunk.begin() + s + n}, refine_in1,
                      refine_in2);
        std::map<std::string, Tensor> out =
            refiner_->Run({{"input1", &refine_in1}, {"input2", &refine_in2}});
        std::copy(out.at("output1").data(), out.at("output1").data() + n * 3,
                  translation.begin() + s * 3);
        std::copy(out.at("output2").data(), out.at("output2").data() + n * 3,
                  rotation.begin() + s * 3);
      }
      UpdateRefinedPoses(chunk, translation.data(), rotation.data(), diameter);
    }

    // The scorer compares the chunk's candidates against each other.
    const int n = static_cast<int>(chunk.size());
    auto scorer = scorers_.find(n);
    if (scorer == scorers_.end()) {
      std::fprintf(stderr, "No scorer for a batch of %d candidates.\n", n);
      std::abort();
    }
    Tensor in1({n, kCropSize, kCropSize, 6}), in2({n, kCropSize, kCropSize, 6});
    builder.Build(chunk, in1, in2);
    std::map<std::string, Tensor> out =
        scorer->second->Run({{"input1", &in1}, {"input2", &in2}});
    const Tensor& scores = out.at("output1");
    int best_local = 0;
    for (int i = 1; i < n; ++i) {
      if (scores[i] > scores[best_local]) best_local = i;
    }
    if (scores[best_local] > best_score) {
      best_score = scores[best_local];
      best_pose = chunk[best_local];
      found = true;
    }
  }
  if (!found) return {Mat4::Identity(), 0.0f};
  return {ApplyMeshCenterOffset(best_pose, MeshCenter(mesh)), best_score};
}

}  // namespace perception::foundationpose
