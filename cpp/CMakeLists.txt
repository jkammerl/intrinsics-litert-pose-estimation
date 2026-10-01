cmake_minimum_required(VERSION 3.24)
project(perception_litert CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# LiteRT C API: headers from the LiteRT release, and the prebuilt libLiteRt.so
# from the matching ai-edge-litert wheel on PyPI. On Android, use libLiteRt.so
# from the LiteRT Maven package instead.
set(LITERT_VERSION 2.2.0)
FetchContent_Declare(
  litert
  URL https://github.com/google-ai-edge/LiteRT/archive/refs/tags/v${LITERT_VERSION}.tar.gz
  URL_HASH SHA256=6d2ce16738199adc5a3cdde76c3c6a6dac636d3b52a1d7790ea524fb0d59f7fc
  SOURCE_SUBDIR headers_only  # Don't build LiteRT; only its headers are used.
)
FetchContent_MakeAvailable(litert)

if(APPLE)
  # macOS arm64: the GPU accelerator runs on Metal.
  set(LITERT_WHEEL_URL https://files.pythonhosted.org/packages/1f/09/778399be516866419e3de6257e4f7ab6b8563b2fc204b6353f2361a8b025/ai_edge_litert-2.2.0-cp312-cp312-macosx_12_0_arm64.whl)
  set(LITERT_WHEEL_SHA256 90fe45c8a0557e15b873f685e33ecaa2302782a8cb09a1ca754d536d5fa30945)
  set(LITERT_LIBRARY libLiteRt.dylib)
  set(LITERT_GPU_LIBRARY libLiteRtMetalAccelerator.dylib)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(LITERT_WHEEL_URL https://files.pythonhosted.org/packages/4f/ef/be6341bb3cc935e47a521ea1a173a022d1680d309545579a12dbc739df40/ai_edge_litert-2.2.0-cp312-cp312-manylinux_2_27_aarch64.whl)
  set(LITERT_WHEEL_SHA256 6d3fe2047461fbdd251425960be7a3674dbbf707bd019a01ffe6eb43cc653434)
else()
  set(LITERT_WHEEL_URL https://files.pythonhosted.org/packages/fa/bc/c4788c401cbe1bb082cb09220b1e35ae8a8f89451b494f4e151a8c93bf4f/ai_edge_litert-2.2.0-cp312-cp312-manylinux_2_27_x86_64.whl)
  set(LITERT_WHEEL_SHA256 4e151f07229b2f714f9b34ea42a9cacffc98953b1f6e832adee4479f6b81f50a)
endif()
if(NOT APPLE)
  set(LITERT_LIBRARY libLiteRt.so)
  set(LITERT_GPU_LIBRARY libLiteRtWebGpuAccelerator.so)
endif()
set(LITERT_WHEEL ${CMAKE_BINARY_DIR}/ai_edge_litert.whl)
# The LiteRT runtime and its GPU accelerator (ML Drift on WebGPU, which runs
# on Vulkan on Linux; Metal on macOS). LiteRT loads the accelerator at runtime
# from LITERT_LIBRARY_DIR.
set(LITERT_LIBRARY_DIR ${CMAKE_BINARY_DIR}/litert_lib/ai_edge_litert)
if(NOT EXISTS ${LITERT_LIBRARY_DIR}/${LITERT_LIBRARY} OR
   NOT EXISTS ${LITERT_LIBRARY_DIR}/${LITERT_GPU_LIBRARY})
  file(DOWNLOAD ${LITERT_WHEEL_URL} ${LITERT_WHEEL}
       EXPECTED_HASH SHA256=${LITERT_WHEEL_SHA256})
  file(ARCHIVE_EXTRACT INPUT ${LITERT_WHEEL}
       DESTINATION ${CMAKE_BINARY_DIR}/litert_lib
       PATTERNS ai_edge_litert/${LITERT_LIBRARY}
                ai_edge_litert/${LITERT_GPU_LIBRARY})
endif()

# LiteRT generates build_config.h at configure time.
set(LITERT_BUILD_CONFIG_DISABLE_GPU 0)
set(LITERT_BUILD_CONFIG_DISABLE_NPU 0)
configure_file(
  ${litert_SOURCE_DIR}/litert/build_common/build_config.h.in
  ${CMAKE_BINARY_DIR}/litert_generated/litert/build_common/build_config.h
)

add_library(litert_c SHARED IMPORTED GLOBAL)
set_target_properties(litert_c PROPERTIES
  IMPORTED_LOCATION ${LITERT_LIBRARY_DIR}/${LITERT_LIBRARY}
  IMPORTED_NO_SONAME TRUE
  INTERFACE_INCLUDE_DIRECTORIES
    "${litert_SOURCE_DIR};${CMAKE_BINARY_DIR}/litert_generated"
)

# Tests are built when this is the top-level project; other projects (e.g.
# the ROS 2 node) can include it with add_subdirectory() for the library.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(PERCEPTION_TOP_LEVEL ON)
else()
  set(PERCEPTION_TOP_LEVEL OFF)
endif()
option(PERCEPTION_BUILD_TESTS "Build the tests" ${PERCEPTION_TOP_LEVEL})

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.tar.gz
  URL_HASH SHA256=65fab701d9829d38cb77c14acdc431d2108bfdbf8979e40eb8ae567edf10b27c
)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
  URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
)
FetchContent_Declare(
  stb
  URL https://github.com/nothings/stb/archive/2c980bb59875b0d32144a71867fbdebb2f77cd20.tar.gz
  URL_HASH SHA256=9a955b1b49a4410088a2e0ee2a9c057c3c907d0c1d75454144cb980aca0ba515
  SOURCE_SUBDIR headers_only
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
if(PERCEPTION_BUILD_TESTS)
  FetchContent_MakeAvailable(googletest)
endif()
FetchContent_MakeAvailable(json stb)

# OpenCV 4, as the IOC service: OpenCV 5 changed warpPerspective's results.
# Homebrew's default OpenCV is 5; prefer its opencv@4.
if(APPLE AND EXISTS /opt/homebrew/opt/opencv@4)
  list(APPEND CMAKE_PREFIX_PATH /opt/homebrew/opt/opencv@4)
endif()
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
if(OpenCV_VERSION VERSION_GREATER_EQUAL 5)
  message(WARNING "OpenCV ${OpenCV_VERSION}: the crops differ from the IOC "
                  "service's (OpenCV 4)")
endif()

# Runs the models with LiteRT; FoundationPose also uses OpenCV for the same
# image warps as the Python service.
add_library(perception STATIC
  foundationpose.cc
  litert_model.cc
  pose_estimator.cc
  rfdetr.cc
)
target_include_directories(perception PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(perception PUBLIC litert_c ${OpenCV_LIBS})
target_include_directories(perception PRIVATE ${OpenCV_INCLUDE_DIRS})
# Don't fuse multiply-adds, so that the arithmetic matches NumPy's.
target_compile_options(perception PRIVATE -ffp-contract=off)
set_target_properties(perception PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Python bindings (python/bindings.cc): the module litert_pose_estimation,
# e.g. for Intrinsic's IOC pose estimator service (see intrinsic/).
option(PERCEPTION_BUILD_PYTHON "Build the Python module" OFF)
if(PERCEPTION_BUILD_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(litert_pose_estimation
    ${CMAKE_CURRENT_SOURCE_DIR}/../python/bindings.cc)
  target_link_libraries(litert_pose_estimation PRIVATE perception)
  target_compile_definitions(litert_pose_estimation PRIVATE
    PERCEPTION_DEFAULT_LITERT_LIBRARY_DIR="${LITERT_LIBRARY_DIR}")
  # Find libLiteRt next to the module, or in the build tree.
  if(APPLE)
    set(PERCEPTION_ORIGIN "@loader_path")
  else()
    set(PERCEPTION_ORIGIN "\$ORIGIN")
  endif()
  set_target_properties(litert_pose_estimation PROPERTIES
    BUILD_RPATH "${PERCEPTION_ORIGIN};${LITERT_LIBRARY_DIR}"
    INSTALL_RPATH "${PERCEPTION_ORIGIN}")
endif()

# Test helpers (images, .npy and JSON files), also used by the ROS 2 node's
# tests.
add_library(perception_test_util STATIC test_util.cc)
target_include_directories(perception_test_util PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR} ${stb_SOURCE_DIR})
target_link_libraries(perception_test_util PUBLIC
  perception nlohmann_json::nlohmann_json)
target_compile_definitions(perception_test_util PUBLIC
  PERCEPTION_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
  LITERT_LIBRARY_DIR="${LITERT_LIBRARY_DIR}"
)
set_target_properties(perception_test_util PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

if(NOT PERCEPTION_BUILD_TESTS)
  return()
endif()

enable_testing()
include(GoogleTest)

add_executable(perception_test
  models_test.cc
  gpu_test.cc
  rfdetr_test.cc
  service_golden_test.cc
)
target_link_libraries(perception_test perception_test_util GTest::gtest_main)
gtest_discover_tests(perception_test DISCOVERY_TIMEOUT 60
  PROPERTIES TIMEOUT 1800)
