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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(LITERT_WHEEL_URL https://files.pythonhosted.org/packages/4f/ef/be6341bb3cc935e47a521ea1a173a022d1680d309545579a12dbc739df40/ai_edge_litert-2.2.0-cp312-cp312-manylinux_2_27_aarch64.whl)
  set(LITERT_WHEEL_SHA256 6d3fe2047461fbdd251425960be7a3674dbbf707bd019a01ffe6eb43cc653434)
else()
  set(LITERT_WHEEL_URL https://files.pythonhosted.org/packages/fa/bc/c4788c401cbe1bb082cb09220b1e35ae8a8f89451b494f4e151a8c93bf4f/ai_edge_litert-2.2.0-cp312-cp312-manylinux_2_27_x86_64.whl)
  set(LITERT_WHEEL_SHA256 4e151f07229b2f714f9b34ea42a9cacffc98953b1f6e832adee4479f6b81f50a)
endif()
set(LITERT_WHEEL ${CMAKE_BINARY_DIR}/ai_edge_litert.whl)
if(NOT EXISTS ${CMAKE_BINARY_DIR}/litert_lib/ai_edge_litert/libLiteRt.so)
  file(DOWNLOAD ${LITERT_WHEEL_URL} ${LITERT_WHEEL}
       EXPECTED_HASH SHA256=${LITERT_WHEEL_SHA256})
  file(ARCHIVE_EXTRACT INPUT ${LITERT_WHEEL}
       DESTINATION ${CMAKE_BINARY_DIR}/litert_lib
       PATTERNS ai_edge_litert/libLiteRt.so)
endif()

# LiteRT generates build_config.h at configure time.
set(LITERT_BUILD_CONFIG_DISABLE_GPU 0)
set(LITERT_BUILD_CONFIG_DISABLE_NPU 0)
configure_file(
  ${litert_SOURCE_DIR}/litert/build_common/build_config.h.in
  ${CMAKE_BINARY_DIR}/litert_generated/litert/build_common/build_config.h
)

add_library(litert_c SHARED IMPORTED)
set_target_properties(litert_c PROPERTIES
  IMPORTED_LOCATION ${CMAKE_BINARY_DIR}/litert_lib/ai_edge_litert/libLiteRt.so
  IMPORTED_NO_SONAME TRUE
  INTERFACE_INCLUDE_DIRECTORIES
    "${litert_SOURCE_DIR};${CMAKE_BINARY_DIR}/litert_generated"
)

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
FetchContent_MakeAvailable(googletest json stb)

# Runs the models; no dependencies other than LiteRT.
add_library(perception
  litert_model.cc
  rfdetr.cc
)
target_include_directories(perception PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(perception PUBLIC litert_c)

enable_testing()
include(GoogleTest)

add_executable(perception_test
  models_test.cc
  rfdetr_test.cc
  test_util.cc
)
target_include_directories(perception_test PRIVATE ${stb_SOURCE_DIR})
target_link_libraries(perception_test
  perception GTest::gtest_main nlohmann_json::nlohmann_json)
target_compile_definitions(perception_test PRIVATE
  PERCEPTION_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/.."
)
gtest_discover_tests(perception_test DISCOVERY_TIMEOUT 60)
