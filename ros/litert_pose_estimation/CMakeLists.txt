cmake_minimum_required(VERSION 3.24)
project(litert_pose_estimation CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(message_filters REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

# The LiteRT pipeline (../../cpp): LiteRT runtime, GPU accelerator, models.
set(PERCEPTION_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)
add_subdirectory(${PERCEPTION_ROOT}/cpp ${CMAKE_BINARY_DIR}/perception)
get_target_property(LITERT_RUNTIME_LIB litert_c IMPORTED_LOCATION)
get_filename_component(LITERT_LIBRARY_DIR ${LITERT_RUNTIME_LIB} DIRECTORY)
set(INSTALLED_LITERT_DIR ${CMAKE_INSTALL_PREFIX}/lib/${PROJECT_NAME}/litert)

add_library(pose_estimation_node SHARED src/pose_estimation_node.cpp)
target_include_directories(pose_estimation_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(pose_estimation_node
  perception
  rclcpp::rclcpp
  rclcpp_components::component
  message_filters::message_filters
  ${sensor_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${vision_msgs_TARGETS}
  ${visualization_msgs_TARGETS}
)
# The installed node loads LiteRT's GPU accelerator from its own lib dir.
target_compile_definitions(pose_estimation_node PRIVATE
  PERCEPTION_LITERT_LIBRARY_DIR="${INSTALLED_LITERT_DIR}")
set_target_properties(pose_estimation_node PROPERTIES
  INSTALL_RPATH "${INSTALLED_LITERT_DIR}")
rclcpp_components_register_node(pose_estimation_node
  PLUGIN "litert_pose_estimation::PoseEstimationNode"
  EXECUTABLE pose_estimation)

add_executable(pose_estimation_main src/main.cpp)
target_link_libraries(pose_estimation_main pose_estimation_node)
set_target_properties(pose_estimation_main PROPERTIES
  OUTPUT_NAME pose_estimation_node
  INSTALL_RPATH "${INSTALLED_LITERT_DIR};${CMAKE_INSTALL_PREFIX}/lib")

install(TARGETS pose_estimation_node
  LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(TARGETS pose_estimation_main RUNTIME DESTINATION lib/${PROJECT_NAME})
install(FILES ${LITERT_LIBRARY_DIR}/libLiteRt.so
              ${LITERT_LIBRARY_DIR}/libLiteRtWebGpuAccelerator.so
        DESTINATION lib/${PROJECT_NAME}/litert)
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  # End-to-end golden test: the node on the IOC pose estimator service's
  # recorded inputs, through ROS topics and the ~/estimate service.
  ament_add_gtest(golden_test test/golden_test.cpp TIMEOUT 1800)
  target_link_libraries(golden_test pose_estimation_node perception_test_util)
  target_compile_definitions(golden_test PRIVATE
    PERCEPTION_SOURCE_ROOT="${PERCEPTION_ROOT}")
endif()

ament_package()
