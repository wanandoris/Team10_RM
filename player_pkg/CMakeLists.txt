cmake_minimum_required(VERSION 3.8)
project(player_pkg)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 14)

# 查找依赖包
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(gazebo_msgs REQUIRED)
find_package(geometry_msgs REQUIRED) 
find_package(OpenCV REQUIRED)
find_package(pose_msg REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(referee_pkg REQUIRED)
find_package(sensor_msgs REQUIRED)

# 手动查找 ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS
        /usr/include
        /usr/local/include
        /opt/onnxruntime/include
        $ENV{ONNXRUNTIME_HOME}/include
)

find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
        /opt/onnxruntime/lib
        $ENV{ONNXRUNTIME_HOME}/lib
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(ONNXRUNTIME_FOUND TRUE)
    message(STATUS "Found ONNX Runtime: ${ONNXRUNTIME_LIBRARY}")
    message(STATUS "ONNX Runtime include dir: ${ONNXRUNTIME_INCLUDE_DIR}")
else()
    set(ONNXRUNTIME_FOUND FALSE)
    message(WARNING "ONNX Runtime not found, DL model will be disabled")
endif()

add_executable(TestNode src/training_node.cpp)
ament_target_dependencies(TestNode
  rclcpp
  std_msgs
  OpenCV 
  geometry_msgs
  cv_bridge
  pose_msg
  referee_pkg
  sensor_msgs
)

# 链接OpenCV库
target_link_libraries(TestNode ${OpenCV_LIBS})

# 链接ONNX Runtime库（如果找到）
if(ONNXRUNTIME_FOUND)
    target_include_directories(TestNode PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(TestNode ${ONNXRUNTIME_LIBRARY})
    target_compile_definitions(TestNode PRIVATE HAVE_ONNXRUNTIME)
endif()

# 安装TestNode
install(TARGETS TestNode
  DESTINATION lib/${PROJECT_NAME}
)

# 包含OpenCV头文件
include_directories(${OpenCV_INCLUDE_DIRS})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()