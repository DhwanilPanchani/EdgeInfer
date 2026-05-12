cmake_minimum_required(VERSION 3.20)
project(EdgeInfer)

set(CMAKE_CXX_STANDARD 17)

add_executable(edge_infer_stub cpp/stub.cpp)

# In production this would link the ONNX Runtime C++ SDK:
# find_package(onnxruntime REQUIRED)
# target_link_libraries(edge_infer_stub PRIVATE onnxruntime::onnxruntime)
