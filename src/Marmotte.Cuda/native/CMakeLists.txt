cmake_minimum_required(VERSION 3.24)
project(MarmotteCudaNative LANGUAGES CXX CUDA)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Marmotte CUDA requires a 64-bit toolchain.")
endif()

add_library(marmotte_cuda SHARED CudaRuntime.cu kernels/Add.cu)
set_target_properties(marmotte_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_RUNTIME_LIBRARY Static
    # Match the static CUDA runtime's C runtime, including in Debug builds.
    MSVC_RUNTIME_LIBRARY MultiThreaded
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native/$<0:>"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native/$<0:>"
)
