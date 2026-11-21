cmake_minimum_required(VERSION 3.24)
project(CudaWeightStorageMeasurement LANGUAGES CXX CUDA)
add_executable(cuda_weight_storage CudaWeightStorage.cu)
set_target_properties(cuda_weight_storage PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED YES
    CUDA_RUNTIME_LIBRARY Static
    MSVC_RUNTIME_LIBRARY MultiThreaded
)
