cmake_minimum_required(VERSION 3.24)

# Metal 3.0 shaders first run on macOS 13. Must be set before project().
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS version")

project(MarmotteMetalNative LANGUAGES OBJCXX)

if(NOT APPLE)
    message(FATAL_ERROR "Marmotte Metal requires macOS.")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(FATAL_ERROR "Marmotte Metal requires a 64-bit toolchain.")
endif()

# Explicit source list: experimental shaders are never compiled by accident.
set(METAL_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/Add.metal"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/SwiGlu.metal"
)
set(METAL_KERNEL_LIBRARY "${CMAKE_BINARY_DIR}/marmotte_kernels.metallib")

# Metal enables fast math by default, which may assume that NaN and infinity never occur.
# Marmotte's kernels must match the CPU reference on those values.
add_custom_command(
    OUTPUT "${METAL_KERNEL_LIBRARY}"
    COMMAND xcrun -sdk macosx metal
        -std=metal3.0
        -fno-fast-math
        -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET}
        -o "${METAL_KERNEL_LIBRARY}"
        ${METAL_KERNEL_SOURCES}
    DEPENDS ${METAL_KERNEL_SOURCES}
    VERBATIM
)
add_custom_target(marmotte_metal_kernels DEPENDS "${METAL_KERNEL_LIBRARY}")

add_library(marmotte_metal SHARED
    MetalRuntime.mm
    kernels/Add.mm
    kernels/SwiGlu.mm
)
add_dependencies(marmotte_metal marmotte_metal_kernels)

target_compile_options(marmotte_metal PRIVATE -fobjc-arc)

# The compiled shaders live in a section of the library, so that deployment stays a single file.
target_link_options(marmotte_metal PRIVATE
    "LINKER:-sectcreate,__TEXT,__metallib,${METAL_KERNEL_LIBRARY}"
)

# Command-line processes get no default Metal device unless Core Graphics is linked.
target_link_libraries(marmotte_metal PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework CoreGraphics"
)

set_target_properties(marmotte_metal PROPERTIES
    OBJCXX_STANDARD 17
    OBJCXX_STANDARD_REQUIRED YES
    OBJCXX_VISIBILITY_PRESET hidden
    LINK_DEPENDS "${METAL_KERNEL_LIBRARY}"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/native/$<0:>"
)
