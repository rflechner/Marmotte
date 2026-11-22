#include "../CudaContext.h"
#include <cstddef>
#include <limits>

namespace {
std::size_t elementBytes(CudaEncoding encoding) {
    switch (encoding) {
        case CudaEncoding::F32: return 4;
        case CudaEncoding::F16: return 2;
        default: return 0;
    }
}
}

// Copies rows [firstRow, firstRow + rows) of a [sourceRows; columns] tensor into a
// [rows; columns] destination. All counts are elements or rows, not bytes; the managed
// backend validates shapes and ranges and handles empty copies before native dispatch.
// Contiguous row-major rows form one byte range, so a single device-to-device copy on the
// context's stream replaces a kernel; it is ordered after earlier work and may complete
// after return. Both allocations must be registered with these exact element counts and
// one encoding. Distinct allocations never overlap; the only aliasing the contract allows
// is the identity copy (same buffer, firstRow 0, every row), which is a no-op here because
// overlapping device copies are undefined.
MARMOTTE_EXPORT int marmotte_cuda_copy_rows(CudaContext* context, void* source, int sourceRows,
                                            int firstRow, void* destination, int rows, int columns) {
    if (sourceRows <= 0 || rows <= 0 || columns <= 0 || firstRow < 0 ||
        firstRow > sourceRows - rows) return cudaErrorInvalidValue;
    const auto sourceCount = static_cast<long long>(sourceRows) * columns;
    const auto destinationCount = static_cast<long long>(rows) * columns;
    if (sourceCount > std::numeric_limits<int>::max()) return cudaErrorInvalidValue;

    const auto allocation = findBuffer(context, source, static_cast<int>(sourceCount));
    if (!allocation || !containsBuffer(context, destination, static_cast<int>(destinationCount), allocation->encoding)) {
        return cudaErrorInvalidValue;
    }
    const auto bytesPerElement = elementBytes(allocation->encoding);
    if (bytesPerElement == 0) return cudaErrorNotSupported;
    if (source == destination) return cudaSuccess;

    const auto selected = cudaSetDevice(context->device);
    if (selected != cudaSuccess) return selected;
    const auto offset = static_cast<std::size_t>(firstRow) * static_cast<std::size_t>(columns) * bytesPerElement;
    const auto bytes = static_cast<std::size_t>(destinationCount) * bytesPerElement;
    return cudaMemcpyAsync(destination, static_cast<const char*>(source) + offset, bytes,
                           cudaMemcpyDeviceToDevice, context->stream);
}
