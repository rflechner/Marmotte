// Optional storage experiment, separate from the backend and normal tests. See docs/cuda.md.
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

// Only a private file view is registered, so unrelated managed objects are never pinned.
class ReadOnlyMapping {
#if defined(_WIN32)
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int file = -1;
#endif
public:
    void* address = nullptr;
    std::size_t bytes = 0;
    explicit ReadOnlyMapping(const char* path) {
#if defined(_WIN32)
        file = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        LARGE_INTEGER length{};
        if (file != INVALID_HANDLE_VALUE && GetFileSizeEx(file, &length) && length.QuadPart > 0) {
            bytes = static_cast<std::size_t>(length.QuadPart);
            mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (mapping) address = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
        }
#else
        file = open(path, O_RDONLY);
        struct stat information{};
        if (file >= 0 && fstat(file, &information) == 0 && information.st_size > 0) {
            bytes = static_cast<std::size_t>(information.st_size);
            address = mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, file, 0);
            if (address == MAP_FAILED) address = nullptr;
        }
#endif
        if (!address) {
            close();
            throw std::runtime_error("Cannot map the benchmark file read-only.");
        }
    }
    void close() {
#if defined(_WIN32)
        if (address) UnmapViewOfFile(address);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
        if (address) munmap(address, bytes);
        if (file >= 0) ::close(file);
#endif
        address = nullptr;
    }
    ~ReadOnlyMapping() { close(); }
    ReadOnlyMapping(const ReadOnlyMapping&) = delete;
    ReadOnlyMapping& operator=(const ReadOnlyMapping&) = delete;
};

// Every byte contributes to an exact checksum. This measures a complete device read,
// not quantized arithmetic or model inference. Identical launches exercise both storage paths.
__global__ void sumBytes(const unsigned char* bytes, std::size_t count, unsigned long long* sums) {
    __shared__ unsigned long long partial[256];
    unsigned long long sum = 0;
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += gridDim.x * blockDim.x) sum += bytes[index];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (int width = 128; width > 0; width /= 2) {
        if (threadIdx.x < width) partial[threadIdx.x] += partial[threadIdx.x + width];
        __syncthreads();
    }
    if (threadIdx.x == 0) sums[blockIdx.x] = partial[0];
}

using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct Trial {
    cudaStream_t stream = nullptr;
    void* registered = nullptr;
    std::vector<void*> allocations;
    unsigned long long* sums = nullptr;
    Trial() {
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        const auto status = cudaMalloc(&sums, 8 * 1024 * sizeof(unsigned long long));
        if (status != cudaSuccess) {
            cudaStreamDestroy(stream);
            check(status);
        }
    }
    ~Trial() {
        cudaStreamSynchronize(stream);
        for (auto pointer : allocations) cudaFree(pointer);
        if (registered) cudaHostUnregister(registered);
        cudaFree(sums);
        cudaStreamDestroy(stream);
    }
};

unsigned long long measure(ReadOnlyMapping& mapping, std::size_t offset, std::size_t matrixBytes,
                           std::size_t stagingBytes, bool mapped) {
    Trial trial;
    std::vector<const unsigned char*> weights;
    weights.reserve(8);
    trial.allocations.reserve(8);
    auto start = Clock::now();
    if (mapped) {
        check(cudaHostRegister(mapping.address, mapping.bytes, cudaHostRegisterMapped | cudaHostRegisterReadOnly));
        trial.registered = mapping.address;
        void* device = nullptr;
        check(cudaHostGetDevicePointer(&device, mapping.address, 0));
        for (int i = 0; i < 8; ++i)
            weights.push_back(static_cast<unsigned char*>(device) + offset + i * matrixBytes);
    } else {
        std::vector<unsigned char> staging(stagingBytes);
        for (int i = 0; i < 8; ++i) {
            void* device = nullptr;
            check(cudaMalloc(&device, matrixBytes));
            trial.allocations.push_back(device);
            weights.push_back(static_cast<unsigned char*>(device));
            for (std::size_t position = 0; position < matrixBytes; position += stagingBytes) {
                const auto count = std::min(stagingBytes, matrixBytes - position);
                std::memcpy(staging.data(), static_cast<unsigned char*>(mapping.address) + offset + i * matrixBytes + position, count);
                check(cudaMemcpyAsync(static_cast<unsigned char*>(device) + position, staging.data(), count,
                                      cudaMemcpyHostToDevice, trial.stream));
                check(cudaStreamSynchronize(trial.stream));
            }
        }
    }
    const auto loadMs = elapsed(start);
    std::vector<unsigned long long> sums(8 * 1024);
    // Warm the kernel before measuring ten full reads of all eight matrices.
    start = Clock::now();
    for (int i = 0; i < 8; ++i) sumBytes<<<1024, 256, 0, trial.stream>>>(weights[i], matrixBytes, trial.sums + i * 1024);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(trial.stream));
    const auto firstReadMs = elapsed(start);
    start = Clock::now();
    for (int repeat = 0; repeat < 10; ++repeat)
        for (int i = 0; i < 8; ++i) sumBytes<<<1024, 256, 0, trial.stream>>>(weights[i], matrixBytes, trial.sums + i * 1024);
    check(cudaGetLastError());
    check(cudaStreamSynchronize(trial.stream));
    const auto readMs = elapsed(start) / 10;
    check(cudaMemcpy(sums.data(), trial.sums, sums.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    unsigned long long sum = 0;
    for (auto partial : sums) sum += partial;
    std::printf("%s: load %.2f ms, first read %.2f ms, repeated read %.2f ms, checksum %llu\n",
        mapped ? "mapped host" : (stagingBytes == matrixBytes ? "whole staging" : "4 MiB staging"),
        loadMs, firstReadMs, readMs, sum);
    return sum;
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 5) throw std::runtime_error("Usage: cuda_weight_storage file.gguf dataOffset matrixBytes vectorBytes (see CudaWeights.fsx).");
        const auto offset = std::stoull(argv[2]);
        const auto matrixBytes = std::stoull(argv[3]);
        const auto vectorBytes = std::stoull(argv[4]);
        ReadOnlyMapping mapping(argv[1]);
        if (offset > mapping.bytes || vectorBytes > mapping.bytes - offset || matrixBytes == 0 ||
            matrixBytes > (mapping.bytes - offset - vectorBytes) / 8)
            throw std::runtime_error("Payload ranges exceed the file.");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, 0));
        std::printf("%s: canMapHost=%d, registerReadOnly=%d, unified=%d\n", properties.name,
            properties.canMapHostMemory, properties.hostRegisterReadOnlySupported, properties.unifiedAddressing);
        if (!properties.canMapHostMemory || !properties.hostRegisterReadOnlySupported)
            throw std::runtime_error("Read-only registered mappings are unsupported on this device.");
        check(cudaSetDevice(0));
        // Warm every file page, including the normalization vector, outside all timed regions.
        volatile unsigned long long pageSum = 0;
        for (std::size_t i = 0; i < mapping.bytes; i += 4096)
            pageSum = pageSum + static_cast<unsigned char*>(mapping.address)[i];
        std::printf("Warm-page checksum: %llu\n", pageSum);
        for (int trial = 0; trial < 5; ++trial) {
            const auto baseline = measure(mapping, offset, matrixBytes, matrixBytes, false);
            if (measure(mapping, offset, matrixBytes, 4 * 1024 * 1024, false) != baseline ||
                measure(mapping, offset, matrixBytes, 0, true) != baseline)
                throw std::runtime_error("Stored byte checksums differ.");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
