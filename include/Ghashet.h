#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace Gbuilder {
namespace Gpu {

template <
        typename id_type = uint16_t,
        uint32_t hash_set_size = 10 * 1024 * 1024>
void __global__ init_kernel(id_type* hash_set_array) {
    auto thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
    auto stride = blockDim.x * gridDim.x;
    for (auto i = thread_idx; i < hash_set_size; i += stride) {
        hash_set_array[i] = 0;
    }
}

template <
        typename id_type = uint16_t,
        uint32_t hash_set_size = 10 * 1024 * 1024>
struct HashSet {
    void init() {
        cudaMalloc(&hash_set_array_, hash_set_size * sizeof(id_type));
        init_kernel<<<72, 128>>>(hash_set_array_);
    }

    auto __device__
    check_empty_or_not_insert(id_type u, id_type* hash_set_array) -> bool {
        return !atomicCAS(hash_set_array + u, 0, 1);
    }

    id_type* hash_set_array_;
    enum Kempty { Empty = 0 };
};

} // namespace Gpu
} // namespace Gbuilder
