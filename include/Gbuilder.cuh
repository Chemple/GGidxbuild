#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace Gbuilder {
namespace Gpu {

template <
        uint32_t grid_size,
        uint32_t block_size,
        uint32_t qeury_num,
        uint32_t base_num,
        uint32_t dim,
        typename data_type = float,
        typename id_type = uint32_t>
__global__ void match_kernel(
        id_type* gt_ids,
        data_type* base_data,
        id_type* top1_ret_graph,
        id_type* top1_tmp_graph,
        id_type* top1_reverse_graph,
        id_type* match_ret_graph,
        id_type* match_tmp_graph,
        id_type* match_reverse_graph,
        int8_t* top1_bit_set,
        int8_t* match_bit_set) {
    constexpr auto stride = block_size * grid_size;
    auto thread_idx = threadIdx.x + block_size * blockIdx.x;

    for (auto query_idx = 0; query_idx < qeury_num; query_idx += stride) {
        auto gt_idx = 0;
    }
}

} // namespace Gpu
} // namespace Gbuilder
