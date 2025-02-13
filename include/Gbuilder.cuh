#pragma once

#include "Ghashset.cuh"
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>

namespace Gbuilder {
namespace Gpu {

template <uint32_t grid_size, uint32_t block_size, uint32_t qeury_num,
          uint32_t topk, uint32_t base_num, uint32_t dim,
          uint32_t tomb = 0xFFFFFFFF, typename data_type = float,
          typename id_type = uint32_t>
__global__ void match_kernel(id_type* gt_ids, data_type* base_data,
                             id_type* top1_ret_graph, id_type* top1_tmp_graph,
                             id_type* top1_reverse_graph,
                             id_type* match_ret_graph, id_type* match_tmp_graph,
                             id_type* match_reverse_graph, int8_t* top1_set,
                             int8_t* match_set) {
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  for (auto query_idx = 0; query_idx < qeury_num; query_idx += stride) {
    auto top1_base_id = gt_ids[query_idx * topk + 0];
    auto gt_idx = 1;
    auto base_id = gt_ids[query_idx * topk + gt_idx];
    // NOTE(shiwen): each first elem is inited to be tomb.
    if (atomicCAS(&top1_ret_graph[top1_base_id * topk + 0], tomb, base_id) ==
        tomb) {
      for (gt_idx = 2; gt_idx < topk; gt_idx++) {
        base_id = gt_ids[query_idx * topk + gt_idx];
        top1_ret_graph[top1_base_id * topk + gt_idx - 1] = base_id;
      }
    } 
  }
}

}  // namespace Gpu
}  // namespace Gbuilder
