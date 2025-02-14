#pragma once

#include "Ghashset.cuh"
#include <cstdint>
// #include <__clang_cuda_intrinsics.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace Gbuilder {
namespace Gpu {

template <uint32_t grid_size, uint32_t block_size, uint32_t query_num,
          uint32_t topk, uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF,
          typename data_type = float, typename id_type = uint32_t>
__global__ void match_kernel(id_type* gt_ids, id_type* top1_match_graph,
                             id_type* match_match_graph) {
  // NOTE(shiwen): max_in_degree is same as topk.
  static_assert(max_in_degree == topk);
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  for (auto query_idx = thread_idx; query_idx < query_num;
       query_idx += stride) {
    auto top1_base_id = gt_ids[query_idx * topk + 0];
    auto gt_idx = 1;
    auto base_id = gt_ids[query_idx * topk + gt_idx];
    // top1 match
    if (atomicCAS(&top1_match_graph[top1_base_id * max_in_degree + 0], tomb,
                  base_id) == tomb) {
      for (gt_idx = 2; gt_idx < topk; gt_idx++) {
        base_id = gt_ids[query_idx * topk + gt_idx];
        auto neighbor_idx = gt_idx - 1;
        top1_match_graph[top1_base_id * max_in_degree + neighbor_idx] = base_id;
      }
    }
    // greedy match
    gt_idx = 0;
    base_id = gt_ids[query_idx * topk + gt_idx];
    auto greedy_match_idx = 1;
    auto greedy_match_base_id = gt_ids[query_idx * topk + greedy_match_idx];
    // find the greedy_match_idx and its base id
    while (
        greedy_match_idx < topk &&
        atomicCAS(&match_match_graph[greedy_match_base_id * max_in_degree + 0],
                  tomb, base_id) != tomb) {
      greedy_match_idx++;
      greedy_match_base_id = gt_ids[query_idx * topk + greedy_match_idx];
    }
    // find the greedy match.
    if (greedy_match_idx < topk) {
      auto neighbor_idx = 1;
      for (gt_idx = 0; gt_idx < topk; gt_idx++) {
        if (gt_idx == greedy_match_idx) {
          continue;
        }
        base_id = gt_ids[query_idx * topk + gt_idx];
        match_match_graph[greedy_match_base_id * max_in_degree + neighbor_idx] =
            base_id;
        neighbor_idx++;
      }
    }
  }
}

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF,
          bool is_strict = false, typename data_type = float,
          typename id_type = uint32_t>
__global__ void rng_prune_kernel(data_type* base_data, id_type* graph,
                                 id_type* pruned_graph) {
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  for (auto base_idx = thread_idx; base_idx < base_num; base_idx += stride) {
    // insert base id of the first neibour first.
    auto neibour_idx = 0;
    auto neibour_base_id = graph[base_idx * max_in_degree + 0];
    pruned_graph[base_idx * max_in_degree + 0] = neibour_base_id;
  }
}

}  // namespace Gpu
}  // namespace Gbuilder
