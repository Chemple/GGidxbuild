#pragma once

#include "GBitonicSort.cuh"
#include "Ghashset.cuh"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

namespace Gbuilder {
namespace Gpu {

template <uint32_t grid_size, uint32_t block_size, uint32_t query_num,
          uint32_t topk, uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF,
          typename data_type = float, typename id_type = uint32_t>
__global__ void match_kernel(id_type const* __restrict__ gt_ids,
                             id_type* __restrict__ top1_match_graph,
                             id_type* __restrict__ match_match_graph) {
  // NOTE(shiwen): max_in_degree is same as topk.
  static_assert(max_in_degree == topk);
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  // TODO(shiwen): use shared memory.
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

// each warp is assigned to calculate all the distance between base_vector_id
// and its neighbor.
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF, uint32_t dim,
          uint32_t shared_memory_size, typename data_type = float,
          typename id_type = uint32_t>
__global__ void compute_ip_distance_kernel(
    data_type const* __restrict__ base_data, id_type const* __restrict__ graph,
    data_type* __restrict__ neighbor_distance) {
  constexpr auto lane_width = 32;
  constexpr auto warp_per_block = block_size / lane_width;
  // each warp need to store the base vector and the neighbor vector to
  // calculate the distance between them.
  constexpr auto shared_memory_size_per_warp = dim * sizeof(data_type) * 2;
  constexpr auto global_warp_num = block_size * grid_size / lane_width;
  // NOTE(shiwen): check the allocation of shared memory is correct.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // NOTE(shiwen): use offset instead of bytes address.
  extern __shared__ data_type sdata[];

  auto const global_warp_id =
      (threadIdx.x + blockDim.x * blockIdx.x) / lane_width;
  auto const local_warp_id = threadIdx.x / lane_width;
  auto const lane_id = threadIdx.x % lane_width;
  auto const base_vector_offset =
      local_warp_id * shared_memory_size_per_warp / sizeof(data_type);
  auto const neighbor_vector_offset = base_vector_offset + dim;

  for (auto base_vector_id = global_warp_id; base_vector_id < base_num;
       base_vector_id += global_warp_num) {
    for (auto i = lane_id; i < dim; i += lane_width) {
      assert(i < dim);
      sdata[base_vector_offset + i] = base_data[base_vector_id * dim + i];
    }

    // NOTE(shiwen): in case of dim % 32 != 0
    // TODO(shiwen): need this primitive?
    __syncwarp();

    for (auto neighbor_idx = 0; neighbor_idx < max_in_degree; neighbor_idx++) {
      assert(neighbor_idx < max_in_degree);
      auto neighbor_base_id =
          graph[base_vector_id * max_in_degree + neighbor_idx];
      if (neighbor_base_id == tomb) {
        break;
      }
      for (auto i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sdata[neighbor_vector_offset + i] =
            base_data[neighbor_base_id * dim + i];
      }

      // NOTE(shiwen): in case of dim % 32 != 0
      // TODO(shiwen): need this primitive?
      __syncwarp();

      data_type sum = 0;
      for (auto i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum +=
            sdata[base_vector_offset + i] * sdata[neighbor_vector_offset + i];
      }
      // warp level reduce.
      for (auto offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      if (lane_id == 0) {
        assert(neighbor_idx < max_in_degree);
        neighbor_distance[base_vector_id * max_in_degree + neighbor_idx] = -sum;
      }

      // NOTE(shiwen): in case of dim % 32 != 0
      // TODO(shiwen): need this primitive?
      __syncwarp();
    }
  }
}

// FIXME(shiwen): too much __syncwarp()!!!
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t max_in_degree, uint32_t tomb = 0xFFFFFFFF, uint32_t dim,
          uint32_t shared_memory_size, typename data_type = float,
          typename id_type = uint32_t>
__global__ void compute_and_sort_ip_distance_kernel(
    data_type const* __restrict__ base_data, id_type* __restrict__ graph,
    data_type* __restrict__ neighbor_distance) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      dim * sizeof(data_type) * 2 +
      max_in_degree * (sizeof(data_type) + sizeof(id_type));
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  extern __shared__ __align__(sizeof(data_type)) uint8_t shared_memory[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  // Shared memory layout per warp
  data_type* base_vector = reinterpret_cast<data_type*>(
      &shared_memory[local_warp_id *
                     (dim * sizeof(data_type) * 2 +
                      max_in_degree * (sizeof(data_type) + sizeof(id_type)))]);
  data_type* neighbor_vector = base_vector + dim;
  data_type* distance_sdata = neighbor_vector + dim;
  id_type* neighbor_id_sdata =
      reinterpret_cast<id_type*>(distance_sdata + max_in_degree);

  for (uint32_t base_vector_id = global_warp_id; base_vector_id < base_num;
       base_vector_id += global_warp_num) {
    // Load base vector
    for (uint32_t i = lane_id; i < dim; i += lane_width) {
      assert(i < dim);
      base_vector[i] = base_data[base_vector_id * dim + i];
    }
    __syncwarp();

    uint32_t min_invalid_neighbor_idx = max_in_degree - 1;
    // Process neighbors and collect distances
    for (uint32_t neighbor_idx = 0; neighbor_idx < max_in_degree;
         ++neighbor_idx) {
      assert(neighbor_idx < max_in_degree);
      id_type const neighbor_id =
          graph[base_vector_id * max_in_degree + neighbor_idx];
      if (neighbor_id == tomb) {
        min_invalid_neighbor_idx = neighbor_idx;
        // NOTE(shiwen): set all distance of tomb id to FLT_MAX
        if (lane_id == 0) {
          for (uint32_t neighbor_idx = min_invalid_neighbor_idx;
               neighbor_idx < max_in_degree; neighbor_idx++) {
            assert(neighbor_idx < max_in_degree);
            distance_sdata[neighbor_idx] = FLT_MAX;
          }
        }
        __syncwarp();
        break;
      }

      // Load neighbor vector
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        neighbor_vector[i] = base_data[neighbor_id * dim + i];
      }
      __syncwarp();

      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += base_vector[i] * neighbor_vector[i];
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      __syncwarp();

      if (lane_id == 0) {
        assert(neighbor_idx < max_in_degree);
        distance_sdata[neighbor_idx] = -sum;
        neighbor_id_sdata[neighbor_idx] = neighbor_id;
      }
      __syncwarp();
    }

    __syncwarp();

    // FIXME(shiwen): check the 3rd template.
    // FIXME(shiwen): check the 3rd template.
    // FIXME(shiwen): check the 3rd template.
    // FIXME(shiwen): check the 3rd template.
    // FIXME(shiwen): check the 3rd template.
    // FIXME(shiwen): check the 3rd template.
    // FIXME(shiwen): check the 3rd template.
    // FIXME(shiwen): check the 3rd template.
    // FIXME(shiwen): check the 3rd template.
    warp_sort<data_type, id_type, max_in_degree, lane_width>(
        distance_sdata, neighbor_id_sdata, true);

    __syncwarp();

    for (uint32_t i = lane_id; i < max_in_degree; i += lane_width) {
      assert(i < max_in_degree);
      assert(i < max_in_degree);
      if (i < min_invalid_neighbor_idx) {
        neighbor_distance[base_vector_id * max_in_degree + i] =
            distance_sdata[i];
        graph[base_vector_id * max_in_degree + i] = neighbor_id_sdata[i];
      } else if (i >= min_invalid_neighbor_idx) {
        neighbor_distance[base_vector_id * max_in_degree + i] = FLT_MAX;
        graph[base_vector_id * max_in_degree + i] = tomb;
      }
    }
    __syncwarp();
  }
}

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t graph_max_in_degree, uint32_t pruned_graph_max_in_degree,
          uint32_t tomb = 0xFFFFFFFF, uint32_t dim, bool is_strict = true,
          typename data_type = float, typename id_type = uint32_t>
__global__ void rng_prune_kernel(
    data_type const* __restrict__ base_data, id_type const* __restrict__ graph,
    data_type const* __restrict__ neighbor_distance,
    id_type* __restrict__ pruned_graph) {
  constexpr auto stride = block_size * grid_size;
  auto thread_idx = threadIdx.x + block_size * blockIdx.x;

  for (auto base_id = thread_idx; base_id < base_num; base_id += stride) {
    // insert base id of the first neighbor first.
    auto neighbor_idx = 0;
    auto pruned_graph_neighbor_idx = 0;
    assert(neighbor_idx < graph_max_in_degree);
    auto neighbor_base_id = graph[base_id * graph_max_in_degree + neighbor_idx];
    if (neighbor_base_id == tomb) {
      continue;
    }
    assert(pruned_graph_neighbor_idx < pruned_graph_max_in_degree);
    pruned_graph[base_id * pruned_graph_max_in_degree +
                 pruned_graph_neighbor_idx] = neighbor_base_id;
    neighbor_idx++;
    auto explore_flag = true;
    for (; (explore_flag && neighbor_idx < graph_max_in_degree);
         neighbor_idx++) {
      assert(neighbor_idx < graph_max_in_degree);
      neighbor_base_id = graph[base_id * graph_max_in_degree + neighbor_idx];
      if (neighbor_base_id == tomb) {
        break;
      }
      auto compare_idx = 0;
      for (; compare_idx <= pruned_graph_neighbor_idx; compare_idx++) {
        assert(compare_idx < pruned_graph_max_in_degree);
        auto compare_base_id =
            pruned_graph[base_id * pruned_graph_max_in_degree + compare_idx];
        assert(compare_base_id != tomb);
        // TODO(shiwen): FP16?
        data_type distance = 0;
        for (auto i = 0; i < dim; i++) {
          assert(i < dim);
          auto x_value = base_data[neighbor_base_id * dim + i];
          // TODO(shiwen): use shared memory.
          assert(i < dim);
          auto y_value = base_data[compare_base_id * dim + i];
          distance += x_value * y_value;
        }
        distance = -distance;
        assert(neighbor_idx < graph_max_in_degree);
        if (distance <
            neighbor_distance[base_id * graph_max_in_degree + neighbor_idx]) {
          break;
        }
      }
      // pass all the distance tests.
      if (compare_idx > pruned_graph_neighbor_idx) {
        pruned_graph_neighbor_idx++;
        // NOTE(shiwen):
        assert(pruned_graph_neighbor_idx < pruned_graph_max_in_degree);
        assert(base_id != neighbor_base_id);
        assert(pruned_graph_neighbor_idx < pruned_graph_max_in_degree);
        pruned_graph[base_id * pruned_graph_max_in_degree +
                     pruned_graph_neighbor_idx] = neighbor_base_id;
        if (pruned_graph_neighbor_idx == pruned_graph_max_in_degree - 1) {
          explore_flag = false;
          break;
        }
      }
    }
    // TODO(shiwen): slight RNG prune?
  }
}

}  // namespace Gpu
}  // namespace Gbuilder
