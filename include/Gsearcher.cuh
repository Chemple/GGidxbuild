#pragma once
#include "GBitonicSort.cuh"
#include "Ghashset.cuh"
#include "Ghashtable.cuh"
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <unistd.h>
namespace Gbuilder {
namespace Gpu {

template <typename id_type = uint32_t, typename data_type = float,
          uint32_t dim = 128, uint32_t Km = 32, uint32_t Kp = 8,
          uint32_t Kd = 16, uint32_t hash_table_size = 1 << 12>
struct search_warp_state {
  data_type query_vector_[dim];
  data_type list_vector_[dim];
  id_type node_id_list_[Km + Kp * Kd];
  data_type node_distance_list_[Km + Kp * Kd];
  // FIXME(shiwen): change the size of hash table.
  // TODO(shiwen): if we use one warp to handle one query, is it possible to use
  // warp-level primitive to replace atomicCAS?
  HashTable<uint32_t, hash_table_size> visit_;
};

// set the MSB of raw_node_id to 1.
__device__ __forceinline__ void set_to_parent(uint32_t& raw_node_id) {
  // assert the MSB of raw_node_id is 0
  assert((0X80000000 & raw_node_id) == 0);
  raw_node_id = 0X80000000 | raw_node_id;
}

// NOTE(shiwen): this only work when Km <=32
template <uint32_t Km = 128, uint32_t Kp = 14, uint32_t Kd = 64,
          typename id_type = uint32_t, typename data_type = float,
          uint32_t tomb = 0XFFFFFFFF>
__device__ __forceinline__ bool collect_top_p(
    id_type* __restrict__ topm_ids, id_type* __restrict__ candidate_ids,
    uint32_t const& lane_id) {
  // NOTE(shiwen): Kp must be smaller than lane_width.
  static_assert(Km <= 32);
  static_assert(Kp < 32);
  id_type id = topm_ids[lane_id];
  bool is_valid = (id & 0x80000000) == 0;
  uint32_t mask = __ballot_sync(0xFFFFFFFF, is_valid);
  auto index = __popc(mask & ((1 << lane_id) - 1));
  auto explore_flag = false;

  for (auto i = 0; i < Kp; i++) {
    uint32_t vote = __ballot_sync(0xFFFFFFFF, is_valid && (index == i));
    auto src_lane_id = -1;
    uint32_t candidate_id = tomb;
    if (vote != 0) {
      explore_flag = true;
      // find the leader thread.(from low to high)
      src_lane_id = __ffs(vote) - 1;
      candidate_id = __shfl_sync(0xFFFFFFFF, id, src_lane_id);
    }
    candidate_ids[i * Kd] = candidate_id;
    if (src_lane_id == lane_id) {
      // set the MSB to 1.
      set_to_parent(topm_ids[lane_id]);
    }
  }
  // FIXME(shiwen): check this return value.
  return explore_flag;
}

template <uint32_t Km = 128, uint32_t Kp = 14, uint32_t Kd = 64,
          typename id_type = uint32_t, typename data_type = float,
          uint32_t tomb = 0XFFFFFFFF>
__device__ __forceinline__ bool collect_top_p_normal(
    id_type* __restrict__ topm_ids, id_type* __restrict__ candidate_ids,
    uint32_t const& lane_id) {
  constexpr uint32_t lane_width = 32;
  assert(Kp < lane_width);
  auto prev = 0;
  for (auto list_idx = lane_id; list_idx < Km && prev < Kp;
       list_idx += lane_width) {
    auto id = topm_ids[list_idx];
    auto is_valid = (id & 0x80000000) == 0;
    auto valid_mask = __ballot_sync(0xFFFFFFFF, is_valid);
    // NOTE(shiwen): the index of topp.
    auto index = __popc(valid_mask & ((1 << lane_id) - 1));
    if (is_valid && index + prev < Kp) {
      candidate_ids[(index + prev) * Kd] = id;
      set_to_parent(topm_ids[list_idx]);
    }
    prev += __popc(valid_mask);
  }
  if (prev == 0) {
    return false;
  }
  if (lane_id == 0) {
    for (auto idx = prev; idx < Kp; idx++) {
      candidate_ids[idx * Kd] = tomb;
    }
  }
  return true;
}

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t dim, uint32_t max_degree, uint32_t shared_memory_size,
          uint32_t Km = 128, uint32_t Kp = 14, uint32_t Kd = 64, uint32_t topk,
          uint32_t tomb = 0XFFFFFFFF, uint32_t hash_table_size = 1 << 12,
          uint32_t reset_iter = 4, typename id_type = uint32_t,
          typename data_type = float>
__global__ void enhance_search(data_type* base_data, id_type* graph,
                               id_type* result) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(search_warp_state<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);
  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  static_assert(shared_memory_size < 49152);
  // the max degree must smaller than Kd.
  static_assert(max_degree <= Kd);
  // the topk must be smaller than Km.
  static_assert(topk < Km);

  extern __shared__
      search_warp_state<id_type, data_type, dim, Km, Kp, Kd, hash_table_size>
          warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  data_type* query_vector_sdata = warp_states[local_warp_id].query_vector_;
  data_type* list_vector_sdata = warp_states[local_warp_id].list_vector_;
  id_type* node_id_list_sdata = warp_states[local_warp_id].node_id_list_;
  data_type* node_distance_list_sdata =
      warp_states[local_warp_id].node_distance_list_;
  HashTable<uint32_t, hash_table_size>* visit_table =
      warp_states[local_warp_id].visit_;

  for (uint32_t query_id = global_warp_id; query_id < base_num; query_id++) {
    // Load base vector
    for (uint32_t i = lane_id; i < dim; i += lane_width) {
      assert(i < dim);
      query_vector_sdata[i] = base_data[query_id * dim + i];
    }

    // FIXME(shiwen): generate the random enter points.
    // gen the enter points.
    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      node_id_list_sdata[Km + i] = query_id + i;
    }
    __syncwarp();

    // init the linear list.
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      // FIXME(shiwen): dummy entries. now set to FLT_MAX.
      node_distance_list_sdata[list_idx] = FLT_MAX;
    }

    for (auto list_idx = Km; list_idx < Km + Kp * Kd; list_idx++) {
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        // read the random ids.
        // NOTE(shiwen): what if warp of threads read the same address of the
        // shared memory? -> in a single memory access transaction
        auto list_vector_base_id = node_id_list_sdata[list_idx];
        assert(list_vector_base_id < base_num);
        list_vector_sdata[i] = base_data[list_vector_base_id * dim + i];
      }
      __syncwarp();
      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += query_vector_sdata[i] * list_vector_sdata[i];
      }

      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      __syncwarp();

      if (lane_id == 0) {
        node_distance_list_sdata[list_idx] = -sum;
      }
      __syncwarp();
    }

    // merge sort the Km + Kp * Kd part.
    // FIXME(shiwen): Km + Kp * Kd must be 2^n
    warp_sort<data_type, id_type, Km + Kp * Kd, lane_width>(
        node_distance_list_sdata, node_id_list_sdata, true);

    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[explore_idx * Kd];
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = 0;
          if (explore_id == tomb) {
            neighbor_id = tomb;
          } else {
            assert(explore_id < base_num);
            // NOTE(shiwen): coalesced memory access.
            // TODO(shiwen): is it necessary to load into shared memory? but
            // shared memory is very limited.
            neighbor_id = graph[explore_id * max_degree + neighbor_idx];
          }
          node_id_list_sdata[Km + explore_idx * Kd + neighbor_idx] =
              neighbor_id;
        }
      }
      // TODO(shiwen): there are two ways to calculate the distance. 1.one is
      // all the threads of warp calculate one distance between a node and
      // query.2.the other is each thread in warp calculate one distance between
      // its node and query. now choose the method 1. need ncu to profile.
      for (auto list_idx = 0; list_idx < Kp * Kd; list_idx++) {
        auto node_id = node_id_list_sdata[list_idx];
        uint32_t node_distance = FLT_MAX;
        if (node_id != tomb && visit_table->test_and_set(node_id)) {
          for (uint32_t i = lane_id; i < dim; i += lane_width) {
            assert(node_id < base_num);
            list_vector_sdata[i] = base_data[node_id * dim + i];
          }
          __syncwarp();
          data_type sum = 0;
          for (uint32_t i = lane_id; i < dim; i += lane_width) {
            assert(i < dim);
            sum += query_vector_sdata[i] * list_vector_sdata[i];
          }

          for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
          }
          __syncwarp();

          if (lane_id == 0) {
            node_distance_list_sdata[list_idx] = -sum;
          }
          __syncwarp();
        } else {
          if (lane_id == 0) {
            node_distance_list_sdata[list_idx] = FLT_MAX;
          }
        }
      }
      // FIXME(shiwen): check the 3rd template.
      // FIXME(shiwen): Km + Kp * Kd is not the 2^n.
      warp_sort<data_type, id_type, Km + Kp * Kd, lane_width>(
          node_distance_list_sdata, node_id_list_sdata, true);

      iter++;
      if (iter == reset_iter) {
        visit_table->reset_sync(lane_id);
        iter = 0;
      }
    }

    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] = node_id_list_sdata[res_idx];
    }
  }
}
}  // namespace Gpu
}  // namespace Gbuilder
