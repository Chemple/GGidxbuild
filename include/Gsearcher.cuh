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
          uint32_t dim = 128, uint32_t Km = 128, uint32_t Kp = 6,
          uint32_t Kd = 64, uint32_t hash_table_size = 1 << 11>
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
template <uint32_t Km = 128, uint32_t Kp = 6, uint32_t Kd = 64,
          typename id_type = uint32_t, typename data_type = float,
          uint32_t tomb = 0XFFFFFFFF>
__device__ __forceinline__ bool collect_top_p(
    id_type* __restrict__ topm_ids, id_type* __restrict__ candidate_ids,
    uint32_t const& lane_id) {
  // FIXME(shiwen): use cudaFuncSetAttribute(MyKernel,
  // cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
  // NOTE(shiwen): Kp must be smaller than lane_width.
  // static_assert(Km <= 32);
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

template <uint32_t Km = 128, uint32_t Kp = 6, uint32_t Kd = 64,
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
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km = 128, uint32_t Kp = 6,
          uint32_t Kd = 64, uint32_t topk, uint32_t tomb = 0XFFFFFFFF,
          uint32_t hash_table_size = 1 << 11, uint32_t reset_iter = 4,
          typename id_type = uint32_t, typename data_type = float>
__global__ void search_shared_hashtable(data_type* base_data,
                                        data_type* query_data, id_type* graph,
                                        id_type* result) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp = sizeof(
      search_warp_state<id_type, data_type, dim, Km, Kp, Kd, hash_table_size>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  // static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

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
      &warp_states[local_warp_id].visit_;

  // NOTE(shiwen): query num is base num.
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
    // Load base vector
    for (uint32_t i = lane_id; i < dim; i += lane_width) {
      query_vector_sdata[i] = query_data[query_id * dim + i];
    }

    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
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

    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      visit_table->thread_level_set(node_id_list_sdata[list_idx] & 0X7FFFFFFF);
    }

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
        auto node_id = node_id_list_sdata[Km + list_idx];
        data_type node_distance = FLT_MAX;
        // NOTE(shiwen): in method1, there is no thread conflit when accessing
        // hash table.
        if (node_id != tomb &&
            visit_table->warp_level_test_and_set(node_id, lane_id)) {
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
            node_distance_list_sdata[Km + list_idx] = -sum;
          }
          __syncwarp();
        } else {
          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = FLT_MAX;
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
        for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
          visit_table->thread_level_set(node_id_list_sdata[list_idx] &
                                        0X7FFFFFFF);
        }
        iter = 0;
      }
    }

    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

template <typename id_type = uint32_t, typename data_type = float,
          uint32_t dim = 128, uint32_t Km = 128, uint32_t Kp = 6,
          uint32_t Kd = 64>
struct search_warp_state_global_hashtable {
  data_type query_vector_[dim];
  data_type list_vector_[dim];
  id_type node_id_list_[Km + Kp * Kd];
  data_type node_distance_list_[Km + Kp * Kd];
  // FIXME(shiwen): change the size of hash table.
  // TODO(shiwen): if we use one warp to handle one query, is it possible to use
  // warp-level primitive to replace atomicCAS?
};

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km = 128, uint32_t Kp = 6,
          uint32_t Kd = 64, uint32_t topk, uint32_t tomb = 0XFFFFFFFF,
          uint32_t hash_table_size = 1 << 12, uint32_t reset_iter = 8,
          typename id_type = uint32_t, typename data_type = float>
__global__ void search_global_hashtable(
    data_type* base_data, data_type* query_data,
    HashTable<uint32_t, 1 << 12>* hash_tables, id_type* graph,
    id_type* result) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp = sizeof(
      search_warp_state_global_hashtable<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  // static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

  extern __shared__
      search_warp_state_global_hashtable<id_type, data_type, dim, Km, Kp, Kd>
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
      &hash_tables[global_warp_id];

  // NOTE(shiwen): query num is base num.
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
    // Load base vector
    for (uint32_t i = lane_id; i < dim; i += lane_width) {
      query_vector_sdata[i] = query_data[query_id * dim + i];
    }

    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
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

    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      visit_table->thread_level_set(node_id_list_sdata[list_idx] & 0X7FFFFFFF);
    }

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
        auto node_id = node_id_list_sdata[Km + list_idx];
        data_type node_distance = FLT_MAX;
        // NOTE(shiwen): in method1, there is no thread conflit when accessing
        // hash table.
        if (node_id != tomb &&
            visit_table->warp_level_test_and_set(node_id, lane_id)) {
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
            node_distance_list_sdata[Km + list_idx] = -sum;
          }
          __syncwarp();
        } else {
          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = FLT_MAX;
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
        for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
          visit_table->thread_level_set(node_id_list_sdata[list_idx] &
                                        0X7FFFFFFF);
        }
        iter = 0;
      }
    }

    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km, uint32_t Kp, uint32_t Kd,
          uint32_t topk, uint32_t tomb = 0XFFFFFFFF, uint32_t hash_table_size,
          uint32_t reset_iter, typename id_type = uint32_t,
          typename data_type = float>
__global__ void link_process_global_hashtable(
    data_type* base_data, HashTable<uint32_t, hash_table_size>* hash_tables,
    id_type* graph, id_type* result /*, id_type* enter_points*/) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp = sizeof(
      search_warp_state_global_hashtable<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

  extern __shared__
      search_warp_state_global_hashtable<id_type, data_type, dim, Km, Kp, Kd>
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
      &hash_tables[global_warp_id];

  // NOTE(shiwen): query num is base num.
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
    // Load base vector
    for (uint32_t i = lane_id; i < dim; i += lane_width) {
      query_vector_sdata[i] = base_data[query_id * dim + i];
    }

    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
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

    visit_table->reset_sync(lane_id);

    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      visit_table->thread_level_set(node_id_list_sdata[list_idx] & 0X7FFFFFFF);
    }

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
        auto node_id = node_id_list_sdata[Km + list_idx];
        data_type node_distance = FLT_MAX;
        // NOTE(shiwen): in method1, there is no thread conflit when accessing
        // hash table.
        if (node_id != tomb &&
            visit_table->warp_level_test_and_set(node_id, lane_id)) {
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
            node_distance_list_sdata[Km + list_idx] = -sum;
          }
          __syncwarp();
        } else {
          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = FLT_MAX;
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
        for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
          visit_table->thread_level_set(node_id_list_sdata[list_idx] &
                                        0X7FFFFFFF);
        }
        iter = 0;
      }
    }

    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

template <typename id_type = uint32_t, typename data_type = float,
          uint32_t dim = 128, uint32_t Km = 128, uint32_t Kp = 6,
          uint32_t Kd = 64>
struct search_warp_state_v0 {
  id_type node_id_list_[Km + Kp * Kd];
  data_type node_distance_list_[Km + Kp * Kd];
  // FIXME(shiwen): change the size of hash table.
  // TODO(shiwen): if we use one warp to handle one query, is it possible to use
  // warp-level primitive to replace atomicCAS?
};

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km, uint32_t Kp, uint32_t Kd,
          uint32_t topk, uint32_t tomb = 0XFFFFFFFF, uint32_t hash_table_size,
          uint32_t reset_iter, typename id_type = uint32_t,
          typename data_type = float>
__global__ void __launch_bounds__(block_size, 2)
    link_process_v0(data_type* base_data,
                    HashTable<uint32_t, hash_table_size>* hash_tables,
                    id_type* graph,
                    id_type* result /*, id_type* enter_points*/) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

  extern __shared__ search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>
      tss_warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  id_type* node_id_list_sdata = tss_warp_states[local_warp_id].node_id_list_;
  data_type* node_distance_list_sdata =
      tss_warp_states[local_warp_id].node_distance_list_;
  HashTable<uint32_t, hash_table_size>* visit_table =
      &hash_tables[global_warp_id];
#pragma unroll
  // NOTE(shiwen): query num is base num.
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
#pragma unroll
    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
    }
    __syncwarp();
#pragma unroll
    // init the linear list.
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      // FIXME(shiwen): dummy entries. now set to FLT_MAX.
      node_distance_list_sdata[list_idx] = FLT_MAX;
    }
#pragma unroll
    for (auto list_idx = Km; list_idx < Km + Kp * Kd; list_idx++) {
      auto list_vector_base_id = node_id_list_sdata[list_idx];
      data_type sum = 0;
#pragma unroll
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += base_data[query_id * dim + i] *
               base_data[list_vector_base_id * dim + i];
      }
#pragma unroll
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

    visit_table->reset_sync(lane_id);
#pragma unroll
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      visit_table->thread_level_set(node_id_list_sdata[list_idx] & 0X7FFFFFFF);
    }

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
#pragma unroll
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
#pragma unroll
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
#pragma unroll
      // TODO(shiwen): there are two ways to calculate the distance. 1.one is
      // all the threads of warp calculate one distance between a node and
      // query.2.the other is each thread in warp calculate one distance between
      // its node and query. now choose the method 1. need ncu to profile.
      for (auto list_idx = 0; list_idx < Kp * Kd; list_idx++) {
        auto node_id = node_id_list_sdata[Km + list_idx];
        data_type node_distance = FLT_MAX;
        // NOTE(shiwen): in method1, there is no thread conflit when accessing
        // hash table.
        if (node_id != tomb &&
            visit_table->warp_level_test_and_set(node_id, lane_id)) {
          data_type sum = 0;
          for (uint32_t i = lane_id; i < dim; i += lane_width) {
            assert(i < dim);
            sum += base_data[query_id * dim + i] * base_data[node_id * dim + i];
          }

          for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
          }

          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = -sum;
          }
          __syncwarp();
        } else {
          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = FLT_MAX;
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
        for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
          visit_table->thread_level_set(node_id_list_sdata[list_idx] &
                                        0X7FFFFFFF);
        }
        iter = 0;
      }
    }

#pragma unroll
    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

// v0->v1 (change the way of hash table insert.)
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km, uint32_t Kp, uint32_t Kd,
          uint32_t topk, uint32_t tomb = 0XFFFFFFFF, uint32_t hash_table_size,
          uint32_t reset_iter, typename id_type = uint32_t,
          typename data_type = float>
__global__ void __launch_bounds__(block_size)
    link_process_v1(data_type* base_data,
                    HashTable<uint32_t, hash_table_size>* hash_tables,
                    id_type* graph,
                    id_type* result /*, id_type* enter_points*/) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

  extern __shared__ search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>
      s_warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  id_type* node_id_list_sdata = s_warp_states[local_warp_id].node_id_list_;
  data_type* node_distance_list_sdata =
      s_warp_states[local_warp_id].node_distance_list_;
  HashTable<uint32_t, hash_table_size>* visit_table =
      &hash_tables[global_warp_id];

  // NOTE(shiwen): query num is base num.
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
    }
    __syncwarp();

    // init the linear list.
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      // FIXME(shiwen): dummy entries. now set to FLT_MAX.
      node_distance_list_sdata[list_idx] = FLT_MAX;
    }

    for (auto list_idx = Km; list_idx < Km + Kp * Kd; list_idx++) {
      auto list_vector_base_id = node_id_list_sdata[list_idx];
      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += base_data[query_id * dim + i] *
               base_data[list_vector_base_id * dim + i];
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

    visit_table->reset_sync(lane_id);

    visit_table->warp_level_set<Km>(node_id_list_sdata, lane_id);

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
        auto node_id = node_id_list_sdata[Km + list_idx];
        data_type node_distance = FLT_MAX;
        // NOTE(shiwen): in method1, there is no thread conflit when accessing
        // hash table.
        if (node_id != tomb &&
            visit_table->warp_level_test_and_set(node_id, lane_id)) {
          data_type sum = 0;
          for (uint32_t i = lane_id; i < dim; i += lane_width) {
            assert(i < dim);
            sum += base_data[query_id * dim + i] * base_data[node_id * dim + i];
          }

          for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
          }

          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = -sum;
          }
          __syncwarp();
        } else {
          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = FLT_MAX;
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
        visit_table->warp_level_set<Km>(node_id_list_sdata, lane_id);
        iter = 0;
      }
    }

    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

template <typename data_type = float, typename id_type = uint32_t, uint32_t dim,
          uint32_t base_num>
__device__ __forceinline__ data_type
distance_between_v0(id_type const& x_id, id_type const& y_id,
                    data_type const* __restrict__ base_data) {
  data_type sum = 0;
#pragma unroll
  for (auto i = 0; i < dim; i++) {
    auto x_value = base_data[x_id * dim + i];
    auto y_value = base_data[y_id * dim + i];
    sum += x_value * y_value;
  }
  return -sum;
}

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km, uint32_t Kp, uint32_t Kd,
          uint32_t topk, uint32_t tomb = 0XFFFFFFFF, uint32_t hash_table_size,
          uint32_t reset_iter, typename id_type = uint32_t,
          typename data_type = float>
__global__ void __launch_bounds__(block_size)
    link_process_v222(data_type* base_data,
                      HashTable<uint32_t, hash_table_size>* hash_tables,
                      id_type* graph,
                      id_type* result /*, id_type* enter_points*/) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

  extern __shared__ search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>
      ttss_warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  id_type* node_id_list_sdata = ttss_warp_states[local_warp_id].node_id_list_;
  data_type* node_distance_list_sdata =
      ttss_warp_states[local_warp_id].node_distance_list_;
  HashTable<uint32_t, hash_table_size>* visit_table =
      &hash_tables[global_warp_id];
#pragma unroll
  // NOTE(shiwen): query num is base num.
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
#pragma unroll
    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
    }
    __syncwarp();
#pragma unroll
    // init the linear list.
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      // FIXME(shiwen): dummy entries. now set to FLT_MAX.
      node_distance_list_sdata[list_idx] = FLT_MAX;
    }

    for (auto i = Km + lane_id; i < Km + Kp * Kd; i += lane_width) {
      auto neighbor_id = node_id_list_sdata[i];
      node_distance_list_sdata[i] =
          distance_between_v0<data_type, id_type, dim, base_num>(
              neighbor_id, query_id, base_data);
    }
    __syncwarp();

    // merge sort the Km + Kp * Kd part.
    // FIXME(shiwen): Km + Kp * Kd must be 2^n
    warp_sort<data_type, id_type, Km + Kp * Kd, lane_width>(
        node_distance_list_sdata, node_id_list_sdata, true);

    visit_table->reset_sync(lane_id);
#pragma unroll
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      visit_table->thread_level_set(node_id_list_sdata[list_idx] & 0X7FFFFFFF);
    }

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
#pragma unroll
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
#pragma unroll
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
#pragma unroll
      // TODO(shiwen): there are two ways to calculate the distance. 1.one is
      // all the threads of warp calculate one distance between a node and
      // query.2.the other is each thread in warp calculate one distance between
      // its node and query. now choose the method 1. need ncu to profile.
      // for (auto list_idx = 0; list_idx < Kp * Kd; list_idx++) {
      //   auto node_id = node_id_list_sdata[Km + list_idx];
      //   data_type node_distance = FLT_MAX;
      //   // NOTE(shiwen): in method1, there is no thread conflit when
      //   accessing
      //   // hash table.
      //   if (node_id != tomb &&
      //       visit_table->warp_level_test_and_set(node_id, lane_id)) {
      //     data_type sum = 0;
      //     for (uint32_t i = lane_id; i < dim; i += lane_width) {
      //       assert(i < dim);
      //       sum += base_data[query_id * dim + i] * base_data[node_id * dim +
      //       i];
      //     }

      //     for (int offset = 16; offset > 0; offset >>= 1) {
      //       sum += __shfl_down_sync(0xffffffff, sum, offset);
      //     }

      //     if (lane_id == 0) {
      //       node_distance_list_sdata[Km + list_idx] = -sum;
      //     }
      //     __syncwarp();
      //   } else {
      //     if (lane_id == 0) {
      //       node_distance_list_sdata[Km + list_idx] = FLT_MAX;
      //     }
      //   }
      // }

      for (auto i = Km + lane_id; i < Km + Kp * Kd; i += lane_width) {
        auto neighbor_id = node_id_list_sdata[i];
        auto dist = FLT_MAX;
        if (neighbor_id != tomb &&
            visit_table->thread_level_test_and_set(neighbor_id & 0X7FFFFFFF)) {
          dist = distance_between_v0<data_type, id_type, dim, base_num>(
              neighbor_id, query_id, base_data);
        }
        node_distance_list_sdata[i] = dist;
      }
      __syncwarp();
      // FIXME(shiwen): check the 3rd template.
      // FIXME(shiwen): Km + Kp * Kd is not the 2^n.
      warp_sort<data_type, id_type, Km + Kp * Kd, lane_width>(
          node_distance_list_sdata, node_id_list_sdata, true);

      iter++;
      if (iter == reset_iter) {
        visit_table->reset_sync(lane_id);
        for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
          visit_table->thread_level_set(node_id_list_sdata[list_idx] &
                                        0X7FFFFFFF);
        }
        iter = 0;
      }
    }

#pragma unroll
    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

// v0 -> v2 (change the excution pattern, each thread in one warp calcualte one
// distance, performance ↓)
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km, uint32_t Kp, uint32_t Kd,
          uint32_t topk, uint32_t tomb = 0XFFFFFFFF, uint32_t hash_table_size,
          uint32_t reset_iter, typename id_type = uint32_t,
          typename data_type = float>
__global__ void __launch_bounds__(block_size)
    link_process_v2(data_type* base_data,
                    HashTable<uint32_t, hash_table_size>* hash_tables,
                    id_type* graph,
                    id_type* result /*, id_type* enter_points*/) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

  extern __shared__ search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>
      s_warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  id_type* node_id_list_sdata = s_warp_states[local_warp_id].node_id_list_;
  data_type* node_distance_list_sdata =
      s_warp_states[local_warp_id].node_distance_list_;
  HashTable<uint32_t, hash_table_size>* visit_table =
      &hash_tables[global_warp_id];

  // NOTE(shiwen): query num is base num.
#pragma unroll
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
#pragma unroll
    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
    }
    __syncwarp();

// init the linear list.
#pragma unroll
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      // FIXME(shiwen): dummy entries. now set to FLT_MAX.
      node_distance_list_sdata[list_idx] = FLT_MAX;
    }
#pragma unroll
    for (auto i = lane_id + Km; i < Km + Kp * Kd; i += lane_width) {
      auto neighbor_id = node_id_list_sdata[i];
      assert(neighbor_id < base_num);
      data_type distance =
          distance_between_v0<data_type, id_type, dim, base_num>(
              neighbor_id, query_id, base_data);
      node_distance_list_sdata[i] = distance;
    }

    // merge sort the Km + Kp * Kd part.
    // FIXME(shiwen): Km + Kp * Kd must be 2^n
    warp_sort<data_type, id_type, Km + Kp * Kd, lane_width>(
        node_distance_list_sdata, node_id_list_sdata, true);

    visit_table->reset_sync(lane_id);

    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      visit_table->thread_level_set(node_id_list_sdata[list_idx] & 0X7FFFFFFF);
    }

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;
    auto debug = 0;
    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
      debug++;
      if (debug > 1000) {
        printf("dead loop\n");
      }
#pragma unroll
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
#pragma unroll
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
#pragma unroll
      for (auto i = lane_id + Km; i < Km + Kp * Kd; i += lane_width) {
        auto neighbor_id = node_id_list_sdata[i];
        data_type distance = FLT_MAX;
        assert(neighbor_id < base_num);
        if (neighbor_id != tomb &&
            visit_table->thread_level_test_and_set(neighbor_id)) {
          distance = distance_between_v0<data_type, id_type, dim, base_num>(
              neighbor_id, query_id, base_data);
        }
        node_distance_list_sdata[i] = distance;
      }

      // FIXME(shiwen): check the 3rd template.
      // FIXME(shiwen): Km + Kp * Kd is not the 2^n.
      warp_sort<data_type, id_type, Km + Kp * Kd, lane_width>(
          node_distance_list_sdata, node_id_list_sdata, true);

      iter++;
      if (iter == reset_iter) {
        visit_table->reset_sync(lane_id);
#pragma unroll
        for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
          visit_table->thread_level_set(node_id_list_sdata[list_idx] &
                                        0X7FFFFFFF);
        }
        iter = 0;
      }
    }

#pragma unroll
    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

// v0 > v3 (just change select_p. performance ----)
template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km, uint32_t Kp, uint32_t Kd,
          uint32_t topk, uint32_t tomb = 0XFFFFFFFF, uint32_t hash_table_size,
          uint32_t reset_iter, typename id_type = uint32_t,
          typename data_type = float>
__global__ void __launch_bounds__(block_size)
    link_process_v3(data_type* base_data,
                    HashTable<uint32_t, hash_table_size>* hash_tables,
                    id_type* graph,
                    id_type* result /*, id_type* enter_points*/) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

  extern __shared__ search_warp_state_v0<id_type, data_type, dim, Km, Kp, Kd>
      s_warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  id_type* node_id_list_sdata = s_warp_states[local_warp_id].node_id_list_;
  data_type* node_distance_list_sdata =
      s_warp_states[local_warp_id].node_distance_list_;
  HashTable<uint32_t, hash_table_size>* visit_table =
      &hash_tables[global_warp_id];

  // NOTE(shiwen): query num is base num.
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
    }
    __syncwarp();

    // init the linear list.
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      // FIXME(shiwen): dummy entries. now set to FLT_MAX.
      node_distance_list_sdata[list_idx] = FLT_MAX;
    }

    for (auto list_idx = Km; list_idx < Km + Kp * Kd; list_idx++) {
      auto list_vector_base_id = node_id_list_sdata[list_idx];
      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += base_data[query_id * dim + i] *
               base_data[list_vector_base_id * dim + i];
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

    visit_table->reset_sync(lane_id);

    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      visit_table->thread_level_set(node_id_list_sdata[list_idx] & 0X7FFFFFFF);
    }

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
        auto node_id = node_id_list_sdata[Km + list_idx];
        data_type node_distance = FLT_MAX;
        // NOTE(shiwen): in method1, there is no thread conflit when accessing
        // hash table.
        if (node_id != tomb &&
            visit_table->warp_level_test_and_set(node_id, lane_id)) {
          data_type sum = 0;
          for (uint32_t i = lane_id; i < dim; i += lane_width) {
            assert(i < dim);
            sum += base_data[query_id * dim + i] * base_data[node_id * dim + i];
          }

          for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
          }

          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = -sum;
          }
          __syncwarp();
        } else {
          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = FLT_MAX;
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
        for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
          visit_table->thread_level_set(node_id_list_sdata[list_idx] &
                                        0X7FFFFFFF);
        }
        iter = 0;
      }
    }

    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

template <typename id_type = uint32_t, typename data_type = float,
          uint32_t dim = 128, uint32_t Km = 128, uint32_t Kp = 6,
          uint32_t Kd = 64>
struct search_warp_state_store_base_data {
  data_type query_base_data[dim];
  id_type node_id_list_[Km + Kp * Kd];
  data_type node_distance_list_[Km + Kp * Kd];
  // FIXME(shiwen): change the size of hash table.
  // TODO(shiwen): if we use one warp to handle one query, is it possible to use
  // warp-level primitive to replace atomicCAS?
};

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km, uint32_t Kp, uint32_t Kd,
          uint32_t topk, uint32_t tomb = 0XFFFFFFFF, uint32_t hash_table_size,
          uint32_t reset_iter, typename id_type = uint32_t,
          typename data_type = float>
__global__ void __launch_bounds__(block_size) link_process_v0_store_base_data(
    data_type* base_data, HashTable<uint32_t, hash_table_size>* hash_tables,
    id_type* graph, id_type* result /*, id_type* enter_points*/) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp = sizeof(
      search_warp_state_store_base_data<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

  extern __shared__
      search_warp_state_store_base_data<id_type, data_type, dim, Km, Kp, Kd>
          sss_warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  id_type* node_id_list_sdata = sss_warp_states[local_warp_id].node_id_list_;
  data_type* node_distance_list_sdata =
      sss_warp_states[local_warp_id].node_distance_list_;
  data_type* query_base_vector_sdata =
      sss_warp_states[local_warp_id].query_base_data;
  HashTable<uint32_t, hash_table_size>* visit_table =
      &hash_tables[global_warp_id];

  // NOTE(shiwen): query num is base num.
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
    for (auto i = lane_id; i < dim; i += lane_width) {
      query_base_vector_sdata[i] = base_data[query_id * dim + i];
    }

    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
    }
    __syncwarp();

    // init the linear list.
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      // FIXME(shiwen): dummy entries. now set to FLT_MAX.
      node_distance_list_sdata[list_idx] = FLT_MAX;
    }

    for (auto list_idx = Km; list_idx < Km + Kp * Kd; list_idx++) {
      auto list_vector_base_id = node_id_list_sdata[list_idx];
      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += query_base_vector_sdata[i] *
               base_data[list_vector_base_id * dim + i];
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

    visit_table->reset_sync(lane_id);

    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      visit_table->thread_level_set(node_id_list_sdata[list_idx] & 0X7FFFFFFF);
    }

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
        auto node_id = node_id_list_sdata[Km + list_idx];
        data_type node_distance = FLT_MAX;
        // NOTE(shiwen): in method1, there is no thread conflit when accessing
        // hash table.
        if (node_id != tomb &&
            visit_table->warp_level_test_and_set(node_id, lane_id)) {
          data_type sum = 0;
          for (uint32_t i = lane_id; i < dim; i += lane_width) {
            assert(i < dim);
            sum += query_base_vector_sdata[i] * base_data[node_id * dim + i];
          }

          for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
          }

          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = -sum;
          }
          __syncwarp();
        } else {
          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = FLT_MAX;
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
        for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
          visit_table->thread_level_set(node_id_list_sdata[list_idx] &
                                        0X7FFFFFFF);
        }
        iter = 0;
      }
    }

    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

// template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
//           uint32_t query_num, uint32_t dim, uint32_t max_degree,
//           uint32_t shared_memory_size, uint32_t Km, uint32_t Kp, uint32_t Kd,
//           uint32_t topk, uint32_t tomb = 0XFFFFFFFF, uint32_t
//           hash_table_size, uint32_t reset_iter, typename id_type = uint32_t,
//           typename data_type = float>
// __global__ void __launch_bounds__(block_size)
//     // NOTE(shiwen): roargraph based search. only search once and prune
//     reverse
//     // at the same time.
//     link_process_v0_store_base_data_roargraph(
//         data_type* base_data, HashTable<uint32_t, hash_table_size>*
//         hash_tables, id_type* graph, id_type* result /*, id_type*
//         enter_points*/) {
//   constexpr uint32_t lane_width = 32;
//   constexpr uint32_t warp_per_block = block_size / lane_width;
//   constexpr uint32_t shared_memory_size_per_warp = sizeof(
//       search_warp_state_store_base_data<id_type, data_type, dim, Km, Kp,
//       Kd>);
//   constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;

//   // for 1-bit parented node management.
//   static_assert(base_num < 2147483648);
//   // for shared_memory size calculation.
//   static_assert(shared_memory_size ==
//                 shared_memory_size_per_warp * warp_per_block);

//   // Total amount of shared memory per block: 49152 bytes
//   // NOTE(shiwen): use attribute to max the allocate of shared memory
//   static_assert(shared_memory_size < 49152);

//   // Kd must be smaller than max_degree
//   static_assert(Kd <= max_degree);
//   // the topk must be smaller than Km.
//   static_assert(topk <= Km + Kp * Kd);

//   extern __shared__
//       search_warp_state_store_base_data<id_type, data_type, dim, Km, Kp, Kd>
//           sss_warp_states[];

//   uint32_t const global_warp_id =
//       (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
//   uint32_t const local_warp_id = threadIdx.x / lane_width;
//   uint32_t const lane_id = threadIdx.x % lane_width;

//   id_type* node_id_list_sdata = sss_warp_states[local_warp_id].node_id_list_;
//   data_type* node_distance_list_sdata =
//       sss_warp_states[local_warp_id].node_distance_list_;
//   data_type* query_base_vector_sdata =
//       sss_warp_states[local_warp_id].query_base_data;
//   HashTable<uint32_t, hash_table_size>* visit_table =
//       &hash_tables[global_warp_id];

//   // NOTE(shiwen): query num is base num.
//   for (uint32_t query_id = global_warp_id; query_id < query_num;
//        query_id += global_warp_num) {
//     for (auto i = lane_id; i < dim; i += lane_width) {
//       query_base_vector_sdata[i] = base_data[query_id * dim + i];
//     }

//     for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
//       // FIXME(shiwen): generate the random enter points.
//       // gen the enter points.
//       auto random_id = (3030517 + i) % base_num;
//       node_id_list_sdata[Km + i] = random_id;
//     }
//     __syncwarp();

//     // init the linear list.
//     for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
//       // FIXME(shiwen): dummy entries. now set to FLT_MAX.
//       node_distance_list_sdata[list_idx] = FLT_MAX;
//     }

//     for (auto list_idx = Km; list_idx < Km + Kp * Kd; list_idx++) {
//       auto list_vector_base_id = node_id_list_sdata[list_idx];
//       data_type sum = 0;
//       for (uint32_t i = lane_id; i < dim; i += lane_width) {
//         assert(i < dim);
//         sum += query_base_vector_sdata[i] *
//                base_data[list_vector_base_id * dim + i];
//       }

//       for (int offset = 16; offset > 0; offset >>= 1) {
//         sum += __shfl_down_sync(0xffffffff, sum, offset);
//       }
//       __syncwarp();

//       if (lane_id == 0) {
//         node_distance_list_sdata[list_idx] = -sum;
//       }
//       __syncwarp();
//     }

//     // merge sort the Km + Kp * Kd part.
//     // FIXME(shiwen): Km + Kp * Kd must be 2^n
//     warp_sort<data_type, id_type, Km + Kp * Kd, lane_width>(
//         node_distance_list_sdata, node_id_list_sdata, true);

//     visit_table->reset_sync(lane_id);

//     for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
//       visit_table->thread_level_set(node_id_list_sdata[list_idx] &
//       0X7FFFFFFF);
//     }

//     // NOTE(shiwen): for hashtable reset.
//     auto iter = 0;

//     // FIXME(shiwen): change the condition?
//     while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
//         node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
//       // fill the explore id list.
//       for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
//         auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
//         for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
//              neighbor_idx += lane_width) {
//           uint32_t neighbor_id = tomb;
//           if (explore_id != tomb) {
//             assert(explore_id < base_num);
//             // NOTE(shiwen): coalesced memory access.
//             // TODO(shiwen): is it necessary to load into shared memory? but
//             // shared memory is very limited.
//             neighbor_id = graph[explore_id * max_degree + neighbor_idx];
//           }
//           node_id_list_sdata[Km + explore_idx * Kd + neighbor_idx] =
//               neighbor_id;
//         }
//       }
//       // TODO(shiwen): there are two ways to calculate the distance. 1.one is
//       // all the threads of warp calculate one distance between a node and
//       // query.2.the other is each thread in warp calculate one distance
//       between
//       // its node and query. now choose the method 1. need ncu to profile.
//       for (auto list_idx = 0; list_idx < Kp * Kd; list_idx++) {
//         auto node_id = node_id_list_sdata[Km + list_idx];
//         data_type node_distance = FLT_MAX;
//         // NOTE(shiwen): in method1, there is no thread conflit when
//         accessing
//         // hash table.
//         if (node_id != tomb &&
//             visit_table->warp_level_test_and_set(node_id, lane_id)) {
//           data_type sum = 0;
//           for (uint32_t i = lane_id; i < dim; i += lane_width) {
//             assert(i < dim);
//             sum += query_base_vector_sdata[i] * base_data[node_id * dim + i];
//           }

//           for (int offset = 16; offset > 0; offset >>= 1) {
//             sum += __shfl_down_sync(0xffffffff, sum, offset);
//           }

//           if (lane_id == 0) {
//             node_distance_list_sdata[Km + list_idx] = -sum;
//           }
//           __syncwarp();
//         } else {
//           if (lane_id == 0) {
//             node_distance_list_sdata[Km + list_idx] = FLT_MAX;
//           }
//         }
//       }
//       // FIXME(shiwen): check the 3rd template.
//       // FIXME(shiwen): Km + Kp * Kd is not the 2^n.
//       warp_sort<data_type, id_type, Km + Kp * Kd, lane_width>(
//           node_distance_list_sdata, node_id_list_sdata, true);

//       iter++;
//       if (iter == reset_iter) {
//         visit_table->reset_sync(lane_id);
//         for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width)
//         {
//           visit_table->thread_level_set(node_id_list_sdata[list_idx] &
//                                         0X7FFFFFFF);
//         }
//         iter = 0;
//       }
//     }

//     // write to result
//     for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
//       assert(query_id < base_num);
//       result[query_id * topk + res_idx] =
//           (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
//     }
//   }
// }

template <typename id_type = uint32_t, typename data_type = float,
          uint32_t dim = 128, uint32_t Km = 128, uint32_t Kp = 6,
          uint32_t Kd = 64>
struct search_warp_state_xxx {
  data_type query_base_data[dim];
  id_type node_id_list_[Km + Kp * Kd];
  data_type node_distance_list_[Km + Kp * Kd];
  HashTable<uint32_t, 4 * (Km + Kp * Kd)> hash_table;
  // FIXME(shiwen): change the size of hash table.
  // TODO(shiwen): if we use one warp to handle one query, is it possible to use
  // warp-level primitive to replace atomicCAS?
};

template <uint32_t grid_size, uint32_t block_size, uint32_t base_num,
          uint32_t query_num, uint32_t dim, uint32_t max_degree,
          uint32_t shared_memory_size, uint32_t Km, uint32_t Kp, uint32_t Kd,
          uint32_t topk, uint32_t tomb = 0XFFFFFFFF, uint32_t hash_table_size,
          uint32_t reset_iter, typename id_type = uint32_t,
          typename data_type = float>
__global__ void __launch_bounds__(block_size)
    link_process_v0_xxx(data_type* base_data, id_type* graph,
                        id_type* result /*, id_type* enter_points*/) {
  constexpr uint32_t lane_width = 32;
  constexpr uint32_t warp_per_block = block_size / lane_width;
  constexpr uint32_t shared_memory_size_per_warp =
      sizeof(search_warp_state_xxx<id_type, data_type, dim, Km, Kp, Kd>);
  constexpr uint32_t global_warp_num = (block_size * grid_size) / lane_width;
  static_assert(hash_table_size == 256);

  // for 1-bit parented node management.
  static_assert(base_num < 2147483648);
  // for shared_memory size calculation.
  static_assert(shared_memory_size ==
                shared_memory_size_per_warp * warp_per_block);

  // Total amount of shared memory per block: 49152 bytes
  // NOTE(shiwen): use attribute to max the allocate of shared memory
  static_assert(shared_memory_size < 49152);

  // Kd must be smaller than max_degree
  static_assert(Kd <= max_degree);
  // the topk must be smaller than Km.
  static_assert(topk <= Km + Kp * Kd);

  extern __shared__ search_warp_state_xxx<id_type, data_type, dim, Km, Kp, Kd>
      xx_warp_states[];

  uint32_t const global_warp_id =
      (blockIdx.x * blockDim.x + threadIdx.x) / lane_width;
  uint32_t const local_warp_id = threadIdx.x / lane_width;
  uint32_t const lane_id = threadIdx.x % lane_width;

  id_type* node_id_list_sdata = xx_warp_states[local_warp_id].node_id_list_;
  data_type* node_distance_list_sdata =
      xx_warp_states[local_warp_id].node_distance_list_;
  data_type* query_base_vector_sdata =
      xx_warp_states[local_warp_id].query_base_data;
  HashTable<uint32_t, hash_table_size>* visit_table =
      &xx_warp_states[local_warp_id].hash_table;

  // NOTE(shiwen): query num is base num.
  for (uint32_t query_id = global_warp_id; query_id < query_num;
       query_id += global_warp_num) {
    for (auto i = lane_id; i < dim; i += lane_width) {
      query_base_vector_sdata[i] = base_data[query_id * dim + i];
    }

    for (uint32_t i = lane_id; i < Kp * Kd; i += lane_width) {
      // FIXME(shiwen): generate the random enter points.
      // gen the enter points.
      auto random_id = (3030517 + i) % base_num;
      node_id_list_sdata[Km + i] = random_id;
    }
    __syncwarp();

    // init the linear list.
    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      // FIXME(shiwen): dummy entries. now set to FLT_MAX.
      node_distance_list_sdata[list_idx] = FLT_MAX;
    }

    for (auto list_idx = Km; list_idx < Km + Kp * Kd; list_idx++) {
      auto list_vector_base_id = node_id_list_sdata[list_idx];
      data_type sum = 0;
      for (uint32_t i = lane_id; i < dim; i += lane_width) {
        assert(i < dim);
        sum += query_base_vector_sdata[i] *
               base_data[list_vector_base_id * dim + i];
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

    visit_table->reset_sync(lane_id);

    for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
      visit_table->thread_level_set(node_id_list_sdata[list_idx] & 0X7FFFFFFF);
    }

    // NOTE(shiwen): for hashtable reset.
    auto iter = 0;

    // FIXME(shiwen): change the condition?
    while (collect_top_p_normal<Km, Kp, Kd, id_type, data_type>(
        node_id_list_sdata, node_id_list_sdata + Km, lane_id)) {
      // fill the explore id list.
      for (auto explore_idx = 0; explore_idx < Kp; explore_idx++) {
        auto explore_id = node_id_list_sdata[Km + explore_idx * Kd];
        for (auto neighbor_idx = lane_id; neighbor_idx < Kd;
             neighbor_idx += lane_width) {
          uint32_t neighbor_id = tomb;
          if (explore_id != tomb) {
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
        auto node_id = node_id_list_sdata[Km + list_idx];
        data_type node_distance = FLT_MAX;
        // NOTE(shiwen): in method1, there is no thread conflit when accessing
        // hash table.
        if (node_id != tomb &&
            visit_table->warp_level_test_and_set(node_id, lane_id)) {
          data_type sum = 0;
          for (uint32_t i = lane_id; i < dim; i += lane_width) {
            assert(i < dim);
            sum += query_base_vector_sdata[i] * base_data[node_id * dim + i];
          }

          for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
          }

          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = -sum;
          }
          __syncwarp();
        } else {
          if (lane_id == 0) {
            node_distance_list_sdata[Km + list_idx] = FLT_MAX;
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
        for (auto list_idx = lane_id; list_idx < Km; list_idx += lane_width) {
          visit_table->thread_level_set(node_id_list_sdata[list_idx] &
                                        0X7FFFFFFF);
        }
        iter = 0;
      }
    }

    // write to result
    for (auto res_idx = lane_id; res_idx < topk; res_idx += lane_width) {
      assert(query_id < base_num);
      result[query_id * topk + res_idx] =
          (node_id_list_sdata[res_idx] & 0X7FFFFFFF);
    }
  }
}

template <typename data_type, typename id_type, uint32_t capacity>
struct MaxHeap {
  struct Element {
    data_type dist;
    id_type id;

    __device__ __forceinline__ bool operator<(Element const& other) const {
      return dist < other.dist;
    }
  };

  Element elements[capacity];
  uint32_t size;

  __device__ __forceinline__ MaxHeap() : size(0) {}

  __device__ __forceinline__ void insert(data_type dist, id_type id) {
    if (size >= capacity) {
      if (dist >= elements[0].dist) {
        return;
      }
      elements[0] = {dist, id};
      heapifyDown();
    } else {
      elements[size++] = {dist, id};
      heapifyUp();
    }
  }

 private:
  __device__ __forceinline__ void heapifyDown() {
    uint32_t idx = 0;
    while (true) {
      uint32_t largest = idx;
      uint32_t const left = 2 * idx + 1;
      uint32_t const right = 2 * idx + 2;

      if (left < size && elements[left] < elements[largest]) {
        largest = left;
      }
      if (right < size && elements[right] < elements[largest]) {
        largest = right;
      }

      if (largest == idx) {
        break;
      }

      swap(idx, largest);
      idx = largest;
    }
  }

  __device__ __forceinline__ void heapifyUp() {
    uint32_t idx = size - 1;
    while (idx > 0) {
      uint32_t const parent = (idx - 1) / 2;
      if (elements[parent] < elements[idx]) {
        swap(parent, idx);
        idx = parent;
      } else {
        break;
      }
    }
  }

  __device__ __forceinline__ void swap(uint32_t const& i, uint32_t const& j) {
    Element const temp = elements[i];
    elements[i] = elements[j];
    elements[j] = temp;
  }
};

template <typename id_type, uint32_t hash_table_size,
          uint32_t re_probe_max_num = 8>
struct ThreadLocalHashTable {
  id_type table[hash_table_size];

  __device__ __forceinline__ void init() {
#pragma unroll
    for (uint32_t i = 0; i < hash_table_size; ++i) {
      table[i] = 0;
    }
  }

  __device__ __forceinline__ bool test_and_set(id_type id) {
    constexpr id_type Empty = 0;
    uint32_t slot = id & (hash_table_size - 1);

#pragma unroll
    for (uint32_t i = 0; i < re_probe_max_num; ++i) {
      if (table[slot] == Empty) {
        table[slot] = id + 1;
        return true;
      }
      if (table[slot] == id + 1) {
        return false;
      }
      slot = (slot + 1) & (hash_table_size - 1);
    }
    return false;
  }

  __device__ __forceinline__ bool contains(id_type id) const {
    uint32_t slot = id & (hash_table_size - 1);

#pragma unroll
    for (uint32_t i = 0; i < re_probe_max_num; ++i) {
      if (table[slot] == id + 1) return true;
      if (table[slot] == 0) return false;
      slot = (slot + 1) & (hash_table_size - 1);
    }
    return false;
  }

  __device__ __forceinline__ bool insert(id_type id) {
    return test_and_set(id);
  }
};

// template <uint32_t grid_size, uint32_t block_size, typename id_type,
//           typename data_type, uint32_t dim, uint32_t base_num,
//           uint32_t query_num, uint32_t Km, uint32_t Kp, uint32_t Kd,
//           uint32_t max_degree, uint32_t topk, uint32_t tomb,
//           uint32_t hash_table_size = 2048>
// __global__ __launch_bounds__(block_size) void per_thread_search_kernel(
//     data_type* base_data, id_type* graph, id_type* result) {
//   constexpr uint32_t stride = grid_size * block_size;

//   ThreadLocalHashTable<id_type, hash_table_size, 8> hash_table;
//   MaxHeap<data_type, id_type, Km + Kp * Kd> heap;
//   data_type query_base_data[dim];

//   // hash_table.init();

//   uint32_t const thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   for (auto query_id = thread_idx; query_id < query_num; query_id += stride)
//   {
//     hash_table.init();
// #pragma unroll
//     for (auto i = 0; i < dim; i++) {
//       query_base_data[i] = graph[query_id * dim + i];
//     }

//     constexpr uint32_t prime = 2654435761;
// #pragma unroll
//     for (uint32_t i = 0; i < Kp; ++i) {
//       id_type enter_id = (query_id * prime + i) % base_num;
//       if (hash_table.test_and_set(enter_id)) {
//         data_type sum = 0;
// #pragma unroll
//         for (uint32_t j = 0; j < dim; j += 4) {
//           sum += query[j] * base_data[enter_id * dim + j];
//           sum += query[j + 1] * base_data[enter_id * dim + j + 1];
//           sum += query[j + 2] * base_data[enter_id * dim + j + 2];
//           sum += query[j + 3] * base_data[enter_id * dim + j + 3];
//         }
//         heap.insert(-sum, enter_id);
//       }
//     }

//     constexpr uint32_t max_iterations = Km * 2;
//     data_type prev_max = -FLT_MAX;
//     uint32_t stagnation = 0;

//     for (uint32_t iter = 0; iter < max_iterations && heap.size > 0; ++iter) {
//       auto [current_dist, current_id] = heap.elements[0];
//       heap.elements[0] = heap.elements[--heap.size];
//       heap.heapifyDown();

//       if (current_dist == prev_max) {
//         if (++stagnation >= 3) break;
//       } else {
//         prev_max = current_dist;
//         stagnation = 0;
//       }

// #pragma unroll
//       for (uint32_t i = 0; i < max_degree; ++i) {
//         id_type neighbor = graph[current_id * max_degree + i];
//         if (neighbor == tomb || neighbor == 0) continue;

//         if (hash_table.test_and_set(neighbor)) {
//           data_type sum = 0;
// #pragma unroll
//           for (uint32_t j = 0; j < dim; j += 4) {
//             sum += query[j] * base_data[neighbor * dim + j];
//             sum += query[j + 1] * base_data[neighbor * dim + j + 1];
//             sum += query[j + 2] * base_data[neighbor * dim + j + 2];
//             sum += query[j + 3] * base_data[neighbor * dim + j + 3];
//           }
//           heap.insert(-sum, neighbor);
//         }
//       }
//     }

//     uint32_t sorted_count = heap.size;
//     for (int i = sorted_count / 2 - 1; i >= 0; --i)
//       heap.heapifyDown(i, sorted_count);

//     for (int i = sorted_count - 1; i > 0; --i) {
//       auto temp = heap.elements[0];
//       heap.elements[0] = heap.elements[i];
//       heap.elements[i] = temp;
//       heap.heapifyDown(0, i);
//     }

// #pragma unroll
//     for (uint32_t i = 0; i < topk; ++i) {
//       result[tid * topk + i] =
//           (i < sorted_count) ? (heap.elements[i].id & 0x7FFFFFFF) : tomb;
//     }
//   }
// }
}  // namespace Gpu
}  // namespace Gbuilder
