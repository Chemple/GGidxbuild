#pragma once
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>

namespace Gbuilder {
namespace Gpu {
template <typename key_type = uint32_t, uint32_t table_size = 1 << 11>
struct HashTable {
  // NOTE(shiwen): the key start at 1. Empty is 0.
  key_type list_[table_size];
  static constexpr uint32_t Kempty = 0;

  __device__ __forceinline__ uint32_t hash(key_type const& key) {
    // NOTE(shiwen): the table size must be 2^n.
    return key & (table_size - 1);
  }

  // NOTE(shiwen): poor performance, fix this.
  __device__ __forceinline__ bool warp_level_test_and_set(
      key_type const& key, uint32_t const& lane_id) {
    auto res = false;
    if (lane_id == 0) {
      auto slot = hash(key);
      // auto old_key = atomicCAS(&list_[slot], Kempty, key + 1);
      auto old_key = list_[slot];
      if (old_key == Kempty) {
        list_[slot] = key + 1;
      }
      while (old_key != Kempty && old_key != key + 1) {
        slot = (slot + 1) & (table_size - 1);
        // old_key = atomicCAS(&list_[slot], Kempty, key + 1);
        old_key = list_[slot];
        if (old_key == Kempty) {
          list_[slot] = key + 1;
        }
      }
      if (old_key == Kempty) {
        res = true;
      } else {
        res = false;
      }
    }
    return __shfl_sync(0XFFFFFFFF, res, 0);
  }

  __device__ __forceinline__ void thread_level_set(key_type const& key) {
    auto slot = hash(key);
    auto old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    while (old_key != Kempty && old_key != key + 1) {
      slot = (slot + 1) & (table_size - 1);
      old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    }
  }

  // NOTE(shiwen): poor performance, fix this.
  template <uint32_t Km>
  __device__ __forceinline__ void warp_level_set(
      uint32_t const* node_id_list_sdata, uint32_t const& lane_id) {
    if (lane_id == 0) {
      for (auto i = 0; i < Km; i++) {
        auto key = node_id_list_sdata[i] & 0X7FFFFFFF;
        auto slot = hash(key);
        // auto old_key = atomicCAS(&list_[slot], Kempty, key + 1);
        auto old_key = list_[slot];
        if (old_key == Kempty) {
          list_[slot] = key + 1;
        }
        while (old_key != Kempty && old_key != key + 1) {
          slot = (slot + 1) & (table_size - 1);
          // old_key = atomicCAS(&list_[slot], Kempty, key + 1);
          old_key = list_[slot];
          if (old_key == Kempty) {
            list_[slot] = key + 1;
          }
        }
      }
    }
  }

  __device__ __forceinline__ bool thread_level_test_and_set(
      key_type const& key) {
    auto slot = hash(key);
    auto old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    while (old_key != Kempty && old_key != key + 1) {
      slot = (slot + 1) & (table_size - 1);
      old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    }
    if (old_key == Kempty) {
      return true;
    }
    return false;
  }

  // NOTE(shiwen): for sync reset
  __device__ __forceinline__ bool reset_sync(uint32_t const& lane_id) {
    // FIXME(shiwen): maybe other cuda API..?
    constexpr uint32_t lane_width = 32;
    for (auto i = lane_id; i < table_size; i += lane_width) {
      list_[i] = Kempty;
    }
  }

  // for thread-level query excution
  __device__ __forceinline__ void reset_thread_local() {
    // NOTE(shiwen): the hash table must be allocate in shared memory or local
    // memory
    memset(&list_, 0, table_size * sizeof(key_type));
  }

  // for thread-level query excution
  __device__ __forceinline__ bool test_and_set_thread_local(
      key_type const& key) {
    auto slot = hash(key);
    auto old_key = list_[slot];
    if (old_key == Kempty) {
      list_[slot] = key + 1;
    }
    while (old_key != Kempty && old_key != key + 1) {
      slot = (slot + 1) & (table_size - 1);
      old_key = list_[slot];
      if (old_key == Kempty) {
        list_[slot] = key + 1;
      }
    }
    if (old_key == Kempty) {
      return true;
    } else {
      return false;
    }
  }

  // // NOTE(shiwen): for async reset
  // __device__ __forceinline__ bool reset_async() {}

  // // NOTE(shiwen): barrier
  // __device__ __forceinline__ bool sync() {}
};
}  // namespace Gpu
}  // namespace Gbuilder
