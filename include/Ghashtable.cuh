#pragma once
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdio>
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

  __device__ __forceinline__ bool test_and_set(key_type const& key) {
    auto slot = hash(key);
    auto old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    while (old_key != Kempty && old_key != key + 1) {
      slot = (slot + 1) & table_size;
      old_key = atomicCAS(&list_[slot], Kempty, key + 1);
    }
    if (old_key == Kempty) {
      return true;
    }
    return false;
  }
};
}  // namespace Gpu
}  // namespace Gbuilder
