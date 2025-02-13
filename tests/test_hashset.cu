#include "Ghashset.cuh"
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <sys/types.h>

namespace Gbuilder {
namespace Gpu {

template <uint32_t hash_set_size = 10 * 1024 * 1024>
void __global__ test_insert_kernel(HashSet<>* d_hash_set) {
  auto thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (thread_idx % 4 == 0 && thread_idx < hash_set_size) {
    d_hash_set->check_empty_or_insert(thread_idx,
                                      d_hash_set->d_hash_set_array_);
  }
}

template <uint32_t hash_set_size = 10 * 1024 * 1024>
void __global__ test_insert_kernel_conflict(HashSet<>* d_hash_set,
                                            uint32_t* sum) {
  auto thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (thread_idx < hash_set_size) {
    if (d_hash_set->check_empty_or_insert(thread_idx % 2,
                                          d_hash_set->d_hash_set_array_)) {
      atomicAdd(sum, 1);
    }
  }
}

void __global__ hello_world() { printf("hello world!\n"); }

TEST(HashSetTest, WarmupGpu) {
  hello_world<<<72, 32>>>();
  cudaDeviceSynchronize();
}

TEST(HashSetTest, InsertConflicTest) {
  HashSet<>* d_set = nullptr;
  auto set = HashSet<>{};
  auto hash_set_size = 10 * 1024 * 1024;
  auto h_hash_set_array = (HashSet<>::cell_type_allied*)malloc(
      hash_set_size * sizeof(HashSet<>::cell_type_allied));

  cudaMalloc(&d_set, sizeof(HashSet<>));
  cudaMemcpy(d_set, &set, sizeof(HashSet<>), cudaMemcpyHostToDevice);
  auto block_size = 256;
  auto grid_size = 72;
  uint32_t* d_sum = nullptr;
  uint32_t* h_sum = (uint32_t*)malloc(sizeof(uint32_t));
  cudaMalloc(&d_sum, sizeof(uint32_t));
  cudaMemset(d_sum, 0, sizeof(uint32_t));
  test_insert_kernel_conflict<<<72, 256>>>(d_set, d_sum);
  cudaMemcpy(h_sum, d_sum, sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  cudaMemcpy(h_hash_set_array, set.d_hash_set_array_,
             hash_set_size * sizeof(HashSet<>::cell_type_allied),
             cudaMemcpyDeviceToHost);

  ASSERT_EQ(*h_sum, 2);

  cudaFree(d_set);
  free(h_hash_set_array);
}

TEST(HashSetTest, InsertNoConflicTest) {
  HashSet<>* d_set = nullptr;
  auto set = HashSet<>{};
  auto hash_set_size = 10 * 1024 * 1024;
  auto h_hash_set_array = (HashSet<>::cell_type_allied*)malloc(
      hash_set_size * sizeof(HashSet<>::cell_type_allied));

  cudaMalloc(&d_set, sizeof(HashSet<>));
  cudaMemcpy(d_set, &set, sizeof(HashSet<>), cudaMemcpyHostToDevice);
  auto block_size = 256;
  auto grid_size = 72;
  test_insert_kernel<<<72, 256>>>(d_set);
  cudaDeviceSynchronize();
  cudaMemcpy(h_hash_set_array, set.d_hash_set_array_,
             hash_set_size * sizeof(HashSet<>::cell_type_allied),
             cudaMemcpyDeviceToHost);

  for (auto i = 0; i < std::min(hash_set_size, block_size * grid_size); i++) {
    if (i % 4 == 0) {
      if (h_hash_set_array[i] != 1) {
        SPDLOG_INFO("the i is {}", i);
      }
      ASSERT_EQ(h_hash_set_array[i], 1);
    } else {
      if (h_hash_set_array[i] != 0) {
        SPDLOG_INFO("the i is {}", i);
      }
      ASSERT_EQ(h_hash_set_array[i], 0);
    }
  }

  cudaFree(d_set);
  free(h_hash_set_array);
}
}  // namespace Gpu
}  // namespace Gbuilder