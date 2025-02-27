#pragma once
#include "GBitonicSort.cuh"
#include "Ghashset.cuh"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <gtest/gtest.h>
#include <algorithm>
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
template <uint32_t grid_size, uint32_t block_size, uint32_t shared_memory_size>
__global__ void testKernel() {
  extern __shared__ uint8_t sharedMem[];
  for (auto i = threadIdx.x; i < shared_memory_size; i += block_size) {
    sharedMem[i] = i % 256;
  }
  __syncthreads();
  auto sum = 0;
  for (auto i = 0; i < shared_memory_size; i++) {
    sum += sharedMem[i];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    printf("the sum is %d", sum);
  }
}

TEST(TestMaxSharedMemory, test) {
  constexpr auto max_shared_memory_size = 49152;
  constexpr auto test_shared_memory_size = 53248;
  constexpr auto grid_size = 72;
  constexpr auto block_size = 256;

  cudaFuncSetCacheConfig(
      testKernel<grid_size, block_size, test_shared_memory_size>,
      cudaFuncCachePreferShared);
  cudaCheckError();

  cudaFuncSetAttribute(
      testKernel<grid_size, block_size, test_shared_memory_size>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, cudaFuncCachePreferShared);

  cudaCheckError();

  SPDLOG_INFO("the test shared memory size is {}", test_shared_memory_size);

  // 启动kernel，指定共享内存大小
  testKernel<grid_size, block_size, test_shared_memory_size>
      <<<grid_size, block_size, test_shared_memory_size>>>();

  cudaCheckError();

  cudaDeviceSynchronize();
}
}  // namespace Gpu
}  // namespace Gbuilder
