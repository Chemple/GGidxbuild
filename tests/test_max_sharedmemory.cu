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
  constexpr auto grid_size = 72;
  constexpr auto block_size = 256;

  // cudaFuncSetCacheConfig(
  //     testKernel<grid_size, block_size, max_shared_memory_size>,
  //     cudaFuncCachePreferShared);
  // cudaCheckError();

  int32_t query_size = 0;

  // 查询设备支持的每块最大共享内存
  cudaDeviceGetAttribute(&query_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);

  SPDLOG_INFO("the max shared memory size is {}", query_size);

  cudaCheckError();

  // 启动kernel，指定共享内存大小
  testKernel<grid_size, block_size, max_shared_memory_size>
      <<<grid_size, block_size, max_shared_memory_size>>>();

  cudaDeviceSynchronize();
}
}  // namespace Gpu
}  // namespace Gbuilder
