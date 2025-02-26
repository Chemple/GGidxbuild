#include "GBitonicSort.cuh"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <gtest/gtest.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <sys/types.h>
#include <unistd.h>

namespace Gbuilder {
namespace Gpu {

template <uint32_t N = 128>
__global__ void test(uint32_t* d_k, uint32_t* d_v) {
  warp_sort<uint32_t, uint32_t, N, 32>(d_k, d_v);
}

void print_vec(std::vector<uint32_t> const& vec) {
  for (auto const& elem : vec) {
    printf("%u ", elem);
  }
  printf("\n");
}

TEST(Bitonictest, SimpleTest) {
  constexpr uint32_t N = 256;
  auto seed = std::chrono::system_clock::now().time_since_epoch().count();

  uint32_t* d_k = nullptr;
  uint32_t* d_v = nullptr;

  auto host_k_vec = std::vector<uint32_t>(N);
  for (auto i = 0; i < host_k_vec.size(); i++) {
    host_k_vec[i] = i;
  }
  auto host_v_vec = std::vector<uint32_t>(N);
  for (auto i = 0; i < host_k_vec.size(); i++) {
    host_v_vec[i] = i;
  }

  auto check_k_vec = std::vector<uint32_t>(N);
  auto check_v_vec = std::vector<uint32_t>(N);

  std::shuffle(host_k_vec.begin(), host_k_vec.end(),
               std::default_random_engine(seed));
  std::shuffle(host_v_vec.begin(), host_v_vec.end(),
               std::default_random_engine(seed));

  print_vec(host_k_vec);
  print_vec(host_v_vec);

  cudaMalloc(&d_k, N * sizeof(uint32_t));
  cudaMalloc(&d_v, N * sizeof(uint32_t));

  cudaMemcpy(d_k, host_k_vec.data(), N * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, host_v_vec.data(), N * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  test<N><<<1, 32>>>(d_k, d_v);

  cudaMemcpy(check_k_vec.data(), d_k, N * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(check_v_vec.data(), d_v, N * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  print_vec(check_k_vec);
  print_vec(check_v_vec);
}
}  // namespace Gpu
}  // namespace Gbuilder
