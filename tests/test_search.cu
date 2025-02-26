#include "Gsearcher.cuh"
#include "spdlog/spdlog.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>
#include <omp.h>

namespace Gbuilder {
namespace Gpu {
TEST(TestSearch, SizeTest) {
  SPDLOG_INFO(
      "the size is {}",
      sizeof(search_warp_state<uint32_t, float, 128, 128, 14, 64, 1 << 12>));
}

template <uint32_t Km = 128, uint32_t Kp = 14, uint32_t Kd = 64,
          typename id_type = uint32_t, typename data_type = float,
          uint32_t tomb = 0XFFFFFFFF>
__global__ void test(id_type* __restrict__ topm_ids,
                     id_type* __restrict__ candidate_ids) {
  auto lane_id = threadIdx.x % 32;
  auto res = collect_top_p_normal<Km, Kp, Kd, id_type, data_type, tomb>(
      topm_ids, candidate_ids, lane_id);
  // printf("the res is %d\n", res);
}

template <uint32_t Km = 128, uint32_t Kp = 14, uint32_t Kd = 64,
          typename id_type = uint32_t, typename data_type = float,
          uint32_t tomb = 0XFFFFFFFF>
bool cpu_ref(id_type* __restrict__ topm_ids,
             id_type* __restrict__ candidate_ids) {
  auto find_num = 0;
  for (auto i = 0; i < Km; i++) {
    if (find_num < Kp) {
      if ((topm_ids[i] & 0X80000000) == 0) {
        candidate_ids[find_num * Kd] = topm_ids[i];
        topm_ids[i] |= 0X80000000;
        find_num++;
      }
    } else {
      break;
    }
  }
  if (find_num == 0) {
    return false;
  }
  while (find_num < Kp) {
    candidate_ids[find_num * Kd] = tomb;
    find_num++;
  }
  return true;
}

void print_vec(std::vector<uint32_t> const& vec) {
  for (auto const& elem : vec) {
    printf("%u,", elem);
  }
  printf("\n");
}

void check_vec(std::vector<uint32_t> const& vec0,
               std::vector<uint32_t> const& vec1) {
  for (auto i = 0; i < vec0.size(); i++) {
    if (vec0[i] != vec1[i]) {
      printf(
          "the error loc is %d, the left value is %d, the right value is %d\n",
          i, vec0[i], vec1[i]);
    }
  }
}

TEST(TestSearch, TestSelectTopPSimple) {
  constexpr uint32_t Km = 128;
  constexpr uint32_t Kp = 14;
  constexpr uint32_t Kd = 64;
  constexpr uint32_t list_length = Km + Kp * Kd;

  auto host_list = std::vector<uint32_t>(list_length);

  for (auto i = 0; i < host_list.size(); i++) {
    if (i % 2 == 0) {
      host_list[i] = i;
    } else {
      host_list[i] = i | 0x80000000;
    }
  }

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(host_list.begin(), host_list.end(),
               std::default_random_engine(seed));

  // copy
  auto check_host_list = host_list;
  uint32_t* d_list = nullptr;

  // print_vec(host_list);

  cudaMalloc(&d_list, sizeof(uint32_t) * list_length);
  cudaMemcpy(d_list, host_list.data(), sizeof(uint32_t) * list_length,
             cudaMemcpyHostToDevice);

  test<Km, Kp, Kd, uint32_t, float, 0XFFFFFFFF><<<1, 32>>>(d_list, d_list + Km);
  cpu_ref<Km, Kp, Kd, uint32_t, float, 0XFFFFFFFF>(check_host_list.data(),
                                                   check_host_list.data() + Km);

  cudaMemcpy(host_list.data(), d_list, sizeof(uint32_t) * list_length,
             cudaMemcpyDeviceToHost);

  // print_vec(host_list);
  // print_vec(check_host_list);

  // check_vec(host_list, check_host_list);

  ASSERT_EQ(host_list, check_host_list);
}
}  // namespace Gpu
}  // namespace Gbuilder
