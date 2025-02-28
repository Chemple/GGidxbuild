#include "Gbuilder.cuh"
#include "Gsearcher.cuh"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <queue>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <sys/types.h>

namespace Gbuilder {
namespace Gpu {

TEST(MatchTest, T2ITest) {
  constexpr uint32_t gt_topk = 129;
  constexpr uint32_t top1_match_graph_degree = 128;
  constexpr uint32_t base_num = 10 * 1000 * 1000;
  // TODO(shiwen): fix this.
  constexpr uint32_t gt_query_num = 10 * 1000 * 1000;
  constexpr uint32_t dim = 0;
  constexpr uint32_t tomb = 0XFFFFFFFF;

  auto gt_file_name =
      "/home/shiwen/project/GGidxbuild/data/gt.train.10M.129.gpu";

  auto h_gt = std::vector<uint32_t>(gt_query_num * gt_topk);
  read_vec_from_file(h_gt, gt_file_name);

  uint32_t* d_gt = nullptr;

  uint32_t* d_top1_match_graph = nullptr;

  auto h_init_graph =
      std::vector<uint32_t>(base_num * top1_match_graph_degree, tomb);

  cudaMalloc(&d_gt, gt_query_num * gt_topk * sizeof(uint32_t));
  cudaMalloc(&d_top1_match_graph,
             base_num * top1_match_graph_degree * sizeof(uint32_t));

  cudaMemcpy(d_gt, h_gt.data(), gt_query_num * gt_topk * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_top1_match_graph, h_init_graph.data(),
             base_num * top1_match_graph_degree * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  // NOTE(shiwen):kernel launch here:
  constexpr uint32_t match_grid_size = 144;
  constexpr uint32_t match_block_size = 512;
  SPDLOG_INFO("begin gpu match");
  match_top1_kernel<match_grid_size, match_block_size, gt_query_num, gt_topk,
                    top1_match_graph_degree, tomb, float, uint32_t>
      <<<match_grid_size, match_block_size>>>(d_gt, d_top1_match_graph);
  SPDLOG_INFO("end gpu match");

  auto h_check = std::vector<uint32_t>(base_num * top1_match_graph_degree);
  cudaMemcpy(h_check.data(), d_top1_match_graph,
             base_num * top1_match_graph_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  dump_vec2_file(h_check,
                 "/home/shiwen/project/GGidxbuild/data/gpu_match_res.ibin");

  auto count = 0;
  for (auto i = 0; i < gt_query_num; i++) {
    if (h_check[i * top1_match_graph_degree] != tomb) {
      count++;
    }
  }
  SPDLOG_INFO("the match count is {}", count);
}
}  // namespace Gpu
}  // namespace Gbuilder
