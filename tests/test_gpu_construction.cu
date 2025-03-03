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
// unimplement.
void cpu_async_topnn_projection(void*) {}

void parallel_compute_and_sort_ip_distance_cpu(
    float const* base_data, uint32_t* graph, float* neighbor_distance,
    uint32_t base_num, uint32_t dim, uint32_t max_in_degree, uint32_t tomb,
    uint32_t num_threads = std::thread::hardware_concurrency()) {
  // 计算每个线程的工作量
  uint32_t const chunk_size = (base_num + num_threads - 1) / num_threads;
  std::vector<std::thread> workers;

  auto worker_task = [&](uint32_t start, uint32_t end) {
    end = std::min(end, base_num);
    for (uint32_t base_idx = start; base_idx < end; ++base_idx) {
      float const* base_vector = &base_data[base_idx * dim];
      uint32_t* current_graph = &graph[base_idx * max_in_degree];
      float* current_dist = &neighbor_distance[base_idx * max_in_degree];

      // 收集有效邻居和距离
      std::vector<std::pair<float, uint32_t>> valid_neighbors;
      valid_neighbors.reserve(max_in_degree);

      for (uint32_t i = 0; i < max_in_degree; ++i) {
        uint32_t neighbor_id = current_graph[i];
        if (neighbor_id == tomb) {
          valid_neighbors.emplace_back(FLT_MAX, neighbor_id);
          continue;
        }

        float const* neighbor_vector = &base_data[neighbor_id * dim];
        float sum = 0.0f;

        // 手动展开循环以提高计算效率
        uint32_t j = 0;
        for (; j + 3 < dim; j += 4) {
          sum += base_vector[j] * neighbor_vector[j] +
                 base_vector[j + 1] * neighbor_vector[j + 1] +
                 base_vector[j + 2] * neighbor_vector[j + 2] +
                 base_vector[j + 3] * neighbor_vector[j + 3];
        }
        for (; j < dim; ++j) {
          sum += base_vector[j] * neighbor_vector[j];
        }

        valid_neighbors.emplace_back(-sum, neighbor_id);
      }

      // 并行排序（使用pdqsort替代std::sort）
      std::sort(valid_neighbors.begin(), valid_neighbors.end(),
                [](auto const& a, auto const& b) { return a.first < b.first; });

      // 写回结果
      uint32_t valid_count = valid_neighbors.size();
      for (uint32_t i = 0; i < max_in_degree; ++i) {
        if (i < valid_count) {
          current_dist[i] = valid_neighbors[i].first;
          current_graph[i] = valid_neighbors[i].second;
        } else {
          current_dist[i] = 0.0f;
          current_graph[i] = tomb;
        }
      }
    }
  };

  // 创建并启动工作线程
  for (uint32_t t = 0; t < num_threads; ++t) {
    uint32_t start = t * chunk_size;
    uint32_t end = start + chunk_size;
    workers.emplace_back(worker_task, start, end);
  }

  // 等待所有线程完成
  for (auto& t : workers) {
    t.join();
  }
}

TEST(GpuConstructionTime, T2ITest) {
  constexpr uint32_t gt_topk = 129;
  constexpr uint32_t top1_match_graph_degree = 128;
  constexpr uint32_t base_num = 10 * 1000 * 1000;
  // TODO(shiwen): fix this.
  constexpr uint32_t gt_query_num = 10 * 1000 * 1000;
  constexpr uint32_t dim = 200;
  constexpr uint32_t tomb = 0XFFFFFFFF;

  constexpr uint32_t top1_pruneed_graph_degree = 15;
  constexpr uint32_t prune_final_graph_degree = 55;
  constexpr uint32_t reverse_graph_degree = 128;

  auto gt_file_name =
      "/home/shiwen/project/GGidxbuild/data/gt.train.10M.129.gpu";
  auto basedata_file_name =
      "/home/shiwen/project/GGidxbuild/data/10M_200/vector.fbin";

  auto h_gt = std::vector<uint32_t>(gt_query_num * gt_topk);
  auto h_base_data = std::vector<float>(base_num * dim);

  read_vec_from_file(h_gt, gt_file_name);
  read_vec_from_file(h_base_data, basedata_file_name);

  // TODO(shiwen): fix this.
  cpu_async_topnn_projection(nullptr);

  uint32_t* d_gt = nullptr;
  float* d_base_data = nullptr;

  auto init_graph =
      std::vector<uint32_t>(base_num * top1_match_graph_degree, tomb);
  uint32_t* d_top1_match_graph = nullptr;

  uint32_t* d_top1_pruned_graph = nullptr;
  uint32_t* d_reverse_graph = nullptr;
  uint32_t* d_prune_final_graph = nullptr;

  // TODO(shiwen): allocate all the GPU resource at the beginning of the time.
  // TODO(shiwen): use cuda graph to hide the kernel launch time.
  // TODO(shiwen): how to async the memcpy after one kernel launch?
  cudaMalloc(&d_gt, gt_query_num * gt_topk * sizeof(uint32_t));
  cudaCheckError();
  cudaMalloc(&d_top1_match_graph,
             base_num * top1_match_graph_degree * sizeof(uint32_t));
  cudaCheckError();

  SPDLOG_INFO("allocate {} GB global memory",
              1.0 *
                  (gt_query_num * gt_topk + base_num * dim +
                   base_num * top1_match_graph_degree +
                   base_num * top1_match_graph_degree +
                   base_num * top1_pruneed_graph_degree +
                   base_num * reverse_graph_degree +
                   base_num * prune_final_graph_degree) *
                  4 / 1024 / 1024 / 1024);

  cudaMemcpy(d_gt, h_gt.data(), gt_query_num * gt_topk * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaCheckError();

  cudaMemcpy(d_top1_match_graph, init_graph.data(),
             base_num * top1_match_graph_degree * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaCheckError();

  // cudaStream_t main_stream;
  // cudaStreamCreate(&main_stream);
  // cudaEvent_t start_construction;
  // cudaEvent_t end_construction;
  // cudaEventCreate(&start_construction);
  // cudaEventCreate(&end_construction);
  // cudaGraph_t graph;
  // cudaGraphExec_t instance;
  // cudaStreamBeginCapture(main_stream, cudaStreamCaptureModeGlobal);

  // cudaEventRecordWithFlags(start_construction, main_stream,
  //                          cudaEventRecordExternal);

  // NOTE(shiwen):kernel launch here:
  constexpr uint32_t match_grid_size = 144;
  constexpr uint32_t match_block_size = 512;
  match_top1_kernel<match_grid_size, match_block_size, gt_query_num, gt_topk,
                    top1_match_graph_degree, tomb, float, uint32_t>
      <<<match_grid_size, match_block_size, 0 /*, main_stream*/>>>(
          d_gt, d_top1_match_graph);

  cudaFree(d_gt);
  cudaCheckError();
  cudaDeviceSynchronize();

  auto h_top1_match_g =
      std::vector<uint32_t>(base_num * top1_match_graph_degree);
  cudaMemcpy(h_top1_match_g.data(), d_top1_match_graph,
             base_num * top1_match_graph_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaCheckError();

  dump_vec2_file(h_top1_match_g,
                 "/home/shiwen/project/GGidxbuild/data/top1_match_graph.ibin");

  // auto host_top1_match =
  //     std::vector<uint32_t>(base_num * top1_match_graph_degree);
  // cudaMemcpy(host_top1_match.data(), d_top1_match_graph,
  //            base_num * top1_match_graph_degree * sizeof(uint32_t),
  //            cudaMemcpyDeviceToHost);

  // auto h_neighbor_distance =
  //     std::vector<float>(base_num * top1_match_graph_degree);

  // parallel_compute_and_sort_ip_distance_cpu(
  //     h_base_data.data(), host_top1_match.data(), h_neighbor_distance.data(),
  //     base_num, dim, top1_match_graph_degree, 0xFFFFFFFF);

  constexpr uint32_t sort_neighbor_grid_size = 144;
  constexpr uint32_t sort_neighbor_block_size = 256;
  constexpr uint32_t sort_neighbor_share_memory_size =
      sizeof(compute_sort_warp_state<uint32_t, float, top1_match_graph_degree,
                                     dim>) *
      sort_neighbor_block_size / 32;

  cudaMalloc(&d_base_data, base_num * dim * sizeof(float));
  cudaCheckError();

  SPDLOG_INFO("allocate {} memory for d_neighbor_distance",
              base_num * top1_match_graph_degree * sizeof(float));
  cudaMemcpy(d_base_data, h_base_data.data(), base_num * dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaCheckError();
  cudaDeviceSynchronize();

  // TODO(shiwen): change this kernel to gain more performance❗❗❗❗❗❗❗
  sort_neighbor_kernel<
      sort_neighbor_grid_size, sort_neighbor_block_size, base_num,
      top1_match_graph_degree, tomb, dim, sort_neighbor_share_memory_size,
      float, uint32_t><<<sort_neighbor_grid_size, sort_neighbor_block_size,
                         sort_neighbor_share_memory_size /*,
                         main_stream*/>>>(
      d_base_data, d_top1_match_graph);

  auto h_top1_match_sorted_g =
      std::vector<uint32_t>(base_num * top1_match_graph_degree);
  cudaMemcpy(h_top1_match_sorted_g.data(), d_top1_match_graph,
             base_num * top1_match_graph_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  dump_vec2_file(h_top1_match_sorted_g,
                 "/home/shiwen/project/GGidxbuild/data/top1_match_sorted.ibin");

  // SPDLOG_INFO("finish gpu compute");

  // auto check_graph = std::vector<uint32_t>(base_num *
  // top1_match_graph_degree); auto check_neighbor_distance =
  //     std::vector<float>(base_num * top1_match_graph_degree);

  // cudaMemcpy(check_graph.data(), d_top1_match_graph,
  //            sizeof(uint32_t) * base_num * top1_match_graph_degree,
  //            cudaMemcpyDeviceToHost);
  // cudaMemcpy(check_neighbor_distance.data(), d_neighbor_distance,
  //            sizeof(float) * base_num * top1_match_graph_degree,
  //            cudaMemcpyDeviceToHost);

  // dump_vec2_file(
  //     check_graph,
  //     "/home/shiwen/project/GGidxbuild/data/after_match_sort_distance.fbin");
  // dump_vec2_file(
  //     check_graph,
  //     "/home/shiwen/project/GGidxbuild/data/after_match_sort_id.ibin");

  // for (auto i = 0; i < base_num; i++) {
  //   auto gpu_neighbor_set = std::unordered_set<uint32_t>{};
  //   auto cpu_neighbor_set = std::unordered_set<uint32_t>{};
  //   for (auto j = 0; j < top1_match_graph_degree; j++) {
  //     ASSERT_NEAR(check_neighbor_distance[i * top1_match_graph_degree + j],
  //                 h_neighbor_distance[i * top1_match_graph_degree + j],
  //                 1e-3);
  //     if (check_graph[i * top1_match_graph_degree + j] !=
  //         host_top1_match[i * top1_match_graph_degree + j]) {
  //       gpu_neighbor_set.insert(check_graph[i * top1_match_graph_degree +
  //       j]); cpu_neighbor_set.insert(
  //           host_top1_match[i * top1_match_graph_degree + j]);
  //     }
  //   }
  //   for (auto const& gpu_id : gpu_neighbor_set) {
  //     ASSERT_TRUE(cpu_neighbor_set.find(gpu_id) != cpu_neighbor_set.end());
  //   }
  // }

  cudaMalloc(&d_top1_pruned_graph,
             base_num * top1_pruneed_graph_degree * sizeof(uint32_t));
  cudaCheckError();
  cudaMalloc(&d_reverse_graph,
             base_num * reverse_graph_degree * sizeof(uint32_t));
  cudaCheckError();

  constexpr uint32_t rng_reverse_grid_size = 144;
  constexpr uint32_t rng_reverse_block_size = 512;
  rng_prune_and_add_reverse_kernel<
      rng_reverse_grid_size, rng_reverse_block_size, base_num,
      top1_match_graph_degree, top1_pruneed_graph_degree, reverse_graph_degree,
      tomb, dim, true, float, uint32_t>
      <<<rng_reverse_grid_size, rng_reverse_block_size>>>(
          d_base_data, d_top1_match_graph, d_reverse_graph,
          d_top1_pruned_graph);

  auto h_pre_prune_g =
      std::vector<uint32_t>(base_num * top1_pruneed_graph_degree);
  cudaMemcpy(h_pre_prune_g.data(), d_top1_pruned_graph,
             base_num * top1_pruneed_graph_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  dump_vec2_file(
      h_pre_prune_g,
      "/home/shiwen/project/GGidxbuild/data/top1_pre_pruned_graph.ibin");

  auto h_pre_prune_reverse_g =
      std::vector<uint32_t>(base_num * reverse_graph_degree);
  cudaMemcpy(h_pre_prune_reverse_g.data(), d_reverse_graph,
             base_num * reverse_graph_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  dump_vec2_file(
      h_pre_prune_reverse_g,
      "/home/shiwen/project/GGidxbuild/data/top1_pre_pruned_reverse.ibin");

  // cudaMalloc(&d_prune_final_graph,
  //            base_num * prune_final_graph_degree * sizeof(uint32_t));
  // cudaCheckError();

  constexpr uint32_t merge_to_reverse_grid_size = 144;
  constexpr uint32_t merge_to_reverse_block_size = 512;
  merge_prune_graph_to_reverse_graph_kernel<
      merge_to_reverse_grid_size, merge_to_reverse_block_size, base_num,
      top1_pruneed_graph_degree, reverse_graph_degree, tomb, dim, true, float,
      uint32_t><<<merge_to_reverse_grid_size, merge_to_reverse_block_size>>>(
      d_base_data, d_reverse_graph, d_top1_pruned_graph);

  auto h_merge_g = std::vector<uint32_t>(base_num * reverse_graph_degree);
  cudaMemcpy(h_merge_g.data(), d_reverse_graph,
             base_num * reverse_graph_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  dump_vec2_file(h_merge_g,
                 "/home/shiwen/project/GGidxbuild/data/top1_merge_graph.ibin");

  constexpr uint32_t sort_neighbor_grid_size1 = 144;
  constexpr uint32_t sort_neighbor_block_size1 = 256;
  constexpr uint32_t sort_neighbor_share_memory_size1 =
      sizeof(
          compute_sort_warp_state<uint32_t, float, reverse_graph_degree, dim>) *
      sort_neighbor_block_size / 32;
  ;
  reverse_sort_kernel<
      sort_neighbor_grid_size1, sort_neighbor_block_size1, base_num,
      reverse_graph_degree, tomb, dim, sort_neighbor_share_memory_size1,
      float, uint32_t><<<sort_neighbor_grid_size1, sort_neighbor_block_size1,
                         sort_neighbor_share_memory_size1 /*,
                         main_stream*/>>>(
      d_base_data, d_reverse_graph);

  auto h_merge_sort_g = std::vector<uint32_t>(base_num * reverse_graph_degree);
  cudaMemcpy(h_merge_sort_g.data(), d_reverse_graph,
             base_num * reverse_graph_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  dump_vec2_file(
      h_merge_sort_g,
      "/home/shiwen/project/GGidxbuild/data/top1_merge_sort_graph.ibin");

  constexpr uint32_t final_prune_grid_size = 144;
  constexpr uint32_t final_prune_block_size = 512;
  prune_merge_graph_without_distance<
      final_prune_grid_size, final_prune_block_size, base_num,
      reverse_graph_degree, top1_pruneed_graph_degree, tomb, dim, true, float,
      uint32_t><<<final_prune_grid_size, final_prune_block_size>>>(
      d_base_data, d_reverse_graph, d_top1_pruned_graph);

  auto h_top1_projection_graph =
      std::vector<uint32_t>(base_num * top1_pruneed_graph_degree);
  cudaMemcpy(h_top1_projection_graph.data(), d_top1_pruned_graph,
             base_num * top1_pruneed_graph_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  dump_vec2_file(
      h_top1_projection_graph,
      "/home/shiwen/project/GGidxbuild/data/top1_projection_graph.ibin");

  // NOTE(shiwen): now the d_top1_match_graph is free.

  // cudaEventRecordWithFlags(end_construction, main_stream,
  //                          cudaEventRecordExternal);
  // cudaStreamEndCapture(main_stream, &graph);
  // cudaGraphInstantiate(&instance, graph);
  // cudaGraphLaunch(instance, main_stream);
  // cudaStreamSynchronize(main_stream);
  // float construction_time;
  // cudaEventElapsedTime(&construction_time, start_construction,
  //                      end_construction);
}
}  // namespace Gpu
}  // namespace Gbuilder
