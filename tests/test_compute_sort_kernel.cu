#include "Gbuilder.cuh"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <gtest/gtest.h>
#include <cassert>
#include <cstdint>
#include <random>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <sys/types.h>

namespace Gbuilder {
namespace Gpu {

// CPU参考实现
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

void init_full_host_graph(std::vector<uint32_t>& host_graph, uint32_t degree,
                          uint32_t base_num) {
  assert(degree * base_num == host_graph.size());

#pragma omp parallel for
  for (uint32_t i = 0; i < base_num; i++) {
    std::mt19937 gen(11);
    std::uniform_int_distribution<uint32_t> id_dist(0, base_num - 1);
    auto set = std::unordered_set<uint32_t>{i};
    for (uint32_t j = 0; j < degree; j++) {
      uint32_t select = id_dist(gen);
      while (set.find(select) != set.end()) {
        select = (select + 1) % base_num;
      }
      if (select >= base_num) {
        SPDLOG_ERROR("the neighbor id is wrong!");
      }
      // #pragma omp critical
      // {
      set.insert(select);
      // }
      host_graph[i * degree + j] = select;
    }
  }

  for (uint32_t i = 0; i < base_num; i++) {
    for (uint32_t j = 0; j < degree; j++) {
      if (host_graph[i * degree + j] >= base_num) {
        SPDLOG_ERROR("post the neighbor id is wrong!");
      }
    }
  }
}

void init_partial_host_graph(std::vector<uint32_t>& host_graph, uint32_t degree,
                             uint32_t base_num) {
  auto invalid_row_percent = 70;
  auto invalid_col_percent = 20;
  assert(degree * base_num == host_graph.size());
  std::mt19937 gen(11);
  std::uniform_int_distribution<uint32_t> id_dist(0, base_num - 1);
#pragma omp parallel for
  for (uint32_t i = 0; i < base_num; i++) {
    auto set = std::unordered_set<uint32_t>{i};
    for (uint32_t j = 0; j < degree; j++) {
      uint32_t select = id_dist(gen);
      while (set.find(select) != set.end()) {
        select = (select + 1) % base_num;
      }
      if (select >= base_num) {
        SPDLOG_ERROR("the neighbor id is wrong!");
      }
#pragma omp critical
      {
        set.insert(select);
      }
      host_graph[i * degree + j] = select;
    }
  }

  for (uint32_t i = 0; i < base_num; i++) {
    for (uint32_t j = 0; j < degree; j++) {
      if (host_graph[i * degree + j] >= base_num) {
        SPDLOG_ERROR("post the neighbor id is wrong!");
      }
    }
  }
}

void init_base_data(std::vector<float>& base_data, uint32_t dim,
                    uint32_t base_num) {
  assert(dim * base_num == base_data.size());
  std::mt19937 gen(11);
  std::uniform_real_distribution<float> dist(-1, 1);
#pragma omp parallel for
  for (uint32_t i = 0; i < base_num; i++) {
    for (uint32_t j = 0; j < dim; j++) {
      auto select = dist(gen);
      base_data[i * dim + j] = select;
    }
  }
}

TEST(compute_test, allvalidtest) {
  omp_set_num_threads(64);
  constexpr uint32_t base_number = 1024 * 1024 * 10;
  constexpr uint32_t degree = 64;
  constexpr uint32_t dim = 128;
  constexpr uint32_t grid_size = 72;
  constexpr uint32_t block_size = 256;
  constexpr uint32_t share_memory_size =
      sizeof(compute_sort_warp_state<uint32_t, float, degree, dim>) *
      block_size / 32;
  auto host_graph = std::vector<uint32_t>(base_number * degree);
  auto host_base_data = std::vector<float>(base_number * dim);
  init_base_data(host_base_data, dim, base_number);
  init_full_host_graph(host_graph, degree, base_number);

  uint32_t* d_graph;
  float* d_base_data;
  float* d_neighbor_distance;

  cudaMalloc(&d_graph, sizeof(uint32_t) * base_number * degree);
  cudaCheckError();
  cudaMalloc(&d_base_data, sizeof(float) * base_number * dim);
  cudaCheckError();
  cudaMalloc(&d_neighbor_distance, sizeof(float) * base_number * degree);
  cudaCheckError();

  cudaMemcpy(d_base_data, host_base_data.data(),
             sizeof(float) * base_number * dim, cudaMemcpyHostToDevice);
  cudaCheckError();

  cudaMemcpy(d_graph, host_graph.data(),
             sizeof(uint32_t) * base_number * degree, cudaMemcpyHostToDevice);
  cudaCheckError();

  SPDLOG_INFO("finish cuda malloc and cuda memcpy");
  compute_and_sort_ip_distance_kernel<grid_size, block_size, base_number,
                                      degree, 0xFFFFFFFF, dim,
                                      share_memory_size>
      <<<grid_size, block_size, share_memory_size>>>(d_base_data, d_graph,
                                                     d_neighbor_distance);

  cudaCheckError();

  cudaDeviceSynchronize();
  SPDLOG_INFO("finish gpu compute");

  auto h_neighbor_distance = std::vector<float>(base_number * degree);

  auto host_grah_test = std::vector<uint32_t>{};
  host_grah_test = host_graph;

  parallel_compute_and_sort_ip_distance_cpu(
      host_base_data.data(), host_graph.data(), h_neighbor_distance.data(),
      base_number, dim, degree, 0xFFFFFFFF);

  auto check_graph = std::vector<uint32_t>(base_number * degree);
  auto check_neighbor_distance = std::vector<float>(base_number * degree);

  cudaMemcpy(check_graph.data(), d_graph,
             sizeof(uint32_t) * base_number * degree, cudaMemcpyDeviceToHost);
  cudaMemcpy(check_neighbor_distance.data(), d_neighbor_distance,
             sizeof(float) * base_number * degree, cudaMemcpyDeviceToHost);

  for (auto i = 0; i < base_number; i++) {
    for (auto j = 0; j < degree; j++) {
      ASSERT_NEAR(check_neighbor_distance[i * degree + j],
                  h_neighbor_distance[i * degree + j], 1e-5);
      if (check_graph[i * degree + j] != host_graph[i * degree + j]) {
        printf(" %ud %ud\n", check_graph[i * degree + j],
               host_graph[i * degree + j]);
      }
    }
  }

  // cudaCheckError();
}

TEST(compute_test, DISABLED_someinvalidtest) {
  omp_set_num_threads(64);
  constexpr uint32_t base_number = 1024 * 1024 * 10;
  constexpr uint32_t degree = 64;
  constexpr uint32_t dim = 128;
  constexpr uint32_t grid_size = 72;
  constexpr uint32_t block_size = 256;
  constexpr uint32_t share_memory_size =
      sizeof(compute_sort_warp_state<uint32_t, float, degree, dim>) *
      block_size / 32;
  auto host_graph = std::vector<uint32_t>(base_number * degree);
  auto host_base_data = std::vector<float>(base_number * dim);
  init_base_data(host_base_data, dim, base_number);
  init_full_host_graph(host_graph, degree, base_number);

  uint32_t* d_graph;
  float* d_base_data;
  float* d_neighbor_distance;

  cudaMalloc(&d_graph, sizeof(uint32_t) * base_number * degree);
  cudaCheckError();
  cudaMalloc(&d_base_data, sizeof(float) * base_number * dim);
  cudaCheckError();
  cudaMalloc(&d_neighbor_distance, sizeof(float) * base_number * degree);
  cudaCheckError();

  cudaMemcpy(d_base_data, host_base_data.data(),
             sizeof(float) * base_number * dim, cudaMemcpyHostToDevice);
  cudaCheckError();

  cudaMemcpy(d_graph, host_graph.data(),
             sizeof(uint32_t) * base_number * degree, cudaMemcpyHostToDevice);
  cudaCheckError();

  SPDLOG_INFO("finish cuda malloc and cuda memcpy");
  compute_and_sort_ip_distance_kernel<grid_size, block_size, base_number,
                                      degree, 0xFFFFFFFF, dim,
                                      share_memory_size>
      <<<grid_size, block_size, share_memory_size>>>(d_base_data, d_graph,
                                                     d_neighbor_distance);

  cudaCheckError();

  cudaDeviceSynchronize();
  SPDLOG_INFO("finish gpu compute");

  auto h_neighbor_distance = std::vector<float>(base_number * degree);

  auto host_grah_test = std::vector<uint32_t>{};
  host_grah_test = host_graph;

  parallel_compute_and_sort_ip_distance_cpu(
      host_base_data.data(), host_graph.data(), h_neighbor_distance.data(),
      base_number, dim, degree, 0xFFFFFFFF);

  auto check_graph = std::vector<uint32_t>(base_number * degree);
  auto check_neighbor_distance = std::vector<float>(base_number * degree);

  cudaMemcpy(check_graph.data(), d_graph,
             sizeof(uint32_t) * base_number * degree, cudaMemcpyDeviceToHost);
  cudaMemcpy(check_neighbor_distance.data(), d_neighbor_distance,
             sizeof(float) * base_number * degree, cudaMemcpyDeviceToHost);

  for (auto i = 0; i < base_number; i++) {
    for (auto j = 0; j < degree; j++) {
      ASSERT_NEAR(check_neighbor_distance[i * degree + j],
                  h_neighbor_distance[i * degree + j], 1e-5);
      if (check_graph[i * degree + j] != host_graph[i * degree + j]) {
        printf(" %ud %ud\n", check_graph[i * degree + j],
               host_graph[i * degree + j]);
      }
    }
  }

  // cudaCheckError();
}
}  // namespace Gpu
}  // namespace Gbuilder
