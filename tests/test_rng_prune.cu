#include "Gbuilder.cuh"
#include "spdlog/spdlog.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <random>
#include <set>
#include <vector>
#include <cuda_runtime.h>
#include <omp.h>

namespace Gbuilder {
namespace Gpu {
template <typename data_type, typename id_type>
void cpu_rng_prune(data_type const* base_data, id_type const* graph,
                   data_type const* neighbor_distance, id_type* pruned_graph,
                   uint32_t base_num, uint32_t dim,
                   uint32_t graph_max_in_degree,
                   uint32_t pruned_graph_max_in_degree,
                   id_type tomb = 0xFFFFFFFF) {
  omp_set_num_threads(64);
#pragma omp parallel for
  for (uint32_t base_id = 0; base_id < base_num; ++base_id) {
    std::vector<id_type> pruned_neighbors;
    id_type const* neighbors = &graph[base_id * graph_max_in_degree];
    data_type const* distances =
        &neighbor_distance[base_id * graph_max_in_degree];

    // 添加第一个有效邻居
    uint32_t neighbor_idx = 0;
    while (neighbor_idx < graph_max_in_degree &&
           neighbors[neighbor_idx] == tomb) {
      ++neighbor_idx;
    }
    if (neighbor_idx >= graph_max_in_degree) continue;
    pruned_neighbors.push_back(neighbors[neighbor_idx++]);

    // 处理后续邻居
    for (; neighbor_idx < graph_max_in_degree; ++neighbor_idx) {
      id_type curr_neighbor = neighbors[neighbor_idx];
      if (curr_neighbor == tomb) break;

      bool keep = false;
      for (id_type existing : pruned_neighbors) {
        // 计算curr_neighbor与existing的内积距离
        data_type ip = 0;
        for (uint32_t d = 0; d < dim; ++d) {
          ip += base_data[curr_neighbor * dim + d] *
                base_data[existing * dim + d];
        }
        data_type curr_dist = -ip;

        // 检查是否满足RNG条件
        if (curr_dist < distances[neighbor_idx]) {
          keep = true;
          break;
        }
      }

      if (!keep) {
        pruned_neighbors.push_back(curr_neighbor);
        if (pruned_neighbors.size() >= pruned_graph_max_in_degree) break;
      }
    }

    // 写入结果
    uint32_t write_size =
        std::min(pruned_neighbors.size(), (size_t)pruned_graph_max_in_degree);
    for (uint32_t i = 0; i < write_size; ++i) {
      pruned_graph[base_id * pruned_graph_max_in_degree + i] =
          pruned_neighbors[i];
    }
    // // 填充tombstone
    // for (uint32_t i = write_size; i < pruned_graph_max_in_degree; ++i) {
    //   pruned_graph[base_id * pruned_graph_max_in_degree + i] = tomb;
    // }
  }
}

class RNGPruneTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 初始化随机数生成器
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    // 分配主机内存
    host_base_data.resize(base_num * dim);
    host_graph.resize(base_num * graph_max_in_degree);
    host_pruned_gpu.resize(base_num * pruned_max_in_degree);
    host_pruned_cpu.resize(base_num * pruned_max_in_degree);

    // 生成随机base数据
    for (auto& v : host_base_data) v = dist(gen);

    SPDLOG_INFO("begin gen the graph");
    // 生成随机图结构（填充有效邻居）
    std::uniform_int_distribution<uint32_t> id_dist(0, base_num - 1);
    omp_set_num_threads(64);

#pragma omp parallel for
    for (uint32_t i = 0; i < base_num; ++i) {
      auto set = std::set<uint32_t>{};
      set.insert(i);
      for (uint32_t j = 0; j < graph_max_in_degree; ++j) {
        auto select = id_dist(gen);
        while (set.find(select) != set.end()) {
          select = (select + 1) % base_num;
          // SPDLOG_INFO("the select is {}, i is {}", select, i);
        }
        set.insert(select);
        host_graph[i * graph_max_in_degree + j] = select;
      }
    }

    SPDLOG_INFO("finish setup...");
  }

  using data_type = float;
  using id_type = uint32_t;

  // 测试参数
  static constexpr uint32_t base_num = 10 * 1024 * 1024;
  static constexpr uint32_t dim = 128;
  static constexpr uint32_t graph_max_in_degree = 128;
  static constexpr uint32_t pruned_max_in_degree = 32;
  static constexpr uint32_t tomb = 0XFFFFFFFF;
  static constexpr uint32_t grid_size = 72;
  static constexpr uint32_t block_size = 256;

  static constexpr uint32_t compute_sort_share_memory =
      (dim * sizeof(data_type) * 2 +
       graph_max_in_degree * (sizeof(data_type) + sizeof(id_type))) *
      block_size / 32;

  // 主机数据
  std::vector<float> host_base_data;
  std::vector<uint32_t> host_graph;
  std::vector<uint32_t> host_pruned_gpu;
  std::vector<uint32_t> host_pruned_cpu;
};

TEST_F(RNGPruneTest, BasicPrune) {
  // 分配设备内存
  float* d_base_data;
  uint32_t *d_graph, *d_pruned;
  float* d_neighbor_dist;
  std::vector<data_type> h_distance(base_num * graph_max_in_degree);

  cudaMalloc(&d_base_data, host_base_data.size() * sizeof(float));
  cudaMalloc(&d_graph, host_graph.size() * sizeof(uint32_t));
  cudaMalloc(&d_pruned, host_pruned_gpu.size() * sizeof(uint32_t));
  cudaMalloc(&d_neighbor_dist, base_num * graph_max_in_degree * sizeof(float));

  // 拷贝数据到设备
  cudaMemcpy(d_base_data, host_base_data.data(),
             host_base_data.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_graph, host_graph.data(), host_graph.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
  }
  cudaDeviceSynchronize();

  SPDLOG_INFO("finish coping data to GPU");

  // 运行第一个kernel（排序）
  compute_and_sort_ip_distance_kernel<grid_size, block_size, base_num,
                                      graph_max_in_degree, tomb, dim,
                                      compute_sort_share_memory>
      <<<grid_size, block_size, compute_sort_share_memory>>>(
          d_base_data, d_graph, d_neighbor_dist);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
  }
  cudaDeviceSynchronize();

  SPDLOG_INFO("finish kernel1");

  // 运行第二个kernel（剪枝）
  rng_prune_kernel<grid_size, block_size, base_num, graph_max_in_degree,
                   pruned_max_in_degree, tomb, dim, true>
      <<<grid_size, block_size>>>(d_base_data, d_graph, d_neighbor_dist,
                                  d_pruned);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
  }
  cudaDeviceSynchronize();

  SPDLOG_INFO("finish kernel2");

  // 拷贝结果回主机
  cudaMemcpy(host_pruned_gpu.data(), d_pruned,
             host_pruned_gpu.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  cudaMemcpy(h_distance.data(), d_neighbor_dist,
             base_num * graph_max_in_degree * sizeof(float),
             cudaMemcpyDeviceToHost);

  cudaMemcpy(host_graph.data(), d_graph,
             base_num * graph_max_in_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
  }
  cudaDeviceSynchronize();

  SPDLOG_INFO("finish data coping to host");

  // 运行CPU版本
  cpu_rng_prune<float, uint32_t>(host_base_data.data(), host_graph.data(),
                                 h_distance.data(), host_pruned_cpu.data(),
                                 base_num, dim, graph_max_in_degree,
                                 pruned_max_in_degree);

  SPDLOG_INFO("finish computing in host");

  // 比较结果
  for (uint32_t i = 0; i < base_num; ++i) {
    for (uint32_t j = 0; j < pruned_max_in_degree; ++j) {
      uint32_t gpu_val = host_pruned_gpu[i * pruned_max_in_degree + j];
      uint32_t cpu_val = host_pruned_cpu[i * pruned_max_in_degree + j];

      // 验证有效邻居是否一致
      if (gpu_val != 0xFFFFFFFF) {
        EXPECT_EQ(gpu_val, cpu_val)
            << "Mismatch at base " << i << ", neighbor " << j;
      }
    }
  }

  // 释放设备内存
  cudaFree(d_base_data);
  cudaFree(d_graph);
  cudaFree(d_pruned);
  cudaFree(d_neighbor_dist);
}

}  // namespace Gpu
}  // namespace Gbuilder
