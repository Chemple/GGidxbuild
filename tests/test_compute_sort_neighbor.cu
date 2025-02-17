#include "Gbuilder.cuh"
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>
#include <cuda_runtime.h>

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

// 辅助函数：比较两个float数组是否近似相等
void compare_float_arrays(float const* a, float const* b, size_t n,
                          float epsilon = 1e-5) {
  for (size_t i = 0; i < n; ++i) {
    ASSERT_NEAR(a[i], b[i], epsilon) << "at index " << i;
  }
}

void compare_uint32_arrays(uint32_t const* a, uint32_t const* b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    ASSERT_EQ(a[i], b[i]) << "at index " << i;
  }
}

void compare_id_and_distance(uint32_t const* a, uint32_t const* b,
                             float const* x, float const* y, size_t n,
                             float epsilon = 1e-5) {
  for (size_t i = 0; i < n; ++i) {
    if (a[i] != b[i]) {
      if (x[i] != y[i]) {
        ASSERT_NEAR(x[i], y[i], epsilon) << "at index " << i;
      }
    }
  }
  for (size_t i = 0; i < n; ++i) {
    ASSERT_NEAR(x[i], y[i], epsilon) << "at index " << i;
  }
}

// 测试固件类
class IPDistanceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 初始化随机数生成器
    std::mt19937 gen(43);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    base_data.resize(base_num * dim);
    for (auto& val : base_data) {
      val = dist(gen);
    }

    initial_graph.resize(base_num * graph_max_in_degree);
    std::uniform_int_distribution<uint32_t> id_dist(
        0, base_num - 2);  // 避免自连接

    for (uint32_t i = 0; i < base_num; ++i) {
      // 前50%为有效邻居，后50%为tomb
      uint32_t valid_count = graph_max_in_degree / 2;
      for (uint32_t j = 0; j < graph_max_in_degree; ++j) {
        if (j < valid_count) {
          // 生成不等于当前节点i的ID
          uint32_t neighbor_id = id_dist(gen);
          neighbor_id += (neighbor_id >= i);  // 跳过当前节点
          initial_graph[i * graph_max_in_degree + j] = neighbor_id;
        } else {
          initial_graph[i * graph_max_in_degree + j] = tomb;
        }
      }
    }

    // 分配设备内存
    cudaMalloc(&d_base_data, base_data.size() * sizeof(float));
    cudaMalloc(&d_graph, initial_graph.size() * sizeof(uint32_t));
    cudaMalloc(&d_neighbor_distance,
               base_num * graph_max_in_degree * sizeof(float));

    // 拷贝数据到设备
    cudaMemcpy(d_base_data, base_data.data(), base_data.size() * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph, initial_graph.data(),
               initial_graph.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);
  }

  void TearDown() override {
    cudaFree(d_base_data);
    cudaFree(d_graph);
    cudaFree(d_neighbor_distance);
    cudaDeviceReset();
  }

  // 公共测试参数
  static constexpr uint32_t base_num = 64;
  static constexpr uint32_t dim = 128;
  static constexpr uint32_t graph_max_in_degree = 32;
  static constexpr uint32_t pruned_max_in_degree = 16;
  static constexpr uint32_t tomb = 0XFFFFFFFF;
  static constexpr uint32_t grid_size = 72;
  static constexpr uint32_t block_size = 256;

  static constexpr uint32_t shared_memory_size =
      (dim * sizeof(float) * 2 +
       graph_max_in_degree * (sizeof(uint32_t) + sizeof(float))) *
      block_size / 32;

  // 测试数据
  std::vector<float> base_data{};

  std::vector<uint32_t> initial_graph{};

  // 设备指针
  float* d_base_data = nullptr;
  uint32_t* d_graph = nullptr;
  float* d_neighbor_distance = nullptr;
};

TEST_F(IPDistanceTest, BasicFunctionality) {
  SPDLOG_INFO("begin gpu computation");
  SPDLOG_INFO("the share memory size is {}", shared_memory_size);

  // 启动内核
  compute_and_sort_ip_distance_kernel<grid_size, block_size, base_num,
                                      graph_max_in_degree, tomb, dim,
                                      shared_memory_size>
      <<<grid_size, block_size, shared_memory_size>>>(d_base_data, d_graph,
                                                      d_neighbor_distance);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(error));
  }

  cudaDeviceSynchronize();

  SPDLOG_INFO("finish gpu computation");

  // 拷贝结果回主机
  std::vector<uint32_t> gpu_graph(initial_graph.size());
  std::vector<float> gpu_dist(base_num * graph_max_in_degree);
  cudaMemcpy(gpu_graph.data(), d_graph, initial_graph.size() * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(gpu_dist.data(), d_neighbor_distance,
             base_num * graph_max_in_degree * sizeof(float),
             cudaMemcpyDeviceToHost);

  SPDLOG_INFO("finish gpu data transfering");

  // 计算CPU结果
  std::vector<uint32_t> cpu_graph = initial_graph;
  std::vector<float> cpu_dist(base_num * graph_max_in_degree);
  parallel_compute_and_sort_ip_distance_cpu(base_data.data(), cpu_graph.data(),
                                            cpu_dist.data(), base_num, dim,
                                            graph_max_in_degree, tomb);

  SPDLOG_INFO("finish cpu data computation");

  compare_id_and_distance(gpu_graph.data(), cpu_graph.data(), gpu_dist.data(),
                          cpu_dist.data(), base_num * graph_max_in_degree);

  // // 验证结果
  // compare_uint32_arrays(gpu_graph.data(), cpu_graph.data(),
  //                       base_num * max_in_degree);
  // compare_float_arrays(gpu_dist.data(), cpu_dist.data(),
  //                      base_num * max_in_degree);
}

TEST_F(IPDistanceTest, AllTombstoneCase) {
  // 修改初始图为全tombstone
  std::fill(initial_graph.begin(), initial_graph.end(), tomb);
  cudaMemcpy(d_graph, initial_graph.data(),
             initial_graph.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

  // 启动内核
  compute_and_sort_ip_distance_kernel<grid_size, block_size, base_num,
                                      graph_max_in_degree, tomb, dim,
                                      shared_memory_size>
      <<<grid_size, block_size, shared_memory_size>>>(d_base_data, d_graph,
                                                      d_neighbor_distance);

  // 获取GPU结果
  std::vector<uint32_t> gpu_graph(initial_graph.size());
  std::vector<float> gpu_dist(base_num * graph_max_in_degree);
  cudaMemcpy(gpu_graph.data(), d_graph, initial_graph.size() * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(gpu_dist.data(), d_neighbor_distance,
             base_num * graph_max_in_degree * sizeof(float),
             cudaMemcpyDeviceToHost);

  // 预期结果：所有位置都是tomb和0
  std::vector<uint32_t> expected_graph(initial_graph.size(), tomb);
  std::vector<float> expected_dist(base_num * graph_max_in_degree, FLT_MAX);

  ASSERT_EQ(gpu_graph, expected_graph);
  ASSERT_EQ(gpu_dist, expected_dist);
}

}  // namespace Gpu
}  // namespace Gbuilder
