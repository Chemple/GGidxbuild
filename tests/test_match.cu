#include "Gbuilder.cuh"
#include <gtest/gtest.h>
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <sys/types.h>
#include <random>

class MatchKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 测试参数设置
    query_num = 10000;
    dim = 128;
    topk = 100;
    max_in_degree = 100;
    tomb = 0xFFFFFFFF;

    // 生成gt_ids数据
    gt_ids_host.resize(query_num * topk);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(0, 999);  // 生成0-999的随机数

    // 填充gt_ids并找到最大值
    base_num = 0;
    for (size_t i = 0; i < query_num * topk; ++i) {
      gt_ids_host[i] = dis(gen);
      base_num = std::max(base_num, gt_ids_host[i]);
    }
    base_num++;  // 最大值+1作为base_num

    // 计算graph大小
    graph_size = base_num * (topk - 1) * sizeof(uint32_t);

    // 分配设备内存
    cudaMalloc(&gt_ids_dev, query_num * topk * sizeof(uint32_t));
    cudaMalloc(&top1_match_graph_dev, graph_size);
    cudaMalloc(&match_match_graph_dev, graph_size);

    // 拷贝gt_ids到设备
    cudaMemcpy(gt_ids_dev, gt_ids_host.data(),
               query_num * topk * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // 初始化graph为tomb值
    std::vector<uint32_t> init_graph(base_num * (topk - 1), tomb);
    cudaMemcpy(top1_match_graph_dev, init_graph.data(), graph_size,
               cudaMemcpyHostToDevice);
    cudaMemcpy(match_match_graph_dev, init_graph.data(), graph_size,
               cudaMemcpyHostToDevice);
  }

  void TearDown() override {
    cudaFree(gt_ids_dev);
    cudaFree(top1_match_graph_dev);
    cudaFree(match_match_graph_dev);
  }

  uint32_t query_num;
  uint32_t dim;
  uint32_t topk;
  uint32_t max_in_degree;
  uint32_t tomb;
  uint32_t base_num;
  size_t graph_size;

  std::vector<uint32_t> gt_ids_host;
  uint32_t* gt_ids_dev;
  uint32_t* top1_match_graph_dev;
  uint32_t* match_match_graph_dev;
};

TEST_F(MatchKernelTest, LargeScaleMatchTest) {
  // 设置kernel启动参数
  constexpr uint32_t block_size = 256;
  uint32_t grid_size = (query_num + block_size - 1) / block_size;

  // 启动kernel
  Gbuilder::Gpu::match_kernel<block_size, 256, 100, 4, 4>
      <<<grid_size, block_size>>>(gt_ids_dev, top1_match_graph_dev,
                                  match_match_graph_dev);

  cudaDeviceSynchronize();

  // 检查CUDA错误
  cudaError_t error = cudaGetLastError();
  EXPECT_EQ(error, cudaSuccess) << "CUDA error: " << cudaGetErrorString(error);

  // 获取结果
  std::vector<uint32_t> top1_match_result(base_num * (topk - 1));
  std::vector<uint32_t> match_match_result(base_num * (topk - 1));

  cudaMemcpy(top1_match_result.data(), top1_match_graph_dev, graph_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(match_match_result.data(), match_match_graph_dev, graph_size,
             cudaMemcpyDeviceToHost);

  // 验证结果
  // 1. 检查是否所有非tomb值都是来自gt_ids
  for (size_t i = 0; i < base_num * (topk - 1); ++i) {
    if (top1_match_result[i] != tomb) {
      EXPECT_TRUE(std::find(gt_ids_host.begin(), gt_ids_host.end(),
                            top1_match_result[i]) != gt_ids_host.end())
          << "Invalid value in top1_match_graph at position " << i;
    }
    if (match_match_result[i] != tomb) {
      EXPECT_TRUE(std::find(gt_ids_host.begin(), gt_ids_host.end(),
                            match_match_result[i]) != gt_ids_host.end())
          << "Invalid value in match_match_graph at position " << i;
    }
  }

  // 2. 验证每个base_id的邻居数量不超过topk-1
  for (uint32_t base_id = 0; base_id < base_num; ++base_id) {
    int top1_neighbors = 0;
    int match_neighbors = 0;

    for (uint32_t j = 0; j < topk - 1; ++j) {
      if (top1_match_result[base_id * (topk - 1) + j] != tomb) top1_neighbors++;
      if (match_match_result[base_id * (topk - 1) + j] != tomb)
        match_neighbors++;
    }

    EXPECT_LE(top1_neighbors, topk - 1)
        << "Too many neighbors in top1_match_graph for base_id " << base_id;
    EXPECT_LE(match_neighbors, topk - 1)
        << "Too many neighbors in match_match_graph for base_id " << base_id;
  }
}
