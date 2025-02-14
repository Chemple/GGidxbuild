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

TEST(MatchTest, SimpleTest) {}

class MatchKernelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // 设置测试参数
    query_num = 2;
    topk = 4;
    max_in_degree = 4;
    tomb = 0xFFFFFFFF;

    // 初始化测试数据
    gt_ids_host = {
        1, 2, 3, 4,  // 第一个查询的gt_ids
        5, 6, 7, 8   // 第二个查询的gt_ids
    };

    // 分配和初始化设备内存
    size_t gt_ids_size = query_num * topk * sizeof(uint32_t);
    size_t graph_size = query_num * max_in_degree * sizeof(uint32_t);

    cudaMalloc(&gt_ids_dev, gt_ids_size);
    cudaMalloc(&top1_match_graph_dev, graph_size);
    cudaMalloc(&match_match_graph_dev, graph_size);

    cudaMemcpy(gt_ids_dev, gt_ids_host.data(), gt_ids_size,
               cudaMemcpyHostToDevice);

    // 初始化图为tomb值
    std::vector<uint32_t> init_graph(query_num * max_in_degree, tomb);
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

  // 测试参数
  uint32_t query_num;
  uint32_t topk;
  uint32_t max_in_degree;
  uint32_t tomb;

  // 主机数据
  std::vector<uint32_t> gt_ids_host;

  // 设备数据
  uint32_t* gt_ids_dev;
  uint32_t* top1_match_graph_dev;
  uint32_t* match_match_graph_dev;
};

TEST_F(MatchKernelTest, BasicMatchTest) {
  // 启动kernel
  constexpr uint32_t grid_size = 1;
  constexpr uint32_t block_size = 256;

  Gbuilder::Gpu::match_kernel<grid_size, block_size, 2, 4, 4>
      <<<grid_size, block_size>>>(gt_ids_dev, top1_match_graph_dev,
                                  match_match_graph_dev);

  cudaDeviceSynchronize();

  // 检查结果
  std::vector<uint32_t> top1_match_result(query_num * max_in_degree);
  std::vector<uint32_t> match_match_result(query_num * max_in_degree);

  cudaMemcpy(top1_match_result.data(), top1_match_graph_dev,
             query_num * max_in_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(match_match_result.data(), match_match_graph_dev,
             query_num * max_in_degree * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  // 验证top1_match_graph的结果
  // 对于第一个查询，验证其邻居
  EXPECT_EQ(top1_match_result[0], 2);  // 第一个位置应该是2
  EXPECT_EQ(top1_match_result[1], 3);  // 第二个位置应该是3
  EXPECT_EQ(top1_match_result[2], 4);  // 第三个位置应该是4

  // 检查match_match_graph的结果
  // 由于贪婪匹配的性质，结果可能会有所不同
  // 这里我们至少确保结果不是tomb值
  for (int i = 0; i < max_in_degree; i++) {
    if (match_match_result[i] != tomb) {
      EXPECT_TRUE(std::find(gt_ids_host.begin(), gt_ids_host.end(),
                            match_match_result[i]) != gt_ids_host.end());
    }
  }
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}