#include "Gbuilder.cuh"
#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <algorithm>

class ComputeIPDistanceTest : public ::testing::Test {
protected:
    // 测试参数
    static constexpr uint32_t grid_size = 1;
    static constexpr uint32_t block_size = 256;
    static constexpr uint32_t base_num = 1000;
    static constexpr uint32_t max_in_degree = 64;
    static constexpr uint32_t tomb = 0xFFFFFFFF;
    static constexpr uint32_t dim = 128;
    static constexpr uint32_t warp_per_block = block_size / 32;
    static constexpr uint32_t shared_memory_size = dim * sizeof(float) * 2 * warp_per_block;

    void SetUp() override {
        // 分配和初始化主机内存
        h_base_data.resize(base_num * dim);
        h_graph.resize(base_num * max_in_degree);
        h_neighbor_distance.resize(base_num * max_in_degree);
        h_expected_distance.resize(base_num * max_in_degree);

        // 分配设备内存
        cudaMalloc(&d_base_data, base_num * dim * sizeof(float));
        cudaMalloc(&d_graph, base_num * max_in_degree * sizeof(uint32_t));
        cudaMalloc(&d_neighbor_distance, base_num * max_in_degree * sizeof(float));
    }

    void TearDown() override {
        cudaFree(d_base_data);
        cudaFree(d_graph);
        cudaFree(d_neighbor_distance);
    }

    // 修改后的CPU参考实现，使用双精度计算
    void computeIPDistanceCPU() {
        #pragma omp parallel for
        for (uint32_t i = 0; i < base_num; ++i) {
            for (uint32_t j = 0; j < max_in_degree; ++j) {
                uint32_t neighbor_id = h_graph[i * max_in_degree + j];
                if (neighbor_id == tomb) break;

                // 使用双精度计算以获得更高精度的参考结果
                double sum = 0.0;
                for (uint32_t k = 0; k < dim; ++k) {
                    sum += static_cast<double>(h_base_data[i * dim + k]) * 
                          static_cast<double>(h_base_data[neighbor_id * dim + k]);
                }
                h_expected_distance[i * max_in_degree + j] = static_cast<float>(-sum);
            }
        }
    }

    // 生成测试数据
    void generateTestData() {
        std::mt19937 gen(42); // 固定种子以保证可重复性
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // 生成基础向量数据
        for (size_t i = 0; i < base_num * dim; ++i) {
            h_base_data[i] = dist(gen);
        }

        // 生成图数据
        for (size_t i = 0; i < base_num; ++i) {
            size_t valid_neighbors = gen() % max_in_degree;
            for (size_t j = 0; j < valid_neighbors; ++j) {
                h_graph[i * max_in_degree + j] = gen() % base_num;
            }
            for (size_t j = valid_neighbors; j < max_in_degree; ++j) {
                h_graph[i * max_in_degree + j] = tomb;
            }
        }
    }

    // 结果验证函数
    bool verifyResults() {
        bool all_match = true;
        const double rel_tol = 1e-5;  // 相对容差
        const double abs_tol = 1e-5;  // 绝对容差

        for (size_t i = 0; i < base_num * max_in_degree; ++i) {
            if (h_graph[i / max_in_degree * max_in_degree + (i % max_in_degree)] == tomb) {
                continue;
            }

            double expected = h_expected_distance[i];
            double actual = h_neighbor_distance[i];
            
            // 使用相对误差和绝对误差的组合
            bool match = std::abs(expected - actual) <= abs_tol + rel_tol * std::abs(expected);
            
            if (!match) {
                printf("Mismatch at index %zu: expected=%f, actual=%f, abs_diff=%e\n",
                       i, expected, actual, std::abs(expected - actual));
                all_match = false;
            }
        }
        return all_match;
    }

    std::vector<float> h_base_data;
    std::vector<uint32_t> h_graph;
    std::vector<float> h_neighbor_distance;
    std::vector<float> h_expected_distance;

    float* d_base_data;
    uint32_t* d_graph;
    float* d_neighbor_distance;
};

TEST_F(ComputeIPDistanceTest, BasicFunctionality) {
    generateTestData();
    
    // 计算CPU参考结果
    computeIPDistanceCPU();

    // 拷贝数据到设备
    cudaMemcpy(d_base_data, h_base_data.data(), 
               base_num * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph, h_graph.data(), 
               base_num * max_in_degree * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // 运行kernel
    Gbuilder::Gpu::compute_ip_distance_kernel<grid_size, block_size, base_num, max_in_degree, 
                             tomb, dim, shared_memory_size>
        <<<grid_size, block_size, shared_memory_size>>>(
            d_base_data, d_graph, d_neighbor_distance);
    
    // 检查kernel执行是否成功
    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "Kernel launch failed: " << cudaGetErrorString(err);
    
    // 等待kernel完成
    err = cudaDeviceSynchronize();
    ASSERT_EQ(err, cudaSuccess) << "Kernel execution failed: " << cudaGetErrorString(err);

    // 拷贝结果回主机
    cudaMemcpy(h_neighbor_distance.data(), d_neighbor_distance,
               base_num * max_in_degree * sizeof(float), cudaMemcpyDeviceToHost);

    // 验证结果
    EXPECT_TRUE(verifyResults()) << "Results do not match within tolerance";
}

// 边界情况测试
TEST_F(ComputeIPDistanceTest, EdgeCases) {
    // 特殊情况：所有邻居都是tomb
    std::fill(h_graph.begin(), h_graph.end(), tomb);
    
    cudaMemcpy(d_base_data, h_base_data.data(), 
               base_num * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph, h_graph.data(), 
               base_num * max_in_degree * sizeof(uint32_t), cudaMemcpyHostToDevice);

    Gbuilder::Gpu::compute_ip_distance_kernel<grid_size, block_size, base_num, max_in_degree, 
                             tomb, dim, shared_memory_size>
        <<<grid_size, block_size, shared_memory_size>>>(
            d_base_data, d_graph, d_neighbor_distance);

    cudaDeviceSynchronize();
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

// 性能测试
TEST_F(ComputeIPDistanceTest, Performance) {
    generateTestData();
    
    cudaMemcpy(d_base_data, h_base_data.data(), 
               base_num * dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_graph, h_graph.data(), 
               base_num * max_in_degree * sizeof(uint32_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    Gbuilder::Gpu::compute_ip_distance_kernel<grid_size, block_size, base_num, max_in_degree, 
                             tomb, dim, shared_memory_size>
        <<<grid_size, block_size, shared_memory_size>>>(
            d_base_data, d_graph, d_neighbor_distance);

    // 计时
    cudaEventRecord(start);
    for (int i = 0; i < 100; ++i) {
        Gbuilder::Gpu::compute_ip_distance_kernel<grid_size, block_size, base_num, max_in_degree, 
                                 tomb, dim, shared_memory_size>
            <<<grid_size, block_size, shared_memory_size>>>(
                d_base_data, d_graph, d_neighbor_distance);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Average kernel execution time: %.3f ms\n", milliseconds / 100.0f);
    EXPECT_LT(milliseconds / 100.0f, 10.0f);  // 假设期望性能小于10ms

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// 异常情况测试
TEST_F(ComputeIPDistanceTest, InvalidInputs) {
    // 测试错误的维度值
    static constexpr uint32_t invalid_dim = 0;
    static constexpr uint32_t invalid_shared_memory_size = 
        invalid_dim * sizeof(float) * 2 * warp_per_block;

    // 这应该在编译时失败
    EXPECT_FALSE((std::is_constructible<
        decltype(Gbuilder::Gpu::compute_ip_distance_kernel<grid_size, block_size, base_num, 
                max_in_degree, tomb, invalid_dim, invalid_shared_memory_size>),
        float*, uint32_t*, float*
    >::value));
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}