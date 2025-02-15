#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <unordered_map>
#include <algorithm>

// 优化参数
#define BLOCK_SIZE 256
#define NUM_SMS 72
#define GRID_SIZE (NUM_SMS * 32)
#define VALUE_RANGE 10000000
#define HASH_BIN_COUNT 8192  // 2^13, 用于初步归约
#define ELEMENTS_PER_THREAD 8

// CUDA错误检查宏
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 使用向量加载来减少内存事务
__global__ void computeHistogramOptimized(const int4* __restrict__ matrix,
                                        unsigned int* __restrict__ hashHist,
                                        const size_t matrixSize) {
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ unsigned int localHist[HASH_BIN_COUNT];
    
    // 初始化共享内存
    for(int i = tid; i < HASH_BIN_COUNT; i += blockDim.x) {
        localHist[i] = 0;
    }
    __syncthreads();
    
    // 每个线程处理多个元素
    const size_t stride = gridDim.x * blockDim.x;
    size_t i = gid;
    
    while(i * 4 < matrixSize) {
        int4 data = matrix[i];
        
        // 使用位操作进行快速哈希
        atomicAdd(&localHist[(data.x & (HASH_BIN_COUNT-1))], 1);
        atomicAdd(&localHist[(data.y & (HASH_BIN_COUNT-1))], 1);
        atomicAdd(&localHist[(data.z & (HASH_BIN_COUNT-1))], 1);
        atomicAdd(&localHist[(data.w & (HASH_BIN_COUNT-1))], 1);
        
        i += stride;
    }
    __syncthreads();
    
    // 归约到全局内存
    for(int i = tid; i < HASH_BIN_COUNT; i += blockDim.x) {
        if(localHist[i] > 0) {
            atomicAdd(&hashHist[i], localHist[i]);
        }
    }
}

__global__ void computeFinalHistogram(const int4* __restrict__ matrix,
                                    const unsigned int* __restrict__ hashHist,
                                    unsigned int* __restrict__ finalHist,
                                    int* __restrict__ result,
                                    unsigned int* __restrict__ maxCount,
                                    const size_t matrixSize) {
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    __shared__ unsigned int localHist[256];
    __shared__ unsigned int sharedMaxCount;
    __shared__ int sharedMaxValue;
    
    if(tid == 0) {
        sharedMaxCount = 0;
        sharedMaxValue = -1;
    }
    
    // 只处理哈希直方图中计数较大的桶
    const unsigned int threshold = matrixSize / (VALUE_RANGE * 2);  // 启发式阈值
    
    const size_t stride = gridDim.x * blockDim.x;
    size_t i = gid;
    
    while(i * 4 < matrixSize) {
        int4 data = matrix[i];
        int values[4] = {data.x, data.y, data.z, data.w};
        
        #pragma unroll
        for(int j = 0; j < 4; j++) {
            int value = values[j];
            if(value < VALUE_RANGE && hashHist[value & (HASH_BIN_COUNT-1)] > threshold) {
                unsigned int count = atomicAdd(&finalHist[value], 1);
                if(count > sharedMaxCount) {
                    sharedMaxCount = count;
                    sharedMaxValue = value;
                }
            }
        }
        
        i += stride;
    }
    __syncthreads();
    
    // 更新全局最大值
    if(tid == 0 && sharedMaxCount > 0) {
        atomicMax(maxCount, sharedMaxCount);
        if(sharedMaxCount == *maxCount) {
            *result = sharedMaxValue;
        }
    }
}

// 保存矩阵到文件
void saveMatrixToFile(const int* data, size_t size, const char* filename) {
    FILE* fp = fopen(filename, "wb");
    if(!fp) {
        fprintf(stderr, "Failed to open file for writing\n");
        return;
    }
    
    fwrite(&size, sizeof(size_t), 1, fp);
    fwrite(data, sizeof(int), size, fp);
    
    fclose(fp);
}

// 生成测试数据
void generateTestData(int* data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, VALUE_RANGE-1);
    
    // 随机生成数据
    #pragma omp parallel for
    for(size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    
    // 确保有一个明确的最频繁元素
    size_t frequent_count = size / 4;
    int frequent_value = dis(gen);
    #pragma omp parallel for
    for(size_t i = 0; i < frequent_count; i++) {
        data[i] = frequent_value;
    }
}

// CPU验证
int findMostFrequentCPU(const int* matrix, size_t size) {
    std::unordered_map<int, size_t> freq;
    #pragma omp parallel
    {
        std::unordered_map<int, size_t> local_freq;
        #pragma omp for nowait
        for(size_t i = 0; i < size; i++) {
            local_freq[matrix[i]]++;
        }
        
        #pragma omp critical
        {
            for(const auto& pair : local_freq) {
                freq[pair.first] += pair.second;
            }
        }
    }
    
    size_t maxFreq = 0;
    int result = -1;
    for(const auto& pair : freq) {
        if(pair.second > maxFreq) {
            maxFreq = pair.second;
            result = pair.first;
        }
    }
    return result;
}

int main() {
    const size_t rows = 10000000;
    const size_t cols = 100;
    const size_t matrixSize = rows * cols;
    
    printf("Matrix size: %zu x %zu (%zu elements, %.2f GB)\n", 
           rows, cols, matrixSize, matrixSize * sizeof(int) / (1024.0 * 1024.0 * 1024.0));
    
    // 分配内存
    int* h_matrix;
    CHECK_CUDA_ERROR(cudaMallocHost(&h_matrix, matrixSize * sizeof(int)));  // 使用锁页内存
    
    printf("Generating test data...\n");
    generateTestData(h_matrix, matrixSize);
    
    printf("Saving matrix to file...\n");
    saveMatrixToFile(h_matrix, matrixSize, "test_matrix.bin");
    
    // 分配设备内存
    int4* d_matrix;
    unsigned int* d_hashHist;
    unsigned int* d_finalHist;
    int* d_result;
    unsigned int* d_maxCount;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrixSize * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_hashHist, HASH_BIN_COUNT * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_finalHist, VALUE_RANGE * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_maxCount, sizeof(unsigned int)));
    
    // 初始化设备内存
    CHECK_CUDA_ERROR(cudaMemset(d_hashHist, 0, HASH_BIN_COUNT * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMemset(d_finalHist, 0, VALUE_RANGE * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMemset(d_maxCount, 0, sizeof(unsigned int)));
    
    // 传输数据到设备
    printf("Copying data to device...\n");
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix, matrixSize * sizeof(int), cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    double memcpy_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // GPU计算
    printf("Running GPU version...\n");
    start = std::chrono::high_resolution_clock::now();
    
    computeHistogramOptimized<<<GRID_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<int4*>(d_matrix), d_hashHist, matrixSize);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    computeFinalHistogram<<<GRID_SIZE, BLOCK_SIZE>>>(
        reinterpret_cast<int4*>(d_matrix), d_hashHist, d_finalHist, d_result, d_maxCount, matrixSize);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    int gpu_result;
    CHECK_CUDA_ERROR(cudaMemcpy(&gpu_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    
    end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // CPU验证
    printf("Running CPU version for verification...\n");
    start = std::chrono::high_resolution_clock::now();
    int cpu_result = findMostFrequentCPU(h_matrix, matrixSize);
    end = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 输出结果
    printf("\nResults:\n");
    printf("GPU Result: %d (took %.2f ms)\n", gpu_result, gpu_time);
    printf("CPU Result: %d (took %.2f ms)\n", cpu_result, cpu_time);
    printf("Memory copy time: %.2f ms\n", memcpy_time);
    printf("Results match: %s\n", (gpu_result == cpu_result) ? "Yes" : "No");
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    
    // 清理内存
    CHECK_CUDA_ERROR(cudaFreeHost(h_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_hashHist));
    CHECK_CUDA_ERROR(cudaFree(d_finalHist));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    CHECK_CUDA_ERROR(cudaFree(d_maxCount));
    
    return 0;
}