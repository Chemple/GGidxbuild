#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <chrono>
#include <unordered_map>
#include <algorithm>

// A10 优化参数
#define BLOCK_SIZE 1024
#define NUM_SMS 72
#define GRID_SIZE (NUM_SMS * 32)
#define HIST_BINS 1024
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)
#define ELEMENTS_PER_THREAD 16
#define SHARED_HIST_SIZE 256  // 减小共享内存使用量


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

// Warp级别的归约函数
__device__ unsigned int warpReduceSum(unsigned int val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// 计算直方图的kernel
__global__ void computeHistogramOptimized(const int* __restrict__ matrix,
                                        unsigned int* __restrict__ globalHist,
                                        const size_t matrixSize) {
    const uint4* matrix4 = reinterpret_cast<const uint4*>(matrix);
    
    // 减小共享内存大小，每次处理部分直方图
    __shared__ unsigned int sharedHist[WARPS_PER_BLOCK][SHARED_HIST_SIZE];
    
    const unsigned int warpId = threadIdx.x >> 5;
    const unsigned int laneId = threadIdx.x & 31;
    
    // 分多次处理直方图
    for(int histOffset = 0; histOffset < HIST_BINS; histOffset += SHARED_HIST_SIZE) {
        // 初始化共享内存
        #pragma unroll
        for(int i = threadIdx.x; i < SHARED_HIST_SIZE * WARPS_PER_BLOCK; i += BLOCK_SIZE) {
            reinterpret_cast<unsigned int*>(sharedHist)[i] = 0;
        }
        __syncthreads();
        
        size_t startIdx = (blockIdx.x * BLOCK_SIZE + threadIdx.x) * ELEMENTS_PER_THREAD;
        const size_t stride = BLOCK_SIZE * gridDim.x * ELEMENTS_PER_THREAD;
        
        // 局部计数器
        unsigned int localBins[SHARED_HIST_SIZE / 32] = {0};
        
        // 处理数据
        for(size_t i = startIdx; i < matrixSize; i += stride) {
            if(i + 4 <= matrixSize) {
                uint4 data = matrix4[i >> 2];
                unsigned int values[4] = {data.x, data.y, data.z, data.w};
                
                #pragma unroll
                for(int j = 0; j < 4; j++) {
                    unsigned int val = values[j];
                    // 检查值是否在当前处理的范围内
                    if(val >= histOffset && val < histOffset + SHARED_HIST_SIZE) {
                        unsigned int localIdx = val - histOffset;
                        localBins[localIdx / 32]++;
                    }
                }
            }
        }
        
        // Warp归约
        #pragma unroll
        for(int i = 0; i < SHARED_HIST_SIZE / 32; i++) {
            unsigned int sum = warpReduceSum(localBins[i]);
            if(laneId == 0) {
                atomicAdd(&sharedHist[warpId][i * 32], sum);
            }
        }
        __syncthreads();
        
        // 更新全局直方图
        for(int i = threadIdx.x; i < SHARED_HIST_SIZE; i += BLOCK_SIZE) {
            unsigned int sum = 0;
            #pragma unroll
            for(int w = 0; w < WARPS_PER_BLOCK; w++) {
                sum += sharedHist[w][i];
            }
            if(sum > 0) {
                atomicAdd(&globalHist[histOffset + i], sum);
            }
        }
        __syncthreads();
    }
}

__global__ void findMaxFrequencyOptimized(const unsigned int* __restrict__ histogram,
                                        int* __restrict__ result,
                                        unsigned int* __restrict__ maxCount) {
    __shared__ unsigned int sharedMaxCount[WARPS_PER_BLOCK];
    __shared__ int sharedMaxValue[WARPS_PER_BLOCK];
    
    const unsigned int warpId = threadIdx.x >> 5;
    const unsigned int laneId = threadIdx.x & 31;
    
    unsigned int localMaxCount = 0;
    int localMaxValue = -1;
    
    for(int i = threadIdx.x + blockIdx.x * BLOCK_SIZE; i < HIST_BINS; i += BLOCK_SIZE * gridDim.x) {
        unsigned int count = histogram[i];
        if(count > localMaxCount) {
            localMaxCount = count;
            localMaxValue = i;
        }
    }
    
    #pragma unroll
    for(int offset = 16; offset > 0; offset >>= 1) {
        unsigned int tmpCount = __shfl_down_sync(0xffffffff, localMaxCount, offset);
        int tmpValue = __shfl_down_sync(0xffffffff, localMaxValue, offset);
        if(tmpCount > localMaxCount) {
            localMaxCount = tmpCount;
            localMaxValue = tmpValue;
        }
    }
    
    if(laneId == 0) {
        sharedMaxCount[warpId] = localMaxCount;
        sharedMaxValue[warpId] = localMaxValue;
    }
    __syncthreads();
    
    if(warpId == 0 && laneId < WARPS_PER_BLOCK) {
        localMaxCount = sharedMaxCount[laneId];
        localMaxValue = sharedMaxValue[laneId];
        
        #pragma unroll
        for(int offset = WARPS_PER_BLOCK/2; offset > 0; offset >>= 1) {
            unsigned int tmpCount = __shfl_down_sync(0xffffffff, localMaxCount, offset);
            int tmpValue = __shfl_down_sync(0xffffffff, localMaxValue, offset);
            if(tmpCount > localMaxCount) {
                localMaxCount = tmpCount;
                localMaxValue = tmpValue;
            }
        }
        
        if(laneId == 0) {
            atomicMax(maxCount, localMaxCount);
            if(localMaxCount == *maxCount) {
                *result = localMaxValue;
            }
        }
    }
}

// GPU版本查找最频繁元素
int findMostFrequentGPU(const int* d_matrix, size_t matrixSize) {
    unsigned int *d_histogram;
    int *d_result;
    unsigned int *d_maxCount;
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_histogram, HIST_BINS * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_result, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_maxCount, sizeof(unsigned int)));
    
    CHECK_CUDA_ERROR(cudaMemset(d_histogram, 0, HIST_BINS * sizeof(unsigned int)));
    CHECK_CUDA_ERROR(cudaMemset(d_maxCount, 0, sizeof(unsigned int)));
    
    computeHistogramOptimized<<<GRID_SIZE, BLOCK_SIZE>>>(d_matrix, d_histogram, matrixSize);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    findMaxFrequencyOptimized<<<NUM_SMS, BLOCK_SIZE>>>(d_histogram, d_result, d_maxCount);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    int result;
    CHECK_CUDA_ERROR(cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost));
    
    CHECK_CUDA_ERROR(cudaFree(d_histogram));
    CHECK_CUDA_ERROR(cudaFree(d_result));
    CHECK_CUDA_ERROR(cudaFree(d_maxCount));
    
    return result;
}

// CPU版本查找最频繁元素（用于验证）
int findMostFrequentCPU(const int* matrix, size_t size) {
    std::unordered_map<int, int> freq;
    for(size_t i = 0; i < size; i++) {
        freq[matrix[i]]++;
    }
    
    int maxFreq = 0;
    int result = -1;
    for(const auto& pair : freq) {
        if(pair.second > maxFreq) {
            maxFreq = pair.second;
            result = pair.first;
        }
    }
    return result;
}

// 生成测试数据
void generateTestData(int* data, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, HIST_BINS-1);
    
    for(size_t i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
    
    // 确保有一个明确的最频繁元素
    size_t frequent_count = size / 4;  // 25%的数据设置为同一个值
    for(size_t i = 0; i < frequent_count; i++) {
        data[i] = HIST_BINS/2;
    }
}

int main() {
    // 设置矩阵大小
    const size_t rows = 10000000;  // 10M
    const size_t cols = 100;
    const size_t matrixSize = rows * cols;
    
    printf("Matrix size: %zu x %zu (%zu elements, %.2f GB)\n", 
           rows, cols, matrixSize, matrixSize * sizeof(int) / (1024.0 * 1024.0 * 1024.0));
    
    // 分配主机内存
    int* h_matrix = (int*)malloc(matrixSize * sizeof(int));
    if(!h_matrix) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return -1;
    }
    
    // 生成测试数据
    printf("Generating test data...\n");
    generateTestData(h_matrix, matrixSize);
    
    // 分配设备内存
    int* d_matrix;
    CHECK_CUDA_ERROR(cudaMalloc(&d_matrix, matrixSize * sizeof(int)));
    
    // 传输数据到设备
    printf("Copying data to device...\n");
    auto start = std::chrono::high_resolution_clock::now();
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, h_matrix, matrixSize * sizeof(int), cudaMemcpyHostToDevice));
    auto end = std::chrono::high_resolution_clock::now();
    double memcpy_time = std::chrono::duration<double, std::milli>(end - start).count();
    
    // GPU计算
    printf("Running GPU version...\n");
    start = std::chrono::high_resolution_clock::now();
    int gpu_result = findMostFrequentGPU(d_matrix, matrixSize);
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
    free(h_matrix);
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    
    return 0;
}