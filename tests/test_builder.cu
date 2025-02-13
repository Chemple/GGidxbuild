#include <gtest/gtest.h>
#include <cstdio>
#include "Gbuilder.h"

void __global__ hello_world() {
    printf("hello world!\n");
}

TEST(test, simple_test) {
    hello_world<<<16, 64>>>();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
    cudaDeviceSynchronize();
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
    }
}
