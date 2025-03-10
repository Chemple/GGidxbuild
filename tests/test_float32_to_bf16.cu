#include "spdlog/spdlog.h"
#include <gtest/gtest.h>
#include <algorithm>
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
// #include <__clang_cuda_builtin_vars.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <immintrin.h>
#include <sys/types.h>

template <uint32_t num>
void convert_array_avx512(float const* src, uint16_t* dst) {
  constexpr size_t simd_width = 16;
  static_assert(num % 16 == 0);
  size_t i = 0;
  for (; i + simd_width <= num; i += simd_width) {
    __m512 f32_vec = _mm512_loadu_ps(src + i);
    __m512i u32_vec = _mm512_castps_si512(f32_vec);
    __m512i round_offset = _mm512_set1_epi32(0x7FFF);
    __m512i high_bit =
        _mm512_and_si512(_mm512_srli_epi32(u32_vec, 16), _mm512_set1_epi32(1));
    round_offset = _mm512_add_epi32(round_offset, high_bit);
    u32_vec = _mm512_add_epi32(u32_vec, round_offset);
    __m512i shifted = _mm512_srli_epi32(u32_vec, 16);
    __m256i bf16_packed = _mm512_cvtusepi32_epi16(shifted);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), bf16_packed);
  }
}

__global__ void bf16DotProductKernel(__nv_bfloat16 const* vec1,
                                     __nv_bfloat16 const* vec2,
                                     __nv_bfloat16* __restrict__ result,
                                     int n) {
  __nv_bfloat16 thread_sum = 0.0f;
  for (auto i = 0; i < n; i++) {
    __nv_bfloat16 v1 = vec1[i];
    __nv_bfloat16 v2 = vec2[i];
    thread_sum = __hadd(thread_sum, __hmul(v1, v2));
  }
  *result = thread_sum;
}

TEST(TestBf16, SimpleConvert) {
  constexpr uint32_t float_number = 16;
  std::random_device rd;
  std::mt19937 gen(rd());
  float min_val = -1.0f;
  float max_val = 1.0f;
  std::uniform_real_distribution<float> dist(min_val, max_val);
  float* src =
      static_cast<float*>(aligned_alloc(64, float_number * sizeof(float)));
  for (auto i = 0; i < float_number; i++) {
    src[i] = dist(gen);
  }
  uint16_t* dst = static_cast<uint16_t*>(
      aligned_alloc(64, float_number * sizeof(uint16_t)));

  auto sum = 0.0f;
  for (auto i = 0; i < float_number; i++) {
    sum += src[i] * src[i];
  }
  SPDLOG_INFO("the check res is {}", sum);

  convert_array_avx512<float_number>(src, dst);

  auto cpu_bf16_sum = 0.0f;
  for (auto i = 0; i < float_number; i++) {
    cpu_bf16_sum += __bfloat162float(dst[i]) * __bfloat162float(dst[i]);
  }
  SPDLOG_INFO("the cpu_bf_sum is {}", cpu_bf16_sum);

  uint16_t* d_vec = nullptr;
  uint16_t* d_res = nullptr;
  cudaMemcpy(d_vec, dst, float_number * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  bf16DotProductKernel<<<1, 1>>>((__nv_bfloat16 const*)d_vec,
                                 (__nv_bfloat16 const*)d_vec,
                                 (__nv_bfloat16*)d_res, float_number);

  __nv_bfloat16 h_check_res;
  cudaMemcpy(&h_check_res, d_res, sizeof(uint16_t), cudaMemcpyDeviceToHost);
  float fp32 = __bfloat162float(h_check_res);
  SPDLOG_INFO("the res is {}", fp32);
}
