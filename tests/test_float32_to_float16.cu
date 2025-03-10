#include "spdlog/spdlog.h"
#include <gtest/gtest.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <sys/types.h>

template <uint32_t size>
void float32_to_float16_simd(float const* src, uint16_t* dst) {
  static_assert(size % 8 == 0);
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    __m256 f32_vec = _mm256_loadu_ps(src + i);
    __m128i f16_vec =
        _mm256_cvtps_ph(f32_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), f16_vec);
  }
}

template <uint32_t size>
void float16_to_float32_simd(uint16_t const* src, float* dst) {
  static_assert(size % 8 == 0);
  size_t i = 0;
  for (; i + 8 <= size; i += 8) {
    __m128i f16_vec =
        _mm_loadu_si128(reinterpret_cast<__m128i const*>(src + i));
    __m256 f32_vec = _mm256_cvtph_ps(f16_vec);
    _mm256_storeu_ps(dst + i, f32_vec);
  }
}

struct ErrorStats {
  double max_abs_error = 0;
  double max_rel_error = 0;
  double avg_abs_error = 0;
  double avg_rel_error = 0;
  int count = 0;
};

ErrorStats analyze_errors(float const* orig, float const* reconstructed,
                          size_t len) {
  ErrorStats stats;
  double total_abs = 0;
  double total_rel = 0;

  for (size_t i = 0; i < len; ++i) {
    float const a = orig[i];
    float const b = reconstructed[i];

    if (std::isnan(a) || std::isinf(a) || std::isnan(b) || std::isinf(b))
      continue;

    double const abs_error = std::abs(a - b);
    double const rel_error = (a != 0) ? abs_error / std::abs(a) : 0;

    stats.max_abs_error = std::max(stats.max_abs_error, abs_error);
    stats.max_rel_error = std::max(stats.max_rel_error, rel_error);
    total_abs += abs_error;
    total_rel += rel_error;
    stats.count++;
  }

  if (stats.count > 0) {
    stats.avg_abs_error = total_abs / stats.count;
    stats.avg_rel_error = total_rel / stats.count;
  }
  return stats;
}

TEST(TestFloat16, simpletest) {
  constexpr uint32_t float_number = 32;
  float* h_raw_float = (float*)std::aligned_alloc(64, float_number);
  uint16_t* h_fp16 = (uint16_t*)std::aligned_alloc(64, float_number);
  float32_to_float16_simd<float_number>(h_raw_float, h_fp16);
  float* h_trans_float = (float*)std::aligned_alloc(64, float_number);
  float16_to_float32_simd<float_number>(h_fp16, h_raw_float);
  auto state = analyze_errors(h_raw_float, h_trans_float, float_number);
  SPDLOG_INFO("the state is {},{},{},{},{}", state.avg_abs_error,
              state.avg_rel_error, state.max_abs_error, state.max_rel_error,
              state.count);
}