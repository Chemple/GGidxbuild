#include "Gsearcher.cuh"
#include "spdlog/spdlog.h"
#include <gtest/gtest.h>
#include <cstdint>
#include <omp.h>

namespace Gbuilder {
namespace Gpu {
TEST(TestSearch, SizeTest) {
  SPDLOG_INFO("the size is {}",
              sizeof(search_warp_state<uint32_t, float, 128, 128, 16, 64>));
}
}  // namespace Gpu
}  // namespace Gbuilder
