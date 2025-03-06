#include "Gbuilder.cuh"
#include "Gsearcher.cuh"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <gtest/gtest.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <sys/types.h>
#include <x86intrin.h>

namespace Gbuilder {
namespace Gpu {

float compare(float const* a, float const* b, uint32_t size) {
  __m512 msum0 = _mm512_setzero_ps();

  while (size >= 16) {
    __m512 mx = _mm512_loadu_ps(a);
    __m512 my = _mm512_loadu_ps(b);
    a += 16;
    b += 16;
    msum0 = _mm512_fmadd_ps(mx, my, msum0);  // fma: mx * my + msum0
    size -= 16;
  }

  __m256 msum1 =
      _mm512_extractf32x8_ps(msum0, 1) + _mm512_extractf32x8_ps(msum0, 0);

  if (size >= 8) {
    __m256 mx = _mm256_loadu_ps(a);
    __m256 my = _mm256_loadu_ps(b);
    a += 8;
    b += 8;
    msum1 = _mm256_fmadd_ps(mx, my, msum1);
    size -= 8;
  }

  __m128 msum2 =
      _mm256_extractf128_ps(msum1, 1) + _mm256_extractf128_ps(msum1, 0);

  if (size >= 4) {
    __m128 mx = _mm_loadu_ps(a);
    __m128 my = _mm_loadu_ps(b);
    a += 4;
    b += 4;
    msum2 = _mm_fmadd_ps(mx, my, msum2);
    size -= 4;
  }

  if (size > 0) {
    __m128i mask = _mm_set_epi32(size > 2 ? -1 : 0, size > 1 ? -1 : 0,
                                 size > 0 ? -1 : 0, 0);
    __m128 mx = _mm_maskload_ps(a, mask);
    __m128 my = _mm_maskload_ps(b, mask);
    msum2 = _mm_fmadd_ps(mx, my, msum2);
  }

  msum2 = _mm_hadd_ps(msum2, msum2);
  msum2 = _mm_hadd_ps(msum2, msum2);
  return -1.0f * _mm_cvtss_f32(msum2);
}

struct SimpleNeighbor {
  uint32_t id;
  float distance;
  SimpleNeighbor() = default;
  SimpleNeighbor(uint32_t id, float distance) : id{id}, distance{distance} {}
  inline bool operator<(SimpleNeighbor const& other) const {
    return distance < other.distance ||
           (distance == other.distance && id < other.id);
  }
  inline bool operator>(SimpleNeighbor const& other) const {
    return distance > other.distance ||
           (distance == other.distance && id > other.id);
  }
  friend void swap(SimpleNeighbor& a, SimpleNeighbor& b) {
    std::swap(a.id, b.id);
    std::swap(a.distance, b.distance);
  }
};

void RNGPrune(uint32_t M, std::vector<SimpleNeighbor>& full_set,
              uint32_t target_id, std::vector<uint32_t>& pruned_list,
              float const* data, bool is_strict, uint32_t num_base,
              uint32_t const dimension) {
  uint32_t M_ctr = M;
  uint32_t start = 0;

  while (start < full_set.size() && full_set[start].id == target_id) start++;
  if (start == full_set.size()) {
    return;
  }

  std::vector<uint32_t> result;
  result.reserve(M_ctr);
  result.emplace_back(full_set[start].id);

  while (result.size() < M_ctr && (++start) < full_set.size()) {
    auto& p = full_set[start];
    bool occlude = false;
    for (size_t i = 0; i < result.size(); ++i) {
      if (p.id == result[i]) {
        occlude = true;
        break;
      }
      float djk = compare(data + dimension * p.id, data + dimension * result[i],
                          dimension);

      if (djk < p.distance) {
        occlude = true;
        break;
      }
    }
    if (!occlude) {
      if (p.id != target_id &&
          std::find(result.begin(), result.end(), p.id) == result.end()) {
        result.emplace_back(p.id);
      }
    }
  }

  start = 0;  // double check
  while (start < full_set.size() && full_set[start].id == target_id) start++;
  while (result.size() < M_ctr && (++start) < full_set.size()) {
    auto& p = full_set[start];
    bool occlude = false;
    for (size_t i = 0; i < result.size(); ++i) {
      if (p.id == result[i]) {
        occlude = true;
        break;
      }
      float djk = compare(data + dimension * p.id, data + dimension * result[i],
                          dimension);
      if (djk < p.distance) {
        occlude = true;
        break;
      }
    }
    if (!occlude) {
      if (p.id != target_id &&
          std::find(result.begin(), result.end(), p.id) == result.end()) {
        result.emplace_back(p.id);
      }
    }
  }

  if (!is_strict) {
    for (size_t i = 1; i < full_set.size() && result.size() < M_ctr; ++i) {
      if (std::find(result.begin(), result.end(), full_set[i].id) ==
              result.end() &&
          full_set[i].id < num_base) {
        if (full_set[i].id != target_id) {
          result.emplace_back(full_set[i].id);
        }
      }
    }
  }

  pruned_list = result;
}

std::vector<std::vector<uint32_t>> MatchNN(
    uint32_t num_base, uint32_t num_query, uint32_t max_degree, uint32_t N_ctr,
    uint32_t M_nn, uint32_t const* query_knn, uint32_t& ep, float const* data,
    uint32_t const dimension, int thread_limit) {
  int original_threads = omp_get_max_threads();
  omp_set_num_threads(thread_limit);

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<uint32_t> match(num_query, num_base + 1);
  std::vector<uint32_t> frequency(num_base, 0);
  std::vector<bool> vis(num_base, false);

  for (uint32_t it_nq = 0; it_nq < num_query; ++it_nq) {
    uint32_t base = num_base + 1;
    bool ifmatch = false;
    for (uint32_t j = 0; j < N_ctr; j++) {
      uint32_t nn = query_knn[it_nq * N_ctr + j];
      if (nn >= num_base || ifmatch) break;
      ++frequency[nn];
      if (vis[nn]) {
        continue;
      } else {
        vis[nn] = true;
        ifmatch = true;
        base = nn;
      }
    }
    match[it_nq] = base;
  }

  auto after_match = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> match_duration = after_match - start_time;
  SPDLOG_INFO("MatchNN: Initial matching completed in {:.2f} seconds",
              match_duration.count());

  std::vector<uint32_t> order(num_base);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](uint32_t i, uint32_t j) {
    return frequency[i] > frequency[j];
  });
  assert(order[0] < num_base);
  ep = order[0];

  uint32_t const MAX_DEGREE = 128 - M_nn;
  std::vector<std::vector<uint32_t>> match_graph(num_base);
  std::vector<std::vector<uint32_t>> tmp_graph(
      num_base, std::vector<uint32_t>(MAX_DEGREE, UINT32_MAX));
  std::vector<std::atomic<uint32_t>> degrees(num_base);
  std::vector<bool> is_full(num_base);

  auto before_parallel = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 200)
  for (uint32_t it_nq = 0; it_nq < num_query; ++it_nq) {
    uint32_t cur_main_id = match[it_nq];
    if (cur_main_id >= num_base) {
      continue;
    }
    std::set<uint32_t> vis;
    std::vector<SimpleNeighbor> full_set;
    vis.insert(cur_main_id);
    for (uint32_t j = 0; j < N_ctr; j++) {
      uint32_t base_id = query_knn[it_nq * N_ctr + j];
      if (base_id >= num_base) break;
      if (vis.find(base_id) != vis.end()) continue;
      vis.insert(base_id);
      float distance = compare(data + dimension * base_id,
                               data + dimension * cur_main_id, dimension);
      full_set.emplace_back(SimpleNeighbor(base_id, distance));
    }
    std::sort(full_set.begin(), full_set.end());
    std::vector<uint32_t> pruned_list;
    RNGPrune(M_nn, full_set, cur_main_id, pruned_list, data, false, num_base,
             dimension);
    for (uint32_t des_node : pruned_list) {
      if (is_full[des_node]) continue;
      uint32_t cur_degree =
          degrees[des_node].fetch_add(1, std::memory_order_relaxed);
      if (cur_degree < MAX_DEGREE) {
        tmp_graph[des_node][cur_degree] = cur_main_id;
      } else if (cur_degree == MAX_DEGREE) {
        is_full[des_node] = true;
      }
    }
    match_graph[cur_main_id] = pruned_list;
  }

  auto after_parallel = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> parallel_duration =
      after_parallel - before_parallel;
  SPDLOG_INFO("MatchNN: Initial graph building completed in {:.2f} seconds",
              parallel_duration.count());

  auto before_final_prune = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 200)
  for (uint32_t it_nb = 0; it_nb < num_base; ++it_nb) {
    std::vector<uint32_t> const& vec1 = match_graph[it_nb];
    std::vector<uint32_t> const& vec2 = tmp_graph[it_nb];

    uint32_t actual_size = degrees[it_nb].load(std::memory_order_relaxed);
    actual_size = std::min(actual_size, MAX_DEGREE);
    std::unordered_set<uint32_t> mergedSet;
    mergedSet.reserve(vec1.size() + actual_size);
    mergedSet.insert(vec1.begin(), vec1.end());
    mergedSet.insert(vec2.begin(), vec2.begin() + actual_size);

    match_graph[it_nb] =
        std::vector<uint32_t>(mergedSet.begin(), mergedSet.end());
    if (match_graph[it_nb].size() > M_nn) {
      std::vector<SimpleNeighbor> full_set;
      for (uint32_t& base_id : match_graph[it_nb]) {
        float distance = compare(data + dimension * base_id,
                                 data + dimension * it_nb, dimension);

        full_set.emplace_back(SimpleNeighbor(base_id, distance));
      }
      std::sort(full_set.begin(), full_set.end());
      std::vector<uint32_t> pruned_list;
      RNGPrune(M_nn, full_set, it_nb, pruned_list, data, true, num_base,
               dimension);
      match_graph[it_nb] = std::move(pruned_list);
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> final_duration = end_time - before_final_prune;
  std::chrono::duration<double> total_duration = end_time - start_time;

  SPDLOG_INFO("MatchNN: Final pruning completed in {:.2f} seconds",
              final_duration.count());
  SPDLOG_INFO("MatchNN: Total execution time: {:.2f} seconds",
              total_duration.count());

  omp_set_num_threads(original_threads);
  
  return match_graph;
}

std::vector<std::vector<uint32_t>> FusionFinal(
    uint32_t num_base, uint32_t M_supply, uint32_t M_link, uint32_t M_final,
    float const* data, std::vector<uint32_t>& supply_graph_,
    std::vector<std::vector<uint32_t>>& bipartite_graph_,
    std::vector<uint32_t>& link_graph_, uint32_t const dimension, int thread_limit) {
  int original_threads = omp_get_max_threads();
  omp_set_num_threads(thread_limit);
              
  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<std::vector<uint32_t>> final_graph_(num_base);

#pragma omp parallel for schedule(dynamic, 100)
  for (uint32_t it_nb = 0; it_nb < num_base; ++it_nb) {
    std::vector<uint32_t> final_set;
    std::vector<SimpleNeighbor> full_set;
    std::set<uint32_t> vis;
    final_set.reserve(M_final);
    full_set.reserve(70);

    // 处理 supply graph (top1 projection)
    for (uint32_t j = 0; j < M_supply; j++) {
      uint32_t base_id = supply_graph_[it_nb * M_supply + j];
      if (base_id >= num_base) continue;
      if (vis.find(base_id) != vis.end() || base_id == it_nb ||
          base_id >= num_base)
        continue;
      vis.insert(base_id);
      float distance = compare(data + dimension * it_nb,
                               data + dimension * base_id, dimension);

      full_set.push_back(SimpleNeighbor(base_id, distance));
    }
    std::sort(full_set.begin(), full_set.end());
    RNGPrune(M_supply, full_set, it_nb, final_set, data, false, num_base,
             dimension);
    final_graph_[it_nb] = final_set;

    // 处理 bipartite graph (topnn projection)
    for (uint32_t& base_id : bipartite_graph_[it_nb]) {
      if (vis.find(base_id) != vis.end() || base_id == it_nb ||
          base_id >= num_base)
        continue;
      vis.insert(base_id);
      float distance = compare(data + dimension * it_nb,
                               data + dimension * base_id, dimension);
      full_set.push_back(SimpleNeighbor(base_id, distance));
    }

    // 处理 link graph (从GPU获取的第二阶段结果)
    for (uint32_t j = 0; j < M_link; j++) {
      uint32_t base_id = link_graph_[it_nb * M_link + j];
      if (base_id >= num_base) continue;
      if (vis.find(base_id) != vis.end()) continue;  // 避免重复
      vis.insert(base_id);
      float distance = compare(data + dimension * it_nb,
                               data + dimension * base_id, dimension);
      full_set.push_back(SimpleNeighbor(base_id, distance));
    }

    // 合并和修剪
    std::sort(full_set.begin(), full_set.end());
    std::vector<uint32_t> pruned_list;
    RNGPrune(M_final, full_set, it_nb, pruned_list, data, true, num_base,
             dimension);
    std::unordered_set<uint32_t> final_set_lookup(final_graph_[it_nb].begin(),
                                                  final_graph_[it_nb].end());
    std::vector<uint32_t> ok_insert;
    ok_insert.reserve(M_final);
    size_t const remaining_slots = M_final - final_graph_[it_nb].size();
    for (uint32_t candidate : pruned_list) {
      if (ok_insert.size() >= remaining_slots) break;
      if (final_set_lookup.find(candidate) == final_set_lookup.end()) {
        ok_insert.push_back(candidate);
      }
    }
    final_graph_[it_nb].insert(final_graph_[it_nb].end(), ok_insert.begin(),
                               ok_insert.end());
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = end_time - start_time;

  SPDLOG_INFO("FusionFinal: Completed in {:.2f} seconds",
              total_duration.count());

  omp_set_num_threads(original_threads);
  
  return final_graph_;
}

void SaveGraph(std::vector<std::vector<uint32_t>> const& graph,
               std::string const& file_path, uint32_t ep, uint32_t base_num) {
  auto start_time = std::chrono::high_resolution_clock::now();

  std::ofstream out(file_path, std::ios::binary | std::ios::out);
  if (!out.is_open()) {
    throw std::runtime_error("cannot open file");
  }

  // 写入元数据
  out.write((char*)&ep, sizeof(uint32_t));
  out.write((char*)&base_num, sizeof(uint32_t));

  // 写入图数据
  for (uint32_t i = 0; i < base_num; ++i) {
    uint32_t nbr_size = graph[i].size();
    out.write((char*)&nbr_size, sizeof(uint32_t));
    out.write((char*)graph[i].data(), sizeof(uint32_t) * nbr_size);
  }

  out.close();

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> total_duration = end_time - start_time;

  SPDLOG_INFO("SaveGraph: Completed in {:.2f} seconds", total_duration.count());
}

TEST(GpuConstructionTime, TestEnd2EndCPUGPU) {

  cudaDeviceReset();

  // 配置参数
  constexpr uint32_t gt_degree = 128;
  constexpr uint32_t match_degree = 128;
  constexpr uint32_t base_num = 10000000;
  constexpr uint32_t dim = 200;
  constexpr uint32_t tomb = 0XFFFFFFFF;
  constexpr uint32_t reverse_edge_num = 111;
  constexpr uint32_t pruned_edge_num = 15;
  constexpr uint32_t top1_projection_degree = 16;
  constexpr uint32_t num_element_pr_list =
      reverse_edge_num + pruned_edge_num + 2;
  uint32_t ep = 0;  // ep变量

  // search相关参数
  constexpr uint32_t query_num = 10000000;

  constexpr uint32_t first_round_search_grid_size = 144;
  constexpr uint32_t first_round_search_block_size = 1024;
  constexpr uint32_t Km = 32;
  constexpr uint32_t Kp = 2;
  constexpr uint32_t Kd = 16;
  constexpr uint32_t topk = Km + Kp * Kd;
  constexpr uint32_t reset_iter = 15;
  constexpr uint32_t hashtable_size = 1 << 12;
  constexpr uint32_t shared_memory_size =
      (first_round_search_block_size / 32) *
      sizeof(search_warp_state_v0<uint32_t, float, dim, Km, Kp, Kd>);

  constexpr uint32_t global_warp_num =
      first_round_search_grid_size * first_round_search_block_size / 32;

  constexpr uint32_t first_round_pruned_edge_num = 55;
  constexpr uint32_t first_round_reverse_edge_num = 71;

  constexpr uint32_t second_round_pruned_edge_num = 55;
  constexpr uint32_t second_round_reverse_edge_num = 71;

  constexpr uint32_t second_search_query_num = 10000000;

  constexpr uint32_t second_round_search_grid_size = 144;
  constexpr uint32_t second_round_search_block_size = 1024;
  constexpr uint32_t second_Km = 32;
  constexpr uint32_t second_Kp = 2;
  constexpr uint32_t second_Kd = 16;
  constexpr uint32_t second_topk = Km + Kp * Kd;
  constexpr uint32_t second_reset_iter = 15;
  constexpr uint32_t second_hashtable_size = 1 << 12;
  constexpr uint32_t second_shared_memory_size =
      (second_round_search_block_size / 32) *
      sizeof(search_warp_state_v0<uint32_t, float, dim, second_Km, second_Kp,
                                  second_Kd>);

  constexpr uint32_t second_global_warp_num =
      second_round_search_grid_size * second_round_search_block_size / 32;

  constexpr uint32_t final_degree = 55;

  auto basedata_file_name =
      "/home/shiwen/project/GGidxbuild/data/10M_200/vector.fbin";
  auto gt_file = "/home/shiwen/project/GGidxbuild/data/gt.train.10M.128.gpu";

  // CPU线程数限制
  int max_threads = omp_get_max_threads();
  int cpu_thread_limit = std::min(max_threads, 64);  // 限制最大线程数为64

  // 创建CUDA流和事件
  cudaStream_t compute_stream, copy_stream;
  cudaEvent_t data_ready, gpu_phase1_done, gpu_phase2_done;

  cudaStreamCreate(&compute_stream);
  cudaStreamCreate(&copy_stream);
  cudaEventCreate(&data_ready);
  cudaEventCreate(&gpu_phase1_done);
  cudaEventCreate(&gpu_phase2_done);

  // 总体时间测量事件
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // 读取数据
  auto data_load_start = std::chrono::high_resolution_clock::now();

  auto h_gt_data = std::vector<uint32_t>(base_num * gt_degree);
  auto h_base_data = std::vector<float>(base_num * dim);

  read_vec_from_file(h_gt_data, gt_file);
  read_vec_from_file(h_base_data, basedata_file_name);

  auto data_load_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> data_load_duration =
      data_load_end - data_load_start;
  SPDLOG_INFO("Data loading completed in {:.2f} seconds",
              data_load_duration.count());

  // 预先声明需要的变量
  auto h_projection = std::vector<uint32_t>(base_num * top1_projection_degree);
  auto h_second_round_search_merge =
      std::vector<uint32_t>(base_num * final_degree);
  std::vector<std::vector<uint32_t>> final_graph;

  // 分配GPU内存
  auto gpu_alloc_start = std::chrono::high_resolution_clock::now();

  float* d_base_data = nullptr;
  uint32_t* d_space_128_xx = nullptr;
  uint32_t* d_space_128_yy = nullptr;
  uint32_t* d_top1_projection = nullptr;
  HashTable<uint32_t, 1 << 12>* d_hashtables = nullptr;

  cudaMalloc(&d_base_data, base_num * dim * sizeof(float));
  cudaMalloc(&d_space_128_xx, base_num * gt_degree * sizeof(uint32_t));
  cudaMalloc(
      &d_space_128_yy,
      base_num *
          sizeof(
              pr_neighbor_list<uint32_t, reverse_edge_num, pruned_edge_num>));
  cudaMalloc(&d_top1_projection,
             base_num * top1_projection_degree * sizeof(uint32_t));
  cudaMalloc(&d_hashtables,
             global_warp_num * hashtable_size * sizeof(uint32_t));

  auto gpu_alloc_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu_alloc_duration =
      gpu_alloc_end - gpu_alloc_start;
  SPDLOG_INFO("GPU memory allocation completed in {:.2f} seconds",
              gpu_alloc_duration.count());

  // 开始计时
  cudaEventRecord(start, compute_stream);

  // 1. 异步数据传输到GPU (在copy_stream上)
  auto data_transfer_start = std::chrono::high_resolution_clock::now();

  cudaMemcpyAsync(d_base_data, h_base_data.data(),
                  base_num * dim * sizeof(float), cudaMemcpyHostToDevice,
                  copy_stream);

  cudaMemcpyAsync(d_space_128_xx, h_gt_data.data(),
                  base_num * gt_degree * sizeof(uint32_t),
                  cudaMemcpyHostToDevice, copy_stream);

  auto h_init_match = std::vector<uint32_t>(base_num * match_degree, tomb);

  cudaMemcpyAsync(d_space_128_yy, h_init_match.data(),
                  base_num * match_degree * sizeof(uint32_t),
                  cudaMemcpyHostToDevice, copy_stream);

  // 标记数据传输完成
  cudaEventRecord(data_ready, copy_stream);
 
  auto test_start_time = std::chrono::high_resolution_clock::now();
  // 2. GPU计算第一阶段 - top1 projection
  cudaStreamWaitEvent(compute_stream, data_ready, 0);

  // GPU Kernels - 第一阶段 (top1 projection)
  constexpr uint32_t match_grid_size = 144;
  constexpr uint32_t match_block_size = 512;

  match_top1_kernel_v0<match_grid_size, match_block_size, base_num, gt_degree,
                       match_degree, 127, tomb, float, uint32_t>
      <<<match_grid_size, match_block_size, 0, compute_stream>>>(
          d_space_128_xx, d_space_128_yy);

  constexpr uint32_t grid_size = 144;
  constexpr uint32_t block_size = 256;

  init_pr_lists<grid_size, block_size, base_num, match_degree, pruned_edge_num,
                reverse_edge_num, tomb, dim, true, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          (pr_neighbor_list<uint32_t, reverse_edge_num, pruned_edge_num>*)
              d_space_128_xx);

  fusion_prune_reverse_kernel_v0<grid_size, block_size, base_num, match_degree,
                                 pruned_edge_num, reverse_edge_num, tomb, dim,
                                 true, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          d_base_data, d_space_128_yy,
          (pr_neighbor_list<uint32_t, reverse_edge_num, pruned_edge_num>*)
              d_space_128_xx);

  constexpr uint32_t projection_grid_size = 144;
  constexpr uint32_t projection_block_size = 256;

  fusion_merge_sort_prune_kernel<projection_grid_size, projection_block_size,
                                 base_num, pruned_edge_num, reverse_edge_num,
                                 top1_projection_degree, tomb, dim, true, float,
                                 uint32_t>
      <<<projection_grid_size, projection_block_size, 0, compute_stream>>>(
          d_base_data,
          (pr_neighbor_list<uint32_t, reverse_edge_num, pruned_edge_num>*)
              d_space_128_xx,
          d_top1_projection);

  // 创建计算完成事件等待copy_stream
  cudaEvent_t phase1_compute_done;
  cudaEventCreate(&phase1_compute_done);
  cudaEventRecord(phase1_compute_done, compute_stream);

  // 确保数据拷贝前计算已完成
  cudaStreamWaitEvent(copy_stream, phase1_compute_done, 0);

  // 在计算完成后异步拷贝投影数据供CPU使用
  cudaMemcpyAsync(h_projection.data(), d_top1_projection,
                  base_num * top1_projection_degree * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, copy_stream);

  cudaEventDestroy(phase1_compute_done);

  // 标记GPU第一阶段完成 - 这个事件用于CPU线程同步
  cudaEventRecord(gpu_phase1_done, copy_stream);  // 使用copy_stream确保数据拷贝完成

  // 3. 等待GPU第一阶段完成，然后CPU计算topnn projection
  auto gpu1_wait_start = std::chrono::high_resolution_clock::now();

  cudaError_t phase1_result = cudaEventSynchronize(gpu_phase1_done);

  auto gpu1_wait_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu1_wait_duration =
      gpu1_wait_end - gpu1_wait_start;
  SPDLOG_INFO("CPU: GPU phase 1 wait completed in {:.2f} seconds",
              gpu1_wait_duration.count());

  // CPU计算topnn projection (与GPU第二阶段并行)
  auto match_start = std::chrono::high_resolution_clock::now();

  auto topnn_projection_graph =
      MatchNN(base_num, query_num, match_degree, gt_degree, 40,
              const_cast<uint32_t*>(h_gt_data.data()), ep,
              const_cast<float*>(h_base_data.data()), dim, cpu_thread_limit);

  auto match_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> match_duration = match_end - match_start;
  SPDLOG_INFO("CPU: MatchNN (topnn projection) completed in {:.2f} seconds",
              match_duration.count());

  // 4. GPU计算第二阶段 - 两轮link (并行于CPU的topnn projection计算)

  link_process_v0<first_round_search_grid_size, first_round_search_block_size,
                  base_num, query_num, dim, top1_projection_degree,
                  shared_memory_size, Km, Kp, Kd, topk, 0XFFFFFFFF,
                  hashtable_size, reset_iter, uint32_t, float>
      <<<first_round_search_grid_size, first_round_search_block_size,
         shared_memory_size, compute_stream>>>(
          d_base_data, d_hashtables, d_top1_projection, d_space_128_xx);

  init_pr_lists<grid_size, block_size, base_num, match_degree,
                first_round_pruned_edge_num, first_round_reverse_edge_num, tomb,
                dim, true, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          (pr_neighbor_list<uint32_t, first_round_reverse_edge_num,
                            first_round_pruned_edge_num>*)d_space_128_yy);

  fusion_prune_reverse_kernel_v0<
      grid_size, block_size, base_num, topk, first_round_pruned_edge_num,
      first_round_reverse_edge_num, tomb, dim, true, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          d_base_data, d_space_128_xx,
          (pr_neighbor_list<uint32_t, first_round_reverse_edge_num,
                            first_round_pruned_edge_num>*)d_space_128_yy);

  fusion_merge_sort_prune_kernel<grid_size, block_size, base_num,
                                 first_round_pruned_edge_num,
                                 first_round_reverse_edge_num, final_degree,
                                 tomb, dim, true, float, uint32_t>
      <<<projection_grid_size, projection_block_size, 0, compute_stream>>>(
          d_base_data,
          (pr_neighbor_list<uint32_t, first_round_reverse_edge_num,
                            first_round_pruned_edge_num>*)d_space_128_yy,
          d_space_128_xx);

  link_process_v0<second_round_search_grid_size, second_round_search_block_size,
                  base_num, query_num, dim, final_degree,
                  second_shared_memory_size, second_Km, second_Kp, second_Kd,
                  second_topk, 0XFFFFFFFF, hashtable_size, reset_iter, uint32_t,
                  float>
      <<<first_round_search_grid_size, first_round_search_block_size,
         shared_memory_size, compute_stream>>>(
          d_base_data, d_hashtables, d_space_128_xx, (uint32_t*)d_space_128_yy);

  init_pr_lists<grid_size, block_size, base_num, match_degree,
                second_round_pruned_edge_num, second_round_reverse_edge_num,
                tomb, dim, true, float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          (pr_neighbor_list<uint32_t, second_round_reverse_edge_num,
                            second_round_pruned_edge_num>*)d_space_128_xx);

  fusion_prune_reverse_kernel_v0<grid_size, block_size, base_num, second_topk,
                                 second_round_pruned_edge_num,
                                 second_round_reverse_edge_num, tomb, dim, true,
                                 float, uint32_t>
      <<<grid_size, block_size, 0, compute_stream>>>(
          d_base_data, (uint32_t*)d_space_128_yy,
          (pr_neighbor_list<uint32_t, second_round_reverse_edge_num,
                            second_round_pruned_edge_num>*)d_space_128_xx);

  fusion_merge_sort_prune_kernel<grid_size, block_size, base_num,
                                 second_round_pruned_edge_num,
                                 second_round_reverse_edge_num, final_degree,
                                 tomb, dim, true, float, uint32_t>
      <<<projection_grid_size, projection_block_size, 0, compute_stream>>>(
          d_base_data,
          (pr_neighbor_list<uint32_t, second_round_reverse_edge_num,
                            second_round_pruned_edge_num>*)d_space_128_xx,
          (uint32_t*)d_space_128_yy);

  // 第二阶段完成，准备数据传输
  cudaEvent_t compute_done;
  cudaEventCreate(&compute_done);
  cudaEventRecord(compute_done, compute_stream);

  // 确保数据拷贝前计算已完成
  cudaStreamWaitEvent(copy_stream, compute_done, 0);

  // 5. 异步拷贝最终结果给CPU使用
  cudaMemcpyAsync(h_second_round_search_merge.data(), (uint32_t*)d_space_128_yy,
                  base_num * final_degree * sizeof(uint32_t),
                  cudaMemcpyDeviceToHost, copy_stream);

  cudaEventDestroy(compute_done);

  // 标记GPU第二阶段完成 - 这个事件用于CPU同步
  cudaEventRecord(gpu_phase2_done, copy_stream);  // 使用copy_stream确保数据拷贝完成

  // 等待GPU第二阶段完成
  auto gpu2_wait_start = std::chrono::high_resolution_clock::now();

  cudaError_t phase2_result = cudaEventSynchronize(gpu_phase2_done);

  auto gpu2_wait_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> gpu2_wait_duration =
      gpu2_wait_end - gpu2_wait_start;
  SPDLOG_INFO("CPU: GPU phase 2 wait completed in {:.2f} seconds",
              gpu2_wait_duration.count());

  // 6. 最终阶段 - 融合三个图
  auto fusion_start = std::chrono::high_resolution_clock::now();

  final_graph =
      FusionFinal(base_num, top1_projection_degree, final_degree, 70,
                  const_cast<float*>(h_base_data.data()), h_projection,
                  topnn_projection_graph, h_second_round_search_merge, dim, cpu_thread_limit);

  auto fusion_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> fusion_duration = fusion_end - fusion_start;
  SPDLOG_INFO("CPU: FusionFinal completed in {:.2f} seconds",
              fusion_duration.count());

  // 停止GPU计时
  cudaEventRecord(stop, compute_stream);

  auto test_end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> test_duration = test_end_time - test_start_time;
  SPDLOG_INFO("Total test execution time: {:.2f} seconds",
              test_duration.count());

  // 7. 保存最终图
  auto save_start = std::chrono::high_resolution_clock::now();

  SaveGraph(final_graph,
            "/home/yuxiang/Alaya/GGidxbuild/data/final_test_gpu_graph", ep,
            base_num);

  auto save_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> save_duration = save_end - save_start;
  SPDLOG_INFO("CPU: Graph saving completed in {:.2f} seconds",
              save_duration.count());

  // 等待GPU完成并计算时间
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  SPDLOG_INFO("GPU kernel execution time: {:.3f} seconds", milliseconds / 1000);

  // 清理资源
  cudaStreamDestroy(compute_stream);
  cudaStreamDestroy(copy_stream);
  cudaEventDestroy(data_ready);
  cudaEventDestroy(gpu_phase1_done);
  cudaEventDestroy(gpu_phase2_done);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_base_data);
  cudaFree(d_space_128_xx);
  cudaFree(d_space_128_yy);
  cudaFree(d_top1_projection);
  cudaFree(d_hashtables);

  // 计算CPU部分总时间
  auto cpu_total_duration = match_duration + gpu1_wait_duration + 
                           gpu2_wait_duration + fusion_duration + save_duration;
  
  // 各阶段时间统计
  SPDLOG_INFO("CPU time breakdown:");
  SPDLOG_INFO("  - GPU wait 1:   {:.2f}s ({:.1f}%)", gpu1_wait_duration.count(),
              100.0 * gpu1_wait_duration.count() / cpu_total_duration.count());
  SPDLOG_INFO("  - MatchNN:      {:.2f}s ({:.1f}%)", match_duration.count(),
              100.0 * match_duration.count() / cpu_total_duration.count());
  SPDLOG_INFO("  - GPU wait 2:   {:.2f}s ({:.1f}%)", gpu2_wait_duration.count(),
              100.0 * gpu2_wait_duration.count() / cpu_total_duration.count());
  SPDLOG_INFO("  - FusionFinal:  {:.2f}s ({:.1f}%)", fusion_duration.count(),
              100.0 * fusion_duration.count() / cpu_total_duration.count());
  SPDLOG_INFO("  - SaveGraph:    {:.2f}s ({:.1f}%)", save_duration.count(),
              100.0 * save_duration.count() / cpu_total_duration.count());
  SPDLOG_INFO("  - Total CPU:    {:.2f}s", cpu_total_duration.count());

}

}  // namespace Gpu
}  // namespace Gbuilder