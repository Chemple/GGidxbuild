#include "Gsearcher.cuh"
#include "spdlog/spdlog.h"
#include "utils.hpp"
#include <gtest/gtest.h>
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
#include <omp.h>
#include <sys/types.h>

namespace Gbuilder {
namespace Gpu {
TEST(TestSearch, SizeTest) {
  SPDLOG_INFO(
      "the size is {}",
      sizeof(search_warp_state<uint32_t, float, 128, 128, 6, 64, 1 << 11>));
}

template <uint32_t Km = 128, uint32_t Kp = 14, uint32_t Kd = 64,
          typename id_type = uint32_t, typename data_type = float,
          uint32_t tomb = 0XFFFFFFFF>
__global__ void test(id_type* __restrict__ topm_ids,
                     id_type* __restrict__ candidate_ids) {
  auto lane_id = threadIdx.x % 32;
  auto res = collect_top_p_normal<Km, Kp, Kd, id_type, data_type, tomb>(
      topm_ids, candidate_ids, lane_id);
  // printf("the res is %d\n", res);
}

template <uint32_t Km = 128, uint32_t Kp = 14, uint32_t Kd = 64,
          typename id_type = uint32_t, typename data_type = float,
          uint32_t tomb = 0XFFFFFFFF>
bool cpu_ref(id_type* __restrict__ topm_ids,
             id_type* __restrict__ candidate_ids) {
  auto find_num = 0;
  for (auto i = 0; i < Km; i++) {
    if (find_num < Kp) {
      if ((topm_ids[i] & 0X80000000) == 0) {
        candidate_ids[find_num * Kd] = topm_ids[i];
        topm_ids[i] |= 0X80000000;
        find_num++;
      }
    } else {
      break;
    }
  }
  if (find_num == 0) {
    return false;
  }
  while (find_num < Kp) {
    candidate_ids[find_num * Kd] = tomb;
    find_num++;
  }
  return true;
}

template <typename vec_tye>
void print_vec(std::vector<vec_tye> const& vec) {
  for (auto const& elem : vec) {
    std::cout << elem << ",";
  }
  printf("\n");
}

void check_vec(std::vector<uint32_t> const& vec0,
               std::vector<uint32_t> const& vec1) {
  for (auto i = 0; i < vec0.size(); i++) {
    if (vec0[i] != vec1[i]) {
      printf(
          "the error loc is %d, the left value is %d, the right value is %d\n",
          i, vec0[i], vec1[i]);
    }
  }
}

template <typename data_type = float, typename id_type = uint32_t>
using candidate_pool = std::priority_queue<std::pair<data_type, id_type>>;

template <typename data_type = float, typename id_type = uint32_t,
          uint32_t base_num, uint32_t dim, uint32_t degree>
candidate_pool<data_type, id_type> search(id_type const* graph,
                                          data_type const* data,
                                          data_type const* query, uint32_t topk,
                                          uint32_t L, uint32_t ep) {
  auto dist_func = [](data_type const* query, data_type const* candidate,
                      uint32_t dimention) {
    data_type dist = 0;
    for (auto i = 0; i < dimention; i++) {
      // SPDLOG_INFO("query[i] is {}, candidate[i] is {}", query[i],
      // candidate[i]);
      dist += query[i] * candidate[i];
    }
    return -dist;
  };

  auto visit_table = std::unordered_set<id_type>{};
  candidate_pool<data_type, id_type> top_candidates;
  candidate_pool<data_type, id_type> candidate_set;

  data_type dist = dist_func(query, data + ep * dim, dim);
  top_candidates.emplace(dist, ep);
  candidate_set.emplace(-dist, ep);
  visit_table.insert(ep);

  while (!candidate_set.empty()) {
    auto current_node_pair = candidate_set.top();
    candidate_set.pop();
    if ((-current_node_pair.first) > top_candidates.top().first &&
        top_candidates.size() == L)
      break;
    for (size_t m = 0; m < degree; m++) {
      unsigned candidate_id = graph[current_node_pair.second * degree + m];
      if (candidate_id == 0XFFFFFFFF) {
        continue;
      }
      if (visit_table.find(candidate_id) != visit_table.end()) continue;
      visit_table.insert(candidate_id);
      float dist = dist_func(query, data + candidate_id * dim, dim);
      if (top_candidates.size() < L || top_candidates.top().first > dist) {
        candidate_set.emplace(-dist, candidate_id);
        top_candidates.emplace(dist, candidate_id);
        if (top_candidates.size() > L) top_candidates.pop();
      }
    }
  }
  while (top_candidates.size() > topk) {
    top_candidates.pop();
  }
  return top_candidates;
}

template <uint32_t query_num, uint32_t base_num, uint32_t dim, uint32_t topk,
          uint32_t gt_topk, uint32_t degree, typename data_type,
          typename id_type>
void testSearch(id_type const* graph, data_type* data, data_type* query,
                id_type* gt, uint32_t L) {
  std::vector<candidate_pool<data_type, id_type>> knn(query_num);

  for (size_t i = 0; i < query_num; i++) {
    knn[i] = search<float, uint32_t, base_num, dim, degree>(
        graph, data, query + i * dim, topk, L, 1000);
  }

  float recall = 0;
  for (size_t i = 0; i < query_num; i++) {
    auto tmp = knn[i];
    auto hash_table = std::unordered_set<uint32_t>{};
    float num = 0;
    while (!tmp.empty()) {
      hash_table.insert(tmp.top().second);
      tmp.pop();
    }
    for (auto j = 0; j < topk; j++) {
      if (hash_table.find(gt[i * gt_topk + j]) != hash_table.end()) {
        num++;
      }
    }
    recall += num / topk;
  }
  std::cout << "recall: " << recall / query_num << std::endl;
}

TEST(TestSearch, TestSelectTopPSimple) {
  constexpr uint32_t Km = 128;
  constexpr uint32_t Kp = 14;
  constexpr uint32_t Kd = 64;
  constexpr uint32_t list_length = Km + Kp * Kd;

  auto host_list = std::vector<uint32_t>(list_length);

  for (auto i = 0; i < host_list.size(); i++) {
    if (i % 2 == 0) {
      host_list[i] = i;
    } else {
      host_list[i] = i | 0x80000000;
    }
  }

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(host_list.begin(), host_list.end(),
               std::default_random_engine(seed));

  // copy
  auto check_host_list = host_list;
  uint32_t* d_list = nullptr;

  // print_vec(host_list);

  cudaMalloc(&d_list, sizeof(uint32_t) * list_length);
  cudaMemcpy(d_list, host_list.data(), sizeof(uint32_t) * list_length,
             cudaMemcpyHostToDevice);

  test<Km, Kp, Kd, uint32_t, float, 0XFFFFFFFF><<<1, 32>>>(d_list, d_list + Km);
  cpu_ref<Km, Kp, Kd, uint32_t, float, 0XFFFFFFFF>(check_host_list.data(),
                                                   check_host_list.data() + Km);

  cudaMemcpy(host_list.data(), d_list, sizeof(uint32_t) * list_length,
             cudaMemcpyDeviceToHost);

  // print_vec(host_list);
  // print_vec(check_host_list);

  // check_vec(host_list, check_host_list);

  ASSERT_EQ(host_list, check_host_list);
}

template <typename vec_type>
bool read_vec_from_file(std::vector<vec_type>& vec, char const* file_path) {
  if (std::filesystem::exists(file_path)) {
    std::ifstream file(file_path, std::ios::binary);
    if (file.is_open()) {
      file.read(reinterpret_cast<char*>(vec.data()),
                vec.size() * sizeof(vec_type));
      file.close();
      SPDLOG_INFO("loaded from file: {}", file_path);
      return true;
    } else {
      SPDLOG_ERROR("Failed to open file: {}", file_path);
      return false;
    }
  }
  SPDLOG_ERROR("file is not exist: {}", file_path);
  return false;
}

TEST(TestSearch, DISABLED_TestSearch) {
  constexpr uint32_t dim = 128;
  constexpr uint32_t degree = 64;
  constexpr uint32_t base_num = 10000;
  constexpr uint32_t query_num = 10000;
  constexpr uint32_t gt_topk = 100;
  constexpr uint32_t topk = 100;

  auto host_graph = std::vector<uint32_t>(degree * base_num);
  auto host_data = std::vector<float>(dim * base_num);
  // auto host_query = std::vector<float>(query_num * dim);
  auto gt = std::vector<uint32_t>(gt_topk * base_num);

  auto graph_file_name =
      "/home/shiwen/project/GGidxbuild/data/search_graph.bin";
  auto data_file_name = "/home/shiwen/project/GGidxbuild/data/vectors.fbin";
  auto gt_file_name = "/home/shiwen/project/GGidxbuild/data/groundtruth.ibin";

  read_vec_from_file(host_graph, graph_file_name);
  read_vec_from_file(host_data, data_file_name);
  read_vec_from_file(gt, gt_file_name);

  // print_vec(host_data);

  // std::mt19937
  // gen(std::chrono::system_clock::now().time_since_epoch().count());
  // std::uniform_real_distribution<float> random_val(-1, 1);
  // for (auto& elem : host_query) {
  //   elem = random_val(gen);
  // }

  uint32_t* d_graph = nullptr;
  uint32_t* d_base_data = nullptr;
  uint32_t* d_result = nullptr;

  testSearch<query_num, base_num, dim, topk, gt_topk, degree, float, uint32_t>(
      host_graph.data(), host_data.data(), host_data.data(), gt.data(), 128);
}

// TEST(TestSearch, DISABLED_GpuSearchOneQueryTest) {
//   constexpr uint32_t dim = 128;
//   constexpr uint32_t degree = 64;
//   constexpr uint32_t base_num = 10000;
//   constexpr uint32_t query_num = 10000;
//   constexpr uint32_t gt_topk = 100;
//   constexpr uint32_t topk = 10;
//   constexpr uint32_t grid_size = 144;
//   constexpr uint32_t block_size = 128;
//   constexpr uint32_t Km = 64;
//   constexpr uint32_t Kp = 2;
//   constexpr uint32_t Kd = 32;
//   constexpr uint32_t reset_iter = 3;
//   constexpr uint32_t hashtable_size = 1 << 10;
//   constexpr uint32_t shared_memory_size =
//       (block_size / 32) *
//       sizeof(
//           search_warp_state<uint32_t, float, dim, Km, Kp, Kd,
//           hashtable_size>);

//   SPDLOG_INFO("the shared memory size is {}", shared_memory_size);

//   auto host_graph = std::vector<uint32_t>(degree * base_num);
//   auto host_data = std::vector<float>(dim * base_num);
//   // auto host_query = std::vector<float>(query_num * dim);
//   auto gt = std::vector<uint32_t>(gt_topk * base_num);

//   auto graph_file_name =
//       "/home/shiwen/project/GGidxbuild/data/search_graph.bin";
//   auto data_file_name = "/home/shiwen/project/GGidxbuild/data/vectors.fbin";
//   auto gt_file_name =
//   "/home/shiwen/project/GGidxbuild/data/groundtruth.ibin";

//   read_vec_from_file(host_graph, graph_file_name);
//   read_vec_from_file(host_data, data_file_name);
//   read_vec_from_file(gt, gt_file_name);

//   // print_vec(host_data);

//   // std::mt19937
//   // gen(std::chrono::system_clock::now().time_since_epoch().count());
//   // std::uniform_real_distribution<float> random_val(-1, 1);
//   // for (auto& elem : host_query) {
//   //   elem = random_val(gen);
//   // }

//   uint32_t* d_graph = nullptr;
//   float* d_base_data = nullptr;
//   uint32_t* d_result = nullptr;
//   // float* d_distance = nullptr;

//   // testSearch<query_num, base_num, dim, topk, gt_topk, degree, float,
//   // uint32_t>(
//   //     host_graph.data(), host_data.data(), host_data.data(), gt.data(),
//   128);

//   cudaMalloc(&d_graph, degree * base_num * sizeof(uint32_t));
//   cudaMalloc(&d_base_data, dim * base_num * sizeof(float));
//   cudaMalloc(&d_result, topk * base_num * sizeof(uint32_t));
//   // cudaMalloc(&d_distance, Km * base_num * sizeof(float));

//   cudaMemcpy(d_graph, host_graph.data(), degree * base_num *
//   sizeof(uint32_t),
//              cudaMemcpyHostToDevice);
//   cudaMemcpy(d_base_data, host_data.data(), dim * base_num * sizeof(float),
//              cudaMemcpyHostToDevice);

//   auto carveout = 100;
//   cudaFuncSetAttribute(
//       enhance_search<grid_size, block_size, base_num, query_num, dim, degree,
//                      shared_memory_size, Km, Kp, Kd, topk, 0XFFFFFFFF,
//                      hashtable_size, reset_iter, uint32_t, float>,
//       cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
//   cudaCheckError();

//   SPDLOG_INFO("begin gpu searching");
//   enhance_search<grid_size, block_size, base_num, query_num, dim, degree,
//                  shared_memory_size, Km, Kp, Kd, topk, 0XFFFFFFFF,
//                  hashtable_size, reset_iter, uint32_t, float>
//       <<<grid_size, block_size, shared_memory_size>>>(d_base_data,
//       d_base_data,
//                                                       d_graph, d_result);

//   cudaCheckError();
//   cudaDeviceSynchronize();
//   cudaCheckError();
//   SPDLOG_INFO("finish gpu searching");

//   auto check_res = std::vector<uint32_t>(topk * base_num);
//   // auto check_distance = std::vector<float>(Km * base_num);
//   cudaMemcpy(check_res.data(), d_result, topk * base_num * sizeof(uint32_t),
//              cudaMemcpyDeviceToHost);
//   // cudaMemcpy(check_distance.data(), d_distance,
//   //            Km * base_num * sizeof(uint32_t), cudaMemcpyDeviceToHost);

//   cudaCheckError();

//   uint32_t hit = 0;
//   for (auto i = 0; i < query_num; i++) {
//     auto hash_table = std::unordered_set<uint32_t>{};
//     // auto local_hit = 0;
//     for (auto j = 0; j < topk; j++) {
//       // SPDLOG_INFO("{}", check_res[i * topk + j]);
//       hash_table.insert(check_res[i * topk + j]);
//     }
//     for (auto j = 0; j < topk; j++) {
//       if (hash_table.find(gt[i * gt_topk + j]) != hash_table.end()) {
//         hit++;
//         // local_hit++;
//       }
//     }
//     // SPDLOG_INFO("the local recall is {}", (1.0 * local_hit) / topk);
//   }
//   auto final_recall = 1.0 * hit / (query_num * topk);
//   SPDLOG_INFO("the final recall is {}", final_recall);
// }

// TEST(TestSearch, GpuScaleSharedHash) {
//   constexpr uint32_t dim = 200;
//   constexpr uint32_t degree = 64;
//   constexpr uint32_t base_num = 10 * 1000 * 1000;
//   constexpr uint32_t query_num = 10 * 1000;
//   constexpr uint32_t gt_topk = 500;
//   constexpr uint32_t topk = 10;
//   constexpr uint32_t grid_size = 144;
//   constexpr uint32_t block_size = 128 + 64;
//   constexpr uint32_t Km = 128;
//   constexpr uint32_t Kp = 2;
//   constexpr uint32_t Kd = 64;
//   constexpr uint32_t reset_iter = 3;
//   constexpr uint32_t hashtable_size = 1 << 10;
//   constexpr uint32_t shared_memory_size =
//       (block_size / 32) *
//       sizeof(
//           search_warp_state<uint32_t, float, dim, Km, Kp, Kd,
//           hashtable_size>);

//   SPDLOG_INFO("the shared memory size is {}", shared_memory_size);

//   auto host_graph = std::vector<uint32_t>(degree * base_num);
//   auto host_data = std::vector<float>(dim * base_num);
//   auto host_query = std::vector<float>(dim * query_num);
//   // auto host_query = std::vector<float>(query_num * dim);
//   auto gt = std::vector<uint32_t>(gt_topk * base_num);

//   auto graph_file_name =
//       "/home/shiwen/project/GGidxbuild/data/10M_200/fixed_graph.bin";
//   auto data_file_name =
//       "/home/shiwen/project/GGidxbuild/data/10M_200/vector.fbin";
//   auto query_file_name =
//       "/home/shiwen/project/GGidxbuild/data/10M_200/query.fbin";
//   auto gt_file_name = "/home/shiwen/project/GGidxbuild/data/10M_200/gt.ibin";

//   read_vec_from_file(host_graph, graph_file_name);
//   read_vec_from_file(host_data, data_file_name);
//   read_vec_from_file(host_query, query_file_name);
//   read_vec_from_file(gt, gt_file_name);

//   // print_vec(host_data);

//   // std::mt19937
//   // gen(std::chrono::system_clock::now().time_since_epoch().count());
//   // std::uniform_real_distribution<float> random_val(-1, 1);
//   // for (auto& elem : host_query) {
//   //   elem = random_val(gen);
//   // }

//   uint32_t* d_graph = nullptr;
//   float* d_base_data = nullptr;
//   float* d_query_data = nullptr;
//   uint32_t* d_result = nullptr;
//   // float* d_distance = nullptr;

//   // testSearch<query_num, base_num, dim, topk, gt_topk, degree, float,
//   // uint32_t>(
//   //     host_graph.data(), host_data.data(), host_data.data(), gt.data(),
//   128);

//   cudaMalloc(&d_graph, degree * base_num * sizeof(uint32_t));
//   cudaMalloc(&d_base_data, dim * base_num * sizeof(float));
//   cudaMalloc(&d_query_data, dim * query_num * sizeof(float));
//   cudaMalloc(&d_result, topk * base_num * sizeof(uint32_t));
//   // cudaMalloc(&d_distance, Km * base_num * sizeof(float));

//   cudaMemcpy(d_graph, host_graph.data(), degree * base_num *
//   sizeof(uint32_t),
//              cudaMemcpyHostToDevice);
//   cudaMemcpy(d_base_data, host_data.data(), dim * base_num * sizeof(float),
//              cudaMemcpyHostToDevice);
//   cudaMemcpy(d_query_data, host_query.data(), dim * query_num *
//   sizeof(float),
//              cudaMemcpyHostToDevice);

//   // auto carveout = 100;
//   // cudaFuncSetAttribute(
//   //     enhance_search<grid_size, block_size, base_num, query_num, dim,
//   degree,
//   //                    shared_memory_size, Km, Kp, Kd, topk, 0XFFFFFFFF,
//   //                    hashtable_size, reset_iter, uint32_t, float>,
//   //     cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
//   // cudaCheckError();

//   SPDLOG_INFO("begin gpu searching");
//   enhance_search<grid_size, block_size, base_num, query_num, dim, degree,
//                  shared_memory_size, Km, Kp, Kd, topk, 0XFFFFFFFF,
//                  hashtable_size, reset_iter, uint32_t, float>
//       <<<grid_size, block_size, shared_memory_size>>>(d_base_data,
//       d_query_data,
//                                                       d_graph, d_result);

//   cudaCheckError();
//   cudaDeviceSynchronize();
//   cudaCheckError();
//   SPDLOG_INFO("finish gpu searching");

//   auto check_res = std::vector<uint32_t>(topk * base_num);
//   // auto check_distance = std::vector<float>(Km * base_num);
//   cudaMemcpy(check_res.data(), d_result, topk * base_num * sizeof(uint32_t),
//              cudaMemcpyDeviceToHost);
//   // cudaMemcpy(check_distance.data(), d_distance,
//   //            Km * base_num * sizeof(uint32_t), cudaMemcpyDeviceToHost);

//   cudaCheckError();

//   uint32_t hit = 0;
//   for (auto i = 0; i < query_num; i++) {
//     auto hash_table = std::unordered_set<uint32_t>{};
//     // auto local_hit = 0;
//     for (auto j = 0; j < topk; j++) {
//       // SPDLOG_INFO("{}", check_res[i * topk + j]);
//       hash_table.insert(check_res[i * topk + j]);
//     }
//     for (auto j = 0; j < topk; j++) {
//       if (hash_table.find(gt[i * gt_topk + j]) != hash_table.end()) {
//         hit++;
//         // local_hit++;
//       }
//     }
//     // SPDLOG_INFO("the local recall is {}", (1.0 * local_hit) / topk);
//   }
//   auto final_recall = 1.0 * hit / (query_num * topk);
//   SPDLOG_INFO("the final recall is {}", final_recall);
// }

// TEST(TestSearch, GpuScaleGlobalHash) {
//   constexpr uint32_t dim = 200;
//   constexpr uint32_t degree = 64;
//   constexpr uint32_t base_num = 10 * 1000 * 1000;
//   constexpr uint32_t query_num = 10 * 1000;
//   constexpr uint32_t gt_topk = 500;
//   constexpr uint32_t topk = 10;
//   constexpr uint32_t grid_size = 144;
//   constexpr uint32_t block_size = 416;
//   constexpr uint32_t Km = 128;
//   constexpr uint32_t Kp = 2;
//   constexpr uint32_t Kd = 64;
//   constexpr uint32_t reset_iter = 15;
//   constexpr uint32_t hashtable_size = 1 << 12;
//   constexpr uint32_t shared_memory_size =
//       (block_size / 32) *
//       sizeof(
//           search_warp_state_global_hashtable<uint32_t, float, dim, Km, Kp,
//           Kd>);

//   constexpr uint32_t global_warp_num = grid_size * block_size / 32;

//   SPDLOG_INFO("the shared memory size is {}", shared_memory_size);

//   auto host_graph = std::vector<uint32_t>(degree * base_num);
//   auto host_data = std::vector<float>(dim * base_num);
//   auto host_query = std::vector<float>(dim * query_num);
//   // auto host_query = std::vector<float>(query_num * dim);
//   auto gt = std::vector<uint32_t>(gt_topk * base_num);

//   auto graph_file_name =
//       "/home/shiwen/project/GGidxbuild/data/10M_200/fixed_graph.bin";
//   auto data_file_name =
//       "/home/shiwen/project/GGidxbuild/data/10M_200/vector.fbin";
//   auto query_file_name =
//       "/home/shiwen/project/GGidxbuild/data/10M_200/query.fbin";
//   auto gt_file_name = "/home/shiwen/project/GGidxbuild/data/10M_200/gt.ibin";

//   read_vec_from_file(host_graph, graph_file_name);
//   read_vec_from_file(host_data, data_file_name);
//   read_vec_from_file(host_query, query_file_name);
//   read_vec_from_file(gt, gt_file_name);

//   // print_vec(host_data);

//   // std::mt19937
//   // gen(std::chrono::system_clock::now().time_since_epoch().count());
//   // std::uniform_real_distribution<float> random_val(-1, 1);
//   // for (auto& elem : host_query) {
//   //   elem = random_val(gen);
//   // }

//   uint32_t* d_graph = nullptr;
//   float* d_base_data = nullptr;
//   float* d_query_data = nullptr;
//   uint32_t* d_result = nullptr;
//   HashTable<uint32_t, 1 << 12>* d_hashtables = nullptr;
//   // float* d_distance = nullptr;

//   // testSearch<query_num, base_num, dim, topk, gt_topk, degree, float,
//   // uint32_t>(
//   //     host_graph.data(), host_data.data(), host_data.data(), gt.data(),
//   // 128);

//   cudaMalloc(&d_graph, degree * base_num * sizeof(uint32_t));
//   cudaMalloc(&d_base_data, dim * base_num * sizeof(float));
//   cudaMalloc(&d_query_data, dim * query_num * sizeof(float));
//   cudaMalloc(&d_result, topk * base_num * sizeof(uint32_t));
//   cudaMalloc(&d_hashtables,
//              global_warp_num * hashtable_size * sizeof(uint32_t));
//   // cudaMalloc(&d_distance, Km * base_num * sizeof(float));

//   cudaMemcpy(d_graph, host_graph.data(), degree * base_num *
//   sizeof(uint32_t),
//              cudaMemcpyHostToDevice);
//   cudaMemcpy(d_base_data, host_data.data(), dim * base_num * sizeof(float),
//              cudaMemcpyHostToDevice);
//   cudaMemcpy(d_query_data, host_query.data(), dim * query_num *
//   sizeof(float),
//              cudaMemcpyHostToDevice);

//   // auto carveout = 100;
//   // cudaFuncSetAttribute(
//   //     enhance_search<grid_size, block_size, base_num, query_num, dim,
//   // degree,
//   //                    shared_memory_size, Km, Kp, Kd, topk, 0XFFFFFFFF,
//   //                    hashtable_size, reset_iter, uint32_t, float>,
//   //     cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
//   // cudaCheckError();

//   SPDLOG_INFO("begin gpu searching");
//   search_global_hashtable<grid_size, block_size, base_num, query_num, dim,
//                           degree, shared_memory_size, Km, Kp, Kd, topk,
//                           0XFFFFFFFF, hashtable_size, reset_iter, uint32_t,
//                           float><<<grid_size, block_size,
//                           shared_memory_size>>>(
//       d_base_data, d_query_data, d_hashtables, d_graph, d_result);

//   cudaCheckError();
//   cudaDeviceSynchronize();
//   cudaCheckError();
//   SPDLOG_INFO("finish gpu searching");

//   auto check_res = std::vector<uint32_t>(topk * base_num);
//   // auto check_distance = std::vector<float>(Km * base_num);
//   cudaMemcpy(check_res.data(), d_result, topk * base_num * sizeof(uint32_t),
//              cudaMemcpyDeviceToHost);
//   // cudaMemcpy(check_distance.data(), d_distance,
//   //            Km * base_num * sizeof(uint32_t), cudaMemcpyDeviceToHost);

//   cudaCheckError();

//   uint32_t hit = 0;
//   for (auto i = 0; i < query_num; i++) {
//     auto hash_table = std::unordered_set<uint32_t>{};
//     // auto local_hit = 0;
//     for (auto j = 0; j < topk; j++) {
//       // SPDLOG_INFO("{}", check_res[i * topk + j]);
//       hash_table.insert(check_res[i * topk + j]);
//     }
//     for (auto j = 0; j < topk; j++) {
//       if (hash_table.find(gt[i * gt_topk + j]) != hash_table.end()) {
//         hit++;
//         // local_hit++;
//       }
//     }
//     // SPDLOG_INFO("the local recall is {}", (1.0 * local_hit) / topk);
//   }
//   auto final_recall = 1.0 * hit / (query_num * topk);
//   SPDLOG_INFO("the final recall is {}", final_recall);
// }

TEST(TestSearch, GpuEnhanceLink) {
  constexpr uint32_t dim = 200;
  constexpr uint32_t degree = 16;
  constexpr uint32_t base_num = 10 * 1000 * 1000;
  constexpr uint32_t grid_size = 144;
  constexpr uint32_t block_size = 512;
  constexpr uint32_t Km = 32;
  constexpr uint32_t Kp = 2;
  constexpr uint32_t Kd = 16;
  constexpr uint32_t topk = Km + Kp * Kd;
  constexpr uint32_t reset_iter = 15;
  constexpr uint32_t hashtable_size = 1 << 12;
  constexpr uint32_t shared_memory_size =
      (block_size / 32) *
      sizeof(
          search_warp_state_global_hashtable<uint32_t, float, dim, Km, Kp, Kd>);

  constexpr uint32_t global_warp_num = grid_size * block_size / 32;

  SPDLOG_INFO("the shared memory size is {}", shared_memory_size);

  auto host_graph = std::vector<uint32_t>(degree * base_num);
  auto host_data = std::vector<float>(dim * base_num);
  // auto host_query = std::vector<float>(dim * query_num);
  // auto host_query = std::vector<float>(query_num * dim);
  // auto gt = std::vector<uint32_t>(gt_topk * base_num);

  auto graph_file_name =
      "/home/shiwen/project/GGidxbuild/data/10M_200/top1_projection_graph.bin";
  auto data_file_name =
      "/home/shiwen/project/GGidxbuild/data/10M_200/vector.fbin";
  // auto query_file_name =
  //     "/home/shiwen/project/GGidxbuild/data/10M_200/query.fbin";
  // auto gt_file_name = "/home/shiwen/project/GGidxbuild/data/10M_200/gt.ibin";

  read_vec_from_file(host_graph, graph_file_name);
  read_vec_from_file(host_data, data_file_name);
  // read_vec_from_file(host_query, query_file_name);
  // read_vec_from_file(gt, gt_file_name);

  // print_vec(host_data);

  // std::mt19937
  // gen(std::chrono::system_clock::now().time_since_epoch().count());
  // std::uniform_real_distribution<float> random_val(-1, 1);
  // for (auto& elem : host_query) {
  //   elem = random_val(gen);
  // }

  uint32_t* d_graph = nullptr;
  float* d_base_data = nullptr;
  // float* d_query_data = nullptr;
  uint32_t* d_result = nullptr;
  HashTable<uint32_t, 1 << 12>* d_hashtables = nullptr;
  // float* d_distance = nullptr;

  // testSearch<query_num, base_num, dim, topk, gt_topk, degree, float,
  // uint32_t>(
  //     host_graph.data(), host_data.data(), host_data.data(), gt.data(),
  // 128);

  cudaMalloc(&d_graph, degree * base_num * sizeof(uint32_t));
  cudaMalloc(&d_base_data, dim * base_num * sizeof(float));
  // cudaMalloc(&d_query_data, dim * query_num * sizeof(float));
  cudaMalloc(&d_result, topk * base_num * sizeof(uint32_t));
  cudaMalloc(&d_hashtables,
             global_warp_num * hashtable_size * sizeof(uint32_t));
  // cudaMalloc(&d_distance, Km * base_num * sizeof(float));

  cudaMemcpy(d_graph, host_graph.data(), degree * base_num * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_base_data, host_data.data(), dim * base_num * sizeof(float),
             cudaMemcpyHostToDevice);
  // cudaMemcpy(d_query_data, host_query.data(), dim * query_num *
  // sizeof(float),
  //            cudaMemcpyHostToDevice);

  // auto carveout = 100;
  // cudaFuncSetAttribute(
  //     enhance_search<grid_size, block_size, base_num, query_num, dim,
  // degree,
  //                    shared_memory_size, Km, Kp, Kd, topk, 0XFFFFFFFF,
  //                    hashtable_size, reset_iter, uint32_t, float>,
  //     cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
  // cudaCheckError();

  SPDLOG_INFO("begin gpu searching");
  link_process_global_hashtable<
      grid_size, block_size, base_num, dim, degree, shared_memory_size, Km, Kp,
      Kd, topk, 0XFFFFFFFF, hashtable_size, reset_iter, uint32_t, float>
      <<<grid_size, block_size, shared_memory_size>>>(d_base_data, d_hashtables,
                                                      d_graph, d_result);

  cudaCheckError();
  cudaDeviceSynchronize();
  cudaCheckError();
  SPDLOG_INFO("finish gpu searching");

  auto check_res = std::vector<uint32_t>(topk * base_num);
  // auto check_distance = std::vector<float>(Km * base_num);
  cudaMemcpy(check_res.data(), d_result, topk * base_num * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  // cudaMemcpy(check_distance.data(), d_distance,
  //            Km * base_num * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  cudaCheckError();

  // uint32_t hit = 0;
  // for (auto i = 0; i < query_num; i++) {
  //   auto hash_table = std::unordered_set<uint32_t>{};
  //   // auto local_hit = 0;
  //   for (auto j = 0; j < topk; j++) {
  //     // SPDLOG_INFO("{}", check_res[i * topk + j]);
  //     hash_table.insert(check_res[i * topk + j]);
  //   }
  //   for (auto j = 0; j < topk; j++) {
  //     if (hash_table.find(gt[i * gt_topk + j]) != hash_table.end()) {
  //       hit++;
  //       // local_hit++;
  //     }
  //   }
  //   // SPDLOG_INFO("the local recall is {}", (1.0 * local_hit) / topk);
  // }
  // auto final_recall = 1.0 * hit / (query_num * topk);
  // SPDLOG_INFO("the final recall is {}", final_recall);
}

}  // namespace Gpu
}  // namespace Gbuilder
