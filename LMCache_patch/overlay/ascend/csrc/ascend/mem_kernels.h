#pragma once

#include <cstdint>

#include <torch/extension.h>

#include "kernels/types.h"

enum class TransferDirection : int {
  H2D = 0,
  D2H = 1,
};

enum class GPUKVFormat : int {
  NB_NL_TWO_BS_NH_HS = 0,
  NL_X_TWO_NB_BS_NH_HS = 1,
  NL_X_NB_TWO_BS_NH_HS = 2,
  NL_X_NB_BS_HS = 3,
  TWO_X_NL_X_NBBS_NH_HS = 4,
  NL_X_NBBS_ONE_HS = 5,
};

namespace kvcache_ops {
void multi_layer_kv_transfer_kernel_v2(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    kvcache_ops::KVCacheFormat kvcache_format, uint32_t blockDim, void* stream,
    uint8_t* pagedKVCaches, uint8_t* dstCacheTensor, uint8_t* slotmappings,
    int64_t hiddenDims, int32_t kvs, int32_t numLayers, int64_t pageBuffSize,
    int32_t numTokensChunk, int64_t perLoopBuffer, int32_t maxTokensPerLoop,
    bool page2L);
}  // namespace kvcache_ops

void multi_layer_kv_transfer(torch::Tensor& key_value,
                             const torch::Tensor& key_value_ptrs,
                             const torch::Tensor& slot_mapping,
                             const torch::Device& paged_memory_device,
                             int page_buffer_size,
                             TransferDirection direction,
                             GPUKVFormat gpu_kv_format,
                             int block_size = 0);

void get_multi_layer_kv_transfer(torch::Tensor& key_value,
                                 const torch::Tensor& key_value_ptrs,
                                 const torch::Tensor& slot_mapping,
                                 const torch::Device& paged_memory_device,
                                 int page_buffer_size,
                                 TransferDirection direction,
                                 GPUKVFormat gpu_kv_format,
                                 int block_size = 0);
