#include "utils.h"

#include "mem_kernels.h"

#include <torch/extension.h>

namespace wings_ascend {

kvcache_ops::AscendType get_dtype_from_torch(at::ScalarType scalarType) {
  switch (scalarType) {
    case at::ScalarType::Float:
      return kvcache_ops::AscendType::FP32;
    case at::ScalarType::BFloat16:
      return kvcache_ops::AscendType::BF16;
    case at::ScalarType::Half:
      return kvcache_ops::AscendType::FP16;
    case at::ScalarType::Long:
      return kvcache_ops::AscendType::INT64;
    case at::ScalarType::Int:
      return kvcache_ops::AscendType::INT32;
    default:
      TORCH_CHECK(false, "Unsupported Ascend scalar type: ", scalarType);
  }
}

kvcache_ops::KVCacheFormat get_kvcache_format(GPUKVFormat gpu_kv_format) {
  switch (gpu_kv_format) {
    case GPUKVFormat::NL_X_TWO_NB_BS_NH_HS:
      return kvcache_ops::KVCacheFormat::MERGED_KV;
    case GPUKVFormat::NL_X_NB_TWO_BS_NH_HS:
      return kvcache_ops::KVCacheFormat::SEPARATE_KV;
    default:
      TORCH_CHECK(false, "Unsupported Ascend GPU KV format: ",
                  static_cast<int>(gpu_kv_format));
  }
}

bool is_page_to_lmcache(TransferDirection direction) {
  return direction == TransferDirection::D2H;
}

}  // namespace wings_ascend
