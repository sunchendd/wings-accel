#pragma once

#include <ATen/ATen.h>

#include "kernels/types.h"

enum class TransferDirection : int;
enum class GPUKVFormat : int;

namespace wings_ascend {

kvcache_ops::AscendType get_dtype_from_torch(at::ScalarType scalarType);
kvcache_ops::KVCacheFormat get_kvcache_format(GPUKVFormat gpu_kv_format);
bool is_page_to_lmcache(TransferDirection direction);

}  // namespace wings_ascend
