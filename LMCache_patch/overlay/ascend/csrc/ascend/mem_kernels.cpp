#include "mem_kernels.h"

#include <Python.h>

#include <algorithm>
#include <c10/core/DeviceGuard.h>
#include <pybind11/pybind11.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/npu/Module.h>
#include "tiling/platform/platform_ascendc.h"

#include "utils.h"

namespace py = pybind11;

namespace {

[[noreturn]] void throw_not_supported(const char* message) {
  PyErr_SetString(PyExc_NotImplementedError, message);
  throw py::error_already_set();
}

void validate_native_transfer(const torch::Tensor& key_value,
                              const torch::Tensor& key_value_ptrs,
                              const torch::Tensor& slot_mapping,
                              GPUKVFormat gpu_kv_format) {
  if (gpu_kv_format != GPUKVFormat::NL_X_TWO_NB_BS_NH_HS &&
      gpu_kv_format != GPUKVFormat::NL_X_NB_TWO_BS_NH_HS) {
    throw_not_supported(
        "Ascend native save/load currently supports only the [2, nb, bs, nh, hs] and [nb, 2, bs, nh, hs] vLLM KV layouts.");
  }

  if (key_value.dim() != 4 || key_value.size(0) != 2) {
    TORCH_CHECK(false, "Ascend native transfer expects [2, num_layers, num_tokens, hidden]");
  }

  TORCH_CHECK(key_value_ptrs.dim() == 1,
              "Ascend native transfer expects a 1-D pointer tensor");
  TORCH_CHECK(slot_mapping.dim() == 1,
              "Ascend native transfer expects a 1-D slot mapping tensor");

  const int expected_ptrs =
      gpu_kv_format == GPUKVFormat::NL_X_NB_TWO_BS_NH_HS ? key_value.size(1) * 2
                                                         : key_value.size(1);
  TORCH_CHECK(
      key_value_ptrs.size(0) == expected_ptrs,
      "Ascend native transfer received an unexpected pointer tensor length");
}

}  // namespace

void multi_layer_kv_transfer(torch::Tensor& key_value,
                             const torch::Tensor& key_value_ptrs,
                             const torch::Tensor& slot_mapping,
                             const torch::Device& paged_memory_device,
                             int page_buffer_size,
                             TransferDirection direction,
                             GPUKVFormat gpu_kv_format,
                             int block_size) {
  (void)block_size;
  validate_native_transfer(key_value, key_value_ptrs, slot_mapping, gpu_kv_format);

  auto* key_value_ptr = static_cast<uint8_t*>(key_value.data_ptr());
  auto* page_buffer_ptrs = static_cast<uint8_t*>(key_value_ptrs.data_ptr());
  auto* slot_mapping_ptr = static_cast<uint8_t*>(slot_mapping.data_ptr());

  int num_layers = static_cast<int>(key_value.size(1));
  int num_tokens = static_cast<int>(slot_mapping.size(0));
  int hidden_dims = static_cast<int>(key_value.size(-1));
  int kv_size = static_cast<int>(key_value.size(0));

  const c10::OptionalDeviceGuard device_guard(paged_memory_device);

  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();
  at::ScalarType scalar_type = key_value.scalar_type();
  at::ScalarType slot_type = slot_mapping.scalar_type();

  const char* soc_name = aclrtGetSocName();
  auto ascendc_platform =
      platform_ascendc::PlatformAscendCManager::GetInstance(soc_name);
  uint64_t ub_size = 0;
  ascendc_platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);

  uint32_t aiv_num = static_cast<uint32_t>(std::min(num_layers, 4));
  int32_t num_buffs_on_dev = 2;
  int64_t base_buff_size =
      static_cast<int64_t>(num_buffs_on_dev) * hidden_dims *
      key_value.element_size();

  TORCH_CHECK(ub_size >= static_cast<uint64_t>(base_buff_size),
              "Ascend UB size is smaller than the per-token transfer buffer");

  int32_t max_tokens_per_loop =
      static_cast<int32_t>(ub_size / static_cast<uint64_t>(base_buff_size)) - 1;
  max_tokens_per_loop = std::max(1, std::min(max_tokens_per_loop, num_tokens));

  int64_t total_per_loop_buffer =
      static_cast<int64_t>(max_tokens_per_loop) * base_buff_size;
  TORCH_CHECK(ub_size >= static_cast<uint64_t>(total_per_loop_buffer),
              "Ascend UB size is smaller than the computed per-loop transfer buffer");

  int64_t single_per_loop_buffer = total_per_loop_buffer / num_buffs_on_dev;
  auto kvcache_format = wings_ascend::get_kvcache_format(gpu_kv_format);
  bool page_to_lmcache = wings_ascend::is_page_to_lmcache(direction);

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_kv_transfer_kernel_v2");
  cmd.SetCustomHandler([=]() -> int {
    auto slot_num = wings_ascend::get_dtype_from_torch(slot_type);
    auto dtype_num = wings_ascend::get_dtype_from_torch(scalar_type);
    kvcache_ops::multi_layer_kv_transfer_kernel_v2(
        dtype_num, slot_num, kvcache_format, aiv_num,
        const_cast<void*>(reinterpret_cast<const void*>(stream)), page_buffer_ptrs,
        key_value_ptr, slot_mapping_ptr, hidden_dims, kv_size, num_layers,
        page_buffer_size, num_tokens, single_per_loop_buffer,
        max_tokens_per_loop, page_to_lmcache);
    return 0;
  });
  cmd.Run();
}

void get_multi_layer_kv_transfer(torch::Tensor& key_value,
                                 const torch::Tensor& key_value_ptrs,
                                 const torch::Tensor& slot_mapping,
                                 const torch::Device& paged_memory_device,
                                 int page_buffer_size,
                                 TransferDirection direction,
                                 GPUKVFormat gpu_kv_format,
                                 int block_size) {
  multi_layer_kv_transfer(key_value, key_value_ptrs, slot_mapping,
                          paged_memory_device, page_buffer_size, direction,
                          gpu_kv_format, block_size);
}
