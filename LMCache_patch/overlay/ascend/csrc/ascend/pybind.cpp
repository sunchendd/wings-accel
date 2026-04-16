// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>

#include "common/dcmi_management.h"
#include "mem_alloc.h"
#include "mem_kernels.h"

namespace py = pybind11;

PYBIND11_MODULE(c_ops, m) {
  py::enum_<TransferDirection>(m, "TransferDirection")
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();

  py::enum_<GPUKVFormat>(m, "GPUKVFormat")
      .value("NB_NL_TWO_BS_NH_HS", GPUKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", GPUKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_BS_NH_HS", GPUKVFormat::NL_X_NB_TWO_BS_NH_HS)
      .value("NL_X_NB_BS_HS", GPUKVFormat::NL_X_NB_BS_HS)
      .value("TWO_X_NL_X_NBBS_NH_HS", GPUKVFormat::TWO_X_NL_X_NBBS_NH_HS)
      .value("NL_X_NBBS_ONE_HS", GPUKVFormat::NL_X_NBBS_ONE_HS)
      .export_values();

  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_numa_ptr", &alloc_numa_ptr);
  m.def("free_numa_ptr", &free_numa_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("alloc_shm_pinned_ptr", &alloc_shm_pinned_ptr);
  m.def("free_shm_pinned_ptr", &free_shm_pinned_ptr);
  m.def("get_gpu_pci_bus_id", &get_npu_pci_bus_id);
}
