#include "dcmi_management.h"

#include <dlfcn.h>

#include <iomanip>
#include <sstream>

#include "exception.h"

namespace dcmi_ascend {

using PcieInfoV2Func = int (*)(int, int, dcmi_pcie_info_all*);
using InitFunc = int (*)();

DCMIManager& DCMIManager::GetInstance() {
  static DCMIManager instance;
  return instance;
}

DCMIManager::DCMIManager() : libHandle_(nullptr) {
  libHandle_ = dlopen("libdcmi.so", RTLD_LAZY | RTLD_GLOBAL);
  WINGS_ASCEND_CHECK(libHandle_ != nullptr, "dlopen libdcmi.so failed");
  auto init_func = reinterpret_cast<InitFunc>(dlsym(libHandle_, "dcmi_init"));
  WINGS_ASCEND_CHECK(init_func != nullptr, "dlsym dcmi_init failed");
  int ret = init_func();
  WINGS_ASCEND_CHECK(ret == 0, "dcmi_init failed with ret=", ret);
}

DCMIManager::~DCMIManager() {
  if (libHandle_ != nullptr) {
    dlclose(libHandle_);
  }
}

std::string DCMIManager::GetDevicePcieInfoV2(int cardId, int deviceId,
                                             dcmi_pcie_info_all* pcieInfo) {
  auto func = reinterpret_cast<PcieInfoV2Func>(
      dlsym(libHandle_, "dcmi_get_device_pcie_info_v2"));
  WINGS_ASCEND_CHECK(func != nullptr,
                     "dlsym dcmi_get_device_pcie_info_v2 failed");
  int ret = func(cardId, deviceId, pcieInfo);
  WINGS_ASCEND_CHECK(ret == 0, "dcmi_get_device_pcie_info_v2 failed with ret=", ret);

  std::ostringstream oss;
  oss << std::setfill('0') << std::hex << std::setw(4) << pcieInfo->domain
      << ":" << std::setw(2) << pcieInfo->bdf_busId << ":" << std::setw(2)
      << pcieInfo->bdf_deviceId << "." << std::setw(1) << pcieInfo->bdf_funcId;
  return oss.str();
}

}  // namespace dcmi_ascend

std::string get_npu_pci_bus_id(int device) {
  auto& dcmi = dcmi_ascend::DCMIManager::GetInstance();
  dcmi_ascend::dcmi_pcie_info_all pcieInfo;
  return dcmi.GetDevicePcieInfoV2(device, 0, &pcieInfo);
}
