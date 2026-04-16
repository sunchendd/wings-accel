#pragma once

#include <string>

namespace dcmi_ascend {

struct dcmi_pcie_info_all {
  unsigned int vendorId;
  unsigned int subvendorId;
  unsigned int deviceId;
  unsigned int subDeviceId;
  int domain;
  unsigned int bdf_busId;
  unsigned int bdf_deviceId;
  unsigned int bdf_funcId;
  unsigned char reserved[32];
};

class DCMIManager {
 public:
  static DCMIManager& GetInstance();
  ~DCMIManager();

  std::string GetDevicePcieInfoV2(int cardId, int deviceId,
                                  dcmi_pcie_info_all* pcieInfo);

 private:
  DCMIManager();

  void* libHandle_;
};

}  // namespace dcmi_ascend

std::string get_npu_pci_bus_id(int device);
