#ifndef UNIFIEDCACHE_IDEVICE_H
#define UNIFIEDCACHE_IDEVICE_H

#include <functional>
#include <memory>
#include "status/status.h"

namespace UC {

class IDevice {
public:
    IDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : deviceId{deviceId}, bufferSize{bufferSize}, bufferNumber{bufferNumber}
    {
    }
    virtual ~IDevice() = default;
    virtual Status Setup() = 0;
    virtual std::shared_ptr<std::byte> GetBuffer(const size_t size) = 0;
    virtual Status H2DSync(std::byte* dst, const std::byte* src, const size_t count) = 0;
    virtual Status D2HSync(std::byte* dst, const std::byte* src, const size_t count) = 0;
    virtual Status H2DAsync(std::byte* dst, const std::byte* src, const size_t count) = 0;
    virtual Status D2HAsync(std::byte* dst, const std::byte* src, const size_t count) = 0;
    virtual Status AppendCallback(std::function<void(bool)> cb) = 0;
    virtual Status Synchronized() = 0;
    virtual Status H2DBatchSync(std::byte* dArr[], const std::byte* hArr[], const size_t number,
                                const size_t count) = 0;
    virtual Status D2HBatchSync(std::byte* hArr[], const std::byte* dArr[], const size_t number,
                                const size_t count) = 0;

protected:
    virtual std::shared_ptr<std::byte> MakeBuffer(const size_t size) = 0;
    const int32_t deviceId;
    const size_t bufferSize;
    const size_t bufferNumber;
};

class DeviceFactory {
public:
    static std::unique_ptr<IDevice> Make(const int32_t deviceId, const size_t bufferSize,
                                         const size_t bufferNumber);
};

} // namespace UC

#endif
