#ifndef UNIFIEDCACHE_IBUFFERED_DEVICE_H
#define UNIFIEDCACHE_IBUFFERED_DEVICE_H

#include "idevice.h"

namespace UC {

class IBufferedDevice : public IDevice {
    class LinearBuffer {
        std::shared_ptr<std::byte> addr_{nullptr};
        size_t index_{0};
        size_t number_{0};
        size_t size_{0};

    public:
        void Setup(std::shared_ptr<std::byte> addr, const size_t number, const size_t size)
        {
            this->addr_ = addr;
            this->number_ = number;
            this->size_ = size;
            this->Reset();
        }
        void Reset() noexcept { this->index_ = 0; }
        bool Full() const noexcept { return this->index_ == this->number_; }
        bool Available(const size_t size) const noexcept { return this->size_ >= size; }
        std::shared_ptr<std::byte> Get() noexcept
        {
            auto addr = this->addr_.get();
            auto buffer = addr + this->size_ * this->index_;
            ++this->index_;
            return std::shared_ptr<std::byte>(buffer, [](auto) {});
        }
    };
    LinearBuffer buffer_;

public:
    IBufferedDevice(const int32_t deviceId, const size_t bufferSize, const size_t bufferNumber)
        : IDevice{deviceId, bufferSize, bufferNumber}
    {
    }
    Status Setup() override
    {
        auto totalSize = this->bufferSize * this->bufferNumber;
        if (totalSize == 0) { return Status::OK(); }
        auto addr = this->MakeBuffer(totalSize);
        if (!addr) { return Status::OutOfMemory(); }
        this->buffer_.Setup(addr, this->bufferNumber, this->bufferSize);
        return Status::OK();
    }
    virtual std::shared_ptr<std::byte> GetBuffer(const size_t size) override
    {
        if (this->buffer_.Full()) {
            auto status = this->Synchronized();
            if (status.Failure()) { return nullptr; }
            this->buffer_.Reset();
        }
        return this->buffer_.Available(size) ? this->buffer_.Get() : this->MakeBuffer(size);
    }
};

} // namespace UC

#endif
