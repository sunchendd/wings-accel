#ifndef UNIFIEDCACHE_LOCALSTORE_H
#define UNIFIEDCACHE_LOCALSTORE_H

#include "kvstore.h"

namespace UC {

class LocalStore : public CCStore<> {
public:
    struct Config {
        size_t ioSize;
        size_t capacity;
        void* backend;
        int32_t deviceId;
        Config(const size_t ioSize, const size_t capacity)
            : ioSize{ioSize}, capacity{capacity}, backend{nullptr}, deviceId{-1}
        {
        }
    };

public:
    LocalStore() : impl_{nullptr} {}
    ~LocalStore() override
    {
        if (this->impl_) { delete this->impl_; }
    }
    int32_t Setup(const Config& config);
    int32_t Alloc(const std::string& block) override { return this->impl_->Alloc(block); }
    bool Lookup(const std::string& block) override { return this->impl_->Lookup(block); }
    void Commit(const std::string& block, const bool success) override
    {
        this->impl_->Commit(block, success);
    }
    std::list<int32_t> Alloc(const std::list<std::string>& blocks) override
    {
        return this->impl_->Alloc(blocks);
    }
    std::list<bool> Lookup(const std::list<std::string>& blocks) override
    {
        return this->impl_->Lookup(blocks);
    }
    void Commit(const std::list<std::string>& blocks, const bool success) override
    {
        this->impl_->Commit(blocks, success);
    }
    size_t Submit(Task&& task) override { return this->impl_->Submit(std::move(task)); }
    int32_t Wait(const size_t task) override { return this->impl_->Wait(task); }
    int32_t Check(const size_t task, bool& finish) override
    {
        return this->impl_->Check(task, finish);
    }

private:
    LocalStore* impl_;
};

} // namespace UC

#endif
