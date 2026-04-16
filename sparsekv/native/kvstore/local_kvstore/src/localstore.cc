#include "localstore.h"
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include "device/idevice.h"
#include "logger/logger.h"
#include "status/status.h"
#include "task/task_manager.h"

namespace UC {

namespace {

class LocalStoreQueue final : public TaskQueue {
public:
    using ShardList = std::list<Task::Shard>;

    explicit LocalStoreQueue(std::function<void(ShardList&&)> submit) : submit_{std::move(submit)} {}

    void Push(ShardList& shards) noexcept override
    {
        if (shards.empty()) { return; }
        {
            std::lock_guard<std::mutex> lg(mutex_);
            pending_.emplace_back(std::move(shards));
        }
        cv_.notify_one();
    }

    void Run()
    {
        worker_ = std::thread([this] {
            for (;;) {  // loop until stop_ is set
                ShardList shards;
                {
                    std::unique_lock<std::mutex> ul(mutex_);
                    cv_.wait(ul, [this] { return stop_ || !pending_.empty(); });
                    if (stop_ && pending_.empty()) { return; }
                    shards = std::move(pending_.front());
                    pending_.pop_front();
                }
                submit_(std::move(shards));
            }
        });
    }

    void Stop()
    {
        {
            std::lock_guard<std::mutex> lg(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    ~LocalStoreQueue() override { Stop(); }

private:
    std::function<void(ShardList&&)> submit_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::deque<ShardList> pending_;
    bool stop_{false};
    std::thread worker_;
};

class LocalTaskManager final : public TaskManager {
public:
    void Init(std::vector<std::shared_ptr<TaskQueue>> queues, const size_t timeoutMs) noexcept
    {
        this->queues_ = std::move(queues);
        this->timeoutMs_ = timeoutMs;
    }

    Status SubmitTask(Task&& task, size_t& taskId) noexcept { return Submit(std::move(task), taskId); }

    Status WaitTask(const size_t taskId) noexcept { return Wait(taskId); }

    Status CheckTask(const size_t taskId, bool& finish) noexcept { return Check(taskId, finish); }

    void MarkFailure(const size_t taskId) noexcept { this->failureSet_.Insert(taskId); }
};

class LocalStoreWithoutBackendImpl : public LocalStore {
public:
    ~LocalStoreWithoutBackendImpl() override = default;

    int32_t Setup(const size_t ioSize, const size_t capacity, const int32_t deviceId)
    {
        if (ioSize == 0 || capacity == 0) { return Status::InvalidParam().Underlying(); }
        ioSize_ = ioSize;
        capacity_ = capacity;

        device_ = DeviceFactory::Make(deviceId, ioSize_, capacity_);
        if (!device_) {
            UC_ERROR("Failed to make store device.");
            return Status::OutOfMemory().Underlying();
        }
        auto status = device_->Setup();
        if (status.Failure()) { return status.Underlying(); }

        freeBuffers_.reserve(capacity_);
        for (size_t i = 0; i < capacity_; i++) {
            auto buffer = device_->GetBuffer(ioSize_);
            if (!buffer) { return Status::OutOfMemory().Underlying(); }
            freeBuffers_.push_back(std::move(buffer));
        }

        auto submit = [this](LocalStoreQueue::ShardList&& shards) { this->SubmitShardList(shards); };
        queue_ = std::make_shared<LocalStoreQueue>(std::move(submit));
        taskManager_.Init({queue_}, 0);
        queue_->Run();

        return Status::OK().Underlying();
    }

    int32_t Alloc(const std::string& block) override
    {
        std::lock_guard<std::mutex> lg(blocksMutex_);
        auto it = blocks_.find(block);
        if (it != blocks_.end()) { return Status::OK().Underlying(); }
        if (freeBuffers_.empty()) { return Status::OutOfMemory().Underlying(); }
        auto buffer = std::move(freeBuffers_.back());
        freeBuffers_.pop_back();
        blocks_.emplace(block, BlockEntry{std::move(buffer), false});
        return Status::OK().Underlying();
    }

    bool Lookup(const std::string& block) override
    {
        std::lock_guard<std::mutex> lg(blocksMutex_);
        auto it = blocks_.find(block);
        return it != blocks_.end() && it->second.ready;
    }

    void Commit(const std::string& block, const bool success) override
    {
        std::lock_guard<std::mutex> lg(blocksMutex_);
        auto it = blocks_.find(block);
        if (it == blocks_.end()) { return; }
        if (success) {
            it->second.ready = true;
            return;
        }
        freeBuffers_.push_back(std::move(it->second.buffer));
        blocks_.erase(it);
    }

    std::list<int32_t> Alloc(const std::list<std::string>& blocks) override
    {
        std::list<int32_t> results;
        results.resize(blocks.size(), Status::OK().Underlying());
        auto rit = results.begin();
        for (const auto& block : blocks) {
            *rit = this->Alloc(block);
            ++rit;
        }
        return results;
    }

    std::list<bool> Lookup(const std::list<std::string>& blocks) override
    {
        std::list<bool> results;
        for (const auto& block : blocks) { results.push_back(this->Lookup(block)); }
        return results;
    }

    void Commit(const std::list<std::string>& blocks, const bool success) override
    {
        for (const auto& block : blocks) { this->Commit(block, success); }
    }

    size_t Submit(Task&& task) override
    {
        size_t taskId = Task::invalid;
        auto status = taskManager_.SubmitTask(std::move(task), taskId);
        if (status.Failure()) { return Task::invalid; }
        return taskId;
    }

    int32_t Wait(const size_t task) override { return taskManager_.WaitTask(task).Underlying(); }

    int32_t Check(const size_t task, bool& finish) override
    {
        return taskManager_.CheckTask(task, finish).Underlying();
    }

private:
    struct BlockEntry {
        std::shared_ptr<std::byte> buffer;
        bool ready{false};
    };

    Status GetBlockPtr(const std::string& block, const size_t offset, const size_t length,
                       std::byte*& outPtr, const bool allowAlloc)
    {
        if (offset + length > ioSize_) { return Status::InvalidParam(); }
        std::lock_guard<std::mutex> lg(blocksMutex_);
        auto it = blocks_.find(block);
        if (it == blocks_.end()) {
            if (!allowAlloc) { return Status::NotFound(); }
            if (freeBuffers_.empty()) { return Status::OutOfMemory(); }
            auto buffer = std::move(freeBuffers_.back());
            freeBuffers_.pop_back();
            auto [iter, _] = blocks_.emplace(block, BlockEntry{std::move(buffer), false});
            it = iter;
        }
        if (!allowAlloc && !it->second.ready) { return Status::NotFound(); }
        outPtr = it->second.buffer.get() + offset;
        return Status::OK();
    }

    void SubmitShardList(LocalStoreQueue::ShardList& shards)
    {
        if (shards.empty()) { return; }

        auto done = shards.back().done;
        const auto owner = shards.back().owner;
        bool hasDeviceOp = false;

        for (auto& shard : shards) {
            if (shard.length == 0) {
                taskManager_.MarkFailure(owner);
                if (done) { done(); }
                return;
            }

            std::byte* storePtr = nullptr;
            const bool allowAlloc = (shard.type == Task::Type::DUMP);
            auto status = GetBlockPtr(shard.block, shard.offset, shard.length, storePtr, allowAlloc);
            if (status.Failure()) {
                taskManager_.MarkFailure(owner);
                if (done) { done(); }
                return;
            }

            if (shard.location == Task::Location::DEVICE) {
                hasDeviceOp = true;
                if (shard.type == Task::Type::LOAD) {
                    status = device_->H2DAsync(reinterpret_cast<std::byte*>(shard.address), storePtr,
                                               shard.length);
                } else {
                    status =
                        device_->D2HAsync(storePtr, reinterpret_cast<std::byte*>(shard.address),
                                          shard.length);
                }
                if (status.Failure()) {
                    taskManager_.MarkFailure(owner);
                    if (done) { done(); }
                    return;
                }
            } else {
                auto* hostPtr = reinterpret_cast<std::byte*>(shard.address);
                if (shard.type == Task::Type::LOAD) {
                    std::memcpy(hostPtr, storePtr, shard.length);
                } else {
                    std::memcpy(storePtr, hostPtr, shard.length);
                }
            }
        }

        if (!hasDeviceOp) {
            if (done) { done(); }
            return;
        }

        auto cb = [this, owner, doneFn = done](bool ok) {
            if (!ok) { this->taskManager_.MarkFailure(owner); }
            if (doneFn) { doneFn(); }
        };
        auto status = device_->AppendCallback(std::move(cb));
        if (status.Failure()) {
            taskManager_.MarkFailure(owner);
            if (done) { done(); }
            return;
        }
    }

private:
    LocalTaskManager taskManager_;
    size_t ioSize_{0};
    size_t capacity_{0};
    std::unique_ptr<IDevice> device_{nullptr};

    std::mutex blocksMutex_;
    std::unordered_map<std::string, BlockEntry> blocks_;
    std::vector<std::shared_ptr<std::byte>> freeBuffers_;

    std::shared_ptr<LocalStoreQueue> queue_{nullptr};
};

class LocalStoreWithBackendImpl : public LocalStore {
public:
    int32_t Setup(const size_t ioSize, const size_t capacity, void* backend, const int32_t deviceId)
    {
        (void)ioSize;
        (void)capacity;
        (void)deviceId;
        if (!backend) { return Status::InvalidParam().Underlying(); }
        backend_ = static_cast<UC::CCStore<>*>(backend);
        return Status::OK().Underlying();
    }

    int32_t Alloc(const std::string& block) override { return backend_->Alloc(block); }
    bool Lookup(const std::string& block) override { return backend_->Lookup(block); }
    void Commit(const std::string& block, const bool success) override { backend_->Commit(block, success); }
    std::list<int32_t> Alloc(const std::list<std::string>& blocks) override { return backend_->Alloc(blocks); }
    std::list<bool> Lookup(const std::list<std::string>& blocks) override { return backend_->Lookup(blocks); }
    void Commit(const std::list<std::string>& blocks, const bool success) override
    {
        backend_->Commit(blocks, success);
    }
    size_t Submit(Task&& task) override { return backend_->Submit(std::move(task)); }
    int32_t Wait(const size_t task) override { return backend_->Wait(task); }
    int32_t Check(const size_t task, bool& finish) override { return backend_->Check(task, finish); }

private:
    UC::CCStore<>* backend_{nullptr};
};

} // namespace

int32_t LocalStore::Setup(const Config& config)
{
    if (config.backend) {
        auto impl = new (std::nothrow) LocalStoreWithBackendImpl();
        if (!impl) {
            UC_ERROR("Out of memory.");
            return Status::OutOfMemory().Underlying();
        }
        this->impl_ = impl;
        return impl->Setup(config.ioSize, config.capacity, config.backend, config.deviceId);
    }
    auto impl = new (std::nothrow) LocalStoreWithoutBackendImpl();
    if (!impl) {
        UC_ERROR("Out of memory.");
        return Status::OutOfMemory().Underlying();
    }
    this->impl_ = impl;
    return impl->Setup(config.ioSize, config.capacity, config.deviceId);
}

} // namespace UC
