#ifndef UNIFIEDCACHE_INFRA_LATCH_H
#define UNIFIEDCACHE_INFRA_LATCH_H

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include "time/now_time.h"

namespace UC {

class Latch {
public:
    Latch() : startTp{NowTime::Now()} {}
    void Set(size_t expected) noexcept { this->counter_.store(expected); }
    void SetEpilog(std::function<void(void)> finish) noexcept { finish_ = std::move(finish); }
    void Up() { ++this->counter_; }
    void Done(std::function<void(void)>&& finish = nullptr) noexcept
    {
        auto counter = this->counter_.load(std::memory_order_acquire);
        while (counter > 0) {
            auto desired = counter - 1;
            if (this->counter_.compare_exchange_weak(counter, desired, std::memory_order_acq_rel)) {
                if (desired == 0) {
                    auto& fn = finish ? finish : finish_;
                    if (fn) { fn(); }
                    std::lock_guard<std::mutex> lg(this->mutex_);
                    this->cv_.notify_all();
                }
                return;
            }
        }
    }
    void Wait()
    {
        std::unique_lock<std::mutex> lk(this->mutex_);
        if (this->counter_ == 0) { return; }
        this->cv_.wait(lk, [this] { return this->counter_ == 0; });
    }
    bool WaitFor(size_t timeoutMs) noexcept
    {
        if (timeoutMs == 0) {
            this->Wait();
            return true;
        }
        std::unique_lock<std::mutex> lk(this->mutex_);
        if (this->counter_ == 0) { return true; }
        auto elapsed = std::chrono::duration<double>(NowTime::Now() - startTp);
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
        auto timeMs = std::chrono::milliseconds(timeoutMs);
        if (timeMs <= elapsedMs) { return false; }
        auto remainMs = timeMs - elapsedMs;
        return this->cv_.wait_for(lk, remainMs, [this] { return this->counter_ == 0; });
    }
    bool Check() noexcept { return this->counter_ == 0; }

public:
    double startTp{0};

protected:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<size_t> counter_{0};
    std::function<void(void)> finish_{nullptr};
};

}  // namespace UC

#endif  // UNIFIEDCACHE_INFRA_LATCH_H
