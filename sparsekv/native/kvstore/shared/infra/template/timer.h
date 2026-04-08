#ifndef UNIFIEDCACHE_INFRA_TIMER_H
#define UNIFIEDCACHE_INFRA_TIMER_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace UC {

template <typename Callable>
class Timer {
public:
    Timer(const std::chrono::seconds& interval, Callable&& callable)
        : interval_(interval), callable_(callable), running_(false)
    {
    }
    ~Timer()
    {
        {
            std::lock_guard<std::mutex> lg(this->mutex_);
            this->running_ = false;
            this->cv_.notify_one();
        }
        if (this->thread_.joinable()) { this->thread_.join(); }
    }
    bool Start()
    {
        {
            std::lock_guard<std::mutex> lg(this->mutex_);
            if (this->running_) { return true; }
        }
        try {
            this->running_ = true;
            this->thread_ = std::thread(&Timer::Runner, this);
            return true;
        } catch (...) {
            return false;
        }
    }

private:
    void Runner()
    {
        while (this->running_) {
            {
                std::unique_lock<std::mutex> lg(this->mutex_);
                this->cv_.wait_for(lg, this->interval_, [this] { return !this->running_; });
                if (!this->running_) { break; }
            }
            this->callable_();
        }
    }

private:
    std::chrono::seconds interval_;
    Callable callable_;
    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_;
};

} // namespace UC

#endif
