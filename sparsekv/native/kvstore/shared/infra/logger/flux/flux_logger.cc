#include <atomic>
#include <chrono>
#include <condition_variable>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <type_traits>
#include <list>
#include <mutex>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include "logger/logger.h"

namespace UC::Logger {

static constexpr size_t FLUSH_BATCH_SIZE = 1024;
static constexpr auto FlushLatency = std::chrono::milliseconds(10);
static const char* g_levelStrs[] = {"D", "I", "W", "E"};

class FluxLogger : public ILogger {
    using Buffer = std::list<std::string>;
    std::atomic<bool> stop_{false};
    std::chrono::steady_clock::time_point lastFlush_{std::chrono::steady_clock::now()};
    Buffer frontBuf_;
    Buffer backBuf_;
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_;

    FluxLogger() : worker_(&FluxLogger::WorkerLoop, this)
    {
        pthread_atfork(Prepare, ParentPost, ChildPost);
    }
    void WorkerLoop()
    {
        Buffer localBuf;
        while (true) {  // loop until stop_ is set
            {
                std::unique_lock ul(this->mtx_);
                auto triggered = this->cv_.wait_for(ul, FlushLatency, [this] {
                    return this->stop_.load(std::memory_order_relaxed) || !this->backBuf_.empty();
                });
                if (this->stop_.load(std::memory_order_relaxed)) { break; }
                localBuf.splice(localBuf.end(), this->backBuf_);
                if (!triggered) { localBuf.splice(localBuf.end(), this->frontBuf_); }
            }
            if (localBuf.empty()) { continue; }
            for (const auto& s : localBuf) { std::fwrite(s.data(), 1, s.size(), stdout); }
            std::fflush(stdout);
            localBuf.clear();
        }
        while (!this->backBuf_.empty()) {
            localBuf.splice(localBuf.end(), this->backBuf_);
            for (const auto& s : localBuf) { std::fwrite(s.data(), 1, s.size(), stdout); }
            std::fflush(stdout);
            localBuf.clear();
        }
    }
    static void Prepare() { Instance()->mtx_.lock(); }
    static void ParentPost() { Instance()->mtx_.unlock(); }
    static void ChildPost()
    {
        Instance()->mtx_.unlock();
        new (&Instance()->mtx_) std::mutex;
    }

public:
    FluxLogger(const FluxLogger&) = delete;
    FluxLogger& operator=(const FluxLogger&) = delete;
    static FluxLogger* Instance()
    {
        static FluxLogger t;
        return &t;
    }
    ~FluxLogger()
    {
        this->stop_.store(true, std::memory_order_relaxed);
        {
            std::lock_guard lg(this->mtx_);
            this->backBuf_.splice(this->backBuf_.end(), this->frontBuf_);
            this->cv_.notify_one();
        }
        if (this->worker_.joinable()) { this->worker_.join(); }
    }

protected:
    void Log(Level&& lv, SourceLocation&& loc, std::string&& msg) override
    {
        using namespace std::chrono;
        static const auto PID = getpid();
        static thread_local const auto TID = syscall(SYS_gettid);
        static thread_local seconds lastSec{0};
        static thread_local char datetime[32];
        auto systemNow = system_clock::now();
        auto currentSec = time_point_cast<seconds>(systemNow);
        if (lastSec != currentSec.time_since_epoch()) {
            auto systemTime = system_clock::to_time_t(systemNow);
            std::tm systemTm;
            localtime_r(&systemTime, &systemTm);
            fmt::format_to_n(datetime, sizeof(datetime), "{:%F %T}", systemTm);
            lastSec = currentSec.time_since_epoch();
        }
        auto us = duration_cast<microseconds>(systemNow - currentSec).count();
        // use g_levelStrs indexed by log level value
        auto payload = fmt::format("[{}.{:06d}][UC][{}] {} [{},{}][{}:{},{}]\n", datetime, us,
                                   g_levelStrs[static_cast<int>(lv)], msg, PID, TID,
                                   basename(loc.file), loc.line, loc.func);
        auto steadyNow = steady_clock::now();
        std::lock_guard lg(this->mtx_);
        this->frontBuf_.push_back(std::move(payload));
        bool byCount = this->frontBuf_.size() >= FLUSH_BATCH_SIZE;
        bool byTime = steadyNow - this->lastFlush_ >= FlushLatency;
        if (byCount || byTime) {
            this->backBuf_.splice(this->backBuf_.end(), this->frontBuf_);
            this->lastFlush_ = steadyNow;
            this->cv_.notify_one();
        }
    }
};

ILogger* Make() { return FluxLogger::Instance(); }

} // namespace UC::Logger
