#ifndef UNIFIEDCACHE_INFRA_STOPWATCH_H
#define UNIFIEDCACHE_INFRA_STOPWATCH_H

#include <chrono>

namespace UC {

class StopWatch {
    using clock = std::chrono::steady_clock;
    std::chrono::time_point<clock> startTp_;

public:
    StopWatch() : startTp_{clock::now()} {}
    std::chrono::duration<double> Elapsed() const
    {
        return std::chrono::duration<double>(clock::now() - startTp_);
    }
    std::chrono::milliseconds ElapsedMs() const
    {
        return std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - startTp_);
    }
    void Reset() { startTp_ = clock::now(); }
};

} // namespace UC

#endif
