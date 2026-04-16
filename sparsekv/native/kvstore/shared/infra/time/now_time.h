#ifndef UNIFIEDCACHE_SHARED_INFRA_TIME_NOW_TIME_H
#define UNIFIEDCACHE_SHARED_INFRA_TIME_NOW_TIME_H

#include <chrono>

namespace UC {

class NowTime {
public:
    static auto Now()
    {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double>(now).count();
    }
};

}  // namespace UC

#endif
