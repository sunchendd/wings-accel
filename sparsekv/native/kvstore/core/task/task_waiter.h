#ifndef UNIFIEDCACHE_ITASK_WAITER_H
#define UNIFIEDCACHE_ITASK_WAITER_H

#include "thread/latch.h"

namespace UC {

class TaskWaiter : public Latch {
public:
    TaskWaiter(const size_t expected, const double startTp) : Latch{}
    {
        this->startTp = startTp;
        Set(expected);
    }
    using Latch::Wait;
    virtual bool Wait(const size_t timeoutMs) noexcept { return WaitFor(timeoutMs); }
    virtual bool Finish() noexcept { return Check(); }
};

}  // namespace UC

#endif
