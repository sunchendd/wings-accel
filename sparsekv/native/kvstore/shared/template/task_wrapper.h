#ifndef UNIFIEDCACHE_STORE_DETAIL_TEMPLATE_TASK_WRAPPER_H
#define UNIFIEDCACHE_STORE_DETAIL_TEMPLATE_TASK_WRAPPER_H

#include <shared_mutex>
#include <unordered_map>
#include "status/status.h"
#include "template/hashset.h"
#include "thread/latch.h"

namespace UC::Detail {

template <typename Task, typename TaskHandle, typename TaskWaiter = Latch>
class TaskWrapper {
protected:
    using TaskPtr = std::shared_ptr<Task>;
    using WaiterPtr = std::shared_ptr<TaskWaiter>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using TaskSet = std::unordered_map<TaskHandle, TaskPair>;
    using TaskIdSet = HashSet<TaskHandle>;
    size_t timeoutMs_;
    TaskIdSet failureSet_;
    TaskSet tasks_{};
    std::shared_mutex mutex_{};
    virtual void Dispatch(TaskPtr t, WaiterPtr w) = 0;

public:
    Expected<TaskHandle> Submit(Task task)
    {
        auto handle = task.id;
        TaskPtr t = nullptr;
        WaiterPtr w = nullptr;
        try {
            t = std::make_shared<Task>(std::move(task));
            w = std::make_shared<TaskWaiter>();
            std::unique_lock<std::shared_mutex> lock(mutex_);
            auto inserted = tasks_.emplace(handle, TaskPair{t, w}).second;
            if (!inserted) [[unlikely]] { return Status::DuplicateKey(); }
        } catch (const std::exception& e) {
            return Status::Error(e.what());
        }
        Dispatch(t, w);
        return handle;
    }
    Expected<bool> Check(TaskHandle taskId)
    {
        WaiterPtr w = nullptr;
        {
            std::shared_lock<std::shared_mutex> lock(mutex_);
            auto iter = tasks_.find(taskId);
            if (iter == tasks_.end()) [[unlikely]] { return Status::NotFound(); }
            w = iter->second.second;
        }
        return w->Check();
    }
    Status Wait(TaskHandle taskId)
    {
        TaskPtr t = nullptr;
        WaiterPtr w = nullptr;
        {
            std::unique_lock<std::shared_mutex> lock(mutex_);
            auto iter = tasks_.find(taskId);
            if (iter == tasks_.end()) [[unlikely]] { return Status::NotFound(); }
            t = iter->second.first;
            w = iter->second.second;
            tasks_.erase(iter);
        }
        auto finished = w->WaitFor(timeoutMs_);
        if (!finished) [[unlikely]] {
            failureSet_.Insert(taskId);
            w->Wait();
            failureSet_.Remove(taskId);
            return Status::Timeout();
        }
        auto failure = failureSet_.Contains(taskId);
        if (failure) [[unlikely]] {
            failureSet_.Remove(taskId);
            return Status::Error();
        }
        return Status::OK();
    }
};

}  // namespace UC::Detail

#endif
