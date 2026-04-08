#ifndef UNIFIEDCACHE_TASK_MANAGER_H
#define UNIFIEDCACHE_TASK_MANAGER_H

#include <unordered_map>
#include "logger/logger.h"
#include "status/status.h"
#include "task_queue.h"
#include "task_set.h"

namespace UC {

class TaskManager {
    using TaskPtr = std::shared_ptr<Task>;
    using WaiterPtr = std::shared_ptr<TaskWaiter>;
    using TaskPair = std::pair<TaskPtr, WaiterPtr>;
    using QueuePtr = std::shared_ptr<TaskQueue>;

public:
    virtual ~TaskManager() = default;
    virtual Status Submit(Task&& task, size_t& taskId) noexcept
    {
        taskId = task.Id();
        const auto taskStr = task.Str();
        TaskPtr taskPtr = nullptr;
        WaiterPtr waiterPtr = nullptr;
        try {
            taskPtr = std::make_shared<Task>(std::move(task));
            waiterPtr = std::make_shared<TaskWaiter>(0, task.StartTp());
        } catch (const std::exception& e) {
            UC_ERROR("Failed({}) to submit task({}).", e.what(), taskStr);
            return Status::OutOfMemory();
        }
        std::lock_guard<std::mutex> lg(mutex_);
        const auto& [iter, success] =
            tasks_.emplace(taskId, std::make_pair(std::move(taskPtr), std::move(waiterPtr)));
        if (!success) {
            UC_ERROR("Failed to submit task({}).", taskStr);
            return Status::OutOfMemory();
        }
        auto shards = iter->second.first->Split(queues_.size(), iter->second.second);
        for (auto& shard : shards) {
            auto& q = queues_[qIndex_++];
            if (qIndex_ == queues_.size()) { qIndex_ = 0; }
            q->Push(shard);
        }
        return Status::OK();
    }
    virtual Status Wait(const size_t taskId) noexcept
    {
        TaskPtr task = nullptr;
        WaiterPtr waiter = nullptr;
        {
            std::lock_guard<std::mutex> lg(mutex_);
            auto iter = tasks_.find(taskId);
            if (iter == tasks_.end()) {
                UC_ERROR("Not found task by id({}).", taskId);
                return Status::NotFound();
            }
            task = iter->second.first;
            waiter = iter->second.second;
            tasks_.erase(iter);
        }
        if (!waiter->Wait(timeoutMs_)) {
            UC_ERROR("Task({}) timeout({}).", task->Str(), timeoutMs_);
            failureSet_.Insert(taskId);
            waiter->Wait();
        }
        auto failure = failureSet_.Contains(taskId);
        if (failure) {
            failureSet_.Remove(taskId);
            UC_ERROR("Task({}) failed.", task->Str());
            return Status::Error();
        }
        return Status::OK();
    }
    virtual Status Check(const size_t taskId, bool& finish) noexcept
    {
        std::lock_guard<std::mutex> lg(mutex_);
        auto iter = tasks_.find(taskId);
        if (iter == tasks_.end()) {
            UC_ERROR("Not found task by id({}).", taskId);
            return Status::NotFound();
        }
        finish = iter->second.second->Finish();
        return Status::OK();
    }

protected:
    std::mutex mutex_;
    std::unordered_map<size_t, TaskPair> tasks_;
    size_t qIndex_{0};
    std::vector<QueuePtr> queues_;
    size_t timeoutMs_{0};
    TaskSet failureSet_;
};

}  // namespace UC

#endif
