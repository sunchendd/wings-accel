#ifndef SPARSE_STORE_KVSTORE_H
#define SPARSE_STORE_KVSTORE_H

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>

#include "task/task_shard.h"

namespace UC {

template <class T = Task>
class CCStore {
    using BlockId = std::string;
    using TaskId = size_t;

public:
    virtual ~CCStore() = default;
    virtual int32_t Alloc(const BlockId& block) = 0;
    virtual bool Lookup(const BlockId& block) = 0;
    virtual void Commit(const BlockId& block, const bool success) = 0;
    virtual std::list<int32_t> Alloc(const std::list<BlockId>& blocks) = 0;
    virtual std::list<bool> Lookup(const std::list<BlockId>& blocks) = 0;
    virtual void Commit(const std::list<BlockId>& blocks, const bool success) = 0;
    virtual TaskId Submit(T&& task) = 0;
    virtual int32_t Wait(const TaskId task) = 0;
    virtual int32_t Check(const TaskId task, bool& finish) = 0;
};

}  // namespace UC

#endif
