#ifndef UNIFIEDCACHE_INFRA_INDEX_POOL_H
#define UNIFIEDCACHE_INFRA_INDEX_POOL_H

#include <atomic>
#include <cstddef>
#include <limits>
#include <vector>

namespace UC {

class IndexPool {
public:
    using Index = uint32_t;
    static constexpr Index npos = std::numeric_limits<Index>::max();

private:
    struct Node {
        Index idx;
        Index next;
    };
    struct Pointer {
        Index slot;
        uint32_t ver;
    };
    static_assert(sizeof(Pointer) == 8, "Pointer must be 64-bit");

public:
    void Setup(const Index capacity) noexcept
    {
        this->capacity_ = capacity;
        this->nodes_.resize(capacity + 1);
        for (Index slot = 1; slot <= capacity; slot++) {
            this->nodes_[slot].idx = slot - 1;
            this->nodes_[slot].next = slot + 1;
        }
        this->nodes_[capacity].next = 0;
        this->pointer_.store({1, 0});
    }
    Index Acquire() noexcept
    {
        for (;;) {
            auto ptr = this->pointer_.load(std::memory_order_acquire);
            if (ptr.slot == 0) { return npos; }
            auto next = this->nodes_[ptr.slot].next;
            Pointer desired{next, ptr.ver + 1};
            if (this->pointer_.compare_exchange_weak(ptr, desired, std::memory_order_release,
                                                     std::memory_order_relaxed)) {
                return this->nodes_[ptr.slot].idx;
            }
        }
    }
    void Release(const Index idx) noexcept
    {
        if (idx >= this->capacity_) { return; }
        auto slot = idx + 1;
        for (;;) {
            auto ptr = this->pointer_.load(std::memory_order_acquire);
            this->nodes_[slot].next = ptr.slot;
            Pointer desired{slot, ptr.ver + 1};
            if (this->pointer_.compare_exchange_weak(ptr, desired, std::memory_order_release,
                                                     std::memory_order_relaxed)) {
                return;
            }
        }
    }

private:
    Index capacity_;
    std::vector<Node> nodes_;
    alignas(64) std::atomic<Pointer> pointer_;
};

} // namespace UC

#endif
