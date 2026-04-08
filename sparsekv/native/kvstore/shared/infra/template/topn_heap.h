#ifndef UNIFIEDCACHE_INFRA_TOP_N_HEAP_H
#define UNIFIEDCACHE_INFRA_TOP_N_HEAP_H

#include <algorithm>
#include <cstdint>
#include <functional>

namespace UC {

template <typename T, typename Compare = std::less<T>>
class TopNHeap {
public:
    using ValueType = T;
    using SizeType = uint32_t;
    using ConstRef = const T&;

private:
    using IndexType = uint32_t;
    std::vector<ValueType> val_{};
    std::vector<IndexType> idx_{};
    SizeType capacity_{0};
    SizeType size_{0};
    Compare cmp_{};

public:
    explicit TopNHeap(const SizeType capacity) noexcept(
        std::is_nothrow_default_constructible_v<Compare>)
    : capacity_{capacity} {
        val_.resize(capacity);
        idx_.resize(capacity);
    }
    TopNHeap(const TopNHeap&) = delete;
    TopNHeap(const TopNHeap&&) = delete;
    TopNHeap& operator=(const TopNHeap&) = delete;
    TopNHeap& operator=(const TopNHeap&&) = delete;
    ~TopNHeap() { Clear(); }
    
    SizeType Size() const noexcept { return size_; }
    SizeType Capacity() const noexcept { return capacity_; }
    bool Empty() const noexcept { return size_ == 0; }

    void Push(ConstRef value) noexcept {
        if (size_ < capacity_) {
            val_[size_] = value;
            idx_[size_] = size_;
            SiftUp(size_);
            size_++;
            return;
        }
        if (cmp_(val_[idx_.front()], value)) {
            val_[idx_.front()] = value;
            SiftDown(0);
        }
    }
    ConstRef Top() const noexcept { return val_[idx_.front()]; }
    void Pop() noexcept {
        idx_[0] = idx_[--size_];
        if (size_) { SiftDown(0); }
    }
    void Clear() noexcept { size_ = 0; }
private:
    static IndexType Parent(IndexType i) noexcept { return (i - 1) / 2; }
    static IndexType Left(IndexType i) noexcept { return 2 * i + 1; }
    static IndexType Right(IndexType i) noexcept { return 2 * i + 2; }
    void SiftUp(IndexType i) noexcept {
        auto pos = i;
        while (pos > 0) {
            auto p = Parent(pos);
            if (!cmp_(val_[idx_[pos]], val_[idx_[p]])) { break; }
            std::swap(idx_[pos], idx_[p]);
            pos = p;
        }
    }
    void SiftDown(IndexType i) noexcept {
        auto pos = i;
        for (;;) {
            auto l = Left(pos);
            auto r = Right(pos);
            auto best = pos;
            if (l < size_ && cmp_(val_[idx_[l]], val_[idx_[best]])) { best = l; }
            if (r < size_ && cmp_(val_[idx_[r]], val_[idx_[best]])) { best = r; }
            if (best == pos) { break; }
            std::swap(idx_[pos], idx_[best]);
            pos = best;
        }
    }
};

template <typename T, size_t N, typename Compare = std::less<T>>
class TopNFixedHeap : public TopNHeap<T, Compare> {
public:
    TopNFixedHeap() : TopNHeap<T, Compare>{N} {}
};

} // namespace UC

#endif
