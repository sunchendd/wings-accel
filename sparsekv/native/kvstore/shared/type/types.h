#ifndef UNIFIEDCACHE_STORE_DETAIL_TYPE_TYPES_H
#define UNIFIEDCACHE_STORE_DETAIL_TYPE_TYPES_H

#include <array>
#include <string>
#include <vector>

namespace UC::Detail {

using BlockId = std::array<std::byte, 16>; /* 16-byte block hash */
using TaskHandle = std::size_t;            /* Opaque task token (0 = invalid) */

/**
 * @brief Hasher of BlockId
 */
struct BlockIdHasher {
    size_t operator()(const BlockId& blockId) const noexcept
    {
        std::string_view sv(reinterpret_cast<const char*>(blockId.data()), blockId.size());
        return std::hash<std::string_view>{}(sv);
    }
};

/**
 * @brief Describes one shard (slice) of a block.
 */
struct Shard {
    BlockId owner;            /* Parent block identifier */
    std::size_t index;        /* Shard index inside the block */
    std::vector<void*> addrs; /* Device-side buffer addresses */
};

/**
 * @brief Batch descriptor for load or dump operations.
 *
 * Inherits from std::vector<Shard> to store the shard list and reserves
 * room for future extensions (priority, deadline, etc.).
 */
struct TaskDesc : std::vector<Shard> {
    using vector::vector; /* Inherit all ctors */
    std::string brief;    /* Description of Task */
};

}  // namespace UC::Detail

#endif
