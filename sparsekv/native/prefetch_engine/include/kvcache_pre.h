#ifndef ATB_KV_CACHE_PRE_H
#define ATB_KV_CACHE_PRE_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <future>
#include <iostream>
#include <kvcache_log.h>
#include <map>
#include <memory>
#include <mutex>
#include <pybind11/numpy.h>
#include <queue>
#include <shared_mutex>
#include <sstream>
#include <stdarg.h>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <thread>
#include <torch/torch.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

#include "kvstore.h"

namespace bmsaprefetch {

typedef struct {
    int topkLen;
    std::string reqID;
    int layerID;
    int topkIndex;
    int bsIndex;
} PrefetchReqInfo;

class ThreadPool {
public:
    static ThreadPool* GetInst()
    {
        static ThreadPool pool(8);
        return &pool;
    }

    ~ThreadPool();

    template <class F, class... Args>
    auto Enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;

    size_t GetActiveThreads() const;

private:
    explicit ThreadPool(size_t threadCount);
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    mutable std::mutex queueMutex;
    bool stop;
    std::condition_variable condition;
    std::atomic<size_t> activeThreads{0};
    size_t maxThreads;
};

void MutliBSThreadFun(void* args);

// ============ 无锁映射表（使用原子操作+RCU）============
struct BlockMapping {
    std::unordered_map<int, int> docsTable;
    std::unordered_map<int, int> blocksMap;
    mutable std::shared_mutex mutex;

    BlockMapping() {
        docsTable.reserve(2048);
        blocksMap.reserve(2048);
    }

    BlockMapping(const BlockMapping&) = delete;
    BlockMapping& operator=(const BlockMapping&) = delete;

    void batch_init(const std::map<int, int>& initMap) {
        std::unique_lock<std::shared_mutex> lock(mutex);
        for (const auto& kv : initMap) {
            docsTable[kv.first] = kv.second;
            blocksMap[kv.second] = kv.first;
        }
    }

    void batch_update(const std::vector<std::pair<int, int>>& updates,
                     const std::vector<int>& erases) {
        std::unique_lock<std::shared_mutex> lock(mutex);

        for (int idx : erases) {
            auto it = docsTable.find(idx);
            if (it != docsTable.end()) {
                blocksMap.erase(it->second);
                docsTable.erase(it);
            }
        }

        for (const auto& kv : updates) {
            auto oldIt = docsTable.find(kv.first);
            if (oldIt != docsTable.end()) {
                blocksMap.erase(oldIt->second);
            }
            docsTable[kv.first] = kv.second;
            blocksMap[kv.second] = kv.first;
        }
    }

    // 优化：一次性拷贝所有数据（减少锁持有时间）
    std::unordered_map<int, int> get_snapshot() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return docsTable;  // 拷贝构造
    }

    std::map<int, int> get_docs_map() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return std::map<int, int>(docsTable.begin(), docsTable.end());
    }

    std::map<int, int> get_blocks_map() const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        return std::map<int, int>(blocksMap.begin(), blocksMap.end());
    }

    void get_free_blocks(int promptLen,
                        const std::unordered_set<int>& excludeBlocks,
                        std::vector<int>& freeBlocks) const {
        std::shared_lock<std::shared_mutex> lock(mutex);
        for (const auto& kv : docsTable) {
            if (kv.first >= promptLen) {
                continue;
            }
            if (excludeBlocks.find(kv.second) == excludeBlocks.end()) {
                freeBlocks.push_back(kv.second);
            }
        }
    }
};

struct RequestMappings {
    std::unique_ptr<BlockMapping[]> layers;
    std::vector<std::string> blockHashes;
    int promptLen;
    int layerNum;

    RequestMappings() = default;
    explicit RequestMappings(int num) : layerNum(num), promptLen(0) {
        layers.reset(new BlockMapping[num]);
    }

    RequestMappings(const RequestMappings&) = delete;
    RequestMappings& operator=(const RequestMappings&) = delete;
};

// ============ 线程本地上下文（避免动态分配）============
struct ThreadLocalContext {
    std::vector<int64_t> indices;
    std::vector<int> missIdxs;
    std::vector<int> freeBlocks;
    std::unordered_set<int> hitBlocks;
    std::vector<std::pair<int, int>> hitBlocksIdx;
    std::vector<std::pair<int, int>> updates;

    ThreadLocalContext() {
        indices.reserve(512);
        missIdxs.reserve(256);
        freeBlocks.reserve(256);
        hitBlocks.reserve(256);
        hitBlocksIdx.reserve(256);
        updates.reserve(256);
    }

    void reset() {
        indices.clear();
        missIdxs.clear();
        freeBlocks.clear();
        hitBlocks.clear();
        hitBlocksIdx.clear();
        updates.clear();
    }
};

class __attribute__((visibility("hidden"))) BMSAPrefetchEngineC {
private:
    mutable std::shared_mutex mGlobalMutex;
    std::unordered_map<std::string, std::unique_ptr<RequestMappings>> mRequestMappings;

    torch::Tensor mLoadSuccessBlocks;
    torch::Tensor mFreeBlock;
    torch::Tensor mFreeBlockLen;
    torch::Tensor mSuccessTableLen;
    torch::Tensor mUseTopkIdxs;

    int mLayerNum;
    int mRank = -1;
    uint32_t mMaxBs = 30;
    uint32_t mMaxTopkLen = 0;
    bool mIsLog = false;
    bool mUseMla = false;
    uint32_t mBlockSize = 128;
    uint32_t mTensorElemSize = 2;
    uint32_t mHeadNum = 40;
    uint32_t mHeadSzie = 128;
    uint32_t mTPSize = 2;
    size_t mKVSzieBytes = 0;
    uint32_t mExtraTopkLen = 16;
    bool mIsPythonLoad = false;

    std::vector<std::string> mReqIdList;
    int* mTopkLenList = nullptr;
    int* mBsIndexList = nullptr;
    uint32_t runBsLen = 0;
    uint32_t mDecodeStep = 0;

    std::atomic<bool> mIsPrefetchDone{true};
    std::unordered_set<std::string> mDelSeqIds;

    UC::CCStore<>* mStore = nullptr;
    std::vector<torch::Tensor> mKvCaches;

    Logger mLogger;
    ThreadPool* mThreadPool;

    std::map<std::string, std::vector<std::vector<int>>> allNeedLoadBlock;
    std::map<std::string, std::vector<std::vector<int>>> allMissIdxs;

    // 新增：预处理缓存（避免重复tensor访问）
    std::vector<int64_t*> mTopkIdxsCache;  // 直接指向tensor数据
    std::vector<int*> mFreeBlockCache;
    std::vector<int*> mLoadSuccessBlocksCache;

    RequestMappings* GetRequestMappingsPtr(const std::string& reqID);

    // 核心：完全并行的prefetch
    void RunOneBsPrefetchParallel(std::string reqID, int topkLen, int bsIndex,
                                  int topkIndex, RequestMappings* reqMappings);

    void UpdateLayerResults(int layerID, int bsIndex, const std::string& reqID,
                           const std::vector<std::pair<int, int>>& hitBlocksIdx,
                           const std::unordered_set<int>& hitBlocks,
                           int promptLen);

public:
    std::mutex mMutex;
    std::atomic<bool> mStopPrefetch{false};

    ~BMSAPrefetchEngineC();

    BMSAPrefetchEngineC(torch::Tensor& freeBlock, torch::Tensor& loadSuccessBlocks,
                        torch::Tensor& freeBlockLen, torch::Tensor& successTableLen,
                        std::vector<uint32_t>& kvShape, bool useMla, bool isLog,
                        int tpSize, int rank, int extraTopkLen, bool isPythonLoad);

    void SetBlocksMap(std::string reqID, std::vector<int>& blockTableList,
                      std::vector<int>& selectIndex, std::vector<std::string>& blocksHash,
                      int maxIdx);

    void SetBlocksMapMultiLayer(std::string reqID,
                               std::vector<std::map<int, int>>& remainMap,
                               std::vector<std::map<int, int>>& prefetchMap,
                               std::vector<std::string>& blocksHash, int maxIdx);

    void AddBlocksMap(std::string reqID, int idx, int blockID);
    void DelBlocksMap(std::string reqID);
    void DelReqIDRun();

    void SetBlockTableInfo(torch::Tensor& blockTables, torch::Tensor& blockLengths,
                          torch::Tensor& inputTopkBuf, int step);

    void RunAsyncPrefetchBs(std::vector<std::string>& reqIDsInput,
                           std::vector<int>& topkLensInput,
                           std::vector<int>& bsIndexInput,
                           std::vector<torch::Tensor>& kvCaches, void* storePtr);

    int CallPrefetchProcessFun();

    bool GetPrefetchStatus();
    void SetPrefetchStatus(bool flag);
    void SetModelRunningStatus(bool flag);

    size_t GetOffset(uint32_t layerID, bool isV);
    size_t GetOffsetNew(uint32_t layerID, bool isV);

    std::map<std::string, std::vector<std::vector<int>>> ObtainLoadBlocks();
    std::map<std::string, std::vector<std::vector<int>>> ObtainMissIdxs();
    std::map<std::string, std::vector<std::map<int, int>>> ObtainBlocksMap();
    std::map<std::string, std::vector<std::map<int, int>>> ObtainDocsMap();

    void CheckInputIndex(uint32_t maxLen, uint32_t index);
    void PrintMap(std::string reqID, int i);
};

} // namespace bmsaprefetch

#endif
