#include "kvcache_pre.h"
#include <kvcache_log.h>
#include <omp.h>
#include <sched.h>
#include <cstdint>

namespace bmsaprefetch {

static constexpr size_t kTaskQueueMultiplier = 2;       // task queue size = maxThreads * multiplier
static constexpr int kMaxRetryCount = 100;              // max retry times for store lookup
static constexpr int kRetryIntervalUs = 50;             // retry interval in microseconds
static constexpr int kFreeBlocksReserveSize = 256;      // initial reserve size for free blocks vector
static constexpr int kKVTensorTopkDim = 2;              // index of topk dimension in KV tensor (sizes()[2])
static constexpr int kKVShapeHeadSizeIdx = 2;           // index of head_size in kvShape vector


ThreadPool::ThreadPool(size_t threadCount) : stop(false), maxThreads(threadCount)
{
    for (size_t i = 0; i < maxThreads; i++) {
        workers.emplace_back([this] {
            while (true) { // NOLINT: intentional infinite loop
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    this->condition.wait(lock,
                                         [this] { return this->stop || !this->tasks.empty(); });

                    if (this->stop && this->tasks.empty()) { return; }

                    task = std::move(this->tasks.front());
                    this->tasks.pop();
                    ++activeThreads;
                }

                task();
                {
                    std::unique_lock<std::mutex> lock(this->queueMutex);
                    --activeThreads;
                    condition.notify_all();
                }
            }
        });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for (std::thread& worker : workers) { worker.join(); }
}

template <class F, class... Args>
auto ThreadPool::Enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));

    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queueMutex);

        condition.wait(lock, [this] {
            return (activeThreads < maxThreads || tasks.size() < maxThreads * kTaskQueueMultiplier);
        });

        if (stop) { throw std::runtime_error("enqueue on stopped ThreadPool"); }

        tasks.emplace([task]() { (*task)(); });
    }
    condition.notify_one();
    return res;
}

size_t ThreadPool::GetActiveThreads() const { return activeThreads; }

void MutliBSThreadFun(void* args)
{
    BMSAPrefetchEngineC* engine = static_cast<BMSAPrefetchEngineC*>(args);
    int ret = engine->CallPrefetchProcessFun();
    engine->mMutex.lock();
    engine->DelReqIDRun();
    engine->mMutex.unlock();
    if (ret == 0) { engine->SetPrefetchStatus(true); }
}

// ============ BMSAPrefetchEngineC实现 ============
BMSAPrefetchEngineC::BMSAPrefetchEngineC(torch::Tensor& freeBlock,
                                         torch::Tensor& loadSuccessBlocks,
                                         torch::Tensor& freeBlockLen,
                                         torch::Tensor& successTableLen,
                                         std::vector<uint32_t>& kvShape,
                                         bool useMla, bool isLog, int tpSize,
                                         int rank, int extraTopkLen, bool isPythonLoad)
    : mLogger("./log/kvcache_pre_log.txt", LogLevel::INFO, isLog)
{
    mLoadSuccessBlocks = loadSuccessBlocks;
    mLayerNum = mLoadSuccessBlocks.sizes()[0];
    mMaxBs = mLoadSuccessBlocks.sizes()[1];
    mMaxTopkLen = mLoadSuccessBlocks.sizes()[kKVTensorTopkDim];
    mFreeBlock = freeBlock;
    mFreeBlockLen = freeBlockLen;
    mSuccessTableLen = successTableLen;
    mIsLog = isLog;

    mBsIndexList = (int*)malloc(sizeof(int) * mMaxBs);
    mTopkLenList = (int*)malloc(sizeof(int) * mMaxBs);
    mIsPrefetchDone = true;
    mThreadPool = ThreadPool::GetInst();
    mUseMla = useMla;
    mHeadSzie = kvShape[kKVShapeHeadSizeIdx];
    mHeadNum = kvShape[1];
    mBlockSize = kvShape[0];
    mTPSize = tpSize;
    mRank = rank;
    mIsPythonLoad = isPythonLoad;
    mExtraTopkLen = extraTopkLen;

    if (mRank != 0) {
        mLogger.SetLevel(LogLevel::WARNING);
        mIsLog = false;
    }

    mLogger.log(LogLevel::INFO,
                "BMSAPrefetchEngineC Init (v4 ultra-parallel) mLayerNum %d mMaxBs %u, "
                "mUseMla %d, mHeadSzie %u, mTPSize %u mBlockSize %u mHeadNum %u\n",
                mLayerNum, mMaxBs, mUseMla, mHeadSzie, mTPSize, mBlockSize, mHeadNum);
}

BMSAPrefetchEngineC::~BMSAPrefetchEngineC()
{
    free(mBsIndexList);
    free(mTopkLenList);
}

RequestMappings* BMSAPrefetchEngineC::GetRequestMappingsPtr(const std::string& reqID)
{
    std::shared_lock<std::shared_mutex> lock(mGlobalMutex);
    auto it = mRequestMappings.find(reqID);
    if (it == mRequestMappings.end()) {
        return nullptr;
    }
    return it->second.get();
}

void BMSAPrefetchEngineC::SetBlocksMap(std::string reqID,
                                       std::vector<int>& blockTableList,
                                       std::vector<int>& selectIndex,
                                       std::vector<std::string>& blocksHash,
                                       int maxIdx)
{
    std::unique_lock<std::shared_mutex> lock(mGlobalMutex);

    auto reqMappings = std::make_unique<RequestMappings>(mLayerNum);
    reqMappings->blockHashes = blocksHash;
    reqMappings->promptLen = maxIdx;

    std::map<int, int> initMap;
    for (auto idx : selectIndex) {
        if (idx >= 0 && idx < (int)blockTableList.size()) {
            initMap[idx] = blockTableList[idx];
        }
    }

    for (int i = 0; i < mLayerNum; i++) {
        reqMappings->layers[i].batch_init(initMap);
    }

    mRequestMappings[reqID] = std::move(reqMappings);

    mLogger.log(LogLevel::DEBUG, "SetBlocksMap for reqID: %s, layers: %d, blocks: %zu\n",
                reqID.c_str(), mLayerNum, initMap.size());
}

void BMSAPrefetchEngineC::SetBlocksMapMultiLayer(std::string reqID,
                                                 std::vector<std::map<int, int>>& remainMap,
                                                 std::vector<std::map<int, int>>& prefetchMap,
                                                 std::vector<std::string>& blocksHash,
                                                 int maxIdx)
{
    std::unique_lock<std::shared_mutex> lock(mGlobalMutex);

    auto reqMappings = std::make_unique<RequestMappings>(mLayerNum);
    reqMappings->blockHashes = blocksHash;
    reqMappings->promptLen = maxIdx;

    for (int i = 0; i < mLayerNum; i++) {
        std::map<int, int> combinedMap;

        for (const auto& kv : remainMap[i]) {
            combinedMap[kv.first] = kv.second;
        }
        for (const auto& kv : prefetchMap[i]) {
            combinedMap[kv.first] = kv.second;
        }

        reqMappings->layers[i].batch_init(combinedMap);
    }

    mRequestMappings[reqID] = std::move(reqMappings);
}

void BMSAPrefetchEngineC::AddBlocksMap(std::string reqID, int idx, int blockID)
{
    auto reqMappings = GetRequestMappingsPtr(reqID);
    if (!reqMappings) {
        return;
    }

    std::vector<std::pair<int, int>> updates;
    updates.emplace_back(idx, blockID);

    int numThreads = std::min(16, mLayerNum);
#pragma omp parallel for num_threads(numThreads) schedule(static)
    for (int i = 0; i < mLayerNum; i++) {
        reqMappings->layers[i].batch_update(updates, {});
    }
}

void BMSAPrefetchEngineC::DelBlocksMap(std::string reqID)
{
    std::lock_guard<std::mutex> lg(mMutex);
    mDelSeqIds.insert(reqID);
    if (mIsPrefetchDone.load(std::memory_order_acquire)) {
        DelReqIDRun();
    }
}

void BMSAPrefetchEngineC::DelReqIDRun()
{
    std::unique_lock<std::shared_mutex> lock(mGlobalMutex);

    for (const auto& reqID : mDelSeqIds) {
        auto it = mRequestMappings.find(reqID);
        if (it != mRequestMappings.end()) {
            mRequestMappings.erase(it);
            mLogger.log(LogLevel::INFO, "Del reqID: %s\n", reqID.c_str());
        }
    }
    mDelSeqIds.clear();
}

// ============ 核心：完全并行的Prefetch（模拟v0但线程安全）============
void BMSAPrefetchEngineC::RunOneBsPrefetchParallel(std::string reqID, int topkLen,
                                                   int bsIndex, int topkIndex,
                                                   RequestMappings* reqMappings)
{
    if (!reqMappings) {
        return;
    }

    const int promptLen = reqMappings->promptLen;
    const std::vector<std::string>& blockHashes = reqMappings->blockHashes;

    // 预计算：提取topk indices到栈上（避免重复tensor访问）
    std::vector<int64_t> allIndices(topkLen);
    {
        bool isInt32 = (mUseTopkIdxs.scalar_type() == torch::kInt32);
        for (int j = 0; j < topkLen; j++) {
            if (isInt32) {
                allIndices[j] = mUseTopkIdxs[0][topkIndex][j].item<int32_t>();
            } else {
                allIndices[j] = mUseTopkIdxs[0][topkIndex][j].item<int64_t>();
            }
        }
    }

    // 动态调整线程数（根据layer数量）
    int numThreads = std::min(std::max(8, mLayerNum / 4), 16);

    // ============ 关键优化：OMP并行处理每一层 ============
#pragma omp parallel num_threads(numThreads)
    {
        // 线程本地变量（无竞争）
        ThreadLocalContext ctx;

#pragma omp for schedule(dynamic, 1) nowait
        for (int layerID = 0; layerID < mLayerNum; layerID++) {
            ctx.reset();

            // 阶段1：无锁查找（使用快照）
            std::unordered_map<int, int> snapshot = reqMappings->layers[layerID].get_snapshot();

            for (int j = 0; j < topkLen && ctx.hitBlocks.size() < (topkLen - mExtraTopkLen); j++) {
                int64_t idx = allIndices[j];
                auto it = snapshot.find(idx);
                if (it != snapshot.end()) {
                    ctx.hitBlocks.insert(it->second);
                    ctx.hitBlocksIdx.emplace_back(idx, it->second);
                } else {
                    ctx.missIdxs.push_back(idx);
                }
            }

            // 阶段2：分配free blocks
            int oneFreeBlockLen = mFreeBlockLen[layerID][bsIndex].item<int>();
            int* freeBlockPtr = mFreeBlock[layerID][bsIndex].data_ptr<int>();

            int freeIdx = 0;
            for (size_t i = 0; i < ctx.missIdxs.size() &&
                 ctx.hitBlocks.size() < (topkLen - mExtraTopkLen) &&
                 freeIdx < oneFreeBlockLen; ) {
                int blockID = freeBlockPtr[freeIdx++];
                if (ctx.hitBlocks.find(blockID) != ctx.hitBlocks.end()) {
                    continue;
                }

                ctx.freeBlocks.push_back(blockID);
                ctx.hitBlocks.insert(blockID);
                ctx.hitBlocksIdx.emplace_back(ctx.missIdxs[i], blockID);
                ctx.updates.emplace_back(ctx.missIdxs[i], blockID);
                i++;
            }

            // 阶段3：I/O加载（线程独立执行）
            if (!mIsPythonLoad && !ctx.freeBlocks.empty()) {
                for (size_t i = 0; i < ctx.freeBlocks.size(); i++) {
                    int blockID = ctx.freeBlocks[i];
                    int missIdx = ctx.updates[i].first;

                    // 检查删除标记
                    bool shouldStop = false;
#pragma omp critical
                    {
                        std::lock_guard<std::mutex> lg(mMutex);
                        shouldStop = (mDelSeqIds.find(reqID) != mDelSeqIds.end());
                    }
                    if (shouldStop) break;

                    // 等待模型完成
                    while (mStopPrefetch.load(std::memory_order_acquire)) {
                        std::this_thread::sleep_for(std::chrono::microseconds(1));
                    }

                    if (missIdx >= (int)blockHashes.size()) continue;
                    std::string blockHash = blockHashes[missIdx];

                    // Lookup with retry
                    bool ready = mStore->Lookup(blockHash);
                    if (!ready) {
                        for (int retry = 0; retry < 100; retry++) {
                            std::this_thread::sleep_for(std::chrono::microseconds(50));
                            ready = mStore->Lookup(blockHash);
                            if (ready) break;
                        }
                    }

                    if (!ready) {
                        ctx.hitBlocks.erase(blockID);
                        ctx.hitBlocksIdx.erase(
                            std::remove_if(ctx.hitBlocksIdx.begin(), ctx.hitBlocksIdx.end(),
                                          [missIdx](const std::pair<int, int>& p) {
                                              return p.first == missIdx;
                                          }), ctx.hitBlocksIdx.end());
                        ctx.updates.erase(
                            std::remove_if(ctx.updates.begin(), ctx.updates.end(),
                                          [missIdx](const std::pair<int, int>& p) {
                                              return p.first == missIdx;
                                          }), ctx.updates.end());
                        continue;
                    }

                    // 构造I/O任务
                    UC::Task task{UC::Task::Type::LOAD, UC::Task::Location::DEVICE, "NFS::S2D"};
                    size_t kOffset = GetOffsetNew(layerID, false);
                    size_t vOffset = GetOffsetNew(layerID, true);

                    if (!mUseMla) {
                        task.Append(blockHash, kOffset,
                                   reinterpret_cast<uintptr_t>(
                                       mKvCaches[layerID][0][blockID].data_ptr()),
                                   mKVSzieBytes);
                        task.Append(blockHash, vOffset,
                                   reinterpret_cast<uintptr_t>(
                                       mKvCaches[layerID][1][blockID].data_ptr()),
                                   mKVSzieBytes);
                    } else {
                        task.Append(blockHash, kOffset,
                                   reinterpret_cast<uintptr_t>(
                                       mKvCaches[layerID][blockID].data_ptr()),
                                   mKVSzieBytes);
                    }

                    // 提交并等待（每个线程独立）
                    size_t taskID = mStore->Submit(std::move(task));
                    if (taskID == UC::Task::invalid) {
                        ctx.hitBlocks.erase(blockID);
                        ctx.hitBlocksIdx.erase(
                            std::remove_if(ctx.hitBlocksIdx.begin(), ctx.hitBlocksIdx.end(),
                                          [missIdx](const std::pair<int, int>& p) {
                                              return p.first == missIdx;
                                          }), ctx.hitBlocksIdx.end());
                        ctx.updates.erase(
                            std::remove_if(ctx.updates.begin(), ctx.updates.end(),
                                          [missIdx](const std::pair<int, int>& p) {
                                              return p.first == missIdx;
                                          }), ctx.updates.end());
                        continue;
                    }

                    auto ret = mStore->Wait(taskID);
                    if (ret != 0) {
                        ctx.hitBlocks.erase(blockID);
                        ctx.hitBlocksIdx.erase(
                            std::remove_if(ctx.hitBlocksIdx.begin(), ctx.hitBlocksIdx.end(),
                                          [missIdx](const std::pair<int, int>& p) {
                                              return p.first == missIdx;
                                          }), ctx.hitBlocksIdx.end());
                        ctx.updates.erase(
                            std::remove_if(ctx.updates.begin(), ctx.updates.end(),
                                          [missIdx](const std::pair<int, int>& p) {
                                              return p.first == missIdx;
                                          }), ctx.updates.end());
                    }
                }
            }

            // 阶段4：更新映射（批量）
            if (!ctx.updates.empty()) {
                reqMappings->layers[layerID].batch_update(ctx.updates, {});
            }

            // 阶段5：记录加载信息
#pragma omp critical
            {
                allNeedLoadBlock[reqID][layerID] = ctx.freeBlocks;
                std::vector<int> missVec(ctx.missIdxs.begin(),
                                        ctx.missIdxs.begin() + ctx.freeBlocks.size());
                allMissIdxs[reqID][layerID] = missVec;
            }

            // 阶段6：更新结果tensor
            UpdateLayerResults(layerID, bsIndex, reqID, ctx.hitBlocksIdx,
                             ctx.hitBlocks, promptLen);
        }
    }  // OMP并行区域结束
}

void BMSAPrefetchEngineC::UpdateLayerResults(int layerID, int bsIndex,
                                             const std::string& reqID,
                                             const std::vector<std::pair<int, int>>& hitBlocksIdx,
                                             const std::unordered_set<int>& hitBlocks,
                                             int promptLen)
{
    int successIndex = 0;
    for (const auto& kv : hitBlocksIdx) {
        mLoadSuccessBlocks[layerID][bsIndex][successIndex] = kv.second;
        successIndex += 1;
    }

    RequestMappings* reqMappings = GetRequestMappingsPtr(reqID);
    if (reqMappings) {
        int* freeBlockPtr = mFreeBlock[layerID][bsIndex].data_ptr<int>();
        int oneFreeBlockIndex = 0;

        std::vector<int> freeBlocks;
        freeBlocks.reserve(kFreeBlocksReserveSize);
        reqMappings->layers[layerID].get_free_blocks(promptLen, hitBlocks, freeBlocks);

        for (int blockID : freeBlocks) {
            freeBlockPtr[oneFreeBlockIndex] = blockID;
            oneFreeBlockIndex += 1;
        }

        mFreeBlockLen[layerID][bsIndex] = oneFreeBlockIndex;
    }

    mSuccessTableLen[layerID][bsIndex] = static_cast<int>(hitBlocks.size());
}

void BMSAPrefetchEngineC::RunAsyncPrefetchBs(std::vector<std::string>& reqIDsInput,
                                             std::vector<int>& topkLensInput,
                                             std::vector<int>& bsIndexInput,
                                             std::vector<torch::Tensor>& kvCaches,
                                             void* storePtr)
{
    if (mKVSzieBytes == 0) {
        mTensorElemSize = kvCaches[0].element_size();
        if (mUseMla) {
            mKVSzieBytes = kvCaches[0].element_size() * kvCaches[0][0].numel();
        } else {
            mKVSzieBytes = kvCaches[0].element_size() * kvCaches[0][0][0].numel();
        }

        if (storePtr == nullptr) {
            mLogger.log(LogLevel::ERROR,
                        "Decode step: %u, |KVCache Prefetch| storePtr is nullptr\n",
                        mDecodeStep);
            std::abort();
        }

        mStore = static_cast<UC::CCStore<>*>(storePtr);
        mLogger.log(LogLevel::INFO,
                    "Decode step: %u, |KVCache Prefetch| Initialized: KVSize=%zu bytes, "
                   "ElemSize=%u, Store=%p\n",
                   mDecodeStep, mKVSzieBytes, mTensorElemSize, mStore);
    }

    mKvCaches = kvCaches;
    runBsLen = reqIDsInput.size();

    if (runBsLen > mMaxBs) {
        mLogger.log(LogLevel::ERROR,
                    "Decode step: %u, |KVCache Prefetch| runBsLen %u exceeds maxBs: %d\n",
                   mDecodeStep, runBsLen, mMaxBs);
        std::abort();
    }

    mReqIdList.clear();
    mReqIdList.assign(reqIDsInput.begin(), reqIDsInput.end());
    memcpy(mTopkLenList, topkLensInput.data(), sizeof(int) * runBsLen);
    memcpy(mBsIndexList, bsIndexInput.data(), sizeof(int) * runBsLen);

    mIsPrefetchDone.store(false, std::memory_order_release);

    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Starting async batch size: %zu\n",
                mDecodeStep, reqIDsInput.size());

    if (mIsPythonLoad) {
        MutliBSThreadFun(this);
    } else {
        mThreadPool->Enqueue(MutliBSThreadFun, this);
    }
}

void BMSAPrefetchEngineC::SetBlockTableInfo(torch::Tensor& blockTables,
                                           torch::Tensor& blockLengths,
                                           torch::Tensor& inputTopkBuf, int step)
{
    mLoadSuccessBlocks = blockTables;
    mSuccessTableLen = blockLengths;
    mUseTopkIdxs = inputTopkBuf;
    mDecodeStep = step;
}

int BMSAPrefetchEngineC::CallPrefetchProcessFun()
{
    auto start = std::chrono::high_resolution_clock::now();

    allNeedLoadBlock.clear();
    allMissIdxs.clear();

    for (size_t i = 0; i < runBsLen; i++) {
        if (mTopkLenList[i] <= 0) {
            mLogger.log(LogLevel::WARNING,
                        "Decode step: %u, |KVCache Prefetch| skip reqID: %s, topkLen: %d\n",
                       mDecodeStep, mReqIdList[i].c_str(), mTopkLenList[i]);
            continue;
        }

        allMissIdxs[mReqIdList[i]] = std::vector<std::vector<int>>(mLayerNum);
        allNeedLoadBlock[mReqIdList[i]] = std::vector<std::vector<int>>(mLayerNum);

        // 预先获取指针（避免重复锁）
        RequestMappings* reqMappings = GetRequestMappingsPtr(mReqIdList[i]);

        RunOneBsPrefetchParallel(mReqIdList[i], mTopkLenList[i],
                                mBsIndexList[i], i, reqMappings);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    mLogger.log(LogLevel::INFO,
                "Decode step: %u, |KVCache Prefetch| Completed in %ld us\n",
                mDecodeStep, duration.count());

    return 0;
}

// ============ 工具方法（保持不变）============
bool BMSAPrefetchEngineC::GetPrefetchStatus()
{
    return mIsPrefetchDone.load(std::memory_order_acquire);
}

void BMSAPrefetchEngineC::SetPrefetchStatus(bool flag)
{
    mIsPrefetchDone.store(flag, std::memory_order_release);
}

void BMSAPrefetchEngineC::SetModelRunningStatus(bool flag)
{
    mStopPrefetch.store(flag, std::memory_order_release);
}

size_t BMSAPrefetchEngineC::GetOffsetNew(uint32_t layerID, bool isV)
{
    size_t kMinDataBlockSize =
        static_cast<size_t>(mBlockSize) * mHeadNum * mHeadSzie * mTensorElemSize;
    size_t layerSize = kMinDataBlockSize * 2;
    size_t kOffset = layerSize * layerID;

    if (mUseMla) {
        layerSize = kMinDataBlockSize;
        kOffset = layerSize * layerID;
        return kOffset;
    }

    size_t vOffset = kOffset + kMinDataBlockSize;
    return isV ? vOffset : kOffset;
}

std::map<std::string, std::vector<std::vector<int>>>
BMSAPrefetchEngineC::ObtainLoadBlocks()
{
    return allNeedLoadBlock;
}

std::map<std::string, std::vector<std::vector<int>>>
BMSAPrefetchEngineC::ObtainMissIdxs()
{
    return allMissIdxs;
}

std::map<std::string, std::vector<std::map<int, int>>>
BMSAPrefetchEngineC::ObtainBlocksMap()
{
    std::map<std::string, std::vector<std::map<int, int>>> result;

    std::shared_lock<std::shared_mutex> lock(mGlobalMutex);
    for (const auto& kv : mRequestMappings) {
        std::vector<std::map<int, int>> layers(kv.second->layerNum);
        for (int i = 0; i < kv.second->layerNum; i++) {
            layers[i] = kv.second->layers[i].get_blocks_map();
        }
        result[kv.first] = layers;
    }

    return result;
}

std::map<std::string, std::vector<std::map<int, int>>>
BMSAPrefetchEngineC::ObtainDocsMap()
{
    std::map<std::string, std::vector<std::map<int, int>>> result;

    std::shared_lock<std::shared_mutex> lock(mGlobalMutex);
    for (const auto& kv : mRequestMappings) {
        std::vector<std::map<int, int>> layers(kv.second->layerNum);
        for (int i = 0; i < kv.second->layerNum; i++) {
            layers[i] = kv.second->layers[i].get_docs_map();
        }
        result[kv.first] = layers;
    }

    return result;
}

} // namespace bmsaprefetch
