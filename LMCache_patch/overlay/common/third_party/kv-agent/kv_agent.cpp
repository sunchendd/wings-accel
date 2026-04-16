#include <cstdio>
#include <cstdlib> 
#include <torch/extension.h>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <stdint.h>
#include <immintrin.h>
#include <sys/time.h>
#include <numa.h>
#include <unistd.h>
#include <string>
#include <sched.h>
#include "./qzip/qzip.h"
#include <chrono>
#include <iomanip>
#include <sstream>
#include <thread>
#include <mutex>
#include <list>
#include <atomic>
#include <cstring>

// 异步释放内存相关变量
static std::list<char*> pending_releases;
static std::mutex release_mutex;
static std::atomic<bool> release_thread_running(true);
static std::thread release_thread;

// Global configuration parameters
static int log_enabled = 0;
static int mantissa_loss_level = 0;
static int qat_instance_num = 4;
static std::string kv_data_dir_path = "./kv_cache_compressed";
static bool config_initialized = false;

//#define OMP_DEBUG
static uint64_t total_compressed_size, total_uncompressed_size;
static __thread char *compressed_buf;
static __thread char *decompressed_buf;
static __thread int buf_pos;
static __thread int input_buf_size;
static __thread int output_buf_size;
static __thread cpu_set_t saved_cpu_mask;
static cpu_set_t all_cpu_mask;

#define JUDGE_MASK_FOR_LOSS_LEVEL_5 (0x60)
#define JUDGE_MASK_FOR_LOSS_LEVEL_4 (0x70)
#define JUDGE_MASK_FOR_LOSS_LEVEL_3 (0x78)
#define JUDGE_MASK_FOR_LOSS_LEVEL_2 (0x7C)
#define JUDGE_MASK_FOR_LOSS_LEVEL_1 (0x7E)

#define CLEAR_MASK_FOR_LOSS_LEVEL_5 (0xE0)
#define CLEAR_MASK_FOR_LOSS_LEVEL_4 (0xF0)
#define CLEAR_MASK_FOR_LOSS_LEVEL_3 (0xF8)
#define CLEAR_MASK_FOR_LOSS_LEVEL_2 (0xFC)
#define CLEAR_MASK_FOR_LOSS_LEVEL_1 (0xFE)

// 日志系统
static std::string current_time() {
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto now_time = std::localtime(&now_time_t);
    auto now_duration = now - std::chrono::system_clock::from_time_t(now_time_t);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now_duration).count() % 1000;

    std::ostringstream oss;
    oss << std::put_time(now_time, "%Y-%m-%d %H:%M:%S") << ',' 
        << std::setfill('0') << std::setw(3) << ms;
    return oss.str();
}

static const char* short_file_name(const char* path) {
    const char* slash = std::strrchr(path, '/');
    return slash ? slash + 1 : path;
}

// 修改后的 LOG 宏定义，只在 INFO 日志中判断 log_enabled
#define LOG(level, message) do { \
    std::ostream& log_stream = (level == std::string("INFO")) ? std::cout : std::cerr; \
    if (level == std::string("INFO") && !log_enabled) { \
        break; /* 使用break代替continue */ \
    } \
    log_stream << "[" << #level << "] [" << current_time() << "] [" << short_file_name(__FILE__) << ":" << __LINE__ << "] " << message << std::endl; \
} while (0)

#define LOG_INFO(message) LOG("INFO", message)
#define LOG_WARN(message) LOG("WARNING", message)
#define LOG_ERROR(message) LOG("ERROR", message)

#define LOG_INFO_ALWAYS(message) do { \
    std::cout << "[INFO] [" << current_time() << "] [" << short_file_name(__FILE__) << ":" << __LINE__ << "] " << message << std::endl; \
} while (0)

// 新增内存释放线程函数（添加日志）
static void release_thread_function() {
    LOG_INFO_ALWAYS("KV Release thread started");
    
    while (release_thread_running) {
        // 每5秒检查一次
        std::this_thread::sleep_for(std::chrono::seconds(5));
        
        std::list<char*> to_release;
        {
            std::lock_guard<std::mutex> lock(release_mutex);
            if (!pending_releases.empty()) {
                LOG_INFO("KV Release thread: Collecting " + std::to_string(pending_releases.size()) + " memory blocks for release");
                to_release.swap(pending_releases);
            }
        }
        
        // 异步释放内存（避免阻塞主线程）
        for (auto ptr : to_release) {
            std::free(ptr);
        }
        
        if (!to_release.empty()) {
            LOG_INFO("KV Release thread: Released " + std::to_string(to_release.size()) + " memory blocks");
        }
    }
    
    LOG_INFO_ALWAYS("KV Release thread stopped");
    
    // 最后一次清理
    std::list<char*> to_release;
    {
        std::lock_guard<std::mutex> lock(release_mutex);
        if (!pending_releases.empty()) {
            LOG_INFO_ALWAYS("KV Release thread: Final release of " + std::to_string(pending_releases.size()) + " memory blocks");
            to_release.swap(pending_releases);
        }
    }
    
    for (auto ptr : to_release) {
        std::free(ptr);
    }
}

// 新增线程安全的内存块添加函数
static void add_pending_release(char* ptr) {
    if (ptr == nullptr) return;

    std::lock_guard<std::mutex> lock(release_mutex);
    pending_releases.push_back(ptr);
}

// 修改thread_cleanup函数
static void thread_cleanup() {
    if (compressed_buf) {
        add_pending_release(compressed_buf);
        compressed_buf = nullptr;
    }
    if (decompressed_buf) {
        add_pending_release(decompressed_buf);
        decompressed_buf = nullptr;
    }
}

// Configuration initialization function
void init_config() {
    if (!config_initialized) {
        LOG_WARN("Using default configuration parameters.");
        log_enabled = 0;
        mantissa_loss_level = 0;
        qat_instance_num = 4;
        kv_data_dir_path = "./kv_cache_compressed";
        config_initialized = true;
    }
}

// Python binding configuration functions
void set_log_enabled(int value) {
    log_enabled = value;
    config_initialized = true;

    LOG_INFO_ALWAYS("Configuration: Log enabled set to " + std::to_string(log_enabled));
}

void start_mem_release_thread() {
    // 启动内存释放线程
    release_thread = std::thread(release_thread_function);

    // 设置线程名称
    pthread_setname_np(release_thread.native_handle(), "KV_Release_Thread");

    // 确保线程不再被 join 或 detach，避免资源泄漏或崩溃
    if (release_thread.joinable()) {
        release_thread.detach();
    }

    LOG_INFO_ALWAYS("KV release thread started.");
}

void set_mantissa_loss_level(int value) {
    if (value >= 0 && value <= 5) {
        mantissa_loss_level = value;
        config_initialized = true;
    } else {
        LOG_WARN("Invalid loss level " + std::to_string(value) + ", using default 0");
        mantissa_loss_level = 0;
    }
    LOG_INFO_ALWAYS("Configuration: Mantissa loss level set to " + std::to_string(mantissa_loss_level));
}

void set_qat_instance_num(int value) {
    if (value > 0) {
        qat_instance_num = value;
        config_initialized = true;
    } else {
        LOG_WARN("Invalid QAT instance num " + std::to_string(value) + ", using default 4");
        qat_instance_num = 4;
    }
    LOG_INFO_ALWAYS("Configuration: QAT instance num set to " + std::to_string(qat_instance_num));

    start_mem_release_thread();
}

void set_kv_data_dir(const std::string& path) {
    kv_data_dir_path = path;
    config_initialized = true;
    LOG_INFO_ALWAYS("Configuration: KV data directory set to " + kv_data_dir_path);
}

// Ensure configuration is initialized before use
void ensure_config_initialized() {
    if (!config_initialized) {
        init_config();
    }
}

static std::string kv_data_dir(void)
{
    ensure_config_initialized();
    return kv_data_dir_path;
}

static void dump_affinity(void)
{
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) == -1) {
        std::cerr << "sched_getaffinity error!" << std::endl;
        std::abort();
    }
    for (int i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &mask)) {
            std::cout << i << " ";
        }
    }
    std::cout << std::endl;
}

static void __save_affinity(void)
{
    CPU_ZERO(&saved_cpu_mask);
    if (sched_getaffinity(0, sizeof(saved_cpu_mask), &saved_cpu_mask) == -1) {
        std::cerr << "sched_getaffinity error!" << std::endl;
        std::abort();
    }
    if (log_enabled) {
        std::cout << "Saved affinity:";
        dump_affinity();
    }

    sched_setaffinity(0, sizeof(all_cpu_mask), &all_cpu_mask);
    if (log_enabled) {
        std::cout << "new affinity:";
        dump_affinity();
    }
}

static void save_affinity(void)
{
    static int init;

    if (!init) {
        int current_cpu = sched_getcpu();
        if (current_cpu == -1) {
            std::cerr << "sched_getcpu  error!" << std::endl;
            std::abort();
        }
        int current_node = numa_node_of_cpu(current_cpu);
        if (current_node == -1) {
            std::cerr << "numa_node_of_cpu error!" << std::endl;
            std::abort();
        }
        struct bitmask *cpumask = numa_allocate_cpumask();
        if (!cpumask) {
            std::cerr << "numa_allocate_cpumask error!" << std::endl;
            std::abort();
        }
        if (numa_node_to_cpus(current_node, cpumask) != 0) {
            std::cerr << "numa_node_to_cpus error!" << std::endl;
            std::abort();
        }
        size_t num_bits = numa_bitmask_nbytes(cpumask) * 8;
        CPU_ZERO(&all_cpu_mask);
        for (unsigned int cpu = 0; cpu < num_bits; cpu++) {
            if (numa_bitmask_isbitset(cpumask, cpu)) {
                CPU_SET(cpu, &all_cpu_mask);
            }
        }
        numa_free_cpumask(cpumask);
        init = 1;
    }

    #pragma omp parallel for schedule(static, 1) num_threads(qat_instance_num)
    for (int i = 0; i < qat_instance_num; i++)
    #ifdef OMP_DEBUG
    #pragma omp critical
    #endif
    {
        __save_affinity();
    }
}

static void __restore_affinity(void)
{
    if (sched_setaffinity(0, sizeof(saved_cpu_mask), &saved_cpu_mask) == -1) {
        perror("restore_affinity, sched_setaffinity failed");
    }
    if (log_enabled) {
        std::cout << "restored affinity:";
        dump_affinity();
    }
}

static void restore_affinity(void)
{
    #pragma omp parallel for schedule(static, 1) num_threads(qat_instance_num)
    for (int i = 0; i < qat_instance_num; i++)
    #ifdef OMP_DEBUG
    #pragma omp critical
    #endif
    {
        __restore_affinity();
    }
}

static void reorganize_kv_cache_block(char *high_bytes, char *low_bytes, char *block_addr, int64_t size)
{
    uint8_t *src = (uint8_t *)block_addr;
    uint8_t* high = (uint8_t*)high_bytes;
    uint8_t* low = (uint8_t*)low_bytes;
    unsigned char tmp_nat_byte;

    for (int64_t index = 0; index < size; index += 2) {
        *high =  *(src + index + 1);
        tmp_nat_byte = *((unsigned char *)(src + index));
        if (mantissa_loss_level) {
            switch (mantissa_loss_level) {
                case 5:
                    /* make sure the highlist 2 bits of mantissa are non-zero, not include the sign bit */
                    if ((tmp_nat_byte & JUDGE_MASK_FOR_LOSS_LEVEL_5) == JUDGE_MASK_FOR_LOSS_LEVEL_5) {
                        tmp_nat_byte = tmp_nat_byte & CLEAR_MASK_FOR_LOSS_LEVEL_5;
                        break;
                    }
                    /* if set loss level is high but data only meet with lower level also go through */
                case 4:
                    if (tmp_nat_byte & JUDGE_MASK_FOR_LOSS_LEVEL_4) {
                        tmp_nat_byte = tmp_nat_byte & CLEAR_MASK_FOR_LOSS_LEVEL_4;
                        break;
                    }
                    /* if set loss level is high but data only meet with lower level also go through */
                case 3:
                    if (tmp_nat_byte & JUDGE_MASK_FOR_LOSS_LEVEL_3) {
                        tmp_nat_byte = tmp_nat_byte & CLEAR_MASK_FOR_LOSS_LEVEL_3;
                        break;
                    }
                    /* if set loss level is high but data only meet with lower level also go through */
                case 2:
                    if (tmp_nat_byte & JUDGE_MASK_FOR_LOSS_LEVEL_2) {
                        tmp_nat_byte = tmp_nat_byte & CLEAR_MASK_FOR_LOSS_LEVEL_2;
                        break;
                    }
                    /* if set loss level is high but data only meet with lower level also go through */
                case 1:
                    if (tmp_nat_byte & JUDGE_MASK_FOR_LOSS_LEVEL_1) {
                        tmp_nat_byte = tmp_nat_byte & CLEAR_MASK_FOR_LOSS_LEVEL_1;
                    }
                    break;
                default:
                    break;
            }
        }
        *low = tmp_nat_byte;
        ++ high;
        ++ low;
    }
}

static void restore_kv_cache_block(char *high_bytes, char *low_bytes, char *block_addr, int64_t size)
{
    uint8_t* dst = (uint8_t*)block_addr;
    uint8_t* high = (uint8_t*)high_bytes;
    uint8_t* low = (uint8_t*)low_bytes;

    if (size % 64 != 0) {
        std::cerr << "KVCACHE_TIERING_DIR is not set" << std::endl;
        std::abort();
    }

#if 1
    for (int64_t i = 0; i < size / 2; i += 1) {
        ((uint16_t*)dst)[i] = (high[i] << 8) | low[i];
    }
#else
    for (uint64_t i = 0; i < size; i += 32) {
        _mm_prefetch((char*)(high + i + 32), _MM_HINT_T0);
        _mm_prefetch((char*)(low + i + 32), _MM_HINT_T0);
        __m256i low_part = _mm256_loadu_si256((__m256i*)(low + i));
        __m512i low_vec = _mm512_cvtepu8_epi16(low_part);
        __m256i high_part = _mm256_loadu_si256((__m256i*)(high + i));
        __m512i high_vec = _mm512_slli_epi16(_mm512_cvtepu8_epi16(high_part), 8);
        __m512i merged = _mm512_or_si512(high_vec, low_vec);
        _mm512_storeu_si512(dst + i * 2, merged);
    }
#endif
}

static void block_save(uint64_t block_hash, const std::string& path = "")
{
    int ret;
    char *inputs[] = {decompressed_buf};
    char *outputs[] = {compressed_buf};
    int in_sizes[] = {input_buf_size};
    int out_sizes[] = {output_buf_size};

    if (buf_pos != input_buf_size) {
        std::cerr << __func__
            << " buf_pos " << buf_pos
            << " != input_buf_size " << input_buf_size
            << std::endl;
        std::abort();
    }

    std::string file_name = path;
    if (path==kv_data_dir() || path==""){
        std::string file_name = kv_data_dir().append("/kv_");
        file_name.append(std::to_string(block_hash));
        file_name.append(".bin");
    }
    
    if (log_enabled) {
        std::cout << __func__
            << " block_hash=" << block_hash
            << " input_buf_size=" << input_buf_size
            << " output_buf_size=" << output_buf_size
            << " file_name=" << file_name
            << std::endl;
    }

    if (std::filesystem::exists(file_name)) {
        std::cerr << __func__
            << " File already exists: " << file_name
            << std::endl;
        std::abort();
    }

    ret = kv_agent_block_compress(inputs, outputs, in_sizes, out_sizes, 1);
    if (ret) {
        std::cerr << "Compress error: " << ret;
        std::abort();
    }
    total_uncompressed_size += in_sizes[0];
    total_compressed_size += out_sizes[0];

    std::ofstream outfile(file_name, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error opening file for writing: " << file_name << std::endl;
        std::abort();
    }

    outfile.write((char *)out_sizes, sizeof(out_sizes));
    if (!outfile) {
        std::cerr << "Error writing metadata to file: " << file_name << std::endl;
        std::abort();
    }
    outfile.write((char *)outputs[0], out_sizes[0]);
    if (!outfile) {
        std::cerr << "Error writing compressed kv cache to file: " << file_name << std::endl;
        std::abort();
    }
    outfile.close();
}

static int block_load(uint64_t block_hash, const std::string& path = "")
{
    struct timeval start, end;
    int ret;
    char *outputs[] = {decompressed_buf};
    char *inputs[] = {compressed_buf};
    int in_sizes[] = {input_buf_size};
    int out_sizes[] = {output_buf_size};

    std::string file_name = path;
    if (path==kv_data_dir() || path==""){
        std::string file_name = kv_data_dir().append("/kv_");
        file_name.append(std::to_string(block_hash));
        file_name.append(".bin");
    }

    if (log_enabled) {
        std::cout << __func__
            << " block_hash=" << block_hash
            << " input_buf_size=" << input_buf_size
            << " output_buf_size=" << output_buf_size
            << " file_name=" << file_name
            << std::endl;
    }

    std::ifstream infile(file_name, std::ios::binary | std::ios::ate);
    if (!infile) {
        if (log_enabled)
            std::cout << "Error opening file for reading: " << file_name << std::endl;
        return -1;
    }
    infile.seekg(0, std::ios::beg);
    std::ifstream ifd(file_name, std::ios::binary);
    if (!infile.read((char *)in_sizes, sizeof(in_sizes))) {
            std::cerr << "Error reading metadata from file: " << file_name << std::endl;
            std::abort();
    }
    if (!infile.read(inputs[0], in_sizes[0]))
    {
        std::cerr << "Error reading compressed kv cache from file: " << file_name << std::endl;
        std::abort();
    }
    infile.close();
    gettimeofday(&start, NULL);
    ret = kv_agent_block_decompress(inputs, outputs, in_sizes, out_sizes, 1);
    if (ret) {
        std::cerr << "Decompress error: " << ret;
        std::abort();
    }
    gettimeofday(&end, NULL);

    if (log_enabled)
    #ifdef OMP_DEBUG
    #pragma omp critical
    #endif
    {
        long seconds = end.tv_sec - start.tv_sec;
        long microseconds = end.tv_usec - start.tv_usec;
        double elapsed = seconds + microseconds*1e-6;
        std::cout << "Decompress took "
            << elapsed
            << " seconds, in size: "
            << in_sizes[0]
            << " out size: "
            << out_sizes[0]
            << std::endl;
    }
    return 0;
}

static void block_data_put(char *data, int size)
{
    char *hi = decompressed_buf + buf_pos;
    buf_pos += size / 2;
    char *lo = decompressed_buf + buf_pos;
    buf_pos += size / 2;
    reorganize_kv_cache_block(hi, lo, data, size);
}

static void block_data_get(char *data, int size)
{
    char *hi = decompressed_buf + buf_pos;
    buf_pos += size / 2;
    char *lo = decompressed_buf + buf_pos;
    buf_pos += size / 2;
    restore_kv_cache_block(hi, lo, data, size);
}

static void block_buf_prepare(int size)
{
    if (size < 0) {
        LOG_ERROR("Invalid buffer size: " << size);
        std::abort();
    }
    
    buf_pos = 0;
    input_buf_size = size;

    // 为compressed_buf分配内存
    if (!compressed_buf) {
        compressed_buf = static_cast<char*>(std::aligned_alloc(64, size));
        if (!compressed_buf) {
            LOG_ERROR("alloc compressed_buf failed.");
            std::abort();
        }
        LOG_INFO("alloc compressed_buf success. size=" << size);
    }

    // 为decompressed_buf分配或重新分配内存
    if (!decompressed_buf || output_buf_size != size) {
        if (decompressed_buf) {
            LOG_INFO("realloc " << size << " to decompressed_buf start. old size=" << output_buf_size);
        }
        
        char* new_buf = static_cast<char*>(std::aligned_alloc(64, size));
        if (!new_buf) {
            LOG_ERROR("alloc decompressed_buf failed. size=" << size);
            std::abort();
        }
        
        if (decompressed_buf) {
            std::free(decompressed_buf);
            LOG_INFO("realloc " << size << " to decompressed_buf success. new size=" << size);
        } else {
            LOG_INFO("alloc decompressed_buf success. size=" << size);
        }
        
        decompressed_buf = new_buf;
    }

    output_buf_size = size;
}

/*
shape:
    nv (2, num_blocks, block_size, num_kv_heads, head_size)
    ipex (2, num_blocks, block_size * num_kv_heads * head_size)
*/
static void blocks_transfer(std::vector<torch::Tensor>& blocks_vec, torch::Tensor& blocks_to_transfer, bool to_save, const std::string& path = "")
{
    const int64_t num_blocks = blocks_to_transfer.size(0);
    const int64_t num_layers = blocks_vec.size();
    const int64_t block_size = (blocks_vec[0][0].element_size() * blocks_vec[0][0][0].numel());
    char** k_mems = new char*[num_layers];
    char** v_mems = new char*[num_layers];

    save_affinity();
    for (int64_t layer = 0; layer < num_layers; layer++) {
        k_mems[layer] = static_cast<char*>(blocks_vec[layer][0].data_ptr());
        v_mems[layer] = static_cast<char*>(blocks_vec[layer][1].data_ptr());
    }
    if (log_enabled) {
        std::cout << __func__
            << (to_save ? " save" : " load")
            << " num_blocks=" << num_blocks
            << " num_layers=" << num_layers
            << " element_size=" << blocks_vec[0][0].element_size()
            << " numel=" << blocks_vec[0][0][0].numel()
            << " block_size=" << block_size
            << " k_mems[0]=" << reinterpret_cast<void*>(k_mems[0])
            << " v_mems[0]=" << reinterpret_cast<void*>(v_mems[0])
            << std::endl;
    }
    #pragma omp parallel for schedule(static, 1) num_threads(qat_instance_num)
    for (int64_t i = 0; i < num_blocks; i++) {
        int64_t block_num = blocks_to_transfer[i][0].item<int64_t>();
        uint64_t block_hash = static_cast<uint64_t>(blocks_to_transfer[i][1].item<int64_t>());
        int64_t block_off = block_num * block_size;
        int64_t block_status = 0;
        int ret;

        block_buf_prepare(block_size * num_layers * 2);
        if (to_save) {
            for (int64_t layer = 0; layer < num_layers; layer++) {
                block_data_put(k_mems[layer] + block_off, block_size);
                block_data_put(v_mems[layer] + block_off, block_size);
            }
            block_save(block_hash, path.empty() ? kv_data_dir() : path);
        } else {
            ret = block_load(block_hash, path.empty() ? kv_data_dir() : path);
            if (ret != 0) {
                block_status = 2;
                blocks_to_transfer[i][2] = torch::tensor(block_status, torch::kInt64);
                continue;
            }

            /* fill the k/v buf with data from outputs */
            for (int64_t layer = 0; layer < num_layers; layer++) {
                block_data_get(k_mems[layer] + block_off, block_size);
                block_data_get(v_mems[layer] + block_off, block_size);
            }
        }
    }
    restore_affinity();
    if (to_save && total_uncompressed_size > 0) {
        double compression_rate = (static_cast<double>(total_compressed_size) / static_cast<double>(total_uncompressed_size)) * 100.0;
        LOG_INFO("compression_rate (compressed/uncompressed): " << compression_rate << "%");
        total_compressed_size = 0;
        total_uncompressed_size = 0;
    }

    // 仅写入时，退出前回收内存
    if (to_save) {
        thread_cleanup();
    }
}

static void blocks_exists(torch::Tensor& blocks_status)
{
    const int64_t num_blocks = blocks_status.size(0);
    for (int64_t i = 0; i < num_blocks; i++)
        blocks_status[i][1] = torch::tensor(0, torch::kInt64);
    for (int64_t i = 0; i < num_blocks; i++) {
        uint64_t block_hash = static_cast<uint64_t>(blocks_status[i][0].item<int64_t>());
        std::string file_name = kv_data_dir().append("/kv_");
        file_name.append(std::to_string(block_hash));
        file_name.append(".bin");
        std::ifstream file(file_name);
        if (std::filesystem::exists(file_name))
            blocks_status[i][1] = torch::tensor(1, torch::kInt64);
        else
            break;
    }
}

static void blocks_save(std::vector<torch::Tensor>& blocks_vec, torch::Tensor& blocks_to_save)
{
    blocks_transfer(blocks_vec, blocks_to_save, true);
}

static void blocks_load(std::vector<torch::Tensor>& blocks_vec, torch::Tensor& blocks_to_load)
{
    blocks_transfer(blocks_vec, blocks_to_load, false);
}

// 新增带路径参数的版本
static void blocks_save_with_path(std::vector<torch::Tensor>& blocks_vec, 
                                torch::Tensor& blocks_to_save,
                                const std::string& path)
{
    blocks_transfer(blocks_vec, blocks_to_save, true, path);
}

static void blocks_load_with_path(std::vector<torch::Tensor>& blocks_vec,
                                torch::Tensor& blocks_to_load,
                                const std::string& path)
{
    blocks_transfer(blocks_vec, blocks_to_load, false, path);
}


/*
shape:
    ipex (2, num_heads, seq_len, head_size)
*/
static void tfmr_blocks_transfer(std::vector<torch::Tensor>& blocks_vec, int block_tokens, torch::Tensor& blocks_to_transfer, bool to_save)
{
    const int64_t num_blocks = blocks_to_transfer.size(0);
    const int64_t num_heads = blocks_vec[0].size(1);
    const int64_t seq_len = blocks_vec[0].size(2);
    const int64_t head_size = blocks_vec[0].size(3);
    const int64_t element_size = blocks_vec[0].element_size();
    const int64_t per_block_size = element_size * head_size * block_tokens;
    const int64_t per_head_size = element_size * head_size * seq_len;
    const int64_t num_layers = blocks_vec.size();
    char** k_mems = new char*[num_layers];
    char** v_mems = new char*[num_layers];

    for (int64_t layer = 0; layer < num_layers; layer++) {
        k_mems[layer] = static_cast<char*>(blocks_vec[layer][0].data_ptr());
        v_mems[layer] = static_cast<char*>(blocks_vec[layer][1].data_ptr());
    }
    if (log_enabled) {
        std::cout << __func__
            << " num_blocks=" << num_blocks
            << " num_layers=" << num_layers
            << " block_tokens=" << block_tokens
            << " seq_len=" << seq_len
            << " num_heads=" << num_heads
            << " head_size=" << head_size
            << " element_size=" << element_size
            << " k_mems[0]=" << reinterpret_cast<void*>(k_mems[0])
            << " v_mems[0]=" << reinterpret_cast<void*>(v_mems[0])
            << std::endl;
    }
    for (int64_t i = 0; i < num_blocks; i++) {
        int64_t block_num = blocks_to_transfer[i][0].item<int64_t>();
        uint64_t block_hash = static_cast<uint64_t>(blocks_to_transfer[i][1].item<int64_t>());
        std::string file_name = kv_data_dir().append("/kv_");
        file_name.append(std::to_string(block_hash));
        file_name.append(".bin");
        if (log_enabled) {
            std::cout << __func__
                << " block_num=" << block_num
                << " block_hash=" << block_hash
                << " file_name=" << file_name
                << std::endl;
        }
        if (to_save) {
            std::ofstream outfile(file_name, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error opening file for writing: " << file_name << std::endl;
                std::abort();
            }

            for (int64_t layer = 0; layer < num_layers; layer++) {
                for (int64_t head = 0; head < num_heads; head++) {
                    outfile.write(k_mems[layer] + (block_num * per_block_size) + (head * per_head_size), per_block_size);
                    if (!outfile) {
                        std::cerr << "Error writing to file: " << file_name << std::endl;
                        std::abort();
                    }
                }
                for (int64_t head = 0; head < num_heads; head++) {
                    outfile.write(v_mems[layer] + (block_num * per_block_size) + (head * per_head_size), per_block_size);
                    if (!outfile) {
                        std::cerr << "Error writing to file: " << file_name << std::endl;
                        std::abort();
                    }
                }
            }
            outfile.close();
        } else {
            std::ifstream infile(file_name, std::ios::binary | std::ios::ate);
            if (!infile) {
                std::cerr << "Error opening file for reading: " << file_name << std::endl;
                int64_t block_status = 2;
                blocks_to_transfer[i][2] = torch::tensor(block_status, torch::kInt64);
                continue;
            }
            infile.seekg(0, std::ios::beg);
            std::ifstream ifd(file_name, std::ios::binary);
            for (int64_t layer = 0; layer < num_layers; layer++) {
                for (int64_t head = 0; head < num_heads; head++) {
                    if (!infile.read(k_mems[layer] + (block_num * per_block_size) + (head * per_head_size), per_block_size)) {
                        std::cerr << "Error reading from file: " << file_name << std::endl;
                        std::abort();
                    }
                }
                for (int64_t head = 0; head < num_heads; head++) {
                    if (!infile.read(v_mems[layer] + (block_num * per_block_size) + (head * per_head_size), per_block_size)) {
                        std::cerr << "Error reading from file: " << file_name << std::endl;
                        std::abort();
                    }
                }
            }
            infile.close();
        }
    }
}

static void tfmr_blocks_save(std::vector<torch::Tensor>& blocks_vec, int block_tokens, torch::Tensor& blocks_to_save)
{
    tfmr_blocks_transfer(blocks_vec, block_tokens, blocks_to_save, true);
}

static void tfmr_blocks_load(std::vector<torch::Tensor>& blocks_vec, int block_tokens, torch::Tensor& blocks_to_load)
{
    tfmr_blocks_transfer(blocks_vec, block_tokens, blocks_to_load, false);
}

/*
shape:
    sgl (size, head_num, head_dim)
*/
static void sgl_blocks_transfer(std::vector<torch::Tensor>& k_buffers, std::vector<torch::Tensor>& v_buffers, torch::Tensor& blocks_to_transfer, bool to_save)
{
    static int is_mla = -1;
    const int64_t num_blocks = blocks_to_transfer.size(0);
    const int64_t per_block_tokens = blocks_to_transfer.size(1) - 2;
    const int64_t num_layers = k_buffers.size();
    const int64_t element_size = k_buffers[0].element_size();
    const int64_t num_blocks_max = k_buffers[0].size(0);
    const int64_t head_num = k_buffers[0].size(1);
    const int64_t head_dim = k_buffers[0].size(2);
    const int64_t per_token_size = k_buffers[0][0].numel() * element_size;
    const int64_t block_size = per_token_size * per_block_tokens;
    char** k_mems = new char*[num_layers];
    char** v_mems = new char*[num_layers];

    if (is_mla == -1) {
        const int64_t v_buffers_size = v_buffers.size();
        if (v_buffers_size == 1)
            is_mla = 1;
        else
            is_mla = 0;
    }
    if (log_enabled) {
        std::cout << __func__
            << (to_save ? " save" : " load")
            << " num_blocks=" << num_blocks
            << " num_blocks_max=" << num_blocks_max
            << " num_layers=" << num_layers
            << " head_num=" << head_num
            << " head_dim=" << head_dim
            << " element_size=" << element_size
            << " per_token_size=" << per_token_size
            << " per_block_tokens=" << per_block_tokens
            << " block_size=" << block_size
            << " is_mla=" << is_mla
            << std::endl;
    }

    save_affinity();
    for (int64_t layer = 0; layer < num_layers; layer++) {
        k_mems[layer] = static_cast<char*>(k_buffers[layer].data_ptr());
        if (!is_mla)
            v_mems[layer] = static_cast<char*>(v_buffers[layer].data_ptr());
    }

    #pragma omp parallel for schedule(static, 1) num_threads(qat_instance_num)
    for (int64_t i = 0; i < num_blocks; i++) {
        uint64_t block_hash = static_cast<uint64_t>(blocks_to_transfer[i][1].item<int64_t>());
        if (log_enabled) {
            std::cout << __func__
                << (to_save ? " save" : " load")
                << " block=" << i
                << " block_hash=" << block_hash
                << " kv_indices=";
            for (int64_t t = 0; t < per_block_tokens; t++)
                std::cout << " " << static_cast<uint64_t>(blocks_to_transfer[i][2+t].item<int64_t>());
            std::cout << std::endl;
        }

        if (!is_mla)
            block_buf_prepare(block_size * num_layers * 2);
        else
            block_buf_prepare(block_size * num_layers);

        if (to_save) {
            for (int64_t t_index = 0; t_index < per_block_tokens; t_index++) {
                uint64_t kv_indice = static_cast<uint64_t>(blocks_to_transfer[i][2+t_index].item<int64_t>());
                for (int64_t layer = 0; layer < num_layers; layer++) {
                    block_data_put(k_mems[layer] + (per_token_size * kv_indice), per_token_size);
                    if (!is_mla)
                        block_data_put(v_mems[layer] + (per_token_size * kv_indice), per_token_size);
                }
            }
            block_save(block_hash);
        } else {
            if (block_load(block_hash) != 0)
                continue;
            for (int64_t t = 0; t < per_block_tokens; t++) {
                uint64_t kv_indice = static_cast<uint64_t>(blocks_to_transfer[i][2+t].item<int64_t>());
                for (int64_t layer = 0; layer < num_layers; layer++) {
                    block_data_get(k_mems[layer] + (per_token_size * kv_indice), per_token_size);
                    if (!is_mla)
                        block_data_get(v_mems[layer] + (per_token_size * kv_indice), per_token_size);
                }
            }
            // set return = 1 for success
            blocks_to_transfer[i][0] = torch::tensor(1, torch::kInt64);
        }
    }
    restore_affinity();
    if (to_save && total_uncompressed_size > 0) {
        double compression_rate = (static_cast<double>(total_compressed_size) / static_cast<double>(total_uncompressed_size)) * 100.0;
        LOG_INFO("compression_rate (compressed/uncompressed): " << compression_rate << "%");
        total_compressed_size = 0;
        total_uncompressed_size = 0;
    }
}

/* blocks_to_save shape: (num_blocks, input/return+block_hash+len(kv_indices)) */
static void sgl_blocks_save(std::vector<torch::Tensor>& k_buffers, std::vector<torch::Tensor>& v_buffers, torch::Tensor& blocks_to_save)
{
    sgl_blocks_transfer(k_buffers, v_buffers, blocks_to_save, true);
}

static void sgl_blocks_load(std::vector<torch::Tensor>& k_buffers, std::vector<torch::Tensor>& v_buffers, torch::Tensor& blocks_to_load)
{
    sgl_blocks_transfer(k_buffers, v_buffers, blocks_to_load, false);
}

static std::string cpu_bind(const std::string& cpu_ids) {
    bitmask* omp_cpu_mask = numa_parse_cpustring(cpu_ids.c_str());
    if (omp_cpu_mask->size <= 0) {
        std::cerr << "Invalid cpu_ids" << std::endl;
        std::abort();
    }
    std::vector<int> omp_cpu_ids;
    omp_cpu_ids.reserve(omp_cpu_mask->size);

    constexpr int group_size = 8 * sizeof(*omp_cpu_mask->maskp);

    for (unsigned long offset = 0; offset < omp_cpu_mask->size; offset += group_size) {
        unsigned long group_mask = omp_cpu_mask->maskp[offset / group_size];
        int i = 0;
        while (group_mask) {
            if (group_mask & 1) {
                omp_cpu_ids.emplace_back(offset + i);
            }
            ++i;
            group_mask >>= 1;
        }
    }

    // Memory node binding
    if (numa_available() != -1) {
        int mem_node_id = numa_node_of_cpu(omp_cpu_ids.front());
        bitmask* mask = numa_parse_nodestring(std::to_string(mem_node_id).c_str());
        bitmask* src_mask = numa_get_membind();

        int pid = getpid();

        // move all existing pages to the specified numa node.
        *(src_mask->maskp) = *(src_mask->maskp) ^ *(mask->maskp);
        int page_num = numa_migrate_pages(pid, src_mask, mask);
        if (page_num == -1) {
            TORCH_CHECK(false,
                    "numa_migrate_pages failed. errno: " + std::to_string(errno));
        }

        // restrict memory allocation node.
        numa_set_membind(mask);
        numa_set_strict(1);
    }

    // OMP threads binding
    omp_set_num_threads((int)omp_cpu_ids.size());
    torch::set_num_threads((int)omp_cpu_ids.size());
    if (static_cast<int>(omp_cpu_ids.size()) != static_cast<int>(torch::get_num_threads())) {
        std::cerr << "set torch threads failed" << std::endl;
        std::abort();
    }
    if (static_cast<int>(omp_cpu_ids.size()) != static_cast<int>(omp_get_max_threads())) {
        std::cerr << "set omp max threads failed" << std::endl;
        std::abort();
    }

    std::vector<std::pair<int, int>> thread_core_mapping;
    thread_core_mapping.reserve(omp_cpu_ids.size());
    omp_lock_t writelock;
    omp_init_lock(&writelock);

#pragma omp parallel for schedule(static, 1)
    for (size_t i = 0; i < omp_cpu_ids.size(); ++i) {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(omp_cpu_ids[i], &mask);
        int ret = sched_setaffinity(0, sizeof(cpu_set_t), &mask);
        if (ret == -1) {
            TORCH_CHECK(false,
                    "sched_setaffinity failed. errno: " + std::to_string(errno));
        }

        omp_set_lock(&writelock);
        thread_core_mapping.emplace_back(gettid(), omp_cpu_ids[i]);
        omp_unset_lock(&writelock);
    }

    omp_destroy_lock(&writelock);

    numa_free_nodemask(omp_cpu_mask);

    std::stringstream ss;
    ss << "OMP threads binding of Process " << getpid() << ":\n";
    std::sort(thread_core_mapping.begin(), thread_core_mapping.end(),
            [](auto&& a, auto&& b) { return a.second < b.second; });
    for (auto&& item : thread_core_mapping) {
        ss << "\t"
            << "OMP tid: " << item.first << ", core " << item.second << "\n";
    }

    if (log_enabled)
        std::cout << ss.str();
    return ss.str();
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for std::string, std::vector support

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("init_config", &init_config, "Initialize with default config");
    m.def("set_log_enabled", &set_log_enabled, "Set log enabled");
    m.def("set_mantissa_loss_level", &set_mantissa_loss_level, "Set mantissa loss level");
    m.def("set_qat_instance_num", &set_qat_instance_num, "Set QAT instance num");
    m.def("set_kv_data_dir", &set_kv_data_dir, "Set KV data directory");

    // 添加带路径参数的新版本
    m.def("blocks_save_with_path", &blocks_save_with_path, "Save blocks with custom path", py::call_guard<py::gil_scoped_release>());
    m.def("blocks_load_with_path", &blocks_load_with_path, "Load blocks with custom path", py::call_guard<py::gil_scoped_release>());

    // 原始 blocks save/load
    m.def("blocks_save", &blocks_save, "Save blocks", py::call_guard<py::gil_scoped_release>());
    m.def("blocks_load", &blocks_load, "Load blocks", py::call_guard<py::gil_scoped_release>());
    m.def("blocks_exists", &blocks_exists, "Check block existence", py::call_guard<py::gil_scoped_release>());

    // Transformer blocks
    m.def("tfmr_blocks_save", &tfmr_blocks_save, "Save transformer blocks", py::call_guard<py::gil_scoped_release>());
    m.def("tfmr_blocks_load", &tfmr_blocks_load, "Load transformer blocks", py::call_guard<py::gil_scoped_release>());

    // Single block transfer functions
    m.def("sgl_blocks_save", &sgl_blocks_save, "Save single blocks", py::call_guard<py::gil_scoped_release>());
    m.def("sgl_blocks_load", &sgl_blocks_load, "Load single blocks", py::call_guard<py::gil_scoped_release>());

    // CPU bind (already CPU-intensive)
    m.def("cpu_bind", &cpu_bind, "Bind threads to specified CPUs", py::call_guard<py::gil_scoped_release>());
}