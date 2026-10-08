// include/strata/platform/profiling.hpp - sherlock-it diagnostic instrumentation for the HIP engine.
//
// Gated behind STRATA_SHERLOCK_IT=1 (set at process start).  When enabled:
//   * kernel launches are timed with CUDA/HIP events
//   * memory allocations / frees are tracked (VRAM bytes in use per device)
//   * memory copies are timed and counted by direction
//   * hipBLAS/rocBLAS calls are timed
//   * CPU phase markers record wall time and RSS
// Results are written to <STRATA_SHERLOCK_OUTDIR>/sherlock-<pid>.csv (or stderr as fallback).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::profiling {

struct KernelSample {
    uint64_t seq;
    const char* name;
    void* stream;
    int32_t grid_x, grid_y, grid_z;
    int32_t block_x, block_y, block_z;
    uint32_t shared_bytes;
    double ms;
    uint64_t cpu_us;
};

struct MemSample {
    uint64_t seq;
    const char* op;      // "alloc", "free", "memcpy"
    const char* site;
    size_t bytes;
    void* ptr;
    int kind;            // cudaMemcpyKind
    double ms;
    size_t vram_used;
    size_t vram_total;
};

struct BlasSample {
    uint64_t seq;
    const char* name;
    int m, n, k;
    const char* type;
    double ms;
};

struct PhaseSample {
    uint64_t seq;
    const char* name;
    double cpu_ms;
    uint64_t rss_kb;
    size_t vram_used;
    size_t vram_total;
};

class Profiler {
public:
    static Profiler& instance();

    void initialize();
    void init_locked();

    bool enabled() const { return enabled_; }

    uint64_t kernel_launch(const char* name, void* stream,
                           int32_t gx, int32_t gy, int32_t gz,
                           int32_t bx, int32_t by, int32_t bz, uint32_t shared);
    void kernel_done(uint64_t id);

    void mem_alloc(const char* site, size_t bytes, void* ptr);
    void mem_free(const char* site, void* ptr);
    void mem_copy(const char* site, size_t bytes, int kind,
                  void* dst, void* src, void* stream);

    void blas_call(const char* name, int m, int n, int k,
                   const char* type, double ms);

    void phase_begin(const char* name);
    void phase_end(const char* name);

    void flush();

private:
    Profiler();
    ~Profiler();
    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    void emit(const char* kind, uint64_t seq, const char* name,
              int m, int n, int k, const char* type, double ms,
              size_t bytes, void* ptr, const char* site, const char* kind_str,
              int grid_x, int grid_y, int grid_z,
              int block_x, int block_y, int block_z, uint32_t shared,
              size_t vram_free_mb, size_t vram_total_mb,
              uint64_t rss_kb, uint64_t t_us);

    struct PendingKernel {
        const char* name;
        void* stream;
        int32_t gx, gy, gz, bx, by, bz;
        uint32_t shared;
        uint64_t t0_us;
        void* start_event;
        void* end_event;
        bool events_ready;
    };

    std::atomic<uint64_t> seq_{0};
    std::atomic<uint32_t> active_kernels_{0};
    bool enabled_ = false;
    bool initialized_ = false;
    std::string out_dir_;
    uint32_t pid_ = 0;
    bool fallback_stderr_ = false;
    std::ofstream out_file_;
    std::mutex out_mu_;
    std::mutex mu_;
    std::unordered_map<uint64_t, PendingKernel> pending_;
    std::unordered_map<std::string, uint64_t> phase_starts_;
};

struct PhaseScope {
    PhaseScope(const char* name);
    ~PhaseScope();
private:
    const char* name_;
    bool active_;
};

}  // namespace strata::profiling

#ifdef STRATA_SHERLOCK_IT
#define STRATA_PROF_KERNEL(name, stream, grid, block, shared) \
    do { \
        ::strata::profiling::Profiler::instance().kernel_launch( \
            name, stream, \
            (int32_t)(grid).x, (int32_t)(grid).y, (int32_t)(grid).z, \
            (int32_t)(block).x, (int32_t)(block).y, (int32_t)(block).z, (uint32_t)(shared)); \
    } while(0)
#define STRATA_PROF_KERNEL_DONE(id) \
    do { ::strata::profiling::Profiler::instance().kernel_done(id); } while(0)
#define STRATA_PROF_ALLOC(site, bytes, ptr) \
    do { ::strata::profiling::Profiler::instance().mem_alloc(site, bytes, ptr); } while(0)
#define STRATA_PROF_FREE(site, ptr) \
    do { ::strata::profiling::Profiler::instance().mem_free(site, ptr); } while(0)
#define STRATA_PROF_COPY(site, bytes, kind, dst, src, stream) \
    do { ::strata::profiling::Profiler::instance().mem_copy(site, bytes, kind, dst, src, stream); } while(0)
#define STRATA_PROF_BLAS(name, m, n, k, type, ms) \
    do { ::strata::profiling::Profiler::instance().blas_call(name, m, n, k, type, ms); } while(0)
#define STRATA_PROF_PHASE(name) \
    ::strata::profiling::PhaseScope _strata_prof_phase_##__LINE__(name)
#else
#define STRATA_PROF_KERNEL(name, stream, grid, block, shared) do {} while(0)
#define STRATA_PROF_KERNEL_DONE(id) do {} while(0)
#define STRATA_PROF_ALLOC(site, bytes, ptr) do {} while(0)
#define STRATA_PROF_FREE(site, ptr) do {} while(0)
#define STRATA_PROF_COPY(site, bytes, kind, dst, src, stream) do {} while(0)
#define STRATA_PROF_BLAS(name, m, n, k, type, ms) do {} while(0)
#define STRATA_PROF_PHASE(name) do {} while(0)
#endif
