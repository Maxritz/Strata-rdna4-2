// src/platform/profiling.cpp - sherlock-it profiling implementation for the HIP engine.
#include "strata/platform/profiling.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#if defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#define STRATA_EVENT_BLOCKING_SYNC hipEventBlockingSync
#define STRATA_EVENT_T hipEvent_t
#define STRATA_EVENT_CREATE hipEventCreate
#define STRATA_EVENT_CREATE_WITH_FLAGS hipEventCreateWithFlags
#define STRATA_EVENT_RECORD hipEventRecord
#define STRATA_EVENT_SYNC hipEventSynchronize
#define STRATA_EVENT_ELAPSED hipEventElapsedTime
#define STRATA_EVENT_DESTROY hipEventDestroy
#define STRATA_EVENT_DISABLE_TIMING hipEventDisableTiming
#define STRATA_MEM_GET_INFO hipMemGetInfo
#define STRATA_MEM_SUCCESS hipSuccess
#define STRATA_STREAM_T hipStream_t
#else
#include <cuda_runtime.h>
#define STRATA_EVENT_BLOCKING_SYNC cudaEventBlockingSync
#define STRATA_EVENT_T cudaEvent_t
#define STRATA_EVENT_CREATE cudaEventCreate
#define STRATA_EVENT_CREATE_WITH_FLAGS cudaEventCreateWithFlags
#define STRATA_EVENT_RECORD cudaEventRecord
#define STRATA_EVENT_SYNC cudaEventSynchronize
#define STRATA_EVENT_ELAPSED cudaEventElapsedTime
#define STRATA_EVENT_DESTROY cudaEventDestroy
#define STRATA_EVENT_DISABLE_TIMING cudaEventDisableTiming
#define STRATA_MEM_GET_INFO cudaMemGetInfo
#define STRATA_MEM_SUCCESS cudaSuccess
#define STRATA_STREAM_T cudaStream_t
#endif

namespace strata::profiling {

namespace {

constexpr size_t kMaxPath = 4096;

uint64_t now_us() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t now_ms() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

uint64_t rss_kb() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return info.WorkingSetSize / 1024;
#else
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_maxrss;
#endif
}

uint32_t pid() {
#ifdef _WIN32
    return GetCurrentProcessId();
#else
    return getpid();
#endif
}

}  // namespace

Profiler& Profiler::instance() {
    static Profiler p;
    if (!p.initialized_) {
        p.init_locked();
    }
    return p;
}

Profiler::Profiler() {
    pid_ = pid();
    active_kernels_ = 0;
}

void Profiler::init_locked() {
    initialized_ = true;
    const char* on = std::getenv("STRATA_SHERLOCK_IT");
    if (!on || on[0] == '\0' || on[0] == '0') {
        enabled_ = false;
        return;
    }
    enabled_ = true;

    const char* outdir = std::getenv("STRATA_SHERLOCK_OUTDIR");
    out_dir_ = outdir ? outdir : ".";

    char fname[kMaxPath];
    std::snprintf(fname, sizeof(fname), "%s/sherlock-%u.csv", out_dir_.c_str(), pid_);
    out_file_.open(fname, std::ios::out | std::ios::trunc);
    if (!out_file_) {
        std::fprintf(stderr, "strata: sherlock-it: cannot open %s for writing; falling back to stderr\n", fname);
        fallback_stderr_ = true;
    } else {
        out_file_ << "kind,seq,name,m,n,k,type,ms,bytes,ptr,site,kind_str,grid_x,grid_y,grid_z,block_x,block_y,block_z,shared,vram_free_mb,vram_total_mb,rss_kb,t_us\n";
        out_file_.flush();
    }

    std::fprintf(stderr, "strata: sherlock-it profiling enabled (pid=%u, out=%s)\n", pid_, out_dir_.c_str());
}

void Profiler::initialize() {
    if (!initialized_) {
        init_locked();
    }
}

Profiler::~Profiler() {
    if (enabled_) {
        flush();
    }
}

uint64_t Profiler::kernel_launch(const char* name, void* stream,
                                 int32_t gx, int32_t gy, int32_t gz,
                                 int32_t bx, int32_t by, int32_t bz, uint32_t shared) {
    if (!enabled_) return 0;
    const uint64_t id = seq_.fetch_add(1);
    const uint64_t t = now_us();

    PendingKernel pk;
    pk.name = name;
    pk.stream = stream;
    pk.gx = gx; pk.gy = gy; pk.gz = gz;
    pk.bx = bx; pk.by = by; pk.bz = bz;
    pk.shared = shared;
    pk.t0_us = t;

    STRATA_EVENT_T start_ev, end_ev;
    STRATA_EVENT_CREATE_WITH_FLAGS(&start_ev, STRATA_EVENT_BLOCKING_SYNC | STRATA_EVENT_DISABLE_TIMING);
    STRATA_EVENT_CREATE_WITH_FLAGS(&end_ev, STRATA_EVENT_BLOCKING_SYNC | STRATA_EVENT_DISABLE_TIMING);
    pk.start_event = start_ev;
    pk.end_event = end_ev;
    STRATA_EVENT_RECORD(start_ev, (STRATA_STREAM_T)stream);

    {
        std::lock_guard<std::mutex> lk(mu_);
        pending_[id] = pk;
        active_kernels_.fetch_add(1);
    }
    return id;
}

void Profiler::kernel_done(uint64_t id) {
    if (!enabled_ || id == 0) return;
    PendingKernel* pk = nullptr;
    {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = pending_.find(id);
        if (it == pending_.end()) return;
        pk = &it->second;
        pk->events_ready = true;
    }
    if (pk) {
        STRATA_EVENT_RECORD((STRATA_EVENT_T)pk->end_event, (STRATA_STREAM_T)pk->stream);
        STRATA_EVENT_SYNC((STRATA_EVENT_T)pk->end_event);
        float elapsed_ms = 0;
        STRATA_EVENT_ELAPSED(&elapsed_ms, (STRATA_EVENT_T)pk->start_event, (STRATA_EVENT_T)pk->end_event);
        STRATA_EVENT_DESTROY((STRATA_EVENT_T)pk->start_event);
        STRATA_EVENT_DESTROY((STRATA_EVENT_T)pk->end_event);

        uint64_t dt_us = now_us() - pk->t0_us;
        size_t free_b = 0, total_b = 0;
        if (STRATA_MEM_GET_INFO(&free_b, &total_b) != STRATA_MEM_SUCCESS) { free_b = 0; total_b = 0; }

        emit("KERNEL", id, pk->name, 0, 0, 0, "none", elapsed_ms,
             0, nullptr, pk->name, "kernel",
             pk->gx, pk->gy, pk->gz, pk->bx, pk->by, pk->bz, pk->shared,
             free_b >> 20, total_b >> 20, rss_kb(), dt_us);
        active_kernels_.fetch_sub(1);

        std::lock_guard<std::mutex> lk2(mu_);
        pending_.erase(id);
    }
}

void Profiler::mem_alloc(const char* site, size_t bytes, void* ptr) {
    if (!enabled_) return;
    uint64_t t = now_us();
    size_t free_b = 0, total_b = 0;
    if (STRATA_MEM_GET_INFO(&free_b, &total_b) == STRATA_MEM_SUCCESS) {
        emit("ALLOC", 0, "", 0, 0, 0, "none", 0.0,
             bytes, ptr, site, "",
             0, 0, 0, 0, 0, 0, 0,
             free_b >> 20, total_b >> 20, rss_kb(), t);
    }
}

void Profiler::mem_free(const char* site, void* ptr) {
    if (!enabled_) return;
    uint64_t t = now_us();
    size_t free_b = 0, total_b = 0;
    if (STRATA_MEM_GET_INFO(&free_b, &total_b) == STRATA_MEM_SUCCESS) {
        emit("FREE", 0, "", 0, 0, 0, "none", 0.0,
             0, ptr, site, "",
             0, 0, 0, 0, 0, 0, 0,
             free_b >> 20, total_b >> 20, rss_kb(), t);
    }
}

void Profiler::mem_copy(const char* site, size_t bytes, int kind,
                         void* dst, void* src, void* stream) {
    if (!enabled_) return;
    uint64_t t = now_us();
    const char* kind_str = "unknown";
    switch (kind) {
        case 0: kind_str = "1to1"; break;
        case 1: kind_str = "htod"; break;
        case 2: kind_str = "dtoh"; break;
        case 3: kind_str = "dtod"; break;
    }
    emit("COPY", 0, "", 0, 0, 0, "none", 0.0,
         bytes, nullptr, site, kind_str,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, rss_kb(), t);
}

void Profiler::blas_call(const char* name, int m, int n, int k,
                         const char* type, double ms) {
    if (!enabled_) return;
    emit("BLAS", 0, name, m, n, k, type, ms,
         0, nullptr, name, type,
         0, 0, 0, 0, 0, 0, 0,
         0, 0, rss_kb(), now_us());
}

void Profiler::phase_begin(const char* name) {
    if (!enabled_) return;
    phase_starts_[name] = now_ms();
}

void Profiler::phase_end(const char* name) {
    if (!enabled_) return;
    auto it = phase_starts_.find(name);
    if (it == phase_starts_.end()) return;
    uint64_t elapsed_ms = now_ms() - it->second;
    phase_starts_.erase(it);
    size_t free_b = 0, total_b = 0;
    if (STRATA_MEM_GET_INFO(&free_b, &total_b) != STRATA_MEM_SUCCESS) { free_b = 0; total_b = 0; }
    emit("PHASE", 0, name, 0, 0, 0, "none", (double)elapsed_ms,
         0, nullptr, name, "",
         0, 0, 0, 0, 0, 0, 0,
         free_b >> 20, total_b >> 20, rss_kb(), now_us());
}

void Profiler::emit(const char* kind, uint64_t seq, const char* name,
                    int m, int n, int k, const char* type, double ms,
                    size_t bytes, void* ptr, const char* site, const char* kind_str,
                    int grid_x, int grid_y, int grid_z,
                    int block_x, int block_y, int block_z, uint32_t shared,
                    size_t vram_free_mb, size_t vram_total_mb,
                    uint64_t rss_kb, uint64_t t_us) {
    if (fallback_stderr_) {
        std::fprintf(stderr, "[sherlock] kind=%s seq=%llu name=%s m=%d n=%d k=%d type=%s ms=%.4f bytes=%zu ptr=%p site=%s copy_kind=%s grid=(%d,%d,%d) block=(%d,%d,%d) shared=%u vram_free=%zu/%zuMB rss_kb=%llu t=%lluus\n",
            kind, (unsigned long long)seq, name, m, n, k, type, ms,
            bytes, ptr, site, kind_str, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared,
            vram_free_mb, vram_total_mb, (unsigned long long)rss_kb, (unsigned long long)t_us);
        return;
    }
    std::lock_guard<std::mutex> lk(out_mu_);
    out_file_ << kind << "," << seq << "," << (name ? name : "") << ","
              << m << "," << n << "," << k << "," << (type ? type : "") << ","
              << ms << "," << bytes << "," << ptr << "," << (site ? site : "") << ","
              << (kind_str ? kind_str : "") << ","
              << grid_x << "," << grid_y << "," << grid_z << ","
              << block_x << "," << block_y << "," << block_z << "," << shared << ","
              << vram_free_mb << "," << vram_total_mb << "," << rss_kb << "," << t_us << "\n";
    out_file_.flush();
}

void Profiler::flush() {
    if (!enabled_) return;
    std::fprintf(stderr, "[sherlock] PROFILER FLUSH pid=%u\n", pid_);
    if (out_file_) {
        std::lock_guard<std::mutex> lk(out_mu_);
        out_file_.flush();
    }
}

PhaseScope::PhaseScope(const char* name) : name_(name), active_(false) {
    if (Profiler::instance().enabled()) {
        active_ = true;
        Profiler::instance().phase_begin(name_);
    }
}

PhaseScope::~PhaseScope() {
    if (active_) {
        Profiler::instance().phase_end(name_);
    }
}

}  // namespace strata::profiling
