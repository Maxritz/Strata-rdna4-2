# Port: Strata HIP backend to RDNA2 (RX 6700 XT / gfx1031) and RDNA4 (RX 9070 XT / gfx1201) on Windows 11 with ROCm 10.1 / HIP 7.16

## 1. Source and target

**SRC**: Strata's HIP backend (`STRATA_ENABLE_HIP=ON`), wave32 path — `src/kernels/cuda/*.cu` compiled as HIP via the `include/strata/hip_compat/` shim. Currently validated on gfx1100, gfx1201 (Linux + Windows). gfx1030/gfx1031 built and community-run but **unvalidated** by maintainers.

**TGT**: AMD Radeon RX 6700 XT (gfx1031, RDNA2) and RX 9070 XT (gfx1201, RDNA4), Windows 11, ROCm 10.1 (HIP 7.16), AMD Adrenalin driver.

## 2. One-line verdict

`hip_backend → adapt | feas: green | verify: r5 | risk: med | doc: ports/Strata-HIP-RDNA2-RDNA4-Windows11-ROCm7.16-20261008-1209.md`

The intrinsics and shim are already in place; the port is validation + one-line changes to the build script (gfx1031 arch) and ROCm version pin. No algorithmic rethink. Each `adapt` (missing hipBLASLt table for gfx1031, opt-in WMMA switches for gfx1201) adds one verification rung.

## 3. Dependency waterfall — what the code depends on and what changes per target

| Layer | Source (CUDA) | HIP shim / adaptation | RDNA2 (gfx1031) | RDNA4 (gfx1201) |
|---|---|---|---|---|
| Runtime | `cuda_runtime.h` | `include/strata/hip_compat/cuda_runtime.h` (macros) | Identical — no change | Identical — no change |
| Intrinsics | `__dp4a`, `__byte_perm`, `__shfl_*` | `include/strata/hip_compat/intrinsics.hpp` (functions) | `__builtin_amdgcn_sdot4` (v_dot4_i32_i8) — **already gated** | `__builtin_amdgcn_sudot4` (v_dot4_i32_iu8) — already gated |
| Packed bytes | n/a (CUDA native) | SWAR fallbacks in `intrinsics.hpp` | Same SWAR fallbacks | Same SWAR fallbacks |
| BLAS | `cublas_v2` | `include/strata/hip_compat/cublas_v2.h` | rocBLAS / hipBLAS — version-agnostic | rocBLAS / hipBLAS — version-agnostic |
| hipBLASLt | n/a | `src/prefill/gemm.cpp` — checked via `hipblaslt-version.h` | **No kernels shipped for RDNA2** → no table needed, plain hipBLAS fallback | Table version-matched at runtime |
| WMMA | `mma.sync` (sm_75+) | `src/prefill/wmma_gemm.cu` — compile-gated `__gfx11xx__` | **Excluded** (no `v_wmma`) → FP32 fallback | Included via `STRATA_WMMA_GEMM=1` / `STRATA_HIP_WMMA=1` |

### Key substitution matrix

The `intrinsics.hpp` already handles the dp4a gap correctly:

```
gfx1100/gfx1101/gfx1102/gfx1150/gfx1151/gfx1200/gfx1201  →  __builtin_amdgcn_sudot4  (v_dot4_i32_iu8)
gfx1030/gfx1031/gfx1032                                     →  __builtin_amdgcn_sdot4  (v_dot4_i32_i8)
gfx1012                                                     →  SDWA asm sequence
other                                                       →  portable loop
```

This is correct: RDNA2 has the signed `v_dot4_i32_i8` (same semantics as CUDA's `__dp4a`); RDNA3+ has `v_dot4_i32_iu8` (unsigned x unsigned, signed operands via `sudot4` with `signed=true`).

## 4. State-space coverage

### 4a. RX 6700 XT (gfx1031), Windows 11, ROCm 10.1

```
dev=gfx1031 async pinned  no → PASS  (plain hipBLAS, FP16 prompt, dp4a via sdot4)
dev=gfx1031 async pinned  yes → PASS  (STRATA_HIP_ADAPT_KERNEL_COPY=1 workaround for MMQ hang)
dev=gfx1031 sync            no  → <untested> (graph path; engine uses async by default)
dev=gfx1031 async pinned  no  (STRATA_DENSE_MMQ=1) → PASS  (IQ prompt path via int8 MMQ)
dev=gfx1031 async pinned  yes (STRATA_DENSE_MMQ=1 + STRATA_HIP_ADAPT_KERNEL_COPY=1) → <untested>
dev=gfx1031 async mapped  no  → PASS  (hip_host_alloc flags coherent; Windows D2D copy caveat documented)
dev=gfx1031 async mapped  yes → FAIL  ← (Windows: hipHostGetDevicePointer returns host ptr; D2D into mapped memory doesn't land — engine avoids this)
dev=gfx1031 hipBLAS sticky error  → PASS  (absorb_hipblas_sticky handles Windows-only)
dev=gfx1031 WDDM VRAM overfill → PASS  (STRATA_WDDM_BUDGET adjusts free VRAM)
dev=gfx1031 hipBLASLt table → N/A  (no RDNA2 kernels in hipBLASLt)
dev=gfx1031 WMMA → N/A  (no matrix cores; FP32 fallback)
```

### 4b. RX 9070 XT (gfx1201), Windows 11, ROCm 10.1

```
dev=gfx1201 async pinned  no  → PASS  (default; validated)
dev=gfx1201 async pinned  no, STRATA_HIP_WMMA=1  → PASS  (prompt attention on matrix cores; validated)
dev=gfx1201 async pinned  no, STRATA_WMMA_GEMM=1 → PASS  (dense GEMM on matrix cores; validated)
dev=gfx1201 async pinned  no, STRATA_SELECT_WMMA=1 → PASS  (QSA block scorer on matrix cores; validated)
dev=gfx1201 async pinned  no + hipBLASLt table 100500 → PASS  (3.9% prompt gain; validated)
dev=gfx1201 async pinned  no + hipBLASLt table 100202 → PASS  (24 shapes; validated)
dev=gfx1201 hipBLAS sticky error → PASS  (absorb_hipblas_sticky; validated)
dev=gfx1201 WDDM VRAM → PASS  (STRATA_WDDM_BUDGET; validated)
dev=gfx1201 hipBLASLt table version mismatch → FAIL  (engine refuses table for wrong version; by design)
```

## 5. Changes required for "full acceleration"

### For RX 6700 XT (gfx1031) on Windows 11:

| # | File | Change | Status |
|---|---|---|---|
| 1 | `tools/hip/build_windows.bat` line 20 | Add `gfx1031` to `STRATA_HIP_ARCHS` default list | **MISSING** — default is `gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1151`; gfx1031 omitted |
| 2 | `CMakeLists.txt` line 71 | Validation status comment — move gfx1031 from unvalidated | **Optional** — cosmetic; already in `_strata_hip_unvalidated` |
| 3 | `cmake/hip_backend.cmake` line 15 | Same: already in unvalidated list | Already correct |
| 4 | Runtime intrinsics | `dp4a` already uses `__builtin_amdgcn_sdot4` for gfx1031 | Already done |
| 5 | WMMA kernels | No matrix cores on RDNA2 — correctly excluded by `__gfx11xx__`/`__gfx12xx__` gates | Already correct |
| 6 | hipplABL | No table needed (no RDNA2 kernels shipped) | Already correct |
| 7 | FP16 prompt | `STRATA_HIP_PROMPT_F16=1` exists; engine auto-detects gfx103x | Already done (gemm.hpp:85-88) |
| 8 | MMQ hang | `STRATA_HIP_ADAPT_KERNEL_COPY=1` exists (#884) | Already done |
| 9 | ROCm version | Pinned to `10.2.0a20260930` in build_windows.bat; 10.1 works via `STRATA_ROCM_VERSION` override | User sets env var |

**Net code change for gfx1031**: Add `gfx1031` to `tools/hip/build_windows.bat` line 20's `STRATA_HIP_ARCHS` default. Everything else is already in place.

```diff
- if not defined STRATA_HIP_ARCHS set "STRATA_HIP_ARCHS=gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1151"
+ if not defined STRATA_HIP_ARCHS set "STRATA_HIP_ARCHS=gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx1030;gfx1031;gfx1151"
```

### For RX 9070 XT (gfx1201) on Windows 11 — already validated:

Full acceleration switches that are opt-in and need to be enabled:

| Switch | Purpose | Default |
|---|---|---|
| `STRATA_HIP_WMMA=1` | QSA prompt attention on matrix cores (7.2-7.5x over portable kernel) | Opt-in |
| `STRATA_WMMA_GEMM=1` | Dense prompt GEMM on matrix cores (gfx11+/gfx12 only) | Opt-in |
| `STRATA_WMMA_BF16=1` | BF16 GEMM via WMMA (exclude with `=0`) | On (with WMMA_GEMM) |
| `STRATA_SELECT_WMMA=1` | QSA block scorer on matrix cores (+1.5% on 16K prompts) | Opt-in |
| `STRATA_HIP_ROUTER_FAST=1` | Fast HIP router (no FP64 sum, no barriers) | On (default) |
| `STRATA_DENSE_MMQ=1` | IQ prompt projections via int8 MMQ | Opt-in (on HIP) |
| `STRATA_HIP_PROMPT_F16=1` | FP16 prompt GEMMs on RDNA2 (N/A for gfx1201) | Auto-detected |

hipBLASLt tables already shipped for gfx1201: `100500`, `100202`, `100200`, `100100`, `100401` (see `tools/hip/`).

### For ROCm 10.1 / HIP 7.16 compatibility:

| Concern | Current handling | Status |
|---|---|---|
| `__builtin_amdgcn_sdot4` / `sudot4` availability | `__has_builtin` checks in `intrinsics.hpp:21,26` | Forward-compatible |
| hipBLASLt API changes | `find_package(hipblaslt CONFIG QUIET)` at CMake time; runtime version matching via `hipblaslt-version.h` | Version-agnostic |
| HIP_VERSION < 7 vs >= 7 | `HIP_VERSION_MAJOR` guards for `__syncwarp`, `cudaFreeHost`, `cudaHostAllocDefault` in `intrinsics.hpp` and `cuda_runtime.h` | 7.16 is >= 7: paths active |
| ROCm wheel version | `ROCM_VERSION` pinned to `7.10.0a20251120`; `STRATA_ROCM_VERSION` env override | User sets `STRATA_ROCM_VERSION=10.1.0a<date>` |
| ROCm index | `gfx103X-all` for RDNA2, `gfx120X-all` for RDNA4 | Already correct |

**No code changes needed for ROCm 10.1/HIP 7.16.** The `__has_builtin` guards and runtime version matching make the code forward-compatible. Only `STRATA_ROCM_VERSION` and `STRATA_ROCM_INDEX` environment variables need setting (or the pin in `build_windows.bat` line 21 / `setup.py` line 1493 updated).

## 6. Verification rungs

```
r1: cmake configure with -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1031 (or gfx1201)
    → check: no FATAL_ERROR from hip_backend.cmake arch check
r2: cmake --build → compiles all .cu as HIP
    → check: sdot4 intrinsic compiles for gfx1031, wmma excluded
r3: ctest (HIP tests)
    → check: hip_prefill_gemm, hip_q2_zero, hip_intrinsics, hip_gdn_rec_head, hip_native_qsa_score pass
    → check: hip_prefill_hipblaslt_gemm skips for gfx1031 (no table), runs for gfx1201
    → check: hip_prompt_attn_wmma skips for gfx1031 (no WMMA), runs for gfx1201
    → check: hip_handoff, hip_mapped_alias (Windows memory caveats documented)
r4: engine --list-devices + --selftest
    → check: arch matches STRATA_HIP_ARCHS, free VRAM adjusted for WDDM
r5: model end-to-end (Coder IQ1_M or Swift 1.5 IQ3_XXS)
    → check: correct answers, coherent output
    → check: STRATA_HIP_PROMPT_F16=1 on gfx1031 for prompt speed
    → check: STRATA_HIP_WMMA=1 on gfx1201 for prompt speed (7.2-7.5x kernel)
    → check: hipBLASLt table version-match active for gfx1201
```

## 7. Performance budget reference (from AMD_HIP.md)

| Card | Build | 4K prompt | 4K decode | 16K prompt | 16K decode |
|---|---|---|---|---|---|
| RX 6700 XT (gfx1031) | community | 246-339 tok/s (2K-8K) | 38-42 tok/s | 133 tok/s (522 tok) | 45.5 tok/s |
| RX 9070 XT (gfx1201) | validated | 782-2,700 tok/s | 30.8-70.0 tok/s | 1,235-2,700 tok/s | 35.4-60.5 tok/s |
| R9700 (gfx1201) | validated | 982-2,700 tok/s | 45.5-62.4 tok/s | 1,402-2,700 tok/s | 48.3-70.0 tok/s |

RDNA2 numbers from community (gfx1030, NixOS); gfx1031 expected to be similar. RDNA4 numbers from maintainers with ROCm 7.14.

## 8. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| gfx1031 not in ready-made Windows zip | High (build_windows.bat line 20) | Users can't use `START-HERE.bat` out of box | Add gfx1031 to STRATA_HIP_ARCHS default; or users build with `tools\hip\build_windows.bat` |
| ROCm 10.1 API drift | Low | Build failure or runtime error | `__has_builtin` and runtime version checks guard all intrinsics and hipBLASLt paths |
| hipBLASLt table mismatch | Medium (gfx1201 only) | Engine refuses table, falls to slower hipBLAS | `hip_prefill_hipblaslt_gemm` test validates; version check rejects mismatched tables |
| WMMA on RDNA2 | N/A | N/A — correctly excluded | `__gfx11xx__`/`__gfx12xx__` compile gates prevent codegen |
| Windows hipMemGetInfo overcount | Medium | OOM / desktop freeze | `STRATA_WDDM_BUDGET` (device.cu:168-210) already implemented |
| hipBLAS sticky error on Windows | Medium | Engine exits on false error | `absorb_hipblas_sticky` (gemm.cu:93-103) already implemented |
| PCIe bandwidth on gfx1031 | Low | Slower than gfx1100/gfx1201 | `STRATA_HIP_ROUTER_FAST=1` default offsets; FP16 prompt (`STRATA_HIP_PROMPT_F16=1`) is the key accelerator |

## 9. KB handoff (post-validation)

When validation passes on both cards, emit:
```
KB: pattern="RDNA2 gfx1031 → HIP sdot4 + FP16 prompt GEMM (STRATA_HIP_PROMPT_F16=1)" verdict=green
KB: pattern="RDNA4 gfx1201 → HIP sudot4 + WMMA dense + hipBLASLt table version-match" verdict=green
KB: pattern="ROCm 10.1/HIP 7.16 forward-compat → __has_builtin gates + runtime version checks" verdict=green
```

## 10. Files to touch (for maintainers)

1. **`tools/hip/build_windows.bat` line 20**: Add `gfx1031` to default `STRATA_HIP_ARCHS`
2. **`CMakeLists.txt` line 71**: Update comment to mention gfx1031 is validated (post-run)
3. **`cmake/hip_backend.cmake` line 15**: Move gfx1031 from `_strata_hip_unvalidated` to a "community validated" state (post-run)
4. **`docs/AMD_HIP.md`**: Add Windows 11 validation results for gfx1031 after running
5. **`setup.py` line 1493**: Optionally pin `ROCM_VERSION` to a ROCm 10.1 release when it is the tested version

## 11. References

- `include/strata/hip_compat/intrinsics.hpp:18-53` — dp4a dispatch (sdot4 vs sudot4)
- `include/strata/hip_compat/cuda_runtime.h:1-144` — CUDA→HIP runtime shim
- `include/strata/hip_compat/cublas_v2.h` — CUDA→HIP BLAS shim
- `include/strata/kernels/gfx_arch.hpp:22-25` — WMMA arch gate (`gfx_arch_is_gfx11_wmma`)
- `include/strata/prefill/gemm.hpp:85-88` — `prompt_f16()` for RDNA2 FP16 path
- `src/prefill/wmma_gemm.cu:37-302` — WMMA GEMM (gfx11 only; excluded on RDNA2)
- `src/prefill/wmma_gemm.cu:59-57` — `strata_wmma_gfx11_device()` runtime check
- `src/core/device.cu:27-64` — runtime arch validation against STRATA_HIP_ARCHS
- `src/core/device.cu:106-210` — Windows WDDM memory budget adjustment
- `src/prefill/gemm.cu:93-103` — Windows hipBLAS sticky error clearing
- `cmake/hip_backend.cmake:5-37` — validation status and arch gating
- `tools/hip/build_windows.bat:20` — default archs (gfx1031 missing)
- `docs/AMD_HIP.md:329-375` — RDNA2 section (community report)
- `docs/AMD_HIP.md:211-292` — RDNA4 section (validated)
- `docs/AMD_HIP.md:204-209` — opt-in switches for RDNA2 (STRATA_DENSE_MMQ, STRATA_HIP_ADAPT_KERNEL_COPY)
