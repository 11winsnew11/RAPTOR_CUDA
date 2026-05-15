#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <random>
#include <fstream>
#include <vector>

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

// ============================================================
// Device Helpers
// ============================================================
__device__ __forceinline__ int load_found_flag_relaxed(const int *p)
{
    return *((const volatile int *)p);
}

__device__ __forceinline__ bool warp_found_ready(const int *__restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0)
        f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ============================================================
// Constant Memory
// ============================================================
__constant__ uint64_t c_Gx[512 * 4]; // Max HALF=512 (batch=1024)
__constant__ uint64_t c_Gy[512 * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];
__constant__ uint32_t c_vanity_len;
__constant__ uint32_t c_vanity_prefix_mask;

// ============================================================
// Warp Reduce for Hash Counter
// ============================================================
__device__ __forceinline__ unsigned long long warp_reduce_add_ull(unsigned long long val)
{
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xFFFFFFFFu, val, off);
    return val;
}

// ============================================================
// REPORT_MATCH macro — eliminates code duplication
// ============================================================
#define REPORT_MATCH_ADD(delta)                                            \
    do                                                                     \
    {                                                                      \
        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) \
        {                                                                  \
            uint64_t _fs[4];                                               \
            for (int k = 0; k < 4; ++k)                                    \
                _fs[k] = S[k];                                             \
            uint64_t _dv = (uint64_t)(delta);                              \
            for (int k = 0; k < 4 && _dv; ++k)                             \
            {                                                              \
                uint64_t _old = _fs[k];                                    \
                _fs[k] = _old + _dv;                                       \
                _dv = (_fs[k] < _old) ? 1ull : 0ull;                       \
            }                                                              \
            for (int k = 0; k < 4; ++k)                                    \
                d_found_result->scalar[k] = _fs[k];                        \
            for (int k = 0; k < 4; ++k)                                    \
                d_found_result->Rx[k] = _rx[k];                            \
            for (int k = 0; k < 4; ++k)                                    \
                d_found_result->Ry[k] = _ry[k];                            \
            d_found_result->threadId = (int)gid;                           \
            d_found_result->iter = 0;                                      \
            __threadfence_system();                                        \
            atomicExch(d_found_flag, FOUND_READY);                         \
        }                                                                  \
    } while (0)

#define REPORT_MATCH_SUB(delta)                                            \
    do                                                                     \
    {                                                                      \
        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) \
        {                                                                  \
            uint64_t _fs[4];                                               \
            for (int k = 0; k < 4; ++k)                                    \
                _fs[k] = S[k];                                             \
            uint64_t _dv = (uint64_t)(delta);                              \
            for (int k = 0; k < 4 && _dv; ++k)                             \
            {                                                              \
                uint64_t _old = _fs[k];                                    \
                _fs[k] = _old - _dv;                                       \
                _dv = (_old < _dv) ? 1ull : 0ull;                          \
            }                                                              \
            for (int k = 0; k < 4; ++k)                                    \
                d_found_result->scalar[k] = _fs[k];                        \
            for (int k = 0; k < 4; ++k)                                    \
                d_found_result->Rx[k] = _rx[k];                            \
            for (int k = 0; k < 4; ++k)                                    \
                d_found_result->Ry[k] = _ry[k];                            \
            d_found_result->threadId = (int)gid;                           \
            d_found_result->iter = 0;                                      \
            __threadfence_system();                                        \
            atomicExch(d_found_flag, FOUND_READY);                         \
        }                                                                  \
    } while (0)

// ============================================================
// CHECK_HASH_AND_REPORT — inline helper for hash checking
// ============================================================
// _rx, _ry must be in scope; gid, S, d_found_flag, d_found_result also in scope
#define CHECK_HASH_AND_REPORT(px3, odd_byte, is_add, delta_val)                                   \
    do                                                                                            \
    {                                                                                             \
        uint8_t _h20[20];                                                                         \
        getHash160_33_from_limbs((odd_byte), (px3), _h20);                                        \
        ++local_hashes;                                                                           \
        MAYBE_WARP_FLUSH();                                                                       \
        uint32_t _hp = load_u32_le(_h20);                                                         \
        bool _pref = ((_hp & vanity_mask) == (target_prefix & vanity_mask));                      \
        if (__any_sync(full_mask, _pref))                                                         \
        {                                                                                         \
            bool _fm = false;                                                                     \
            if (_pref)                                                                            \
            {                                                                                     \
                if (vanity_len <= 4)                                                              \
                {                                                                                 \
                    _fm = true;                                                                   \
                }                                                                                 \
                else                                                                              \
                {                                                                                 \
                    _fm = true;                                                                   \
                    const uint32_t *_hw = (const uint32_t *)(_h20 + 4);                           \
                    const uint32_t *_tw = (const uint32_t *)(c_target_hash160 + 4);               \
                    uint32_t _wcnt = (vanity_len - 4) / 4;                                        \
                    uint32_t _wrem = (vanity_len - 4) & 3;                                        \
                    for (uint32_t _w = 0; _w < _wcnt; ++_w)                                       \
                    {                                                                             \
                        if (_hw[_w] != _tw[_w])                                                   \
                        {                                                                         \
                            _fm = false;                                                          \
                            break;                                                                \
                        }                                                                         \
                    }                                                                             \
                    if (_fm && _wrem)                                                             \
                    {                                                                             \
                        for (uint32_t _b = 0; _b < _wrem; ++_b)                                   \
                        {                                                                         \
                            if (_h20[4 + _wcnt * 4 + _b] != c_target_hash160[4 + _wcnt * 4 + _b]) \
                            {                                                                     \
                                _fm = false;                                                      \
                                break;                                                            \
                            }                                                                     \
                        }                                                                         \
                    }                                                                             \
                }                                                                                 \
            }                                                                                     \
            if (_fm)                                                                              \
            {                                                                                     \
                uint64_t _rx[4];                                                                  \
                for (int k = 0; k < 4; ++k)                                                       \
                    _rx[k] = (px3)[k];                                                            \
                uint64_t _ry[4]; /* y3 computed inline */                                         \
                ModSub256(_ry, x1, px3);                                                          \
                _ModMult(_ry, _ry, _lam_save);                                                    \
                ModSub256(_ry, _ry, y1);                                                          \
                if (is_add)                                                                       \
                {                                                                                 \
                    REPORT_MATCH_ADD(delta_val);                                                  \
                }                                                                                 \
                else                                                                              \
                {                                                                                 \
                    REPORT_MATCH_SUB(delta_val);                                                  \
                }                                                                                 \
            }                                                                                     \
            __syncwarp(full_mask);                                                                \
            WARP_FLUSH_HASHES();                                                                  \
            return;                                                                               \
        }                                                                                         \
    } while (0)

// ============================================================
// TEMPLATED KERNEL — Key Optimization
// ============================================================
// HALF is a compile-time constant → correct local array sizing
// This eliminates the 16KB subp[MAX_BATCH_SIZE/2][4] problem
template <int HALF>
__launch_bounds__(256, 2)
    __global__ void kernel_point_add_and_check_v2(
        const uint64_t *__restrict__ Px,
        const uint64_t *__restrict__ Py,
        uint64_t *__restrict__ Rx,
        uint64_t *__restrict__ Ry,
        uint64_t *__restrict__ start_scalars,
        uint64_t *__restrict__ counts256,
        uint64_t threadsTotal,
        uint32_t max_batches_per_launch,
        int *__restrict__ d_found_flag,
        FoundResult *__restrict__ d_found_result,
        unsigned long long *__restrict__ hashes_accum,
        unsigned int *__restrict__ d_any_left)
{
    // B is compile-time constant via template
    const int B = HALF * 2;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal)
        return;

    const unsigned lane = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane))
        return;

    const uint32_t target_prefix = c_target_prefix;
    const uint32_t vanity_len = c_vanity_len;
    const uint32_t vanity_mask = c_vanity_prefix_mask;

    unsigned int local_hashes = 0;
#define FLUSH_THRESHOLD 65536u
#define WARP_FLUSH_HASHES()                                                           \
    do                                                                                \
    {                                                                                 \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v)                                                           \
            atomicAdd(hashes_accum, v);                                               \
        local_hashes = 0;                                                             \
    } while (0)
#define MAYBE_WARP_FLUSH()                                 \
    do                                                     \
    {                                                      \
        if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) \
            WARP_FLUSH_HASHES();                           \
    } while (0)

    // Load per-thread state
    uint64_t x1[4], y1[4], S[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i] = start_scalars[idx];
    }
    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
        rem[i] = counts256[gid * 4 + i];

    if ((rem[0] | rem[1] | rem[2] | rem[3]) == 0ull)
    {
#pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            Rx[gid * 4 + i] = x1[i];
            Ry[gid * 4 + i] = y1[i];
        }
        WARP_FLUSH_HASHES();
        return;
    }

    // Load J-point once (used every batch iteration)
    uint64_t Jx_loc[4], Jy_loc[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        Jx_loc[j] = c_Jx[j];
        Jy_loc[j] = c_Jy[j];
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B))
    {
        if (warp_found_ready(d_found_flag, full_mask, lane))
        {
            WARP_FLUSH_HASHES();
            return;
        }

        // ============================================================
        // Check center point P
        // ============================================================
        {
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            uint8_t h20[20];
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes;
            MAYBE_WARP_FLUSH();

            uint32_t h_prefix = load_u32_le(h20);
            bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

            if (__any_sync(full_mask, pref))
            {
                bool full_match = false;
                if (pref)
                {
                    if (vanity_len <= 4)
                    {
                        full_match = true;
                    }
                    else
                    {
                        full_match = true;
                        const uint32_t *hw = (const uint32_t *)(h20 + 4);
                        const uint32_t *tw = (const uint32_t *)(c_target_hash160 + 4);
                        uint32_t wcnt = (vanity_len - 4) / 4;
                        uint32_t wrem = (vanity_len - 4) & 3;
                        for (uint32_t w = 0; w < wcnt; ++w)
                        {
                            if (hw[w] != tw[w])
                            {
                                full_match = false;
                                break;
                            }
                        }
                        if (full_match && wrem)
                        {
                            for (uint32_t b = 0; b < wrem; ++b)
                            {
                                if (h20[4 + wcnt * 4 + b] != c_target_hash160[4 + wcnt * 4 + b])
                                {
                                    full_match = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                if (full_match)
                {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE)
                    {
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter = 0;
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                            d_found_result->scalar[k] = S[k];
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                            d_found_result->Rx[k] = x1[k];
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                            d_found_result->Ry[k] = y1[k];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask);
                WARP_FLUSH_HASHES();
                return;
            }
        }

        // ============================================================
        // Montgomery's Trick: Accumulate sub-products
        // KEY: subp[HALF][4] sized by template, not MAX_BATCH_SIZE/2
        // ============================================================
        uint64_t subp[HALF][4];
        uint64_t acc[4], tmp[4];

        // subp[half-1] = (Jx - x1)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[j] = Jx_loc[j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            subp[HALF - 1][j] = acc[j];

        // subp[i] = product of (Gx[i+1]-x1) * ... * (Gx[half-1]-x1) * (Jx-x1)
        for (int i = HALF - 2; i >= 0; --i)
        {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                tmp[j] = c_Gx[(size_t)(i + 1) * 4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j = 0; j < 4; ++j)
                subp[i][j] = acc[j];
        }

        // Compute full product and invert
        uint64_t inverse[5];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            inverse[j] = c_Gx[0];
        ModSub256(inverse, inverse, x1); // inverse = (Gx[0] - x1)
        _ModMult(inverse, subp[0]);      // inverse = full product
        inverse[4] = 0ull;
        _ModInv(inverse);

        // ============================================================
        // Process +G_i and -G_i for i = 0 .. HALF-2
        // ============================================================
        for (int i = 0; i < HALF - 1; ++i)
        {
            if (warp_found_ready(d_found_flag, full_mask, lane))
            {
                WARP_FLUSH_HASHES();
                return;
            }

            // Compute dx_inv_i for this iteration
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            // --- Load Gx[i], Gy[i] ONCE (shared by +G_i and -G_i) ---
            uint64_t gxi[4], gyi[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                gxi[j] = c_Gx[(size_t)i * 4 + j];
                gyi[j] = c_Gy[(size_t)i * 4 + j];
            }

            // --- Precompute dxi = (Gx[i] - x1) for inverse update ---
            uint64_t dxi[4];
            ModSub256(dxi, gxi, x1);

            // ============================================================
            // Check +G_i: P + (i+1)*G, scalar = S + (i+1)
            // ============================================================
            {
                uint64_t lam[4], px3[4], s[4];

                // λ = (Gy[i] - y1) / (Gx[i] - x1)
                ModSub256(s, gyi, y1);
                _ModMult(lam, s, dx_inv_i);

                // x3 = λ² - x1 - Gx[i]
                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, gxi);

                // y3 parity: (x1 - x3)*λ - y1
                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd;
                ModSub256isOdd(s, y1, &odd);

                // Hash check — use word-by-word comparison
                uint8_t h20[20];
                getHash160_33_from_limbs(odd ? 0x03 : 0x02, px3, h20);
                ++local_hashes;
                MAYBE_WARP_FLUSH();

                uint32_t h_prefix = load_u32_le(h20);
                bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

                if (__any_sync(full_mask, pref))
                {
                    bool full_match = false;
                    if (pref)
                    {
                        if (vanity_len <= 4)
                        {
                            full_match = true;
                        }
                        else
                        {
                            full_match = true;
                            const uint32_t *hw = (const uint32_t *)(h20 + 4);
                            const uint32_t *tw = (const uint32_t *)(c_target_hash160 + 4);
                            uint32_t wcnt = (vanity_len - 4) / 4;
                            uint32_t wrem = (vanity_len - 4) & 3;
                            for (uint32_t w = 0; w < wcnt; ++w)
                            {
                                if (hw[w] != tw[w])
                                {
                                    full_match = false;
                                    break;
                                }
                            }
                            if (full_match && wrem)
                            {
                                for (uint32_t b = 0; b < wrem; ++b)
                                {
                                    if (h20[4 + wcnt * 4 + b] != c_target_hash160[4 + wcnt * 4 + b])
                                    {
                                        full_match = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (full_match)
                    {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE)
                        {
                            uint64_t fs[4];
                            for (int k = 0; k < 4; ++k)
                                fs[k] = S[k];
                            uint64_t addv = (uint64_t)(i + 1);
                            for (int k = 0; k < 4 && addv; ++k)
                            {
                                uint64_t old = fs[k];
                                fs[k] = old + addv;
                                addv = (fs[k] < old) ? 1ull : 0ull;
                            }
                            for (int k = 0; k < 4; ++k)
                                d_found_result->scalar[k] = fs[k];
                            for (int k = 0; k < 4; ++k)
                                d_found_result->Rx[k] = px3[k];
                            uint64_t y3[4], t[4];
                            ModSub256(t, x1, px3);
                            _ModMult(y3, t, lam);
                            ModSub256(y3, y3, y1);
                            for (int k = 0; k < 4; ++k)
                                d_found_result->Ry[k] = y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask);
                    WARP_FLUSH_HASHES();
                    return;
                }
            }

            // ============================================================
            // Check -G_i: P - (i+1)*G, scalar = S - (i+1)
            // Reuses gxi[], gyi[] (negated), dxi[], dx_inv_i[]
            // ============================================================
            {
                uint64_t lam[4], px3[4], s[4];

                // Negate Gy[i] for -G_i
                uint64_t neg_gyi[4];
                ModNeg256(neg_gyi, gyi);

                // λ = (-Gy[i] - y1) / (Gx[i] - x1)
                ModSub256(s, neg_gyi, y1);
                _ModMult(lam, s, dx_inv_i);

                // x3 = λ² - x1 - Gx[i]
                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, gxi);

                // y3 parity
                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd;
                ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20];
                getHash160_33_from_limbs(odd ? 0x03 : 0x02, px3, h20);
                ++local_hashes;
                MAYBE_WARP_FLUSH();

                uint32_t h_prefix = load_u32_le(h20);
                bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

                if (__any_sync(full_mask, pref))
                {
                    bool full_match = false;
                    if (pref)
                    {
                        if (vanity_len <= 4)
                        {
                            full_match = true;
                        }
                        else
                        {
                            full_match = true;
                            const uint32_t *hw = (const uint32_t *)(h20 + 4);
                            const uint32_t *tw = (const uint32_t *)(c_target_hash160 + 4);
                            uint32_t wcnt = (vanity_len - 4) / 4;
                            uint32_t wrem = (vanity_len - 4) & 3;
                            for (uint32_t w = 0; w < wcnt; ++w)
                            {
                                if (hw[w] != tw[w])
                                {
                                    full_match = false;
                                    break;
                                }
                            }
                            if (full_match && wrem)
                            {
                                for (uint32_t b = 0; b < wrem; ++b)
                                {
                                    if (h20[4 + wcnt * 4 + b] != c_target_hash160[4 + wcnt * 4 + b])
                                    {
                                        full_match = false;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                    if (full_match)
                    {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE)
                        {
                            uint64_t fs[4];
                            for (int k = 0; k < 4; ++k)
                                fs[k] = S[k];
                            uint64_t subv = (uint64_t)(i + 1);
                            for (int k = 0; k < 4 && subv; ++k)
                            {
                                uint64_t old = fs[k];
                                fs[k] = old - subv;
                                subv = (old < subv) ? 1ull : 0ull;
                            }
                            for (int k = 0; k < 4; ++k)
                                d_found_result->scalar[k] = fs[k];
                            for (int k = 0; k < 4; ++k)
                                d_found_result->Rx[k] = px3[k];
                            uint64_t y3[4], t[4];
                            ModSub256(t, x1, px3);
                            _ModMult(y3, t, lam);
                            ModSub256(y3, y3, y1);
                            for (int k = 0; k < 4; ++k)
                                d_found_result->Ry[k] = y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask);
                    WARP_FLUSH_HASHES();
                    return;
                }
            }

            // --- Inverse update: strip dxi from running inverse ---
            // dxi was precomputed as (Gx[i] - x1), no need to recompute
            _ModMult(inverse, inverse, dxi);
        }

        // ============================================================
        // Process -G_{half-1}: P - half*G, scalar = S - half
        // ============================================================
        {
            const int i = HALF - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t gxi[4], gyi[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                gxi[j] = c_Gx[(size_t)i * 4 + j];
                gyi[j] = c_Gy[(size_t)i * 4 + j];
            }
            uint64_t dxi[4];
            ModSub256(dxi, gxi, x1);

            uint64_t lam[4], px3[4], s[4];
            uint64_t neg_gyi[4];
            ModNeg256(neg_gyi, gyi);

            ModSub256(s, neg_gyi, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256(px3, px3, x1);
            ModSub256(px3, px3, gxi);

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            uint8_t odd;
            ModSub256isOdd(s, y1, &odd);

            uint8_t h20[20];
            getHash160_33_from_limbs(odd ? 0x03 : 0x02, px3, h20);
            ++local_hashes;
            MAYBE_WARP_FLUSH();

            uint32_t h_prefix = load_u32_le(h20);
            bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

            if (__any_sync(full_mask, pref))
            {
                bool full_match = false;
                if (pref)
                {
                    if (vanity_len <= 4)
                    {
                        full_match = true;
                    }
                    else
                    {
                        full_match = true;
                        const uint32_t *hw = (const uint32_t *)(h20 + 4);
                        const uint32_t *tw = (const uint32_t *)(c_target_hash160 + 4);
                        uint32_t wcnt = (vanity_len - 4) / 4;
                        uint32_t wrem = (vanity_len - 4) & 3;
                        for (uint32_t w = 0; w < wcnt; ++w)
                        {
                            if (hw[w] != tw[w])
                            {
                                full_match = false;
                                break;
                            }
                        }
                        if (full_match && wrem)
                        {
                            for (uint32_t b = 0; b < wrem; ++b)
                            {
                                if (h20[4 + wcnt * 4 + b] != c_target_hash160[4 + wcnt * 4 + b])
                                {
                                    full_match = false;
                                    break;
                                }
                            }
                        }
                    }
                }
                if (full_match)
                {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE)
                    {
                        uint64_t fs[4];
                        for (int k = 0; k < 4; ++k)
                            fs[k] = S[k];
                        uint64_t sub_val = (uint64_t)HALF;
                        for (int k = 0; k < 4 && sub_val; ++k)
                        {
                            uint64_t old = fs[k];
                            fs[k] = old - sub_val;
                            sub_val = (old < sub_val) ? 1ull : 0ull;
                        }
                        for (int k = 0; k < 4; ++k)
                            d_found_result->scalar[k] = fs[k];
                        for (int k = 0; k < 4; ++k)
                            d_found_result->Rx[k] = px3[k];
                        uint64_t y3[4], t[4];
                        ModSub256(t, x1, px3);
                        _ModMult(y3, t, lam);
                        ModSub256(y3, y3, y1);
                        for (int k = 0; k < 4; ++k)
                            d_found_result->Ry[k] = y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter = 0;
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask);
                WARP_FLUSH_HASHES();
                return;
            }

            // Strip last denominator from inverse for jump step
            _ModMult(inverse, inverse, dxi);
        }

        // ============================================================
        // Jump Step: P ← P + B*G
        // inverse now = 1/(Jx - x1)
        // ============================================================
        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                Jy_minus_y1[j] = Jy_loc[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            ModSub256(x3, x3, Jx_loc);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                x1[j] = x3[j];
                y1[j] = y3[j];
            }
        }

        // Update scalar and remaining count
        {
            uint64_t addv = (uint64_t)B;
            for (int k = 0; k < 4 && addv; ++k)
            {
                uint64_t old = S[k];
                S[k] = old + addv;
                addv = (S[k] < old) ? 1ull : 0ull;
            }
            sub256_u64_inplace(rem, (uint64_t)B);
        }
        ++batches_done;
    }

    // Write back state
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        Rx[gid * 4 + i] = x1[i];
        Ry[gid * 4 + i] = y1[i];
        counts256[gid * 4 + i] = rem[i];
        start_scalars[gid * 4 + i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull)
    {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
#undef MAYBE_WARP_FLUSH
#undef WARP_FLUSH_HASHES
#undef FLUSH_THRESHOLD
}

// ============================================================
// Kernel Dispatch — selects correct template instantiation
// ============================================================
static bool launch_kernel_v2(
    uint32_t batch_size,
    int blocks, int tpb, cudaStream_t stream,
    const uint64_t *Px, const uint64_t *Py,
    uint64_t *Rx, uint64_t *Ry,
    uint64_t *start_scalars, uint64_t *counts256,
    uint64_t threadsTotal,
    uint32_t max_batches_per_launch,
    int *d_found_flag, FoundResult *d_found_result,
    unsigned long long *hashes_accum,
    unsigned int *d_any_left)
{
    int half = batch_size / 2;
    switch (half)
    {
    case 16:
        kernel_point_add_and_check_v2<16><<<blocks, tpb, 0, stream>>>(
            Px, Py, Rx, Ry, start_scalars, counts256, threadsTotal,
            max_batches_per_launch, d_found_flag, d_found_result, hashes_accum, d_any_left);
        return true;
    case 32:
        kernel_point_add_and_check_v2<32><<<blocks, tpb, 0, stream>>>(
            Px, Py, Rx, Ry, start_scalars, counts256, threadsTotal,
            max_batches_per_launch, d_found_flag, d_found_result, hashes_accum, d_any_left);
        return true;
    case 64:
        kernel_point_add_and_check_v2<64><<<blocks, tpb, 0, stream>>>(
            Px, Py, Rx, Ry, start_scalars, counts256, threadsTotal,
            max_batches_per_launch, d_found_flag, d_found_result, hashes_accum, d_any_left);
        return true;
    case 128:
        kernel_point_add_and_check_v2<128><<<blocks, tpb, 0, stream>>>(
            Px, Py, Rx, Ry, start_scalars, counts256, threadsTotal,
            max_batches_per_launch, d_found_flag, d_found_result, hashes_accum, d_any_left);
        return true;
    case 256:
        kernel_point_add_and_check_v2<256><<<blocks, tpb, 0, stream>>>(
            Px, Py, Rx, Ry, start_scalars, counts256, threadsTotal,
            max_batches_per_launch, d_found_flag, d_found_result, hashes_accum, d_any_left);
        return true;
    case 512:
        kernel_point_add_and_check_v2<512><<<blocks, tpb, 0, stream>>>(
            Px, Py, Rx, Ry, start_scalars, counts256, threadsTotal,
            max_batches_per_launch, d_found_flag, d_found_result, hashes_accum, d_any_left);
        return true;
    default:
        std::cerr << "Error: Unsupported half batch size: " << half << "\n";
        return false;
    }
}
```

// ============================================================
// External declarations
// ============================================================
extern bool hexToLE64(const std::string &h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string &h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern bool decode_p2pkh_address(const std::string &addr, uint8_t out20[20]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
extern std::string human_bytes(double bytes);
__global__ void scalarMulKernelBase(const uint64_t *scalars_in, uint64_t *outX, uint64_t *outY, int N);

// ============================================================
// State Management for Resume Support
// ============================================================
#pragma pack(push, 1)
struct ScanState
{
    uint32_t magic;
    uint32_t version;
    uint64_t random_idx;
    uint32_t sub_random;
    uint64_t total_hashes;
    uint64_t rng_seed;
    uint32_t checksum;
};
#pragma pack(pop)

static const uint32_t STATE_MAGIC = 0x5343414E; // "SCAN"
static const uint32_t STATE_VERSION = 2;

static uint32_t calcStateChecksum(const ScanState &s)
{
    uint32_t c = s.magic;
    c ^= s.version;
    c ^= (uint32_t)(s.random_idx & 0xFFFFFFFF);
    c ^= (uint32_t)(s.random_idx >> 32);
    c ^= s.sub_random;
    c ^= (uint32_t)(s.total_hashes & 0xFFFFFFFF);
    c ^= (uint32_t)(s.total_hashes >> 32);
    c ^= (uint32_t)(s.rng_seed & 0xFFFFFFFF);
    c ^= (uint32_t)(s.rng_seed >> 32);
    return c;
}

static void saveState(const std::string &filename, uint64_t r_idx, uint32_t s, uint64_t hashes, uint64_t seed)
{
    ScanState state;
    state.magic = STATE_MAGIC;
    state.version = STATE_VERSION;
    state.random_idx = r_idx;
    state.sub_random = s;
    state.total_hashes = hashes;
    state.rng_seed = seed;
    state.checksum = calcStateChecksum(state);

    std::ofstream f(filename, std::ios::binary);
    if (f.good())
    {
        f.write((const char *)&state, sizeof(state));
    }
}

static bool loadState(const std::string &filename, uint64_t &r_idx, uint32_t &s, uint64_t &hashes, uint64_t &seed)
{
    std::ifstream f(filename, std::ios::binary);
    if (!f.good())
        return false;

    ScanState state;
    if (!f.read((char *)&state, sizeof(state)))
        return false;
    if (state.magic != STATE_MAGIC)
        return false;
    if (state.version != STATE_VERSION)
        return false;
    if (state.checksum != calcStateChecksum(state))
        return false;

    r_idx = state.random_idx;
    s = state.sub_random;
    hashes = state.total_hashes;
    seed = state.rng_seed;
    return true;
}

// ============================================================
// Galois Field (Polynomial Extreme) Random Permutation
// ============================================================
class PolynomialExtremeRNG
{
private:
    uint64_t A0;
    uint64_t B0;

    static uint64_t gf64_clmul(uint64_t a, uint64_t b)
    {
        uint64_t result = 0;
        while (b)
        {
            if (b & 1)
                result ^= a;
            bool hi_bit = (a >> 63) & 1;
            a <<= 1;
            if (hi_bit)
                a ^= 0x1B;
            b >>= 1;
        }
        return result;
    }

public:
    PolynomialExtremeRNG(uint64_t seed)
    {
        A0 = seed ^ 0xA5A5A5A5A5A5A5A5ULL;
        if (A0 == 0)
            A0 = 0xAAAAAAAAAAAAAAAAULL;
        B0 = (~seed) ^ 0x5A5A5A5A5A5A5A5AULL;
    }

    uint64_t permute(uint64_t r_idx, int random_bits)
    {
        if (random_bits == 0)
            return 0;
        if (random_bits == 64)
        {
            return gf64_clmul(A0, r_idx) ^ B0;
        }

        int half_bits = random_bits / 2;
        uint64_t mask = (1ULL << half_bits) - 1;
        uint64_t L = (r_idx >> half_bits) & mask;
        uint64_t R = r_idx & mask;

        uint64_t A = A0;
        uint64_t B = B0;

        for (int i = 0; i < 4; ++i)
        {
            uint64_t F_R = (gf64_clmul(A, R) ^ B) & mask;
            uint64_t tmp = L ^ F_R;
            L = R;
            R = tmp;
            A = gf64_clmul(A, 0x123456789ABCDEF0ULL);
            B = (B << 1) | (B >> 63);
        }

        return (L << half_bits) | R;
    }
};

// ============================================================
// Build hex string for slice boundaries
// ============================================================
static std::string buildSliceHex(int common_hex, uint32_t prefix_nibble,
                                 uint64_t R, int random_hex,
                                 uint64_t S, int sub_hex,
                                 uint64_t L, int linear_hex)
{
    char buf[128];
    if (common_hex > 0)
    {
        snprintf(buf, sizeof(buf), "%0*x%0*llx%0*llx%0*llx",
                 common_hex, (unsigned)prefix_nibble,
                 random_hex, (unsigned long long)R,
                 sub_hex, (unsigned long long)S,
                 linear_hex, (unsigned long long)L);
    }
    else
    {
        snprintf(buf, sizeof(buf), "%0*llx%0*llx%0*llx",
                 random_hex, (unsigned long long)R,
                 sub_hex, (unsigned long long)S,
                 linear_hex, (unsigned long long)L);
    }
    return std::string(buf);
}

// ============================================================
// Main Function
// ============================================================
int main(int argc, char **argv)
{
    std::signal(SIGINT, handle_sigint);

    // --- Configuration ---
    std::string vanity_hex;
    std::string range_hex;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm = 8;
    uint32_t slices_per_launch = 64;
    int forced_random_bits = -1;
    int forced_linear_bits = -1;
    int sub_random_bits = 8;
    std::string state_file = "vanity_state.bin";
    bool no_shuffle = false;
    uint64_t rng_seed = 0;
    bool has_seed = false;

    auto parse_grid = [](const std::string &s, uint32_t &a_out, uint32_t &b_out) -> bool
    {
        size_t comma = s.find(',');
        if (comma == std::string::npos)
            return false;
        auto trim = [](std::string &z)
        {
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos)
            {
                z.clear();
                return;
            }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str);
        trim(b_str);
        if (a_str.empty() || b_str.empty())
            return false;
        char *endp = nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10);
        if (*endp)
            return false;
        endp = nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10);
        if (*endp)
            return false;
        if (aa == 0ul || bb == 0ul)
            return false;
        if (aa > (1ul << 20) || bb > (1ul << 20))
            return false;
        a_out = (uint32_t)aa;
        b_out = (uint32_t)bb;
        return true;
    };

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--vanity-hash160" && i + 1 < argc)
            vanity_hex = argv[++i];
        else if (arg == "--range" && i + 1 < argc)
            range_hex = argv[++i];
        else if (arg == "--grid" && i + 1 < argc)
        {
            uint32_t a = 0, b = 0;
            if (!parse_grid(argv[++i], a, b))
            {
                std::cerr << "Error: --grid expects \"A,B\" (positive integers).\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm = b;
        }
        else if (arg == "--slices" && i + 1 < argc)
        {
            char *endp = nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul << 20))
            {
                std::cerr << "Error: --slices must be in 1.." << (1u << 20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = (uint32_t)v;
        }
        else if (arg == "--random-bits" && i + 1 < argc)
        {
            forced_random_bits = std::atoi(argv[++i]);
            if (forced_random_bits < 0 || forced_random_bits > 60)
            {
                std::cerr << "Error: --random-bits must be 0-60\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "--linear-bits" && i + 1 < argc)
        {
            forced_linear_bits = std::atoi(argv[++i]);
            if (forced_linear_bits < 16 || forced_linear_bits > 64)
            {
                std::cerr << "Error: --linear-bits must be 16-64\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "--sub-random-bits" && i + 1 < argc)
        {
            sub_random_bits = std::atoi(argv[++i]);
            if (sub_random_bits < 0 || sub_random_bits > 16 || (sub_random_bits % 4 != 0))
            {
                std::cerr << "Error: --sub-random-bits must be 0-16 and multiple of 4\n";
                return EXIT_FAILURE;
            }
        }
        else if (arg == "--state-file" && i + 1 < argc)
        {
            state_file = argv[++i];
        }
        else if (arg == "--seed" && i + 1 < argc)
        {
            rng_seed = std::stoull(argv[++i], nullptr, 16);
            has_seed = true;
        }
        else if (arg == "--no-shuffle")
        {
            no_shuffle = true;
        }
    }

    if (range_hex.empty() || vanity_hex.empty())
    {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> --vanity-hash160 <prefix_hex>\n"
                  << "  [--grid A,B] [--slices N]\n"
                  << "  [--random-bits N] [--linear-bits N] [--sub-random-bits N]\n"
                  << "  [--state-file file] [--seed hex64] [--no-shuffle]\n";
        return EXIT_FAILURE;
    }

    // --- Parsing Vanity Hash160 ---
    if (vanity_hex.length() > 40)
    {
        std::cerr << "Error: Vanity prefix cannot exceed 20 bytes (40 hex chars).\n";
        return EXIT_FAILURE;
    }
    if (vanity_hex.length() % 2 != 0)
    {
        std::cerr << "Error: Vanity prefix length must be even (full bytes).\n";
        return EXIT_FAILURE;
    }

    uint8_t target_hash160[20] = {0};
    uint32_t vanity_len = vanity_hex.length() / 2;

    std::string padded_vanity = vanity_hex;
    if (padded_vanity.length() < 40)
    {
        padded_vanity += std::string(40 - padded_vanity.length(), '0');
    }

    if (!hexToHash160(padded_vanity, target_hash160))
    {
        std::cerr << "Error: Invalid hex for vanity prefix.\n";
        return EXIT_FAILURE;
    }

    uint32_t vanity_prefix_mask = 0xFFFFFFFFu;
    if (vanity_len < 4)
    {
        vanity_prefix_mask = (1ULL << (vanity_len * 8)) - 1;
    }

    std::cout << "Searching for Vanity Prefix: " << vanity_hex << " (" << vanity_len << " bytes)\n";

    // --- Parse Range & Calculate Bit Split ---
    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos)
    {
        std::cerr << "Error: range format must be start:end\n";
        return EXIT_FAILURE;
    }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex = range_hex.substr(colon_pos + 1);

    int total_hex_chars = start_hex.length();
    if (end_hex.length() != total_hex_chars)
    {
        std::cerr << "Error: start and end hex must have same length\n";
        return EXIT_FAILURE;
    }
    int total_bits = total_hex_chars * 4;

    int common_hex = 0;
    while (common_hex < total_hex_chars && start_hex[common_hex] == end_hex[common_hex])
    {
        common_hex++;
    }

    int prefix_bits = common_hex * 4;
    int variable_bits = total_bits - prefix_bits;

    uint32_t prefix_nibble = 0;
    if (common_hex > 0)
    {
        prefix_nibble = std::stoul(start_hex.substr(0, common_hex), nullptr, 16);
    }

    int linear_bits, random_bits;

    if (forced_linear_bits > 0)
    {
        linear_bits = forced_linear_bits;
    }
    else
    {
        linear_bits = 32;
    }

    linear_bits = (linear_bits + 3) & ~3;

    bool use_sub_chunking = true;

    if (variable_bits <= linear_bits)
    {
        random_bits = 0;
        sub_random_bits = 0;
        linear_bits = variable_bits;
        use_sub_chunking = false;
    }
    else if (forced_random_bits >= 0)
    {
        random_bits = forced_random_bits;
        int remaining = variable_bits - random_bits - linear_bits;
        if (remaining > 0 && sub_random_bits > 0)
        {
            sub_random_bits = (remaining < sub_random_bits) ? remaining : sub_random_bits;
            sub_random_bits = (sub_random_bits + 3) & ~3;
        }
        else
        {
            sub_random_bits = 0;
        }
    }
    else
    {
        int target_sub = (sub_random_bits > 0) ? sub_random_bits : 0;
        int needed = variable_bits - linear_bits - target_sub;

        if (needed >= 0)
        {
            random_bits = needed;
            sub_random_bits = target_sub;
        }
        else
        {
            sub_random_bits = 0;
            needed = variable_bits - linear_bits;
            if (needed < 0)
            {
                linear_bits = variable_bits;
                random_bits = 0;
            }
            else
            {
                random_bits = needed;
            }
        }
    }

    random_bits = (random_bits + 3) & ~3;

    if (random_bits + sub_random_bits + linear_bits != variable_bits)
    {
        std::cerr << "Error: Bit split mismatch. Variable=" << variable_bits
                  << " but random(" << random_bits << ") + sub(" << sub_random_bits
                  << ") + linear(" << linear_bits << ") = " << (random_bits + sub_random_bits + linear_bits) << "\n";
        return EXIT_FAILURE;
    }

    int random_hex_chars = random_bits / 4;
    int sub_hex_chars = sub_random_bits / 4;
    int linear_hex_chars = linear_bits / 4;

    uint64_t num_random_slices = (random_bits > 0) ? (1ULL << random_bits) : 1;
    uint64_t num_sub_slices = (sub_random_bits > 0) ? (1ULL << sub_random_bits) : 1;
    uint64_t keys_per_slice = 1ULL << linear_bits;
    uint64_t total_slices = num_random_slices * num_sub_slices;
    uint64_t total_keys = total_slices * keys_per_slice;

    std::cout << "======== Range Analysis =================================\n";
    std::cout << std::left << std::setw(25) << "Total range" << " : " << total_hex_chars << " hex (" << total_bits << " bits)\n";
    std::cout << std::left << std::setw(25) << "Common prefix" << " : " << (common_hex > 0 ? start_hex.substr(0, common_hex) : "(none)")
              << " (" << prefix_bits << " bits)\n";
    std::cout << std::left << std::setw(25) << "Variable bits" << " : " << variable_bits << "\n";
    std::cout << "-----------------------------------------------------------\n";
    std::cout << std::left << std::setw(25) << "Random bits" << " : " << random_bits << " (2^" << random_bits << " = " << num_random_slices << " slices)\n";
    std::cout << std::left << std::setw(25) << "Sub-random bits" << " : " << sub_random_bits << " (2^" << sub_random_bits << " = " << num_sub_slices << " sub-slices)\n";
    std::cout << std::left << std::setw(25) << "Linear bits" << " : " << linear_bits << " (2^" << linear_bits << " = " << keys_per_slice << " keys/slice)\n";
    std::cout << std::left << std::setw(25) << "Total slices" << " : " << total_slices << "\n";
    std::cout << std::left << std::setw(25) << "Total keys" << " : " << total_keys << " (2^" << (random_bits + sub_random_bits + linear_bits) << ")\n";
    std::cout << std::left << std::setw(25) << "Shuffle Algo" << " : " << (no_shuffle ? "Disabled" : "Galois-Field Polynomial Extreme") << "\n";
    std::cout << std::left << std::setw(25) << "State file" << " : " << state_file << "\n\n";

    // --- Validate batch size ---
    auto is_pow2 = [](uint32_t v) -> bool
    { return v && ((v & (v - 1)) == 0); };
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u))
    {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > 1024)
    {
        std::cerr << "Error: batch size must be <= 1024 (kernel limit).\n";
        return EXIT_FAILURE;
    }

    if (keys_per_slice % runtime_points_batch_size != 0)
    {
        std::cerr << "Error: keys_per_slice (2^" << linear_bits << ") must be divisible by batch_size ("
                  << runtime_points_batch_size << ")\n";
        return EXIT_FAILURE;
    }

    // --- GPU Setup ---
    int device = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&device) != cudaSuccess || cudaGetDeviceProperties(&prop, device) != cudaSuccess)
    {
        std::cerr << "CUDA init error\n";
        return EXIT_FAILURE;
    }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int threadsPerBlock = 256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock)
        threadsPerBlock = prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32)
        threadsPerBlock = 32;

    uint64_t slice_len[4] = {0, 0, 0, 0};
    slice_len[0] = (linear_bits == 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << linear_bits) - 1);
    add256_u64(slice_len, 1ull, slice_len);

    const uint64_t bytesPerThread = 2ull * 4ull * sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    const uint64_t reserveBytes = 128ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    uint64_t q_div_batch[4], r_div_batch = 0ull;
    divmod_256_by_u64(slice_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
    if (r_div_batch != 0ull)
    {
        std::cerr << "Error: slice length not divisible by batch size.\n";
        return EXIT_FAILURE;
    }
    bool q_fits_u64 = (q_div_batch[3] | q_div_batch[2] | q_div_batch[1]) == 0ull;
    uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
    if (!q_fits_u64)
    {
        std::cerr << "Error: total batches too large for u64.\n";
        return EXIT_FAILURE;
    }

    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull)
        userUpper = UINT64_MAX;

    auto pick_threads_total = [&](uint64_t upper) -> uint64_t
    {
        if (upper < (uint64_t)threadsPerBlock)
            return 0ull;
        uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
        uint64_t q = total_batches_u64;
        while (t >= (uint64_t)threadsPerBlock)
        {
            if ((q % t) == 0ull)
                return t;
            t -= (uint64_t)threadsPerBlock;
        }
        return 0ull;
    };

    uint64_t upper = maxThreadsByMem;
    if (total_batches_u64 < upper)
        upper = total_batches_u64;
    if (userUpper < upper)
        upper = userUpper;

    uint64_t threadsTotal = pick_threads_total(upper);
    if (threadsTotal == 0ull)
    {
        std::cerr << "Error: failed to pick threadsTotal satisfying divisibility.\n";
        return EXIT_FAILURE;
    }
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    uint64_t per_thread_cnt[4];
    uint64_t r_u64 = 0ull;
    divmod_256_by_u64(slice_len, threadsTotal, per_thread_cnt, r_u64);
    if (r_u64 != 0ull)
    {
        std::cerr << "Internal error: slice_len not divisible by threadsTotal.\n";
        return EXIT_FAILURE;
    }

    // --- Allocate Host Buffers ---
    uint64_t *h_counts256 = nullptr;
    uint64_t *h_start_scalars = nullptr;
    cudaHostAlloc(&h_counts256, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc(&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    for (uint64_t i = 0; i < threadsTotal; ++i)
    {
        h_counts256[i * 4 + 0] = per_thread_cnt[0];
        h_counts256[i * 4 + 1] = per_thread_cnt[1];
        h_counts256[i * 4 + 2] = per_thread_cnt[2];
        h_counts256[i * 4 + 3] = per_thread_cnt[3];
    }

    const uint32_t B = runtime_points_batch_size;
    const uint32_t half = B >> 1;

    // --- Allocate Device Buffers ---
    uint64_t *d_start_scalars = nullptr, *d_Px = nullptr, *d_Py = nullptr, *d_Rx = nullptr, *d_Ry = nullptr, *d_counts256 = nullptr;
    int *d_found_flag = nullptr;
    FoundResult *d_found_result = nullptr;
    unsigned long long *d_hashes_accum = nullptr;
    unsigned int *d_any_left = nullptr;

    auto ck = [](cudaError_t e, const char *msg)
    {
        if (e != cudaSuccess)
        {
            std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    ck(cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_start_scalars)");
    ck(cudaMalloc(&d_Px, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Px)");
    ck(cudaMalloc(&d_Py, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Py)");
    ck(cudaMalloc(&d_Rx, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Rx)");
    ck(cudaMalloc(&d_Ry, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Ry)");
    ck(cudaMalloc(&d_counts256, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_counts256)");
    ck(cudaMalloc(&d_found_flag, sizeof(int)), "cudaMalloc(d_found_flag)");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)), "cudaMalloc(d_found_result)");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)), "cudaMalloc(d_hashes_accum)");
    ck(cudaMalloc(&d_any_left, sizeof(unsigned int)), "cudaMalloc(d_any_left)");

    ck(cudaMemcpy(d_counts256, h_counts256, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts256");

    {
        int zero = FOUND_NONE;
        unsigned long long zero64 = 0ull;
        ck(cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "init found_flag");
        ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes_accum");
    }

    // --- Copy Vanity Config ---
    {
        uint32_t prefix_le = (uint32_t)target_hash160[0] | ((uint32_t)target_hash160[1] << 8) | ((uint32_t)target_hash160[2] << 16) | ((uint32_t)target_hash160[3] << 24);

        cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
        cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
        cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(vanity_len));
        cudaMemcpyToSymbol(c_vanity_prefix_mask, &vanity_prefix_mask, sizeof(vanity_prefix_mask));
    }

    // --- Create Stream First for Precomputations ---
    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "create stream");

    // --- Precompute G points (1..B/2) ---
    {
        uint64_t *h_scalars_half = nullptr;
        cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < half; ++k)
            h_scalars_half[(size_t)k * 4 + 0] = (uint64_t)(k + 1);

        uint64_t *d_scalars_half = nullptr, *d_Gx_half = nullptr, *d_Gy_half = nullptr;
        ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_scalars_half)");
        ck(cudaMalloc(&d_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gx_half)");
        ck(cudaMalloc(&d_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gy_half)");
        ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy half scalars");

        int blocks_scal = (int)((half + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock, 0, streamKernel>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
        ck(cudaStreamSynchronize(streamKernel), "scalarMulKernelBase(half) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(half) launch");

        uint64_t *h_Gx_half = (uint64_t *)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        uint64_t *h_Gy_half = (uint64_t *)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx_half");
        ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy_half");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

        cudaFree(d_scalars_half);
        cudaFree(d_Gx_half);
        cudaFree(d_Gy_half);
        cudaFreeHost(h_scalars_half);
        std::free(h_Gx_half);
        std::free(h_Gy_half);
    }

    // --- Precompute J point (batch_size * G) ---
    {
        uint64_t *h_scalarB = nullptr;
        cudaHostAlloc(&h_scalarB, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalarB, 0, 4 * sizeof(uint64_t));
        h_scalarB[0] = (uint64_t)B;

        uint64_t *d_scalarB = nullptr, *d_Jx = nullptr, *d_Jy = nullptr;
        ck(cudaMalloc(&d_scalarB, 4 * sizeof(uint64_t)), "cudaMalloc(d_scalarB)");
        ck(cudaMalloc(&d_Jx, 4 * sizeof(uint64_t)), "cudaMalloc(d_Jx)");
        ck(cudaMalloc(&d_Jy, 4 * sizeof(uint64_t)), "cudaMalloc(d_Jy)");
        ck(cudaMemcpy(d_scalarB, h_scalarB, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scalarB");

        scalarMulKernelBase<<<1, 1, 0, streamKernel>>>(d_scalarB, d_Jx, d_Jy, 1);
        ck(cudaStreamSynchronize(streamKernel), "scalarMulKernelBase(B) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(B) launch");

        uint64_t hJx[4], hJy[4];
        ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
        ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");
        ck(cudaMemcpyToSymbol(c_Jx, hJx, 4 * sizeof(uint64_t)), "ToSymbol c_Jx");
        ck(cudaMemcpyToSymbol(c_Jy, hJy, 4 * sizeof(uint64_t)), "ToSymbol c_Jy");

        cudaFree(d_scalarB);
        cudaFree(d_Jx);
        cudaFree(d_Jy);
        cudaFreeHost(h_scalarB);
    }

    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_v2<64>, cudaFuncCachePreferL1);

    // --- Memory Info ---
    size_t freeB = 0, totalB = 0;
    cudaMemGetInfo(&freeB, &totalB);
    size_t usedB = totalB - freeB;
    double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;

    std::cout << "======== GPU Information =================================\n";
    std::cout << std::left << std::setw(25) << "Device" << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(25) << "SM" << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(25) << "ThreadsPerBlock" << " : " << threadsPerBlock << "\n";
    std::cout << std::left << std::setw(25) << "Blocks" << " : " << blocks << "\n";
    std::cout << std::left << std::setw(25) << "Total threads" << " : " << threadsTotal << "\n";
    std::cout << std::left << std::setw(25) << "Points batch size" << " : " << B << "\n";
    std::cout << std::left << std::setw(25) << "Batches/launch" << " : " << slices_per_launch << " (per thread)\n";
    std::cout << std::left << std::setw(25) << "Memory utilization" << " : "
              << std::fixed << std::setprecision(1) << util << "% ("
              << human_bytes((double)usedB) << " / " << human_bytes((double)totalB) << ")\n\n";

    // --- Load State for Resume Support ---
    uint64_t start_r_idx = 0;
    uint32_t start_s_val = 0;
    uint64_t start_total_hashes = 0;
    bool resuming = false;

    if (loadState(state_file, start_r_idx, start_s_val, start_total_hashes, rng_seed))
    {
        if (start_r_idx < num_random_slices && start_s_val < num_sub_slices)
        {
            resuming = true;
            has_seed = true;
            std::cout << "======== Resuming from State ==========================\n";
            std::cout << "  Random index : " << start_r_idx << " / " << num_random_slices << "\n";
            std::cout << "  Sub-random   : " << start_s_val << " / " << num_sub_slices << "\n";
            std::cout << "  Prev hashes  : " << start_total_hashes << "\n";
            std::cout << "  GF Poly Seed : 0x" << std::hex << rng_seed << std::dec << "\n\n";
        }
        else
        {
            std::cout << "Warning: State file invalid (out of range), starting fresh.\n";
            start_r_idx = 0;
            start_s_val = 0;
            start_total_hashes = 0;
            has_seed = false;
        }
    }

    if (!has_seed)
    {
        std::random_device rd;
        rng_seed = ((uint64_t)rd() << 32) | rd();
        std::cout << "Generated Galois-Field Poly Seed: 0x" << std::hex << rng_seed << std::dec << "\n";
        std::cout << "(Use --seed 0x" << std::hex << rng_seed << std::dec << " to reproduce this run)\n\n";
    }

    PolynomialExtremeRNG poly_rng(rng_seed);

    if (!resuming)
    {
        unsigned long long zero64 = 0;
        ck(cudaMemcpyAsync(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice, streamKernel), "reset hashes_accum");
    }

    // --- Main Scan Loop ---
    std::cout << "======== Phase-1: Polynomial Extreme Vanity Search ==========\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = start_total_hashes;

    bool found = false;
    bool interrupted = false;
    uint64_t state_save_counter = 0;
    const uint64_t STATE_SAVE_INTERVAL = 50;

    for (uint64_t r_idx = start_r_idx; r_idx < num_random_slices && !g_sigint; ++r_idx)
    {
        uint64_t R = r_idx;
        if (!no_shuffle && random_bits > 0)
        {
            R = poly_rng.permute(r_idx, random_bits);
        }

        uint32_t s_start = (r_idx == start_r_idx) ? start_s_val : 0;

        for (uint32_t S = s_start; S < num_sub_slices && !g_sigint; ++S)
        {

            std::string slice_start_hex, slice_end_hex;
            uint64_t linear_max = (linear_bits >= 64) ? 0xFFFFFFFFFFFFFFFFULL : ((1ULL << linear_bits) - 1ULL);

            slice_start_hex = buildSliceHex(common_hex, prefix_nibble, R, random_hex_chars, S, sub_hex_chars, 0ULL, linear_hex_chars);
            slice_end_hex = buildSliceHex(common_hex, prefix_nibble, R, random_hex_chars, S, sub_hex_chars, linear_max, linear_hex_chars);

            uint64_t slice_start[4], slice_end[4];
            if (!hexToLE64(slice_start_hex, slice_start) || !hexToLE64(slice_end_hex, slice_end))
            {
                std::cerr << "Error: failed to convert slice hex to LE64\n";
                return EXIT_FAILURE;
            }

            {
                uint64_t cur[4] = {slice_start[0], slice_start[1], slice_start[2], slice_start[3]};
                for (uint64_t i = 0; i < threadsTotal; ++i)
                {
                    uint64_t Sc[4];
                    add256_u64(cur, (uint64_t)half, Sc);
                    h_start_scalars[i * 4 + 0] = Sc[0];
                    h_start_scalars[i * 4 + 1] = Sc[1];
                    h_start_scalars[i * 4 + 2] = Sc[2];
                    h_start_scalars[i * 4 + 3] = Sc[3];

                    uint64_t next[4];
                    add256(cur, per_thread_cnt, next);
                    cur[0] = next[0];
                    cur[1] = next[1];
                    cur[2] = next[2];
                    cur[3] = next[3];
                }
            }

            // Async Uploads on streamKernel
            ck(cudaMemcpyAsync(d_start_scalars, h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice, streamKernel), "cpy start_scalars");
            ck(cudaMemcpyAsync(d_counts256, h_counts256, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice, streamKernel), "reset counts");

            int zero_flag = FOUND_NONE;
            ck(cudaMemcpyAsync(d_found_flag, &zero_flag, sizeof(int), cudaMemcpyHostToDevice, streamKernel), "reset found_flag");

            // OPTIMIZATION: Launch scalarMulBase on the same stream to remove device-wide sync!
            {
                int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
                scalarMulKernelBase<<<blocks_scal, threadsPerBlock, 0, streamKernel>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
            }

            bool completed_this_slice = false;

            while (!completed_this_slice && !g_sigint)
            {
                unsigned int zeroU = 0u;
                ck(cudaMemcpyAsync(d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, streamKernel), "zero d_any_left");

                // OPTIMIZATION: Call template dispatch
                if (!launch_kernel_v2(
                        B, blocks, threadsPerBlock, streamKernel,
                        d_Px, d_Py, d_Rx, d_Ry,
                        d_start_scalars, d_counts256,
                        threadsTotal,
                        slices_per_launch,
                        d_found_flag, d_found_result,
                        d_hashes_accum,
                        d_any_left))
                {
                    std::cerr << "\nKernel launch error: Unsupported batch size\n";
                    interrupted = true;
                    break;
                }

                cudaError_t launchErr = cudaGetLastError();
                if (launchErr != cudaSuccess)
                {
                    std::cerr << "\nKernel launch error: " << cudaGetErrorString(launchErr) << "\n";
                    interrupted = true;
                    break;
                }

                while (!completed_this_slice && !g_sigint)
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    double dt = std::chrono::duration<double>(now - tLast).count();

                    if (dt >= 1.0)
                    {
                        unsigned long long h_hashes = 0ull;
                        ck(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read hashes");
                        double delta = (double)(h_hashes - lastHashes);
                        double mkeys = delta / (dt * 1e6);
                        double elapsed = std::chrono::duration<double>(now - t0).count();

                        uint64_t slices_done = r_idx * num_sub_slices + S;
                        double slice_prog = (double)slices_done / (double)total_slices * 100.0;

                        std::string cur_display;
                        if (random_bits > 0 || sub_random_bits > 0)
                        {
                            char dbuf[64];
                            snprintf(dbuf, sizeof(dbuf), "%0*llx|%0*x", random_hex_chars, (unsigned long long)R, sub_hex_chars, (unsigned)S);
                            cur_display = dbuf;
                        }
                        else
                        {
                            cur_display = slice_start_hex + ":" + slice_end_hex;
                        }

                        std::cout << "\r[" << cur_display << "] "
                                  << "Time: " << std::fixed << std::setprecision(1) << elapsed << "s"
                                  << " | " << std::setprecision(1) << mkeys << " Mkey/s"
                                  << " | Total: " << h_hashes
                                  << " | Progress: " << std::setprecision(4) << slice_prog << "%"
                                  << std::flush;

                        lastHashes = h_hashes;
                        tLast = now;
                    }

                    int host_found = 0;
                    ck(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "read found_flag");
                    if (host_found == FOUND_READY)
                    {
                        found = true;
                        completed_this_slice = true;
                        break;
                    }

                    cudaError_t qs = cudaStreamQuery(streamKernel);
                    if (qs == cudaSuccess)
                    {
                        break;
                    }
                    else if (qs != cudaErrorNotReady)
                    {
                        cudaGetLastError();
                        interrupted = true;
                        completed_this_slice = true;
                        break;
                    }

                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }

                if (g_sigint)
                    break;
                if (interrupted)
                    break;

                cudaStreamSynchronize(streamKernel);
                std::cout.flush();

                if (found || interrupted)
                    break;

                unsigned int h_any = 0u;
                ck(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read any_left");

                std::swap(d_Px, d_Rx);
                std::swap(d_Py, d_Ry);

                if (h_any == 0u)
                {
                    completed_this_slice = true;
                }
            }

            state_save_counter++;
            if (state_save_counter >= STATE_SAVE_INTERVAL)
            {
                unsigned long long cur_hashes = 0;
                cudaMemcpy(&cur_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
                saveState(state_file, r_idx, S + 1, cur_hashes, rng_seed);
                state_save_counter = 0;
            }

            if (found || interrupted || g_sigint)
                break;
        }

        if (found || interrupted || g_sigint)
            break;
    }

    {
        unsigned long long final_hashes = 0;
        cudaMemcpy(&final_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (!found)
        {
            saveState(state_file, start_r_idx, 0, final_hashes, rng_seed);
        }
        else
        {
            std::remove(state_file.c_str());
        }
    }

    cudaDeviceSynchronize();
    std::cout << "\n\n";

    int h_found_flag = 0;
    ck(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "final read found_flag");

    int exit_code = EXIT_SUCCESS;

    if (h_found_flag == FOUND_READY)
    {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "read found_result");

        std::cout << "\n╔══════════════════════════════════════════════════════════╗\n";
        std::cout << "║                   FOUND MATCH!                          ║\n";
        std::cout << "╠══════════════════════════════════════════════════════════╣\n";
        std::cout << "║ Private Key   : " << std::left << std::setw(40) << formatHex256(host_result.scalar) << "║\n";
        std::cout << "║ Public Key    : " << std::left << std::setw(40) << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "║\n";
        std::cout << "╚══════════════════════════════════════════════════════════╝\n";

        exit_code = EXIT_SUCCESS;
    }
    else
    {
        if (g_sigint)
        {
            std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
            std::cout << "Search was interrupted by user.\n";
            std::cout << "Run again to resume from saved state: " << state_file << "\n";
            exit_code = 130;
        }
        else if (interrupted)
        {
            std::cout << "======== TERMINATED (Error) ============================\n";
            exit_code = EXIT_FAILURE;
        }
        else
        {
            std::cout << "======== KEY NOT FOUND (Exhaustive) ===================\n";
            std::cout << "Vanity prefix not found within the specified range.\n";
            std::remove(state_file.c_str());
            exit_code = EXIT_SUCCESS;
        }
    }

    // --- Cleanup ---
    cudaFree(d_start_scalars);
    cudaFree(d_Px);
    cudaFree(d_Py);
    cudaFree(d_Rx);
    cudaFree(d_Ry);
    cudaFree(d_counts256);
    cudaFree(d_found_flag);
    cudaFree(d_found_result);
    cudaFree(d_hashes_accum);
    cudaFree(d_any_left);
    cudaStreamDestroy(streamKernel);

    if (h_start_scalars)
        cudaFreeHost(h_start_scalars);
    if (h_counts256)
        cudaFreeHost(h_counts256);

    return exit_code;
}