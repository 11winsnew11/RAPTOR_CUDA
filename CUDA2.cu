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
#include <random> // Untuk random start

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}
__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];
__constant__ int c_vanity_len;

// --- Konstanta Baru untuk Grasshopper Algorithm ---
__constant__ uint64_t c_RangeStart[4];
__constant__ uint64_t c_RangeLen[4];
// Titik P_RangeLen = RangeLen * G (untuk wrap-around)
__constant__ uint64_t c_P_RangeLen_X[4];
__constant__ uint64_t c_P_RangeLen_Y[4];

// Helper Device Function untuk Point Addition Affine (Generik)
// Digunakan untuk wrap-around correction: P = P - P_RangeLen
__device__ void pointAddAffineGeneric(
    const uint64_t x1[4], const uint64_t y1[4],
    const uint64_t x2[4], const uint64_t y2[4],
    uint64_t x3[4], uint64_t y3[4])
{
    // Menghitung lambda = (y2 - y1) * inv(x2 - x1)
    uint64_t dx[4], dy[4], inv_dx[4], lam[4];
    
    ModSub256(dx, x2, x1);
    ModSub256(dy, y2, y1);
    
    // Inverse dx
    // Karena ini jarang dieksekusi (hanya saat wrap), kita pakai _ModInv standalone
    uint64_t inv_tmp[5];
    inv_tmp[0] = dx[0]; inv_tmp[1] = dx[1]; inv_tmp[2] = dx[2]; inv_tmp[3] = dx[3]; inv_tmp[4] = 0;
    _ModInv(inv_tmp);
    inv_dx[0] = inv_tmp[0]; inv_dx[1] = inv_tmp[1]; inv_dx[2] = inv_tmp[2]; inv_dx[3] = inv_tmp[3];
    
    _ModMult(lam, dy, inv_dx);
    
    // x3 = lam^2 - x1 - x2
    _ModSqr(x3, lam);
    ModSub256(x3, x3, x1);
    ModSub256(x3, x3, x2);
    
    // y3 = lam(x1 - x3) - y1
    uint64_t t[4];
    ModSub256(t, x1, x3);
    _ModMult(y3, t, lam);
    ModSub256(y3, y3, y1);
}

__launch_bounds__(256, 2)
__global__ void kernel_point_add_and_check_oneinv(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;
    const int vanity_len = c_vanity_len;

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], S[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i]  = start_scalars[idx];   
    }
    uint64_t rem[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    auto check_vanity = [&](const uint8_t* h20) -> bool {
        if (vanity_len >= 4) {
            if (load_u32_le(h20) != target_prefix) return false;
            #pragma unroll
            for (int k = 4; k < 20; ++k) {
                if (k >= vanity_len) break;
                if (h20[k] != c_target_hash160[k]) return false;
            }
            return true;
        } else {
            for (int k = 0; k < vanity_len; ++k) {
                if (h20[k] != c_target_hash160[k]) return false;
            }
            return true;
        }
    };

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // ... (Bagian Hash Check tetap sama persis) ...
        {
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            bool match = check_vanity(h20);
            if (__any_sync(full_mask, match)) {
                if (match) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=S[k];
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=x1[k];
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y1[k];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }
        }

        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

        #pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Jx[j];
        ModSub256(acc, acc, x1);
        #pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
            #pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
            #pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
        #pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
        #pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        uint64_t sy_neg[4], sx_neg[4];
        ModNeg256(sy_neg, y1);
        ModNeg256(sx_neg, x1);

        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            // --- BLOK +G ---
            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
                #pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);     
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3); 
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool match = check_vanity(h20);
                if (__any_sync(full_mask, match)) {
                    if (match) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t addv=(uint64_t)(i+1);
                            for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                           
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            // --- BLOK -G ---
            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
                #pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                ModNeg256(py_i, py_i); 

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();

                bool match = check_vanity(h20);
                if (__any_sync(full_mask, match)) {
                    if (match) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t sub=(uint64_t)(i+1);
                            for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
                            #pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            uint64_t gxmi[4];
            #pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        // --- BLOK AKHIR (half - 1) -G ---
        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t px3[4], s[4], lam[4];
            uint64_t px_i[4], py_i[4];
            #pragma unroll
            for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
            ModNeg256(py_i, py_i);

            ModSub256(s, py_i, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256(px3, px3, x1);
            ModSub256(px3, px3, px_i);

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, y1, &odd);

            uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            bool match = check_vanity(h20);
            if (__any_sync(full_mask, match)) {
                if (match) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                        uint64_t sub=(uint64_t)half;
                        for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                        uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }

            uint64_t last_dx[4];
            #pragma unroll
            for (int j=0;j<4;++j) last_dx[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(last_dx, last_dx, x1);
            _ModMult(inverse, inverse, last_dx);
        }

        // ==========================================
        // GRASSHOPPER UPDATE LOGIC
        // ==========================================
        
        // 1. Hitung P_temp = P_current + P_B (J)
        uint64_t x_temp[4], y_temp[4];
        {
            uint64_t lam[4], s[4];
            uint64_t Jy_minus_y1[4];
            #pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_Jy[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x_temp, lam);
            ModSub256(x_temp, x_temp, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=c_Jx[j];
            ModSub256(x_temp, x_temp, Jx_local);

            ModSub256(s, x1, x_temp);
            _ModMult(y_temp, s, lam);
            ModSub256(y_temp, y_temp, y1);
        }

        // 2. Hitung S_temp = S + B
        uint64_t s_temp[4];
        uint64_t carry = 0;
        // Add B to S
        {
             __uint128_t res = (__uint128_t)S[0] + (uint64_t)B;
             s_temp[0] = (uint64_t)res;
             carry = (uint64_t)(res >> 64);
             res = (__uint128_t)S[1] + carry;
             s_temp[1] = (uint64_t)res;
             carry = (uint64_t)(res >> 64);
             res = (__uint128_t)S[2] + carry;
             s_temp[2] = (uint64_t)res;
             carry = (uint64_t)(res >> 64);
             s_temp[3] = S[3] + carry;
        }

        // 3. Cek Wrap-Around (Apakah S_temp >= RangeEnd?)
        // Karena range power of 2 dan start aligned, kita cek apakah bit high menyala
        // atau lebih sederhana: bandingkan dengan c_RangeLen.
        // Logika: Jika S_temp >= c_RangeStart + c_RangeLen
        
        bool wrap = false;
        // Cek simple: jika carry out terjadi (S wrap ke 0), atau jika S_temp > RangeEnd.
        // Karena start aligned, kita bisa cek: jika (s_temp & range_len) != 0 ??
        // Tidak, range_len adalah length. RangeEnd = Start + Len.
        // Kita cek apakah s_temp >= c_RangeStart + c_RangeLen.
        // Atau gunakan logika borrow.
        
        // Cara paling aman: cek apakah s_temp >= c_RangeLen (jika start=0).
        // Karena kita pakai absolute coordinates, kita hitung End dulu.
        // Untuk efisiensi, kita asumsikan RangeLen power of 2.
        // Jika S bergeser melewati power of 2 boundary, wrap.
        
        // Logika Wrap:
        // Jika (s_temp - c_RangeStart) >= c_RangeLen
        uint64_t diff[4];
        // Sub256(diff, s_temp, c_RangeStart) -> borrow check
        // Sub manual:
        // ...
        // Lebih cepat: cek flag carry dari penjumlahan sebelumnya? Tidak, B kecil.
        // Kita pakai logika: jika S_new melewati End.
        // Karena RangeLen power of 2, bit ke-n akan menyala jika >= Len (relative).
        // Kalkulasi relatif: Rel = S_temp - c_RangeStart.
        
        // Kita lakukan pengecekan 256-bit biasa (sederhana)
        uint64_t end_local[4];
        // End = Start + Len
        // Kita tidak simpan End, kita simpan Start dan Len.
        // Wrap jika S_temp - Start >= Len.
        // Kita hitung diff = S_temp - Start
        uint64_t b_sub = 0;
        for(int k=0; k<4; ++k) {
            uint64_t val = s_temp[k];
            uint64_t sub_val = c_RangeStart[k];
            uint64_t res = val - sub_val - b_sub;
            diff[k] = res;
            b_sub = (val < sub_val + b_sub) ? 1 : 0;
        }
        
        // Jika diff >= c_RangeLen -> Wrap
        // Bandingkan diff vs c_RangeLen
        // Karena range_len power of 2, kita bisa cek bit.
        // Atau loop perbandingan.
        // Asumsi: Kasus wrap jarang terjadi (hanya jika B mendekati ujung).
        // Perbandingan simple:
        if (diff[3] > c_RangeLen[3] || 
           (diff[3] == c_RangeLen[3] && diff[2] > c_RangeLen[2]) ||
           (diff[3] == c_RangeLen[3] && diff[2] == c_RangeLen[2] && diff[1] > c_RangeLen[1]) ||
           (diff[3] == c_RangeLen[3] && diff[2] == c_RangeLen[2] && diff[1] == c_RangeLen[1] && diff[0] >= c_RangeLen[0])) {
            wrap = true;
        }
        
        if (wrap) {
            // S_new = S_temp - RangeLen
            // P_new = P_temp - P_RangeLen
            uint64_t s_final[4];
            // Sub Manual (S_temp - c_RangeLen)
            b_sub = 0;
            for(int k=0; k<4; ++k) {
                uint64_t val = s_temp[k];
                uint64_t sub_val = c_RangeLen[k];
                s_final[k] = val - sub_val - b_sub;
                b_sub = (val < sub_val + b_sub) ? 1 : 0;
            }
            
            // Point Subtraction: P_temp - c_P_RangeLen = P_temp + (-c_P_RangeLen)
            // -P_RangeLen adalah (X, -Y)
            uint64_t neg_Y_range[4];
            ModNeg256(neg_Y_range, c_P_RangeLen_Y);
            
            pointAddAffineGeneric(x_temp, y_temp, c_P_RangeLen_X, neg_Y_range, x1, y1);
            
            #pragma unroll
            for(int k=0; k<4; ++k) S[k] = s_final[k];
        } else {
            // Tidak wrap
            #pragma unroll
            for(int k=0; k<4; ++k) { x1[k] = x_temp[k]; y1[k] = y_temp[k]; S[k] = s_temp[k]; }
        }

        // 4. Update Remaining (counts)
        // Dalam mode Grasshopper, kita selalu mengurangi 'rem' untuk tracking seberapa banyak kita bekerja
        sub256_u64_inplace(rem, (uint64_t)B);
        
        ++batches_done;
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
    #undef check_vanity
}

// Deklarasi eksternal
extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string range_hex, vanity_hash_hex;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         = 64;

    // Parser args (sama seperti sebelumnya)
    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        char* endp=nullptr;
        unsigned long aa = std::strtoul(a_str.c_str(), &endp, 10); if (*endp) return false;
        endp=nullptr;
        unsigned long bb = std::strtoul(b_str.c_str(), &endp, 10); if (*endp) return false;
        if (aa == 0ul || bb == 0ul) return false;
        if (aa > (1ul<<20) || bb > (1ul<<20)) return false;
        a_out=(uint32_t)aa; b_out=(uint32_t)bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--vanity-hash160" && i + 1 < argc) vanity_hash_hex = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\" (positive integers).\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm    = b;
        }
        else if (arg == "--slices" && i + 1 < argc) {
            char* endp=nullptr;
            unsigned long v = std::strtoul(argv[++i], &endp, 10);
            if (*endp != '\0' || v == 0ul || v > (1ul<<20)) {
                std::cerr << "Error: --slices must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = (uint32_t)v;
        }
    }

    if (range_hex.empty() || vanity_hash_hex.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> --vanity-hash160 <prefix_hex> [--grid A,B] [--slices N]\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }

    // Parse Vanity
    if (vanity_hash_hex.length() > 40 || vanity_hash_hex.length() % 2 != 0) {
        std::cerr << "Error: Vanity hash160 hex length must be even and <= 40 characters.\n";
        return EXIT_FAILURE;
    }
    uint8_t target_hash160[20];
    memset(target_hash160, 0, 20);
    for (size_t i = 0; i < vanity_hash_hex.length() / 2; ++i) {
        std::string byteStr = vanity_hash_hex.substr(i * 2, 2);
        target_hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    int vanity_len = (int)(vanity_hash_hex.length() / 2);

    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << " (kernel limit).\n";
        return EXIT_FAILURE;
    }

    // Hitung Range Length
    uint64_t range_len[4]; 
    sub256(range_end, range_start, range_len); 
    add256_u64(range_len, 1ull, range_len);

    // Validasi Range Power of 2 (Diperlukan untuk Grasshopper logic sederhana)
    auto is_power_of_two_256 = [&](const uint64_t a[4])->bool {
        if ((a[0]|a[1]|a[2]|a[3]) == 0ull) return false;
        uint64_t am1[4]; uint64_t borrow = 1ull;
        for (int i=0;i<4;++i) {
            uint64_t v = a[i] - borrow; borrow = (a[i] < borrow) ? 1ull : 0ull; am1[i] = v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) am1[k] = a[k]; break; }
        }
        uint64_t and0=a[0]&am1[0], and1=a[1]&am1[1], and2=a[2]&am1[2], and3=a[3]&am1[3];
        return (and0|and1|and2|and3)==0ull;
    };
    if (!is_power_of_two_256(range_len)) {
        std::cerr << "Error: For Grasshopper Random Leap, range length (end - start + 1) must be a power of two.\n"; return EXIT_FAILURE;
    }

    // Setup GPU
    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; return EXIT_FAILURE;
    }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int threadsPerBlock=256;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock=prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock=32;

    const uint64_t bytesPerThread = 2ull*4ull*sizeof(uint64_t);
    size_t totalGlobalMem = prop.totalGlobalMem;
    const uint64_t reserveBytes = 64ull * 1024 * 1024;
    uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
    uint64_t maxThreadsByMem = usableMem / bytesPerThread;

    // Kalkulasi Threads Total
    uint64_t q_div_batch[4], r_div_batch = 0ull;
    divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
    if (r_div_batch != 0ull) {
        std::cerr << "Error: range length must be divisible by batch size (" << runtime_points_batch_size << ").\n";
        return EXIT_FAILURE;
    }
    bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
    uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
    if (!q_fits_u64) { std::cerr << "Error: total batches too large for u64.\n"; return EXIT_FAILURE; }

    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;

    auto pick_threads_total = [&](uint64_t upper)->uint64_t {
        if (upper < (uint64_t)threadsPerBlock) return 0ull;
        uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
        uint64_t q = total_batches_u64;
        while (t >= (uint64_t)threadsPerBlock) {
            if ((q % t) == 0ull) return t;
            t -= (uint64_t)threadsPerBlock;
        }
        return 0ull;
    };

    uint64_t upper = maxThreadsByMem;
    if (total_batches_u64 < upper) upper = total_batches_u64;
    if (userUpper         < upper) upper = userUpper;

    uint64_t threadsTotal = pick_threads_total(upper);
    if (threadsTotal == 0ull) {
        std::cerr << "Error: failed to pick threadsTotal satisfying divisibility.\n";
        return EXIT_FAILURE;
    }
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    // Per-thread count untuk Grasshopper:
    // Karena pencarian acak, kita set counts = RangeLen agar thread berjalan lama
    // Atau set sesuai kebutuhan (RangeLen / ThreadsTotal) jika ingin fair.
    // Untuk Grasshopper, kita set sama dengan RangeLen agar thread bisa "melompat" jauh.
    uint64_t per_thread_cnt[4]; 
    for(int k=0;k<4;++k) per_thread_cnt[k] = range_len[k]; // Full range scanning potential

    // Alokasi Memori
    uint64_t* h_counts256     = nullptr;
    uint64_t* h_start_scalars = nullptr;
    cudaHostAlloc(&h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc(&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    // Inisialisasi Counts
    for (uint64_t i = 0; i < threadsTotal; ++i) {
        h_counts256[i*4+0] = per_thread_cnt[0];
        h_counts256[i*4+1] = per_thread_cnt[1];
        h_counts256[i*4+2] = per_thread_cnt[2];
        h_counts256[i*4+3] = per_thread_cnt[3];
    }

    // ==========================================
    // GRASSHOPPER INITIALIZATION
    // ==========================================
    const uint32_t B = runtime_points_batch_size;
    const uint32_t half = B >> 1;
    
    std::cout << "Info: Initializing Grasshopper Random Start positions...\n";
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());

    // Hitung mask untuk modulo bit (karena range power of 2)
    uint64_t len_minus1[4];
    {   uint64_t borrow=1ull;
        for (int i=0;i<4;++i) {
            uint64_t v=range_len[i]-borrow; borrow=(range_len[i]<borrow)?1ull:0ull; len_minus1[i]=v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) len_minus1[k]=range_len[k]; break; }
        }
    }

    for (uint64_t i = 0; i < threadsTotal; ++i) {
        // Generate Random Offset
        uint64_t rand_offset[4];
        rand_offset[0] = dist(gen) & len_minus1[0];
        rand_offset[1] = dist(gen) & len_minus1[1];
        rand_offset[2] = dist(gen) & len_minus1[2];
        rand_offset[3] = dist(gen) & len_minus1[3];
        
        // Start Scalar = RangeStart + RandomOffset
        add256(range_start, rand_offset, &h_start_scalars[i*4]);
    }

    // Copy Constants to Device
    cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(int));
    cudaMemcpyToSymbol(c_RangeStart, range_start, 4*sizeof(uint64_t));
    cudaMemcpyToSymbol(c_RangeLen, range_len, 4*sizeof(uint64_t));

    uint32_t prefix_le = 0;
    if (vanity_len >= 4) {
         prefix_le = (uint32_t)target_hash160[0]
                   | ((uint32_t)target_hash160[1] << 8)
                   | ((uint32_t)target_hash160[2] << 16)
                   | ((uint32_t)target_hash160[3] << 24);
    }
    cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));

    // Device Ptrs
    uint64_t *d_start_scalars=nullptr, *d_Px=nullptr, *d_Py=nullptr, *d_Rx=nullptr, *d_Ry=nullptr, *d_counts256=nullptr;
    int *d_found_flag=nullptr; FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr; unsigned int *d_any_left=nullptr;

    auto ck = [](cudaError_t e, const char* msg){
        if (e != cudaSuccess) {
            std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    ck(cudaMalloc(&d_start_scalars, threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_start_scalars)");
    ck(cudaMalloc(&d_Px,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Px)");
    ck(cudaMalloc(&d_Py,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Py)");
    ck(cudaMalloc(&d_Rx,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Rx)");
    ck(cudaMalloc(&d_Ry,           threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Ry)");
    ck(cudaMalloc(&d_counts256,    threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_counts256)");
    ck(cudaMalloc(&d_found_flag,   sizeof(int)),                         "cudaMalloc(d_found_flag)");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)),                 "cudaMalloc(d_found_result)");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)),          "cudaMalloc(d_hashes_accum)");
    ck(cudaMalloc(&d_any_left,     sizeof(unsigned int)),                "cudaMalloc(d_any_left)");

    ck(cudaMemcpy(d_start_scalars, h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start_scalars");
    ck(cudaMemcpy(d_counts256,     h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts256");
    { int zero = FOUND_NONE; unsigned long long zero64=0ull;
      ck(cudaMemcpy(d_found_flag, &zero,   sizeof(int),                cudaMemcpyHostToDevice), "init found_flag");
      ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes_accum"); }

    // Precompute P_Start (Initial Points dari Random Scalars)
    {
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");
        ck(cudaGetLastError(), "scalarMulKernelBase launch");
    }

    // Precompute G Table (1G ... half*G)
    {
        uint64_t* h_scalars_half = nullptr;
        cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4 + 0] = (uint64_t)(k + 1);

        uint64_t *d_scalars_half=nullptr, *d_Gx_half=nullptr, *d_Gy_half=nullptr;
        ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_scalars_half)");
        ck(cudaMalloc(&d_Gx_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gx_half)");
        ck(cudaMalloc(&d_Gy_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gy_half)");
        ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy half scalars");

        int blocks_scal = (int)((half + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(half) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(half) launch");

        uint64_t* h_Gx_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        uint64_t* h_Gy_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx_half");
        ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy_half");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

        cudaFree(d_scalars_half); cudaFree(d_Gx_half); cudaFree(d_Gy_half);
        cudaFreeHost(h_scalars_half);
        std::free(h_Gx_half); std::free(h_Gy_half);
    }
    
    // Precompute Jump Point J = B * G
    {
        uint64_t* h_scalarB = nullptr;
        cudaHostAlloc(&h_scalarB, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalarB, 0, 4 * sizeof(uint64_t));
        h_scalarB[0] = (uint64_t)B;

        uint64_t *d_scalarB=nullptr, *d_Jx=nullptr, *d_Jy=nullptr;
        ck(cudaMalloc(&d_scalarB, 4 * sizeof(uint64_t)), "cudaMalloc(d_scalarB)");
        ck(cudaMalloc(&d_Jx,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jx)");
        ck(cudaMalloc(&d_Jy,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jy)");
        ck(cudaMemcpy(d_scalarB, h_scalarB, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scalarB");

        scalarMulKernelBase<<<1, 1>>>(d_scalarB, d_Jx, d_Jy, 1);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(B) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(B) launch");

        uint64_t hJx[4], hJy[4];
        ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
        ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");
        ck(cudaMemcpyToSymbol(c_Jx, hJx, 4 * sizeof(uint64_t)), "ToSymbol c_Jx");
        ck(cudaMemcpyToSymbol(c_Jy, hJy, 4 * sizeof(uint64_t)), "ToSymbol c_Jy");

        cudaFree(d_scalarB); cudaFree(d_Jx); cudaFree(d_Jy);
        cudaFreeHost(h_scalarB);
    }

    // Precompute RangeLen * G (untuk Wrap-Around)
    {
        uint64_t* h_scalarRL = nullptr;
        cudaHostAlloc(&h_scalarRL, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        // Copy range_len (sudah uint64[4])
        memcpy(h_scalarRL, range_len, 4 * sizeof(uint64_t));

        uint64_t *d_scalarRL=nullptr, *d_RLx=nullptr, *d_RLy=nullptr;
        ck(cudaMalloc(&d_scalarRL, 4 * sizeof(uint64_t)), "cudaMalloc(d_scalarRL)");
        ck(cudaMalloc(&d_RLx,      4 * sizeof(uint64_t)), "cudaMalloc(d_RLx)");
        ck(cudaMalloc(&d_RLy,      4 * sizeof(uint64_t)), "cudaMalloc(d_RLy)");
        ck(cudaMemcpy(d_scalarRL, h_scalarRL, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scalarRL");

        scalarMulKernelBase<<<1, 1>>>(d_scalarRL, d_RLx, d_RLy, 1);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase(RangeLen) sync");
        ck(cudaGetLastError(), "scalarMulKernelBase(RangeLen) launch");

        uint64_t hRLx[4], hRLy[4];
        ck(cudaMemcpy(hRLx, d_RLx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H RLx");
        ck(cudaMemcpy(hRLy, d_RLy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H RLy");
        ck(cudaMemcpyToSymbol(c_P_RangeLen_X, hRLx, 4 * sizeof(uint64_t)), "ToSymbol c_P_RangeLen_X");
        ck(cudaMemcpyToSymbol(c_P_RangeLen_Y, hRLy, 4 * sizeof(uint64_t)), "ToSymbol c_P_RangeLen_Y");

        cudaFree(d_scalarRL); cudaFree(d_RLx); cudaFree(d_RLy);
        cudaFreeHost(h_scalarRL);
    }

    // Info
    size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB,&totalB);
    size_t usedB = totalB - freeB;
    double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;

    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Mode"              << " : GRASSHOPPER (Random Start + Wrap)\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "ThreadsTotal"      << " : " << (uint64_t)threadsTotal << "\n";
    std::cout << std::left << std::setw(20) << "Batch Size (B)"    << " : " << B << "\n";
    std::cout << std::left << std::setw(20) << "Vanity Target"     << " : " << vanity_hash_hex << " (" << vanity_len << " bytes)\n\n";
    std::cout << "======== Phase-1: Random Search =======================\n";

    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "create stream");

    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv, cudaFuncCachePreferL1);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    bool stop_all = false;
    bool completed_all = false;
    while (!stop_all) {
        if (g_sigint) std::cerr << "\n[Ctrl+C] Interrupt received. Finishing current kernel slice and exiting...\n";

        unsigned int zeroU = 0u;
        ck(cudaMemcpyAsync(d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, streamKernel), "zero d_any_left");

        kernel_point_add_and_check_oneinv<<<blocks, threadsPerBlock, 0, streamKernel>>>(
            d_Px, d_Py, d_Rx, d_Ry,
            d_start_scalars, d_counts256,
            threadsTotal,
            B,
            slices_per_launch,
            d_found_flag, d_found_result,
            d_hashes_accum,
            d_any_left
        );
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "\nKernel launch error: " << cudaGetErrorString(launchErr) << "\n";
            stop_all = true;
        }

        while (!stop_all) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 1.0) {
                unsigned long long h_hashes = 0ull;
                ck(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read hashes");
                double delta = (double)(h_hashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                
                // Karena Random, progress bukan linear. Kita tampilkan "Keys Checked" saja.
                // Bisa hitung "Expected Coverage" probabilitas jika diinginkan.
                long double total_keys_ld = ld_from_u256(range_len);
                long double coverage = 0.0L;
                if (total_keys_ld > 0.0L) {
                     // Simple approximation: (keys / total) * 100
                     coverage = ((long double)h_hashes / total_keys_ld) * 100.0L;
                }
                
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << "s | Speed: " << std::fixed << std::setprecision(2) << mkeys
                          << " Mkeys/s | Total: " << h_hashes
                          << " | Est. Coverage: " << std::fixed << std::setprecision(4) << (double)coverage << " %";
                std::cout.flush();
                lastHashes = h_hashes; tLast = now;
            }

            int host_found = 0;
            ck(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "read found_flag");
            if (host_found == FOUND_READY) { stop_all = true; break; }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess) break;
            else if (qs != cudaErrorNotReady) { cudaGetLastError(); stop_all = true; break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        cudaStreamSynchronize(streamKernel);
        std::cout.flush();
        if (stop_all || g_sigint) break;

        // Swap Buffers
        std::swap(d_Px, d_Rx);
        std::swap(d_Py, d_Ry);

        // Di mode Grasshopper, kita tidak pernah selesai (wrap around), kecuali found.
        // Kecuali jika ada mekanisme stop manual.
        // Kita set h_any = 1 agar loop terus berjalan selama tidak ketemu.
        // Kecuali jika user set specific count limit di counts256.
        unsigned int h_any = 0u;
        ck(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read any_left");
        if (h_any == 0u) { completed_all = true; break; } // Jika counts habis (misal user set limit)
    }

    cudaDeviceSynchronize();
    std::cout << "\n";

    int h_found_flag = 0;
    ck(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "final read found_flag");

    int exit_code = EXIT_SUCCESS;

    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "read found_result");
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
    } else {
        if (g_sigint) {
            std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
            exit_code = 130;
        } else {
            std::cout << "======== SEARCH STOPPED ================================\n";
        }
    }

    // Cleanup
    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py); cudaFree(d_Rx); cudaFree(d_Ry);
    cudaFree(d_counts256); cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_hashes_accum); cudaFree(d_any_left);
    cudaStreamDestroy(streamKernel);

    if (h_start_scalars) cudaFreeHost(h_start_scalars);
    if (h_counts256)     cudaFreeHost(h_counts256);

    return exit_code;
}