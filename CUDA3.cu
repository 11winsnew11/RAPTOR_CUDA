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

// MAX_BATCH_SIZE 1024 (2^10) agar muat di Constant Memory (64KB limit)
// 1024 sudah cukup besar untuk effisiensi batch inversion
#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4]; // Jump Point (B * G)
__constant__ uint64_t c_Jy[4];
__constant__ int c_vanity_len;

// ============================================================
// KERNEL: SYSTEMATIC SWEEP (LINEAR SCAN)
// ============================================================
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

    // Jika count sudah nol, thread ini selesai
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

    // Loop pemrosesan batch
    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // --- 1. HASH CHECK TITIK P (Pusat Batch) ---
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

        // --- 2. BATCH INVERSION SETUP ---
        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

        // Inisialisasi acc dengan (x_J - x_P)
        #pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Jx[j];
        ModSub256(acc, acc, x1);
        #pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        // Akumulasi selisih untuk titik-titik G
        for (int i = half - 2; i >= 0; --i) {
            #pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
            #pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        // Inversi Tunggal
        uint64_t d0[4], inverse[5];
        #pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
        #pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        // --- 3. LOOP PENGECEKAN TETANGGA (+G dan -G) ---
        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            // BLOK +G (P + (i+1)G)
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

            // BLOK -G (P - (i+1)G)
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
                            uint64_t subv=(uint64_t)(i+1);
                            for (int k=0;k<4 && subv;++k){ uint64_t old=fs[k]; fs[k]=old-subv; subv=(old<subv)?1ull:0ull; }
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

            // Update inverse untuk iterasi berikutnya
            uint64_t gxmi[4];
            #pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        // --- 4. BLOK AKHIR (Titik Terjauh: P - half*G) ---
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
                        uint64_t subv=(uint64_t)half;
                        for (int k=0;k<4 && subv;++k){ uint64_t old=fs[k]; fs[k]=old-subv; subv=(old<subv)?1ull:0ull; }
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

        // ==========================================
        // LINEAR SWEEP UPDATE (NO WRAP)
        // ==========================================
        // Saat ini, 'inverse' sudah sama dengan 1 / (x_J - x_P)
        // Kita gunakan ini untuk menghitung P_new = P + J
        
        uint64_t lam_up[4], s_up[4];
        uint64_t x_new[4], y_new[4];
        uint64_t old_x[4], old_y[4];

        // Backup P lama
        #pragma unroll
        for(int j=0;j<4;++j) { old_x[j] = x1[j]; old_y[j] = y1[j]; }

        // Hitung Lambda
        uint64_t dy_up[4];
        #pragma unroll
        for(int j=0;j<4;++j) dy_up[j] = c_Jy[j];
        ModSub256(dy_up, dy_up, old_y); // y_J - y_P
        
        _ModMult(lam_up, dy_up, inverse); // lam = (yJ - yP) / (xJ - xP)

        // Hitung X Baru
        _ModSqr(x_new, lam_up);
        ModSub256(x_new, x_new, old_x);
        uint64_t Jx_loc[4]; 
        #pragma unroll
        for(int j=0;j<4;++j) Jx_loc[j] = c_Jx[j];
        ModSub256(x_new, x_new, Jx_loc);

        // Hitung Y Baru
        ModSub256(s_up, old_x, x_new);
        _ModMult(y_new, s_up, lam_up);
        ModSub256(y_new, y_new, old_y);

        // Update State Register
        #pragma unroll
        for(int j=0;j<4;++j) { x1[j] = x_new[j]; y1[j] = y_new[j]; }

        // Update Scalar S = S + B
        {
            uint64_t carry = 0;
            __uint128_t res = (__uint128_t)S[0] + (uint64_t)B;
            S[0] = (uint64_t)res;
            carry = (uint64_t)(res >> 64);
            res = (__uint128_t)S[1] + carry;
            S[1] = (uint64_t)res;
            carry = (uint64_t)(res >> 64);
            res = (__uint128_t)S[2] + carry;
            S[2] = (uint64_t)res;
            carry = (uint64_t)(res >> 64);
            S[3] = S[3] + carry;
        }

        sub256_u64_inplace(rem, (uint64_t)B);
        ++batches_done;
    }

    // Simpan State Kembali
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

// ============================================================
// HOST CODE
// ============================================================

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);
void mul256_u64(const uint64_t a[4], uint64_t b, uint64_t r[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t res = (__uint128_t)a[i] * b + carry;
        r[i] = (uint64_t)res;
        carry = (uint64_t)(res >> 64);
    }
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string range_hex, vanity_hash_hex;
    uint32_t runtime_points_batch_size = 1024; // Default 1024 (2^10)
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         = 64;

    // Parser args (Sama seperti sebelumnya)
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

    // Validasi Batch Size
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

    // Validasi Range Power of 2 (masih diperlukan agar chunk_size bulat)
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
        std::cerr << "Error: For Linear Sweep, range length (end - start + 1) must be a power of two.\n"; return EXIT_FAILURE;
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

    // Kalkulasi Threads Total
    uint64_t q_div_batch[4], r_div_batch = 0ull;
    divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
    if (r_div_batch != 0ull) {
        std::cerr << "Error: range length must be divisible by batch size.\n";
        return EXIT_FAILURE;
    }
    
    // Total Batches
    bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
    uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
    if (!q_fits_u64) { std::cerr << "Error: total batches too large for u64.\n"; return EXIT_FAILURE; }

    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;

    // Penentuan ThreadsTotal
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

    uint64_t upper = userUpper;
    if (total_batches_u64 < upper) upper = total_batches_u64;

    uint64_t threadsTotal = pick_threads_total(upper);
    if (threadsTotal == 0ull) {
        std::cerr << "Error: failed to pick threadsTotal satisfying divisibility.\n";
        return EXIT_FAILURE;
    }
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    // Hitung chunk per thread
    uint64_t per_thread_cnt[4];
    for(int k=0;k<4;++k) per_thread_cnt[k] = range_len[k]; 
    divmod_256_by_u64(per_thread_cnt, threadsTotal, per_thread_cnt, r_div_batch); // per_thread_cnt = range_len / threadsTotal

    // Alokasi Memori
    uint64_t* h_counts256     = nullptr;
    uint64_t* h_start_scalars = nullptr;
    cudaHostAlloc(&h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc(&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    // ==========================================
    // LINEAR INITIALIZATION (SYSTEMATIC SWEEP)
    // ==========================================
    std::cout << "Info: Initializing Linear Systematic Sweep positions...\n";
    
    for (uint64_t i = 0; i < threadsTotal; ++i) {
        // 1. Hitung Offset = i * per_thread_cnt
        uint64_t offset[4];
        mul256_u64(per_thread_cnt, i, offset);

        // 2. Start Scalar = range_start + offset
        uint64_t start_val[4];
        add256(range_start, offset, start_val);

        // 3. Isi Host Arrays
        h_start_scalars[i*4+0] = start_val[0];
        h_start_scalars[i*4+1] = start_val[1];
        h_start_scalars[i*4+2] = start_val[2];
        h_start_scalars[i*4+3] = start_val[3];

        h_counts256[i*4+0] = per_thread_cnt[0];
        h_counts256[i*4+1] = per_thread_cnt[1];
        h_counts256[i*4+2] = per_thread_cnt[2];
        h_counts256[i*4+3] = per_thread_cnt[3];
    }

    // Copy Constants to Device
    cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(int));

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

    // Precompute P_Start
    {
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");
        ck(cudaGetLastError(), "scalarMulKernelBase launch");
    }

    // Precompute G Table (1G ... half*G)
    {
        const uint32_t B = runtime_points_batch_size;
        const uint32_t half = B >> 1;
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
    
    // Precompute Jump Point J (B * G)
    {
        const uint32_t B = runtime_points_batch_size;
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

    // Info
    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Mode"              << " : LINEAR SWEEP (Systematic Scan)\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "ThreadsTotal"      << " : " << (uint64_t)threadsTotal << "\n";
    std::cout << std::left << std::setw(20) << "Batch Size (B)"    << " : " << runtime_points_batch_size << "\n";
    std::cout << std::left << std::setw(20) << "Vanity Target"     << " : " << vanity_hash_hex << " (" << vanity_len << " bytes)\n\n";
    std::cout << "======== Phase-1: Linear Search =======================\n";

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
            runtime_points_batch_size,
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
                
                long double total_keys_ld = ld_from_u256(range_len);
                long double coverage = 0.0L;
                if (total_keys_ld > 0.0L) {
                     coverage = ((long double)h_hashes / total_keys_ld) * 100.0L;
                }
                
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << "s | Speed: " << std::fixed << std::setprecision(2) << mkeys
                          << " Mkeys/s | Total: " << h_hashes
                          << " | Progress: " << std::fixed << std::setprecision(4) << (double)coverage << " %";
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

        std::swap(d_Px, d_Rx);
        std::swap(d_Py, d_Ry);

        unsigned int h_any = 0u;
        ck(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read any_left");
        if (h_any == 0u) { completed_all = true; break; }
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
            std::cout << "======== SEARCH COMPLETED (100% Scanned) ===============\n";
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