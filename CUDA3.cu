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
#include <algorithm>

// Include headers lainnya (asumsi ada)
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

// Constant Memory
__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];
__constant__ uint64_t c_deltas[(MAX_BATCH_SIZE/2) * 4]; 
__constant__ uint64_t c_J_delta[4]; 

__constant__ uint32_t c_vanity_len;
__constant__ uint32_t c_vanity_prefix_mask;

// Helper Device Functions
__device__ void add256_with_carry(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 res = (unsigned __int128)a[i] + b[i] + carry;
        r[i] = (uint64_t)res;
        carry = (uint64_t)(res >> 64);
    }
}

__device__ void sub256_with_borrow(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t borrow = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint64_t t = a[i] - b[i];
        uint64_t b_new = (a[i] < b[i]) ? 1 : 0;
        r[i] = t - borrow;
        borrow = b_new + ((t < borrow) ? 1 : 0);
    }
}

// Kernel tidak berubah, logika invers tetap sama karena delta antar titik konstan
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
    const uint32_t vanity_len = c_vanity_len;
    const uint32_t vanity_mask = c_vanity_prefix_mask;

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

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // --- Blok Pengecekan Vanity (Base Point P) ---
        {
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            uint32_t h_prefix = load_u32_le(h20);
            bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

            if (__any_sync(full_mask, pref)) {
                bool full_match = false;
                if (pref) {
                    if (vanity_len <= 4) full_match = true;
                    else {
                        full_match = true;
                        for (uint32_t k = 4; k < vanity_len; ++k) {
                            if (h20[k] != c_target_hash160[k]) { full_match = false; break; }
                        }
                    }
                }

                if (full_match) {
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

        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t current_delta[4];
            #pragma unroll
            for (int k=0; k<4; ++k) current_delta[k] = c_deltas[(size_t)i*4 + k];

            // --- Check +R_i ---
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

                uint32_t h_prefix = load_u32_le(h20);
                bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

                if (__any_sync(full_mask, pref)) {
                    bool full_match = false;
                    if (pref) {
                        if (vanity_len <= 4) full_match = true;
                        else {
                            full_match = true;
                            for (uint32_t k=4; k<vanity_len; ++k) if (h20[k] != c_target_hash160[k]) { full_match=false; break; }
                        }
                    }
                    
                    if (full_match) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; 
                            add256_with_carry(S, current_delta, fs);
                            
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

            // --- Check -R_i ---
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

                uint32_t h_prefix = load_u32_le(h20);
                bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

                if (__any_sync(full_mask, pref)) {
                    bool full_match = false;
                    if (pref) {
                        if (vanity_len <= 4) full_match = true;
                        else {
                            full_match = true;
                            for (uint32_t k=4; k<vanity_len; ++k) if (h20[k] != c_target_hash160[k]) { full_match=false; break; }
                        }
                    }
                    
                    if (full_match) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; 
                            sub256_with_borrow(S, current_delta, fs);

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

        // Last Iteration
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

            uint32_t h_prefix = load_u32_le(h20);
            bool pref = ((h_prefix & vanity_mask) == (target_prefix & vanity_mask));

            if (__any_sync(full_mask, pref)) {
                bool full_match = false;
                if (pref) {
                    if (vanity_len <= 4) full_match = true;
                    else {
                        full_match = true;
                        for (uint32_t k=4; k<vanity_len; ++k) if (h20[k] != c_target_hash160[k]) { full_match=false; break; }
                    }
                }
                
                if (full_match) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                         uint64_t current_delta[4];
                        #pragma unroll
                        for (int k=0; k<4; ++k) current_delta[k] = c_deltas[(size_t)i*4 + k];

                        uint64_t fs[4]; 
                        sub256_with_borrow(S, current_delta, fs); 

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

        // Jump Logic
        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_Jy[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=c_Jx[j];
            ModSub256(x3, x3, Jx_local);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

#pragma unroll
            for (int j=0;j<4;++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
        }

        // Update Scalar S
        {
            // S = S + c_J_delta
            // Di mode Strided, c_J_delta = B * R
            uint64_t carry = 0;
            #pragma unroll
            for (int k=0;k<4;++k) {
                unsigned __int128 res = (unsigned __int128)S[k] + c_J_delta[k] + carry;
                S[k] = (uint64_t)res;
                carry = (uint64_t)(res >> 64);
            }
            sub256_u64_inplace(rem, (uint64_t)B); 
        }
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
}

// Host Helpers
extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern bool decode_p2pkh_address(const std::string& addr, uint8_t out20[20]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

void mul256_u64(const uint64_t a[4], uint64_t b, uint64_t r[4]) {
    unsigned __int128 res;
    uint64_t carry = 0;
    res = (unsigned __int128)a[0] * b; r[0] = (uint64_t)res; carry = res >> 64;
    res = (unsigned __int128)a[1] * b + carry; r[1] = (uint64_t)res; carry = res >> 64;
    res = (unsigned __int128)a[2] * b + carry; r[2] = (uint64_t)res; carry = res >> 64;
    res = (unsigned __int128)a[3] * b + carry; r[3] = (uint64_t)res;
}

// Helper penjumlahan 256-bit di Host
void add256_host(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        unsigned __int128 res = (unsigned __int128)a[i] + b[i] + carry;
        r[i] = (uint64_t)res;
        carry = (uint64_t)(res >> 64);
    }
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string vanity_hex, range_hex;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         = 64;
    
    bool random_mode = false;
    uint64_t random_seed = 0;

    // Parsing logic
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
        if      (arg == "--vanity-hash160" && i + 1 < argc) vanity_hex = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex  = argv[++i];
        else if (arg == "--random") {
            random_mode = true;
            if (i + 1 < argc && argv[i+1][0] != '-') {
                random_seed = std::stoull(argv[++i]);
            } else {
                std::random_device rd;
                std::mt19937_64 gen(rd());
                std::uniform_int_distribution<uint64_t> dis;
                random_seed = dis(gen);
            }
        }
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

    if (range_hex.empty() || vanity_hex.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> --vanity-hash160 <prefix_hex> [--random [seed]] [--grid A,B] [--slices N]\n";
        return EXIT_FAILURE;
    }

    if (vanity_hex.length() > 40) { std::cerr << "Error: Vanity prefix too long.\n"; return EXIT_FAILURE; }
    if (vanity_hex.length() % 2 != 0) { std::cerr << "Error: Vanity prefix length must be even.\n"; return EXIT_FAILURE; }

    uint8_t target_hash160[20] = {0}; 
    uint32_t vanity_len = vanity_hex.length() / 2;
    std::string padded_vanity = vanity_hex;
    if (padded_vanity.length() < 40) padded_vanity += std::string(40 - padded_vanity.length(), '0');
    if (!hexToHash160(padded_vanity, target_hash160)) { std::cerr << "Error: Invalid hex for vanity.\n"; return EXIT_FAILURE; }

    uint32_t vanity_prefix_mask = 0xFFFFFFFFu;
    if (vanity_len < 4) vanity_prefix_mask = (1ULL << (vanity_len * 8)) - 1;

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }

    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << " (kernel limit).\n";
        return EXIT_FAILURE;
    }

    uint64_t range_len[4]; sub256(range_end, range_start, range_len); add256_u64(range_len, 1ull, range_len);

    auto is_zero_256 = [](const uint64_t a[4])->bool { return (a[0]|a[1]|a[2]|a[3]) == 0ull; };
    auto is_power_of_two_256 = [&](const uint64_t a[4])->bool {
        if (is_zero_256(a)) return false;
        uint64_t am1[4]; uint64_t borrow = 1ull;
        for (int i=0;i<4;++i) {
            uint64_t v = a[i] - borrow; borrow = (a[i] < borrow) ? 1ull : 0ull; am1[i] = v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) am1[k] = a[k]; break; }
        }
        uint64_t and0=a[0]&am1[0], and1=a[1]&am1[1], and2=a[2]&am1[2], and3=a[3]&am1[3];
        return (and0|and1|and2|and3)==0ull;
    };
    if (!is_power_of_two_256(range_len)) {
        std::cerr << "Error: range length (end - start + 1) must be a power of two.\n"; return EXIT_FAILURE;
    }
    
    uint64_t len_minus1[4];
    {   uint64_t borrow=1ull;
        for (int i=0;i<4;++i) {
            uint64_t v=range_len[i]-borrow; borrow=(range_len[i]<borrow)?1ull:0ull; len_minus1[i]=v;
            if (!borrow && i+1<4) { for (int k=i+1;k<4;++k) len_minus1[k]=range_len[k]; break; }
        }
    }
    {   
        uint64_t and0 = range_start[0] & len_minus1[0];
        uint64_t and1 = range_start[1] & len_minus1[1];
        uint64_t and2 = range_start[2] & len_minus1[2];
        uint64_t and3 = range_start[3] & len_minus1[3];
        if ((and0|and1|and2|and3) != 0ull) {
            std::cerr << "Error: start must be aligned to the range length.\n"; return EXIT_FAILURE;
        }
    }

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

    uint64_t q_div_batch[4], r_div_batch = 0ull;
    divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
    if (r_div_batch != 0ull) {
        std::cerr << "Error: range length must be divisible by batch size.\n";
        return EXIT_FAILURE;
    }
    bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
    uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
    if (!q_fits_u64) { std::cerr << "Error: total batches too large for u64.\n"; return EXIT_FAILURE; }

    uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
    if (userUpper == 0ull) userUpper = UINT64_MAX;

    auto pick_threads_total_pow2 = [&](uint64_t upper)->uint64_t {
        if (upper < (uint64_t)threadsPerBlock) return 0ull;
        uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
        uint64_t q = total_batches_u64;
        
        uint64_t pot = 1ull;
        while (pot <= t) pot <<= 1;
        pot >>= 1; 
        
        while (pot >= (uint64_t)threadsPerBlock) {
            if (pot > 0 && (q % pot) == 0ull) return pot;
            pot >>= 1;
        }
        return 0ull;
    };

    uint64_t upper = maxThreadsByMem;
    if (total_batches_u64 < upper) upper = total_batches_u64;
    if (userUpper         < upper) upper = userUpper;

    uint64_t threadsTotal = pick_threads_total_pow2(upper);
    
    if (threadsTotal == 0ull) {
         std::cerr << "Error: Could not determine valid thread count.\n";
         return EXIT_FAILURE;
    }
    
    int blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

    uint64_t per_thread_cnt[4]; uint64_t r_u64 = 0ull;
    divmod_256_by_u64(range_len, threadsTotal, per_thread_cnt, r_u64);
    if (r_u64 != 0ull) { std::cerr << "Internal error: range_len not divisible by threadsTotal.\n"; return EXIT_FAILURE; }
    
    {   uint64_t qq[4], rr=0ull;
        divmod_256_by_u64(per_thread_cnt, (uint64_t)runtime_points_batch_size, qq, rr);
        if (rr != 0ull) { std::cerr << "Internal error: per-thread count is not a multiple of batch size.\n"; return EXIT_FAILURE; }
    }

    std::mt19937_64 rng(random_seed);

    uint64_t* h_counts256     = nullptr;
    uint64_t* h_start_scalars = nullptr;
    cudaHostAlloc(&h_counts256,     threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
    cudaHostAlloc(&h_start_scalars, threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

    for (uint64_t i = 0; i < threadsTotal; ++i) {
        h_counts256[i*4+0] = per_thread_cnt[0];
        h_counts256[i*4+1] = per_thread_cnt[1];
        h_counts256[i*4+2] = per_thread_cnt[2];
        h_counts256[i*4+3] = per_thread_cnt[3];
    }

    const uint32_t B = runtime_points_batch_size;
    const uint32_t half = B >> 1;

    // ==========================================================
    // --- PRECOMPUTATION: STRIDED RANDOM BATCH LOGIC ---------
    // ==========================================================

    uint64_t* h_batch_deltas = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
    uint64_t h_jump_delta[4] = {0,0,0,0};
    uint64_t h_stride_scalar[4] = {0,0,0,0}; // R

    if (random_mode) {
        std::cout << "======= MODE: STRIDED RANDOM BATCH SEARCH ========\n";
        std::cout << "Seed: " << std::hex << random_seed << std::dec << "\n";
        
        // 1. Generate Random Stride (R)
        // R harus ganjil agar menjamin full-cycle coverage pada range power-of-2
        std::uniform_int_distribution<uint64_t> dis64(1, UINT64_MAX);
        h_stride_scalar[0] = dis64(rng);
        h_stride_scalar[1] = dis64(rng);
        h_stride_scalar[2] = dis64(rng);
        h_stride_scalar[3] = dis64(rng);
        
        // Pastikan ganjil
        h_stride_scalar[0] |= 1ULL; 

        std::cout << "Random Stride (R) : " << formatHex256(h_stride_scalar) << "\n";
        std::cout << "Search path is randomized (Strided) but strictly within range.\n";

        // 2. Compute Batch Deltas: R, 2R, 3R... (mod 2^256 implied by overflow)
        uint64_t current_acc[4] = {0,0,0,0};
        for (uint32_t i = 0; i < half; ++i) {
            add256_host(current_acc, h_stride_scalar, current_acc); // current_acc += R
            std::memcpy(&h_batch_deltas[(size_t)i*4], current_acc, 4*sizeof(uint64_t));
        }

        // 3. Compute Jump Delta: J = B * R
        mul256_u64(h_stride_scalar, (uint64_t)B, h_jump_delta);

    } else {
        // Mode Linear: Stride = 1
        h_stride_scalar[0] = 1; 
        
        for (uint32_t i = 0; i < half; ++i) {
            std::memset(&h_batch_deltas[(size_t)i*4], 0, 4*sizeof(uint64_t));
            h_batch_deltas[(size_t)i*4 + 0] = (uint64_t)(i + 1); 
        }
        h_jump_delta[0] = B; 
    }

    std::cout << "Precomputing Strided Batch Points on GPU...\n";

    // 4. Hitung Titik G * Deltas di GPU
    uint64_t *d_delta_scalars, *d_temp_Gx, *d_temp_Gy;
    cudaMalloc(&d_delta_scalars, (size_t)half * 4 * sizeof(uint64_t));
    cudaMalloc(&d_temp_Gx, (size_t)half * 4 * sizeof(uint64_t));
    cudaMalloc(&d_temp_Gy, (size_t)half * 4 * sizeof(uint64_t));

    cudaMemcpy(d_delta_scalars, h_batch_deltas, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);

    int blocks_scal = (half + 255) / 256;
    scalarMulKernelBase<<<blocks_scal, 256>>>(d_delta_scalars, d_temp_Gx, d_temp_Gy, half);
    cudaDeviceSynchronize();

    uint64_t* h_temp_Gx = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
    uint64_t* h_temp_Gy = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
    cudaMemcpy(h_temp_Gx, d_temp_Gx, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_temp_Gy, d_temp_Gy, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    cudaMemcpyToSymbol(c_Gx, h_temp_Gx, (size_t)half * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_Gy, h_temp_Gy, (size_t)half * 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_deltas, h_batch_deltas, (size_t)half * 4 * sizeof(uint64_t));

    // 5. Hitung Titik Jump (J) dan Delta-nya
    uint64_t *d_jump_scalar, *d_temp_Jx, *d_temp_Jy;
    cudaMalloc(&d_jump_scalar, 4 * sizeof(uint64_t));
    cudaMalloc(&d_temp_Jx, 4 * sizeof(uint64_t));
    cudaMalloc(&d_temp_Jy, 4 * sizeof(uint64_t));
    
    cudaMemcpy(d_jump_scalar, h_jump_delta, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    scalarMulKernelBase<<<1, 1>>>(d_jump_scalar, d_temp_Jx, d_temp_Jy, 1);
    cudaDeviceSynchronize();

    uint64_t h_Jx[4], h_Jy[4];
    cudaMemcpy(h_Jx, d_temp_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Jy, d_temp_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaMemcpyToSymbol(c_Jx, h_Jx, 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_Jy, h_Jy, 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_J_delta, h_jump_delta, 4 * sizeof(uint64_t));

    // Cleanup Precompute Temp Memory
    std::free(h_batch_deltas); std::free(h_temp_Gx); std::free(h_temp_Gy);
    cudaFree(d_delta_scalars); cudaFree(d_temp_Gx); cudaFree(d_temp_Gy);
    cudaFree(d_jump_scalar); cudaFree(d_temp_Jx); cudaFree(d_temp_Jy);

    // ==========================================================
    // --- INISIALISASI SCALAR (RANDOM START) -------------------
    // ==========================================================
    
    // Logika Start Scalar bisa dipertahankan (XOR Shuffle) untuk memastikan
    // thread memulai dari titik yang berbeda jauh (non-overlapping start chunks).
    // Jika ingin purely random independent, bisa diganti, tapi XOR shuffle lebih aman.
    
    if (random_mode) {
        std::cout << "Applying Random Start logic (XOR Shuffle)...\n";
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            uint64_t mask = threadsTotal - 1;
            uint64_t scrambled_idx = i ^ (random_seed & mask);
            
            uint64_t offset_scalar[4] = {0,0,0,0};
            mul256_u64(per_thread_cnt, scrambled_idx, offset_scalar);
            
            uint64_t temp_scalar[4];
            add256(range_start, offset_scalar, temp_scalar);
            
            uint64_t Sc[4];
            add256_u64(temp_scalar, (uint64_t)half, Sc);
            
            h_start_scalars[i*4+0] = Sc[0];
            h_start_scalars[i*4+1] = Sc[1];
            h_start_scalars[i*4+2] = Sc[2];
            h_start_scalars[i*4+3] = Sc[3];
        }
    } else {
        uint64_t cur[4] = { range_start[0], range_start[1], range_start[2], range_start[3] };
        for (uint64_t i = 0; i < threadsTotal; ++i) {
            uint64_t Sc[4]; add256_u64(cur, (uint64_t)half, Sc); 
            h_start_scalars[i*4+0] = Sc[0];
            h_start_scalars[i*4+1] = Sc[1];
            h_start_scalars[i*4+2] = Sc[2];
            h_start_scalars[i*4+3] = Sc[3];

            uint64_t next[4]; add256(cur, per_thread_cnt, next);
            cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
        }
    }

    // Copy Target and Vanity Config
    {
        uint32_t prefix_le = (uint32_t)target_hash160[0]
                           | ((uint32_t)target_hash160[1] << 8)
                           | ((uint32_t)target_hash160[2] << 16)
                           | ((uint32_t)target_hash160[3] << 24);
        
        cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
        cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
        cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(vanity_len));
        cudaMemcpyToSymbol(c_vanity_prefix_mask, &vanity_prefix_mask, sizeof(vanity_prefix_mask));
    }

    // Device Malloc
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

    // Kernel: Generate Initial Points (Px, Py)
    {
        int blocks_scal = (int)((threadsTotal + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_start_scalars, d_Px, d_Py, (int)threadsTotal);
        ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");
        ck(cudaGetLastError(), "scalarMulKernelBase launch");
    }

    // Info Print
    size_t freeB=0,totalB=0; cudaMemGetInfo(&freeB,&totalB);
    size_t usedB = totalB - freeB;
    double util = totalB ? (double)usedB * 100.0 / (double)totalB : 0.0;

    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "SM"                << " : " << prop.multiProcessorCount << "\n";
    std::cout << std::left << std::setw(20) << "ThreadsPerBlock"   << " : " << threadsPerBlock << "\n";
    std::cout << std::left << std::setw(20) << "Blocks"            << " : " << blocks << "\n";
    std::cout << std::left << std::setw(20) << "Points batch size" << " : " << B << "\n";
    std::cout << std::left << std::setw(20) << "Batches/SM"        << " : " << runtime_batches_per_sm << "\n";
    std::cout << std::left << std::setw(20) << "Batches/launch"    << " : " << slices_per_launch << " (per thread)\n";
    std::cout << std::left << std::setw(20) << "Memory utilization"<< " : "
              << std::fixed << std::setprecision(1) << util << "% ("
              << human_bytes((double)usedB) << " / " << human_bytes((double)totalB) << ")\n";
    std::cout << "------------------------------------------------------- \n";
    std::cout << std::left << std::setw(20) << "Total threads"     << " : " << (uint64_t)threadsTotal << "\n\n";
    std::cout << "======== Phase-1: Vanity Search ==========================\n";

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
                long double total_keys_ld = ld_from_u256(range_len);
                long double prog = total_keys_ld > 0.0L ? ((long double)h_hashes / total_keys_ld) * 100.0L : 0.0L;
                if (prog > 100.0L) prog = 100.0L;
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << " s | Speed: " << std::fixed << std::setprecision(1) << mkeys
                          << " Mkeys/s | Count: " << h_hashes
                          << " | Progress: " << std::fixed << std::setprecision(2) << (double)prog << " %";
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

        unsigned int h_any = 0u;
        ck(cudaMemcpy(&h_any, d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read any_left");

        std::swap(d_Px, d_Rx);
        std::swap(d_Py, d_Ry);

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
            std::cout << "Search was interrupted by user. Partial progress above.\n";
            exit_code = 130;
        } else if (completed_all) {
            std::cout << "======== KEY NOT FOUND (exhaustive) ===================\n";
            std::cout << "Vanity prefix not found within the specified range.\n";
        } else {
            std::cout << "======== TERMINATED ===================================\n";
        }
    }

    cudaFree(d_start_scalars); cudaFree(d_Px); cudaFree(d_Py); cudaFree(d_Rx); cudaFree(d_Ry);
    cudaFree(d_counts256); cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_hashes_accum); cudaFree(d_any_left);
    cudaStreamDestroy(streamKernel);

    if (h_start_scalars) cudaFreeHost(h_start_scalars);
    if (h_counts256)     cudaFreeHost(h_counts256);

    return exit_code;
}