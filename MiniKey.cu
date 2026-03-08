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

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_RangeStart[4]; 
__constant__ uint64_t c_RangeMask[4];   // Mask = RangeLen - 1 (untuk rentang pangkat 2)
__constant__ int c_vanity_len;

// ============================================================
// KERNEL: RANDOM SEARCH WITH CORRECT RANGE MASKING
// ============================================================
__launch_bounds__(256, 2)
__global__ void kernel_random_search(
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum,
    uint32_t batch_size,
    uint32_t iters_per_launch 
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    
    // RNG State (Xorshift128+)
    uint64_t rng_s0 = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t rng_s1 = (uint64_t)clock64() ^ (rng_s0 << 32);
    if (rng_s0 == 0) rng_s0 = 1; 
    
    auto xorshift128plus = [&]() -> uint64_t {
        uint64_t s1 = rng_s0;
        uint64_t s0 = rng_s1;
        rng_s0 = s0;
        s1 ^= s1 << 23;
        return (rng_s1 = (s1 ^ s0 ^ (s1 >> 17) ^ (s0 >> 26))) + s0;
    };

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    const uint32_t target_prefix = c_target_prefix;
    const int vanity_len = c_vanity_len;

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

    // Loop utama random search
    for (uint32_t iter = 0; iter < iters_per_launch; ++iter) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        // 1. Generate Random Scalar k (256-bit)
        uint64_t k_rand[4];
        k_rand[0] = xorshift128plus();
        k_rand[1] = xorshift128plus();
        k_rand[2] = xorshift128plus();
        k_rand[3] = xorshift128plus();

        // 2. Apply Range Masking (Power of 2 Range Optimization)
        // k_final = (k_rand & Mask) + RangeStart
        // Ini menjamin k_final berada dalam [Start, Start + Len-1] jika Len adalah pangkat 2.
        uint64_t k_final[4];
        
        // Operasi AND 256-bit
        k_final[0] = k_rand[0] & c_RangeMask[0];
        k_final[1] = k_rand[1] & c_RangeMask[1];
        k_final[2] = k_rand[2] & c_RangeMask[2];
        k_final[3] = k_rand[3] & c_RangeMask[3];

        // Operasi ADD 256-bit: k_final += RangeStart
        // Dengan carry propagation
        {
            uint64_t carry = 0;
            // Limb 0
            __uint128_t res = (__uint128_t)k_final[0] + c_RangeStart[0];
            k_final[0] = (uint64_t)res;
            carry = (uint64_t)(res >> 64);

            // Limb 1
            res = (__uint128_t)k_final[1] + c_RangeStart[1] + carry;
            k_final[1] = (uint64_t)res;
            carry = (uint64_t)(res >> 64);

            // Limb 2
            res = (__uint128_t)k_final[2] + c_RangeStart[2] + carry;
            k_final[2] = (uint64_t)res;
            carry = (uint64_t)(res >> 64);

            // Limb 3
            k_final[3] = k_final[3] + c_RangeStart[3] + carry;
        }

        // 3. Hitung Titik P = k_final * G
        uint64_t x1[4], y1[4];
        scalarMulBaseAffine(k_final, x1, y1);

        // 4. Cek Titik Pusat (P)
        {
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();

            bool match = check_vanity(h20);
            if (__any_sync(full_mask, match)) {
                if (match) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                        d_found_result->iter     = iter;
                        #pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=k_final[k];
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

        // 5. Batch Neighbors Check (Logika sama seperti sebelumnya)
        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

        #pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Gx[(size_t)0*4 + j]; 
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

        uint64_t d0[4];
        uint64_t inverse_full[5]; 
        
        #pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
        #pragma unroll
        for (int j=0;j<4;++j) inverse_full[j] = d0[j];
        inverse_full[4] = 0;
        
        _ModMult(inverse_full, subp[0]); 
        _ModInv(inverse_full); 

        // Loop pengecekan batch (+G dan -G)
        for (int i = 0; i < half - 1; ++i) {
             if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

             uint64_t dx_inv_i[4];
             _ModMult(dx_inv_i, subp[i], inverse_full); 

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
                             uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=k_final[k];
                             uint64_t addv=(uint64_t)(i+1);
                             for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
                             #pragma unroll
                             for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                             #pragma unroll
                             for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                             d_found_result->threadId = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                             d_found_result->iter     = iter;
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
                             uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=k_final[k];
                             uint64_t subv=(uint64_t)(i+1);
                             for (int k=0;k<4 && subv;++k){ uint64_t old=fs[k]; fs[k]=old-subv; subv=(old<subv)?1ull:0ull; }
                             #pragma unroll
                             for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
                             #pragma unroll
                             for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                             d_found_result->threadId = (int)(blockIdx.x * blockDim.x + threadIdx.x);
                             d_found_result->iter     = iter;
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
             _ModMult(inverse_full, inverse_full, gxmi);
        }
    }
    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

// Helper Host Functions
void mul256_u64(const uint64_t a[4], uint64_t b, uint64_t r[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        __uint128_t res = (__uint128_t)a[i] * b + carry;
        r[i] = (uint64_t)res;
        carry = (uint64_t)(res >> 64);
    }
}

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string range_hex, vanity_hash_hex;
    uint32_t runtime_points_batch_size = 1024; 
    uint32_t iters_per_launch          = 64; 

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--vanity-hash160" && i + 1 < argc) vanity_hash_hex = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--grid"           && i + 1 < argc) { i++; /* skip */ }
        else if (arg == "--slices"         && i + 1 < argc) { i++; /* skip */ }
    }

    if (range_hex.empty() || vanity_hash_hex.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> --vanity-hash160 <prefix_hex>\n";
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
    
    if (vanity_hash_hex.length() > 40 || vanity_hash_hex.length() % 2 != 0) { std::cerr << "Error: Vanity hash len\n"; return EXIT_FAILURE; }
    uint8_t target_hash160[20]; memset(target_hash160, 0, 20);
    for (size_t i = 0; i < vanity_hash_hex.length() / 2; ++i) {
        std::string byteStr = vanity_hash_hex.substr(i * 2, 2);
        target_hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    int vanity_len = (int)(vanity_hash_hex.length() / 2);

    // Hitung RangeLen
    uint64_t range_len[4]; 
    sub256(range_end, range_start, range_len); 
    add256_u64(range_len, 1ull, range_len);

    // Hitung RangeMask (Len - 1)
    uint64_t range_mask[4];
    for(int i=0; i<4; ++i) range_mask[i] = range_len[i];
    
    // range_mask = range_len - 1
    uint64_t borrow = 1;
    for (int i = 0; i < 4; ++i) {
        if (range_mask[i] >= borrow) {
            range_mask[i] -= borrow;
            borrow = 0;
        } else {
            range_mask[i] -= borrow; // will underflow
            borrow = 1;
        }
    }

    // Verifikasi apakah RangeLen adalah pangkat 2
    // Jika RangeLen adalah 2^n, maka RangeMask harus semua 1s (biner)
    // Cara cek: (Len & (Len-1)) == 0. 
    // Di 256-bit: pastikan hanya 1 bit yang set di range_len.
    bool is_pow2 = false;
    int bit_count = 0;
    for(int i=0; i<4; ++i) {
        bit_count += __builtin_popcountll(range_len[i]);
    }
    if (bit_count == 1) is_pow2 = true;

    if (!is_pow2) {
        std::cerr << "Warning: Range Length is NOT a power of two. Random distribution will be slightly biased.\n";
        std::cerr << "For perfect distribution, use ranges like 8000...000 to 8FFF...FFF (power of 2 length).\n";
        // Jika bukan pangkat 2, masking bitwise tidak presisi, tapi kita tetap lanjutkan dengan mask saat ini (pembulatan ke bawah)
        // Atau bisa implementasi rejection sampling (lebih lambat).
        // Untuk simplisitas, kita lanjutkan masking bitwise (cepat tapi biased jika range tidak 'bulat').
    }

    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) { std::cerr<<"CUDA init error\n"; return EXIT_FAILURE; }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int threadsPerBlock = 256;
    int blocks = prop.multiProcessorCount * 8; 
    uint64_t threadsTotal = (uint64_t)blocks * threadsPerBlock;

    cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    cudaMemcpyToSymbol(c_vanity_len, &vanity_len, sizeof(int));
    cudaMemcpyToSymbol(c_RangeStart, range_start, 4 * sizeof(uint64_t));
    cudaMemcpyToSymbol(c_RangeMask, range_mask, 4 * sizeof(uint64_t)); // Upload Mask!

    uint32_t prefix_le = 0;
    if (vanity_len >= 4) {
         prefix_le = (uint32_t)target_hash160[0] | ((uint32_t)target_hash160[1] << 8) | ((uint32_t)target_hash160[2] << 16) | ((uint32_t)target_hash160[3] << 24);
    }
    cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));

    int *d_found_flag=nullptr; FoundResult *d_found_result=nullptr;
    unsigned long long *d_hashes_accum=nullptr; 

    auto ck = [](cudaError_t e, const char* msg){ if (e != cudaSuccess) { std::cerr << msg << ": " << cudaGetErrorString(e) << "\n"; std::exit(EXIT_FAILURE); } };

    ck(cudaMalloc(&d_found_flag, sizeof(int)), "malloc flag");
    ck(cudaMalloc(&d_found_result, sizeof(FoundResult)), "malloc result");
    ck(cudaMalloc(&d_hashes_accum, sizeof(unsigned long long)), "malloc accum");
    
    { int zero = FOUND_NONE; unsigned long long zero64=0ull;
      ck(cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "init flag");
      ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init accum"); }

    // Precompute G Table
    {
        const uint32_t B = runtime_points_batch_size;
        const uint32_t half = B >> 1;
        uint64_t* h_scalars_half = nullptr;
        cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        std::memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
        for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4 + 0] = (uint64_t)(k + 1);

        uint64_t *d_scalars_half=nullptr, *d_Gx_half=nullptr, *d_Gy_half=nullptr;
        ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "malloc");
        ck(cudaMalloc(&d_Gx_half,      (size_t)half * 4 * sizeof(uint64_t)), "malloc");
        ck(cudaMalloc(&d_Gy_half,      (size_t)half * 4 * sizeof(uint64_t)), "malloc");
        ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy");

        int blocks_scal = (int)((half + threadsPerBlock - 1) / threadsPerBlock);
        scalarMulKernelBase<<<blocks_scal, threadsPerBlock>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
        ck(cudaDeviceSynchronize(), "sync");

        uint64_t* h_Gx_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        uint64_t* h_Gy_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
        ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H");
        ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H");
        ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol");
        ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol");

        cudaFree(d_scalars_half); cudaFree(d_Gx_half); cudaFree(d_Gy_half);
        cudaFreeHost(h_scalars_half);
        std::free(h_Gx_half); std::free(h_Gy_half);
    }

    std::cout << "======== Mode: RANDOM SEARCH (Strict Range) ========\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Batch Size: " << runtime_points_batch_size << "\n";
    std::cout << "Threads: " << threadsTotal << "\n";
    std::cout << "================================================\n";

    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "create stream");

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    bool stop_all = false;
    while (!stop_all) {
        if (g_sigint) std::cerr << "\n[Interruption] Exiting...\n";

        kernel_random_search<<<blocks, threadsPerBlock, 0, streamKernel>>>(
            d_found_flag, d_found_result, d_hashes_accum,
            runtime_points_batch_size, iters_per_launch
        );
        
        if (cudaGetLastError() != cudaSuccess) { stop_all = true; }

        for (int k=0; k<10 && !stop_all; ++k) {
            auto now = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration<double>(now - tLast).count();
            if (dt >= 0.1) {
                unsigned long long h_hashes = 0ull;
                ck(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read hashes");
                double delta = (double)(h_hashes - lastHashes);
                double mkeys = delta / (dt * 1e6);
                double elapsed = std::chrono::duration<double>(now - t0).count();
                
                std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                          << "s | Speed: " << std::fixed << std::setprecision(2) << mkeys
                          << " Mkeys/s | Total: " << h_hashes << "    ";
                std::cout.flush();
                lastHashes = h_hashes; tLast = now;
            }

            int host_found = 0;
            ck(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "read flag");
            if (host_found == FOUND_READY) { stop_all = true; break; }

            cudaError_t qs = cudaStreamQuery(streamKernel);
            if (qs == cudaSuccess) break; 
            else if (qs != cudaErrorNotReady) { cudaGetLastError(); stop_all = true; break; }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (stop_all || g_sigint) break;
    }

    cudaDeviceSynchronize();
    std::cout << "\n";

    int h_found_flag = 0;
    ck(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "final flag");

    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "read result");
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
    } else {
        std::cout << "Search stopped.\n";
    }

    cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_hashes_accum);
    cudaStreamDestroy(streamKernel);
    return (h_found_flag == FOUND_READY) ? EXIT_SUCCESS : EXIT_FAILURE;
}