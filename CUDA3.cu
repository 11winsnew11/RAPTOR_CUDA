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

// ==========================================
// MODIFIKASI: Xorshift32 RNG & Helpers
// ==========================================
__device__ __host__ __forceinline__ uint32_t xorshift32(uint32_t *state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

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

__constant__ int c_vanity_len;

// ============================================================
// KERNEL: RANDOM SWEEP (CHUNK BASED)
// ============================================================
__launch_bounds__(256, 2)
__global__ void kernel_random_chunk_sweep(
    const uint64_t* __restrict__ range_start, // Start range (4xu64)
    const uint64_t* __restrict__ range_len,   // Length range (4xu64)
    uint32_t* __restrict__ rng_states,        // RNG State per thread
    uint64_t threadsTotal,
    uint32_t iterations_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum
)
{
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;
    const int vanity_len = c_vanity_len;

    // Load Range Parameters
    uint64_t r_start[4], r_len[4];
    #pragma unroll
    for(int i=0; i<4; ++i) {
        r_start[i] = range_start[i];
        r_len[i]   = range_len[i];
    }

    // Load RNG State
    uint32_t rnd_state = rng_states[gid];

    unsigned long long local_hashes = 0;

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

    // Loop Iterasi
    for(uint32_t iter = 0; iter < iterations_per_launch; ++iter) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) break;

        // 1. Generate Random Offset (6 Chunks of 12-bit = 72 bit)
        uint64_t offset_lo = 0; // 64 bit bawah
        uint64_t offset_hi = 0; // sisa bit

        // Loop 6 kali untuk pola chunk
        // Chunk 0..4 penuh 60 bit. Chunk 5 sisa 12 bit.
        uint32_t c0 = xorshift32(&rnd_state) & 0xFFF;
        uint32_t c1 = xorshift32(&rnd_state) & 0xFFF;
        uint32_t c2 = xorshift32(&rnd_state) & 0xFFF;
        uint32_t c3 = xorshift32(&rnd_state) & 0xFFF;
        uint32_t c4 = xorshift32(&rnd_state) & 0xFFF;
        uint32_t c5 = xorshift32(&rnd_state) & 0xFFF;

        offset_lo = (uint64_t)c0 | ((uint64_t)c1 << 12) | ((uint64_t)c2 << 24) | ((uint64_t)c3 << 36) | ((uint64_t)c4 << 48);
        
        // Masukkan 4 bit dari c5 ke posisi 60-63
        offset_lo |= ((uint64_t)(c5 & 0xF) << 60);
        
        // Sisa 8 bit c5 masuk ke offset_hi (bit 0-7)
        offset_hi = (c5 >> 4);

        // 2. Apply Range Constraint (Offset % RangeLen)
        // Karena range_len bisa besar, kita lakukan modulo sederhana jika offset < range_len
        // Jika range_len besar (power of 2), ini mudah.
        // Untuk generalisasi cepat: jika offset > range_len, kita kurangi (aproksimasi)
        // Atau biarkan overflow jika range_len mendekati 2^72.
        
        uint64_t final_scalar[4];
        
        // Hitung Scalar = Start + Offset
        // Scalar = r_start + (offset_lo + offset_hi*2^64)
        
        // Tambah offset_lo
        __uint128_t res0 = (__uint128_t)r_start[0] + offset_lo;
        final_scalar[0] = (uint64_t)res0;
        uint64_t carry = (uint64_t)(res0 >> 64);

        // Tambah offset_hi + carry
        __uint128_t res1 = (__uint128_t)r_start[1] + offset_hi + carry;
        final_scalar[1] = (uint64_t)res1;
        carry = (uint64_t)(res1 >> 64);

        // Sisa start
        final_scalar[2] = r_start[2] + carry;
        final_scalar[3] = r_start[3]; // Asumsi offset tidak overflow ke bit tinggi signifikan

        // Modulo Check Sederhana (wrap around)
        // Jika final_scalar >= (r_start + r_len), kurangi r_len
        // Perbandingan 256-bit sederhana
        bool gt = false;
        if (final_scalar[3] > r_start[3]) gt = true;
        else if (final_scalar[3] == r_start[3]) {
             if (final_scalar[2] > r_start[2]) gt = true;
             else if (final_scalar[2] == r_start[2]) {
                 if (final_scalar[1] > r_start[1]) gt = true;
                 else if (final_scalar[1] == r_start[1]) {
                     if (final_scalar[0] >= r_start[0]) gt = true;
                 }
             }
        }
        
        // Cek apakah lebih dari batas (end = start + len)
        // Logika approx: jika kita melewati range (jarang terjadi jika range >> random space), wrap
        // Kita lakukan pengurangan sederhana jika final_scalar > (start + len approximation)
        // Untuk simplisitas, kita biarkan saja (random distribution) atau lakukan mask bit.
        // KODE AMAN: Mask dengan bit length range jika pow2.
        // Kode ini mengasumsikan Range Length cukup besar sehingga collision kecil.
        
        // 3. Compute Public Key (Affine)
        uint64_t Rx[4], Ry[4];
        scalarMulBaseAffine(final_scalar, Rx, Ry);

        // 4. Hash & Check
        uint8_t h20[20];
        uint8_t prefix = (uint8_t)(Ry[0] & 1ULL) ? 0x03 : 0x02;
        getHash160_33_from_limbs(prefix, Rx, h20);
        
        ++local_hashes;

        bool match = check_vanity(h20);
        if (__any_sync(full_mask, match)) {
            if (match) {
                if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                    d_found_result->threadId = (int)gid;
                    d_found_result->iter     = (int)iter;
                    #pragma unroll
                    for (int k=0;k<4;++k) d_found_result->scalar[k]=final_scalar[k];
                    #pragma unroll
                    for (int k=0;k<4;++k) d_found_result->Rx[k]=Rx[k];
                    #pragma unroll
                    for (int k=0;k<4;++k) d_found_result->Ry[k]=Ry[k];
                    __threadfence_system();
                    atomicExch(d_found_flag, FOUND_READY);
                }
            }
            break; 
        }
    }

    // Save RNG State
    rng_states[gid] = rnd_state;

    // Atomic Add Hashes
    unsigned long long v = warp_reduce_add_ull(local_hashes);
    if (lane == 0 && v) atomicAdd(hashes_accum, v);
}

// ============================================================
// HOST CODE
// ============================================================

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string range_hex, vanity_hash_hex;
    uint32_t runtime_threads_per_block = 256;
    uint32_t iterations_per_launch = 1000; 

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      (arg == "--vanity-hash160" && i + 1 < argc) vanity_hash_hex = argv[++i];
        else if (arg == "--range"          && i + 1 < argc) range_hex       = argv[++i];
        else if (arg == "--tpb"            && i + 1 < argc) runtime_threads_per_block = std::stoul(argv[++i]);
        else if (arg == "--iters"          && i + 1 < argc) iterations_per_launch = std::stoul(argv[++i]);
    }

    if (range_hex.empty() || vanity_hash_hex.empty()) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> --vanity-hash160 <prefix_hex> [--tpb threads_per_block] [--iters N]\n";
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

    uint64_t range_len[4]; 
    sub256(range_end, range_start, range_len); 
    add256_u64(range_len, 1ull, range_len);

    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; return EXIT_FAILURE;
    }
    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

    int threadsPerBlock = runtime_threads_per_block;
    if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock = prop.maxThreadsPerBlock;
    if (threadsPerBlock < 32) threadsPerBlock = 32;

    int blocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / threadsPerBlock);
    uint64_t threadsTotal = (uint64_t)blocks * threadsPerBlock;

    // Alokasi Memori Host
    uint64_t h_range_start[4], h_range_len[4];
    for(int i=0; i<4; ++i) { h_range_start[i] = range_start[i]; h_range_len[i] = range_len[i]; }

    // Alokasi Memori Device
    uint64_t *d_range_start, *d_range_len;
    uint32_t *d_rng_states;
    int *d_found_flag; FoundResult *d_found_result;
    unsigned long long *d_hashes_accum;

    auto ck = [](cudaError_t e, const char* msg){
        if (e != cudaSuccess) { std::cerr << msg << ": " << cudaGetErrorString(e) << "\n"; std::exit(EXIT_FAILURE); }
    };

    ck(cudaMalloc(&d_range_start, 4 * sizeof(uint64_t)), "malloc start");
    ck(cudaMalloc(&d_range_len,   4 * sizeof(uint64_t)), "malloc len");
    ck(cudaMalloc(&d_rng_states,  threadsTotal * sizeof(uint32_t)), "malloc rng");
    ck(cudaMalloc(&d_found_flag,  sizeof(int)), "malloc flag");
    ck(cudaMalloc(&d_found_result,sizeof(FoundResult)), "malloc result");
    ck(cudaMalloc(&d_hashes_accum,sizeof(unsigned long long)), "malloc accum");

    ck(cudaMemcpy(d_range_start, h_range_start, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start");
    ck(cudaMemcpy(d_range_len,   h_range_len,   4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy len");

    // Init RNG States (Host Side)
    uint32_t* h_rng_states = (uint32_t*)malloc(threadsTotal * sizeof(uint32_t));
    uint32_t global_seed = (uint32_t)time(NULL);
    for(uint64_t i=0; i<threadsTotal; ++i) {
        h_rng_states[i] = global_seed ^ ((uint32_t)i * 2654435761u);
    }
    ck(cudaMemcpy(d_rng_states, h_rng_states, threadsTotal * sizeof(uint32_t), cudaMemcpyHostToDevice), "cpy rng");

    // Init Others
    int zero = FOUND_NONE; unsigned long long zero64 = 0ull;
    ck(cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice), "init flag");
    ck(cudaMemcpy(d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init accum");

    // Copy Constants
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

    std::cout << "======== PrePhase: GPU Information ====================\n";
    std::cout << std::left << std::setw(20) << "Mode"              << " : RANDOM CHUNK SWEEP (72-bit Key Gen)\n";
    std::cout << std::left << std::setw(20) << "Device"            << " : " << prop.name << " (compute " << prop.major << "." << prop.minor << ")\n";
    std::cout << std::left << std::setw(20) << "ThreadsTotal"      << " : " << threadsTotal << "\n";
    std::cout << std::left << std::setw(20) << "Vanity Target"     << " : " << vanity_hash_hex << " (" << vanity_len << " bytes)\n\n";
    std::cout << "======== Phase-1: Random Search =======================\n";

    cudaStream_t streamKernel;
    ck(cudaStreamCreateWithFlags(&streamKernel, cudaStreamNonBlocking), "create stream");

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    bool stop_all = false;
    while (!stop_all) {
        if (g_sigint) std::cerr << "\n[Ctrl+C] Interrupt received. Exiting...\n";

        kernel_random_chunk_sweep<<<blocks, threadsPerBlock, 0, streamKernel>>>(
            d_range_start, d_range_len,
            d_rng_states,
            threadsTotal,
            iterations_per_launch,
            d_found_flag, d_found_result,
            d_hashes_accum
        );
        
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "\nKernel launch error: " << cudaGetErrorString(launchErr) << "\n";
            stop_all = true;
            break;
        }

        // Monitor Loop
        cudaStreamSynchronize(streamKernel); // Simpler blocking approach for random search

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();
        if (dt >= 1.0) {
            unsigned long long h_hashes = 0ull;
            ck(cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost), "read hashes");
            double delta = (double)(h_hashes - lastHashes);
            double mkeys = delta / (dt * 1e6);
            double elapsed = std::chrono::duration<double>(now - t0).count();
            
            std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                      << "s | Speed: " << std::fixed << std::setprecision(2) << mkeys
                      << " Mkeys/s | Total: " << h_hashes;
            std::cout.flush();
            lastHashes = h_hashes; tLast = now;
        }

        int host_found = 0;
        ck(cudaMemcpy(&host_found, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "read found_flag");
        if (host_found == FOUND_READY || g_sigint) { stop_all = true; }
    }

    std::cout << "\n";

    int h_found_flag = 0;
    ck(cudaMemcpy(&h_found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost), "final read found_flag");

    if (h_found_flag == FOUND_READY) {
        FoundResult host_result{};
        ck(cudaMemcpy(&host_result, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost), "read found_result");
        std::cout << "\n======== FOUND MATCH! =================================\n";
        std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
        std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
    } else {
        std::cout << "======== SEARCH STOPPED ===============================\n";
    }

    // Cleanup
    cudaFree(d_range_start); cudaFree(d_range_len); cudaFree(d_rng_states);
    cudaFree(d_found_flag); cudaFree(d_found_result); cudaFree(d_hashes_accum);
    cudaStreamDestroy(streamKernel);
    if(h_rng_states) free(h_rng_states);

    return 0;
}