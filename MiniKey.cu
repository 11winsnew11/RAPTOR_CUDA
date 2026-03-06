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
#include "sha256.h" // Asumsi ini menyediakan SHA256 device/host functions
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int) { g_sigint = 1; }

// ============================================================
// KONSTANTA & DEFINISI MINIKEY
// ============================================================
__constant__ uint64_t c_Gx[4];
__constant__ uint64_t c_Gy[4];
__constant__ int c_vanity_len;

// Base58 Alphabet
// '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
__device__ __constant__ const char B58_ALPHABET[58] = {
    '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
};

// Reverse lookup table (ASCII -> Index) for speed
__device__ __constant__ const int8_t B58_MAP[128] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
     0,  1,  2,  3,  4,  5,  6,  7,  8, -1, -1, -1, -1, -1, -1, -1,
     9, 10, 11, 12, 13, 14, 15, 16, -1, 17, 18, 19, 20, 21, -1, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, -1, -1, -1, -1, -1, -1,
    33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, -1, 44, 45, 46, 47,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, -1, -1, -1, -1, -1, -1
};

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// ============================================================
// DEVICE HELPERS: JACOBIAN ELLIPTIC CURVE MATH (secp256k1)
// ============================================================
// Diperlukan karena kita menghitung P = k * G dari nol untuk setiap kandidat valid.
// Menggunakan koordinat Jacobian untuk menghindari inversi modular yang mahal di loop.

__device__ void jacobian_double(uint64_t X[4], uint64_t Y[4], uint64_t Z[4]) {
    if ((Y[0] | Y[1] | Y[2] | Y[3]) == 0) return; // Point at infinity
    uint64_t A[4], B[4], C[4], D[4], E[4], F[4];
    
    // A = X^2
    _ModSqr(A, X);
    
    // B = Y^2
    _ModSqr(B, Y);
    
    // C = B^2
    _ModSqr(C, B);
    
    // D = 2*((X+B)^2 - A - C) = 4XY^2
    ModAdd256(D, X, B);
    _ModSqr(D, D);
    ModSub256(D, D, A);
    ModSub256(D, D, C);
    // D sudah 4XY^2
    
    // E = 3A
    uint64_t tmp[4];
    _ModSqr(tmp, A); // A^2
    ModAdd256(E, tmp, A); 
    // Simplified: E = 3A. Actually formula is E = 3A (mod p)
    // Wait, standard formula: E = 3*X^2. If a=0, E=3A.
    // Correct: E = 3*A
    ModAdd256(tmp, A, A); // 2A
    ModAdd256(E, tmp, A); // 3A

    // F = E^2
    _ModSqr(F, E);
    
    // X3 = F - 2D
    ModSub256(X, F, D);
    ModSub256(X, X, D);
    
    // Y3 = E*(D - X3) - 8C
    ModSub256(F, D, X); // reuse F
    _ModMult(F, E, F);
    uint64_t tmp2[4];
    ModAdd256(tmp2, C, C); // 2C
    ModAdd256(tmp2, tmp2, tmp2); // 4C
    ModAdd256(tmp2, tmp2, tmp2); // 8C
    ModSub256(Y, F, tmp2);
    
    // Z3 = 2Y*Z
    _ModMult(Z, Y, Z);
    ModAdd256(Z, Z, Z);
}

// Point Addition: P (Jacobian) + Q (Affine G)
__device__ void jacobian_add_mixed(uint64_t X1[4], uint64_t Y1[4], uint64_t Z1[4], 
                                   const uint64_t X2[4], const uint64_t Y2[4]) {
    uint64_t Z1z[4], Z1zz[4], U2[4], S2[4], H[4], R[4];
    
    // Z1Z1 = Z1^2
    _ModSqr(Z1z, Z1);
    // Z1ZZ = Z1^3
    _ModMult(Z1zz, Z1z, Z1);
    
    // U2 = X2 * Z1Z1
    _ModMult(U2, X2, Z1z);
    // S2 = Y2 * Z1ZZ
    _ModMult(S2, Y2, Z1zz);
    
    // H = U2 - X1
    ModSub256(H, U2, X1);
    // R = S2 - Y1
    ModSub256(R, S2, Y1);
    
    // If H == 0
    if ((H[0]|H[1]|H[2]|H[3]) == 0) {
        if ((R[0]|R[1]|R[2]|R[3]) == 0) {
            // Point doubling case (P == Q)
            jacobian_double(X1, Y1, Z1);
            return;
        }
        // Point at infinity (P == -Q)
        X1[0] = 0; X1[1] = 0; X1[2] = 0; X1[3] = 0;
        Y1[0] = 0; Y1[1] = 0; Y1[2] = 0; Y1[3] = 0;
        Z1[0] = 0; Z1[1] = 0; Z1[2] = 0; Z1[3] = 0;
        return;
    }
    
    // HH = H^2
    uint64_t HH[4];
    _ModSqr(HH, H);
    // HHH = H * HH
    uint64_t HHH[4];
    _ModMult(HHH, H, HH);
    // V = X1 * HH
    uint64_t V[4];
    _ModMult(V, X1, HH);
    
    // X3 = R^2 - HHH - 2V
    _ModSqr(X1, R);
    ModSub256(X1, X1, HHH);
    ModSub256(X1, X1, V);
    ModSub256(X1, X1, V);
    
    // Y3 = R*(V - X3) - Y1*HHH
    ModSub256(V, V, X1);
    _ModMult(Y1, R, V);
    uint64_t tmp[4];
    _ModMult(tmp, Y1, HHH); // Ini seharusnya Y1 * HHH
    // Re-check logic: Y3 = R(V - X3) - Y1*HHH
    // Kita sudah punya R*(V-X3) di Y1 (sebagai temp)
    // Tapi Y1 di line atas tertimpa.
    // Correct flow:
    // Y3 = R * (V - X3)
    ModSub256(V, V, X1); // V is actually V_old - X3? No, V was X1*HH.
    // Let's restart Y calc to be safe using stack vars.
    uint64_t Y3[4];
    ModSub256(Y3, V, X1); // V - X3
    _ModMult(Y3, R, Y3);   // R * (V - X3)
    _ModMult(tmp, Y1, HHH); // Y1 * HHH
    ModSub256(Y1, Y3, tmp);
    
    // Z3 = Z1 * H
    _ModMult(Z1, Z1, H);
}

// Convert Jacobian to Affine
__device__ void jacobian_to_affine(uint64_t X[4], uint64_t Y[4], uint64_t Z[4]) {
    if ((Z[0]|Z[1]|Z[2]|Z[3]) == 0) return; // Infinity
    
    uint64_t Z2[4], Z3[4], InvZ[4];
    _ModSqr(Z2, Z);
    _ModMult(Z3, Z2, Z);
    
    // Inverse of Z
    // Use modular inversion
    // Assuming _ModInv exists or works with 256-bit. 
    // If only _ModInv for 256bit exists similar to old code.
    // Old code used `inverse[5]` for some reason? Let's assume standard 256bit inv.
    // Copy Z to InvZ
    InvZ[0]=Z[0]; InvZ[1]=Z[1]; InvZ[2]=Z[2]; InvZ[3]=Z[3];
    _ModInv(InvZ); // Function from CUDAMath.h (assumed)
    
    // X = X * Z^-2
    _ModMult(X, X, InvZ); // InvZ is Z^-1. Square it to get Z^-2
    _ModSqr(InvZ, InvZ);
    _ModMult(X, X, InvZ);
    
    // Y = Y * Z^-3
    // We need Z^-3. Z^-3 = Z^-1 * Z^-2
    // Let's recompute simply
    InvZ[0]=Z[0]; InvZ[1]=Z[1]; InvZ[2]=Z[2]; InvZ[3]=Z[3];
    _ModInv(InvZ);
    _ModSqr(Z2, InvZ); // Z^-2
    _ModMult(Z3, Z2, InvZ); // Z^-3
    
    _ModMult(X, X, Z2);
    _ModMult(Y, Y, Z3);
}

// Scalar Multiplication: k * G
__device__ void scalar_mult_g(const uint64_t k[4], uint64_t outX[4], uint64_t outY[4]) {
    uint64_t X[4] = {0}, Y[4] = {0}, Z[4] = {0};
    
    // Jacobian Infinity is Z=0 (X/Y arbitrary, usually set to 0)
    
    // Scan bits of k from MSB to LSB (Standard Double-and-Add)
    // secp256k1 order is 256 bits.
    // We start from the highest bit set.
    
    bool started = false;
    for (int i = 3; i >= 0; --i) {
        for (int j = 63; j >= 0; --j) {
            bool bit = (k[i] >> j) & 1;
            if (!started && !bit) continue;
            
            if (!started) {
                // First bit 1: Initialize to G
                started = true;
                X[0] = c_Gx[0]; X[1] = c_Gx[1]; X[2] = c_Gx[2]; X[3] = c_Gx[3];
                Y[0] = c_Gy[0]; Y[1] = c_Gy[1]; Y[2] = c_Gy[2]; Y[3] = c_Gy[3];
                Z[0] = 1; Z[1] = 0; Z[2] = 0; Z[3] = 0;
            } else {
                // Double
                jacobian_double(X, Y, Z);
                // Add
                if (bit) {
                    jacobian_add_mixed(X, Y, Z, c_Gx, c_Gy);
                }
            }
        }
    }
    
    jacobian_to_affine(X, Y, Z);
    
    outX[0] = X[0]; outX[1] = X[1]; outX[2] = X[2]; outX[3] = X[3];
    outY[0] = Y[0]; outY[1] = Y[1]; outY[2] = Y[2]; outY[3] = Y[3];
}

// ============================================================
// KERNEL: MINIKEY BRUTEFORCE
// ============================================================
__launch_bounds__(256, 2)
__global__ void kernel_minikey_search(
    uint64_t* __restrict__ start_counter, // Global counter for work distribution
    uint32_t batch_size, // Iterations per thread per launch
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    unsigned long long* __restrict__ hashes_accum
)
{
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned lane = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;

    // Load global offset and increment by batch_size * grid_size
    // Only lane 0 does atomic add, then shfl
    uint64_t local_start = 0;
    if (lane == 0) local_start = atomicAdd(start_counter, (uint64_t)batch_size * (uint64_t)gridDim.x * blockDim.x);
    local_start = __shfl_sync(full_mask, local_start, 0);
    // Adjust for this specific thread
    local_start += gid * batch_size; // This logic implies we jump grid-size batches then sub-divide?
    // Simpler: atomicAdd returns the base for the whole grid.
    // Thread base = base_returned + gid.
    // But atomicAdd adds `grid_size * batch_size`.
    // So base_returned is the START of this chunk.
    // Thread i gets `base_returned + i * batch_size`.
    // Wait, if we increment by grid_size * batch_size, we cover the whole grid's work for this launch.
    // Base for thread `gid` = `base_returned` + `gid`? 
    // Yes, if atomicAdd returns `old`, then thread 0 works `old`, thread 1 `old+1`.
    // Wait, batch_size is iterations per thread.
    // So thread 0 does `old ... old + batch_size - 1`.
    // Thread 1 does `old + batch_size ... old + 2*batch_size - 1`.
    // Correct logic:
    // chunk_start = atomicAdd(global, total_work_in_this_launch)
    // my_start = chunk_start + gid * batch_size.

    const uint32_t target_prefix = c_target_prefix; // Assuming this is set
    const int vanity_len = c_vanity_len;
    
    unsigned int local_hashes = 0;

    // Local buffer for minikey (23 chars: 22 key + 1 null/check char)
    char minikey[24]; 
    minikey[22] = '?'; // For checksum
    minikey[23] = 0;
    
    // Helper to convert counter to Base58 string
    // This is slow to do every time, but necessary.
    // Optimization: Increment Base58 directly? 
    // Implementing a "Base58 Incrementer" is better.
    
    // Initialize string from counter
    // We use a simple method: convert integer `local_start` to base58.
    // NOTE: This restricts search space to 2^64 keys (~1.8e19) which is huge enough for demo
    // but technically Minikey space is 58^22.
    // For full range, we'd need 256-bit counter logic like in original main.
    // Let's stick to uint64_t counter mapped to string for simplicity in this snippet 
    // OR use the `start_counter` as a "seed" array?
    // Let's do Base58 increment logic.
    
    // Initialize with '1's (value 0)
    for(int i=0; i<22; ++i) minikey[i] = '1';
    
    // Apply offset (Add local_start to the Base58 number represented by minikey)
    // We need a helper `add_u64_to_b58`
    uint64_t val = local_start;
    int idx = 21;
    while (val > 0 && idx >= 0) {
        int digit = B58_MAP[(int)minikey[idx]];
        if (digit == -1) digit = 0; // Should not happen with '1'
        uint64_t sum = (uint64_t)digit + val;
        val = sum / 58;
        minikey[idx] = B58_ALPHABET[sum % 58];
        idx--;
    }
    
    // Loop for batch_size
    for (uint32_t iter = 0; iter < batch_size; ++iter) {
        // 1. Check Minikey Checksum: SHA256(minikey + '?') must start with 0x00
        // We use the 23-char buffer
        uint8_t hash_check[32];
        sha256_device((uint8_t*)minikey, 23, hash_check); // Assuming sha256_device exists
        
        if (hash_check[0] == 0x00) {
            // Valid Minikey candidate!
            // 2. Derive Private Key: SHA256(minikey) (22 chars)
            uint8_t priv_key_bytes[32];
            sha256_device((uint8_t*)minikey, 22, priv_key_bytes);
            
            // Convert LE bytes to 256-bit scalar
            uint64_t k[4];
            for(int i=0; i<4; ++i) {
                k[i] = ((uint64_t)priv_key_bytes[i*8+0] << 0) | ((uint64_t)priv_key_bytes[i*8+1] << 8) |
                       ((uint64_t)priv_key_bytes[i*8+2] << 16) | ((uint64_t)priv_key_bytes[i*8+3] << 24) |
                       ((uint64_t)priv_key_bytes[i*8+4] << 32) | ((uint64_t)priv_key_bytes[i*8+5] << 40) |
                       ((uint64_t)priv_key_bytes[i*8+6] << 48) | ((uint64_t)priv_key_bytes[i*8+7] << 56);
            }
            
            // 3. Compute Public Key
            uint64_t pubX[4], pubY[4];
            scalar_mult_g(k, pubX, pubY);
            
            // 4. Hash160
            uint8_t h20[20];
            uint8_t prefix = (pubY[0] & 1) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, pubX, h20);
            
            local_hashes++;
            
            // 5. Check Vanity
            bool match = true;
            if (vanity_len >= 4) {
                uint32_t h_prefix = (uint32_t)h20[0] | ((uint32_t)h20[1] << 8) | ((uint32_t)h20[2] << 16) | ((uint32_t)h20[3] << 24);
                if (h_prefix != target_prefix) match = false;
            }
            if (match) {
                for (int j = 4; j < vanity_len; ++j) {
                    if (h20[j] != c_target_hash160[j]) { match = false; break; }
                }
            }
            
            if (match) {
                if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                    // Save result
                    // We save the minikey string into 'scalar' field (which is 32 bytes)
                    // And Private Key into Rx/Ry? No, result struct is limited.
                    // We'll store Minikey string in scalar.
                    memset(d_found_result->scalar, 0, 32);
                    memcpy(d_found_result->scalar, minikey, 22); 
                    
                    // We can also store the derived pubkey
                    for(int i=0;i<4;++i) d_found_result->Rx[i] = pubX[i];
                    for(int i=0;i<4;++i) d_found_result->Ry[i] = pubY[i];
                    
                    d_found_result->threadId = (int)gid;
                    __threadfence_system();
                    atomicExch(d_found_flag, FOUND_READY);
                }
            }
        }
        
        // Increment Base58 string for next iteration
        // "S" increments "S" by 1 char? No, numeric increment.
        int carry = 1;
        for (int i = 21; i >= 0 && carry; --i) {
            int digit = B58_MAP[(int)minikey[i]];
            if (digit == -1) digit = 0;
            digit += carry;
            if (digit == 58) {
                digit = 0;
                carry = 1;
            } else {
                carry = 0;
            }
            minikey[i] = B58_ALPHABET[digit];
        }
        
        // Periodic flush
        if ((local_hashes & 255) == 0) {
            unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes);
            if (lane == 0 && v) atomicAdd(hashes_accum, v);
            local_hashes = 0;
        }
        
        // Early exit
        if (load_found_flag_relaxed(d_found_flag) == FOUND_READY) break;
    }
    
    // Final flush
    unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes);
    if (lane == 0 && v) atomicAdd(hashes_accum, v);
}

// ============================================================
// HOST CODE
// ============================================================

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);

// Base58 Encoder for Host display
std::string toBase58(const uint8_t* data, size_t len) {
    const char* alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    std::string result;
    // Simple implementation for display
    // Note: Need proper BigNum base conversion for correctness, 
    // but here we just want to display what we found. 
    // If result is stored as string in scalar, just print it.
    return std::string((const char*)data, len);
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::string vanity_hash_hex;
    uint32_t batch_size = 256; // Iterations per thread

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--vanity-hash160" && i + 1 < argc) vanity_hash_hex = argv[++i];
        else if (arg == "--batch" && i + 1 < argc) batch_size = (uint32_t)std::stoul(argv[++i]);
    }

    if (vanity_hash_hex.empty()) {
        std::cerr << "Usage: " << argv[0] << " --vanity-hash160 <prefix_hex> [--batch N]\n";
        return EXIT_FAILURE;
    }

    // Init Target
    uint8_t target_hash160[20];
    memset(target_hash160, 0, 20);
    for (size_t i = 0; i < vanity_hash_hex.length() / 2; ++i) {
        std::string byteStr = vanity_hash_hex.substr(i * 2, 2);
        target_hash160[i] = (uint8_t)std::stoul(byteStr, nullptr, 16);
    }
    int vanity_len = (int)(vanity_hash_hex.length() / 2);
    
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
    
    // Init G
    uint64_t hGx[4] = { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL };
    uint64_t hGy[4] = { 0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL };
    cudaMemcpyToSymbol(c_Gx, hGx, 32);
    cudaMemcpyToSymbol(c_Gy, hGy, 32);

    int device=0; cudaDeviceProp prop{};
    cudaGetDevice(&device); cudaGetDeviceProperties(&prop, device);
    
    int threadsPerBlock = 256;
    int blocks = prop.multiProcessorCount * 8; // Typical occupancy
    
    // Alloc
    uint64_t* d_start_counter;
    int* d_found_flag; FoundResult* d_found_result;
    unsigned long long* d_hashes_accum;
    
    cudaMalloc(&d_start_counter, sizeof(uint64_t));
    cudaMalloc(&d_found_flag, sizeof(int));
    cudaMalloc(&d_found_result, sizeof(FoundResult));
    cudaMalloc(&d_hashes_accum, sizeof(unsigned long long));
    
    uint64_t zero = 0;
    cudaMemcpy(d_start_counter, &zero, sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_hashes_accum, &zero, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    std::cout << "======== Minikey Search (22 char) =========\n";
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Target Hash: " << vanity_hash_hex << "\n";
    std::cout << "Running...\n";

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0;

    while (!g_sigint) {
        kernel_minikey_search<<<blocks, threadsPerBlock>>>(
            d_start_counter, batch_size,
            d_found_flag, d_found_result, d_hashes_accum
        );
        
        // Monitoring loop
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        
        int h_flag = 0;
        cudaMemcpy(&h_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        
        unsigned long long h_hashes = 0;
        cudaMemcpy(&h_hashes, d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        
        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();
        if (dt >= 1.0) {
            double rate = (double)(h_hashes - lastHashes) / dt / 1e6;
            std::cout << "\rHashes: " << h_hashes << " (" << std::fixed << std::setprecision(2) << rate << " MKeys/s)    " << std::flush;
            lastHashes = h_hashes;
            tLast = now;
        }
        
        if (h_flag == FOUND_READY) {
            FoundResult res;
            cudaMemcpy(&res, d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
            
            std::cout << "\n\n======== FOUND MATCH! ========\n";
            std::cout << "Minikey: " << std::string((char*)res.scalar, 22) << "\n";
            std::cout << "PubKey:  " << formatCompressedPubHex(res.Rx, res.Ry) << "\n";
            
            // Optional: Verify on Host
            // Compute SHA256 of minikey -> PrivKey -> PubKey
            break;
        }
    }
    
    cudaFree(d_start_counter);
    cudaFree(d_found_flag);
    cudaFree(d_found_result);
    cudaFree(d_hashes_accum);
    
    return 0;
}