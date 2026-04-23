Here is the modified `README.md`. It is designed to be sleek, modern, and minimal with a distinct "hacker" aesthetic.

***

# 🦖 RAPTOR CUDA: GPU Key Hunter

**RAPTOR CUDA** is a high-performance GPU cracker designed to hunt down Satoshi Puzzles with extreme prejudice. Built on CUDA architecture, it combines warp-level parallelism with optimized batch EC operations to deliver devastating brute-force speeds.

Based on the `secp256k1` math from [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch) and [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul), but stripped down and rebuilt for raw speed and simplicity.

> ⚠️ **Intel**: Achieved **6.5 Gkeys/s** on RTX 4090. **8.6 Gkeys/s** on RTX 5090.  
> **Pro Tip**: Prevent throttling by tuning `--slices`. Optimal config for 4090: `--grid 128,128 --slices 16`.

---

## 🛡️ Zero Flaws Protocol

The key skipping anomaly has been neutralized. Integrity verified via `proof.py`. Zero keys left behind.

**Execution:**
```bash
python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512
```

**Result:**
```
================ Summary by blocks ================
Range start A (start+2k)           : total= 128  success= 128  fail=   0
...
Full mod 512 residue coverage      : total= 256  success= 256  fail=   0
Random Q1-Q4                       : total=  80  success=  80  fail=   0

Done. Successes=848 Failures=0
```

---

## ⚙️ The Arsenal

- **Pure GPU Power**: Massive parallel execution on NVIDIA GPUs.
- **Low Profile**: Extremely low VRAM footprint. Optimized for rented rigs.
- **Warp Optimized**: Efficient batch modular inversion.
- **Simple & Clean**: Easy to compile, easy to read, easy to run.

---

## 🚀 CLI Interface

Control the beast with precision.

| Flag | Description |
| :--- | :--- |
| `--range` | Search range (must be power of 2). |
| `--address` | Target P2PKH address. |
| `--target-hash160` | Target hash160 (alternative to address). |
| `--grid` | Kernel geometry: `<PointsPerThread>,<ThreadsPerBatch>`. |
| `--slices` | Batch count per kernel launch. |

---

## 📊 Speed Benchmarks

Community-sourced performance metrics.

| GPU | Configuration | Speed | Status |
| :--- | :--- | :--- | :--- |
| **RTX 5090** | `128,256` | **8408 Mkeys/s** | Verified |
| **RTX 4090** | `128,1024` | **6214 Mkeys/s** | Verified |
| **RTX 4070 Ti S** | `512,1024` | **3170 Mkeys/s** | Verified |
| **RTX 4060** | `512,512` | **1238 Mkeys/s** | Verified |

---

## 💻 Live Execution

**Target: RTX 4090**
```bash
./RAPTOR_CUDA --range 200000000000:3fffffffffff --address 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP --grid 128,128 --slices 16
```
```text
======== PrePhase: System Scan =======================
Device               : NVIDIA GeForce RTX 4090 (compute 8.9)
Memory utilization   : 4.8% (1.14 GB / 23.6 GB)
Total threads        : 4194304

======== Phase-1: BruteForce (Sliced) =================
Time: 393.7 s | Speed: 6127.4 Mkeys/s | Count: 2421341587872

======== TARGET ACQUIRED =============================
Private Key   : 00000000000000000000000000000000000000000000000000002EC18388D544
Public Key    : 03FD5487722D2576CB6D7081426B66A3E2986C1CE8358D479063FB5F2BB6DD5849
```

**Target: RTX 5090**
```bash
./RAPTOR_CUDA --range 200000000000:3fffffffffff --address 1F3JRMWudBaj48EhwcHDdpeuy2jwACNxjP --grid 128,256
```
```text
======== PrePhase: System Scan =======================
Device               : NVIDIA GeForce RTX 5090 (compute 12.0)
Memory utilization   : 1.7% (557.3 MB / 31.4 GB)

======== Phase-1: BruteForce =========================
Time: 7.0 s | Speed: 8408.0 Mkeys/s | Count: 58545467200
```

---

## 🛠️ Installation

Deploy the binary. Requires CUDA Toolkit.

```bash
# Dependencies
apt update && apt install -y build-essential gcc make cuda-toolkit git

# Build
git clone https://github.com/Dookoo2/CUDACyclone.git
cd CUDACyclone
make
```

---

## 🚧 Changelog

*   **V1.3**: Core rewrite. Key skipping eliminated.
*   **V1.2**: Kernel optimization.
*   **V1.1**: Constant memory fix for thermal throttling.
*   **V1.0**: Genesis.

---

## ⚡ Support

**BTC**: ``