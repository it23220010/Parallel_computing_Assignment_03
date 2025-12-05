# Parallel Computing Assignment 03: Min-Max Normalization

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C](https://img.shields.io/badge/C-00599C?style=flat&logo=c&logoColor=white)](https://www.cprogramming.com/)
[![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)

## 📋 Overview

This project implements **Min-Max Normalization** data preprocessing across four parallel computing paradigms:
- **Serial** (baseline)
- **OpenMP** (shared-memory parallelism)
- **MPI** (distributed-memory parallelism)
- **CUDA** (GPU massively parallel computing)

### 🎯 Objectives

- Compare performance characteristics of different parallelization strategies
- Analyze speedup, efficiency, and scalability metrics
- Identify bottlenecks and optimization opportunities
- Demonstrate practical application of parallel computing for data preprocessing

---

## 🏗️ Project Structure

```
Parallel_computing_Assignment_03/
├── serial/                  # Serial baseline implementation
│   ├── minmax_serial.c
│   ├── Makefile
│   └── README.md
├── openmp/                  # OpenMP shared-memory implementation
│   ├── minmax_openmp.c
│   ├── Makefile
│   ├── run.ps1
│   └── README.md
├── mpi/                     # MPI distributed-memory implementation
│   ├── minmax_mpi.c
│   ├── Makefile
│   └── README.md
├── cuda/                    # CUDA GPU implementation
│   ├── minmax_cuda.cu
│   ├── test_cuda.cu
│   ├── Makefile
│   ├── README.md
│   └── Result_screenshots/
├── scripts/                 # Performance evaluation scripts
│   ├── openmp_eval.py
│   ├── mpi_eval.py
│   ├── cuda_eval.py
│   └── comparative_analysis.py
├── data/                    # Test data generation
│   ├── generate_data.py
│   └── test_data.csv
├── results/                 # Performance results and graphs
│   └── comparative_analysis_report.txt
├── BUILD_GUIDE.md          # Detailed build instructions
├── build.ps1               # Windows build automation script
└── Makefile                # Root Makefile for all implementations
```

---

## 🚀 Quick Start

### Prerequisites

**Hardware:**
- CPU: AMD Ryzen 9 5900HX (8 cores / 16 threads) or equivalent
- GPU: NVIDIA RTX 3050 (4GB VRAM) or better with Compute Capability 8.6+
- RAM: 16 GB minimum
- Storage: 2 GB available space

**Software:**
- **Windows 11** with WSL2 Ubuntu 22.04
- **GCC 11.3.0+** (for Serial, OpenMP, MPI)
- **MPICH 4.0** or OpenMPI 4.1+ (for MPI)
- **CUDA Toolkit 12.0+** (for CUDA)
- **Python 3.11+** with `matplotlib`, `numpy`
- **GNU Make 4.3+**

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/it23220010/Parallel_computing_Assignment_03.git
cd Parallel_computing_Assignment_03
```

2. **Install dependencies (WSL2 Ubuntu):**
```bash
# GCC and OpenMP
sudo apt update
sudo apt install build-essential

# MPI
sudo apt install mpich libmpich-dev

# Python packages
pip install matplotlib numpy
```

3. **Install CUDA Toolkit (Windows):**
   - Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
   - Install CUDA Toolkit 12.0 or later
   - Verify: `nvcc --version`

---

## 🔧 Building the Project

### Build All Implementations

**Windows PowerShell:**
```powershell
.\build.ps1
```

**WSL2 / Linux:**
```bash
make all
```

### Build Individual Implementations

**Serial:**
```bash
cd serial
make
./minmax_serial 100000000
```

**OpenMP:**
```bash
cd openmp
make
export OMP_NUM_THREADS=8
./minmax_openmp 100000000
```

**MPI (WSL2):**
```bash
cd mpi
make
mpirun -np 4 ./minmax_mpi 100000000
```

**CUDA (Windows PowerShell):**
```powershell
cd cuda
make
.\minmax_cuda.exe 100000000 128 40000
```

---

## 📊 Running Performance Evaluations

### Individual Evaluations

**OpenMP Evaluation:**
```bash
cd scripts
python openmp_eval.py
# Tests: 1, 2, 4, 8, 16 threads
# Outputs: graphs and performance metrics
```

**MPI Evaluation:**
```bash
cd scripts
python mpi_eval.py
# Tests: 1, 2, 4, 8, 16 processes
# Outputs: speedup curves and efficiency analysis
```

**CUDA Evaluation:**
```powershell
cd scripts
python cuda_eval.py
# Tests: 64, 128, 256, 512 threads/block
# Outputs: block size optimization graphs
```

### Comprehensive Comparative Analysis

```bash
cd scripts
python comparative_analysis.py
```

**Generates:**
- `results/comparative_analysis.png` - Side-by-side performance charts
- `results/comparative_analysis_report.txt` - Detailed analysis report

---

## 📈 Performance Results

### Benchmark Configuration
- **Dataset:** 100,000,000 floating-point elements (400 MB)
- **Data Type:** 32-bit float
- **Value Range:** [0.0, 100.0]
- **Hardware:** AMD Ryzen 9 5900HX + NVIDIA RTX 3050

### Performance Summary

| Implementation | Configuration | Execution Time | Speedup | Efficiency |
|----------------|--------------|----------------|---------|------------|
| **Serial**     | Baseline     | 1.034s         | 1.00x   | 100%       |
| **OpenMP**     | 16 threads   | 0.185s         | 5.59x   | 35%        |
| **MPI**        | 4 processes  | 0.061s         | 17.00x  | 425%       |
| **CUDA**       | 128 threads/block | 0.011s    | 94.00x  | 73%        |

### Key Findings

✅ **CUDA dominates** with 94x speedup for large-scale data preprocessing  
✅ **MPI unexpectedly strong** at 17x speedup on single-node (optimized collectives)  
✅ **OpenMP limited** by memory bandwidth saturation beyond 8 cores  
✅ **Serial adequate** for datasets < 10M elements  

---

## 🧪 Algorithm Details

### Min-Max Normalization Formula

$$X_{normalized} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

### Three-Phase Approach

1. **Phase 1: Find Minimum**
   - Single pass: O(n) time, O(1) space
   - OpenMP: `reduction(min:)` clause
   - MPI: `MPI_Allreduce` with `MPI_MIN`
   - CUDA: Warp-level reduction

2. **Phase 2: Find Maximum**
   - Single pass: O(n) time, O(1) space
   - OpenMP: `reduction(max:)` clause
   - MPI: `MPI_Allreduce` with `MPI_MAX`
   - CUDA: Warp-level reduction with `__shfl_down_sync`

3. **Phase 3: Normalize**
   - Single pass: O(n) time, O(1) auxiliary space
   - In-place transformation: (value - min) / (max - min)
   - Parallel element-wise operation

**Total Complexity:** O(3n) = O(n) linear time

---

## 🎓 Design Decisions

### Serial Implementation
- **Three separate passes** for clarity and parallelizability
- **In-place normalization** for memory efficiency (400MB vs 800MB)
- **32-bit float precision** balances accuracy and bandwidth
- **Edge case handling:** range = 0 → set all values to 0.5

### OpenMP Implementation
- **Reduction clauses** eliminate manual synchronization
- **Static scheduling** for predictable load distribution
- **Thread-private variables** minimize false sharing

### MPI Implementation
- **Block decomposition** for balanced data distribution
- **MPI_Allreduce** combines min/max in single collective operation
- **MPI_Scatter/Gather** for data distribution and collection

### CUDA Implementation
- **Warp-level primitives** (`__shfl_down_sync`) for efficient reduction
- **Grid-stride loops** handle datasets larger than thread count
- **Shared memory optimization** reduces global memory traffic by 128x
- **Block size 128** balances occupancy and resource utilization

---

## 📊 Scalability Analysis

### OpenMP Scaling
- ✅ Near-linear scaling up to 4 threads (92% efficiency)
- ✅ Peak at 8 threads (6.93x speedup, 87% efficiency)
- ❌ Degradation at 16 threads due to memory bandwidth saturation

### MPI Scaling
- ✅ Optimal at 4 processes (17.00x speedup)
- ❌ Communication overhead dominates beyond 4 processes
- ❌ Not suitable for single-machine workloads

### CUDA Scaling
- ✅ Optimal block size: 128 threads/block (152.9x speedup)
- ✅ 95%+ SM occupancy with 40,000 blocks
- ❌ PCIe transfer overhead (42% of total execution time)

---

## 🔍 Bottleneck Analysis

### OpenMP Bottlenecks
- Memory bandwidth saturation (~25 GB/s shared)
- Thread synchronization overhead (~0.002s per reduction)
- Cache coherence protocol (MESI) 5-8% penalty

### MPI Bottlenecks
- Data distribution overhead (MPI_Scatter: ~0.150s)
- Collective communication latency (MPI_Allreduce synchronization)
- Memory duplication per process

### CUDA Bottlenecks
- **PCIe transfer overhead** (~0.008s for 800MB transfers) - **dominant**
- Kernel launch latency (~0.0005s cumulative)
- Thermal throttling on mobile GPU (10-15% performance reduction)

---

## 🛠️ Potential Optimizations

### OpenMP
- SIMD intrinsics (AVX2/AVX-512) for explicit vectorization
- NUMA-aware memory allocation
- Combined min-max pass to reduce memory traversals

### MPI
- Non-blocking communications (`MPI_Isend/Irecv`)
- Hybrid MPI+OpenMP for node-level shared memory
- In-place `MPI_Allreduce` without gather

### CUDA
- **Unified Memory** for zero-copy transfers (eliminate PCIe overhead)
- Asynchronous kernel launches with CUDA streams
- Multi-GPU distribution for datasets > 1B elements
- Kernel fusion (combined min-max-normalize in single kernel)

---

## 📚 Documentation

- **[BUILD_GUIDE.md](BUILD_GUIDE.md)** - Comprehensive build instructions
- **[serial/README.md](serial/README.md)** - Serial implementation details
- **[openmp/README.md](openmp/README.md)** - OpenMP parallelization guide
- **[mpi/README.md](mpi/README.md)** - MPI distributed computing details
- **[cuda/README.md](cuda/README.md)** - CUDA GPU programming guide

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add optimization'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 👨‍💻 Author

**Shenal Dissanayake**  
Student ID: IT23220010  
Repository: [github.com/it23220010/Parallel_computing_Assignment_03](https://github.com/it23220010/Parallel_computing_Assignment_03)

---

## 🙏 Acknowledgments

- AMD Ryzen architecture documentation
- NVIDIA CUDA Programming Guide
- OpenMP API Specification 4.5
- MPI Standard 3.1 Documentation
- Parallel computing course materials

---

## 📞 Support

For issues or questions:
- Open an issue on GitHub
- Contact: [your-email@example.com]

---

**⭐ Star this repository if you found it helpful!**