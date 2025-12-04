# Build System Usage Guide

This project includes two build systems for convenience:

## 1. PowerShell Script (Recommended for Windows)

```powershell
# Show all available commands
.\build.ps1 -Target help

# Build all implementations
.\build.ps1 -Target all

# Run comprehensive comparative analysis
.\build.ps1 -Target comparative

# Run all implementations with custom array size
.\build.ps1 -Target run-all -ArraySize 50000000

# Check which executables are built
.\build.ps1 -Target check

# Clean everything
.\build.ps1 -Target clean
```

## 2. Makefile (For systems with make installed)

```bash
# Show all available commands
make help

# Build all implementations
make all

# Run comprehensive comparative analysis
make comparative

# Run with custom array size
make run-all ARRAY_SIZE=50000000

# Clean everything
make clean
```

## Quick Start

To run the complete comparative analysis:

```powershell
.\build.ps1 -Target comparative
```

This will:
1. Build all implementations (Serial, OpenMP, MPI, CUDA)
2. Run performance tests on each
3. Generate comparative charts
4. Create detailed analysis report

Results will be saved in `results/` directory:
- `comparative_analysis.png` - Side-by-side comparison charts
- `comparative_analysis_report.txt` - Comprehensive analysis

## Available Targets

- **all** - Build all implementations
- **serial** - Build serial implementation only
- **openmp** - Build OpenMP implementation only
- **mpi** - Build MPI implementation only
- **cuda** - Build CUDA implementation only
- **run-all** - Build and run all implementations
- **comparative** - Generate comprehensive comparative analysis
- **quick-compare** - Run quick performance comparison
- **clean** - Remove all executables and results
- **clean-build** - Remove only executables (keep results)
- **check** - Check which executables exist
