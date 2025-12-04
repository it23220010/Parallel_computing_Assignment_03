# Main Makefile for Parallel Computing Assignment 03
# Min-Max Normalization: Serial, OpenMP, MPI, and CUDA implementations

# Compiler settings
GCC = gcc
NVCC = nvcc
MPICC = mpicc
PYTHON = python

# Directories
SERIAL_DIR = serial
OPENMP_DIR = openmp
MPI_DIR = mpi
CUDA_DIR = cuda
SCRIPTS_DIR = scripts
RESULTS_DIR = results

# Executable names
SERIAL_EXE = $(SERIAL_DIR)/minmax_serial.exe
OPENMP_EXE = $(OPENMP_DIR)/minmax_openmp.exe
MPI_EXE = $(MPI_DIR)/minmax_mpi.exe
CUDA_EXE = $(CUDA_DIR)/minmax_cuda.exe

# Compiler flags
CFLAGS = -O3 -Wall
OMPFLAGS = -fopenmp
MPIFLAGS = -I"C:\Program Files (x86)\Microsoft SDKs\MPI\Include" -L"C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64" -lmsmpi
CUDAFLAGS = -O3 -arch=sm_75

# Default array size for testing (100 million elements = 10^8)
ARRAY_SIZE = 100000000

.PHONY: all serial openmp mpi cuda clean run-all test-all comparative help

# Default target
all: serial openmp mpi cuda
	@echo.
	@echo ========================================
	@echo All implementations compiled successfully!
	@echo ========================================
	@echo.

# Build Serial implementation
serial:
	@echo Building Serial implementation...
	@cd $(SERIAL_DIR) && $(GCC) $(CFLAGS) -o minmax_serial.exe minmax_serial.c -lm
	@echo   [OK] Serial executable created

# Build OpenMP implementation
openmp:
	@echo Building OpenMP implementation...
	@cd $(OPENMP_DIR) && $(GCC) $(CFLAGS) $(OMPFLAGS) -o minmax_openmp.exe minmax_openmp.c -lm
	@echo   [OK] OpenMP executable created

# Build MPI implementation
mpi:
	@echo Building MPI implementation...
	@cd $(MPI_DIR) && $(GCC) $(CFLAGS) -o minmax_mpi.exe minmax_mpi.c $(MPIFLAGS) -lm
	@echo   [OK] MPI executable created

# Build CUDA implementation
cuda:
	@echo Building CUDA implementation...
	@cd $(CUDA_DIR) && cmd /c "\"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && $(NVCC) $(CUDAFLAGS) -o minmax_cuda.exe minmax_cuda.cu"
	@echo   [OK] CUDA executable created

# Run all implementations with default array size
run-all: all
	@echo.
	@echo ========================================
	@echo Running all implementations
	@echo Array size: $(ARRAY_SIZE)
	@echo ========================================
	@echo.
	@echo [Serial Implementation]
	@cd $(SERIAL_DIR) && minmax_serial.exe $(ARRAY_SIZE)
	@echo.
	@echo [OpenMP Implementation - 8 threads]
	@cd $(OPENMP_DIR) && minmax_openmp.exe $(ARRAY_SIZE) 8
	@echo.
	@echo [MPI Implementation - 4 processes]
	@cd $(MPI_DIR) && mpiexec -n 4 minmax_mpi.exe $(ARRAY_SIZE)
	@echo.
	@echo [CUDA Implementation - 128 blocks]
	@cd $(CUDA_DIR) && minmax_cuda.exe $(ARRAY_SIZE) 128 781250
	@echo.

# Run performance evaluation scripts
test-serial: serial
	@echo Running Serial performance test...
	@cd $(SERIAL_DIR) && minmax_serial.exe $(ARRAY_SIZE)

test-openmp: openmp
	@echo Running OpenMP performance evaluation...
	@cd $(SCRIPTS_DIR) && $(PYTHON) openmp_eval.py

test-mpi: mpi
	@echo Running MPI performance evaluation...
	@cd $(SCRIPTS_DIR) && $(PYTHON) mpi_eval.py

test-cuda: cuda
	@echo Running CUDA performance evaluation...
	@cd $(SCRIPTS_DIR) && $(PYTHON) cuda_eval.py

test-all: all
	@echo.
	@echo ========================================
	@echo Running all performance evaluations
	@echo ========================================
	@echo.
	@$(MAKE) test-openmp
	@echo.
	@$(MAKE) test-mpi
	@echo.
	@$(MAKE) test-cuda
	@echo.

# Run comprehensive comparative analysis
comparative: all
	@echo.
	@echo ========================================
	@echo Running Comprehensive Comparative Analysis
	@echo ========================================
	@echo This will generate:
	@echo   - Comparative charts (execution time and speedup)
	@echo   - Detailed analysis report
	@echo.
	@cd $(SCRIPTS_DIR) && $(PYTHON) comparative_analysis.py $(ARRAY_SIZE)
	@echo.
	@echo Results saved in $(RESULTS_DIR)/ directory
	@echo.

# Quick comparison (simplified)
quick-compare: all
	@echo Running quick comparison...
	@cd $(SCRIPTS_DIR) && $(PYTHON) quick_compare.py

# Clean all executables and results
clean:
	@echo Cleaning all build artifacts and results...
	@if exist $(SERIAL_EXE) del $(SERIAL_EXE)
	@if exist $(OPENMP_EXE) del $(OPENMP_EXE)
	@if exist $(MPI_EXE) del $(MPI_EXE)
	@if exist $(CUDA_EXE) del $(CUDA_EXE)
	@if exist $(RESULTS_DIR)\*.png del $(RESULTS_DIR)\*.png
	@if exist $(RESULTS_DIR)\*.txt del $(RESULTS_DIR)\*.txt
	@if exist $(RESULTS_DIR)\*.csv del $(RESULTS_DIR)\*.csv
	@echo Clean complete!

# Clean only executables (keep results)
clean-build:
	@echo Cleaning executables only...
	@if exist $(SERIAL_EXE) del $(SERIAL_EXE)
	@if exist $(OPENMP_EXE) del $(OPENMP_EXE)
	@if exist $(MPI_EXE) del $(MPI_EXE)
	@if exist $(CUDA_EXE) del $(CUDA_EXE)
	@echo Executables cleaned!

# Rebuild everything from scratch
rebuild: clean all

# Check if all executables exist
check:
	@echo Checking compiled executables...
	@if exist $(SERIAL_EXE) (echo   [OK] Serial) else (echo   [MISSING] Serial)
	@if exist $(OPENMP_EXE) (echo   [OK] OpenMP) else (echo   [MISSING] OpenMP)
	@if exist $(MPI_EXE) (echo   [OK] MPI) else (echo   [MISSING] MPI)
	@if exist $(CUDA_EXE) (echo   [OK] CUDA) else (echo   [MISSING] CUDA)

# Help target
help:
	@echo.
	@echo ========================================
	@echo Parallel Computing Assignment 03 - Makefile
	@echo ========================================
	@echo.
	@echo Available targets:
	@echo.
	@echo   make all              - Build all implementations (default)
	@echo   make serial           - Build serial implementation only
	@echo   make openmp           - Build OpenMP implementation only
	@echo   make mpi              - Build MPI implementation only
	@echo   make cuda             - Build CUDA implementation only
	@echo.
	@echo   make run-all          - Build and run all implementations
	@echo   make test-all         - Run all performance evaluations
	@echo   make comparative      - Generate comprehensive comparative analysis
	@echo   make quick-compare    - Run quick performance comparison
	@echo.
	@echo   make clean            - Remove all executables and results
	@echo   make clean-build      - Remove only executables (keep results)
	@echo   make rebuild          - Clean and rebuild everything
	@echo   make check            - Check which executables exist
	@echo   make help             - Show this help message
	@echo.
	@echo Example usage:
	@echo   make all                          - Build everything
	@echo   make comparative                  - Run full comparative analysis
	@echo   make run-all ARRAY_SIZE=10000000  - Run with custom array size
	@echo.
