# CUDA Min-Max Normalization Makefile

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -O2 -arch=sm_50 --ptxas-options=-v
TARGET = minmax_cuda
SRC = minmax_cuda.cu

# Default CUDA architecture (adjust based on your GPU)
# Common architectures:
# - sm_50: Maxwell
# - sm_60: Pascal
# - sm_70: Volta
# - sm_75: Turing
# - sm_80: Ampere
# - sm_86: Ampere (laptops)
# - sm_89: Ada Lovelace

# Detect GPU architecture (optional)
# You can manually set this: ARCH = sm_75
ARCH = sm_50

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) -o $(TARGET) $(SRC)

# Debug build
debug: $(SRC)
	$(NVCC) -g -G -O0 -arch=$(ARCH) -o $(TARGET)_debug $(SRC)

# Profile build
profile: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) -lineinfo -o $(TARGET)_profile $(SRC)

# Performance build with specific optimizations
fast: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) --use_fast_math -o $(TARGET)_fast $(SRC)

# Clean build artifacts
clean:
	rm -f $(TARGET) $(TARGET)_debug $(TARGET)_profile $(TARGET)_fast *.o

# Test with default parameters
test: $(TARGET)
	./$(TARGET) 100000000 128 40000

# Quick performance test with different block sizes
test_blocks: $(TARGET)
	@echo "Testing with different block sizes (array size: 1000000, blocks: 256)"
	@echo "================================================================="
	@for block in 32 64 128 256 512 1024; do \
		if [ $$block -le 1024 ]; then \
			echo -n "Block size $$block: "; \
			./$(TARGET) 1000000 $$block 256 2>/dev/null | grep "Kernel execution time" | awk '{print $$4}'; \
		fi; \
	done

# Quick performance test with different number of blocks
test_grids: $(TARGET)
	@echo "Testing with different number of blocks (array size: 1000000, block size: 256)"
	@echo "================================================================="
	@for blocks in 32 64 128 256 512; do \
		echo -n "Blocks $$blocks: "; \
		./$(TARGET) 1000000 256 $$blocks 2>/dev/null | grep "Kernel execution time" | awk '{print $$4}'; \
	done

# Comprehensive test
test_comprehensive: $(TARGET)
	@echo "Running comprehensive CUDA performance tests..."
	@echo "Array size: 1000000"
	@echo ""
	@echo "Block Size | Num Blocks | Time (ms)"
	@echo "-----------|------------|-----------"
	@for block in 128 256 512; do \
		for blocks in 64 128 256 512; do \
			if [ $$block -le 1024 ]; then \
				time=$$(./$(TARGET) 1000000 $$block $$blocks 2>/dev/null | grep "Kernel execution time" | awk '{print $$4}'); \
				printf "%10d | %10d | %9s\n" $$block $$blocks $$time; \
			fi; \
		done; \
	done

# Test with block sizes 64, 128, 256, 512
test_64_128_256_512: $(TARGET)
	@echo "==================================="
	@echo "Running with block size 64:"
	@echo "==================================="
	./$(TARGET) 100000000 64 40000
	@echo ""
	@echo "==================================="
	@echo "Running with block size 128:"
	@echo "==================================="
	./$(TARGET) 100000000 128 40000
	@echo ""
	@echo "==================================="
	@echo "Running with block size 256:"
	@echo "==================================="
	./$(TARGET) 100000000 256 40000
	@echo ""
	@echo "==================================="
	@echo "Running with block size 512:"
	@echo "==================================="
	./$(TARGET) 100000000 512 40000

# Run with default parameters (100M elements, 128 threads/block, 40000 blocks)
run: all
	./$(TARGET) 100000000 128 40000

# Display GPU information
gpu_info:
	nvidia-smi
	nvcc --version

# Compile and show PTX and register usage
verbose: $(SRC)
	$(NVCC) $(NVCC_FLAGS) -arch=$(ARCH) --ptxas-options=-v -o $(TARGET) $(SRC) 2>&1 | grep "ptxas"

# Check CUDA installation
check_cuda:
	@echo "Checking CUDA installation..."
	@which nvcc
	@nvcc --version
	@echo ""
	@echo "Checking GPU..."
	@nvidia-smi --query-gpu=name,compute_capability --format=csv

.PHONY: all clean test test_blocks test_grids test_comprehensive test_64_128_256_512 run debug profile fast gpu_info verbose check_cuda