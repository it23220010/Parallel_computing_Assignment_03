# MPI Min-Max Normalization Makefile

# Compiler and flags
CC = mpicc
CFLAGS = -O2 -Wall -Wextra
TARGET = minmax_mpi
SRC = minmax_mpi.c

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Test with default parameters
test: $(TARGET)
	mpirun -np 4 ./$(TARGET) 100000000

# Performance testing with different process counts
test_performance: $(TARGET)
	@echo "Testing MPI performance with different process counts..."
	@echo "Array size: 100000000"
	@echo ""
	@for procs in 1 2 4 8 16; do \
		echo "Running with $$procs process(es):"; \
		mpirun -np $$procs ./$(TARGET) 100000000; \
		echo ""; \
	done

# Test with 1, 2, 4, 8 processes
test_1_2_4_8: $(TARGET)
	@echo "==================================="
	@echo "Running with 1 process:"
	@echo "==================================="
	mpirun -np 1 ./$(TARGET) 100000000
	@echo ""
	@echo "==================================="
	@echo "Running with 2 processes:"
	@echo "==================================="
	mpirun -np 2 ./$(TARGET) 100000000
	@echo ""
	@echo "==================================="
	@echo "Running with 4 processes:"
	@echo "==================================="
	mpirun -np 4 ./$(TARGET) 100000000
	@echo ""
	@echo "==================================="
	@echo "Running with 8 processes:"
	@echo "==================================="
	mpirun -np 8 ./$(TARGET) 100000000

# Run with default parameters (4 processes, 100M elements)
run: all
	mpirun -np 4 ./$(TARGET) 100000000

# Debug build
debug: $(SRC)
	$(CC) -g -O0 -Wall -Wextra -o $(TARGET)_debug $(SRC)

# Profile build
profile: $(SRC)
	$(CC) -pg -O2 -Wall -Wextra -o $(TARGET)_profile $(SRC)

.PHONY: all clean test test_performance test_1_2_4_8 run debug profile