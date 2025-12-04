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
	mpirun -np 4 ./$(TARGET) 1000000

# Performance testing with different process counts
test_performance: $(TARGET)
	@echo "Testing MPI performance with different process counts..."
	@echo "Array size: 1000000"
	@echo ""
	@for procs in 1 2 4 8 16; do \
		echo "Running with $$procs process(es):"; \
		mpirun -np $$procs ./$(TARGET) 1000000; \
		echo ""; \
	done

# Run with specific parameters
run: $(TARGET)
	@read -p "Enter number of processes: " procs; \
	read -p "Enter array size: " size; \
	mpirun -np $$procs ./$(TARGET) $$size

# Debug build
debug: $(SRC)
	$(CC) -g -O0 -Wall -Wextra -o $(TARGET)_debug $(SRC)

# Profile build
profile: $(SRC)
	$(CC) -pg -O2 -Wall -Wextra -o $(TARGET)_profile $(SRC)

.PHONY: all clean test test_performance run debug profile