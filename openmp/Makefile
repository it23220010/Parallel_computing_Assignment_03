CC = gcc
CFLAGS = -O2 -Wall -Wextra -fopenmp
TARGET = minmax_openmp
SRC = minmax_openmp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o

run: all
	./$(TARGET) 100000000 8

test_threads: $(TARGET)
	for threads in 1 2 4 8 16; do \
		echo "Testing with $$threads threads:"; \
		./$(TARGET) 100000000 $$threads; \
		echo ""; \
	done

test_1_2_4_8: $(TARGET)
	@echo "==================================="
	@echo "Running with 1 thread:"
	@echo "==================================="
	./$(TARGET) 100000000 1
	@echo ""
	@echo "==================================="
	@echo "Running with 2 threads:"
	@echo "==================================="
	./$(TARGET) 100000000 2
	@echo ""
	@echo "==================================="
	@echo "Running with 4 threads:"
	@echo "==================================="
	./$(TARGET) 100000000 4
	@echo ""
	@echo "==================================="
	@echo "Running with 8 threads:"
	@echo "==================================="
	./$(TARGET) 100000000 8

.PHONY: all clean run test_threads test_1_2_4_8