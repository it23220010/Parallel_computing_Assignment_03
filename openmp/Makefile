CC = gcc
CFLAGS = -O2 -Wall -Wextra -fopenmp
TARGET = minmax_openmp
SRC = minmax_openmp.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o

run: $(TARGET)
	./$(TARGET) 1000000 4

test_threads: $(TARGET)
	for threads in 1 2 4 8 16; do \
		echo "Testing with $$threads threads:"; \
		./$(TARGET) 1000000 $$threads; \
		echo ""; \
	done

.PHONY: all clean run test_threads