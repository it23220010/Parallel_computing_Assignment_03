CC = gcc
CFLAGS = -O2 -Wall -Wextra
TARGET = minmax_serial
SRC = minmax_serial.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o

run: $(TARGET)
	./$(TARGET) 1000000

.PHONY: all clean run