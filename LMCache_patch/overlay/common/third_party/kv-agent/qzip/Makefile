# Makefile

# Compiler
CC = gcc

#path to install the library
INSTALL_DIR=/usr/lib

# Compiler flags
CFLAGS = -fPIC -Wall

# Linker flags
LDFLAGS = -shared
LDFLAGS += -lqatzip -lqat_s

# Target shared library
TARGET = libqzip.so

# Source files
SRCS = qzip.c

# Object files
OBJS = $(SRCS:.c=.o)

# Default target
all: $(TARGET)

# Rule to create the shared library
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Rule to compile source files into object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up
clean:
	rm -f $(OBJS) $(TARGET)

install: $(TARGET)
	cp $(TARGET) $(INSTALL_DIR)

.PHONY: all clean install
