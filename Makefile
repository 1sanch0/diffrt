
CC = g++

CFLAGS = -Wall -Wextra -O2 -mtune=native -march=native -I./src
LDFLAGS = -lm

SRCS = src/main.cc \
		   src/bsdf.cc \
		   src/objects.cc

OBJS = $(SRCS:.cc=.o)

TARGET = diffrt

.PHONY: all clean

all: $(TARGET)
	./$(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cc
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
