CC     = cc
CFLAGS = -O3 -ffast-math -march=native -funroll-loops -Wall -Wextra -std=c11 -pthread

# Dependencies: libm (expf, sqrtf) and libpthread.
# On macOS pthread is part of libSystem; on Linux it's a separate link.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LDFLAGS = -lm
else
  LDFLAGS = -lm -lpthread
endif

SRCS = main.c blas.c threadpool.c tensor_ops.c encoder.c decoder.c mel.c weights.c wav.c sentencepiece.c
OBJS = $(SRCS:.c=.o)
BIN  = parakeet

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# weights.c runs once at load time; -ffast-math can produce NaN from
# int4 dequant (compiler may reorder/eliminate 0*scale writes).
weights.o: weights.c parakeet.h
	$(CC) -O3 -march=native -funroll-loops -Wall -Wextra -std=c11 -pthread -c -o $@ $<

%.o: %.c parakeet.h blas.h threadpool.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(BIN)

.PHONY: all clean
