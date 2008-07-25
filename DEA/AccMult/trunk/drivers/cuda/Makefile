ifeq ($(CUDA), 1)
	DEPENDENCIES += driver_cuda.o
endif

CFLAGS += -I../../

all: driver_cuda.o

%.o: %.c %.h
	$(CC) $(CFLAGS) $< -c -o $@

clean:
	@rm -f *.o
