CFLAGS += -Wall -g3 -gdwarf-2 -O3 

LIBS+=$$(pkg-config --libs libstarpu) -lcufft
CFLAGS+=$$(pkg-config --cflags libstarpu)

LIBS+=$$(pkg-config --libs fftw3f)
CFLAGS+=$$(pkg-config --cflags fftw3f)

all: starpu-audio-processing

starpu-audio-processing.o: starpu-audio-processing.c

starpu-audio-processing: starpu-audio-processing.o
	$(CC) $(LDFLAGS) starpu-audio-processing.o -o starpu-audio-processing $(LIBS)

clean:
	rm -f *.o
	rm -f starpu-audio-processing
	rm -f input.dat input.raw
	rm -f output.dat output.wav output.raw
