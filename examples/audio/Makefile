CFLAGS += -Wall -g3 -gdwarf-2 -O3 

LIBS+=$$(pkg-config --libs libstarpu) -lcufft
CFLAGS+=$$(pkg-config --cflags libstarpu)

LIBS+=$$(pkg-config --libs fftw3f)
CFLAGS+=$$(pkg-config --cflags fftw3f)

all: starpu_audio_processing

starpu_audio_processing.o: starpu_audio_processing.c

starpu_audio_processing: starpu_audio_processing.o
	$(CC) $(LDFLAGS) starpu_audio_processing.o -o starpu_audio_processing $(LIBS)

clean:
	rm -f *.o
	rm -f starpu_audio_processing
	rm -f input.dat input.raw
	rm -f output.dat output.wav output.raw
