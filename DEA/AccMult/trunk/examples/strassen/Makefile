RIBSCHED= ../../ribsched.a
CFLAGS+= -I../..

all: strassen.o strassen_kernels.o

strassen_kernels.o: strassen.h strassen_kernels.c
	$(CC) $(CFLAGS) strassen_kernels.c -c -o strassen_kernels.o

strassen.o: strassen.h strassen.c
	$(CC) $(CFLAGS) strassen.c -c -o strassen.o

clean:
	@rm -f *.o *.d
