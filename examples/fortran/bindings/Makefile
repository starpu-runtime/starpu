FC=gfortran
CFLAGS+= -g -Wall
FFLAGS+= -g -Wall

all: hello

hello.F: StarPU-fortran.h

hello: hello.F hello-c.o

clean:
	rm -f *.o *.mod hello
