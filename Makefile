CFLAGS+= -g -Wall
LDFLAGS+= -lm
#LDFLAGS+=/home/gonnet/Labri/DEA/BLAS/NetLIB/src/ATLAS/lib/Linux_P4SSE2/libcblas.a /home/gonnet/Labri/DEA/BLAS/NetLIB/src/ATLAS/lib/Linux_P4SSE2/libatlas.a

ATLASDIR=/home/gonnet/DEA/BLAS/ATLAS/ATLAS/
BLASARCH=Linux_UNKNOWNSSE2

CFLAGS+= -I$(ATLASDIR)/include/
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a


all: mesh

mesh.o: mesh.c
	gcc $(CFLAGS) mesh.c -c -o mesh.o

mesh: mesh.o
	gcc mesh.o -o mesh $(LDFLAGS)

clean:
	rm -f *.o mesh
