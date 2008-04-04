CFLAGS += -g -O3 -Wall #-fPIC

ATLASDIR=/home/gonnet/DEA/BLAS/ATLAS/ATLAS/
BLASARCH=Linux_UNKNOWNSSE2

CFLAGS+= -I$(ATLASDIR)/include/
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a

LDFLAGS+= -lm

ifdef DONTBIND
	CFLAGS += -DDONTBIND
endif

ifdef ATLAS
	CFLAGS += -DUSE_CPU_BLAS
endif

ifdef CPUS
	CFLAGS += -DUSE_CPUS #-DDEBUG
	CFLAGS += -DNMAXCORES=$(CPUS)
endif

CFLAGS += -I../../

all: mult_core.o

mult_core.o: mult_core.c  mult_core.h
	$(CC) $(CFLAGS) mult_core.c -c -o mult_core.o

clean:
	@rm -f *.o 
