CFLAGS += -g -O3 -Wall #-fPIC

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

all: driver_core.o

driver_core.o: driver_core.c  driver_core.h
	$(CC) $(CFLAGS) driver_core.c -c -o driver_core.o

clean:
	@rm -f *.o 
