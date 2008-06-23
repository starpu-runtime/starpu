# to use CUDA
CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
CUDAINSTALLDIR=/usr/local/cuda/

TARGET=incrementer heat tag_example 

# for compile time tuning 
DEFAULTSIZE=2048
DEFAULTGRAIN=128

DEFAULTNSAMPLE=10

ifdef NSAMPLE
	CFLAGS += -DNSAMPLE=$(NSAMPLE)
else
	CFLAGS += -DNSAMPLE=$(DEFAULTNSAMPLE)
endif

LDFLAGS+= -lm

ifdef OPENGL_RENDER
	CFLAGS+=-DOPENGL_RENDER
	LDFLAGS+=-lglut -lGL -lGLU 
endif

DEPENDENCIES=comp.o threads.o mult.o jobs.o mult_core.o matrix.o
EXTRADEP=

ifeq ($(MARCEL), 1)
	CC=$(shell pm2-config --cc)
	LDFLAGS += $(shell pm2-config --libs) 
	CFLAGS += -DUSE_MARCEL $(shell pm2-config --cflags)
else
	LDFLAGS += -lpthread
endif

ifdef DONTBIND
	CFLAGS += -DDONTBIND
endif

ifeq ($(CHECK), 1)
	CFLAGS += -DCHECK_OUTPUT
	CFLAGS += -DCHECK_RESULTS
endif

ifdef SIZE 
	CFLAGS += -DSIZE=$(SIZE)
else
	CFLAGS += -DSIZE=$(DEFAULTSIZE)
endif

ifeq ($(CUDA), 1)
	CFLAGS += -DUSE_CUDA 
	NVCCFLAGS +=  -DUSE_CUDA 
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	NVCCFLAGS += -I$(CUDAINSTALLDIR)/include 
	OBJDEPS += ../drivers/cuda/mult_cuda.o
	DEPENDENCIES += mult_cuda.o
	EXTRADEP += comp_cuda.cubin 
endif

ifeq ($(CUBLAS), 1)
	CFLAGS += -DUSE_CUBLAS
ifdef CUBLASEMU
	LDFLAGS += -lcublasemu
else
	LDFLAGS += -lcublas
endif

	LDFLAGS += -L$(CUDASDKDIR)/lib
	LDFLAGS += -L$(CUDAINSTALLDIR)/lib -lcudart -lcuda

	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	DEPENDENCIES += mult_cublas.o
	OBJDEPS += ../drivers/cublas/mult_cublas.o
endif

ifdef SPU
	OBJDEPS += ../drivers/spu/ppu/mult_spu_ppu.o
	OBJDEPS += ../drivers/spu/spu/spu_worker_program.o
endif

ifdef GORDON
	OBJDEPS += ../drivers/gordon/mult_gordon.o
	OBJDEPS += ../drivers/gordon/externals/scalp/util/util.o
	OBJDEPS += ../drivers/gordon/externals/scalp/cell/gordon/gordon.o 
	OBJDEPS += ../drivers/gordon/externals/scalp/cell/gordon/pputil.o 
	OBJDEPS += ../drivers/gordon/externals/scalp/cell/gordon/spuembed.o 
	CFLAGS += -I../drivers/gordon/externals/scalp/
endif

ifdef ATLAS
	CFLAGS += -DUSE_CPU_BLAS
        CFLAGS+= -I$(ATLASDIR)/include/
        LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
        LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a
	TARGET += dw_mult # dw_factolu
endif


ifdef CPUS
	CFLAGS += -DUSE_CPUS #-DDEBUG
	CFLAGS += -DNMAXCORES=$(CPUS)
	NVCCFLAGS += -DUSE_CPUS
	OBJDEPS += ../drivers/core/mult_core.o
endif

ifdef GRAIN
	CFLAGS += -DGRAIN=$(GRAIN)
	NVCCFLAGS += -DGRAIN=$(GRAIN)
else
	CFLAGS += -DGRAIN=$(DEFAULTGRAIN)
	NVCCFLAGS += -DGRAIN=$(DEFAULTGRAIN)
endif

NVCCFLAGS +=  -DUNIX -DO3
NVCCFLAGS += -I$(CUTILDIR)/inc/

LDFLAGS += -L../core/ -L../common/
CFLAGS += -I../

OBJDEPS += ../common/matrix.o ../common/threads.o ../common/timing.o 
OBJDEPS += ../core/jobs.o ../core/workers.o ../core/tags.o ../core/htable.o
OBJDEPS += ../common/mutex.o ../common/rwlock.o 

ifdef USE_FXT
OBJDEPS += ../common/fxt.o
LDFLAGS += -lfxt
endif

LDFLAGS += -L../datawizard/
CFLAGS += -I../datawizard/
OBJDEPS += ../datawizard/copy-driver.o ../datawizard/coherency.o 
OBJDEPS += ../datawizard/hierarchy.o ../datawizard/filters.o
OBJDEPS += ../datawizard/memalloc.o

all: $(TARGET)

incrementer.o: incrementer.c
	$(CC) $(CFLAGS) incrementer.c -c -o incrementer.o

incrementer: incrementer.o
	$(CC) incrementer.o -o incrementer $(OBJDEPS) $(LDFLAGS)

gpuirq.o: gpuirq.c
	$(CC) $(CFLAGS) gpuirq.c -c -o gpuirq.o

gpuirq: gpuirq.o
	$(CC) gpuirq.o -o gpuirq $(OBJDEPS) $(LDFLAGS)

multiplication2.o:
	$(CC) $(CFLAGS) multiplication2.c -c -o multiplication2.o

multiplication2: multiplication2.o
	$(CC) multiplication2.o -o multiplication2 $(OBJDEPS) $(LDFLAGS) 


multiplication.o:
	$(CC) $(CFLAGS) multiplication.c -c -o multiplication.o

dw_factolu_tag.o: dw_factolu_tag.c dw_factolu.h
	$(CC) $(CFLAGS) dw_factolu_tag.c -c -o dw_factolu_tag.o

dw_factolu.o: dw_factolu.c dw_factolu.h
	$(CC) $(CFLAGS) dw_factolu.c -c -o dw_factolu.o

dw_factolu: dw_factolu.o dw_factolu_tag.o
	$(CC) dw_factolu.o -o dw_factolu $(OBJDEPS) $(LDFLAGS)

dw_mult.o: dw_mult.c
	$(CC) $(CFLAGS) dw_mult.c -c -o dw_mult.o

tag_example.o: tag_example.c
	$(CC) $(CFLAGS) tag_example.c -c -o tag_example.o

dw_mult: dw_mult.o
	$(CC) dw_mult.o -o dw_mult $(OBJDEPS) $(LDFLAGS)

multiplication: multiplication.o
	$(CC) multiplication.o -o multiplication $(OBJDEPS) $(LDFLAGS) 

factoLU: factoLU.o
	$(CC) factoLU.o -o factoLU $(OBJDEPS) $(LDFLAGS) 

factoLU.o: factoLU.c
	$(CC) $(CFLAGS) factoLU.c -c -o factoLU.o

heat.o: heat.c
	$(CC) $(CFLAGS) heat.c -c -o heat.o

heat: dw_factolu.o dw_factolu_tag.o heat.o
	$(CC) dw_factolu.o dw_factolu_tag.o heat.o -o heat $(OBJDEPS) $(LDFLAGS)

tag_example: tag_example.o
	$(CC) tag_example.o -o tag_example $(OBJDEPS) $(LDFLAGS)



clean:
	@rm -f multiplication multiplication2 factoLU heat gpuirq incrementer dw_mult dw_factolu *.o
