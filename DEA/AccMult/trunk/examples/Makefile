TARGET=tag_example 
TARGET+= incrementer
TARGET+= heat 
TARGET+= dw_spmv

ifdef OPENGL_RENDER
	CFLAGS+=-DOPENGL_RENDER
	LDFLAGS+=-lglut -lGL -lGLU 
endif

DEPENDENCIES=comp.o threads.o mult.o jobs.o driver_core.o
EXTRADEP=

ifeq ($(CHECK), 1)
	CFLAGS += -DCHECK_OUTPUT
	CFLAGS += -DCHECK_RESULTS
endif

ifeq ($(CUDA), 1)
	OBJDEPS += ../drivers/cuda/driver_cuda.o
	DEPENDENCIES += driver_cuda.o
	EXTRADEP += cuda/incrementer_cuda.cubin 
	EXTRADEP += cuda/spmv_cuda.cubin 
endif

ifdef SPU
	OBJDEPS += ../drivers/spu/ppu/driver_spu_ppu.o
	OBJDEPS += ../drivers/spu/spu/spu_worker_program.o
endif

ifdef GORDON
	OBJDEPS += ../drivers/gordon/driver_gordon.o
	OBJDEPS += ../drivers/gordon/externals/scalp/util/util.o
	OBJDEPS += ../drivers/gordon/externals/scalp/cell/gordon/gordon.o 
	OBJDEPS += ../drivers/gordon/externals/scalp/cell/gordon/pputil.o 
	OBJDEPS += ../drivers/gordon/externals/scalp/cell/gordon/spuembed.o 
	CFLAGS += -I../drivers/gordon/externals/scalp/
endif

ifdef ATLAS
	TARGET += dw_mult # dw_factolu
endif


ifdef CPUS
	OBJDEPS += ../drivers/core/driver_core.o
endif

NVCCFLAGS +=  -DUNIX -DO3
NVCCFLAGS += -I$(CUTILDIR)/inc/

LDFLAGS += -L../core/ -L../common/
CFLAGS += -I../

OBJDEPS += ../common/threads.o ../common/timing.o 
OBJDEPS += ../core/jobs.o ../core/workers.o ../core/tags.o ../core/htable.o
OBJDEPS += ../common/mutex.o ../common/rwlock.o 

ifdef USE_FXT
OBJDEPS += ../common/fxt.o
LDFLAGS += -lfxt
endif

LDFLAGS += -L../datawizard/
CFLAGS += -I../datawizard/
OBJDEPS += ../datawizard/copy-driver.o ../datawizard/coherency.o 
OBJDEPS += ../datawizard/hierarchy.o ../datawizard/interfaces/blas_filters.o
OBJDEPS += ../datawizard/memalloc.o  ../datawizard/interfaces/blas_interface.o
OBJDEPS += ../datawizard/interfaces/csr_interface.o


all: $(TARGET)

incrementer.o: incrementer.c
	$(CC) $(CFLAGS) incrementer.c -c -o incrementer.o

incrementer: $(EXTRADEP) incrementer.o
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

dw_spmv: $(EXTRADEP) dw_spmv.o
	$(CC) dw_spmv.o -o dw_spmv $(OBJDEPS) $(LDFLAGS)

dw_mult: dw_mult.o
	$(CC) dw_mult.o -o dw_mult $(OBJDEPS) $(LDFLAGS)

multiplication: multiplication.o
	$(CC) multiplication.o -o multiplication $(OBJDEPS) $(LDFLAGS) 

heat.o: heat.c
	$(CC) $(CFLAGS) heat.c -c -o heat.o

heat: dw_factolu.o dw_factolu_tag.o heat.o
	$(CC) dw_factolu.o dw_factolu_tag.o heat.o -o heat $(OBJDEPS) $(LDFLAGS)

tag_example: tag_example.o
	$(CC) tag_example.o -o tag_example $(OBJDEPS) $(LDFLAGS)

cuda/incrementer_cuda.cubin: cuda/incrementer_cuda.cu cuda/incrementer_cuda.h
	$(NVCC) -cubin cuda/incrementer_cuda.cu -o cuda/incrementer_cuda.cubin -m32 --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) 

cuda/spmv_cuda.cubin: cuda/spmv.cu
	$(NVCC) -cubin cuda/spmv.cu -o cuda/spmv_cuda.cubin -m32 --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) 

clean:
	@rm -f tag_example multiplication multiplication2 heat gpuirq dw_spmv incrementer dw_mult dw_factolu *.o
	@rm -f cuda/incrementer_cuda.cubin cuda/spmv_cuda.cubin
