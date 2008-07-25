TARGET=tag_example 
TARGET+= incrementer
#TARGET+= dw_sparse_cg

OBJDEPS=../ribsched.a

ifdef OPENGL_RENDER
	CFLAGS+=-DOPENGL_RENDER
	LDFLAGS+=-lglut -lGL -lGLU 
endif

EXTRADEP=

ifeq ($(CHECK), 1)
	CFLAGS += -DCHECK_OUTPUT
	CFLAGS += -DCHECK_RESULTS
endif

ifeq ($(CUDA), 1)
	EXTRADEP += cuda/incrementer_cuda.cubin 
	EXTRADEP += cuda/spmv_cuda.cubin 
endif

ifdef ATLAS
	TARGET += dw_mult # dw_factolu
	TARGET+= dw_spmv
	TARGET+= heat 
endif

CFLAGS += -I../

ifdef USE_FXT
	LDFLAGS += -lfxt
endif

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

dw_spmv.o: dw_spmv.c dw_spmv.h
	$(CC) $(CFLAGS) dw_spmv.c -c -o dw_spmv.o

dw_spmv: $(EXTRADEP) dw_spmv.o
	$(CC) dw_spmv.o -o dw_spmv $(OBJDEPS) $(LDFLAGS)

dw_sparse_cg_kernels.o: dw_sparse_cg_kernels.c dw_sparse_cg.h
	$(CC) $(CFLAGS) dw_sparse_cg_kernels.c -c -o dw_sparse_cg_kernels.o

dw_sparse_cg.o: dw_sparse_cg.c dw_sparse_cg.h
	$(CC) $(CFLAGS) dw_sparse_cg.c -c -o dw_sparse_cg.o

#dw_sparse_cg: $(EXTRADEP) dw_sparse_cg.o dw_sparse_cg_kernels.o
#	$(CC) dw_sparse_cg.o dw_sparse_cg_kernels.o -o dw_sparse_cg $(OBJDEPS) $(LDFLAGS)



dw_mult: dw_mult.o
	$(CC) dw_mult.o -o dw_mult $(OBJDEPS) $(LDFLAGS)

multiplication: multiplication.o
	$(CC) multiplication.o -o multiplication $(OBJDEPS) $(LDFLAGS) 

heat_display.o: heat_display.c heat.h
	$(CC) $(CFLAGS) heat_display.c -c -o heat_display.o

heat.o: heat.c
	$(CC) $(CFLAGS) heat.c -c -o heat.o

heat: dw_sparse_cg.o dw_sparse_cg_kernels.o dw_factolu.o dw_factolu_tag.o heat_display.o heat.o
	$(CC) dw_sparse_cg.o dw_sparse_cg_kernels.o dw_factolu.o dw_factolu_tag.o heat_display.o heat.o -o heat $(OBJDEPS) $(LDFLAGS)

tag_example: tag_example.o
	$(CC) tag_example.o -o tag_example $(OBJDEPS) $(LDFLAGS)

cuda/incrementer_cuda.cubin: cuda/incrementer_cuda.cu cuda/incrementer_cuda.h
	$(NVCC) -cubin cuda/incrementer_cuda.cu -o cuda/incrementer_cuda.cubin --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) 

cuda/spmv_cuda.cubin: cuda/spmv.cu
	$(NVCC) -cubin cuda/spmv.cu -o cuda/spmv_cuda.cubin --compiler-options -fno-strict-aliasing  $(NVCCFLAGS) 

clean:
	@rm -f tag_example multiplication multiplication2 heat gpuirq dw_spmv incrementer dw_mult dw_factolu *.o
	@rm -f cuda/incrementer_cuda.cubin cuda/spmv_cuda.cubin dw_sparse_cg
