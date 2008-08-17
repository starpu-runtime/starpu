export 

CC=gcc -m64
NVCC = nvcc -m64
NVCCFLAGS += -O3

CFLAGS += -g -O3 -Wall -fPIC
CFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses
CFLAGS += -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type 
CFLAGS += -Wno-unused-function  -Wstrict-prototypes -Wnested-externs -fno-strict-aliasing

LDFLAGS+= -lm


ifdef PERTURB_AMPL
	CFLAGS+="-DUSE_PERTURBATION"
	CFLAGS+="-DAMPL=$(PERTURB_AMPL)"
endif

ifdef MODEL_DEBUG
	CFLAGS+="-DMODEL_DEBUG "
endif

# to use CUDA
CUDASDKDIR=/home/gonnet/NVIDIA_CUDA_SDK/
CUDAINSTALLDIR=/usr/local/cuda/

ifndef ATLASDIR
ATLASDIR=/home/gonnet/DEA/BLAS/ATLAS/ATLAS/
endif
ifndef BLASARCH
BLASARCH=Linux_UNKNOWNSSE2
endif

ifdef PERF_DEBUG
	CFLAGS += -DPERF_DEBUG
	LDFLAGS += -pg
endif

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

ifdef USE_OVERLOAD
	CFLAGS += -DUSE_OVERLOAD
endif

ifeq ($(CUDA), 1)
	CFLAGS += -DUSE_CUDA 
	NVCCFLAGS +=  -DUSE_CUDA 
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
	NVCCFLAGS += -I$(CUDAINSTALLDIR)/include 
	LDFLAGS += -lcuda -L/usr/local/cuda/lib 
	LDFLAGS += -lcublas
endif

ifeq ($(CUBLAS), 1)
	CFLAGS += -DUSE_CUBLAS
	CFLAGS += -I$(CUDAINSTALLDIR)/include 
ifdef CUBLASEMU
	LDFLAGS += -lcublasemu
else
	LDFLAGS += -lcublas
endif

	LDFLAGS += -L$(CUDASDKDIR)/lib
	LDFLAGS += -L$(CUDAINSTALLDIR)/lib -lcudart -lcuda

	CFLAGS += -I$(CUDAINSTALLDIR)/include 
endif

ifdef ATLAS
	CFLAGS+= -I$(ATLASDIR)/include/
	LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libcblas.a
	LDFLAGS+= $(ATLASDIR)/lib/$(BLASARCH)/libatlas.a
endif


ifdef CPUS
	CFLAGS += -DUSE_CPUS
	CFLAGS += -DNMAXCORES=$(CPUS)
	NVCCFLAGS += -DUSE_CPUS
endif

ifdef SPU
	CFLAGS += -DUSE_SPU
	CFLAGS += -DMAXSPUS=$(SPU)
	LDFLAGS += -lspe2
endif

ifdef GORDON
	CFLAGS += -DUSE_GORDON
	LDFLAGS += -lspe2
endif

ifndef FXTDIR
	FXTDIR=/home/gonnet/FxT/target
endif

ifdef USE_FXT
	CFLAGS += -DUSE_FXT -I$(FXTDIR)/include/ -DCONFIG_FUT
	LDFLAGS += -lfxt -L$(FXTDIR)/lib/
endif

ifdef NO_PRIO
	CFLAGS += -DNO_PRIO
endif

#
#	To create the static and dynamic libraries, we need a list of all the object
#	files that are needed by the application which use our runtime
#

OBJDEPS += common/threads.o
OBJDEPS += common/timing.o 
OBJDEPS += core/jobs.o core/workers.o
OBJDEPS += core/dependencies/tags.o
OBJDEPS += core/dependencies/htable.o
OBJDEPS += core/mechanisms/queues.o
OBJDEPS += core/mechanisms/priority_queues.o
OBJDEPS += core/mechanisms/deque_queues.o
OBJDEPS += core/policies/sched_policy.o
OBJDEPS += core/policies/eager-central-policy.o
OBJDEPS += core/policies/eager-central-priority-policy.o
OBJDEPS += core/policies/work-stealing-policy.o
OBJDEPS += core/policies/deque-modeling-policy.o
OBJDEPS += core/policies/random-policy.o
OBJDEPS += common/mutex.o
OBJDEPS += common/rwlock.o 
OBJDEPS += datawizard/copy-driver.o
OBJDEPS += datawizard/coherency.o 
OBJDEPS += datawizard/hierarchy.o 
OBJDEPS += datawizard/interfaces/blas_filters.o
OBJDEPS += datawizard/interfaces/csr_filters.o
OBJDEPS += datawizard/interfaces/vector_filters.o
OBJDEPS += datawizard/memalloc.o
OBJDEPS += datawizard/interfaces/blas_interface.o
OBJDEPS += datawizard/interfaces/csr_interface.o
OBJDEPS += datawizard/interfaces/vector_interface.o
OBJDEPS += task-models/blas_model.o

ifeq ($(CUDA), 1)
	OBJDEPS += drivers/cuda/driver_cuda.o
endif

ifeq ($(CUBLAS), 1)
	OBJDEPS += drivers/cublas/driver_cublas.o
endif

ifdef SPU
	OBJDEPS += drivers/spu/ppu/driver_spu_ppu.o
	OBJDEPS += drivers/spu/spu/spu_worker_program.o
endif

ifdef GORDON
	OBJDEPS += drivers/gordon/driver_gordon.o
	OBJDEPS += drivers/gordon/externals/scalp/util/util.o
	OBJDEPS += drivers/gordon/externals/scalp/cell/gordon/gordon.o 
	OBJDEPS += drivers/gordon/externals/scalp/cell/gordon/pputil.o 
	OBJDEPS += drivers/gordon/externals/scalp/cell/gordon/spuembed.o 
	CFLAGS += -I./drivers/gordon/externals/scalp/
endif

ifdef CPUS
	OBJDEPS += drivers/core/driver_core.o
endif

ifdef USE_FXT
	LDFLAGS += -lfxt
	OBJDEPS += common/fxt.o
endif



all:
	@make -C common
	@make -C core
	@make -C drivers
	@make -C datawizard
	@make -C task-models
	@make -C tools

ribsched.so: all
	gcc --shared -o ribsched.so $(OBJDEPS)

ribsched.a: all
	$(AR) rcs $@ $(OBJDEPS)

libs: ribsched.so ribsched.a

examples: ribsched.a ribsched.so
	@make -C examples


clean:
	@make -C common clean
	@make -C core clean
	@make -C drivers clean
	@make -C datawizard clean
	@make -C task-models clean
	@make -C examples clean
	@make -C tools clean
	@rm -f *.so *.a

help:
	@echo "Possible options (default value) "
	@echo "  CUDA use of cuda or not (0)"
	@echo "  CUBLAS use of cublas or not (0)"
	@echo "  CPUS number of CPUs in use (0)"
	@echo "  CPUS run ATLAS functions (0)"
	@echo "  CHECK makes sure the output is correct (0)"
