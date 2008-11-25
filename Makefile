export 

CC=gcc -m64
NVCC = nvcc -m64
NVCCFLAGS += -O3

CFLAGS += -g -O0 -Wall
CFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses
CFLAGS += -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type 
CFLAGS += -Wno-unused-function  -Wstrict-prototypes -Wnested-externs -fno-strict-aliasing


ifdef DYNAMIC
CFLAGS+= -fPIC
endif

LDFLAGS+= -lm

ifdef COVERAGE
#	CFLAGS += -fprofile-arcs -ftest-coverage
	CFLAGS += --coverage
	LDFLAGS += --coverage
endif

ifdef TRANSFER_OVERHEAD
	CFLAGS += -DTRANSFER_OVERHEAD
endif

ifdef PERTURB_AMPL
	CFLAGS+="-DUSE_PERTURBATION"
	CFLAGS+="-DAMPL=$(PERTURB_AMPL)"
endif

ifdef MODEL_DEBUG
	CFLAGS+="-DMODEL_DEBUG "
endif

ifdef PERF_MODEL_DIR
	CFLAGS+="-DPERF_MODEL_DIR=\"$(PERF_MODEL_DIR)/\""
else
	CFLAGS+="-DPERF_MODEL_DIR=\"$(PWD)/.sampling/\""
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
	CFLAGS += -pg
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
	FXTDIR=/home/gonnet/These/Libs/FxT/target
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
OBJDEPS += common/hash.o
OBJDEPS += common/timing.o 
OBJDEPS += common/htable32.o 
OBJDEPS += common/mutex.o
OBJDEPS += common/rwlock.o 
OBJDEPS += core/perfmodel/perfmodel.o 
OBJDEPS += core/perfmodel/regression.o 
OBJDEPS += core/perfmodel/perfmodel_history.o 
OBJDEPS += core/jobs.o core/workers.o
OBJDEPS += core/dependencies/tags.o
OBJDEPS += core/dependencies/htable.o
OBJDEPS += core/mechanisms/queues.o
OBJDEPS += core/mechanisms/priority_queues.o
OBJDEPS += core/mechanisms/deque_queues.o
OBJDEPS += core/mechanisms/fifo_queues.o
OBJDEPS += core/policies/sched_policy.o
OBJDEPS += core/policies/no-prio-policy.o
OBJDEPS += core/policies/eager-central-policy.o
OBJDEPS += core/policies/eager-central-priority-policy.o
OBJDEPS += core/policies/work-stealing-policy.o
OBJDEPS += core/policies/deque-modeling-policy.o
OBJDEPS += core/policies/random-policy.o
OBJDEPS += datawizard/copy-driver.o
OBJDEPS += datawizard/coherency.o 
OBJDEPS += datawizard/hierarchy.o 
OBJDEPS += datawizard/memalloc.o
OBJDEPS += datawizard/footprint.o
OBJDEPS += datawizard/interfaces/blas_filters.o
OBJDEPS += datawizard/interfaces/csr_filters.o
OBJDEPS += datawizard/interfaces/bcsr_filters.o
OBJDEPS += datawizard/interfaces/vector_filters.o
OBJDEPS += datawizard/interfaces/blas_interface.o
OBJDEPS += datawizard/interfaces/csr_interface.o
OBJDEPS += datawizard/interfaces/bcsr_interface.o
OBJDEPS += datawizard/interfaces/vector_interface.o
OBJDEPS += task-models/blas_model.o

ifeq ($(CUDA), 1)
	OBJDEPS += drivers/cuda/driver_cuda.o
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


LIBS=starpu.a
ifdef DYNAMIC
	LIBS+= starpu.so
endif



all:
	@make -C common
	@make -C core
	@make -C drivers
	@make -C datawizard
	@make -C task-models
	@make -C tools

starpu.so: all
	gcc --shared -o starpu.so $(OBJDEPS)

starpu.a: all
	$(AR) rcs $@ $(OBJDEPS)

libs: $(LIBS)

examples: libs
	@make -C examples

tags:
	find .|xargs ctags

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
	@echo "  CPUS number of CPUs in use (0)"
	@echo "  CPUS run ATLAS functions (0)"
	@echo "  CHECK makes sure the output is correct (0)"
