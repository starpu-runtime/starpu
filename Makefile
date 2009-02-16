.PHONY: tags drivers examples common core tools tests task-models datawizard

export 

CC=gcc -m64
NVCC = nvcc -m64
NVCCFLAGS += -O3

CFLAGS += -g -O3 -Wall
CFLAGS += -W -Wall -Wimplicit -Wswitch -Wformat -Wchar-subscripts -Wparentheses
CFLAGS += -Wmultichar -Wtrigraphs -Wpointer-arith -Wcast-align -Wreturn-type 
CFLAGS += -Wno-unused-function  -Wstrict-prototypes -Wnested-externs -fno-strict-aliasing

# This will be useful for program which use CUDA (and .cubin files) which need some path
# to the CUDA code at runtime.
CFLAGS+="-DSTARPUDIR=\"$(PWD)\""

ifdef DYNAMIC
CFLAGS+= -fPIC
endif

LDFLAGS+= -lm
LDFLAGS += -lpthread

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

ifdef DATA_STATS
	CFLAGS+="-DDATA_STATS "
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
ATLASDIR=/home/gonnet/These/Libs/ATLAS/
endif

ifndef GOTODIR
GOTODIR=/home/gonnet/These/Libs/GotoBLAS/GotoBLAS/
endif

ifndef SCALPDIR
SCALPDIR=/home/gonnet/Scalp/scalp/trunk/
endif

ifdef PERF_DEBUG
	CFLAGS += -DPERF_DEBUG
	LDFLAGS += -pg
	CFLAGS += -pg
endif

EXTRADEP=

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
	CFLAGS += -I$(ATLASDIR)/include/
	CFLAGS += -DATLAS
	EXTRALDFLAGS+= $(ATLASDIR)/lib/libcblas.a
	EXTRALDFLAGS+= $(ATLASDIR)/lib/libatlas.a
endif

ifdef GOTO
	CFLAGS += -I$(GOTODIR)
	CFLAGS += -DGOTO
	EXTRALDFLAGS += $(GOTODIR)/libgoto.a
endif

ifdef CPUS
	CFLAGS += -DUSE_CPUS
	CFLAGS += -DNMAXCORES=$(CPUS)
	NVCCFLAGS += -DUSE_CPUS
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
OBJDEPS += core/dependencies/data-concurrency.o
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
OBJDEPS += core/policies/deque-modeling-policy-data-aware.o
OBJDEPS += core/policies/random-policy.o
OBJDEPS += datawizard/progress.o
OBJDEPS += datawizard/copy-driver.o
OBJDEPS += datawizard/data_request.o
OBJDEPS += datawizard/coherency.o 
OBJDEPS += datawizard/hierarchy.o 
OBJDEPS += datawizard/memalloc.o
OBJDEPS += datawizard/footprint.o
OBJDEPS += datawizard/datastats.o
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

ifdef GORDON
	OBJDEPS += drivers/gordon/driver_gordon.o
	EXTRALDFLAGS+= $(SCALPDIR)/cell/gordon/libgordon.a
	EXTRALDFLAGS+= $(SCALPDIR)/util/libsp@ceutil.a
	EXTRALDFLAGS+= $(SCALPDIR)/util/libsp@ceutil.spu.a
	CFLAGS += -I$(SCALPDIR)
endif

ifdef CPUS
	OBJDEPS += drivers/core/driver_core.o
endif

ifdef USE_FXT
	LDFLAGS += -lfxt
	OBJDEPS += common/fxt.o
endif

ifdef GOTO
	OBJDEPS += common/blas.o
endif

ifdef ATLAS
	OBJDEPS += common/blas.o
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
