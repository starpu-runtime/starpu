.PHONY: core cuda cublas spu

DRIVERS=

ifdef CPUS
	DRIVERS+= core
endif

ifeq ($(CUDA), 1)
	DRIVERS+= cuda
endif

ifeq ($(CUBLAS), 1)
	DRIVERS += cublas
endif

ifdef SPU
	DRIVERS += spu
endif

all: $(DRIVERS)

core:
	@make -C core

cuda:
	@make -C cuda

cublas:
	@make -C cublas

spu:
	@make -C spu

clean:
	@make -C core clean
	@make -C cuda clean
	@make -C cublas clean
	@make -C spu clean
