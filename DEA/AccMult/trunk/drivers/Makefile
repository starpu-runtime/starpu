.PHONY: core cuda cublas spu gordon

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

ifdef GORDON
	DRIVERS += gordon
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

gordon:
	@make -C gordon

clean:
	@make -C core clean
	@make -C cuda clean
	@make -C cublas clean
	@make -C spu clean
	@make -C gordon clean
