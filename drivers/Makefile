.PHONY: core cuda spu gordon

DRIVERS=

ifdef CPUS
	DRIVERS+= core
endif

ifeq ($(CUDA), 1)
	DRIVERS+= cuda
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

spu:
	@make -C spu

gordon:
	@make -C gordon

clean:
	@make -C core clean
	@make -C cuda clean
	@make -C spu clean
	@make -C gordon clean
