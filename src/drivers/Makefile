.PHONY: core cuda gordon

DRIVERS=

ifdef CPUS
	DRIVERS+= core
endif

ifeq ($(CUDA), 1)
	DRIVERS+= cuda
endif

ifdef GORDON
	DRIVERS += gordon
endif

all: $(DRIVERS)

core:
	@make -C core

cuda:
	@make -C cuda

gordon:
	@make -C gordon

clean:
	@make -C core clean
	@make -C cuda clean
	@make -C gordon clean
