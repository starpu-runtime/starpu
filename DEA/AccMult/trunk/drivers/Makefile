.PHONY: core cuda cublas

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

all: $(DRIVERS)

core:
	make -C core

cuda:
	make -C cuda

cublas:
	make -C cublas

clean:
	make -C core clean
	make -C cuda clean
	make -C cublas clean
