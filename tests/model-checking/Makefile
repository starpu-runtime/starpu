# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#
STARPU=../../
CPPFLAGS=-I$(STARPU)/src -I$(STARPU)/include -I.
CFLAGS+=-Wall -Wextra -g -DNOCONFIG
LDFLAGS+=-lsimgrid -lm -Wl,-znorelro -Wl,-znoseparate-code

MC_FLAGS=--cfg=model-check/reduction:none

ifeq (1,0)
MC_FLAGS+=--cfg=contexts/factory:ucontext
MC_FLAGS+=--cfg=model-check/sparse-checkpoint:yes
MC_FLAGS+=--cfg=model-check/visited:1000
endif

# To record the failing trace
ifeq (1,0)
MC_FLAGS+=--cfg=model-check/record:1
# And replay it without simgrid-mc
#MC_FLAGS+=--cfg=model-check/reply:'1;3;4'
endif

# To see which simix calls are made
#MC_FLAGS+=--log=simix_popping.thres:debug

test: prio_list
	simgrid-mc ./prio_list platform.xml MAIN $(MC_FLAGS)

debug: prio_list
	simgrid-mc ./prio_list platform.xml MAIN --log=mc_safety.thres:debug $(MC_FLAGS)

test-barrier: starpu_barrier
	simgrid-mc ./starpu_barrier platform.xml MAIN $(MC_FLAGS)

debug-barrier: starpu_barrier
	simgrid-mc ./starpu_barrier platform.xml MAIN --log=mc_safety.thres:debug $(MC_FLAGS)

all: prio_list prio_list2 starpu_barrier

clean:
	rm -f prio_list prio_list2 starpu_barrier
