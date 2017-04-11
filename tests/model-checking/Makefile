# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017  Université de Bordeaux
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

STARPU=../../
CPPFLAGS=-I$(STARPU)/src
CFLAGS=-Wall -Wextra -g $(STARPU)/src/common/rbtree.c
LDFLAGS=-lsimgrid

MC_FLAGS=--cfg=model-check/reduction:none
#MC_FLAGS+=--cfg=contexts/factory:ucontext
#MC_FLAGS+=--cfg=model-check/sparse-checkpoint:yes
#MC_FLAGS+=--cfg=model-check/visited:1000

test: prio_list
	simgrid-mc ./prio_list platform.xml MAIN $(MC_FLAGS)

debug: prio_list
	simgrid-mc ./prio_list platform.xml MAIN --log=mc_safety.thres:debug $(MC_FLAGS)

clean:
	rm -f prio_list
