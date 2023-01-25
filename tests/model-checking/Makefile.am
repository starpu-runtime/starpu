# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/make/starpu-tests.mk

EXTRA_DIST =		\
	platform.xml	\
	prio_list.sh	\
	barrier.sh	\
	starpu-mc.sh.in

AM_CPPFLAGS = -I$(top_builddir)/src -I$(top_srcdir)/src -I$(top_builddir)/include -I$(top_srcdir)/include $(SIMGRID_CFLAGS)
AM_LDFLAGS  = -Wl,-znorelro -Wl,-znoseparate-code
LIBS += $(SIMGRID_LDFLAGS)

noinst_PROGRAMS = \
		  prio_list \
		  prio_list2 \
		  prio_list3 \
		  starpu_barrier

TESTS = $(SHELL_TESTS)
SHELL_TESTS =

# takes 1s
SHELL_TESTS += prio_list.sh

# https://github.com/simgrid/simgrid/issues/166
#SHELL_TESTS += barrier.sh

if !STARPU_QUICK_CHECK
# takes 7m
SHELL_TESTS += prio_list2.sh
endif

if STARPU_LONG_CHECK
# takes 25m
SHELL_TESTS += prio_list3.sh
endif

#MC_FLAGS=--cfg=model-check/reduction:none

#MC_FLAGS+=--cfg=contexts/factory:ucontext
#MC_FLAGS+=--cfg=model-check/sparse-checkpoint:yes
#MC_FLAGS+=--cfg=model-check/visited:1000

# To record the failing trace
#MC_FLAGS+=--cfg=model-check/record:1
## And replay it without simgrid-mc
#MC_FLAGS+=--cfg=model-check/reply:'1;3;4'

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
