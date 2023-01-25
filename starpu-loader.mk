# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

noinst_PROGRAMS		=

#
# Test loading goes through a lot of launchers:
#
# - $(LAUNCHER) is called first, to run the test through starpu_msexec, i.e.
#   either mpirun or starpu_tcpipexec
#
# - $(LOADER), i.e. tests/loader, is then called to implement timeout, running
#   gdb, etc. But if it detects that the test is a .sh script, it just executes
#   it
#
# - $(STARPU_CHECK_LAUNCHER) $(STARPU_CHECK_LAUNCHER_ARGS) is called by loader
#   to run the program through e.g. valgrind.sh
#
# When the program is a shell script, additionally:
#
# - $(STARPU_SUB_PARALLEL) is called to control parallelism (see below)
#
# - $(MS_LAUNCHER) is called to run the test through starpu_msexec
#
# - $(STARPU_LAUNCH) was set by tests/loader to its own path, to run the program
#   through it.
#
# - $(STARPU_CHECK_LAUNCHER) $(STARPU_CHECK_LAUNCHER_ARGS) is called by loader
#

export LAUNCHER
LAUNCHER_ENV	=

if HAVE_PARALLEL
# When GNU parallel is available and -j is passed to make, run tests through
# parallel, using a "starpu" semaphore.
# Also make test shell scripts run its tests through parallel, using a
# "substarpu" semaphore. This brings some overload, but only one level.
STARPU_SUB_PARALLEL=$(shell echo $(MAKEFLAGS) | sed -ne 's/.*-j\([0-9]\+\).*/parallel --semaphore --id substarpu --fg --fg-exit -j \1/p')
export STARPU_SUB_PARALLEL
endif

export MS_LAUNCHER
if STARPU_USE_MPI_MASTER_SLAVE
# Make tests run through mpiexec
LAUNCHER			= $(abs_top_srcdir)/tools/starpu_msexec
MS_LAUNCHER 			= $(STARPU_MPIEXEC)
LAUNCHER_ENV			+= $(MPI_RUN_ENV) STARPU_NMPIMSTHREADS=4
endif

if STARPU_USE_TCPIP_MASTER_SLAVE
LAUNCHER			= $(abs_top_srcdir)/tools/starpu_msexec
MS_LAUNCHER			= $(abs_top_builddir)/tools/starpu_tcpipexec -np 2 -nobind -ncpus 1
# switch off local socket usage
#MS_LAUNCHER			= $(abs_top_builddir)/tools/starpu_tcpipexec -np 2 -nobind -ncpus 1 -nolocal
LAUNCHER_ENV			+= STARPU_RESERVE_NCPU=2
endif

if STARPU_USE_MPI
LAUNCHER			= $(STARPU_MPIEXEC)
LAUNCHER_ENV			+= $(MPI_RUN_ENV)
endif

LAUNCHER	?=

if STARPU_HAVE_WINDOWS
LOADER_BIN		=	$(LAUNCHER) $(EXTERNAL)
else
LOADER			?=	./loader
loader_CPPFLAGS 	= 	$(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER_BIN		=	$(LAUNCHER) $(LOADER) $(EXTERNAL)
noinst_PROGRAMS		+=	loader
endif

LSAN_OPTIONS ?= suppressions=$(abs_top_srcdir)/tools/dev/lsan/suppressions
TSAN_OPTIONS ?= suppressions=$(abs_top_srcdir)/tools/dev/tsan/starpu.suppr
export LSAN_OPTIONS
export TSAN_OPTIONS

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	$(LAUNCHER_ENV) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER	 	=	$(LOADER_BIN)
else
TESTS_ENVIRONMENT 	=	$(LAUNCHER_ENV) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(LOADER_BIN)
endif

AM_TESTS_FD_REDIRECT = 9>&2
