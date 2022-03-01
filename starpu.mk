# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

AM_CFLAGS = $(GLOBAL_AM_CFLAGS)
AM_CXXFLAGS = $(GLOBAL_AM_CXXFLAGS)
AM_FFLAGS = $(GLOBAL_AM_FFLAGS)
AM_FCFLAGS = $(GLOBAL_AM_FCFLAGS)

if HAVE_PARALLEL
# When GNU parallel is available and -j is passed to make, run tests through
# parallel, using a "starpu" semaphore.
# Also make test shell scripts run its tests through parallel, using a
# "substarpu" semaphore. This brings some overload, but only one level.
STARPU_SUB_PARALLEL=$(shell echo $(MAKEFLAGS) | sed -ne 's/.*-j\([0-9]\+\).*/parallel --semaphore --id substarpu --fg --fg-exit -j \1/p')
export STARPU_SUB_PARALLEL
endif

# These are always defined, both for starpu-mpi and for mpi-ms
# For MPI tests we don't want to oversubscribe the system
MPI_RUN_ENV			= STARPU_WORKERS_NOBIND=1 STARPU_NCPU=3
if STARPU_SIMGRID
STARPU_MPIEXEC			= $(abs_top_builddir)/tools/starpu_smpirun -np 4 -platform $(abs_top_srcdir)/tools/perfmodels/cluster.xml -hostfile $(abs_top_srcdir)/tools/perfmodels/hostfile
else
STARPU_MPIEXEC			= $(MPIEXEC) $(MPIEXEC_ARGS) -np 4
endif

# LAUNCHER should be always put in front of the test loader
LAUNCHER			=
# LAUNCHER_ENV should be always put in TESTS_ENVIRONMENT
LAUNCHER_ENV		=

if STARPU_USE_MPI_MASTER_SLAVE
# Make tests run through mpiexec
LAUNCHER 			+= $(STARPU_MPIEXEC)
LAUNCHER_ENV		+= $(MPI_RUN_ENV) STARPU_NMPIMSTHREADS=4
endif

if STARPU_USE_TCPIP_MASTER_SLAVE
LAUNCHER			+=$(abs_top_srcdir)/tools/starpu_tcpipexec -np 2 -nobind -ncpus 1
# switch off local socket usage
#LAUNCHER			+=$(abs_top_srcdir)/tools/starpu_tcpipexec -np 2 -nobind -ncpus 1 -nolocal
LAUNCHER_ENV			+= STARPU_RESERVE_NCPU=2
endif

V_nvcc_  = $(V_nvcc_$(AM_DEFAULT_VERBOSITY))
V_nvcc_0 = @echo "  NVCC    " $@;
V_nvcc_1 =
V_nvcc   = $(V_nvcc_$(V))

V_icc_  = $(V_icc_$(AM_DEFAULT_VERBOSITY))
V_icc_0 = @echo "  ICC     " $@;
V_icc_1 =
V_icc   = $(V_icc_$(V))

V_ln_  = $(V_ln_$(AM_DEFAULT_VERBOSITY))
V_ln_0 = @echo "  LN      " $@;
V_ln_1 =
V_ln   = $(V_ln_$(V))

V_help2man_  = $(V_help2man_$(AM_DEFAULT_VERBOSITY))
V_help2man_0 = @echo "  HELP2MAN" $@;
V_help2man_1 =
V_help2man   = $(V_help2man_$(V))

showfailed:
	@! grep "^FAIL " $(TEST_LOGS) /dev/null
	@! grep -l "ERROR: AddressSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -l "WARNING: AddressSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -l "ERROR: ThreadSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -l "WARNING: ThreadSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -l "ERROR: LeakSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -l "WARNING: LeakSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -l " runtime error: " $(TEST_LOGS) /dev/null
	@RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showfailed || RET=1 ; \
	done ; \
	exit $$RET

showcheck:
	-cat $(TEST_LOGS) /dev/null
	@! grep -q "ERROR: AddressSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -q "WARNING: AddressSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -q "ERROR: ThreadSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -q "WARNING: ThreadSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -q "ERROR: LeakSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -q "WARNING: LeakSanitizer: " $(TEST_LOGS) /dev/null
	@! grep -q " runtime error: " $(TEST_LOGS) /dev/null
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showcheck || RET=1 ; \
	done ; \
	exit $$RET

showsuite:
	-cat $(TEST_SUITE_LOG) /dev/null
	@! grep -q "ERROR: AddressSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "WARNING: AddressSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "ERROR: ThreadSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "WARNING: ThreadSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "ERROR: LeakSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q "WARNING: LeakSanitizer: " $(TEST_SUITE_LOG) /dev/null
	@! grep -q " runtime error: " $(TEST_SUITE_LOG) /dev/null
	RET=0 ; \
	for i in $(SUBDIRS) ; do \
		make -C $$i showsuite || RET=1 ; \
	done ; \
	exit $$RET

if STARPU_SIMGRID
export STARPU_PERF_MODEL_DIR=$(abs_top_srcdir)/tools/perfmodels/sampling
export STARPU_HOSTNAME=mirage
export MALLOC_PERTURB_=0

env:
	@echo export STARPU_PERF_MODEL_DIR=$(STARPU_PERF_MODEL_DIR)
	@echo export STARPU_HOSTNAME=$(STARPU_HOSTNAME)
	@echo export MALLOC_PERTURB_=$(MALLOC_PERTURB_)
endif

if STARPU_SIMGRID
export STARPU_SIMGRID=1
endif

if STARPU_QUICK_CHECK
export STARPU_QUICK_CHECK=1
endif

if STARPU_LONG_CHECK
export STARPU_LONG_CHECK=1
endif
