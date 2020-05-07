# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

if STARPU_USE_MPI_MASTER_SLAVE
MPI_LAUNCHER 			= $(MPIEXEC)  $(MPIEXEC_ARGS) -np 4
MPI_RUN_ARGS			= STARPU_WORKERS_NOBIND=1 STARPU_NCPU=4 STARPU_NMPIMSTHREADS=4
endif

V_nvcc_  = $(V_nvcc_$(AM_DEFAULT_VERBOSITY))
V_nvcc_0 = @echo "  NVCC    " $@;
V_nvcc_1 =
V_nvcc   = $(V_nvcc_$(V))

V_icc_  = $(V_icc_$(AM_DEFAULT_VERBOSITY))
V_icc_0 = @echo "  ICC     " $@;
V_icc_1 =
V_icc   = $(V_icc_$(V))

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
STARPU_PERF_MODEL_DIR=$(abs_top_srcdir)/tools/perfmodels/sampling
STARPU_HOSTNAME=mirage
MALLOC_PERTURB_=0
export STARPU_PERF_MODEL_DIR
export STARPU_HOSTNAME
export MALLOC_PERTURB_

env:
	@echo export STARPU_PERF_MODEL_DIR=$(STARPU_PERF_MODEL_DIR)
	@echo export STARPU_HOSTNAME=$(STARPU_HOSTNAME)
	@echo export MALLOC_PERTURB_=$(MALLOC_PERTURB_)
endif

if STARPU_SIMGRID
STARPU_SIMGRID=1
export STARPU_SIMGRID
endif

if STARPU_QUICK_CHECK
STARPU_QUICK_CHECK=1
export STARPU_QUICK_CHECK
endif

if STARPU_LONG_CHECK
STARPU_LONG_CHECK=1
export STARPU_LONG_CHECK
endif
