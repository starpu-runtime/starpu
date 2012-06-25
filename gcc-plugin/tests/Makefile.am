# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011, 2012 Institut National de Recherche en Informatique et Automatique
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


gcc_tests =					\
  base.c					\
  pointers.c					\
  output-pointer.c				\
  output-pointer-errors.c			\
  register.c					\
  register-errors.c				\
  acquire.c					\
  acquire-errors.c				\
  release.c					\
  release-errors.c				\
  unregister.c					\
  unregister-errors.c				\
  task-errors.c					\
  scalar-tasks.c				\
  pointer-tasks.c				\
  external-task-impl.c				\
  no-initialize.c				\
  lib-user.c					\
  wait-errors.c					\
  heap-allocated.c				\
  heap-allocated-errors.c			\
  verbose.c					\
  debug-tree.c					\
  opencl.c					\
  opencl-errors.c				\
  shutdown-errors.c

EXTRA_DIST =

if !STARPU_USE_OPENCL

# XXX: This test simulates a buggy OpenCL implementation, and thus
# cannot be run then a real <cl_platform.h> is included.
gcc_tests += opencl-types.c

# This test simulates errors when lacking an OpenCL implementation.
gcc_tests += opencl-lacking.c

else STARPU_USE_OPENCL

EXTRA_DIST +=					\
  opencl-types.c				\
  opencl-lacking.c

endif STARPU_USE_OPENCL


dist_noinst_HEADERS = mocks.h

CLEANFILES = *.gimple *.o			\
  base						\
  pointers					\
  register					\
  release					\
  scalar-tasks					\
  pointer-tasks					\
  lib-user					\
  output-pointer				\
  unregister					\
  heap-allocated				\
  acquire					\
  opencl


EXTRA_DIST += ./run-test.in			\
  my-lib.h my-lib.c				\
  test.cl					\
  $(gcc_tests)

# The test suite assumes that the CPU back-end is available.
if RUN_GCC_PLUGIN_TESTS

TESTS = $(gcc_tests)
TESTS_ENVIRONMENT = ./run-test

else !RUN_GCC_PLUGIN_TESTS

check-hook:
	-@echo "GNU Guile or CPU back-end not available, test suite not run."

endif !RUN_GCC_PLUGIN_TESTS

showcheck:
	-cat $(TEST_LOGS) /dev/null
