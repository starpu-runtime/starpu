# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique
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
  unregister.c					\
  unregister-errors.c				\
  task-errors.c					\
  scalar-tasks.c				\
  pointer-tasks.c				\
  no-initialize.c				\
  lib-user.c					\
  wait-errors.c					\
  heap-allocated.c				\
  heap-allocated-errors.c			\
  shutdown-errors.c

dist_noinst_HEADERS = mocks.h

CLEANFILES = *.gimple				\
  base						\
  pointers					\
  register					\
  scalar-tasks					\
  pointer-tasks

EXTRA_DIST = ./run-test.in			\
  my-lib.h my-lib.c

if HAVE_GUILE

TESTS = $(gcc_tests)
TESTS_ENVIRONMENT = ./run-test

else !HAVE_GUILE

EXTRA_DIST += $(gcc_tests)

check-hook:
	-@echo "GNU Guile not available, test suite not run."

endif !HAVE_GUILE
