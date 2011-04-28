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


TESTS = $(check_PROGRAMS)

check_PROGRAMS =				\
  base						\
  pointers					\
  scalar-tasks

scalar_tasks_LDADD = $(top_builddir)/src/libstarpu.la

dist_noinst_HEADERS = lib.h

AM_CPPFLAGS = -I$(builddir)

# Unfortunately `libtool --mode=execute' doesn't help here, so
# hard-code the real file name.
AM_CFLAGS =					\
  -fplugin="$(builddir)/../src/.libs/starpu.so"	\
  -dH -fdump-tree-gimple -Wall

CLEANFILES = *.gimple
