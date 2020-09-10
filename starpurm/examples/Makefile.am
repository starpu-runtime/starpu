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
include $(top_srcdir)/starpu.mk
SUBDIRS =

CLEANFILES = *.gcno *.gcda *.linkinfo

AM_CPPFLAGS	= -I$(top_srcdir)/include -I$(top_srcdir)/src -I$(top_builddir)/src -I$(top_builddir)/include
AM_CPPFLAGS	+= -I$(top_srcdir)/starpurm/include -I$(top_srcdir)/starpurm/src -I$(top_builddir)/starpurm/src -I$(top_builddir)/starpurm/include
AM_CFLAGS	= -Wall -g $(HWLOC_CFLAGS) $(DLB_CFLAGS)
LDADD	= $(top_builddir)/starpurm/src/libstarpurm-@STARPU_EFFECTIVE_VERSION@.la $(top_builddir)/src/@LIBSTARPU_LINK@ $(HWLOC_LIBS) $(DLB_LIBS)

examplebindir = $(libdir)/starpu/examples/starpurm

examplebin_PROGRAMS =
examplebin_PROGRAMS += 01_init_exit
examplebin_PROGRAMS += 02_list_units
examplebin_PROGRAMS += 03_cpusets
examplebin_PROGRAMS += 04_drs_enable
examplebin_PROGRAMS += 05_vector_scale
examplebin_PROGRAMS += 06_spawn
examplebin_PROGRAMS += 07_async_spawn

01_init_exit_SOURCES = 01_init_exit.c

02_list_units_SOURCES = 02_list_units.c

03_cpusets_SOURCES = 03_cpusets.c

04_drs_enable_SOURCES = 04_drs_enable.c

05_vector_scale_SOURCES = 05_vector_scale.c

06_spawn_SOURCES = 06_spawn.c

07_async_spawn_SOURCES = 07_async_spawn.c
