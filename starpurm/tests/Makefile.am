# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2017-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/starpu-tests.mk
SUBDIRS =

CLEANFILES = *.gcno *.gcda *.linkinfo

AM_CFLAGS	+= $(DLB_CFLAGS)
AM_CPPFLAGS	= -I$(top_srcdir)/include -I$(top_srcdir)/src -I$(top_builddir)/src -I$(top_builddir)/include
AM_CPPFLAGS	+= -I$(top_srcdir)/starpurm/include -I$(top_srcdir)/starpurm/src -I$(top_builddir)/starpurm/src -I$(top_builddir)/starpurm/include $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/starpurm/src/libstarpurm-@STARPU_EFFECTIVE_VERSION@.la $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(HWLOC_LIBS) $(DLB_LIBS)

noinst_PROGRAMS		=
noinst_PROGRAMS += $(myPROGRAMS)

check_PROGRAMS	=	$(myPROGRAMS)

TESTS = $(myPROGRAMS)

myPROGRAMS =
myPROGRAMS += 01_init_exit
myPROGRAMS += 02_list_units
myPROGRAMS += 03_cpusets
myPROGRAMS += 04_drs_enable
