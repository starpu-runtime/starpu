# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2019-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/starpu-loader.mk

AM_CFLAGS += $(APP_CFLAGS)
AM_CXXFLAGS += $(APP_CXXFLAGS)
AM_FFLAGS += $(APP_FFLAGS)
AM_FCFLAGS += $(APP_FCFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/src -I$(top_srcdir)/src/ $(STARPU_H_CPPFLAGS)
AM_CPPFLAGS += -I$(top_srcdir)/bubble/include
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(HWLOC_LIBS)
LIBS += $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS)
#LIBS += ../src/libstarpububble-@STARPU_EFFECTIVE_VERSION@.la

BUILT_SOURCES =

if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA =
endif

EXTRA_DIST =		\
	basic/basic.h

CLEANFILES = *.gcno *.gcda *.linkinfo core starpu_idle_microsec.log

#####################################
# What to install and what to check #
#####################################

if STARPU_HAVE_WINDOWS
check_PROGRAMS		=	$(myPROGRAMS)
else
check_PROGRAMS		=	$(LOADER) $(myPROGRAMS)
endif

TESTS = $(myPROGRAMS)

myPROGRAMS =

if !STARPU_SIMGRID
if STARPU_BUBBLE
myPROGRAMS +=			\
	basic/b			\
	basic/bb		\
	basic/btb		\
	basic/btb_func		\
	basic/bbt		\
	basic/btt		\
	basic/bbtt		\
	basic/tbbt		\
	basic/tbtbt		\
	basic/brt		\
	basic/brbtt		\
	basic/sync		\
	basic/gemm_dag		\
	basic/b2t		\
	basic/brec		\
	basic/brec_level	\
	basic/read		\
	basic/tbrbtt		\
	vector/vector
endif
endif

noinst_PROGRAMS	+= $(myPROGRAMS)
