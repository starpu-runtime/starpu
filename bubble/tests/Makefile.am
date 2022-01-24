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
include $(top_srcdir)/starpu.mk

AM_CFLAGS += -Wno-unused
AM_CXXFLAGS += -Wno-unused
AM_FFLAGS += -Wno-unused -Wno-unused-dummy-argument
AM_FCFLAGS += -Wno-unused -Wno-unused-dummy-argument
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

if STARPU_USE_CUDA

NVCCFLAGS += --compiler-options -fno-strict-aliasing $(AM_CPPFLAGS) -I$(top_builddir)/include $(HWLOC_CFLAGS) $(SIMGRID_CFLAGS)

.cu.o:
	$(V_nvcc) $(NVCC) $< -c -o $@ $(NVCCFLAGS)
endif

#####################################
# What to install and what to check #
#####################################

if STARPU_HAVE_WINDOWS
check_PROGRAMS		=	$(myPROGRAMS)
else
check_PROGRAMS		=	$(LOADER) $(myPROGRAMS)
endif

if !STARPU_HAVE_WINDOWS
## test loader program
if !STARPU_CROSS_COMPILING
LOADER			=	loader
loader_CPPFLAGS 	=	$(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER_BIN		=	$(abs_top_builddir)/bubble/tests/$(LOADER)
loader_SOURCES		=	../../tests/loader.c
else
LOADER			=
LOADER_BIN		=	$(top_builddir)/examples/loader-cross.sh
endif

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER		=	$(LOADER_BIN)
else
TESTS_ENVIRONMENT	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(LOADER_BIN)
endif

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

noinst_PROGRAMS	= $(myPROGRAMS)
