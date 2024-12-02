# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2019-2024   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/make/starpu-tests.mk
include $(top_srcdir)/make/starpu-loader.mk

AM_CFLAGS += $(APP_CFLAGS)
AM_CXXFLAGS += $(APP_CXXFLAGS)
AM_FFLAGS += $(APP_FFLAGS)
AM_FCFLAGS += $(APP_FCFLAGS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_builddir)/src -I$(top_srcdir)/src/ $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)
LIBS += $(HWLOC_LIBS)
LIBS += $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS)

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
if STARPU_RECURSIVE_TASKS
myPROGRAMS +=			\
	basic/b			\
	basic/bb		\
	basic/btb		\
	basic/btb_func		\
	basic/bbt		\
	basic/brob		\
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
	basic/P2bUt		\
	basic/Nbrbt		\
	basic/brbrbt		\
	basic/b-nbt		\
	basic/tbtbt-3data	\
	basic/turnRW		\
	basic/unpartitionRO	\
	basic/flush		\
	basic/brbtbrb           \
	vector/vector
endif
endif

basic_b_SOURCES = basic/b.c
basic_bb_SOURCES = basic/bb.c
basic_btb_SOURCES = basic/btb.c
basic_btb_func_SOURCES = basic/btb_func.c
basic_bbt_SOURCES = basic/bbt.c
basic_brob_SOURCES = basic/brob.c
basic_btt_SOURCES = basic/btt.c
basic_bbtt_SOURCES = basic/bbtt.c
basic_tbbt_SOURCES = basic/tbbt.c
basic_tbtbt_SOURCES = basic/tbtbt.c
basic_brt_SOURCES = basic/brt.c
basic_brbtt_SOURCES = basic/brbtt.c
basic_gemm_dag_SOURCES = basic/gemm_dag.c
basic_b2t_SOURCES = basic/b2t.c
basic_brec_SOURCES = basic/brec.c
basic_brec_level_SOURCES = basic/brec_level.c
basic_read_SOURCES = basic/read.c
basic_tbrbtt_SOURCES = basic/tbrbtt.c
basic_P2bUt_SOURCES = basic/P2bUt.c
basic_Nbrbt_SOURCES = basic/Nbrbt.c
basic_brbrbt_SOURCES = basic/brbrbt.c
basic_b_nbt_SOURCES = basic/b-nbt.c
basic_tbtbt_3data_SOURCES = basic/tbtbt-3data.c
basic_turnRW_SOURCES = basic/turnRW.c
basic_unpartitionRO_SOURCES = basic/unpartitionRO.c
basic_flush_SOURCES = basic/flush.c
basic_brbtbrb_SOURCES = basic/brbtbrb.c
vector_vector_SOURCES = vector/vector.c

if STARPU_USE_CUDA
myPROGRAMS +=			\
	basic/b_on_cuda
basic_b_on_cuda_SOURCES = basic/b_on_cuda.c
basic_b_on_cuda_SOURCES += basic/basic.cu
basic_b_SOURCES += basic/basic.cu
basic_bb_SOURCES += basic/basic.cu
basic_btb_SOURCES += basic/basic.cu
basic_btb_func_SOURCES += basic/basic.cu
basic_brob_SOURCES += basic/basic.cu
basic_btt_SOURCES += basic/basic.cu
basic_brbrbt_SOURCES += basic/basic.cu
basic_tbtbt_3data_SOURCES += basic/basic.cu
basic_brbtbrb_SOURCES += basic/basic.cu
basic_read_SOURCES += basic/basic.cu
basic_b_nbt_SOURCES += basic/basic.cu
basic_brt_SOURCES += basic/basic.cu
basic_b2t_SOURCES += basic/basic.cu
basic_bbt_SOURCES += basic/basic.cu
basic_flush_SOURCES += basic/basic.cu
basic_unpartitionRO_SOURCES += basic/basic.cu
basic_turnRW_SOURCES += basic/basic.cu
basic_P2bUt_SOURCES += basic/basic.cu
basic_tbbt_SOURCES += basic/basic.cu
basic_tbrbtt_SOURCES += basic/basic.cu
basic_brbtt_SOURCES += basic/basic.cu
basic_bbtt_SOURCES += basic/basic.cu
basic_Nbrbt_SOURCES += basic/basic.cu
basic_tbtbt_SOURCES += basic/basic.cu
endif

noinst_PROGRAMS	+= $(myPROGRAMS)
