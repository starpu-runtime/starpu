# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
LIBS += $(STARPU_OPENCL_LDFLAGS) $(STARPU_CUDA_LDFLAGS) $(STARPU_HIP_LDFLAGS)
LIBS += $(STARPU_MAX_FPGA_LDFLAGS)

EXTRA_DIST =				\
	basic.h

CLEANFILES = 					\
	*.gcno *.gcda *.linkinfo core starpu_idle_microsec.log *.mod *.png *.output tasks.rec perfs.rec */perfs.rec */*/perfs.rec perfs2.rec fortran90/starpu_mod.f90 bandwidth-*.dat bandwidth.gp bandwidth.eps bandwidth.svg *.csv *.md *.Rmd *.pdf *.html

BUILT_SOURCES =
SUBDIRS =

examplebindir = $(libdir)/starpu/examples

if STARPU_USE_OPENCL
nobase_STARPU_OPENCL_DATA_DATA =
endif

#####################################
# What to install and what to check #
#####################################

TESTS = $(myPROGRAMS)

myPROGRAMS =

if STARPU_HAVE_WINDOWS
check_PROGRAMS	=	$(myPROGRAMS)
else
check_PROGRAMS	=	$(LOADER) $(myPROGRAMS)
endif
noinst_PROGRAMS += $(myPROGRAMS)

###################
# Recursive tasks #
###################

if !STARPU_SIMGRID
if STARPU_RECURSIVE_TASKS
myPROGRAMS +=		\
	b		\
	bb		\
	bb_strongdeps	\
	btb		\
	btb_func	\
	bbt		\
	brob		\
	btt		\
	bbtt		\
	tbbt		\
	tbtbt		\
	brt		\
	brbtt		\
	gemm_dag	\
	b2t		\
	brec		\
	brec_level	\
	read		\
	tbrbtt		\
	P2bUt		\
	Nbrbt		\
	brbrbt		\
	b-nbt		\
	tbtbt-3data	\
	turnRW		\
	unpartitionRO	\
	flush		\
	brbtbrb
endif
endif

b_SOURCES = b.c
bb_SOURCES = bb.c
bb_strongdeps_SOURCES = bb_strongdeps.c
btb_SOURCES = btb.c
btb_func_SOURCES = btb_func.c
bbt_SOURCES = bbt.c
brob_SOURCES = brob.c
btt_SOURCES = btt.c
bbtt_SOURCES = bbtt.c
tbbt_SOURCES = tbbt.c
tbtbt_SOURCES = tbtbt.c
brt_SOURCES = brt.c
brbtt_SOURCES = brbtt.c
gemm_dag_SOURCES = gemm_dag.c
b2t_SOURCES = b2t.c
brec_SOURCES = brec.c
brec_level_SOURCES = brec_level.c
read_SOURCES = read.c
tbrbtt_SOURCES = tbrbtt.c
P2bUt_SOURCES = P2bUt.c
Nbrbt_SOURCES = Nbrbt.c
brbrbt_SOURCES = brbrbt.c
b_nbt_SOURCES = b-nbt.c
tbtbt_3data_SOURCES = tbtbt-3data.c
turnRW_SOURCES = turnRW.c
unpartitionRO_SOURCES = unpartitionRO.c
flush_SOURCES = flush.c
brbtbrb_SOURCES = brbtbrb.c

if STARPU_RECURSIVE_TASKS
if STARPU_USE_CUDA
myPROGRAMS +=			\
	b_on_cuda
b_on_cuda_SOURCES = b_on_cuda.c
b_on_cuda_SOURCES += basic.cu
b_SOURCES += basic.cu
bb_SOURCES += basic.cu
bb_strongdeps_SOURCES += basic.cu
btb_SOURCES += basic.cu
btb_func_SOURCES += basic.cu
brob_SOURCES += basic.cu
btt_SOURCES += basic.cu
brbrbt_SOURCES += basic.cu
tbtbt_3data_SOURCES += basic.cu
brbtbrb_SOURCES += basic.cu
brec_SOURCES += basic.cu
brec_level_SOURCES += basic.cu
read_SOURCES += basic.cu
b_nbt_SOURCES += basic.cu
brt_SOURCES += basic.cu
b2t_SOURCES += basic.cu
bbt_SOURCES += basic.cu
flush_SOURCES += basic.cu
unpartitionRO_SOURCES += basic.cu
turnRW_SOURCES += basic.cu
P2bUt_SOURCES += basic.cu
tbbt_SOURCES += basic.cu
tbrbtt_SOURCES += basic.cu
brbtt_SOURCES += basic.cu
bbtt_SOURCES += basic.cu
Nbrbt_SOURCES += basic.cu
tbtbt_SOURCES += basic.cu
endif
endif

