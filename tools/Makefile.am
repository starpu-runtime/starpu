# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
# Copyright (C) 2013       Joris Pablo
# Copyright (C) 2013       Thibaut Lambert
# Copyright (C) 2017       Erwan Leria
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

AM_CFLAGS = $(HWLOC_CFLAGS) $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(STARPU_COI_CPPFLAGS) $(GLOBAL_AM_CFLAGS)
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@ $(FXT_LIBS)
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/tools/ -I$(top_srcdir)/mpi/ -I$(top_builddir)/src -I$(top_srcdir)/src
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@ $(STARPU_COI_LDFLAGS) $(STARPU_SCIF_LDFLAGS)

bin_PROGRAMS =
dist_bin_SCRIPTS =

dist_pkgdata_DATA = gdbinit

pkgdata_perfmodels_sampling_busdir = $(datarootdir)/starpu/perfmodels/sampling/bus
pkgdata_perfmodels_sampling_codeletsdir = $(datarootdir)/starpu/perfmodels/sampling/codelets/45

dist_pkgdata_perfmodels_sampling_bus_DATA = \
	perfmodels/sampling/bus/attila.affinity	\
	perfmodels/sampling/bus/attila.bandwidth	\
	perfmodels/sampling/bus/attila.config	\
	perfmodels/sampling/bus/attila.latency	\
	perfmodels/sampling/bus/attila.platform.xml	\
	perfmodels/sampling/bus/attila.platform.v4.xml	\
	perfmodels/sampling/bus/hannibal.affinity	\
	perfmodels/sampling/bus/hannibal.bandwidth	\
	perfmodels/sampling/bus/hannibal.config	\
	perfmodels/sampling/bus/hannibal.latency	\
	perfmodels/sampling/bus/hannibal.platform.xml	\
	perfmodels/sampling/bus/hannibal.platform.v4.xml	\
	perfmodels/sampling/bus/hannibal-pitch.affinity	\
	perfmodels/sampling/bus/hannibal-pitch.bandwidth	\
	perfmodels/sampling/bus/hannibal-pitch.config	\
	perfmodels/sampling/bus/hannibal-pitch.latency	\
	perfmodels/sampling/bus/hannibal-pitch.platform.xml	\
	perfmodels/sampling/bus/hannibal-pitch.platform.v4.xml	\
	perfmodels/sampling/bus/idgraf.affinity	\
	perfmodels/sampling/bus/idgraf.bandwidth	\
	perfmodels/sampling/bus/idgraf.config	\
	perfmodels/sampling/bus/idgraf.latency	\
	perfmodels/sampling/bus/idgraf.platform.xml	\
	perfmodels/sampling/bus/idgraf.platform.v4.xml	\
	perfmodels/sampling/bus/mirage.affinity	\
	perfmodels/sampling/bus/mirage.bandwidth	\
	perfmodels/sampling/bus/mirage.config	\
	perfmodels/sampling/bus/mirage.latency	\
	perfmodels/sampling/bus/mirage.platform.xml	\
	perfmodels/sampling/bus/mirage.platform.v4.xml	\
	perfmodels/sampling/bus/sirocco.affinity	\
	perfmodels/sampling/bus/sirocco.bandwidth	\
	perfmodels/sampling/bus/sirocco.config	\
	perfmodels/sampling/bus/sirocco.latency	\
	perfmodels/sampling/bus/sirocco.platform.xml	\
	perfmodels/sampling/bus/sirocco.platform.v4.xml

dist_pkgdata_perfmodels_sampling_codelets_DATA = \
	perfmodels/sampling/codelets/45/chol_model_11.attila	\
	perfmodels/sampling/codelets/45/chol_model_21.attila	\
	perfmodels/sampling/codelets/45/chol_model_22.attila	\
	perfmodels/sampling/codelets/45/cl_update.attila	\
	perfmodels/sampling/codelets/45/save_cl_bottom.attila	\
	perfmodels/sampling/codelets/45/save_cl_top.attila	\
	perfmodels/sampling/codelets/45/starpu_sgemm_gemm.attila	\
	perfmodels/sampling/codelets/45/starpu_dgemm_gemm.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_openblas.attila	\
	perfmodels/sampling/codelets/45/overlap_sleep_1024_24.attila	\
\
	perfmodels/sampling/codelets/45/chol_model_11.hannibal	\
	perfmodels/sampling/codelets/45/chol_model_21.hannibal	\
	perfmodels/sampling/codelets/45/chol_model_22.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_atlas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_atlas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_atlas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_atlas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_goto.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_goto.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_goto.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_goto.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_openblas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_openblas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_openblas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_openblas.hannibal	\
\
	perfmodels/sampling/codelets/45/chol_model_11.hannibal-pitch	\
	perfmodels/sampling/codelets/45/chol_model_21.hannibal-pitch	\
	perfmodels/sampling/codelets/45/chol_model_22.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_atlas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_atlas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_atlas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_atlas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_goto.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_goto.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_goto.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_goto.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_openblas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_openblas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_openblas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_openblas.hannibal-pitch	\
\
	perfmodels/sampling/codelets/45/chol_model_11.idgraf	\
	perfmodels/sampling/codelets/45/chol_model_21.idgraf	\
	perfmodels/sampling/codelets/45/chol_model_22.idgraf	\
	perfmodels/sampling/codelets/45/cl_update.idgraf	\
	perfmodels/sampling/codelets/45/save_cl_bottom.idgraf	\
	perfmodels/sampling/codelets/45/save_cl_top.idgraf	\
	perfmodels/sampling/codelets/45/starpu_sgemm_gemm.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dgemm_gemm.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_openblas.idgraf	\
\
	perfmodels/sampling/codelets/45/chol_model_11.mirage	\
	perfmodels/sampling/codelets/45/chol_model_21.mirage	\
	perfmodels/sampling/codelets/45/chol_model_22.mirage	\
	perfmodels/sampling/codelets/45/cl_update.mirage	\
	perfmodels/sampling/codelets/45/save_cl_bottom.mirage	\
	perfmodels/sampling/codelets/45/save_cl_top.mirage	\
	perfmodels/sampling/codelets/45/starpu_sgemm_gemm.mirage	\
	perfmodels/sampling/codelets/45/starpu_dgemm_gemm.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_openblas.mirage	\
	perfmodels/sampling/codelets/45/overlap_sleep_1024_24.mirage	\
\
	perfmodels/sampling/codelets/45/chol_model_11.sirocco	\
	perfmodels/sampling/codelets/45/chol_model_21.sirocco	\
	perfmodels/sampling/codelets/45/chol_model_22.sirocco	\
	perfmodels/sampling/codelets/45/cl_update.sirocco	\
	perfmodels/sampling/codelets/45/save_cl_bottom.sirocco	\
	perfmodels/sampling/codelets/45/save_cl_top.sirocco	\
	perfmodels/sampling/codelets/45/starpu_sgemm_gemm.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dgemm_gemm.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_11_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_12_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_21_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_22_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_11_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_12_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_21_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_22_openblas.sirocco	\
	perfmodels/sampling/codelets/45/overlap_sleep_1024_24.sirocco

EXTRA_DIST =				\
	dev/checker/rename.sed		\
	dev/checker/rename.sh		\
	dev/cppcheck/suppressions.txt	\
	dev/valgrind/bash.suppr		\
	dev/valgrind/fxt.suppr		\
	dev/valgrind/glpk.suppr		\
	dev/valgrind/hdf5.suppr		\
	dev/valgrind/hwloc.suppr	\
	dev/valgrind/libc.suppr		\
	dev/valgrind/libgomp.suppr	\
	dev/valgrind/libnuma.suppr	\
	dev/valgrind/madmpi.suppr	\
	dev/valgrind/opencl.suppr	\
	dev/valgrind/openmpi.suppr	\
	dev/valgrind/openmp.suppr	\
	dev/valgrind/padico.suppr	\
	dev/valgrind/pthread.suppr	\
	dev/valgrind/starpu.suppr	\
	dev/valgrind/valgrind.suppr	\
	dev/valgrind/valgrind.sh	\
	dev/valgrind/valgrind_xml.sh	\
	dev/valgrind/helgrind.sh	\
	dev/tsan/starpu.suppr		\
	dev/lsan/suppressions		\
	perfmodels/README		\
	perfmodels/cluster.xml		\
	perfmodels/hostfile		\
	perfmodels/sampling/codelets/tmp/mlr_init.out	 \
	msvc/starpu_clean.bat		\
	msvc/starpu_open.bat		\
	msvc/starpu_exec.bat		\
	msvc/starpu_var.bat		\
	msvc/starpu.sln			\
	msvc/starpu/starpu.vcxproj

CLEANFILES = *.gcno *.gcda *.linkinfo starpu_idle_microsec.log figure/* mlr_*

#####################################
# What to install and what to check #
#####################################

STARPU_TOOLS	=
TESTS		= $(STARPU_TOOLS)

noinst_PROGRAMS	=
check_PROGRAMS	=	$(STARPU_TOOLS)

if !STARPU_HAVE_WINDOWS
## test loader program
if !STARPU_CROSS_COMPILING
LOADER			=	loader
loader_CPPFLAGS =  $(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER_BIN		=	$(abs_top_builddir)/tools/$(LOADER)
loader_SOURCES		=	../tests/loader.c
noinst_PROGRAMS		+=	loader
else
LOADER			=
LOADER_BIN		=	$(top_builddir)/tests/loader-cross.sh
endif

if STARPU_USE_MPI_MASTER_SLAVE 
LOADER_BIN2	= $(MPI_LAUNCHER) $(LOADER_BIN)
else
LOADER_BIN2	= $(LOADER_BIN)
endif


if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	$(MPI_RUN_ARGS) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER		=	$(LOADER_BIN2)
else
TESTS_ENVIRONMENT	=	$(MPI_RUN_ARGS) top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(LOADER_BIN2)
endif

endif

if STARPU_USE_FXT
bin_PROGRAMS += 			\
	starpu_fxt_tool			\
	starpu_fxt_stats		\
	starpu_fxt_data_trace

STARPU_TOOLS += 			\
	starpu_fxt_tool			\
	starpu_fxt_stats		\
	starpu_fxt_data_trace

starpu_fxt_tool_CPPFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS) $(FXT_CFLAGS)
starpu_fxt_tool_LDADD = $(FXT_LIBS)
starpu_fxt_tool_LDFLAGS = $(FXT_LDFLAGS)

starpu_fxt_stats_CPPFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS) $(FXT_CFLAGS)
starpu_fxt_stats_LDADD = $(FXT_LIBS)
starpu_fxt_stats_LDFLAGS = $(FXT_LDFLAGS)

starpu_fxt_data_trace_CPPFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS) $(FXT_CFLAGS)
starpu_fxt_data_trace_LDADD = $(FXT_LIBS)
starpu_fxt_data_trace_LDFLAGS = $(FXT_LDFLAGS)
endif

bin_PROGRAMS += 			\
	starpu_perfmodel_display	\
	starpu_perfmodel_plot 		\
	starpu_calibrate_bus		\
	starpu_machine_display		\
	starpu_sched_display		\
	starpu_tasks_rec_complete	\
	starpu_lp2paje			\
	starpu_perfmodel_recdump

if STARPU_SIMGRID
bin_PROGRAMS += 			\
	starpu_replay

starpu_replay_SOURCES = \
	starpu_replay.c \
	starpu_replay_sched.c
endif

starpu_perfmodel_plot_CPPFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS) $(FXT_CFLAGS)

if STARPU_LONG_CHECK
STARPU_TOOLS	+=			\
	starpu_calibrate_bus
endif

STARPU_TOOLS	+=			\
	starpu_machine_display		\
	starpu_sched_display

if !STARPU_HAVE_WINDOWS
STARPU_TOOLS	+=			\
	starpu_perfmodel_display	\
	starpu_perfmodel_plot
endif

dist_bin_SCRIPTS +=			\
	starpu_workers_activity		\
	starpu_codelet_histo_profile	\
	starpu_codelet_profile		\
	starpu_mpi_comm_matrix.py	\
	starpu_paje_draw_histogram	\
	starpu_paje_draw_histogram.R	\
	starpu_paje_summary		\
	starpu_paje_summary.Rmd		\
	starpu_mlr_analysis		\
	starpu_mlr_analysis.Rmd		\
	starpu_paje_state_stats		\
	starpu_paje_state_stats.R	\
	starpu_env			\
	starpu_trace_state_stats.py

if STARPU_USE_AYUDAME2
dist_bin_SCRIPTS +=			\
	starpu_temanejo2.sh
dist_pkgdata_DATA +=			\
	ayudame.cfg
endif

if STARPU_HAVE_WINDOWS
STARPU_MSVC_dir		 =	$(bindir)
nobase_STARPU_MSVC__DATA =		\
	msvc/starpu_clean.bat		\
	msvc/starpu_open.bat		\
	msvc/starpu_exec.bat		\
	msvc/starpu_var.bat		\
	msvc/starpu.sln			\
	msvc/starpu/starpu.vcxproj
endif

if STARPU_HAVE_HELP2MAN
starpu_calibrate_bus.1: starpu_calibrate_bus$(EXEEXT)
	help2man --no-discard-stderr -N -n "Force StarPU bus calibration" --output=$@ ./$<
starpu_machine_display.1: starpu_machine_display$(EXEEXT)
	help2man --no-discard-stderr -N -n "Display machine StarPU information" --output=$@ ./$<
starpu_perfmodel_display.1: starpu_perfmodel_display$(EXEEXT)
	help2man --no-discard-stderr -N -n "Display StarPU performance model" --output=$@ ./$<
starpu_perfmodel_plot.1: starpu_perfmodel_plot$(EXEEXT)
	help2man --no-discard-stderr -N -n "Plot StarPU performance model" --output=$@ ./$<
starpu_tasks_rec_complete.1: starpu_tasks_rec_complete$(EXEEXT)
	help2man --no-discard-stderr -N -n "Complete StarPU tasks.rec file" --output=$@ ./$<
starpu_lp2paje.1: starpu_lp2paje$(EXEEXT)
	help2man --no-discard-stderr -N -n "Convert lp StarPU schedule into Paje format" --output=$@ ./$<
starpu_workers_activity.1: starpu_workers_activity
	chmod +x $<
	help2man --no-discard-stderr -N -n "Display StarPU workers activity" --output=$@ ./$<
starpu_codelet_profile.1: starpu_codelet_profile
	chmod +x $<
	help2man --no-discard-stderr -N -n "Draw StarPU codelet profile" --output=$@ ./$<
starpu_env.1: starpu_env
	chmod +x $<
	help2man --no-discard-stderr -N -n "Set StarPU environment variables" --output=$@ ./$<
starpu_codelet_histo_profile.1: starpu_codelet_histo_profile
	chmod +x $<
	help2man --no-discard-stderr -N -n "Draw StarPU codelet histogram" --output=$@ ./$<
starpu_mpi_comm_matrix.1: starpu_mpi_comm_matrix.py
	chmod +x $<
	help2man --no-discard-stderr -N -n "Draw StarPU MPI communications matrix" --output=$@ ./$<
starpu_paje_draw_histogram.1: starpu_paje_draw_histogram
	chmod +x $<
	help2man --no-discard-stderr -N -n "Draw StarPU trace histogram" --output=$@ ./$<
starpu_paje_state_stats.1: starpu_paje_state_stats
	chmod +x $<
	help2man --no-discard-stderr -N -n "Print statistics from StarPU trace" --output=$@ ./$<

if STARPU_USE_FXT
starpu_fxt_tool.1: starpu_fxt_tool$(EXEEXT)
	help2man --no-discard-stderr -N -n "Convert raw StarPU FxT trace to various traces" --output=$@ ./$<
starpu_fxt_stats.1: starpu_fxt_stats$(EXEEXT)
	help2man --no-discard-stderr -N -n "Print statistics from raw StarPU FxT trace" --output=$@ ./$<
starpu_fxt_data_trace.1: starpu_fxt_data_trace$(EXEEXT)
	help2man --no-discard-stderr -N -n "Print data trace from raw StarPU FxT trace" --output=$@ ./$<
endif

dist_man1_MANS =\
	starpu_calibrate_bus.1 \
	starpu_machine_display.1 \
	starpu_perfmodel_display.1 \
	starpu_perfmodel_plot.1	\
	starpu_tasks_rec_complete.1 \
	starpu_lp2paje.1	\
	starpu_workers_activity.1 \
	starpu_codelet_profile.1 \
	starpu_codelet_histo_profile.1 \
	starpu_env.1 \
	starpu_mpi_comm_matrix.1 \
	starpu_paje_draw_histogram.1 \
	starpu_paje_state_stats.1

if STARPU_USE_FXT
dist_man1_MANS +=\
	starpu_fxt_tool.1 \
	starpu_fxt_stats.1 \
	starpu_fxt_data_trace.1
endif

clean-local:
	$(RM) $(dist_man1_MANS)

endif

if STARPU_SIMGRID
dist_pkgdata_DATA += starpu_smpi.xslt
dist_bin_SCRIPTS += starpu_smpirun
endif
