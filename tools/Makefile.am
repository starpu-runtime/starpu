# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
include $(top_srcdir)/starpu-tests.mk
include $(top_srcdir)/starpu-loader.mk

SUBDIRS =

AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/tools/ -I$(top_srcdir)/mpi/ -I$(top_builddir)/src -I$(top_srcdir)/src $(STARPU_H_CPPFLAGS)
AM_LDFLAGS = @STARPU_EXPORT_DYNAMIC@
LIBS += $(top_builddir)/src/@LIBSTARPU_LINK@ $(STARPU_EXPORTED_LIBS)

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
	perfmodels/sampling/codelets/45/chol_model_potrf.attila	\
	perfmodels/sampling/codelets/45/chol_model_trsm.attila	\
	perfmodels/sampling/codelets/45/chol_model_syrk.attila	\
	perfmodels/sampling/codelets/45/chol_model_gemm.attila	\
	perfmodels/sampling/codelets/45/cl_update.attila	\
	perfmodels/sampling/codelets/45/save_cl_bottom.attila	\
	perfmodels/sampling/codelets/45/save_cl_top.attila	\
	perfmodels/sampling/codelets/45/starpu_sgemm_gemm.attila	\
	perfmodels/sampling/codelets/45/starpu_dgemm_gemm.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_atlas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_goto.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_openblas.attila	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_openblas.attila	\
	perfmodels/sampling/codelets/45/overlap_sleep_1024_24.attila	\
\
	perfmodels/sampling/codelets/45/chol_model_potrf.hannibal	\
	perfmodels/sampling/codelets/45/chol_model_trsm.hannibal	\
	perfmodels/sampling/codelets/45/chol_model_syrk.hannibal	\
	perfmodels/sampling/codelets/45/chol_model_gemm.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_atlas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_atlas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_atlas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_atlas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_goto.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_goto.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_goto.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_goto.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_openblas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_openblas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_openblas.hannibal	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_openblas.hannibal	\
\
	perfmodels/sampling/codelets/45/chol_model_potrf.hannibal-pitch	\
	perfmodels/sampling/codelets/45/chol_model_trsm.hannibal-pitch	\
	perfmodels/sampling/codelets/45/chol_model_syrk.hannibal-pitch	\
	perfmodels/sampling/codelets/45/chol_model_gemm.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_atlas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_atlas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_atlas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_atlas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_goto.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_goto.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_goto.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_goto.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_openblas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_openblas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_openblas.hannibal-pitch	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_openblas.hannibal-pitch	\
\
	perfmodels/sampling/codelets/45/chol_model_potrf.idgraf	\
	perfmodels/sampling/codelets/45/chol_model_trsm.idgraf	\
	perfmodels/sampling/codelets/45/chol_model_syrk.idgraf	\
	perfmodels/sampling/codelets/45/chol_model_gemm.idgraf	\
	perfmodels/sampling/codelets/45/cl_update.idgraf	\
	perfmodels/sampling/codelets/45/save_cl_bottom.idgraf	\
	perfmodels/sampling/codelets/45/save_cl_top.idgraf	\
	perfmodels/sampling/codelets/45/starpu_sgemm_gemm.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dgemm_gemm.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_atlas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_goto.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_openblas.idgraf	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_openblas.idgraf	\
\
	perfmodels/sampling/codelets/45/chol_model_potrf.mirage	\
	perfmodels/sampling/codelets/45/chol_model_trsm.mirage	\
	perfmodels/sampling/codelets/45/chol_model_syrk.mirage	\
	perfmodels/sampling/codelets/45/chol_model_gemm.mirage	\
	perfmodels/sampling/codelets/45/cl_update.mirage	\
	perfmodels/sampling/codelets/45/save_cl_bottom.mirage	\
	perfmodels/sampling/codelets/45/save_cl_top.mirage	\
	perfmodels/sampling/codelets/45/starpu_sgemm_gemm.mirage	\
	perfmodels/sampling/codelets/45/starpu_dgemm_gemm.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_atlas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_goto.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_openblas.mirage	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_openblas.mirage	\
	perfmodels/sampling/codelets/45/overlap_sleep_1024_24.mirage	\
\
	perfmodels/sampling/codelets/45/chol_model_potrf.sirocco	\
	perfmodels/sampling/codelets/45/chol_model_trsm.sirocco	\
	perfmodels/sampling/codelets/45/chol_model_syrk.sirocco	\
	perfmodels/sampling/codelets/45/chol_model_gemm.sirocco	\
	perfmodels/sampling/codelets/45/cl_update.sirocco	\
	perfmodels/sampling/codelets/45/save_cl_bottom.sirocco	\
	perfmodels/sampling/codelets/45/save_cl_top.sirocco	\
	perfmodels/sampling/codelets/45/starpu_sgemm_gemm.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dgemm_gemm.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_atlas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_goto.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_getrf_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ll_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_trsm_ru_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_slu_lu_model_gemm_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_getrf_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ll_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_trsm_ru_openblas.sirocco	\
	perfmodels/sampling/codelets/45/starpu_dlu_lu_model_gemm_openblas.sirocco	\
	perfmodels/sampling/codelets/45/overlap_sleep_1024_24.sirocco	\
\
	perfmodels/sampling/codelets/45/null.idgraf	\
	perfmodels/sampling/codelets/45/null.sirocco

EXTRA_DIST =				\
	dev/checker/rename.sed		\
	dev/checker/rename.sh		\
	dev/cppcheck/suppressions.txt	\
	dev/valgrind/blas.suppr		\
	dev/valgrind/nvidia.suppr	\
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
	dev/valgrind/p11-kit.suppr	\
	dev/valgrind/padico.suppr	\
	dev/valgrind/papi.suppr		\
	dev/valgrind/pthread.suppr	\
	dev/valgrind/starpu.suppr	\
	dev/valgrind/starpupy.suppr	\
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
	msvc/starpu/starpu.vcxproj	\
	release/Makefile		\
	release/README			\
	patch-ayudame			\
	perfs/bench_sgemm.sh		\
	perfs/error_model.gp		\
	perfs/error_model.sh		\
	distrib/distrib.r		\
	distrib/distrib.sh

CLEANFILES = *.gcno *.gcda *.linkinfo starpu_idle_microsec.log figure/* mlr_*

#####################################
# What to install and what to check #
#####################################

STARPU_TOOLS	=
TESTS		= $(STARPU_TOOLS)

check_PROGRAMS	=	$(STARPU_TOOLS)

if STARPU_USE_FXT
bin_PROGRAMS += 			\
	starpu_fxt_tool			\
	starpu_fxt_stats		\
	starpu_fxt_data_trace

STARPU_TOOLS += 			\
	starpu_fxt_tool			\
	starpu_fxt_stats		\
	starpu_fxt_data_trace

starpu_fxt_tool_CPPFLAGS = $(AM_CPPFLAGS) $(FXT_CFLAGS)
starpu_fxt_tool_LDADD = $(FXT_LIBS)
starpu_fxt_tool_LDFLAGS = $(FXT_LDFLAGS)

starpu_fxt_stats_CPPFLAGS = $(AM_CPPFLAGS) $(FXT_CFLAGS)
starpu_fxt_stats_LDADD = $(FXT_LIBS)
starpu_fxt_stats_LDFLAGS = $(FXT_LDFLAGS)

starpu_fxt_data_trace_CPPFLAGS = $(AM_CPPFLAGS) $(FXT_CFLAGS)
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

starpu_perfmodel_plot_CPPFLAGS = $(AM_CPPFLAGS) $(FXT_CFLAGS)

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

dist_bin_SCRIPTS +=				\
	starpu_workers_activity			\
	starpu_codelet_histo_profile		\
	starpu_codelet_profile			\
	starpu_env				\
	starpu_config				\
	starpu_mpi_comm_matrix.py		\
	starpu_fxt_number_events_to_names.py	\
	starpu_paje_draw_histogram		\
	starpu_paje_draw_histogram.R		\
	starpu_paje_summary			\
	starpu_paje_summary.Rmd			\
	starpu_mlr_analysis			\
	starpu_mlr_analysis.Rmd			\
	starpu_paje_state_stats			\
	starpu_paje_state_stats.R			\
	starpu_send_recv_data_use.py 		\
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
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Force StarPU bus calibration" --output=$@ ./$<
starpu_machine_display.1: starpu_machine_display$(EXEEXT)
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Display machine StarPU information" --output=$@ ./$<
starpu_perfmodel_display.1: starpu_perfmodel_display$(EXEEXT)
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Display StarPU performance model" --output=$@ ./$<
starpu_perfmodel_plot.1: starpu_perfmodel_plot$(EXEEXT)
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Plot StarPU performance model" --output=$@ ./$<
starpu_tasks_rec_complete.1: starpu_tasks_rec_complete$(EXEEXT)
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Complete StarPU tasks.rec file" --output=$@ ./$<
starpu_lp2paje.1: starpu_lp2paje$(EXEEXT)
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Convert lp StarPU schedule into Paje format" --output=$@ ./$<
starpu_workers_activity.1: starpu_workers_activity
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Display StarPU workers activity" --output=$@ ./$<
starpu_codelet_profile.1: starpu_codelet_profile
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Draw StarPU codelet profile" --output=$@ ./$<
starpu_env.1: starpu_env
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Set StarPU environment variables" --output=$@ ./$<
starpu_codelet_histo_profile.1: starpu_codelet_histo_profile
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Draw StarPU codelet histogram" --output=$@ ./$<
starpu_mpi_comm_matrix.1: starpu_mpi_comm_matrix.py
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Draw StarPU MPI communications matrix" --output=$@ ./$<
starpu_fxt_number_events_to_names.1: starpu_fxt_number_events_to_names.py
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Convert events in StarPU traces" --output=$@ ./$<
starpu_paje_draw_histogram.1: starpu_paje_draw_histogram
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Draw StarPU trace histogram" --output=$@ ./$<
starpu_paje_state_stats.1: starpu_paje_state_stats
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Print statistics from StarPU trace" --output=$@ ./$<
starpu_config.1: starpu_config
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Display StarPU configuration" --output=$@ ./$<

if STARPU_USE_FXT
starpu_fxt_tool.1: starpu_fxt_tool$(EXEEXT)
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Convert raw StarPU FxT trace to various traces" --output=$@ ./$<
starpu_fxt_stats.1: starpu_fxt_stats$(EXEEXT)
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Print statistics from raw StarPU FxT trace" --output=$@ ./$<
starpu_fxt_data_trace.1: starpu_fxt_data_trace$(EXEEXT)
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Print data trace from raw StarPU FxT trace" --output=$@ ./$<
endif

if STARPU_USE_TCPIP_MASTER_SLAVE
starpu_tcpipexec.1: starpu_tcpipexec
	@chmod +x $<
	$(V_help2man) LC_ALL=C help2man --no-discard-stderr -N -n "Execute TCPIP application" --output=$@ ./$<
endif

dist_man1_MANS = \
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
	starpu_fxt_number_events_to_names.1 \
	starpu_paje_draw_histogram.1 \
	starpu_paje_state_stats.1	\
	starpu_config.1

if STARPU_USE_FXT
dist_man1_MANS +=\
	starpu_fxt_tool.1 \
	starpu_fxt_stats.1 \
	starpu_fxt_data_trace.1
endif

if STARPU_USE_TCPIP_MASTER_SLAVE
dist_man1_MANS +=\
	starpu_tcpipexec.1
endif

clean-local:
	$(RM) $(dist_man1_MANS) starpu_config.cfg

endif

if STARPU_SIMGRID
dist_pkgdata_DATA += starpu_smpi.xslt
dist_bin_SCRIPTS += starpu_smpirun
endif

if STARPU_USE_TCPIP_MASTER_SLAVE
dist_bin_SCRIPTS += starpu_tcpipexec
endif

dist_pkgdata_DATA += starpu_config.cfg
starpu_config.cfg: $(top_builddir)/src/common/config.h
	grep STARPU $< | grep '#' > $@

