# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

AM_CFLAGS = $(HWLOC_CFLAGS) $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS) $(GLOBAL_AM_CFLAGS)
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/tools/ -I$(top_srcdir)/mpi/ -I$(top_builddir)/src -I$(top_srcdir)/src

bin_PROGRAMS =
dist_bin_SCRIPTS =

dist_pkgdata_DATA = gdbinit

pkgdata_perfmodels_sampling_busdir = $(datarootdir)/starpu/perfmodels/sampling/bus
pkgdata_perfmodels_sampling_codeletsdir = $(datarootdir)/starpu/perfmodels/sampling/codelets/42

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
	perfmodels/sampling/bus/mirage.affinity	\
	perfmodels/sampling/bus/mirage.bandwidth	\
	perfmodels/sampling/bus/mirage.config	\
	perfmodels/sampling/bus/mirage.latency	\
	perfmodels/sampling/bus/mirage.platform.xml	\
	perfmodels/sampling/bus/mirage.platform.v4.xml

dist_pkgdata_perfmodels_sampling_codelets_DATA = \
	perfmodels/sampling/codelets/42/chol_model_11.attila	\
	perfmodels/sampling/codelets/42/chol_model_21.attila	\
	perfmodels/sampling/codelets/42/chol_model_22.attila	\
	perfmodels/sampling/codelets/42/cl_update.attila	\
	perfmodels/sampling/codelets/42/save_cl_bottom.attila	\
	perfmodels/sampling/codelets/42/save_cl_top.attila	\
	perfmodels/sampling/codelets/42/starpu_sgemm_gemm.attila	\
	perfmodels/sampling/codelets/42/starpu_dgemm_gemm.attila	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_11.attila	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_12.attila	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_21.attila	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_22.attila	\
	perfmodels/sampling/codelets/42/starpu_dlu_lu_model_11.attila	\
	perfmodels/sampling/codelets/42/starpu_dlu_lu_model_12.attila	\
	perfmodels/sampling/codelets/42/starpu_dlu_lu_model_21.attila	\
	perfmodels/sampling/codelets/42/starpu_dlu_lu_model_22.attila	\
	perfmodels/sampling/codelets/42/chol_model_11.hannibal	\
	perfmodels/sampling/codelets/42/chol_model_21.hannibal	\
	perfmodels/sampling/codelets/42/chol_model_22.hannibal	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_11.hannibal	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_12.hannibal	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_21.hannibal	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_22.hannibal	\
	perfmodels/sampling/codelets/42/chol_model_11.hannibal-pitch	\
	perfmodels/sampling/codelets/42/chol_model_21.hannibal-pitch	\
	perfmodels/sampling/codelets/42/chol_model_22.hannibal-pitch	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_11.hannibal-pitch	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_12.hannibal-pitch	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_21.hannibal-pitch	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_22.hannibal-pitch	\
	perfmodels/sampling/codelets/42/chol_model_11.mirage	\
	perfmodels/sampling/codelets/42/chol_model_21.mirage	\
	perfmodels/sampling/codelets/42/chol_model_22.mirage	\
	perfmodels/sampling/codelets/42/cl_update.mirage	\
	perfmodels/sampling/codelets/42/save_cl_bottom.mirage	\
	perfmodels/sampling/codelets/42/save_cl_top.mirage	\
	perfmodels/sampling/codelets/42/starpu_sgemm_gemm.mirage	\
	perfmodels/sampling/codelets/42/starpu_dgemm_gemm.mirage	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_11.mirage	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_12.mirage	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_21.mirage	\
	perfmodels/sampling/codelets/42/starpu_slu_lu_model_22.mirage	\
	perfmodels/sampling/codelets/42/starpu_dlu_lu_model_11.mirage	\
	perfmodels/sampling/codelets/42/starpu_dlu_lu_model_12.mirage	\
	perfmodels/sampling/codelets/42/starpu_dlu_lu_model_21.mirage	\
	perfmodels/sampling/codelets/42/starpu_dlu_lu_model_22.mirage

EXTRA_DIST =				\
	dev/rename.sed			\
	dev/rename.sh			\
	perfmodels/README		\
	valgrind/libc.suppr		\
	valgrind/hwloc.suppr		\
	valgrind/libnuma.suppr		\
	valgrind/openmpi.suppr		\
	valgrind/opencl.suppr		\
	valgrind/pthread.suppr		\
	valgrind/starpu.suppr		\
	valgrind/libgomp.suppr		\
	valgrind/helgrind.sh		\
	dev/tsan/starpu.suppr		\
	dev/lsan/suppressions		\
	msvc/starpu_clean.bat		\
	msvc/starpu_open.bat		\
	msvc/starpu_exec.bat		\
	msvc/starpu_var.bat		\
	msvc/starpu.sln			\
	msvc/starpu/starpu.vcxproj	\
	cppcheck/suppressions.txt

CLEANFILES = *.gcno *.gcda *.linkinfo

#####################################
# What to install and what to check #
#####################################

STARPU_TOOLS	=
if !STARPU_SIMGRID
TESTS		= $(STARPU_TOOLS)
endif

if STARPU_HAVE_WINDOWS
check_PROGRAMS	=	$(STARPU_TOOLS)
else
check_PROGRAMS	=	$(LOADER) $(STARPU_TOOLS)
endif

if !STARPU_HAVE_WINDOWS
## test loader program
LOADER			=	loader
loader_CPPFLAGS =  $(AM_CFLAGS) $(AM_CPPFLAGS) -I$(top_builddir)/src/
LOADER_BIN		=	$(abs_top_builddir)/tools/$(LOADER)
loader_SOURCES		=	../tests/loader.c

if STARPU_HAVE_AM111
TESTS_ENVIRONMENT	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)"
LOG_COMPILER		=	$(LOADER_BIN)
else
TESTS_ENVIRONMENT	=	top_builddir="$(abs_top_builddir)" top_srcdir="$(abs_top_srcdir)" $(LOADER_BIN)
endif

endif

if STARPU_USE_FXT
bin_PROGRAMS += 			\
	starpu_fxt_tool			\
	starpu_fxt_stats

STARPU_TOOLS += 			\
	starpu_fxt_tool			\
	starpu_fxt_stats

starpu_fxt_tool_CPPFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS) $(FXT_CFLAGS)
starpu_fxt_tool_LDADD = $(FXT_LIBS)
starpu_fxt_tool_LDFLAGS = $(FXT_LDFLAGS)

starpu_fxt_stats_CPPFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS) $(FXT_CFLAGS)
starpu_fxt_stats_LDADD = $(FXT_LIBS)
starpu_fxt_stats_LDFLAGS = $(FXT_LDFLAGS)
endif

bin_PROGRAMS += 			\
	starpu_perfmodel_display	\
	starpu_perfmodel_plot 		\
	starpu_calibrate_bus		\
	starpu_machine_display		\
	starpu_lp2paje

starpu_perfmodel_plot_CPPFLAGS = $(AM_CFLAGS) $(AM_CPPFLAGS) $(FXT_CFLAGS)

if STARPU_LONG_CHECK
STARPU_TOOLS	+=			\
	starpu_calibrate_bus
endif

STARPU_TOOLS	+=			\
	starpu_machine_display

if !STARPU_HAVE_WINDOWS
STARPU_TOOLS	+=			\
	starpu_perfmodel_display	\
	starpu_perfmodel_plot
endif

dist_bin_SCRIPTS +=			\
	starpu_workers_activity		\
	starpu_codelet_histo_profile	\
	starpu_codelet_profile		\
	starpu_paje_draw_histogram	\
	starpu_paje_draw_histogram.R	\
	starpu_paje_state_stats

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
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_machine_display.1: starpu_machine_display$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_perfmodel_display.1: starpu_perfmodel_display$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_perfmodel_plot.1: starpu_perfmodel_plot$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_lp2paje.1: starpu_lp2paje$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_workers_activity.1: starpu_workers_activity
	chmod +x $<
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_codelet_profile.1: starpu_codelet_profile
	chmod +x $<
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_codelet_histo_profile.1: starpu_codelet_histo_profile
	chmod +x $<
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_paje_draw_histogram.1: starpu_paje_draw_histogram
	chmod +x $<
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_paje_state_stats.1: starpu_paje_state_stats
	chmod +x $<
	help2man --no-discard-stderr -N --output=$@ ./$<

if STARPU_USE_FXT
starpu_fxt_tool.1: starpu_fxt_tool$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_fxt_stats.1: starpu_fxt_stats$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
endif

dist_man1_MANS =\
	starpu_calibrate_bus.1 \
	starpu_machine_display.1 \
	starpu_perfmodel_display.1 \
	starpu_perfmodel_plot.1	\
	starpu_lp2paje.1	\
	starpu_workers_activity.1 \
	starpu_codelet_profile.1 \
	starpu_codelet_histo_profile.1 \
	starpu_paje_draw_histogram.1 \
	starpu_paje_state_stats.1

if STARPU_USE_FXT
dist_man1_MANS +=\
	starpu_fxt_tool.1 \
	starpu_fxt_stats.1
endif

clean-local:
	$(RM) $(dist_man1_MANS)

endif

