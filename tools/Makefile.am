# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2009-2013  Université de Bordeaux 1
# Copyright (C) 2010, 2011, 2012, 2013  Centre National de la Recherche Scientifique
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

SUBDIRS =

AM_CFLAGS = $(HWLOC_CFLAGS) $(STARPU_CUDA_CPPFLAGS) $(STARPU_OPENCL_CPPFLAGS)
LIBS = $(top_builddir)/src/@LIBSTARPU_LINK@ @LIBS@
AM_CPPFLAGS = -I$(top_srcdir)/include/ -I$(top_srcdir)/tools/ -I$(top_srcdir)/mpi/ -I$(top_builddir)/src -I$(top_srcdir)/src

bin_PROGRAMS =
dist_bin_SCRIPTS =

dist_pkgdata_DATA = gdbinit

EXTRA_DIST =					\
  dev/rename.sed				\
  dev/rename.sh

CLEANFILES = *.gcno *.gcda *.linkinfo

#####################################
# What to install and what to check #
#####################################

STARPU_TOOLS	=
TESTS		= $(STARPU_TOOLS)

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
	starpu_machine_display

if STARPU_LONG_CHECK
STARPU_TOOLS	+=			\
	starpu_calibrate_bus
endif

STARPU_TOOLS	+=			\
	starpu_perfmodel_display	\
	starpu_perfmodel_plot 		\
	starpu_machine_display

noinst_PROGRAMS =	cbc2paje lp2paje

dist_bin_SCRIPTS +=			\
	starpu_workers_activity		\
	starpu_codelet_histo_profile	\
	starpu_codelet_profile


if STARPU_HAVE_HELP2MAN
starpu_calibrate_bus.1: starpu_calibrate_bus$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_machine_display.1: starpu_machine_display$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_perfmodel_display.1: starpu_perfmodel_display$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_perfmodel_plot.1: starpu_perfmodel_plot$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_workers_activity.1: starpu_workers_activity$(EXEEXT)
	chmod +x $<
	help2man --no-discard-stderr -N --output=$@ ./$<
starpu_codelet_profile.1: starpu_codelet_profile$(EXEEXT)
	chmod +x $<
	help2man --no-discard-stderr -N --output=$@ $<
starpu_codelet_histo_profile.1: starpu_codelet_histo_profile$(EXEEXT)
	chmod +x $<
	help2man --no-discard-stderr -N --output=$@ $<

if STARPU_USE_FXT
starpu_fxt_tool.1: starpu_fxt_tool$(EXEEXT)
	chmod +x starpu_fxt_tool$(EXEEXT)
	help2man --no-discard-stderr -N --output=$@ ./$<
endif

dist_man1_MANS =\
	starpu_calibrate_bus.1 \
	starpu_machine_display.1 \
	starpu_perfmodel_display.1 \
	starpu_perfmodel_plot.1	\
	starpu_workers_activity.1 \
	starpu_codelet_profile.1 \
	starpu_codelet_histo_profile.1

if STARPU_USE_FXT
dist_man1_MANS +=\
	starpu_fxt_tool.1
endif

clean-local:
	$(RM) $(dist_man1_MANS)

endif

showcheck:
	-cat $(TEST_LOGS) /dev/null
